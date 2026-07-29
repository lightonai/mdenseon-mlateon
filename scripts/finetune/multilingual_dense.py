"""
Contrastive (Matryoshka MNRL) + KL-divergence distillation fine-tuning of a dense
retriever on the multilingual and code datasets (query + 1 positive +
stored hard negatives + teacher cross-encoder scores).
The script is adapted for dense long context training under limited GPU memory.

What to tweak
-------------
CLI:
  --model_name        Checkpoint to fine-tune (typically the pretraining output).
  --learning_rate / --num_train_epochs
  --batch_size        Global batch, split across devices; defines the pool of
                      in-batch negatives, which are gathered across devices.
  --temperature       Temperature of the contrastive loss.
  --student_temperature / --teacher_temperature
                      Softmax temperatures of the KL-div loss applied to
                      teacher and student relevance scores.
  --contrastive_weight / --kldiv_weight
  --matryoshka_dims   Truncation dimensions for Matryoshka MNRL loss.
  --mini_batch_size   GradCache chunk size to reduce GPU memory usage. 0 disables
                      GradCache.
  --mini_batch_size_by_prefix
                      GradCache chunk size per-split overrides for splits (e.g. mldr:8).
  --alpha             Multinomial sampling exponent: each batch is drawn from a
                      dataset with probability proportional to size ** alpha
                      (1.0 = proportional to size, 0.5 = smoothed, 0 = uniform).

Data build (load_train_datasets — baked into the on-disk cache, delete
{cache_dir}/{split} after changing any of them):
  code_splits / multilingual_splits  Which subsets to train on.
  nv_threshold=0.95   A document counts as a negative only if its retrieval
                      score is < 0.95 * positive score (false-negative filter;
                      lower = stricter).
  num_negatives=10    Hard negatives stored per query.

In main():
  num_negatives=7 (collator)  Negatives sampled per step from the stored 10.
  model.max_seq_length        Truncation length (8192).
  prompts                     "query: " / "document: " prefixes; keep the
                              collator and the evaluator in sync.
"""

from __future__ import annotations

import itertools
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(42)

import argparse

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.evaluation.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from sentence_transformers.model_card import SentenceTransformerModelCardData
from sentence_transformers.sampler import MultiDatasetDefaultBatchSampler
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import all_gather_with_grad
from torch.utils.checkpoint import get_device_states, set_device_states
from tqdm import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

# Disable dataset metrics
SentenceTransformerModelCardData.compute_dataset_metrics = lambda self, dataset, dataset_info, loss: {}

logger = logging.getLogger(__name__)

class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.stop_at_step:
            print(f"\n Reached target step {self.stop_at_step}. Stopping training...")
            control.should_training_stop = True
        return control


class MultilingualNanoBEIREvaluator(NanoBEIREvaluator):
    """Evaluate dense models on the Multilingual NanoBEIR collection, per language and aggregated."""

    DATASETS = {
        "climatefever": "ClimateFEVER", "dbpedia": "DBPedia", "fever": "FEVER",
        "fiqa2018": "FiQA2018", "hotpotqa": "HotpotQA", "msmarco": "MSMARCO",
        "nfcorpus": "NFCorpus", "nq": "NQ", "quoraretrieval": "QuoraRetrieval",
        "scidocs": "SCIDOCS", "arguana": "ArguAna", "scifact": "SciFact",
        "touche2020": "Touche2020",
    }
    SUPPORTED_LANGUAGES = ["ar", "de", "en", "es", "fr", "it", "no", "pt", "sv"]

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        languages: list[str] | None = None,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
        dataset_path: str = "lightonai/nanobeir-multilingual",
        **ir_evaluator_kwargs: Any,
    ):
        self.dataset_names = dataset_names or list(self.DATASETS)
        self.languages = languages or self.SUPPORTED_LANGUAGES
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key
        self.dataset_path = dataset_path
        self.write_csv = False
        self.name = f"MultilingualNanoBEIR_{aggregate_key}"
        self.primary_metric = f"{self.name}_cosine_ndcg@10"

        if unknown := [name for name in self.dataset_names if name.lower() not in self.DATASETS]:
            raise ValueError(f"Dataset(s) {unknown} not found. Valid: {list(self.DATASETS)}")
        if unsupported := [language for language in self.languages if language not in self.SUPPORTED_LANGUAGES]:
            raise ValueError(f"Language(s) {unsupported} not supported. Supported: {self.SUPPORTED_LANGUAGES}")

        self.query_prompts = self._validate_prompts(query_prompts)
        self.corpus_prompts = self._validate_prompts(corpus_prompts)
        self.ir_evaluator_kwargs = {"show_progress_bar": show_progress_bar, "batch_size": batch_size, "write_csv": False, **ir_evaluator_kwargs}

        self.evaluators, self.evaluator_languages = [], []
        for language in self.languages:
            for name in self.dataset_names:
                try:
                    self.evaluators.append(self._load_dataset(name, language))
                    self.evaluator_languages.append(language)
                except Exception as exception:
                    logger.warning(f"Failed to load {name}-{language}: {exception}")

        if not self.evaluators:
            raise ValueError("No evaluators created. Please check your dataset_names and languages.")

    def _validate_prompts(self, prompts: str | dict[str, str] | None) -> dict[str, str] | None:
        """Broadcast a single prompt to every dataset, or check that a per-dataset mapping is complete."""
        if isinstance(prompts, str):
            return {name: prompts for name in self.dataset_names}
        if prompts and (missing := [name for name in self.dataset_names if name not in prompts]):
            raise ValueError(f"Missing prompts for: {missing}")
        return prompts

    def _load_dataset(self, dataset_name: str, language: str) -> InformationRetrievalEvaluator:
        base_name = f"Nano{self.DATASETS[dataset_name.lower()]}"
        corpus = load_dataset(self.dataset_path, f"{base_name}_{language}", split="corpus")
        queries = load_dataset(self.dataset_path, f"{base_name}_{language}", split="queries")
        qrels = load_dataset(self.dataset_path, base_name, split="qrels")

        relevant_docs = defaultdict(set)
        for sample in qrels:
            relevant_docs[sample["query-id"]].add(sample["corpus-id"])

        prompt_kwargs = {
            key: prompts.get(dataset_name)
            for prompts, key in [(self.query_prompts, "query_prompt"), (self.corpus_prompts, "corpus_prompt")]
            if prompts is not None
        }
        return InformationRetrievalEvaluator(
            queries={sample["_id"]: sample["text"] for sample in queries if sample.get("text")},
            corpus={sample["_id"]: sample["text"] for sample in corpus if sample.get("text")},
            relevant_docs=dict(relevant_docs),
            name=f"{base_name}-{language}",
            **self.ir_evaluator_kwargs,
            **prompt_kwargs,
        )

    def __call__(
        self,
        model,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        logger.info(f"Multilingual NanoBEIR Evaluation on {len(self.evaluators)} dataset-language pairs")
        results, per_metric, per_language = {}, defaultdict(list), defaultdict(lambda: defaultdict(list))

        for evaluator, language in tqdm(
            list(zip(self.evaluators, self.evaluator_languages)),
            desc="Evaluating datasets",
            disable=not self.show_progress_bar,
        ):
            logger.info(f"Evaluating {evaluator.name}")
            for full_key, value in evaluator(model, output_path, epoch, steps).items():
                metric = full_key.removeprefix(f"{evaluator.name}_")
                results[full_key] = value
                per_metric[metric].append(value)
                per_language[language][metric].append(value)

        results.update({f"{self.name}_{metric}": self.aggregate_fn(values) for metric, values in per_metric.items()})
        results.update({
            f"{self.name}_{language}_{metric}": self.aggregate_fn(values)
            for language, metrics in per_language.items()
            for metric, values in metrics.items()
        })

        logger.info(f"Aggregated results ({self.aggregate_key}): " + self._format(per_metric))
        for language, metrics in per_language.items():
            logger.info(f"  {language}: " + self._format(metrics))
        return results

    def _format(self, metrics: dict[str, list[float]]) -> str:
        return ", ".join(f"{metric}={self.aggregate_fn(values):.4f}" for metric, values in sorted(metrics.items()))

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "dataset_names": self.dataset_names,
            "languages": self.languages,
            "aggregate_key": self.aggregate_key,
            "dataset_path": self.dataset_path,
        }


class MultinomialBatchSampler(MultiDatasetDefaultBatchSampler):
    """Pick each batch's dataset with probability proportional to len(dataset) ** alpha.

    alpha=1.0 samples proportionally to dataset size, 0.5 smooths towards the small datasets,
    0 is uniform over datasets.
    """

    def __init__(self, dataset, batch_samplers, alpha: float = 1.0, generator=None, seed: int = 0):
        super().__init__(dataset, batch_samplers, generator, seed)
        weights = [len(dataset) ** alpha for dataset in self.dataset.datasets]
        self.sampling_probs = [weight / sum(weights) for weight in weights]

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        offsets = [0] + list(itertools.accumulate(len(dataset) for dataset in self.dataset.datasets))
        iterators = [iter(sampler) for sampler in self.batch_samplers]
        resets = [0] * len(self.batch_samplers)
        probs = torch.tensor(self.sampling_probs, dtype=torch.float32)

        emitted = 0
        while emitted < len(self) and float(probs.sum()) > 0.0:
            index = torch.multinomial(probs, num_samples=1, replacement=True, generator=self.generator).item()
            batch = next(iterators[index], None)
            if batch is None:  # exhausted dataset: reseed and restart it so it keeps contributing
                resets[index] += 1
                iterators[index] = self._restart(index, resets[index])
                batch = next(iterators[index], None)
            if batch is None:
                logger.warning("MultinomialBatchSampler: dataset index %d yields no batches, excluding it.", index)
                probs[index] = 0.0
                continue
            emitted += 1
            yield [i + offsets[index] for i in batch]

    def _restart(self, index: int, reset_count: int):
        """Reseed the underlying sampler so a restarted dataset yields a fresh permutation."""
        sampler = self.batch_samplers[index]
        inner = getattr(sampler, "sampler", None)
        if hasattr(inner, "generator"):
            inner.generator.manual_seed(self.seed + self.epoch * 100000 + reset_count)
        elif hasattr(inner, "set_epoch"):
            inner.set_epoch(self.epoch * 100000 + reset_count)
        return iter(sampler)

    def __len__(self) -> int:
        return sum(len(sampler) for sampler in self.batch_samplers)


class RandContext:
    """Save/restore RNG state so a re-forward sees the same dropout pattern.
       Used for GradCache in the loss below.
    """

    def __init__(self, *tensors: torch.Tensor):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)


class MatryoshkaMultipleNegativesRankingKLDivLoss(torch.nn.Module):
    """Matryoshka MNRL + KL-div distillation of teacher scores.

    The loss is ``contrastive_weight * MNRL + kldiv_weight * KL``:

    - **MNRL**: cross-entropy over the cosine similarities between each query and every document
      of every rank (positives and hard negatives are all-gathered, so the negative pool is
      `world_size * batch_size * (1 + k)`), scaled by `scale` and uniformly averaged over the
      `matryoshka_dims` truncations of the embedding.
    - **KL**: per-query KL divergence between the student distribution over its own
      `[positive, negative_0, ..., negative_k]` and the teacher's over the same documents, on the
      full embedding dimension only. Each side is softmaxed at its own temperature, which affects
      this term only.

    Args:
        model (`SentenceTransformer`):
            Model embedding the queries and documents; shared by both terms.
        scale (`float`, *optional*, defaults to 50.0):
            Inverse temperature multiplying the cosine similarities of the MNRL term.
        student_temperature (`float`, *optional*, defaults to 1.0):
            Softmax temperature of the student scores in the KL term. Lower is sharper.
        teacher_temperature (`float`, *optional*, defaults to 1.0):
            Softmax temperature of the teacher scores in the KL term. Lower is sharper.
        contrastive_weight (`float`, *optional*, defaults to 1.0):
            Weight of the MNRL term.
        kldiv_weight (`float`, *optional*, defaults to 1.0):
            Weight of the KL term. 0 skips it and trains on MNRL alone.
        matryoshka_dims (`list[int]`, *optional*):
            Embedding truncations the MNRL term is averaged over. `None` or a single dimension
            disables MRL.
        mini_batch_size (`int`, *optional*):
            GradCache chunk size. `None` embeds the whole batch in one forward; any value splits
            it into chunks to bound activation memory, at the cost of a second forward pass, and
            yields exactly the full-batch loss and gradients.

    Requirements:
        1. Columns ordered `(query, positive, negative_0, ..., negative_k)`.
        2. When `kldiv_weight` > 0, a `label` of shape `(batch_size, 1 + k)` holding the teacher
           scores of those same documents, in the same order.

    Example:
        ::

            loss = MatryoshkaMultipleNegativesRankingKLDivLoss(
                model,
                scale=1 / 0.02,
                student_temperature=0.02,
                teacher_temperature=0.1,
                matryoshka_dims=[768, 512, 256, 128],
                mini_batch_size=16,
            )
    """

    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 50.0,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        contrastive_weight: float = 1.0,
        kldiv_weight: float = 1.0,
        matryoshka_dims: list[int] | None = None,
        mini_batch_size: int | None = None,
    ):
        super().__init__()
        self.model = model
        self.scale = scale
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.contrastive_weight = contrastive_weight
        self.kldiv_weight = kldiv_weight
        self.matryoshka_dims = matryoshka_dims
        self.mini_batch_size = mini_batch_size
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def loss_from_embeddings(self, embeddings: list[torch.Tensor], labels: torch.Tensor | None) -> torch.Tensor:
        queries, batch_size = embeddings[0], embeddings[0].size(0)

        # Matryoshka InfoNCE against the documents of every rank, positives on the block diagonal
        documents = torch.cat([all_gather_with_grad(embedding) for embedding in embeddings[1:]], dim=0)
        rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        targets = torch.arange(batch_size, device=queries.device) + rank * batch_size

        contrastive_loss = torch.stack([
            self.cross_entropy_loss(
                torch.nn.functional.normalize(queries[:, :dim], dim=1)
                @ torch.nn.functional.normalize(documents[:, :dim], dim=1).T
                * self.scale,
                targets,
            )
            for dim in (self.matryoshka_dims or [queries.size(1)])
        ]).mean()

        if labels is None or self.kldiv_weight == 0:
            return self.contrastive_weight * contrastive_loss

        # KL-div over this row's own candidates only: (B, 1 + K) student vs teacher distributions
        candidates = torch.nn.functional.normalize(torch.stack(embeddings[1:], dim=1), dim=-1)
        student_scores = torch.bmm(torch.nn.functional.normalize(queries, dim=-1).unsqueeze(1), candidates.transpose(1, 2)).squeeze(1)
        teacher_scores = labels.to(student_scores.device, dtype=student_scores.dtype)
        kldiv_loss = self.kl_loss(
            torch.nn.functional.log_softmax(student_scores / self.student_temperature, dim=-1),
            torch.nn.functional.log_softmax(teacher_scores / self.teacher_temperature, dim=-1),
        )
        return self.contrastive_weight * contrastive_loss + self.kldiv_weight * kldiv_loss

    def forward(self, sentence_features: list[dict[str, torch.Tensor]], labels: torch.Tensor | None = None) -> torch.Tensor:
        if not torch.is_grad_enabled() or self.mini_batch_size is None:
            return self.loss_from_embeddings([self.model(features)["sentence_embedding"] for features in sentence_features], labels)

        # GradCache: chunked no-grad forward, then the exact full-batch loss on the concatenated embeddings
        chunked_embeddings, random_states = self._chunked_no_grad_forward(sentence_features)
        with torch.no_grad():
            loss = self.loss_from_embeddings([torch.cat(chunks, dim=0) for chunks in chunked_embeddings], labels).detach()

        state = {
            "sentence_features": sentence_features,
            "random_states": random_states,
            "chunked_embeddings": chunked_embeddings,
            "labels": labels,
        }
        detached_loss = loss.clone().requires_grad_(True)
        detached_loss.register_hook(lambda grad_output: self._backward(grad_output, state))
        return detached_loss

    def _chunked_no_grad_forward(self, sentence_features: list[dict[str, torch.Tensor]]):
        """Embed every column in mini-batches without grad, capturing the RNG state of each chunk."""
        chunked_embeddings, random_states = [], []
        for features in sentence_features:
            chunks, states = [], []
            for start in range(0, self._batch_size(features), self.mini_batch_size):
                chunk = self._slice(features, start, start + self.mini_batch_size)
                states.append(RandContext(*[value for value in chunk.values() if isinstance(value, torch.Tensor)]))
                with torch.no_grad():
                    chunks.append(self.model(chunk)["sentence_embedding"].detach())
            chunked_embeddings.append(chunks)
            random_states.append(states)
        return chunked_embeddings, random_states

    def _backward(self, grad_output: torch.Tensor, state: dict) -> None:
        """Cache the gradient of the loss w.r.t. the embeddings, then re-forward each chunk with grad."""
        leaves = [torch.cat(chunks, dim=0).detach().requires_grad_(True) for chunks in state["chunked_embeddings"]]
        with torch.enable_grad():
            (self.loss_from_embeddings(leaves, state["labels"]) * grad_output).backward()
        cached_grads = [leaf.grad.detach() for leaf in leaves]

        with torch.enable_grad():
            for features, states, cached_grad in zip(state["sentence_features"], state["random_states"], cached_grads):
                offset = 0
                for chunk_index, random_state in enumerate(states):
                    chunk = self._slice(features, chunk_index * self.mini_batch_size, (chunk_index + 1) * self.mini_batch_size)
                    with random_state:
                        embeddings = self.model(chunk)["sentence_embedding"]
                    (embeddings * cached_grad[offset : offset + embeddings.size(0)].detach()).sum().backward()
                    offset += embeddings.size(0)
                    del embeddings

    @staticmethod
    def _batch_size(features: dict[str, Any]) -> int:
        """Number of sequences in a sentence_features dict, padded (B, L) or FA2-packed (cu_seq_lens_q)."""
        if "cu_seq_lens_q" in features:
            return len(features["cu_seq_lens_q"]) - 1
        for key in ("input_ids", "attention_mask"):
            if isinstance(features.get(key), torch.Tensor):
                return features[key].shape[0]
        for value in features.values():
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                return value.size(0)
        raise ValueError("No batched tensor found in sentence_features dict.")

    @classmethod
    def _slice(cls, features: dict[str, Any], begin: int, end: int) -> dict[str, Any]:
        """Slice sequences [begin:end], handling both padded and FA2-packed (varlen) layouts."""
        end = min(end, cls._batch_size(features))
        if "cu_seq_lens_q" not in features:
            return {k: v[begin:end] if isinstance(v, torch.Tensor) and v.dim() >= 1 else v for k, v in features.items()}

        cu_seq_lens = features["cu_seq_lens_q"]
        token_begin, token_end, total_tokens = int(cu_seq_lens[begin]), int(cu_seq_lens[end]), int(cu_seq_lens[-1])
        sliced_cu_seq_lens = cu_seq_lens[begin : end + 1] - cu_seq_lens[begin]
        seq_lens = sliced_cu_seq_lens[1:] - sliced_cu_seq_lens[:-1]

        sliced = {}
        for key, value in features.items():
            if key in ("cu_seq_lens_q", "cu_seq_lens_k"):
                sliced[key] = sliced_cu_seq_lens
            elif key in ("max_length_q", "max_length_k"):
                sliced[key] = int(seq_lens.max()) if seq_lens.numel() > 0 else 0
            elif key == "seq_idx":
                sliced[key] = value[..., token_begin:token_end] - begin
            elif isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[-1] == total_tokens:
                sliced[key] = value[..., token_begin:token_end]
            else:
                sliced[key] = value
        return sliced


def mini_batch_size_per_split(splits: list[str], default: int, overrides: list[str] | None) -> dict[str, int | None]:
    """Resolve '<split_prefix>:<mini_batch_size>' overrides per split, where 0 means no GradCache."""
    parsed = {}
    for override in overrides or []:
        prefix, _, value = override.partition(":")
        if not prefix.strip() or not value.strip():
            raise ValueError(f"Invalid override '{override}'. Expected '<split_prefix>:<mini_batch_size>'.")
        parsed[prefix.strip()] = int(value) or None

    def resolve(split: str) -> int | None:
        return next((size for prefix, size in parsed.items() if split == prefix or split.startswith(f"{prefix}_")), default or None)

    return {split: resolve(split) for split in splits}


class KDToContrastive:
    """Converts the HF datasets with reranking scores into a contrastive one with query-positive-negatives format.

    Rows keep the teacher's reranker scores of the kept documents, in the same order as the
    positive and negative columns, so the collator can build the KL-div targets.
    """

    def __init__(
        self,
        queries: Dataset,
        documents: Dataset,
        num_negatives: int = 10,
        nv_threshold: float = 0.95,
    ) -> None:
        self.queries = dict(zip(queries["query_id"], queries["query"]))
        self.documents = dict(zip(documents["document_id"], documents["document"]))
        self.num_negatives = num_negatives
        self.nv_threshold = nv_threshold

    def negative_indices(self, example) -> list[int]:
        """Documents scored clearly below the positive and with a non-empty text, capped at num_negatives."""
        threshold = self.nv_threshold * example["scores"][0]
        return [
            i
            for i in range(1, len(example["document_ids"]))
            if example["scores"][i] < threshold and self.documents.get(example["document_ids"][i], "").strip()
        ][: self.num_negatives]

    def has_enough_negatives(self, example) -> bool:
        return bool(
            example["document_ids"]
            and self.queries.get(example["query_id"], "").strip()
            and self.documents.get(example["document_ids"][0], "").strip()
            and len(self.negative_indices(example)) >= self.num_negatives
        )

    def map_to_query_positive_negatives(self, example) -> dict[str, Any]:
        negatives = self.negative_indices(example)
        return {
            "query": self.queries[example["query_id"]],
            "positive": self.documents[example["document_ids"][0]],
            "teacher_scores": [float(example["rerank_scores"][i]) for i in [0, *negatives]],
            **{f"negative_{n}": self.documents[example["document_ids"][i]] for n, i in enumerate(negatives)},
        }


@dataclass
class SentenceTransformerDataCollatorSampleNeg(SentenceTransformerDataCollator):
    """Collator that samples k negatives from available negative columns per batch.

    The sampled negatives also select which teacher_scores end up in the KL-div label, kept as a
    (batch_size, 1 + k) tensor ordered as [positive, sampled negatives].
    """

    num_negatives: int | None = field(default=None)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        has_teacher_scores = "teacher_scores" in column_names
        if has_teacher_scores:
            column_names.remove("teacher_scores")

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        if not has_teacher_scores:
            for label_column in self.valid_label_columns:
                if label_column in column_names:
                    batch["label"] = torch.tensor([row[label_column] for row in features])
                    column_names.remove(label_column)
                    break

        router_mapping = self._resolve_router_mapping(batch)
        prompts = self._resolve_prompts(batch)

        negative_columns = [column for column in column_names if column.startswith("negative_")]
        other_columns = [column for column in column_names if not column.startswith("negative_")]

        if self.num_negatives is not None and negative_columns:
            negative_columns = random.sample(negative_columns, min(self.num_negatives, len(negative_columns)))

        # teacher_scores are stored as [positive, negative_0, ...], so index them by the sampled negatives
        if has_teacher_scores:
            indices = [0] + [int(column.rsplit("_", 1)[-1]) + 1 for column in negative_columns]
            batch["label"] = torch.tensor([[row["teacher_scores"][i] for i in indices] for row in features])

        for column_name in other_columns + negative_columns:
            task = router_mapping.get(column_name, None)
            prompt = self._get_prompt_for_column(prompts, column_name) if prompts else None

            sample = features[0][column_name]
            text_key = ("document" if "document" in sample else "query") if isinstance(sample, dict) else None
            inputs = [row[column_name][text_key] if text_key else row[column_name] for row in features]

            for key, value in self.preprocess_fn(inputs, prompt=prompt, task=task).items():
                batch[f"{column_name}_{key}"] = value

        return batch


def load_train_datasets(cache_dir: str) -> DatasetDict:
    """Build/load the code and multilingual reranked KD splits, one cached Dataset per split.

    Every split shares the row schema (query, positive, negative_0..k, teacher_scores), so one
    builder covers them all and only the source repo differs: one repo per language for the
    <dataset>_<language> splits, one for code, one for the code-edit splits.
    """
    languages = ["en", "de", "it", "pt", "ar", "fr", "es", "sv", "no"]
    code_splits = [
        "apps", "synthetictext2sql", "cosqa", "codefeedbackst", "codefeedbackmt",
        "stackoverflowqa", "codetranscontest", "codetransdl",
        *[f"CodeSearchNet{ccr}_{language}" for language in ["go", "java", "javascript", "php", "python", "ruby"] for ccr in ["", "_ccr"]],
        *[f"CodeEditSearch_{language}" for language in ["c", "cpp", "go", "java", "javascript", "php", "python", "ruby", "rust", "scala", "shell", "swift", "typescript"]],
    ]
    multilingual_splits = [
        *[f"{split}_{language}" for split in ["hotpotqa", "nq", "msmarco", "fever", "squadv2", "fiqa", "trivia"] for language in languages],
        *[f"miracl_{language}" for language in ["ar", "en", "es", "fr"]],
        *[f"mldr_{language}" for language in ["en", "de", "es", "fr", "it", "pt", "ar"]],
    ]

    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
    for split, is_multilingual in [*[(s, False) for s in code_splits], *[(s, True) for s in multilingual_splits]]:
        try:
            dataset = Dataset.load_from_disk(f"{cache_dir}/{split}")
            if "teacher_scores" not in dataset.column_names:
                raise FileNotFoundError(f"cached {split} predates teacher_scores, rebuilding")
            print(f"Loaded dataset from disk: {split}")
            # Older caches carry a label column the KL-div collator would fight over
            if "label" in dataset.column_names:
                dataset = dataset.remove_columns("label")
        except FileNotFoundError:
            suffix = split.rsplit("_", 1)[-1] if is_multilingual else ("code-edit" if split.startswith("CodeEditSearch") else "code")
            repo = f"lightonai/embeddings-fine-tuning-filtered-{suffix}"
            print(f"Creating dataset: {split} (from {repo})")
            load = lambda config: load_dataset(repo, name=config, data_files=f"{config}/{split}-*", split="train", verification_mode="no_checks")
            scores = load("scores")
            # Baked into the cache — delete {cache_dir}/{split} after changing
            processor = KDToContrastive(load("queries"), load("documents"), num_negatives=10, nv_threshold=0.95)
            dataset = scores.filter(
                processor.has_enough_negatives,
                desc=f"Filtering examples with <10 negatives ({split})",
            ).map(
                processor.map_to_query_positive_negatives,
                remove_columns=scores.column_names,
                desc=f"Creating query-positive-negatives dataset ({split})",
            )
            print(f"  [{split}] {len(dataset):,}/{len(scores):,} rows kept, saving to disk")
            dataset.save_to_disk(f"{cache_dir}/{split}")
        train_dataset[split] = dataset
    return train_dataset


def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(
        description="Fine-tune dense model with Matryoshka contrastive + KL-div distillation"
    )

    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.02,
        help="Contrastive (MNRL) temperature, loss scale = 1 / temperature. Not used by the KL-div term",
    )
    parser.add_argument(
        "--student_temperature",
        type=float,
        default=0.02,
        help="KL distillation loss only: softmax temperature of the student cosine similarities",
    )
    parser.add_argument(
        "--teacher_temperature",
        type=float,
        default=0.1,
        help="KL distillation loss only: softmax temperature of the teacher reranker scores",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument(
        "--mini_batch_size_by_prefix",
        type=str,
        nargs="*",
        default=["mldr:8"],
        help="Per-split GradCache overrides as '<split_prefix>:<mini_batch_size>', 0 disables GradCache",
    )
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--kldiv_weight", type=float, default=1.0)
    parser.add_argument("--matryoshka_dims", type=int, nargs="+", default=[768, 512, 256, 128])
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dataset sampling exponent: probability proportional to size ** alpha (1.0 = proportional, 0 = uniform)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lightonai/mDenseOn-unsupervised",
    )
    parser.add_argument(
        "--stop_at_step",
        type=int,
        default=-1,
        help="Stop training after this many steps (set to -1 to disable)",
    )
    parser.add_argument("--eval_steps", type=int, default=40000)
    parser.add_argument("--save_steps", type=int, default=20000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "--datasets_cache_dir",
        type=str,
        default="./cache/multilingual",
        help="Cache root: one subdirectory per built split, plus the HF hub/datasets caches",
    )

    args = parser.parse_args()

    # Build/cache the datasets on the main process only; other ranks wait,
    # then load the cached version from disk.
    with accelerator.main_process_first():
        train_dataset = load_train_datasets(cache_dir=args.datasets_cache_dir)
    print(train_dataset)

    model_shortname = args.model_name.rstrip("/").split("/")[-1]
    if model_shortname.startswith("checkpoint"):  # checkpoints are named after their run, not themselves
        model_shortname = args.model_name.rstrip("/").split("/")[-2]

    lr_str = f"{args.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e")
    temp_str = "-".join(
        f"{name}{str(temperature).replace('.', '')}"
        for name, temperature in [("ctemp", args.temperature), ("stemp", args.student_temperature), ("ttemp", args.teacher_temperature)]
    )
    run_name = (
        f"{model_shortname}-finetune-"
        f"lr{lr_str}-{temp_str}-"
        f"bs{args.batch_size}-"
        f"nv-retriever-0.95-10negs-7sampled-"
        f"kldiv{args.kldiv_weight}_contrastive{args.contrastive_weight}"
    )
    run_name += f"-alpha{args.alpha}" + ("-noMRL" if len(args.matryoshka_dims) == 1 else "-MRL")

    output_dir = f"{args.output_path}/{model_shortname}/{run_name}"
    mini_batch_sizes = mini_batch_size_per_split(list(train_dataset), args.mini_batch_size, args.mini_batch_size_by_prefix)

    print(f"\n{'=' * 60}")
    print("Training Configuration:")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Contrastive Temperature: {args.temperature} (scale={1 / args.temperature:.1f})")
    print(f"KL Distillation Temperatures: student={args.student_temperature}, teacher={args.teacher_temperature}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Mini Batch Size: {args.mini_batch_size or 'GradCache disabled'} (overrides: {args.mini_batch_size_by_prefix})")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Stop at Step: {args.stop_at_step if args.stop_at_step > 0 else 'Disabled'}")
    print(f"Loss: MNRL (weight={args.contrastive_weight}) + KL-div (weight={args.kldiv_weight})")
    print(f"Matryoshka Dims: {args.matryoshka_dims} (uniformly averaged)")
    print(f"Dataset Sampling: alpha={args.alpha} (1.0 = proportional to size, 0 = uniform)")
    print(f"Resume from Checkpoint: {args.resume_from_checkpoint or '<none>'}")
    print(f"Run Name: {run_name}")
    print(f"Output Dir: {output_dir}")
    print(f"{'=' * 60}\n")

    # load in fp32 to avoid errors but training runs in bf16 with flash attention 2
    model = SentenceTransformer(
        model_name_or_path=args.model_name,
        model_kwargs={"attn_implementation": "flash_attention_2", "dtype": torch.float32},
    )
    model.max_seq_length = 8192

    dev_evaluator = MultilingualNanoBEIREvaluator(
        query_prompts="query: ",
        corpus_prompts="document: ",
    )

    # One loss per split so the long-document splits can use a smaller GradCache chunk
    train_loss = {
        split: MatryoshkaMultipleNegativesRankingKLDivLoss(
            model=model,
            scale=1 / args.temperature,
            student_temperature=args.student_temperature,
            teacher_temperature=args.teacher_temperature,
            contrastive_weight=args.contrastive_weight,
            kldiv_weight=args.kldiv_weight,
            matryoshka_dims=args.matryoshka_dims,
            mini_batch_size=mini_batch_size,
        )
        for split, mini_batch_size in mini_batch_sizes.items()
    }

    callbacks = []
    if args.stop_at_step > 0:
        callbacks.append(StopAtStepCallback(stop_at_step=args.stop_at_step))

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=partial(MultinomialBatchSampler, alpha=args.alpha),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        fp16=False,
        bf16=True,
        seed=42,
        run_name=run_name,
        learning_rate=args.learning_rate,
        dataloader_num_workers=8,
        dataloader_drop_last=True,
        accelerator_config={
            "split_batches": True,
        },
        eval_on_start=False,
    )

    data_collator = SentenceTransformerDataCollatorSampleNeg(
        preprocess_fn=model.preprocess,
        num_negatives=7,  # sampled per step from the stored 10
        prompts={
            "query": "query: ",
            **{
                k: "document: "
                for k in ["positive", *[f"negative_{i}" for i in range(50)]]
            },
        },
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        callbacks=callbacks if callbacks else None,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(f"{output_dir}/final")

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Model saved to: {output_dir}/final")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
