"""
Contrastive fine-tuning of a dense retriever on the NV-Embed KD dataset
(query + 1 positive + stored hard negatives).

What to tweak
-------------
CLI:
  --model_name        Checkpoint to fine-tune (typically the pretraining
                      output).
  --learning_rate / --temperature / --num_train_epochs
  --batch_size        Also the pool of in-batch negatives.

Data build (load_train_datasets — these are baked into the on-disk cache,
delete {cache_dir}/{split} after changing any of them):
  splits              Which subsets of the KD dataset to train on.
  nv_threshold=0.99   A document counts as a negative only if its KD score
                      is < 0.99 * positive score (false-negative filter;
                      lower = stricter).
  num_negatives=50    Hard negatives stored per query.

In main():
  num_negatives=7 (collator)  Negatives actually sampled per step from the
                              stored 50.
  model.max_seq_length        Truncation length (512).
  prompts                     "query: " / "document: " prefixes; keep the
                              collator and the evaluator in sync.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(42)

import argparse

from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import MultiDatasetBatchSamplers
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


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


class KDToContrastive:
    """Converts a KD dataset into a contrastive one with query-positive-negatives format."""

    def __init__(
        self,
        queries: datasets.Dataset | datasets.DatasetDict,
        documents: datasets.Dataset | datasets.DatasetDict,
        split: str = "train",
        num_negatives: int = 32,
        nv_threshold: float = 0.95,
    ) -> None:
        if isinstance(queries, datasets.DatasetDict):
            self.queries = queries[split]
        else:
            self.queries = queries

        if isinstance(documents, datasets.DatasetDict):
            self.documents = documents[split]
        else:
            self.documents = documents

        self.num_negatives = num_negatives
        self.nv_threshold = nv_threshold

        self.queries_index = {
            query_id: i for i, query_id in enumerate(iterable=self.queries["query_id"])
        }
        self.documents_index = {
            document_id: i
            for i, document_id in enumerate(iterable=self.documents["document_id"])
        }

    def has_enough_negatives(self, example):
        scores = example["scores"]
        positive_score = scores[0]
        count = sum(
            1 for score in scores[1:] if score < self.nv_threshold * positive_score
        )
        return count >= self.num_negatives

    def map_to_query_positive_negatives(self, example):
        query_id = example["query_id"]
        document_ids = example["document_ids"]
        scores = example["scores"]

        query_text = self.queries[self.queries_index[query_id]]["query"]
        positive_id = document_ids[0]
        positive_text = self.documents[self.documents_index[positive_id]]["document"]
        positive_score = scores[0]

        row = {"query": query_text, "positive": positive_text}

        total_negatives = 0
        for i in range(1, len(document_ids)):
            if scores[i] < self.nv_threshold * positive_score:
                negative_id = document_ids[i]
                row[f"negative_{total_negatives}"] = self.documents[
                    self.documents_index[negative_id]
                ]["document"]
                total_negatives += 1
                if total_negatives >= self.num_negatives:
                    break

        return row


@dataclass
class SentenceTransformerDataCollatorSampleNeg(SentenceTransformerDataCollator):
    """Collator that samples k negatives from available negative columns per batch."""

    num_negatives: int | None = field(default=None)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        router_mapping = self.router_mapping
        if (
            router_mapping
            and isinstance(router_mapping, dict)
            and isinstance(next(iter(router_mapping.values())), dict)
        ):
            if "dataset_name" in batch and batch["dataset_name"] in router_mapping:
                router_mapping = router_mapping[batch["dataset_name"]]
            else:
                router_mapping = {}

        prompts = self.prompts
        if prompts and isinstance(prompts, dict):
            is_multi_dataset = "dataset_name" in batch
            if is_multi_dataset and batch["dataset_name"] in prompts:
                prompts = prompts[batch["dataset_name"]]
            elif isinstance(next(iter(prompts.values())), dict):
                if not is_multi_dataset:
                    raise ValueError(
                        "The prompts provided to the trainer are a nested dictionary. In this setting, the first "
                        "level of the dictionary should map to dataset names and the second level to column names. "
                        "However, as the provided dataset is a not a DatasetDict, no dataset names can be inferred. "
                        f"The keys to the provided prompts dictionary are {list(prompts.keys())!r}"
                    )
                else:
                    prompts = {}

        negative_columns = [col for col in column_names if col.startswith("negative_")]
        other_columns = [col for col in column_names if not col.startswith("negative_")]

        if self.num_negatives is not None and negative_columns:
            k = min(self.num_negatives, len(negative_columns))
            sampled_negatives = random.sample(negative_columns, k)
            columns_to_process = other_columns + sampled_negatives
        else:
            columns_to_process = column_names

        for column_name in columns_to_process:
            task = router_mapping.get(column_name, None)

            prompt = None
            if isinstance(prompts, str):
                prompt = prompts
            elif isinstance(prompts, dict) and column_name in prompts:
                prompt = prompts[column_name]

            if prompt:
                if self.include_prompt_lengths:
                    prompt_length = self._get_prompt_length(prompt, task=task)
                    if prompt_length is not None:
                        batch[f"{column_name}_prompt_length"] = torch.tensor(
                            [prompt_length] * len(features), dtype=torch.int
                        )
                inputs = [prompt + row[column_name] for row in features]
            else:
                inputs = [row[column_name] for row in features]

            tokenized = self.tokenize_fn(inputs, task=task)
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch


def load_train_datasets(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    train_dataset = DatasetDict()
    splits = ["fiqa", "trivia", "hotpotqa", "nq", "msmarco", "fever", "squadv2"]

    for split in splits:
        try:
            dataset = Dataset.load_from_disk(f"{cache_dir}/{split}")
            print("Loaded dataset from disk")
        except FileNotFoundError:
            print("Creating dataset")
            dataset = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup",
                name="scores",
                num_proc=144,
                split=split,
            )
            queries = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup",
                name="queries",
                num_proc=144,
                split=split,
            )
            documents = load_dataset(
                "lightonai/nv-embed-supervised-distill-dedup",
                name="documents",
                num_proc=144,
                split=split,
            )
            # Baked into the cache — delete {cache_dir}/{split} after changing
            processor = KDToContrastive(
                queries, documents, num_negatives=50, nv_threshold=0.99
            )
            dataset = dataset.filter(
                processor.has_enough_negatives,
                desc="Filtering examples with <50 negatives",
            ).map(
                processor.map_to_query_positive_negatives,
                remove_columns=dataset.column_names,
                desc="Creating query-positive-negatives dataset",
            )
            dataset.save_to_disk(f"{cache_dir}/{split}")
        train_dataset[split] = dataset
    return train_dataset


def main():
    accelerator = Accelerator()

    parser = argparse.ArgumentParser(
        description="Fine-tune dense model with configurable hyperparameters"
    )

    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model_name",
        type=str,
        default="lightonai/DenseOn-unsupervised",
    )
    parser.add_argument(
        "--stop_at_step",
        type=int,
        default=-1,
        help="Stop training after this many steps (set to -1 to disable)",
    )
    parser.add_argument("--eval_steps", type=int, default=3000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache/nv_embed_cached_supervised_99",
    )

    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    # Build/cache the datasets on the main process only; other ranks wait,
    # then load the cached version from disk.
    with accelerator.main_process_first():
        train_dataset = load_train_datasets(cache_dir=args.cache_dir)
    print(train_dataset)

    model_shortname = args.model_name.split("/")[-1]

    lr_str = f"{args.learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e")
    temp_str = str(args.temperature).replace(".", "")
    run_name = (
        f"{model_shortname}-finetune-"
        f"lr{lr_str}-temp{temp_str}-"
        f"bs{args.batch_size}-"
        f"nv-embed-0.99-7negs"
    )

    output_dir = f"output/{model_shortname}/{run_name}"

    print(f"\n{'=' * 60}")
    print("Training Configuration:")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Stop at Step: {args.stop_at_step if args.stop_at_step > 0 else 'Disabled'}")
    print(f"Run Name: {run_name}")
    print(f"Output Dir: {output_dir}")
    print(f"{'=' * 60}\n")

    model = SentenceTransformer(model_name_or_path=args.model_name)
    model.max_seq_length = 512

    dev_evaluator = NanoBEIREvaluator(
        query_prompts="query: ",
        corpus_prompts="document: ",
    )
    train_loss = MultipleNegativesRankingLoss(
        model, scale=1 / (args.temperature)
    )

    callbacks = []
    if args.stop_at_step > 0:
        callbacks.append(StopAtStepCallback(stop_at_step=args.stop_at_step))

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
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
        accelerator_config={
            "split_batches": True,
        },
    )

    data_collator = SentenceTransformerDataCollatorSampleNeg(
        tokenize_fn=model.tokenize,
        num_negatives=7,  # sampled per step from the stored 50
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

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Model saved to: {output_dir}/final")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
