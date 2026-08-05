"""MTEB evaluation of late-interaction (ColBERT) models, with multi-GPU encoding
through accelerate and PLAID retrieval through fast-plaid.

Encoding is distributed across all GPUs with token-budget batch packing: texts are
sorted by length and packed into variable-size batches under a character budget, so
short documents form large batches and long documents small ones without per-task
tuning. Indexing and retrieval then run on the main process with fast-plaid.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch eval/late_interaction.py \
        --models lightonai/mLateOn \
        --tasks MIRACLRetrievalHardNegatives MultiLongDocRetrieval \
        --results_folder results/late_interaction

Models and tasks run sequentially; results accumulate under --results_folder and
existing (task, subset) results are skipped, so reruns only compute what is missing.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import heapq
import json
import logging
import os
import pickle
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mteb
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.cache import ResultCache
from mteb.models.abs_encoder import get_prompt
from mteb.models.model_implementations.pylate_models import MultiVectorModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def token_budget_batches(
    texts: list[str], char_budget: int, max_batch: int = 2048,
    attn_budget: float = 2.5e9,
) -> list[list[int]]:
    """Length-sorted batches packed to a character budget (chars ~ 4x tokens).

    Ascending lengths with a budget bound (peak activation memory), a 2x
    padding-waste bound, and a batch*length^2 bound (quadratic attention memory
    on long documents when flash attention is unavailable), so short docs pack
    huge batches and long docs get small ones without per-task tuning.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    batches: list[list[int]] = []
    current: list[int] = []
    char_sum = 0
    for idx in order:
        length = max(len(texts[idx]), 1)
        padded = (len(current) + 1) * length
        if current and (
            padded > char_budget
            or padded * length > attn_budget
            or padded > 2 * (char_sum + length)
            or len(current) >= max_batch
        ):
            batches.append(current)
            current, char_sum = [], 0
        current.append(idx)
        char_sum += length
    if current:
        batches.append(current)
    return batches


def to_fp16_numpy(embs) -> list[np.ndarray]:
    """One flat device-to-host copy for a list of variable-length embeddings."""
    if not isinstance(embs, (list, tuple)):
        embs = list(embs)
    if not embs:
        return []
    if not isinstance(embs[0], torch.Tensor):
        return [np.asarray(e, dtype=np.float16) for e in embs]
    lengths = [e.shape[0] for e in embs]
    flat = torch.cat(list(embs), dim=0).half().cpu().numpy()
    out, offset = [], 0
    for length in lengths:
        out.append(flat[offset : offset + length])
        offset += length
    return out


def gather_chunk_bounds(
    texts: list[str], max_tokens: int, embed_dim: int, budget_bytes: float
) -> list[int]:
    """Boundaries cutting texts into chunks of ~budget_bytes of fp16 embeddings.

    Sizes are estimated at one token per character (worst case), capped at
    max_tokens, so every gather round moves a bounded amount of data to the main
    process regardless of document length. All ranks compute identical bounds
    from the global text list, keeping the gather collectives in lockstep.
    """
    bounds, acc = [0], 0.0
    for i, text in enumerate(texts):
        acc += min(len(text), max_tokens) * embed_dim * 2
        if acc >= budget_bytes:
            bounds.append(i + 1)
            acc = 0.0
    if bounds[-1] < len(texts) or len(bounds) == 1:
        bounds.append(len(texts))
    return bounds


class AccelerateMultiVectorModel(MultiVectorModel):
    """MultiVectorModel that encodes across GPUs with accelerate and searches with fast-plaid."""

    def __init__(
        self,
        accelerator: Accelerator,
        gather_gb: float = 4.0,
        encode_char_budget: int = 3_000_000,
        index_gpu_memory: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.accelerator = accelerator
        self.encode_char_budget = encode_char_budget
        self.gather_gb = gather_gb
        # fast-plaid index placement at search time: auto = by free VRAM, low/medium/high = fixed tiers.
        self.index_gpu_memory = index_gpu_memory
        self.model.to(accelerator.device)
        self.model.eval()

    def _log_embedding_stats(self, label: str, embeddings) -> None:
        if not self.accelerator.is_main_process or not embeddings:
            return
        lengths = np.asarray([int(emb.shape[0]) for emb in embeddings], dtype=np.int64)
        p50, p90, p95, p99, p100 = np.percentile(lengths, [50, 90, 95, 99, 100])
        print(
            f"{label} token lengths: n={len(lengths):,}, "
            f"min={lengths.min()}, p50={p50:.0f}, p90={p90:.0f}, "
            f"p95={p95:.0f}, p99={p99:.0f}, max={p100:.0f}",
            flush=True,
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        prompt = get_prompt(self.model_prompts, task_metadata, prompt_type)
        all_texts = [text for batch in inputs for text in batch["text"]]

        max_tokens = (
            self.model.query_length
            if prompt_type == PromptType.query
            else self.model.document_length
        ) or 8192
        embed_dim = self.model.get_sentence_embedding_dimension() or 128
        bounds = gather_chunk_bounds(
            all_texts, max_tokens, embed_dim, self.gather_gb * 1e9
        )
        if self.accelerator.is_main_process:
            print(
                f"Encoding {len(all_texts)} texts with {self.accelerator.num_processes} GPUs "
                f"in {len(bounds) - 1} gather round(s) (prompt_type={prompt_type})",
                flush=True,
            )
        rounds = [
            list(range(start, end))[
                self.accelerator.process_index :: self.accelerator.num_processes
            ]
            for start, end in zip(bounds, bounds[1:])
        ]

        all_embeddings = [] if self.accelerator.is_main_process else None
        all_indices = [] if self.accelerator.is_main_process else None

        desc = f"GPU {self.accelerator.process_index} {'Queries' if prompt_type == PromptType.query else 'Docs'}"
        pbar = tqdm(
            total=sum(len(r) for r in rounds),
            desc=desc,
            position=self.accelerator.process_index,
            leave=True,
            disable=not self.accelerator.is_local_main_process,
        )

        for chunk_indices in rounds:
            chunk_texts = [all_texts[i] for i in chunk_indices]

            chunk_embeddings = [None] * len(chunk_texts)
            with torch.no_grad():
                for batch_ids in token_budget_batches(
                    chunk_texts, self.encode_char_budget
                ):
                    batch = [chunk_texts[i] for i in batch_ids]
                    try:
                        embs = self.model.encode(
                            batch,
                            prompt=prompt,
                            is_query=prompt_type == PromptType.query,
                            convert_to_tensor=True,
                            batch_size=len(batch),
                            show_progress_bar=False,
                        )
                    except torch.OutOfMemoryError:
                        # Halve the batch once; the budget is conservative so this is rare.
                        torch.cuda.empty_cache()
                        half = max(1, len(batch) // 2)
                        embs = []
                        for j in range(0, len(batch), half):
                            embs.extend(
                                self.model.encode(
                                    batch[j : j + half],
                                    prompt=prompt,
                                    is_query=prompt_type == PromptType.query,
                                    convert_to_tensor=True,
                                    batch_size=half,
                                    show_progress_bar=False,
                                )
                            )
                    for local_i, emb in zip(batch_ids, to_fp16_numpy(embs)):
                        chunk_embeddings[local_i] = emb
                    pbar.update(len(batch))

            gathered_embs = gather_object(chunk_embeddings)
            gathered_idxs = gather_object(chunk_indices)
            if self.accelerator.is_main_process:
                all_embeddings.extend(gathered_embs)
                all_indices.extend(gathered_idxs)
                pbar.set_postfix({"gathered": len(all_embeddings)})

            del chunk_embeddings, gathered_embs, gathered_idxs
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()

        pbar.close()

        if not self.accelerator.is_main_process:
            return []
        ordered = [None] * len(all_texts)
        for idx, emb in zip(all_indices, all_embeddings):
            ordered[idx] = emb
        return ordered

    def search(
        self,
        queries,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        top_k,
        encode_kwargs,
        top_ranked=None,
        num_proc=None,
    ):
        """Encode queries and corpus on all GPUs, then index and retrieve on the main process."""
        from mteb._create_dataloaders import create_dataloader

        queries_dataloader = create_dataloader(
            queries,
            task_metadata=task_metadata,
            prompt_type=PromptType.query,
            batch_size=encode_kwargs.get("batch_size", 32),
        )
        query_embeddings = self.encode(
            queries_dataloader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        self._log_embedding_stats("Query embeddings", query_embeddings)
        query_idx_to_id = {i: row["id"] for i, row in enumerate(queries)}

        corpus_dataloader = create_dataloader(
            self.task_corpus,
            task_metadata=task_metadata,
            prompt_type=PromptType.document,
            batch_size=encode_kwargs.get("batch_size", 32),
        )
        corpus_embeddings = self.encode(
            corpus_dataloader,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )
        self._log_embedding_stats("Corpus embeddings", corpus_embeddings)
        self.accelerator.wait_for_everyone()

        # Free the encoder before the long single-process indexing and search work.
        self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.accelerator.wait_for_everyone()

        safe_index_name = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in self._index_name
        )
        results_path = os.path.join(
            tempfile.gettempdir(), f"mteb_plaid_results_{safe_index_name}.pkl"
        )
        ready_flag = results_path + ".ready"
        failed_flag = results_path + ".failed"

        if self.accelerator.is_main_process:
            for path in (ready_flag, results_path, failed_flag):
                if os.path.exists(path):
                    os.remove(path)
        # Quick NCCL sync before the long single-process work begins.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            try:
                if top_ranked is not None:
                    result_heaps = self._rerank(
                        query_idx_to_id, query_embeddings, corpus_embeddings, top_ranked, top_k
                    )
                else:
                    result_heaps = self._index_and_retrieve(
                        query_idx_to_id, query_embeddings, corpus_embeddings, top_k
                    )

                results = {qid: {} for qid in query_idx_to_id.values()}
                for qid in result_heaps:
                    for score, corpus_id in result_heaps[qid]:
                        results[qid][corpus_id] = score

                # Write results and signal ready (no NCCL involved).
                with open(results_path, "wb") as f:
                    pickle.dump(results, f)
                with open(ready_flag, "w") as f:
                    f.write("done")
            except BaseException as exc:
                # CUDA OOM in fast-plaid Rust threads deadlocks Python cleanup; signal failure and force-exit.
                traceback.print_exc()
                try:
                    with open(failed_flag, "w") as f:
                        f.write(f"{type(exc).__name__}: {exc}\n")
                except Exception:
                    pass
                os._exit(1)
        else:
            # Poll for results or failure flag instead of blocking in an NCCL barrier.
            while not os.path.exists(ready_flag):
                if os.path.exists(failed_flag):
                    os._exit(1)
                time.sleep(1)
            with open(results_path, "rb") as f:
                results = pickle.load(f)

        self.accelerator.wait_for_everyone()
        self.model.to(self.accelerator.device)
        return results

    def _rerank(self, query_idx_to_id, query_embeddings, corpus_embeddings, top_ranked, top_k):
        """Rerank the provided candidate documents with MaxSim (no index needed)."""
        from pylate import rank

        doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(self.task_corpus)}
        result_heaps = {qid: [] for qid in query_idx_to_id.values()}
        for q_idx, qid in query_idx_to_id.items():
            candidate_ids = [
                d for d in top_ranked.get(qid, []) if d in doc_id_to_idx
            ]
            if not candidate_ids:
                continue
            reranked = rank.rerank(
                documents_ids=[candidate_ids],
                queries_embeddings=[query_embeddings[q_idx]],
                documents_embeddings=[
                    [corpus_embeddings[doc_id_to_idx[d]] for d in candidate_ids]
                ],
            )
            for item in reranked[0]:
                heapq.heappush(
                    result_heaps[qid], (float(item["score"]), str(item["id"]))
                )
            if len(result_heaps[qid]) > top_k:
                result_heaps[qid] = heapq.nlargest(top_k, result_heaps[qid])
        return result_heaps

    def _index_and_retrieve(self, query_idx_to_id, query_embeddings, corpus_embeddings, top_k):
        """Build the PLAID index from the corpus embeddings and retrieve top_k per query."""
        from fast_plaid import search as fast_plaid_search
        from pylate import indexes, retrieve

        doc_ids = [str(x) for x in self.task_corpus["id"]]
        index = indexes.PLAID(
            index_folder=str(self._index_dir),
            index_name=self._index_name,
            use_triton=False,
            batch_size="auto",
            **self.index_kwargs,
        )
        print(
            "PLAID search settings: "
            f"n_full_scores={index._index.n_full_scores}, "
            f"n_ivf_probe={index._index.n_ivf_probe}, "
            f"top_k={top_k}, index_gpu_memory={self.index_gpu_memory}",
            flush=True,
        )

        print(f"Building PLAID index for {len(doc_ids):,} documents...", flush=True)
        index_start = time.perf_counter()
        index.add_documents(
            documents_ids=doc_ids, documents_embeddings=corpus_embeddings
        )
        print(f"PLAID index ready in {time.perf_counter() - index_start:.1f}s.", flush=True)

        # Reload the on-disk index for search, freeing temporary GPU state from construction.
        fast_plaid_index_path = index._index.fast_plaid_index_path
        index._index.fast_plaid.close()
        del corpus_embeddings
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        index._index.fast_plaid = fast_plaid_search.FastPlaid(
            index=fast_plaid_index_path, index_gpu_memory=self.index_gpu_memory
        )

        print(
            f"Starting PLAID search for {len(query_embeddings):,} queries (top_k={top_k})...",
            flush=True,
        )
        search_start = time.perf_counter()
        retriever = retrieve.ColBERT(index=index)
        scores = retriever.retrieve(queries_embeddings=query_embeddings, k=top_k)
        print(f"PLAID search finished in {time.perf_counter() - search_start:.1f}s.", flush=True)

        result_heaps = {qid: [] for qid in query_idx_to_id.values()}
        for q_idx, qid in query_idx_to_id.items():
            for item in scores[q_idx]:
                heapq.heappush(
                    result_heaps[qid], (float(item["score"]), str(item["id"]))
                )

        # Release PLAID GPU memory before the encoder moves back.
        index._index.fast_plaid.close()
        del scores, index
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return result_heaps


def build_model_meta(model_path: str, name: str | None = None) -> ModelMeta:
    """Construct a ModelMeta for a PyLate ColBERT checkpoint (local dir or HF Hub repo)."""
    path = Path(model_path)
    is_local = path.is_dir()

    if name is None:
        name = f"local/{path.parent.name}__{path.name}" if is_local else model_path
    elif "/" not in name:
        name = f"local/{name}"

    if is_local:
        backbone_cfg = json.loads((path / "config.json").read_text())
        dense_cfg_paths = [
            p / "config.json"
            for p in sorted(path.glob("*_Dense"))
            if p.is_dir() and (p / "config.json").exists()
        ]
        last_dense_cfg_text = dense_cfg_paths[-1].read_text() if dense_cfg_paths else None
    else:
        from huggingface_hub import HfApi, hf_hub_download

        backbone_cfg = json.loads(
            Path(hf_hub_download(model_path, "config.json")).read_text()
        )
        dense_files = sorted(
            f for f in HfApi().list_repo_files(model_path)
            if f.endswith("_Dense/config.json")
        )
        last_dense_cfg_text = (
            Path(hf_hub_download(model_path, dense_files[-1])).read_text()
            if dense_files
            else None
        )

    max_tokens = backbone_cfg.get("max_position_embeddings")
    embed_dim = (
        json.loads(last_dense_cfg_text).get("out_features")
        if last_dense_cfg_text
        else None
    )

    return ModelMeta(
        loader=MultiVectorModel,
        name=name,
        revision="local",
        release_date=None,
        languages=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=max_tokens,
        embed_dim=embed_dim,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["PyLate", "ColBERT", "Sentence Transformers", "safetensors"],
        reference=None,
        similarity_fn_name=ScoringFunction.MAX_SIM,
        use_instructions=False,
        training_datasets=None,
        model_type=["late-interaction"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate late-interaction models on MTEB tasks with accelerate"
    )
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model names or paths")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="MTEB task names")
    parser.add_argument(
        "--results_folder", type=str, default="results/late_interaction",
        help="MTEB result cache directory",
    )
    parser.add_argument(
        "--encode_char_budget", type=int, default=3_000_000,
        help="Chars per encode batch (length-sorted packing); lower if encoding OOMs",
    )
    parser.add_argument(
        "--gather_gb", type=float, default=4.0,
        help="Estimated fp16 embedding GB gathered to the main process per round",
    )
    parser.add_argument(
        "--index_gpu_memory", type=str, default="auto",
        choices=["auto", "low", "medium", "high"],
        help="fast-plaid index placement at search time (auto = by free VRAM)",
    )
    parser.add_argument(
        "--index_dir", type=str, default=None, help="PLAID index root directory"
    )
    parser.add_argument(
        "--query_length", type=int, default=None,
        help="Query max length (model config default if omitted)",
    )
    parser.add_argument(
        "--document_length", type=int, default=None,
        help="Document max length (model config default if omitted)",
    )
    parser.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated language codes to filter the task",
    )
    parser.add_argument(
        "--fa2", action="store_true",
        help="Encode with FlashAttention-2 instead of sdpa (requires flash-attn)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=20000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
        tasks = mteb.get_tasks(tasks=args.tasks, languages=languages)
        if accelerator.is_main_process:
            print(f"Filtered tasks to languages: {', '.join(languages)}")
    else:
        tasks = mteb.get_tasks(tasks=args.tasks)

    cache = ResultCache(args.results_folder)

    for model_name in args.models:
        # mteb puts the model name in result filenames; keep parent/base so long local paths fit NAME_MAX.
        model_path = Path(model_name)
        short_name = (
            f"{model_path.parent.name}/{model_path.name}"
            if model_path.parent.name
            else model_path.name
        )
        model_meta = build_model_meta(model_name, name=short_name)

        mv_kwargs = {}
        if args.index_dir:
            mv_kwargs["index_dir"] = args.index_dir
        if args.query_length is not None:
            mv_kwargs["query_length"] = args.query_length
        if args.document_length is not None:
            mv_kwargs["document_length"] = args.document_length

        model = AccelerateMultiVectorModel(
            accelerator=accelerator,
            gather_gb=args.gather_gb,
            encode_char_budget=args.encode_char_budget,
            index_gpu_memory=args.index_gpu_memory,
            model_name=model_name,
            trust_remote_code=True,
            model_kwargs={
                "dtype": torch.bfloat16,
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2" if args.fa2 else "sdpa",
            },
            index_kwargs={"override": True},
            # Override saved do_query_expansion; else queries are MASK-padded to the full query_length.
            do_query_expansion=False,
            **mv_kwargs,
        )
        model.mteb_model_meta = model_meta

        # Multi-subset tasks run one subset per evaluate call so a crash only loses the in-flight subset.
        for task in tasks:
            if accelerator.is_main_process:
                print(
                    f"=== Task: {task.metadata.name} | Model: {model_name} | "
                    f"GPUs: {accelerator.num_processes} | "
                    f"index_gpu_memory={args.index_gpu_memory} ==="
                )
            all_subsets = list(task.hf_subsets) if task.hf_subsets else []
            if len(all_subsets) > 1:
                # Preload once so per-subset evaluate calls don't drop and reload data.
                if not task.data_loaded:
                    task.load_data()
                for subset in all_subsets:
                    task.hf_subsets = [subset]
                    if accelerator.is_main_process:
                        print(f"--- Subset: {subset} ---")
                    # Barrier: keep other ranks out until rank 0 finishes writing the prior subset.
                    accelerator.wait_for_everyone()
                    mteb.evaluate(model, [task], cache=cache)
                    accelerator.wait_for_everyone()
                task.hf_subsets = all_subsets
            else:
                mteb.evaluate(model, [task], cache=cache)

        # Free the model on every rank before loading the next one.
        accelerator.wait_for_everyone()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print(f"=== Finished: {len(args.models)} model(s) x {len(args.tasks)} task(s) ===")


if __name__ == "__main__":
    main()
