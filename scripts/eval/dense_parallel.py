#!/usr/bin/env python3
"""Task-parallel MTEB evaluation for dense models: one task per GPU.

Each task runs as an independent single-GPU subprocess, up to one per GPU in
parallel, with the same token-budget batch packing as scripts/eval/dense_sequential.py (budget halved
on CUDA OOM). Dataset loading, encoding, and scoring of different tasks overlap
fully, so this is the fastest option for sweeps of small tasks where per-task CPU
work dominates. For large encode-bound corpora use scripts/eval/dense_sequential.py, which puts every
GPU on a single task.

Results use the same layout as scripts/eval/dense_sequential.py (one subfolder per model), so both
scripts can share a results folder and completed tasks are skipped on rerun.

Usage:
    python scripts/eval/dense_parallel.py \
        --gpus 0,1,2,3,4,5,6,7 --bf16 \
        --results_folder results/dense \
        --models lightonai/mDenseOn \
        --tasks TRECCOVID FiQA2018 SciFact QuoraRetrieval
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dense_sequential import (
    get_st_model,
    load_model,
    path_to_folder_name,
    progress,
    setup_st_forward_compat,
    task_meta_subsets,
    task_remaining,
    token_budget_batches,
)


def setup_budget_encode(st_model, char_budget: int) -> None:
    """Wrap the model's encode with single-process token-budget batch packing."""

    import numpy as np
    import torch

    original_encode = st_model.encode

    def budget_encode(sentences, **kwargs):
        if not isinstance(sentences, list) or len(sentences) <= 1:
            return original_encode(sentences, **kwargs)
        batches = token_budget_batches(sentences, st_model.encode_char_budget)
        outs = [
            original_encode([sentences[i] for i in batch_ids],
                            **{**kwargs, "batch_size": len(batch_ids), "show_progress_bar": False})
            for batch_ids in batches
        ]
        # Invert the length-sorted packing so outputs line up with the input order.
        order = [i for batch_ids in batches for i in batch_ids]
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        if isinstance(outs[0], torch.Tensor):
            return torch.cat(outs)[inverse]
        if isinstance(outs[0], np.ndarray):
            return np.concatenate(outs, axis=0)[inverse]
        flat = [e for out in outs for e in out]
        return [flat[i] for i in inverse]

    st_model.encode_char_budget = char_budget
    st_model.encode = budget_encode


def run_worker(args: argparse.Namespace) -> None:
    """Run one task on the single visible GPU, halving the char budget on OOM."""

    import mteb
    import torch

    setup_st_forward_compat()
    task_name = args.worker
    model = load_model(args.models[0], bf16=args.bf16, fa2=args.fa2, local=args.local)
    st_model = get_st_model(model)
    if st_model is not None:
        setup_budget_encode(st_model, args.encode_char_budget)
    cache = mteb.ResultCache(cache_path=args.results_folder)

    while True:
        try:
            task = mteb.get_tasks(tasks=[task_name])[0]
            splits = list(task.metadata.eval_splits)
            if "test" in splits and len(splits) > 1:
                task.metadata.eval_splits = ["test"]
            mteb.evaluate(
                model,
                task,
                encode_kwargs={"show_progress_bar": False},
                cache=cache,
                overwrite_strategy="only-missing",
                show_progress_bar=False,
            )
            return
        except Exception as error:  # noqa: BLE001
            if st_model is None or "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            if st_model.encode_char_budget // 2 < 10_000:
                raise
            st_model.encode_char_budget //= 2
            print(f"RETRYING {task_name} (OOM) -> encode_char_budget={st_model.encode_char_budget}", flush=True)


def schedule(args: argparse.Namespace, gpus: list[str]) -> int:
    """Dispatch (model, task) jobs across GPUs, one subprocess per free GPU."""

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, str]] = []
    skipped = 0
    for model_path in args.models:
        model_results_dir = os.path.join(args.results_folder, path_to_folder_name(model_path))
        for task_name in args.tasks:
            subsets, splits = task_meta_subsets(task_name)
            if "test" in splits and len(splits) > 1:
                splits = ["test"]
            if task_remaining(model_results_dir, task_name, subsets, splits):
                jobs.append((model_path, task_name, model_results_dir))
            else:
                skipped += 1

    print("=" * 60)
    print("MTEB Dense Evaluation - TASK-PARALLEL (one task per GPU)")
    print(f"Models: {len(args.models)} | GPUs: {len(gpus)} | Jobs: {len(jobs)} | Skipped: {skipped}")
    print("=" * 60)

    running: dict[str, tuple[subprocess.Popen, str, str, float]] = {}
    passthrough = (["--bf16"] if args.bf16 else []) + (["--fa2"] if args.fa2 else []) \
        + (["--local"] if args.local else []) \
        + ["--encode_char_budget", str(args.encode_char_budget)]
    failed = 0
    while jobs or running:
        for gpu, (proc, model_path, task_name, started) in list(running.items()):
            if proc.poll() is None:
                continue
            elapsed = time.perf_counter() - started
            if proc.returncode == 0:
                progress(f"[GPU {gpu}] Completed: {model_path} / {task_name} in {elapsed:.1f}s")
            else:
                failed += 1
                progress(f"[GPU {gpu}] FAILED: {model_path} / {task_name} "
                         f"(exit {proc.returncode}, see {log_dir}/{path_to_folder_name(model_path)}_{task_name}.log)")
            del running[gpu]

        while jobs and (free := [g for g in gpus if g not in running]):
            gpu = free[0]
            model_path, task_name, model_results_dir = jobs.pop(0)
            log_path = log_dir / f"{path_to_folder_name(model_path)}_{task_name}.log"
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--worker", task_name,
                   "--models", model_path,
                   "--results_folder", model_results_dir,
                   *passthrough]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            progress(f"[GPU {gpu}] Starting: {model_path} / {task_name}")
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            running[gpu] = (proc, model_path, task_name, time.perf_counter())

        time.sleep(1)

    print(f"Done. failed={failed} | results: {args.results_folder}")
    return failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""

    repo = Path(__file__).resolve().parent.parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", help="Comma-separated GPU ids, e.g. 0,1,2,3,4,5,6,7")
    p.add_argument("--results_folder",
                   help="Folder for results; one subfolder per model is created under it")
    p.add_argument("--models", nargs="+", default=[],
                   help="Hub ids or local paths, e.g. lightonai/mDenseOn")
    p.add_argument("--tasks", nargs="+", default=[],
                   help="MTEB task names, e.g. SyntecRetrieval GermanDPR")
    p.add_argument("--encode_char_budget", type=int, default=3_000_000,
                   help="Chars per encode batch (length-sorted packing); lower if encoding OOMs")
    p.add_argument("--log_dir", default=str(repo / "logs" / "task_parallel"),
                   help="Directory for per-task worker logs")
    p.add_argument("--bf16", action="store_true",
                   help="Load models in BF16")
    p.add_argument("--fa2", action="store_true",
                   help="Load models with FlashAttention-2 (implies --bf16; requires flash-attn)")
    p.add_argument("--local", action="store_true",
                   help="Force loading all models via SentenceTransformer (for local checkpoints)")
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)

    args = p.parse_args(argv)
    required = ("results_folder", "models") if args.worker else ("gpus", "results_folder", "models", "tasks")
    missing = [n for n in required if not getattr(args, n)]
    if missing:
        p.error("the following arguments are required: " + ", ".join("--" + n for n in missing))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
        return

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    Path(args.results_folder).mkdir(parents=True, exist_ok=True)
    if schedule(args, gpus):
        sys.exit(1)


if __name__ == "__main__":
    main()
