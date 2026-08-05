#!/usr/bin/env python3
"""Multi-GPU MTEB evaluation for dense models.

Each task is parallelized across GPUs through a persistent SentenceTransformer
multi-process pool with token-budget batch packing: texts are sorted by length and
packed into variable-size batches under a character budget, so short documents form
large batches and long documents small ones without per-task batch-size tuning.
On CUDA OOM the budget is halved and the task retried; results are checkpointed per
subset, so a crash mid-task only re-runs the subsets still missing.

Usage:
    python eval/dense_sequential.py \
        --gpus 0,1,2,3,4,5,6,7 --bf16 \
        --results_folder results/dense \
        --models lightonai/mDenseOn \
        --tasks MIRACLRetrievalHardNegatives MultiLongDocRetrieval
"""

from __future__ import annotations

import argparse
import gc
import importlib
import itertools
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

Pool = dict[str, Any]
ModelWrapper = Any

# On sys.path here because spawn workers re-import this module before unpickling the
# model, letting them import transformers_modules.* for trust_remote_code models.
_HF_MODULES = os.environ.get("HF_MODULES_CACHE") or os.path.expanduser("~/.cache/huggingface/modules")
if os.path.isdir(_HF_MODULES) and _HF_MODULES not in sys.path:
    sys.path.append(_HF_MODULES)

logger = logging.getLogger(__name__)


def progress(msg: str) -> None:
    """Progress line: shown on the terminal AND written to the detail log."""

    print(msg, file=sys.stderr, flush=True)
    logger.info(msg)


def setup_logging(detail_log: str) -> None:
    """Route library chatter (datasets cache hits, MTEB internals, progress bars)
    to detail_log; keep the terminal for our own progress lines only."""

    import datasets  # import first so its own logger setup runs before we override it

    try:
        import transformers  # noqa: F401
    except Exception:  # noqa: BLE001
        pass

    Path(detail_log).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(detail_log)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    for name in ("datasets", "transformers", "sentence_transformers", "mteb",
                 "huggingface_hub", "fsspec", "filelock", "urllib3"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
        lg.setLevel(logging.INFO)

    datasets.disable_progress_bars()
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass


def path_to_folder_name(path: str) -> str:
    """Safe results folder name for a model path or hub id."""

    p = Path(path)
    if p.is_absolute():
        return f"{p.parent.name}__{p.name}"
    return path.replace("/", "_").replace(".", "_")


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


def find_result_json(search_root: str, task_name: str) -> Path | None:
    """Path to `{task_name}.json` under `search_root`, or None if not found.

    Results may be fragmented across revision subfolders; on multiple matches the
    last one in sorted-path order is returned (callers only need any one result).
    """

    root = Path(search_root)
    matches = sorted(root.rglob(f"{task_name}.json")) if root.is_dir() else []
    return matches[-1] if matches else None


def task_meta_subsets(task_name: str) -> tuple[list[str], list[str]]:
    """(hf_subsets, eval_splits) from task metadata (no data load)."""

    import mteb

    task = mteb.get_tasks(tasks=[task_name])[0]
    return list(task.hf_subsets), list(task.metadata.eval_splits)


def task_remaining(search_root: str, task_name: str, subsets: list[str], splits: list[str]) -> bool:
    """True if any (split, subset) of this task is still missing on disk.

    Single-subset tasks: an existing json means done. Multi-subset (multilingual)
    tasks resume per language — MTEB writes the json after each subset, so a
    crash/OOM mid-bench only re-runs the missing ones.
    """

    match = find_result_json(search_root, task_name)
    if match is None:
        return True
    if len(subsets) <= 1:
        return False
    scores = json.loads(match.read_text()).get("scores", {})
    needed = set(subsets)
    return any(not needed.issubset({r.get("hf_subset") for r in scores.get(sp, [])}) for sp in splits)


def load_model(model_path: str, bf16: bool = False, fa2: bool = False, local: bool = False) -> ModelWrapper:
    """Load an MTEB-compatible encoder for `model_path` (hub id or local path)."""

    import mteb

    model_kwargs: dict[str, Any] = {}
    if fa2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if bf16 or fa2:
        model_kwargs["torch_dtype"] = "bfloat16"

    progress(f"Loading model: {model_path}"
             + (" (FA2+BF16)" if fa2 else " (BF16)" if bf16 else ""))
    if local or model_path.startswith("/"):
        return SentenceTransformer(model_path, trust_remote_code=True, model_kwargs=model_kwargs)
    return mteb.get_model(model_path, trust_remote_code=True, model_kwargs=model_kwargs)


def get_st_model(wrapper: ModelWrapper) -> SentenceTransformer | None:
    """Underlying SentenceTransformer (for pool creation), or None."""

    if isinstance(wrapper, SentenceTransformer):
        return wrapper
    candidate = getattr(wrapper, "model", None)
    return candidate if isinstance(candidate, SentenceTransformer) else None


class _WorkerError:
    """Sentinel passed back through the output queue when a pool worker raises.

    Carries the error message so the collector can re-raise it in the main
    process instead of blocking forever on a worker that died (e.g. on OOM).
    """

    def __init__(self, message: str) -> None:
        self.message = message


def _oom_safe_worker(
    target_device: str, model: ModelWrapper, input_queue: Any, output_queue: Any
) -> None:
    """OOM-safe replacement for SentenceTransformer's pool worker.

    Stock ST only catches queue.Empty, so a CUDA OOM kills the worker and the main
    process blocks forever on the output queue. This variant catches any error,
    frees the CUDA cache, and returns a `_WorkerError` sentinel so the collector can
    turn it into a normal exception. Runs in a spawned process (hence module-level).
    """

    import torch

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent process drives shutdown
    while True:
        chunk_id, inputs, kwargs = input_queue.get()
        try:
            embeddings = model.encode(inputs, device=target_device, **kwargs)
            if isinstance(embeddings, torch.Tensor) and embeddings.device.type != "cpu":
                embeddings = embeddings.cpu()
            output_queue.put([chunk_id, embeddings])
        except Exception as e:  # noqa: BLE001
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            output_queue.put([chunk_id, _WorkerError(f"{type(e).__name__}: {e}")])


def _budget_multi_process(self: SentenceTransformer, inputs: list[str], show_progress_bar: bool = True,
                          input_was_string: bool = False, pool: Pool | None = None,
                          device: str | list[str] | None = None, chunk_size: int | None = None,
                          **encode_kwargs: Any) -> Any:
    """Token-budget replacement for SentenceTransformer._encode_multi_process.

    Packs `inputs` into length-sorted batches under `self.encode_char_budget` and
    dispatches each batch to a pool worker as a single forward pass, instead of
    fixed-count chunks at a fixed batch size. Outputs are re-ordered to match the
    input order. A `_WorkerError` from any worker is re-raised as a RuntimeError
    (so the caller can retry at a smaller budget) instead of hanging.
    """

    import numpy as np
    import torch

    convert_to_tensor = encode_kwargs.get("convert_to_tensor", False)
    convert_to_numpy = encode_kwargs.get("convert_to_numpy", False)
    encode_kwargs["show_progress_bar"] = False

    created_pool = pool is None and isinstance(device, list)
    if created_pool:
        pool = self.start_multi_process_pool(device)

    try:
        batches = token_budget_batches(inputs, self.encode_char_budget)
        input_queue, output_queue = pool["input"], pool["output"]
        for chunk_id, batch_ids in enumerate(batches):
            batch = [inputs[i] for i in batch_ids]
            input_queue.put([chunk_id, batch, {**encode_kwargs, "batch_size": len(batch)}])

        output_list = sorted([output_queue.get() for _ in range(len(batches))], key=lambda x: x[0])
        errors = [o[1].message for o in output_list if isinstance(o[1], _WorkerError)]
        if errors:
            raise RuntimeError(f"multi-process worker failed (e.g. CUDA OOM): {errors[0]}")
        if input_was_string:
            return output_list[0][1][0]

        embeddings = [o[1] for o in output_list]
        # Invert the length-sorted packing so outputs line up with the input order.
        sorted_order = [i for batch_ids in batches for i in batch_ids]
        inverse = np.empty(len(sorted_order), dtype=np.int64)
        inverse[sorted_order] = np.arange(len(sorted_order))
        if embeddings and isinstance(embeddings[0], list):
            flat = list(itertools.chain.from_iterable(embeddings))
            return [flat[i] for i in inverse]
        if embeddings and isinstance(embeddings[0], torch.Tensor):
            return torch.cat(embeddings)[inverse]
        if embeddings and isinstance(embeddings[0], np.ndarray):
            return np.concatenate(embeddings, axis=0)[inverse]
        if convert_to_tensor:
            return torch.tensor([])
        if convert_to_numpy:
            return np.array([])
        return []
    finally:
        if created_pool:
            self.stop_multi_process_pool(pool)


def setup_oom_safe_multigpu() -> None:
    """Monkeypatch SentenceTransformer's pool worker and collector with the
    OOM-safe token-budget variants above."""

    SentenceTransformer._encode_multi_process_worker = staticmethod(_oom_safe_worker)
    SentenceTransformer._encode_multi_process = _budget_multi_process


def setup_st_forward_compat() -> None:
    """Load checkpoints saved by newer sentence-transformers versions: their modules.json
    references reorganized module paths (e.g. sentence_transformers.base.modules.transformer.Transformer)
    and their module configs can carry keys unknown to the installed version."""

    import inspect

    from sentence_transformers import models
    from sentence_transformers.util import misc

    # The class shadows the module as a package attribute, so resolve the module explicitly.
    st_module = importlib.import_module("sentence_transformers.SentenceTransformer")
    original = misc.import_from_string

    def compat_import_from_string(dotted_path: str) -> Any:
        try:
            return original(dotted_path)
        except ImportError:
            # Reorganized path: fall back to the class name under sentence_transformers.models.
            return getattr(models, dotted_path.rsplit(".", 1)[1])

    misc.import_from_string = compat_import_from_string
    st_module.import_from_string = compat_import_from_string

    renamed_keys = {"embedding_dimension": "word_embedding_dimension"}

    def adapt_config_keys(cls) -> None:
        params = inspect.signature(cls.__init__).parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return
        original_init = cls.__init__

        def adapted_init(self, *args, **kwargs):
            # Map renamed keys back, then drop keys introduced by newer versions (e.g. transformer_task).
            kwargs = {renamed_keys.get(k, k): v for k, v in kwargs.items()}
            original_init(self, *args, **{k: v for k, v in kwargs.items() if k in params})

        cls.__init__ = adapted_init

    for module_cls in (models.Transformer, models.Pooling, models.Dense, models.Normalize):
        adapt_config_keys(module_cls)


def _interrupt_handler(signum: int, frame: Any) -> None:
    os.write(2, b"\nInterrupted: killing GPU workers and exiting.\n")
    for child in multiprocessing.active_children():
        child.kill()
    try:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        os._exit(130)


def setup_signal_handlers() -> None:
    """Make Ctrl+C (and SIGTERM) tear down instantly: SIGKILL the pool workers —
    no join() wait, which otherwise hangs on busy CUDA kernels — and exit."""

    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)


def plan_encoding(wrapper: ModelWrapper, model_path: str, devices: list[str]) -> tuple[Pool | None, list[str]]:
    """Return (pool, encode_devices). Encoding runs through a persistent
    SentenceTransformer pool (spawned once, reused across tasks) so the
    token-budget batch packing applies on any device count."""

    st_model = get_st_model(wrapper)
    if st_model is None:
        raise RuntimeError(f"{model_path} has no underlying SentenceTransformer; "
                           "the multi-GPU pool requires one.")
    _make_picklable_for_spawn(st_model)
    try:
        progress(f"Starting multi-process pool on {len(devices)} GPUs")
        return st_model.start_multi_process_pool(target_devices=devices), []
    except Exception as e:  # noqa: BLE001
        # Safety net: if a model can't be pickled to spawn workers, run unsharded.
        progress(f"WARNING: {model_path} can't use a multi-process pool "
                 f"({type(e).__name__}); running unsharded on {devices[0]}.")
        try:
            st_model.to(devices[0])
        except Exception:  # noqa: BLE001
            pass
        return None, [devices[0]]


def _make_picklable_for_spawn(st_model: SentenceTransformer) -> None:
    """trust_remote_code models load their class into a dynamic
    `transformers_modules.*` module; pickle's identity check then fails when the model
    is sent to spawn workers. Re-register each such class into its module so the
    reference matches (workers re-import it from HF_MODULES_CACHE, on sys.path above)."""

    for module in st_model.modules():
        cls = type(module)
        if cls.__module__.startswith("transformers_modules") and "." not in cls.__qualname__:
            try:
                setattr(importlib.import_module(cls.__module__), cls.__qualname__, cls)
            except Exception:  # noqa: BLE001
                pass


def stop_pool(wrapper: ModelWrapper, pool: Pool | None) -> None:
    """Stop the multi-process pool if one was created; log and swallow any error."""

    if pool is None:
        return
    try:
        get_st_model(wrapper).stop_multi_process_pool(pool)
    except Exception as e:  # noqa: BLE001
        logger.warning("failed to stop pool cleanly: %s", e)


def run_eval(model: ModelWrapper, task_name: str, result_cache: Any,
             encode_pool: Pool | None, encode_devices: list[str]) -> None:
    """Evaluate one MTEB task with `mteb.evaluate`, writing into `result_cache`.

    A spawned pool drives the token-budget multi-GPU path; otherwise a single-device
    fallback is used. overwrite_strategy="only-missing" makes mteb skip the
    splits/subsets already in the cache, so a retried crash/OOM only re-runs
    what's left.
    """

    import mteb

    task = mteb.get_tasks(tasks=[task_name])[0]
    subsets = list(task.hf_subsets)
    splits = list(task.metadata.eval_splits)
    if "test" in splits and len(splits) > 1:
        splits = ["test"]
        task.metadata.eval_splits = splits
    total_jobs = len(subsets) * len(splits)
    encode_kwargs: dict[str, Any] = {"show_progress_bar": False}
    if encode_pool is not None:
        encode_kwargs["pool"] = encode_pool
        encode_mode = f"token-budget pool ({len(encode_pool['processes'])} GPUs)"
    elif encode_devices:
        encode_kwargs["device"] = encode_devices[0]
        encode_mode = f"single {encode_devices[0]}"
    else:
        encode_mode = "unsharded"

    match = find_result_json(result_cache.cache_path, task_name)
    cached_pairs: set[tuple[str, str]] = set()
    if match is not None:
        scores = json.loads(match.read_text()).get("scores", {})
        cached_pairs = {(sp, r.get("hf_subset")) for sp in splits for r in scores.get(sp, [])}
    num_pending = total_jobs - len(cached_pairs)
    if cached_pairs:
        cached_subsets = sorted({subset for _, subset in cached_pairs})
        progress(f"Running {task_name} ({num_pending}/{total_jobs} split×subset jobs, "
                 f"{len(cached_pairs)} cached) encode={encode_mode}")
        progress(f"  resuming — cached: {', '.join(cached_subsets)}")
    else:
        progress(f"Running {task_name} ({total_jobs} split×subset jobs) encode={encode_mode}")
    pending_subsets = [s for s in subsets if any((sp, s) not in cached_pairs for sp in splits)]

    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)

    if len(subsets) <= 1:
        mteb.evaluate(
            model,
            task,
            encode_kwargs=encode_kwargs,
            cache=result_cache,
            overwrite_strategy="only-missing",
            show_progress_bar=False,
        )
    else:
        for subset in pending_subsets:
            progress(f"  ▸ {subset}")
            sub_task = mteb.get_tasks(tasks=[task_name])[0]
            sub_task.hf_subsets = [subset]
            sub_task.metadata.eval_splits = splits
            mteb.evaluate(
                model,
                sub_task,
                encode_kwargs=encode_kwargs,
                cache=result_cache,
                overwrite_strategy="only-missing",
                show_progress_bar=False,
            )


def run_job_with_retry(model: ModelWrapper, task_name: str, result_cache: Any,
                       encode_pool: Pool | None, encode_devices: list[str]) -> bool:
    """Run a task; on CUDA OOM, halve the encode char budget and retry. Other
    errors aren't retryable (a smaller budget won't help), so fail fast.

    Returns True if the task completed, False if it failed (OOM at the minimum
    budget or a non-OOM error).
    """

    st_model = get_st_model(model)
    while True:
        try:
            budget = st_model.encode_char_budget
            run_eval(model, task_name, result_cache, encode_pool, encode_devices)
            progress(f"Completed: {task_name} (encode_char_budget={budget})")
            return True
        except Exception as error:  # noqa: BLE001
            logger.exception("%s failed at encode_char_budget=%s", task_name, budget)  # traceback -> log file
            if "out of memory" not in str(error).lower():
                progress(f"FAILED: {task_name} ({type(error).__name__}; not an OOM — not retrying)")
                return False
            if st_model is None or budget // 2 < 10_000:
                progress(f"FAILED: {task_name} (OOM; gave up below encode_char_budget=10000)")
                return False
            st_model.encode_char_budget = budget // 2
            progress(f"RETRYING: {task_name} (OOM) -> encode_char_budget={budget // 2}")


def free_model(wrapper: ModelWrapper) -> None:
    """Drop the model reference and free CUDA memory (gc + empty_cache)."""

    import torch

    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_evals(models: list[str], tasks: list[str], output_folder: str, devices: list[str],
              encode_char_budget: int, bf16: bool = False, fa2: bool = False,
              local: bool = False) -> None:
    """Evaluate every model over `tasks`, writing results under `output_folder`.

    For each model: skip tasks whose results already exist on disk, load the model
    once, run the remaining tasks (with OOM retry), then free it. Each model's
    results land in its own subfolder of `output_folder`. Prints a per-run summary
    and exits non-zero if any task failed.
    """

    import mteb

    total_jobs = len(tasks) * len(models)
    completed = skipped = failed = 0
    print("\n" + "=" * 60)
    print("MTEB Dense Evaluation - MULTI-GPU")
    print(f"Models: {len(models)} | GPUs: {len(devices)} ({', '.join(devices)}) | Jobs: {total_jobs}")
    print("=" * 60)

    for model_path in models:
        model_results_dir = os.path.join(output_folder, path_to_folder_name(model_path))
        result_cache = mteb.ResultCache(cache_path=model_results_dir)
        pending_tasks = []
        for task_name in tasks:
            subsets, splits = task_meta_subsets(task_name)
            if task_remaining(model_results_dir, task_name, subsets, splits):
                pending_tasks.append(task_name)
        num_skipped = len(tasks) - len(pending_tasks)
        skipped += num_skipped
        if not pending_tasks:
            print(f"[{model_path}] all tasks already completed — skipping load.")
            continue

        print(f"\n##### {model_path} ({len(pending_tasks)} pending, {num_skipped} skipped) #####")
        try:
            model = load_model(model_path, bf16=bf16, fa2=fa2, local=local)
        except Exception as error:  # noqa: BLE001
            logger.exception("%s failed to load", model_path)
            progress(f"[{model_path}] FAILED TO LOAD: {type(error).__name__}: {error}")
            failed += len(pending_tasks)
            continue

        st_model = get_st_model(model)
        if st_model is not None:
            st_model.encode_char_budget = encode_char_budget
        encode_pool, encode_devices = plan_encoding(model, model_path, devices)
        try:
            for task_name in pending_tasks:
                print(f"\n--- [{model_path}] {task_name} ---")
                start_time = time.perf_counter()
                if run_job_with_retry(model, task_name, result_cache, encode_pool, encode_devices):
                    completed += 1
                    print(f"--- [{model_path}] {task_name} done in {time.perf_counter() - start_time:.1f}s ---")
                else:
                    failed += 1
        finally:
            stop_pool(model, encode_pool)
            free_model(model)

    print("\n" + "=" * 60)
    print(f"Done. completed={completed} skipped={skipped} failed={failed} (of {total_jobs})")
    print(f"  results: {output_folder}")
    print("=" * 60)
    if failed:
        sys.exit(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""

    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus",
                   help="Comma-separated GPU ids, e.g. 0,1,2,3,4,5,6,7")
    p.add_argument("--results_folder",
                   help="Folder for results; one subfolder per model is created under it, "
                        "e.g. results/dense")
    p.add_argument("--models", nargs="+", default=[],
                   help="Hub ids or local paths, e.g. lightonai/mDenseOn")
    p.add_argument("--tasks", nargs="+", default=[],
                   help="MTEB task names, e.g. MIRACLRetrievalHardNegatives SyntecRetrieval")
    p.add_argument("--encode_char_budget", type=int, default=3_000_000,
                   help="Chars per encode batch (length-sorted packing); lower if encoding OOMs")
    p.add_argument("--log_file", default=str(repo / "logs" / "eval_dense_detail.log"),
                   help="Detail log for library chatter")
    p.add_argument("--bf16", action="store_true",
                   help="Load models in BF16")
    p.add_argument("--fa2", action="store_true",
                   help="Load models with FlashAttention-2 (implies --bf16; requires flash-attn)")
    p.add_argument("--local", action="store_true",
                   help="Force loading all models via SentenceTransformer (for local checkpoints)")

    args = p.parse_args(argv)

    missing = [n for n in ("gpus", "results_folder", "models", "tasks") if not getattr(args, n)]
    if missing:
        p.error("the following arguments are required: " + ", ".join("--" + n for n in missing))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_file)

    # Needed for the token-budget batching and OOM retry logic.
    setup_oom_safe_multigpu()
    setup_st_forward_compat()
    setup_signal_handlers()

    devices = [f"cuda:{i.strip()}" for i in args.gpus.split(",") if i.strip()]
    output_folder = str(Path(args.results_folder))
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    run_evals(args.models, args.tasks, output_folder, devices,
              encode_char_budget=args.encode_char_budget,
              bf16=args.bf16, fa2=args.fa2, local=args.local)


if __name__ == "__main__":
    main()
