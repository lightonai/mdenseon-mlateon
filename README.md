<p align="center">
  <img src="assets/mDenseOn-mLateOn.png" alt="mDenseOn and mLateOn" width="560">
</p>

# mDenseOn and mLateOn training code

Training scripts for the open mDenseOn dense retriever and mLateOn late-interaction retriever. The models cover multilingual, long-context, and code search.
We also include the training scripts for their English-only counterparts, DenseOn and LateOn.

For the data recipe, experiments, and results, check out our [mDenseOn and mLateOn blog](https://huggingface.co/blog/lightonai/mdenseon-mlateon), the [DenseOn and LateOn blog](https://huggingface.co/blog/lightonai/denseon-lateon) and our [paper](https://arxiv.org/abs/2607.27178).

## Setup

The scripts require Python 3.10 or newer and are intended for CUDA GPUs with bfloat16 support.

```bash
git clone git@github.com:lightonai/mdenseon-mlateon.git
cd mdenseon-mlateon

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

You can also install the environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

For faster training, install [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) with `uv sync --extra flash` and add `"attn_implementation": "flash_attention_2"` to `model_kwargs` when instantiating the models.

## Training scripts

| Stage | Dense | Late interaction |
| --- | --- | --- |
| Multilingual pre-training | `scripts/pretrain/multilingual_dense.py` | `scripts/pretrain/multilingual_late_interaction.py` |
| Multilingual fine-tuning | `scripts/finetune/multilingual_dense.py` | `scripts/finetune/multilingual_late_interaction.py` |
| English pre-training | `scripts/pretrain/english_dense.py` | `scripts/pretrain/english_late_interaction.py` |
| English fine-tuning | `scripts/finetune/english_dense.py` | `scripts/finetune/english_late_interaction.py` |

The multilingual fine-tuning scripts combine a contrastive loss with KL-divergence distillation from stored cross-encoder teacher scores, and cover the multilingual, long-context, and code datasets. The English ones are contrastive-only on the NV-Embed KD dataset.

Every script exposes its options through `--help`. Start a distributed run with `accelerate launch`, for example:

```bash
accelerate launch scripts/pretrain/multilingual_dense.py --help

accelerate launch scripts/pretrain/multilingual_dense.py \
  --model_name jhu-clsp/mmBERT-base \
  --batch_size 16384 \
  --mini_batch_size 16
```

Outputs are written under `output/`. Hugging Face datasets and checkpoints are downloaded on first use, so pre-training requires substantial storage as well as multi-GPU compute. Use `--stop_at_step` for a short trial run before starting a full job.

## Evaluation

The `scripts/eval/` folder contains multi-GPU [MTEB](https://github.com/embeddings-benchmark/mteb) evaluation scripts for both model families. They require `uv sync --extra eval`.

All scripts pack length-sorted texts into variable-size batches under a character budget (`--encode_char_budget`, default 3M characters), so short documents form large batches and long documents small ones without per-task batch-size tuning. Encoding OOMs recover automatically: the dense scripts halve the budget and retry the task, re-computing only the results still missing, while the late-interaction script re-encodes the offending batch in two halves. To further speed up evals, pass `--fa2` to enable encoding with FlashAttention-2 (see [Setup](#setup) for installing the `flash` extra). All scripts expose their options through `--help`.

### Dense

Dense models can be evaluated in two ways, sharing the same results layout (one subfolder per model; already-completed (task, language) pairs are skipped on rerun, so the two scripts can fill the same results folder):

- `scripts/eval/dense_sequential.py` runs tasks **sequentially**, distributing each task's encoding across all GPUs. This is fastest for large, encode-bound tasks (MSMARCO, MLDR, MIRACL, CodeSearchNet).
- `scripts/eval/dense_parallel.py` runs tasks in **parallel**, each GPU owning its task's encoding end to end. This is fastest for running several small, retrieval-bound tasks (TREC-COVID, FiQA, SciFact, Quora).

```bash
python scripts/eval/dense_sequential.py \
  --gpus 0,1,2,3,4,5,6,7 --bf16 \
  --results_folder results/dense \
  --models lightonai/mDenseOn \
  --tasks MIRACLRetrievalHardNegatives MultiLongDocRetrieval

python scripts/eval/dense_parallel.py \
  --gpus 0,1,2,3,4,5,6,7 --bf16 \
  --results_folder results/dense \
  --models lightonai/mDenseOn \
  --tasks TRECCOVID FiQA2018 SciFact QuoraRetrieval
```

### Late interaction

Late-interaction models encode with `accelerate` across all GPUs and retrieve with [FastPLAID](https://github.com/lightonai/fast-plaid) index (`fast-plaid>=1.5`), where tasks run **sequentially** as the search step is GPU-bound.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch scripts/eval/late_interaction.py \
  --models lightonai/mLateOn \
  --tasks MIRACLRetrievalHardNegatives MultiLongDocRetrieval \
  --results_folder results/late_interaction
```

## Data and models

- [mDenseOn](https://huggingface.co/lightonai/mDenseOn)
- [mLateOn](https://huggingface.co/lightonai/mLateOn)
- [Multilingual collection, including data](https://huggingface.co/collections/lightonai/mdenseon-and-mlateon)
- [DenseOn](https://huggingface.co/lightonai/DenseOn)
- [LateOn](https://huggingface.co/lightonai/LateOn)
- [English-only collection, including data](https://huggingface.co/collections/lightonai/denseon-and-lateon)

## Citation
If you use our code, models or datasets in your research, please consider citing our work:

```bibtex
@misc{sourty2026denseonlateonfullyopen,
  title         = {DenseOn with the LateOn: Fully Open Dense and Late-Interaction Models for Multilingual, Long-Context, and Code Search},
  author        = {Raphaël Sourty and Antoine Chaffin and Paulo Roberto Moura Junior and Amélie Chatelain},
  year          = {2026},
  eprint        = {2607.27178},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.27178},
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
