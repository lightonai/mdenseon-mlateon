<p align="center">
  <img src="assets/mDenseOn-mLateOn.png" alt="mDenseOn and mLateOn" width="560">
</p>

# mDenseOn and mLateOn training code

Training scripts for the open mDenseOn dense retriever and mLateOn late-interaction retriever. The models cover multilingual, long-context, and code search.
We also include the training scripts for their English-only counterparts, DenseOn and LateOn.

For the data recipe, experiments, and results, read the [mDenseOn and mLateOn release post](https://huggingface.co/blog/lightonai/mdenseon-mlateon), as well as the [original DenseOn and LateOn post](https://huggingface.co/blog/lightonai/denseon-lateon).

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
@misc{sourty2026mdenseonmlateon,
  title        = {{mDenseOn with the mLateOn}: Open Multilingual, Long-Context, and Code Retrieval Models},
  author       = {Sourty, Raphael and Chaffin, Antoine and Moura Junior, Paulo Roberto and Chatelain, Amelie},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/blog/lightonai/mDenseOn-mLateOn}}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
