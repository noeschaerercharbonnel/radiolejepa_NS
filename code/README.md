# RadioLeJEPA code

Pretrain ViT models on RGZ20k and evaluate FR-I/FR-II classification on MiraBest.
See the [project README](../README.md) for the experimental design, attribution
and saved figures.

## Setup

Use Python 3.10+. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
cd code
```

Set your dataset paths in:

- [baseline_config.yaml](baseline_config.yaml): `dataset.batches_root` for RGZ20k.
- [probe_config.yaml](probe_config.yaml): `dataset.root` for MiraBest.

These files also define training settings and output paths. Set
`wandb.enabled: false` in both files, or configure your own W&B account.
Run all commands below from `code/`.

## Train and evaluate

```bash
# One pretraining configuration, followed by probing and fine-tuning.
python run_pipeline.py \
  --model vit_tiny_patch16_224 --sample-size 2500 --seed 14 \
  --methods linear finetune --representation backbone --confident-only

# Reduced-label evaluation; requires the model's 16,000-image checkpoint.
python run_probe_scaling.py \
  --model vit_base_patch16_224 --fraction 0.25 \
  --methods linear finetune --representation backbone --confident-only
```

- `--confident-only`: confidently labelled FR-I/FR-II sources (default).
- `--no-confident-only`: confident + uncertain FR-I/FR-II sources; hybrids excluded.
- `--representation projector`: evaluate projector outputs instead of backbone features.

Compatible checkpoints and completed evaluations are reused. Outputs are
separated by dataset selection and representation under `../outputs/`.
See the project README for GPU sweeps with `run_mimosa.sh`.

## Plot saved results

```bash
python results.py --collect-only --metric macro_f1 \
  --representation backbone --confident-only
python results.py --pretraining-only
```

Use `--no-confident-only` to collect the expanded selection. Plot regeneration
requires the saved per-run metrics; the bundled PNGs alone are insufficient.
Without `--collect-only`, `results.py` can launch missing evaluations.
