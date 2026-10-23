# Train and evaluate

Run all commands from `code/`. See the [project README](../README.md) for
installation, data requirements and the experimental design.

## Complete pipeline

On the CUDA training machine:

```bash
bash run_mimosa.sh all
```

This runs the full model/pretraining-size grid, probing and fine-tuning at
each labelled training fraction, and result plotting. Compatible completed
runs are reused.

Choose `MODELS` and `SUBSET_SIZES` in [run_mimosa.sh](run_mimosa.sh).
Choose `dataset.confident_only` and `model.representation` in
[probe_config.yaml](probe_config.yaml); these settings apply to the complete pipeline.

## Individual experiments

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

Outputs are separated by dataset selection and representation under `../outputs/`.

## Plot saved results

```bash
python results.py --collect-only --metric macro_f1 \
  --representation backbone --confident-only
python results.py --pretraining-only
```

Use `--no-confident-only` to collect the expanded selection. Plot regeneration
requires the saved per-run metrics; the bundled PNGs alone are insufficient.
Without `--collect-only`, `results.py` can launch missing evaluations.
