# RadioLeJEPA: pretraining and downstream classification study

Code and figures for a PhD application coding task studying how pretraining
dataset size, ViT size and supervised adaptation affect radio-galaxy
classification. ViT-Tiny, ViT-Small and ViT-Base are pretrained on RGZ20k and
evaluated on the binary FR-I/FR-II task in MiraBest.

The implementation builds on [Erica Lastufka's RadioLeJEPA reference
code](https://anonymous.4open.science/r/radio_lejepa-E92D/), retaining its
LeJEPA objective, SIGReg regularization and augmentation recipe. Extensions
for this study include configurable dataset selections, deterministic nested
training subsets, automated experimental sweeps, backbone/projector evaluation,
compatible checkpoint reuse, saved metrics and comparison plots.

## Repository layout

```text
radiolejepa_NS/
├── README.md
├── code/
│   ├── baseline_config.yaml       # RGZ20k, model grid and pretraining settings
│   ├── probe_config.yaml          # MiraBest selection and downstream settings
│   ├── requirements.txt
│   ├── lejepa_baseline.py         # LeJEPA pretraining
│   ├── lejepa_probe.py            # Frozen probing and fine-tuning
│   ├── run_pipeline.py            # One pretraining/evaluation configuration
│   ├── run_probe_scaling.py       # Labelled-training-fraction experiments
│   ├── run_mimosa.sh              # GPU experiment launcher
│   ├── checkpoint_compat.py       # Compatibility checks and output reuse
│   ├── rgz20k_data.py
│   ├── mirabest_probe_data.py
│   ├── experiment_tracking.py
│   ├── results.py                 # Metric aggregation and plots
│   └── visualization.py           # Dataset and augmentation examples
└── outputs/
    ├── rgz20k_augmentation_grid.png
    ├── mirabest_probe_grid.png
    └── results/
        ├── pretraining_*.png
        ├── backbone/             # Earlier confident + uncertain results
        ├── projector/            # Earlier confident + uncertain results
        └── MBFRConfident/         # Confident-only results
```

The repository includes source code, configurations and exported figures.
Datasets, trained checkpoints and per-run metric histories are not bundled.
This README describes the current folder layout; the longer
[implementation notes](code/README.md) also contain historical examples.

## Experimental design

- **Pretraining:** a fixed 4,000-image RGZ20k validation set and nested training
  subsets of 2,500, 5,000, 7,500, 10,000, 13,000 and 16,000 images; 200 epochs;
  seed 14. The checkpoint with the lowest pretraining validation loss is used.
- **Downstream methods:** frozen features with a trainable LayerNorm and linear
  classifier, or end-to-end fine-tuning. The projector is also updated when
  its output is selected for fine-tuning.
- **Labelled fractions:** 10%, 25%, 50%, 75% and 100% of the training split,
  using each model's selected checkpoint at 16,000 pretraining images.
  Validation and test sets remain fixed across fractions within each selection.
- **Checkpoint selection:** validation balanced accuracy for downstream runs.
  The representation and method-comparison figures report test macro F1.

| MiraBest selection | CLI flag | FR-I fine labels | FR-II fine labels | Train / validation / test |
| --- | --- | --- | --- | --- |
| Confident only (`MBFRConfident`) | `--confident-only` | 0, 1, 2 | 5, 6 | 583 / 146 / 104 |
| Confident + uncertain (`MBFRAll`) | `--no-confident-only` | 0, 1, 2, 3, 4 | 5, 6, 7 | 855 / 214 / 153 |

Both selections exclude hybrid labels 8 and 9. The second mode restores the
earlier selection; it is not an uncertain-only sample. Split counts use seed
14 and a 20% validation split of the selected published training records.
Filtering is followed by a new split, so training/validation membership and
the test population differ between selections. These are single-seed results.

## Installation and data

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
cd code
```

Run the following Python commands from **`code/`**. Relative configuration
and output paths are resolved from that working directory.

Supply the original batch archives and edit:

- [`baseline_config.yaml`](code/baseline_config.yaml):
  `dataset.batches_root` points to RGZ20k, containing `data_batch_1` through
  `data_batch_10`.
- [`probe_config.yaml`](code/probe_config.yaml): `dataset.root` points to
  MiraBest, containing `data_batch_1` through `data_batch_7`, `test_batch`
  and `batches.meta`.

The checked-in dataset paths refer to the author's Mimosa environment.
For archives placed under the repository's `datasets/` folder, use
`../datasets/rgz20k-batches-py` and `../datasets/mirabest-batches-py` respectively.
Data files and model weights are excluded by `.gitignore`.

Weights & Biases logging is enabled in both configurations. Set
`wandb.enabled: false` to run without it, or configure your own entity and
project and run `wandb login`. Dependency versions in `requirements.txt`
are not an exact environment lock; a fixed seed does not guarantee bitwise
identical results across hardware and software versions.

## Run experiments

Run or reuse one pretraining configuration and evaluate both downstream methods:

```bash
python run_pipeline.py \
  --model vit_tiny_patch16_224 --sample-size 2500 --seed 14 \
  --methods linear finetune --representation backbone --confident-only
```

Use `--no-confident-only` for the confident + uncertain selection and
`--representation projector` for projector features. These flags also work
with `lejepa_probe.py`, `run_probe_scaling.py` and `results.py`.

Once the 16,000-image pretraining checkpoint exists, run a reduced-label experiment:

```bash
python run_probe_scaling.py \
  --model vit_base_patch16_224 --fraction 0.25 \
  --methods linear finetune --representation backbone --confident-only
```

The compatible RGZ checkpoint is shared between dataset selections. Completed
downstream results are reused only when their dataset mode, representation,
pretraining epoch and training fraction match. Older compatible output layouts
are supported. New evaluations are stored under separate `MBFRConfident/`
and `MBFRAll/` directories.

### GPU sweeps

On a CUDA machine, `run_mimosa.sh` schedules the model/size grid and labelled
fractions across available GPUs. Edit `MODELS` and `SUBSET_SIZES` in that
script, and choose the dataset mode and representation in `probe_config.yaml`.
The launcher supports `grid`, `probe-scaling`, `results` and `all`:

```bash
PYTHON_BIN="../.venv/bin/python" \
DATA_ROOT="/path/to/datasets" \
bash run_mimosa.sh all
```

`DATA_ROOT` must contain the two batch-archive directories listed above.
Optional `GPU_IDS="0 1"` selects specific GPUs. The launcher uses
`nvidia-smi`; the individual Python entry points also support the devices
selected by their configurations.

## Collect results and inspect figures

With the per-run metrics available at the configured output location,
regenerate plots without launching training:

```bash
python results.py --collect-only --metric macro_f1 \
  --representation backbone --confident-only

python results.py --collect-only --metric macro_f1 \
  --representation backbone --no-confident-only

python results.py --pretraining-only
```

Without `--collect-only`, the results command can run missing downstream
evaluations. Representation-comparison plots read both saved representations
when available, even when backbone is selected for evaluation. The PNG files
alone cannot regenerate the underlying metric histories or checkpoints.

The bundled expanded-selection figures retain their earlier locations under
`outputs/results/`; newly generated equivalents are placed under
`outputs/results/MBFRAll/`.

| Figure | Confident sources | Confident + uncertain sources |
| --- | --- | --- |
| Backbone: probe and fine-tuning versus labelled fraction | [View](outputs/results/MBFRConfident/backbone/method_fraction_comparison.png) | [View](outputs/results/backbone/method_fraction_comparison.png) |
| Frozen backbone versus projector | [View](outputs/results/MBFRConfident/linear_representation_comparison.png) | [View](outputs/results/linear_representation_comparison.png) |
| Fine-tuned backbone versus projector | [View](outputs/results/MBFRConfident/finetune_representation_comparison.png) | [View](outputs/results/finetune_representation_comparison.png) |

Pretraining validation histories are available
[by epoch](outputs/results/pretraining_validation_loss.png) and
[by optimizer updates](outputs/results/pretraining_validation_loss_updates.png).
At a fixed epoch, larger subsets also involve more optimizer updates; these
plots do not isolate dataset size from compute. The downstream selections have
different test populations, so their score gap does not isolate the effect of
uncertain training labels.
