# RGZ LeJEPA scaling study

For the current repository layout and runnable commands, start with the
[repository README](../README.md). The detailed notes below retain historical
cluster paths and Slurm examples; the referenced `sh_files/` launchers are not
included in the current checkout.

This repository measures radio-representation quality as a function of
ViT parameter count `P` and unlabelled RGZ20k pretraining sample size `N`:

```text
(P, N) -> MiraBest frozen-probe and full-fine-tuning performance
```

The pretraining source is the same `RGZ20k` batch archive loaded by Erica's
[reference code](https://anonymous.4open.science/r/radio_lejepa-E92D/). It
contains 20,000 unlabelled, sigma-clipped radio cutouts derived from NVSS/FIRST
observations. It is not an arbitrary 20,000-row sample of the later
[RGZ DR1 release](https://zenodo.org/records/14195049), whose catalogue contains
100,185 classifications. DR1 provides the catalogue provenance and defines
component/peak morphology labels such as `1c1p`, `1c2p`, and `2c2p`.

## Controlled variables

Only the model name and nested pretraining subset size vary across the grid.
The following are validated as fixed:

- Two global and eight local views.
- Global random-resized-crop scale `(0.5, 1.0)`.
- Local scale `(0.4, 0.7)` at 98 pixels, resized to 224 pixels.
- Color jitter probability/strengths, blur, horizontal flip, and normalization.
- LeJEPA all-view centre, SIGReg settings, optimiser, schedule, batch, epochs,
  resolution, and seed set.
- The binary MiraBest FR-I/FR-II probe with fixed validation and test records.

The augmentation values are defined once under `views` in
`baseline_config.yaml`. Their defaults reproduce
`code_useful/lejepa_main.py::HFDataset`. The exact configured training-view
order is:

```text
Global (2): RandomResizedCrop(224, scale=0.5-1.0)
            -> RandomApply(ColorJitter(0.4, 0.4, 0.2, 0.1), p=0.8)
            -> RandomApply(GaussianBlur(kernel=7, sigma=0.1-2.0), p=0.5)
            -> RandomHorizontalFlip(p=0.5)
            -> float conversion -> ImageNet normalization

Local (8):  RandomResizedCrop(98, scale=0.4-0.7)
            -> the same jitter, blur, and flip
            -> Resize(224 x 224)
            -> float conversion -> ImageNet normalization
```

The commented-out grayscale and solarization operations in the original code
remain disabled. Torchvision's original bilinear interpolation and antialiasing
defaults are preserved explicitly.

RGZ20k cutouts are read directly from the original `data_batch_1` through
`data_batch_10` files, reshaped to 150 x 150, and replicated into the three
channels expected by Erica's original ViTs and normalization. No intermediate
PNG, FITS, NumPy, or manifest files are used for RGZ20k pretraining.

The implementation removes `feuerzeug`, Hydra, and CUDA assumptions from
the original experiment. W&B tracks pretraining and classification using the
entity `noe-schaerer-unige` and project `radiolejepa`. The Mimosa data locations
are explicit in the two YAML files.

## File guide

| File | Purpose |
| --- | --- |
| `README.md` | Describes the experiment, data preparation, commands, outputs, and reproducibility safeguards. |
| `requirements.txt` | Lists the Python packages required for data preparation, training, evaluation, and plotting. |
| `baseline_config.yaml` | Defines the fixed RGZ20k dataset, LeJEPA augmentation and optimization settings, ViT model grid, nested sample sizes, and output paths. |
| `probe_config.yaml` | Defines the MiraBest root, backbone/projector representation, frozen-probe and fine-tuning settings, and output paths. |
| `rgz20k_data.py` | Loads RGZ20k batches and produces the augmented views used for pre-training. |
| `mirabest_probe_data.py` | Loads confident-only or all non-hybrid MiraBest sources, maps them to FR-I/FR-II, and produces probe examples. |
| `lejepa_baseline.py` | Implements ViT-based LeJEPA pretraining, SIGReg, scheduling, metrics, and checkpoints for one `(P, N, seed)` run. |
| `lejepa_probe.py` | Loads a pretrained checkpoint and runs frozen probing or full MiraBest fine-tuning. |
| `run_pipeline.py` | Runs/reuses one pretraining combination, then evaluates both MiraBest methods. |
| `run_mimosa.sh` | Runs the complete workflow directly across Mimosa GPUs without Slurm. |
| `test_small.sh` | Creates dataset visualizations and runs a two-epoch ViT-Tiny W&B smoke test on 1,000 training images. |
| `sh_files/run_pipeline_small.sh` | Runs Tiny/Small at 2.5k images on `shared-gpu`. |
| `sh_files/run_pipeline_large.sh` | Runs the remaining twenty-two pretraining jobs on `public-gpu`. |
| `run_probe_scaling.py` | Uses each model's RGZ20k-pretrained checkpoint at one labelled-data fraction. |
| `sh_files/run_probe_scaling.sh` | Runs all reduced-labelled-data probes as a GPU array. |
| `results.py` | Evaluates best-validation encoders on MiraBest, reuses matching results, and plots downstream performance plus saved pretraining training/validation losses by epoch and optimizer updates. |
| `sh_files/results.sh` | Runs the evaluation and results step as a Yggdrasil GPU job. |
| `visualization.py` | Produces an RGZ20k augmentation grid and a labelled MiraBest probe-image grid. |
| `sh_files/visualization.sh` | Runs both dataset visualizations as a Baobab CPU job. |

The normal execution path is:

```text
RGZ20k: 20,000 images
    │
    ├── Fixed validation set: 4,000 images, shared by all runs
    └── Training pool: 16,000 images
            │
            └── Pretrain each ViT type at every N in run_mimosa.sh
                N = 2,500 / 5,000 / 7,500 / 10,000 / 13,000 / 16,000
                    │
                    └── checkpoint_best_validation.pt for each (model, N)
                            │
                            ├── Compare RGZ training sizes
                            │   Use each N's checkpoint + 100% of MB training data
                            │
                            └── Compare MB training fractions
                                Use each model's checkpoint at the largest N
                                + 10% / 25% / 50% / 75% / 100% of MB training data

Both comparisons: frozen probe AND full fine-tuning → MB test scores → results.py
```

The pretraining checkpoint is selected by the lowest RGZ validation loss.
For classification, early stopping selects the best MB validation balanced
accuracy. MB validation and test images stay fixed across both comparisons;
fractions change only the labelled training subset.

The pre-training and probe data paths are independent. The two YAML files keep
experimental settings out of the Python implementation.

## Data layout

Data are intentionally not bundled. On Mimosa, `baseline_config.yaml`
points directly to:

```text
/home/schaern8/code_NS/datasets/rgz20k-batches-py/
├── data_batch_1
├── ...
└── data_batch_10
```

The loader requires all ten batches and exactly 2,000 records per batch. The
AstroLeJEPA paper reports the cutouts as 132 x 132 pixels, while the public
RGZ20k loader reshapes the archived arrays to 150 x 150 pixels. This pipeline
follows the archive and reference loader, then applies the fixed 224 x 224 view
augmentation.

The MiraBest probe reads the Mimosa batch directory directly:

```text
/home/schaern8/code_NS/datasets/mirabest-batches-py/
├── data_batch_1
├── ...
├── data_batch_7
├── test_batch
└── batches.meta
```

The loader reads the seven published training batches and the published test
batch directly. `dataset.confident_only` in `probe_config.yaml` selects two modes:

| Setting | Sources | FR-I (`0`) | FR-II (`1`) | Excluded fine labels |
| --- | --- | --- | --- | --- |
| `true` (default) | Certain sources, `MBFRConfident` | `0`, `1`, `2` (100, 102, 104) | `5`, `6` (200, 201) | `3`, `4`, `7`, `8`, `9` |
| `false` | Previous selection, `MBFRAll` | `0` through `4` | `5` through `7` | `8`, `9` (hybrids) |

The selected filter applies to training, validation, and test. The stored
150 x 150 uint8 images are already sigma-clipped and linearly normalized. They
are divided by 255, repeated into three identical channels, resized to the
configured ViT input size, and normalized using the values saved in the
pretraining checkpoint. A deterministic 20% validation subset is taken from
the selected published training records; selected records in `test_batch`
remain held out.

You can override the config with `--confident-only` (certain sources) or
`--no-confident-only` (previous selection) in `lejepa_probe.py`, `run_pipeline.py`,
`run_probe_scaling.py`, `results.py`, and `visualization.py`. For example:

```bash
python results.py --confident-only
python results.py --no-confident-only
```

Each mode saves its own downstream outputs under `MBFRConfident/` or `MBFRAll/`.
With `skip_completed_runs: true`, compatible existing evaluations are reused,
including the older output layouts. Older metrics without a subset name are
identified from `probe_data.excluded_fine_labels`: `[8, 9]` means the previous
selection, and `[3, 4, 7, 8, 9]` means confident-only. The representation,
pretraining epoch, and training fraction must also match. Switching modes
preserves the other mode's results and reuses the same RGZ pretraining.

## Setup

On Yggdrasil, the Slurm launchers load `GCCcore/11.3.0` followed by
`Python/3.10.4`, then activate the `radiolejepa` virtual environment from
`/home/users/s/schaern8/scratch/envs/radiolejepa`.

```bash
python -m pip install -r requirements.txt
```

RGZ20k needs no preparation command. A fixed 20% holdout is selected from the
full 20,000-image archive before any training subsets are drawn
(`dataset.validation_fraction: 0.20`, `dataset.split_seed: 14`). Every model,
training seed, and `N` uses exactly the same 4,000 validation images. Training
subsets are prefixes of a seeded permutation of the remaining 16,000 records,
so smaller training subsets are nested and never overlap validation.
The training series is **2,500, 5,000, 7,500, 10,000, 13,000, and 16,000**
images per model. `sample_size` and `train_size` both mean the actual training
count, excluding validation. Requests above the 16,000 available training
images raise an error. No model trains on the holdout.

Run directories use `n_<N>`, where N is the actual number of training images:

```text
<model>/n_2500/seed_14/pretraining/
<model>/n_5000/seed_14/pretraining/
<model>/n_7500/seed_14/pretraining/
<model>/n_10000/seed_14/pretraining/
<model>/n_13000/seed_14/pretraining/
<model>/n_16000/seed_14/pretraining/
```

W&B run names and Mimosa logs also state `rgz_train_<N>` and the validation
count. Existing folders are left untouched; checkpoint metadata is checked
before reusing a completed training run. `output.directory: null` derives the direct
pretraining output path from the actual model, training count, and seed;
`--output-dir` can still set an explicit destination.

Defaults use 200 epochs and an AdamW base learning rate of `3e-5`, with the
existing warmup/cosine schedule retained. Validation logs the same LeJEPA loss
components without updating weights; augmentation and SIGReg seeds repeat
each epoch for comparable measurements.
Validation monitors convergence and does not stop pretraining early.

Checkpoints trained without this holdout cannot resume with a 20% holdout:
those images have already contributed to training. For a fresh split-based
study, use new `output.directory` and `output.scaling_directory` locations and
clear `output.legacy_directories` so completed old runs are not reused by the
pipeline. Old checkpoints remain available for downstream evaluation.

Both YAML files enable W&B by default. The SDK uses your existing login; install
`requirements.txt` in the environment that runs the job. One run is created per
pretraining combination or downstream method/fraction. Runs log actual model,
dataset, training settings, epoch losses, validation metrics and learning rates;
classification also logs test scores after best-weight restoration. A local
`wandb_run.json` retains the pretraining run ID for online continuation. Runs
finish on success or failure, and completed jobs are skipped without new runs.
Set `wandb.mode: offline` for disconnected jobs, or `wandb.enabled: false` to
disable logging. W&B files stay within each configured output directory.

To create dataset visualizations and check W&B without running the whole pipeline:

```bash
./test_small.sh
```

After generating visualizations, this runs two full pretraining epochs on 1,000 RGZ training images with the
same fixed 4,000-image validation set, logging to `noe-schaerer-unige/radiolejepa`.
It uses a fresh `../outputs/wandb_smoke/` directory per invocation and skips
MiraBest training. Logs are saved to `logs/mimosa/wandb_test.log`. The script
uses the existing `PYTHON_BIN`/`RADIOLEJEPA_ENV` environment selection; set
`PRETRAINING_ROOT` to override the configured RGZ archive location.

Classification defaults are fixed rates within the paper's `1e-5`–`1e-3` range:
`1e-3` for the classifier and `1e-5` for the fine-tuned backbone, plus the
projector when `model.representation: projector` is selected.
Early stopping already uses validation balanced accuracy with patience 10.
There is no automatic learning-rate search. To record rates selected using
validation performance, add model entries to `training.model_learning_rates`
or `finetuning.model_learning_rates` in `probe_config.yaml`, for example:

```yaml
model_learning_rates:
  vit_tiny_patch16_224:
    learning_rate: 1.0e-4
    backbone_learning_rate: 1.0e-5  # Used only for finetuning.
```

Finetuning model entries override training model entries. Use separate output
directories for rate comparisons so previous completed evaluations are not
reused. Test data must not drive rate selection.

## Inspect both datasets

```bash
python visualization.py --dataset both
```

On Yggdrasil, submit the corresponding CPU job with:

```bash
mkdir -p logs/visualization
sbatch sh_files/visualization.sh
```

This writes `outputs/rgz20k_augmentation_grid.png` with the original RGZ20k
cutout plus its two global and eight local views, and
`outputs/mirabest_probe_grid.png` with FR-I and FR-II MiraBest examples. Either
plot can be produced alone with `--dataset pretraining` or `--dataset probe`.

## Single pretraining and probe run

```bash
python lejepa_baseline.py \
  --model vit_tiny_patch16_224 \
  --sample-size 2500

python lejepa_probe.py \
  --checkpoint ../outputs/rgz20k_full_scaling/vit_tiny_patch16_224/n_2500/seed_14/pretraining/checkpoint_best_validation.pt \
  --output-dir ../outputs/rgz20k_full_scaling/vit_tiny_patch16_224/n_2500/seed_14/probe
```

The default probe uses the pooled backbone features before the SSL projector.
It freezes the backbone and trains only LayerNorm plus a linear classifier.
The classifier input uses each model's native feature dimension (192 for Tiny,
384 for Small, and 768 for Base). Its labelled split never depends on `N`.

Select the representation for both probing and fine-tuning in `probe_config.yaml`:

```yaml
model:
  representation: backbone  # Default; use projector for the SSL projector output.
```

`projector` uses the pretrained projection dimension (128 in the baseline
config) for LayerNorm and the classifier input. The classifier structure is
LayerNorm followed by a linear layer for either representation. Keep the same
representation across ViT sizes within a comparison.

`lejepa_probe.py`, `run_pipeline.py`, `run_probe_scaling.py`, and `results.py`
each accept `--representation backbone` or `--representation projector` to
override the probe config for that invocation. For example, to collect the
existing projector evaluations separately:

```bash
python results.py --collect-only --representation projector
```

## Two MiraBest evaluation methods and older checkpoints

`probe_config.yaml` enables `training.methods: [linear, finetune]`. The pipeline
loads the original pretrained checkpoint independently for each method:

- `linear` freezes the feature extractor and trains LayerNorm plus a linear classifier.
- `finetune` trains the backbone and the same classifier. With `projector`
  selected, it also trains the SSL projector; the default backbone evaluation
  excludes the projector from the forward pass and optimizer.
  Defaults are batch size 16, backbone learning rate `1e-5`,
  classifier learning rate `1e-3`, at most 50 epochs, and patience 10.

Both use identical MiraBest splits and class-stratified training subsets. Both
stop after 10 epochs without improvement in validation balanced accuracy and
restore their best trained weights before testing. Fine-tuning saves the
backbone and classifier, plus projector weights when `projector` is selected.
For the 100% MiraBest training
fraction, both methods use `checkpoint_best_validation.pt` at each RGZ training
size (2,500 through 16,000). This selects the lowest pretraining validation loss,
which may occur before the last epoch. The pipeline completes its configured pretraining epoch target before
classification starts. Reduced MiraBest fractions
use `checkpoint_best_validation.pt` from `n_16000`.
Validation still selects the best downstream epoch, not the pretraining subset.
The 100% fraction is reused from the matching full-data evaluation for each
method, or trained if missing. Other fractions can run directly once pretraining
finishes, without first completing the 100% MiraBest run.

Evaluations use separate `backbone/` and `projector/` subdirectories under each
evaluation directory, after `epochs_<best_ssl_epoch>/` for new-format checkpoints.
Legacy flat evaluation files remain untouched and are not reused or plotted.
Existing explicit `projector/` results are considered only when `projector` is
selected and the saved representation metadata matches; backbone evaluations
use only matching `backbone/` results.
Frozen outputs use `linear_probe.pt` and `linear_probe_metrics.json`;
fine-tuning uses `finetune.pt` and `finetune_metrics.json`. Reduced-data
fine-tuning outputs use `finetune_fractions/`; frozen fractions keep
`probe_fractions/`. Existing outputs are not overwritten when reuse is enabled.

The pipeline searches `output.scaling_directory` and the listed
`output.legacy_directories` for matching model, sample size, and seed. Completed
pretraining (final metrics plus a checkpoint) and frozen results are reused in
place. It accepts both `checkpoint` and the older `best_checkpoint` metrics
fields and recovers relocated checkpoint filenames from the run's pretraining
directory. Earlier project checkpoint metadata, `model_state_dict`, and
distributed `module.` prefixes are supported. All backbone weights must match;
missing metadata or incompatible architectures raise an error. Reused runs
retain their original pretraining provenance, including any earlier holdout
recipe. Only configured model/sample-size combinations appear in the grid.

Run the full workflow on Mimosa after transferring the updated files:

```bash
cd /home/schaern8/code_NS/code_NS_radiolejepa
bash run_mimosa.sh all
```

To evaluate an existing checkpoint directly, without pretraining:

```bash
source ~/envs/radiolejepa/bin/activate
python lejepa_probe.py --method finetune \
  --checkpoint /path/to/checkpoint_best_validation.pt \
  --output-dir ../outputs/manual_finetune
```

For pipeline evaluation of a specific checkpoint, `run_pipeline.py` also
accepts `--checkpoint /path/to/checkpoint.pt` and `--methods linear finetune`.
Use the checkpoint's actual model name, sample size, and seed in those arguments.

`results.py` runs or reuses 100% MiraBest evaluations using the best-validation
checkpoint at every configured RGZ training size, for both enabled methods.
For the reduced 10%, 25%, 50%, and 75% labelled-training fractions, it continues
to select each model's `checkpoint_best_validation.pt` trained on 16,000 RGZ images. It preserves the fixed MiraBest validation and test sets, logs training
to W&B, and writes per-run checkpoints/metrics to the normal evaluation
directories. Missing or unfinished RGZ pretraining is reported and skipped;
Reduced-fraction runs require the best-validation 16k encoder; full-data runs require
the best-validation checkpoint at their corresponding N.

It saves downstream PNG plots to `../outputs/results/<subset>/backbone/` or
`../outputs/results/<subset>/projector/`, below the configured `output.results_directory`,
including `method_comparison.png` and `method_fraction_comparison.png`.
The four shared pretraining-loss figures stay directly in
`output.results_directory`.
Two backbone-versus-projector macro-F1 figures are saved in the subset directory:
`linear_representation_comparison.png` and `finetune_representation_comparison.png`.
It reports missing runs by stage in the
terminal. To plot existing results without training, use
`python results.py --collect-only`. Use a GPU for the default evaluation mode.
The original per-run JSON metrics remain necessary inputs for reuse and plots.


## Full P-by-N grid

This is a new pretraining data/model scaling study. Figure 5 in the AstroLeJEPA
paper instead varies the fraction of labelled downstream training data after
pretraining; it is not a pretraining `P`-by-`N` experiment.

Run the full Tiny/Small/Base/Large grid over
`N = 2.5k, 5k, 7.5k, 10k, 13k, 16k` on Yggdrasil. Create the log directory before
submission because Slurm opens its log
files before the job script starts:

```bash
cd /home/users/s/schaern8/scratch/code_NS/code_NS_radiolejepa
mkdir -p logs/run_pipeline_small logs/run_pipeline_large logs/run_probe_scaling logs/results
sbatch sh_files/run_pipeline_small.sh
sbatch sh_files/run_pipeline_large.sh
```

After both pretraining arrays and their full probes finish, run the labelled
MiraBest fraction study:

```bash
sbatch sh_files/run_probe_scaling.sh
```

After that array finishes, submit the results job:

```bash
sbatch sh_files/results.sh
```

`run_pipeline_small.sh` submits Tiny and Small at `N = 2.5k` to the 12-hour
`shared-gpu` partition. `run_pipeline_large.sh` submits the remaining twenty-two
combinations to the two-day `public-gpu` partition. All tasks are submitted
without a concurrency throttle. The launchers request compute-capability-7.5
GPUs because the installed PyTorch build does not support Yggdrasil's
compute-capability-7.0 V100. Existing completed pretraining and probe outputs
are reused instead of recomputed.

### Mimosa (no Slurm)

Edit only `MODELS` and `SUBSET_SIZES` near the top of `run_mimosa.sh` to
change the Mimosa experiment grid. The launcher generates all combinations
and saves a run-specific config in `logs/mimosa/`, then passes that config to
the grid, fraction study, and results stages. The fraction study automatically
uses the largest size in `SUBSET_SIZES` for each model type. Values in
`baseline_config.yaml` remain defaults for standalone Python/Slurm commands.

The active Mimosa grid includes Tiny, Small, and Base only: 18 pretraining
combinations and 12 reduced-data combinations, each evaluated with both
MiraBest methods. Large is excluded from scheduling and result completeness
checks. The Slurm launchers under `sh_files/` use the same six training sizes and also include Large (24 combinations in total).

The Mimosa launcher uses `~/envs/radiolejepa/bin/python`. It derives the dataset
directory from the code location, so code in `~/code_NS/code_NS_radiolejepa`
uses `~/code_NS/datasets` automatically. It starts one worker per detected GPU,
keeps at most one experiment on each GPU, and distributes the remaining jobs
between them. Before launching, it verifies PyTorch CUDA access and skips GPUs
that already have active compute processes. It stops without starting jobs if
no GPU is free.

Run the stages separately:

```bash
./test_small.sh
./run_mimosa.sh grid
./run_mimosa.sh probe-scaling
./run_mimosa.sh results
```

`test_small.sh` creates the
RGZ20k augmentation and MiraBest image grids. It then starts the two-epoch W&B smoke pretraining run, without
a MiraBest probe.

Or run the complete workflow in order:

```bash
./run_mimosa.sh all
```

Paths and GPU selection can be overridden without editing files:

```bash
PRETRAINING_ROOT=/path/to/rgz20k-batches-py \
PROBE_ROOT=/path/to/mirabest-batches-py \
GPU_IDS="0 1 2 3 4 5 6 7" \
./run_mimosa.sh grid
```

Per-experiment logs are written to `logs/mimosa/`. Completed outputs are
reused or skipped by the Python pipeline. The grid evaluates both methods on
100% of the MiraBest training split for every checkpoint. Those full-data
results supply the `1.00` fraction; the fraction stage adds the 10%, 25%, 50%,
and 75% evaluations for both methods without duplicating the 100% run.

For each ViT type, the fraction study uses its best-validation checkpoint at the
largest configured RGZ training size (currently 16,000 images)
(`n_16000/seed_14/pretraining/checkpoint_best_validation.pt`). The encoder remains frozen while LayerNorm and a linear
classifier are trained with 10%, 25%, 50%, 75%, and 100% of the fixed MiraBest
training split. Smaller-N fraction outputs remain on disk but are not used in
the fraction plots. Fine-tuning uses the same best-validation 16,000-image pretraining
source. Fraction results from older checkpoints without the fixed holdout are
recomputed when requested and excluded from fraction plots. Plot labels use
actual training counts, so the full-pool checkpoint is labelled 16,000.

The results job runs:

```bash
python results.py --require-complete
```

This creates `full_probe_scaling.png`, `probe_fraction_f1.png`,
`method_comparison.png`, and `method_fraction_comparison.png` under
`../outputs/results/<subset>/<representation>/`, when the relevant results are available.
Here `<subset>` is `MBFRConfident` or `MBFRAll`.
The representation defaults to `backbone`. The following shared figures are
written directly to `../outputs/results/`, when the corresponding saved metrics
are available:

| Figure | Loss history | Horizontal axis |
| --- | --- | --- |
| `pretraining_loss.png` | Training | Pretraining epoch |
| `pretraining_validation_loss.png` | Validation | Pretraining epoch |
| `pretraining_loss_updates.png` | Training | Cumulative optimizer updates |
| `pretraining_validation_loss_updates.png` | Validation | Cumulative training optimizer updates |

Two additional shared figures compare backbone and projector test macro-F1:
`linear_representation_comparison.png` for frozen linear probing and
`finetune_representation_comparison.png` for fine-tuning, saved under
`../outputs/results/<subset>/`. Each shows every
configured model across RGZ training sizes, using 100% of the MiraBest training
split and each run's best SSL validation checkpoint. Model colors stay fixed;
backbone features use solid lines with circles, and projector features use
dashed lines with squares. The two figures share the same vertical limits.
They read both representations' saved evaluations regardless of
`--representation`, including when using `--collect-only`. Missing or invalid
evaluations are reported and left as gaps; the comparison step does not train
additional representations.

To generate just the four pretraining-loss figures from existing metrics without
pretraining or downstream evaluation, run:

```bash
python results.py --pretraining-only
```

This reads the configured model/N/seed histories from
`pretraining/metrics.json` in the current or legacy output directories. It
does not require MiraBest data or completed probe runs, and it does not select
downstream checkpoints. To collect both pretraining and existing downstream
plots without training, use:

```bash
python results.py --collect-only --representation backbone
```

Each pretraining figure has one row per architecture and columns for total
LeJEPA loss, unweighted invariance loss, and unweighted SIGReg loss. Colors
identify RGZ subset sizes consistently across the four figures and in the
model-parameter scaling plot. Training values are epoch averages; validation
is recorded after the epoch. On the update plots, each epoch's recorded value
is placed at the cumulative number of training optimizer updates at its end.
Validation itself performs no optimizer updates.

Saved update/step metadata in the metrics takes priority; otherwise the plots
use saved checkpoint step metadata when available. For older runs without this
metadata, the script explicitly reports an estimate from the saved training
size and saved training configuration, or the current configuration when the
original settings are unavailable: `train_size // batch_size`, capped by a positive
`training.max_steps_per_epoch` when configured, gives updates per epoch.
These runs use `drop_last=True` and one optimizer update per training batch.
Estimated curves are marked with `*`. An estimate depends on the configuration
matching the original run, including any step overrides.

Only epochs up to the configured target are shown. Missing validation values,
components, or usable update metadata are reported and skipped; the script
does not rerun pretraining or compute missing validation losses.
The total loss is `(1 - lambda_sigreg) * invariance + lambda_sigreg * sigreg`,
so the two raw components should not be directly added.

When the repository is at `~/code_NS/code_NS_radiolejepa`, results are written
outside the repository below `~/code_NS/outputs`:

```text
../outputs/rgz20k_full_scaling/<model>/n_<N>/seed_<seed>/
├── pretraining/
│   ├── checkpoint_last.pt
│   ├── checkpoint_best_validation.pt
│   ├── metrics.json
│   └── subset.json
└── probe/epochs_<best_ssl_epoch>/<representation>/<subset>/
    ├── linear_probe.pt
    └── linear_probe_metrics.json

../outputs/rgz20k_full_scaling/probe_fractions/<model>/n_16000/seed_<seed>/mb_train_fraction_<f>/
└── epochs_<best_ssl_epoch>/<representation>/<subset>/
    ├── linear_probe.pt
    └── linear_probe_metrics.json

../outputs/results/
├── pretraining_loss.png
├── pretraining_validation_loss.png
├── pretraining_loss_updates.png
├── pretraining_validation_loss_updates.png
├── MBFRConfident/
│   ├── linear_representation_comparison.png
│   ├── finetune_representation_comparison.png
│   ├── backbone/
│   └── projector/
└── MBFRAll/  # The same layout for the previous dataset selection.
```

The comparison plots show full-data performance across pretraining sizes and
macro-F1 as the labelled MiraBest training fraction changes. The same model
uses the same fixed color in the model comparison plots: Tiny is blue,
Small is orange,
Base is green, and Large is red. Solid lines denote linear probing and dashed
lines denote fine-tuning in the method comparisons. The model-parameter panel
and pretraining-loss plots use colors for RGZ subset size instead. Existing
summary files from older runs are not deleted.

Performance plots use vertical limits around the observed scores with padding,
bounded by 0 and 1 and spanning at least 0.10. The two panels in the full-probe
scaling figure share the same vertical limits. Pretraining-loss axes retain
automatic limits with a small margin.

## Comparability safeguards

- The cosine schedule decays from `3e-5` to `1e-6`; its minimum cannot
  exceed its peak.
- SIGReg rejects object batches smaller than four; the default is 16.
- Mean projector standard deviation is logged for monitoring.
- Downstream classification defaults to native backbone features; selecting
  `model.representation: projector` evaluates the pretrained SSL projector.
  All backbone weights must match, and projector evaluation additionally
  requires compatible pretrained projector weights.
- Checkpoints store the model, augmentation, subset sizes, and metric history.

For a publication-quality fit, use at least three seeds and report uncertainty
across seeds before fitting a scaling surface to `P`, `N`, and probe error.

## Resuming training

New pretraining runs atomically save `checkpoint_last.pt` after every completed
epoch, including model/projector weights, AdamW moments and step counters,
scheduler state, random states, data-loader generator state, subset IDs, and
training settings. Run the same launcher again after an interruption: training
continues at the next epoch (an unfinished epoch is repeated).

`checkpoint_best_validation.pt` saves the model with the lowest total LeJEPA
validation loss, updating only on strict improvement. `checkpoint_last.pt`
saves the latest completed epoch and is the final model when training finishes.
Both contain the full model and
training state; best-validation loss/epoch are also recorded in checkpoint
metadata, `metrics.json`, and W&B. Resuming or extending a run retains its
earlier best checkpoint unless a later epoch improves validation loss. With
validation disabled, only the latest checkpoint is saved.
The pipeline uses `checkpoint_best_validation.pt` for full-data classification
at each N, for both frozen probing and full fine-tuning. Reduced MiraBest
fractions use the best-validation 16k checkpoint.

To extend a new run, change `training.epochs` in `baseline_config.yaml` to the
total desired target, for example 200, then run `bash run_mimosa.sh all`.
AdamW is restored without reinitialization. The existing cosine schedule is
stretched to the new target, recalculating the current learning rate on that
longer curve; this is not identical to having trained for 200 epochs originally.
Keep other training/model/data settings unchanged. GPU kernels may still be
nondeterministic; bitwise reproducibility is not guaranteed.

Frozen and fine-tuning evaluations from new checkpoints are saved separately
under `epochs_<best_ssl_epoch>/<representation>/<subset>/`, including the fraction studies.
Existing results are reused only for the selected pretraining checkpoint epoch
and representation, training fraction, and dataset mode. Previous evaluation
files are retained; aggregate plots show the selected mode, representation,
and checkpoints.

Older weight-only checkpoints remain usable for evaluation but cannot resume
training without resetting AdamW, so continuation is explicitly rejected.
For a fresh new-version run, choose a **new** `output.scaling_directory` (and
`output.directory` for standalone training) and clear `output.legacy_directories`
in the config so historical checkpoints are not selected. No outputs need deletion.
# radiolejepa_NS
