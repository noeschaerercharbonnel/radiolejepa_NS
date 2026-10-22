#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${RADIOLEJEPA_ENV:-${HOME}/envs/radiolejepa}/bin/python}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/mimosa}"
LOG_FILE="${LOG_DIR}/visualization.log"
WANDB_LOG_FILE="${LOG_DIR}/wandb_test.log"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUN_DIR="${SCRIPT_DIR}/../outputs/wandb_smoke/vit_tiny_n_1000_epochs_2_${RUN_ID}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}"

echo "Creating RGZ20k and MiraBest visualizations" | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -u visualization.py --dataset both 2>&1 | tee -a "${LOG_FILE}"

echo "Running ViT-Tiny W&B test: 1,000 training images, 2 epochs" | tee "${WANDB_LOG_FILE}"
"${PYTHON_BIN}" -u - "${PRETRAINING_ROOT:-}" "${RUN_DIR}" 2>&1 <<'PY' | tee -a "${WANDB_LOG_FILE}"
import sys
from lejepa_baseline import load_config, train

if __name__ == "__main__":
    config = load_config("baseline_config.yaml")
    config["dataset"].update(sample_size=1000, validation_fraction=0.20, split_seed=14)
    if sys.argv[1]:
        config["dataset"]["batches_root"] = sys.argv[1]
    config["model"]["name"] = "vit_tiny_patch16_224"
    config["training"].update(epochs=2, max_steps_per_epoch=None)
    config["wandb"].update(enabled=True, mode="online",
                           entity="noe-schaerer-unige", project="radiolejepa")
    checkpoint = train(config, output_dir_override=sys.argv[2])
    print(f"W&B smoke run complete; checkpoint: {checkpoint}")
PY
