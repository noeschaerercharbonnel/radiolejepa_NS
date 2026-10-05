#!/bin/bash
#SBATCH --job-name=radiolejepa_probe_scaling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=shared-gpu
#SBATCH --constraint=COMPUTE_CAPABILITY_7_5
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/run_probe_scaling/probe_%A_%a.out
#SBATCH --error=logs/run_probe_scaling/probe_%A_%a.err

set -euo pipefail

CODE_DIR="/home/users/s/schaern8/scratch/code_NS/code_NS_radiolejepa"
PROBE_ROOT="/home/users/s/schaern8/scratch/code_NS/datasets/mirabest-batches-py"
ENV_DIR="${RADIOLEJEPA_ENV:-/home/users/s/schaern8/scratch/envs/radiolejepa}"

MODELS=(
    vit_tiny_patch16_224
    vit_small_patch16_224
    vit_base_patch16_224
    vit_large_patch16_224
)
FRACTIONS=(0.10 0.25 0.50 0.75)

N_FRACTIONS=${#FRACTIONS[@]}
MODEL=${MODELS[SLURM_ARRAY_TASK_ID / N_FRACTIONS]}
FRACTION=${FRACTIONS[SLURM_ARRAY_TASK_ID % N_FRACTIONS]}

module load GCCcore/11.3.0
module load Python/3.10.4
source "${ENV_DIR}/bin/activate"
cd "${CODE_DIR}"

python -u run_probe_scaling.py \
    --model "${MODEL}" \
    --fraction "${FRACTION}" \
    --probe-root "${PROBE_ROOT}"
