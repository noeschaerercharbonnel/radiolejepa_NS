#!/bin/bash
#SBATCH --job-name=radiolejepa_small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/run_pipeline_small/grid_%A_%a.out
#SBATCH --error=logs/run_pipeline_small/grid_%A_%a.err

set -euo pipefail

CODE_DIR="/home/users/s/schaern8/scratch/code_NS/code_NS_radiolejepa"
PRETRAINING_ROOT="/home/users/s/schaern8/scratch/code_NS/datasets/rgz20k-batches-py"
PROBE_ROOT="/home/users/s/schaern8/scratch/code_NS/datasets/mirabest-batches-py"
ENV_DIR="${RADIOLEJEPA_ENV:-/home/users/s/schaern8/scratch/envs/radiolejepa}"

RUNS=(
    "vit_tiny_patch16_224 2500"
    "vit_small_patch16_224 2500"
)

read -r MODEL SAMPLE_SIZE <<< "${RUNS[SLURM_ARRAY_TASK_ID]}"
SEED=14

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

module load GCCcore/11.3.0
module load Python/3.10.4
source "${ENV_DIR}/bin/activate"
cd "${CODE_DIR}"

echo "model=${MODEL} rgz_train_images=${SAMPLE_SIZE} rgz_val_images=4000 seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES:-not_set}"

python -u run_pipeline.py \
    --model "${MODEL}" \
    --sample-size "${SAMPLE_SIZE}" \
    --seed "${SEED}" \
    --pretraining-root "${PRETRAINING_ROOT}" \
    --probe-root "${PROBE_ROOT}"

echo "Completed model=${MODEL} rgz_train_images=${SAMPLE_SIZE} rgz_val_images=4000 seed=${SEED}"
