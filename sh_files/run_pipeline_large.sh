#!/bin/bash
#SBATCH --job-name=radiolejepa_large
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --partition=public-gpu
#SBATCH --constraint=COMPUTE_CAPABILITY_7_5
#SBATCH --time=2-00:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/run_pipeline_large/grid_%A_%a.out
#SBATCH --error=logs/run_pipeline_large/grid_%A_%a.err

set -euo pipefail

CODE_DIR="/home/users/s/schaern8/scratch/code_NS/code_NS_radiolejepa"
PRETRAINING_ROOT="/home/users/s/schaern8/scratch/code_NS/datasets/rgz20k-batches-py"
PROBE_ROOT="/home/users/s/schaern8/scratch/code_NS/datasets/mirabest-batches-py"
ENV_DIR="${RADIOLEJEPA_ENV:-/home/users/s/schaern8/scratch/envs/radiolejepa}"

RUNS=(
    "vit_tiny_patch16_224 5000"
    "vit_tiny_patch16_224 7500"
    "vit_tiny_patch16_224 10000"
    "vit_tiny_patch16_224 13000"
    "vit_tiny_patch16_224 16000"
    "vit_small_patch16_224 5000"
    "vit_small_patch16_224 7500"
    "vit_small_patch16_224 10000"
    "vit_small_patch16_224 13000"
    "vit_small_patch16_224 16000"
    "vit_base_patch16_224 2500"
    "vit_base_patch16_224 5000"
    "vit_base_patch16_224 7500"
    "vit_base_patch16_224 10000"
    "vit_base_patch16_224 13000"
    "vit_base_patch16_224 16000"
    "vit_large_patch16_224 2500"
    "vit_large_patch16_224 5000"
    "vit_large_patch16_224 7500"
    "vit_large_patch16_224 10000"
    "vit_large_patch16_224 13000"
    "vit_large_patch16_224 16000"
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
