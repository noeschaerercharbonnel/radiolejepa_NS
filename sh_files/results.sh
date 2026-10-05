#!/bin/bash
#SBATCH --job-name=radiolejepa_results
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --partition=public-gpu
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/results/results_%j.out
#SBATCH --error=logs/results/results_%j.err

set -euo pipefail

CODE_DIR="/home/users/s/schaern8/scratch/code_NS/code_NS_radiolejepa"
PROBE_ROOT="/home/users/s/schaern8/scratch/code_NS/datasets/mirabest-batches-py"
ENV_DIR="${RADIOLEJEPA_ENV:-/home/users/s/schaern8/scratch/envs/radiolejepa}"

if [ ! -f "${ENV_DIR}/bin/activate" ]; then
    echo "Python environment not found: ${ENV_DIR}"
    exit 1
fi
module load GCCcore/11.3.0
module load Python/3.10.4
source "${ENV_DIR}/bin/activate"

cd "${CODE_DIR}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

echo "Python: $(command -v python)"

python -u results.py --require-complete --probe-root "${PROBE_ROOT}"

echo "Plots saved to ../outputs/results/"
