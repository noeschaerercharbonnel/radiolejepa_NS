#!/bin/bash
#SBATCH --job-name=radiolejepa_visualization
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=shared-cpu
#SBATCH --time=00:15:00
#SBATCH --output=logs/visualization/visualization_%j.out
#SBATCH --error=logs/visualization/visualization_%j.err

set -euo pipefail

CODE_DIR="/home/users/s/schaern8/scratch/code_NS/code_NS_radiolejepa"
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

python -u visualization.py --dataset both

echo "Saved visualizations:"
echo "  ../outputs/rgz20k_augmentation_grid.png"
echo "  ../outputs/mirabest_probe_grid.png"
