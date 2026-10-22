#!/usr/bin/env bash

# Direct multi-GPU launcher for Mimosa, which does not use Slurm.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${RADIOLEJEPA_ENV:-${HOME}/envs/radiolejepa}/bin/python}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../datasets}"
PRETRAINING_ROOT="${PRETRAINING_ROOT:-${DATA_ROOT}/rgz20k-batches-py}"
PROBE_ROOT="${PROBE_ROOT:-${DATA_ROOT}/mirabest-batches-py}"
SEED="${SEED:-14}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/mimosa}"
PHASE="${1:-grid}"

# Edit the experiment grid here; all Mimosa stages use these lists.
MODELS=(
    vit_tiny_patch16_224
    vit_small_patch16_224
    vit_base_patch16_224
)
SUBSET_SIZES=(2500 5000 7500 10000 13000 16000)
MB_TRAIN_FRACTIONS=(0.10 0.25 0.50 0.75)

GRID_JOBS=()
PROBE_JOBS=()
LARGEST_SUBSET=0
for sample_size in "${SUBSET_SIZES[@]}"; do
    ((sample_size > LARGEST_SUBSET)) && LARGEST_SUBSET="${sample_size}"
done
for model in "${MODELS[@]}"; do
    for sample_size in "${SUBSET_SIZES[@]}"; do
        GRID_JOBS+=("${model} ${sample_size}")
    done
    for fraction in "${MB_TRAIN_FRACTIONS[@]}"; do
        PROBE_JOBS+=("${model} ${fraction}")
    done
done

usage() {
    cat <<EOF
Usage: $0 [grid|probe-scaling|results|all]

  grid           Run ${#GRID_JOBS[@]} pretraining jobs and both MiraBest methods (default).
  probe-scaling  Run ${#PROBE_JOBS[@]} fractions with both methods; grid supplies 100% results.
  results        Evaluate missing MiraBest fractions and create plots.
  all            Run grid, probe-scaling, then results.

Overrides: GPU_IDS, PYTHON_BIN, DATA_ROOT, PRETRAINING_ROOT, PROBE_ROOT, SEED,
           CPU_THREADS_PER_JOB, LOG_DIR
EOF
}

initialise_gpus() {
    local gpu_id active_pids
    local -a candidates=() busy_gpus=()

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi was not found; cannot verify GPU availability." >&2
        return 1
    fi
    if [[ -n "${GPU_IDS:-}" ]]; then
        read -r -a candidates <<< "${GPU_IDS}"
    else
        mapfile -t candidates < <(
            nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        )
    fi

    GPUS=()
    for gpu_id in "${candidates[@]}"; do
        active_pids="$(
            nvidia-smi -i "${gpu_id}" \
                --query-compute-apps=pid \
                --format=csv,noheader,nounits 2>/dev/null \
                | sed -n '/^[[:space:]]*[0-9][0-9]*[[:space:]]*$/p'
        )"
        if [[ -n "${active_pids}" ]]; then
            busy_gpus+=("${gpu_id}")
        else
            GPUS+=("${gpu_id}")
        fi
    done

    if [[ ${#busy_gpus[@]} -gt 0 ]]; then
        echo "Skipping GPUs with active compute processes: ${busy_gpus[*]}" >&2
    fi
    if [[ ${#GPUS[@]} -eq 0 ]]; then
        echo "No free GPUs are available; no jobs were started." >&2
        return 1
    fi
    echo "Free GPUs selected: ${GPUS[*]}"
}

validate_common() {
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Python environment not found: ${PYTHON_BIN}" >&2
        return 1
    fi
    if [[ "${PHASE}" != "results" ]] && ! "${PYTHON_BIN}" -c \
        'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
        echo "PyTorch cannot access CUDA in ${PYTHON_BIN}." >&2
        return 1
    fi
    mkdir -p "${LOG_DIR}"
    cd "${SCRIPT_DIR}" || return 1
    MIMOSA_CONFIG="${LOG_DIR}/mimosa_config_$$.yaml"
    "${PYTHON_BIN}" - "${MIMOSA_CONFIG}" "${MODELS[*]}" "${SUBSET_SIZES[*]}" \
        "${SEED}" "${PRETRAINING_ROOT}" <<'PY' || return 1
import sys
from pathlib import Path
import yaml

output, models, sizes, seed, root = sys.argv[1:]
config = yaml.safe_load(Path("baseline_config.yaml").read_text())
sizes = list(map(int, sizes.split()))
config["scaling"].update(models=models.split(), subset_sizes=sizes)
config["dataset"].update(sample_size=max(sizes), batches_root=root)
config["training"]["seed"] = int(seed)
Path(output).write_text(yaml.safe_dump(config, sort_keys=False))
PY
}

run_grid_worker() {
    local worker_index="$1" gpu_id="$2"
    local job_index model sample_size log_file
    local worker_status=0

    for ((job_index = worker_index; job_index < ${#GRID_JOBS[@]}; job_index += ${#GPUS[@]})); do
        read -r model sample_size <<< "${GRID_JOBS[job_index]}"
        log_file="${LOG_DIR}/grid_${model}_n_${sample_size}_seed_${SEED}.log"
        echo "GPU ${gpu_id}: starting ${model}, RGZ train=${sample_size}, val=4000; log=${log_file}"
        if CUDA_VISIBLE_DEVICES="${gpu_id}" \
            OMP_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            MKL_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            "${PYTHON_BIN}" -u run_pipeline.py \
                --pretraining-config "${MIMOSA_CONFIG}" \
                --model "${model}" \
                --sample-size "${sample_size}" \
                --seed "${SEED}" \
                --pretraining-root "${PRETRAINING_ROOT}" \
                --probe-root "${PROBE_ROOT}" \
                >"${log_file}" 2>&1; then
            echo "GPU ${gpu_id}: completed ${model}, RGZ train=${sample_size}, val=4000"
        else
            echo "GPU ${gpu_id}: FAILED ${model}, RGZ train=${sample_size}, val=4000; see ${log_file}" >&2
            worker_status=1
        fi
    done
    return "${worker_status}"
}

run_probe_worker() {
    local worker_index="$1" gpu_id="$2"
    local job_index model fraction log_file
    local worker_status=0

    for ((job_index = worker_index; job_index < ${#PROBE_JOBS[@]}; job_index += ${#GPUS[@]})); do
        read -r model fraction <<< "${PROBE_JOBS[job_index]}"
        log_file="${LOG_DIR}/probe_${model}_n_${LARGEST_SUBSET}_mb_train_fraction_${fraction}_seed_${SEED}.log"
        echo "GPU ${gpu_id}: starting ${model}, fraction=${fraction}; log=${log_file}"
        if CUDA_VISIBLE_DEVICES="${gpu_id}" \
            OMP_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            MKL_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_JOB}" \
            "${PYTHON_BIN}" -u run_probe_scaling.py \
                --pretraining-config "${MIMOSA_CONFIG}" \
                --model "${model}" \
                --fraction "${fraction}" \
                --seed "${SEED}" \
                --probe-root "${PROBE_ROOT}" \
                >"${log_file}" 2>&1; then
            echo "GPU ${gpu_id}: completed ${model}, fraction=${fraction}"
        else
            echo "GPU ${gpu_id}: FAILED ${model}, fraction=${fraction}; see ${log_file}" >&2
            worker_status=1
        fi
    done
    return "${worker_status}"
}

run_parallel() {
    local worker_function="$1" description="$2"
    local worker_index pid status=0
    local -a pids=()

    echo "Launching ${description} across ${#GPUS[@]} GPUs: ${GPUS[*]}"
    for worker_index in "${!GPUS[@]}"; do
        "${worker_function}" "${worker_index}" "${GPUS[worker_index]}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    return "${status}"
}

run_grid() {
    if [[ ! -d "${PRETRAINING_ROOT}" || ! -d "${PROBE_ROOT}" ]]; then
        echo "Dataset directory missing: ${PRETRAINING_ROOT} or ${PROBE_ROOT}" >&2
        return 1
    fi
    initialise_gpus && run_parallel run_grid_worker "${#GRID_JOBS[@]} grid jobs"
}

run_probe_scaling() {
    if [[ ! -d "${PROBE_ROOT}" ]]; then
        echo "MiraBest directory missing: ${PROBE_ROOT}" >&2
        return 1
    fi
    initialise_gpus && run_parallel run_probe_worker "${#PROBE_JOBS[@]} probe jobs"
}

run_results() {
    local matplotlib_dir="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/radiolejepa-matplotlib}"
    mkdir -p "${matplotlib_dir}"
    MPLCONFIGDIR="${matplotlib_dir}" "${PYTHON_BIN}" -u results.py \
        --pretraining-config "${MIMOSA_CONFIG}" --require-complete --probe-root "${PROBE_ROOT}"
}

case "${PHASE}" in -h|--help|help) usage; exit 0 ;; esac
validate_common || exit 1
case "${PHASE}" in
    grid) run_grid ;;
    probe-scaling) run_probe_scaling ;;
    results) run_results ;;
    all) run_grid && run_probe_scaling && run_results ;;
    -h|--help|help) usage ;;
    *) echo "Unknown phase: ${PHASE}" >&2; usage >&2; exit 2 ;;
esac
