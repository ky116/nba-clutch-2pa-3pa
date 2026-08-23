#!/bin/bash
#SBATCH --job-name=nba_full_data_catboost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=144:00:00
#SBATCH --chdir=.
#SBATCH -o logs/%j.txt

set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  for cand in "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${PWD}/../.." "${THIS_SCRIPT_DIR}/../.."; do
    if [[ -n "${cand}" && -f "${cand}/scripts/core/fit_wp_model.r" ]]; then
      PROJECT_DIR="$(cd "${cand}" && pwd)"
      break
    fi
  done
fi
if [[ -z "${PROJECT_DIR:-}" || ! -f "${PROJECT_DIR}/scripts/core/fit_wp_model.r" ]]; then
  echo "[error] cannot resolve PROJECT_DIR from PWD=${PWD} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} THIS_SCRIPT_DIR=${THIS_SCRIPT_DIR}" >&2
  exit 1
fi
cd "$PROJECT_DIR"

# Strip scheduler-only args so they are not forwarded.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exclude)
      shift
      [[ $# -gt 0 ]] && shift
      ;;
    --exclude=*)
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p logs

# Thread settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# CatBoost GPU settings (consumed by dml_models.py)
export CATBOOST_USE_GPU=1
export CATBOOST_DEVICES=${CATBOOST_DEVICES:-0}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-UNSET}"
echo "CATBOOST_USE_GPU=${CATBOOST_USE_GPU}"
echo "CATBOOST_DEVICES=${CATBOOST_DEVICES}"
nvidia-smi || true

CMD=(
  bash "${PROJECT_DIR}/scripts/helpers/run_full_data.sh"
  --threads "${SLURM_CPUS_PER_TASK:-8}"
)

echo "Running: ${CMD[*]} ${EXTRA_ARGS[*]}"
"${CMD[@]}" "${EXTRA_ARGS[@]}"
