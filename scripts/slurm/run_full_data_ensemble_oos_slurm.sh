#!/bin/bash
#SBATCH --job-name=nba_full_data_ensemble_oos
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00
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
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

CMD=(
  bash "${PROJECT_DIR}/scripts/pipelines/run_full_data_ensemble_oos.sh"
  --threads "${SLURM_CPUS_PER_TASK:-32}"
)

echo "Running: ${CMD[*]} $*"
"${CMD[@]}" "$@"
