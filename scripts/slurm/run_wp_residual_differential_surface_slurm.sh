#!/bin/bash
#SBATCH --job-name=wp_resid_diff_surface
#SBATCH --output=logs/%j.txt
#SBATCH --error=logs/%j.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --chdir=.

set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  for cand in "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${PWD}/../.." "${THIS_SCRIPT_DIR}/../.."; do
    if [[ -n "${cand}" && -f "${cand}/scripts/helpers/fit_wp_residual_differential_surface.py" ]]; then
      PROJECT_DIR="$(cd "${cand}" && pwd)"
      break
    fi
  done
fi
if [[ -z "${PROJECT_DIR:-}" || ! -f "${PROJECT_DIR}/scripts/helpers/fit_wp_residual_differential_surface.py" ]]; then
  echo "[error] cannot resolve PROJECT_DIR from PWD=${PWD} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} THIS_SCRIPT_DIR=${THIS_SCRIPT_DIR}" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs .joblib

if [[ -d "${PROJECT_DIR}/.venv" ]]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

PYTHON_CMD="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_CMD}" ]]; then
  PYTHON_CMD="$(command -v python3)"
fi

THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export LGBM_NUM_THREADS="${THREADS}"
export XGB_NUM_THREADS="${THREADS}"
export JOBLIB_TEMP_FOLDER="${PROJECT_DIR}/.joblib"

PANEL="${PANEL:-data/analysis/shotchoice_panel_clutch_rs.parquet}"
WITH_WP="${WITH_WP:-data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz}"
OUTDIR="${OUTDIR:-results/wp_calibration/differential_surface}"
PARAMS_JSON="${PARAMS_JSON:-results/nested_wf_lgbm/train2000_2021_test2022_2024/meta.json}"
MAX_ESTIMATORS="${MAX_ESTIMATORS:-1200}"
N_SAMPLE="${N_SAMPLE:-100000}"

echo "================================================================================"
echo "WP residual differential calibration surface"
echo "================================================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Node=${SLURMD_NODENAME:-unknown}"
echo "Threads=${THREADS}"
echo "Python=${PYTHON_CMD}"
echo "Panel=${PANEL}"
echo "With WP=${WITH_WP}"
echo "Outdir=${OUTDIR}"
echo "Params=${PARAMS_JSON}"
echo "Max estimators=${MAX_ESTIMATORS}"
echo "Surface marginalization sample=${N_SAMPLE}"
echo ""

"${PYTHON_CMD}" -c "import numpy, pandas, sklearn, lightgbm, joblib"

exec "${PYTHON_CMD}" scripts/helpers/fit_wp_residual_differential_surface.py \
  --panel "${PANEL}" \
  --with-wp "${WITH_WP}" \
  --outdir "${OUTDIR}" \
  --params-json "${PARAMS_JSON}" \
  --max-estimators "${MAX_ESTIMATORS}" \
  --n-sample "${N_SAMPLE}" \
  "$@"
