#!/bin/bash
#SBATCH --job-name=wp_features_rs
#SBATCH --cpus-per-task=25
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH -o ./logs/%j.txt
#
# Submit examples:
#   sbatch scripts/slurm/run_wp_features_per_season_slurm.sh
#   sbatch --cpus-per-task=32 run_wp_features_per_season_slurm.sh
#   sbatch scripts/slurm/run_wp_features_per_season_slurm.sh --start-season 2000 --end-season 2024 --seasontype rs

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
cd "${PROJECT_DIR}"

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

START_SEASON="${START_SEASON:-2000}"
END_SEASON="${END_SEASON:-2024}"
SEASONTYPE="${SEASONTYPE:-rs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${JOBS:-25}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p logs

if [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

echo "================================================================"
echo "WP feature build per season (SLURM)"
echo "================================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "START_SEASON=${START_SEASON}"
echo "END_SEASON=${END_SEASON}"
echo "SEASONTYPE=${SEASONTYPE}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "JOBS=${JOBS}"
echo "PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

bash "${PROJECT_DIR}/scripts/helpers/run_wp_features_per_season.sh" \
  --start-season "${START_SEASON}" \
  --end-season "${END_SEASON}" \
  --seasontype "${SEASONTYPE}" \
  --python-bin "${PYTHON_BIN}" \
  --jobs "${JOBS}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "================================================================"
echo "Combining per-season WP outputs"
echo "================================================================"
"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/core/combine_wp_states.py" \
  --start-season "${START_SEASON}" \
  --end-season "${END_SEASON}" \
  --seasontype "${SEASONTYPE}"

echo ""
echo "[done] build + combine finished"
