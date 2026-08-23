#!/bin/bash
#SBATCH --job-name=nba_nested_wf_xgb_lgbm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH -o ./logs/%j.txt
#
# Submit example:
#   sbatch scripts/slurm/run_nested_walk_forward_xgb_lgbm_slurm.sh
#   sbatch scripts/slurm/run_nested_walk_forward_xgb_lgbm_slurm.sh \
#     --input data/analysis/shotchoice_panel_clutch_rs.parquet \
#     --outdir-base results/nested_wf

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

INPUT="${INPUT:-data/analysis/shotchoice_panel_clutch_rs.parquet}"
OUTDIR_BASE="${OUTDIR_BASE:-results/nested_wf}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
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

echo "================================================================================"
echo "Nested walk-forward (SLURM): XGBoost & LightGBM"
echo "================================================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Input: $INPUT"
echo "Outdir base: $OUTDIR_BASE"
echo "Threads: $THREADS"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

bash "${PROJECT_DIR}/scripts/pipelines/run_nested_walk_forward_xgb_lgbm.sh" \
  --input "$INPUT" \
  --outdir-base "$OUTDIR_BASE" \
  --threads "$THREADS" \
  "${EXTRA_ARGS[@]}"
