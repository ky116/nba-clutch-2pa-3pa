#!/bin/bash
#
# run_nested_walk_forward_xgb_lgbm.sh
# Run nested walk-forward for XGBoost and LightGBM sequentially.
#
# Usage examples:
#   bash scripts/pipelines/run_nested_walk_forward_xgb_lgbm.sh
#   bash scripts/pipelines/run_nested_walk_forward_xgb_lgbm.sh --input data/analysis/shotchoice_panel_clutch_rs.parquet --outdir-base results/nested_wf
#   bash scripts/pipelines/run_nested_walk_forward_xgb_lgbm.sh --threads 16 --train-start 2000 --train-end-init 2009 --max-season 2024
#

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

INPUT="data/analysis/shotchoice_panel_clutch_rs.parquet"
OUTDIR_BASE="results/nested_wf"
NUM_THREADS=""
TREATMENT_SCHEME="binary"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT="$2"
            shift 2
            ;;
        --outdir-base)
            OUTDIR_BASE="$2"
            shift 2
            ;;
        --threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --treatment-scheme)
            TREATMENT_SCHEME="$2"
            shift 2
            ;;
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

AVAILABLE_CPUS="$(nproc 2>/dev/null || echo 1)"
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-}"
if [[ -z "${NUM_THREADS}" ]]; then
    if [[ -n "${SLURM_CPUS}" ]]; then
        NUM_THREADS="${SLURM_CPUS}"
    else
        NUM_THREADS="${AVAILABLE_CPUS}"
    fi
fi
if [[ "${NUM_THREADS}" -gt "${AVAILABLE_CPUS}" ]]; then
    if [[ -n "${SLURM_CPUS}" && "${NUM_THREADS}" -eq "${SLURM_CPUS}" ]]; then
        echo "nproc reports ${AVAILABLE_CPUS} CPUs, but using SLURM_CPUS_PER_TASK=${SLURM_CPUS}."
    else
        echo "Requested threads (${NUM_THREADS}) exceed available CPUs (${AVAILABLE_CPUS}); capping."
        NUM_THREADS="${AVAILABLE_CPUS}"
    fi
fi
if [[ "${NUM_THREADS}" -lt 1 ]]; then
    NUM_THREADS="1"
fi

mkdir -p logs
mkdir -p .joblib

if [[ -d ".venv" ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"
export NUMEXPR_NUM_THREADS="$NUM_THREADS"
export XGB_NUM_THREADS="$NUM_THREADS"
export LGBM_NUM_THREADS="$NUM_THREADS"
export JOBLIB_TEMP_FOLDER="$PWD/.joblib"

echo "================================================================================"
echo "Nested walk-forward: XGBoost & LightGBM"
echo "================================================================================"
echo "Input: $INPUT"
echo "Outdir base: $OUTDIR_BASE"
echo "Available CPUs (nproc): $AVAILABLE_CPUS"
echo "Threads: $NUM_THREADS"
echo "Treatment scheme: $TREATMENT_SCHEME"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}"
echo ""

echo "Running preflight checks..."
python3 "${PROJECT_DIR}/scripts/core/preflight_nested_walk_forward.py" \
  --input "$INPUT" \
  --treatment-scheme "$TREATMENT_SCHEME" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "logs/preflight_nested_wf_xgb_lgbm.log"

START_TIME=$(date +%s)

EXIT_XGB=0
EXIT_LGBM=0

echo "Starting XGBoost nested walk-forward..."
python3 "${PROJECT_DIR}/scripts/core/run_nested_walk_forward.py" \
  --input "$INPUT" \
  --outdir "${OUTDIR_BASE}_xgb" \
  --prop-model xgb \
  --outcome-model xgb \
  --tau-model xgb \
  --treatment-scheme "$TREATMENT_SCHEME" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "logs/nested_wf_xgb.log" || EXIT_XGB=$?

echo ""
echo "Starting LightGBM nested walk-forward..."
python3 "${PROJECT_DIR}/scripts/core/run_nested_walk_forward.py" \
  --input "$INPUT" \
  --outdir "${OUTDIR_BASE}_lgbm" \
  --prop-model lgbm \
  --outcome-model lgbm \
  --tau-model lgbm \
  --treatment-scheme "$TREATMENT_SCHEME" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "logs/nested_wf_lgbm.log" || EXIT_LGBM=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "Run complete"
echo "================================================================================"
echo "Exit codes:"
echo "  XGBoost:  $EXIT_XGB"
echo "  LightGBM: $EXIT_LGBM"
echo "Total elapsed: $ELAPSED sec ($((ELAPSED / 60)) min)"

if [[ $EXIT_XGB -ne 0 || $EXIT_LGBM -ne 0 ]]; then
    echo "Warning: Some runs failed. Check logs/nested_wf_xgb.log and logs/nested_wf_lgbm.log"
    exit 1
fi

echo "All runs completed successfully."
