#!/bin/bash
#SBATCH --job-name=nba_nested_wf_lgbm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=144:00:00
#SBATCH --chdir=.
#SBATCH -o logs/%j.txt
#
# Submit example:
#   sbatch scripts/slurm/run_nested_walk_forward_lgbm_slurm.sh
#   sbatch scripts/slurm/run_nested_walk_forward_lgbm_slurm.sh \
#     --input data/analysis/shotchoice_panel_clutch_rs.parquet \
#     --outdir results/nested_wf_lgbm

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

# Strip scheduler-only args so they are not forwarded to Python.
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

INPUT="${INPUT:-data/analysis/shotchoice_panel_clutch_rs.parquet}"
OUTDIR="${OUTDIR:-results/nested_wf_lgbm}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
TREATMENT_SCHEME="${TREATMENT_SCHEME:-binary}"

mkdir -p logs
mkdir -p .joblib

if [[ -d ".venv" ]]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
fi

PYTHON_SITE="${PYTHON_SITE:-}"
PYTHON_CMD=""
_try_python_cmd() {
  local cand="$1"
  local label="${2:-$1}"
  local py_mm=""
  [[ -x "$cand" ]] || return 0
  py_mm="$("$cand" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
  case "$py_mm" in
    3.10|3.11|3.12)
      PYTHON_CMD="$cand"
      ;;
    *)
      echo "[warn] ignoring ${label} (requires Python >= 3.10, got ${py_mm:-unknown})"
      ;;
  esac
}
if [[ -n "${PYTHON_BIN:-}" ]]; then
  _try_python_cmd "$PYTHON_BIN" "PYTHON_BIN=${PYTHON_BIN}"
fi
if [[ -z "$PYTHON_CMD" ]]; then
  _try_python_cmd "$PWD/.venv/bin/python" "$PWD/.venv/bin/python"
fi
if [[ -z "$PYTHON_CMD" ]]; then
  _try_python_cmd "$PWD/.venv/bin/python3.10" "$PWD/.venv/bin/python3.10"
fi
if [[ -z "$PYTHON_CMD" ]]; then
  _try_python_cmd "/usr/bin/python3.10" "/usr/bin/python3.10"
fi
if [[ -z "$PYTHON_CMD" ]]; then
  _try_python_cmd "/usr/bin/python3" "/usr/bin/python3"
fi
if [[ -z "$PYTHON_CMD" ]]; then
  echo "[error] Python >= 3.10 interpreter not found on node ${SLURMD_NODENAME:-unknown}." >&2
  echo "[error] Set PYTHON_BIN to a compatible Python executable or exclude this node." >&2
  exit 1
fi
if [[ -z "$PYTHON_SITE" ]]; then
  PYTHON_SITE="$("$PYTHON_CMD" -c 'import sysconfig; print(sysconfig.get_paths().get("purelib",""))' 2>/dev/null || true)"
fi
if [[ -n "${PYTHON_SITE:-}" && -d "${PYTHON_SITE}" ]]; then
  export PYTHONPATH="${PYTHON_SITE}${PYTHONPATH:+:$PYTHONPATH}"
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export LGBM_NUM_THREADS="$THREADS"
export JOBLIB_TEMP_FOLDER="$PWD/.joblib"

echo "================================================================================"
echo "Nested walk-forward (SLURM): LightGBM"
echo "================================================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Input: $INPUT"
echo "Outdir: $OUTDIR"
echo "Available CPUs (nproc): $(nproc 2>/dev/null || echo 1)"
echo "Threads: $THREADS"
echo "Python: $PYTHON_CMD"
echo "Python version: $("$PYTHON_CMD" -V 2>&1)"
echo "Python site-packages: ${PYTHON_SITE:-<none>}"
echo "Treatment scheme: $TREATMENT_SCHEME"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo ""

echo "Checking Python dependencies..."
"$PYTHON_CMD" -c "import numpy, pandas, sklearn, lightgbm"

echo "Running preflight checks..."
"$PYTHON_CMD" scripts/core/preflight_nested_walk_forward.py \
  --input "$INPUT" \
  --treatment-scheme "$TREATMENT_SCHEME" \
  "${EXTRA_ARGS[@]}"

echo "Starting LightGBM nested walk-forward..."
CMD=(
  "$PYTHON_CMD" scripts/core/run_nested_walk_forward.py
  --input "$INPUT"
  --outdir "$OUTDIR"
  --prop-model lgbm
  --outcome-model lgbm
  --tau-model lgbm
  --treatment-scheme "$TREATMENT_SCHEME"
)
"${CMD[@]}" \
  "${EXTRA_ARGS[@]}"

echo "LightGBM run completed successfully."
