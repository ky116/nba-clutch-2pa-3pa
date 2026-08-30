#!/bin/bash
#SBATCH --job-name=nba_nested_wf_catboost_gpu
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

# Defaults (override via env or CLI)
INPUT="${INPUT:-data/analysis/shotchoice_panel_clutch_rs.parquet}"
OUTDIR="${OUTDIR:-results/nested_wf_catboost_gpu}"
TRAIN_START="${TRAIN_START:-2000}"
TRAIN_END_INIT="${TRAIN_END_INIT:-2009}"
TEST_SPAN="${TEST_SPAN:-3}"
STEP="${STEP:-3}"
MAX_SEASON="${MAX_SEASON:-}"
INNER_TRAIN_INIT_SPAN="${INNER_TRAIN_INIT_SPAN:-4}"
INNER_BLOCK_SPAN="${INNER_BLOCK_SPAN:-3}"
INNER_STEP="${INNER_STEP:-3}"
RANDOM_STATE="${RANDOM_STATE:-123}"
MIN_SAMPLES_PER_TREAT="${MIN_SAMPLES_PER_TREAT:-200}"
MIN_PROP="${MIN_PROP:-1e-2}"
MAX_PROP="${MAX_PROP:-1.0}"
TREAT_A="${TREAT_A:-three-point}"
TREAT_B="${TREAT_B:-two-point}"
TREATMENT_SCHEME="${TREATMENT_SCHEME:-binary}"

mkdir -p logs

if [[ -d ".venv" ]]; then
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

# Thread settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# CatBoost GPU settings (consumed by dml_models.py)
export CATBOOST_USE_GPU=1
export CATBOOST_DEVICES=${CATBOOST_DEVICES:-0}

echo "Python: $PYTHON_CMD"
echo "Python version: $("$PYTHON_CMD" -V 2>&1)"
echo "Python site-packages: ${PYTHON_SITE:-<none>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-UNSET}"
echo "CATBOOST_USE_GPU=${CATBOOST_USE_GPU}"
echo "CATBOOST_DEVICES=${CATBOOST_DEVICES}"
nvidia-smi || true

PRECHECK=(
  "$PYTHON_CMD" scripts/core/preflight_nested_walk_forward.py
  --input "$INPUT"
  --treat-a "$TREAT_A"
  --treat-b "$TREAT_B"
  --treatment-scheme "$TREATMENT_SCHEME"
  --train-start "$TRAIN_START"
  --train-end-init "$TRAIN_END_INIT"
  --test-span "$TEST_SPAN"
  --step "$STEP"
  --inner-train-init-span "$INNER_TRAIN_INIT_SPAN"
  --inner-block-span "$INNER_BLOCK_SPAN"
  --inner-step "$INNER_STEP"
  --min-samples-per-treat "$MIN_SAMPLES_PER_TREAT"
)
if [[ -n "$MAX_SEASON" ]]; then
  PRECHECK+=(--max-season "$MAX_SEASON")
fi
echo "Running preflight: ${PRECHECK[*]} ${EXTRA_ARGS[*]}"
"${PRECHECK[@]}" "${EXTRA_ARGS[@]}"

CMD=(
  "$PYTHON_CMD" scripts/core/run_nested_walk_forward.py
  --input "$INPUT"
  --outdir "$OUTDIR"
  --train-start "$TRAIN_START"
  --train-end-init "$TRAIN_END_INIT"
  --test-span "$TEST_SPAN"
  --step "$STEP"
  --inner-train-init-span "$INNER_TRAIN_INIT_SPAN"
  --inner-block-span "$INNER_BLOCK_SPAN"
  --inner-step "$INNER_STEP"
  --prop-model catboost
  --outcome-model catboost
  --tau-model catboost
  --random-state "$RANDOM_STATE"
  --min-samples-per-treat "$MIN_SAMPLES_PER_TREAT"
  --min-prop "$MIN_PROP"
  --max-prop "$MAX_PROP"
  --treat-a "$TREAT_A"
  --treat-b "$TREAT_B"
  --treatment-scheme "$TREATMENT_SCHEME"
)

if [[ -n "$MAX_SEASON" ]]; then
  CMD+=(--max-season "$MAX_SEASON")
fi
# Pass additional CLI args to scripts/core/run_nested_walk_forward.py
echo "Running: ${CMD[*]} ${EXTRA_ARGS[*]}"
"${CMD[@]}" "${EXTRA_ARGS[@]}"
