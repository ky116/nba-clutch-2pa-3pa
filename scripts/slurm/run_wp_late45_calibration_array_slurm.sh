#!/bin/bash
#SBATCH --job-name=wp_late45_cal
#SBATCH --output=logs/%x_%A_%a.txt
#SBATCH --error=logs/%x_%A_%a.txt
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --exclude=chiron,nevera

set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  for cand in "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${PWD}/../.." "${THIS_SCRIPT_DIR}/../.."; do
    if [[ -n "${cand}" && -f "${cand}/scripts/core/fit_wp_and_score_shots_late45.r" ]]; then
      PROJECT_DIR="$(cd "${cand}" && pwd)"
      break
    fi
  done
fi
if [[ -z "${PROJECT_DIR:-}" || ! -f "${PROJECT_DIR}/scripts/core/fit_wp_and_score_shots_late45.r" ]]; then
  echo "[error] cannot resolve PROJECT_DIR from PWD=${PWD} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} THIS_SCRIPT_DIR=${THIS_SCRIPT_DIR}" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs data/wp results/wp_calibration tmp

START_SEASON="${START_SEASON:-2000}"
END_SEASON="${END_SEASON:-2024}"
SEASONTYPE="${SEASONTYPE:-rs}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
SEASON_OOF_JOBS="${SEASON_OOF_JOBS:-1}"
BAM_NTHREADS="${BAM_NTHREADS:-8}"
LATE_TAIL_TIME_SEC="${LATE_TAIL_TIME_SEC:-45}"
LATE_TAIL_SCORE_ABS="${LATE_TAIL_SCORE_ABS:-7}"

VARIANTS=(none main band score-band surface)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#VARIANTS[@]}" ]]; then
  echo "[error] SLURM_ARRAY_TASK_ID out of range: ${TASK_ID}" >&2
  exit 1
fi
VARIANT="${LATE_TAIL_VARIANT:-${VARIANTS[$TASK_ID]}}"
SAFE_VARIANT="${VARIANT//-/_}"
RUN_TAG="${RUN_TAG:-late45_${SAFE_VARIANT}}"

TRAIN_PATH="${TRAIN_PATH:-data/wp/wp_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.csv.gz}"
SHOT_PATH="${SHOT_PATH:-data/wp/shot_decision_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.csv.gz}"
WP_OUT="${WP_OUT:-data/wp/shot_decision_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_with_wp_${RUN_TAG}.csv.gz}"
FULL_OUT="${FULL_OUT:-data/wp/shot_decision_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_with_wp_full_${RUN_TAG}.csv.gz}"
DML_OUT="${DML_OUT:-data/wp/shot_decision_panel_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_dml_${RUN_TAG}.csv.gz}"
METRICS_OUT="${METRICS_OUT:-results/wp_calibration/wp_calibration_${RUN_TAG}_next_offense_late45.csv}"
TRAILING3_OUT="${TRAILING3_OUT:-results/wp_calibration/wp_calibration_trailing3_made_${RUN_TAG}.csv}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export TMPDIR="${PROJECT_DIR}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

if [[ -z "${R_LIBS_USER:-}" ]]; then
  cand="$(find "${HOME}/R" -maxdepth 3 -type d -name "*-linux-gnu-library" 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "${cand}" ]]; then
    ver="$(find "${cand}" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n1 || true)"
    if [[ -n "${ver}" ]]; then
      export R_LIBS_USER="${ver}"
    fi
  fi
fi

echo "[run] project=${PROJECT_DIR}"
echo "[run] variant=${VARIANT} run_tag=${RUN_TAG}"
echo "[run] seasons=${START_SEASON}-${END_SEASON} seasontype=${SEASONTYPE}"
echo "[run] train_path=${TRAIN_PATH}"
echo "[run] shot_path=${SHOT_PATH}"
echo "[run] threads=${THREADS} season_oof_jobs=${SEASON_OOF_JOBS} bam_nthreads=${BAM_NTHREADS}"
echo "[run] R_LIBS_USER=${R_LIBS_USER:-<unset>}"
echo "[run] TMPDIR=${TMPDIR}"

"${RSCRIPT_BIN}" scripts/core/fit_wp_and_score_shots_late45.r \
  --start-season "${START_SEASON}" \
  --end-season "${END_SEASON}" \
  --seasontype "${SEASONTYPE}" \
  --train-path "${TRAIN_PATH}" \
  --shot-path "${SHOT_PATH}" \
  --protocol-m0-spec \
  --oof-template-protocol-m0-spec \
  --season-oof-jobs "${SEASON_OOF_JOBS}" \
  --bam-nthreads "${BAM_NTHREADS}" \
  --late-tail-variant "${VARIANT}" \
  --late-tail-time-sec "${LATE_TAIL_TIME_SEC}" \
  --late-tail-score-abs "${LATE_TAIL_SCORE_ABS}" \
  --wp-out "${WP_OUT}" \
  --full-out "${FULL_OUT}" \
  --dml-out "${DML_OUT}"

"${PYTHON_BIN}" scripts/helpers/wp_calibration_metrics.py \
  --with-wp "${WP_OUT}" \
  --target next \
  --offense-oriented \
  --exclude-terminal-next \
  --max-time "${LATE_TAIL_TIME_SEC}" \
  --score-abs "${LATE_TAIL_SCORE_ABS}" \
  --out "${METRICS_OUT}"

"${PYTHON_BIN}" scripts/helpers/summarize_trailing3_made_wp_calibration.py \
  --with-wp "${WP_OUT}" \
  --out "${TRAILING3_OUT}"

echo "[done] wp_out=${WP_OUT}"
echo "[done] metrics=${METRICS_OUT}"
echo "[done] trailing3=${TRAILING3_OUT}"
