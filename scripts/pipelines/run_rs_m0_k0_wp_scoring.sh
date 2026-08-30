#!/usr/bin/env bash
set -euo pipefail

# Score RS shot-decision states with the frozen primary WP specification:
# baseline M0 GAM + localized tail surface for 0-45 s and |score_diff| <= 7.
# This script only writes WP-scored shot-state files; panel construction is
# handled by run_rs_m0_k0_panel_from_wp.sh.

START_SEASON="${START_SEASON:-2000}"
END_SEASON="${END_SEASON:-2024}"
SEASONTYPE="${SEASONTYPE:-rs}"

TRAIN_PATH="${TRAIN_PATH:-data/wp/wp_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.csv.gz}"
SHOT_PATH="${SHOT_PATH:-data/wp/shot_decision_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.csv.gz}"

ADOPTED_MODEL_IN="${ADOPTED_MODEL_IN:-models/wp_gam_m0_elo_k0_2000_2024_rs.rds}"
MODEL_IN="${MODEL_IN:-${ADOPTED_MODEL_IN}}"
MODEL_OUT="${MODEL_OUT:-models/wp_gam_m0_elo_k0_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.rds}"
TRAIN_MODEL="${TRAIN_MODEL:-0}"
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"
NO_ELO="${NO_ELO:-1}"
ALLOW_NON_ADOPTED_MODEL="${ALLOW_NON_ADOPTED_MODEL:-0}"
OOF_TEMPLATE_PROTOCOL_M0="${OOF_TEMPLATE_PROTOCOL_M0:-1}"

WP_SCORE_SCRIPT="${WP_SCORE_SCRIPT:-scripts/core/fit_wp_and_score_shots_late45.r}"
LATE_TAIL_VARIANT="${LATE_TAIL_VARIANT:-surface}"
LATE_TAIL_TIME_SEC="${LATE_TAIL_TIME_SEC:-45}"
LATE_TAIL_SCORE_ABS="${LATE_TAIL_SCORE_ABS:-7}"

RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
SEASON_OOF_JOBS="${SEASON_OOF_JOBS:-1}"
BAM_NTHREADS="${BAM_NTHREADS:-1}"

if [[ -z "${R_LIBS_USER:-}" ]]; then
  cand="$(find "${HOME}/R" -maxdepth 3 -type d -name "*-linux-gnu-library" 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "${cand}" ]]; then
    ver="$(find "${cand}" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n1 || true)"
    [[ -n "${ver}" ]] && export R_LIBS_USER="${ver}"
  fi
fi

if [[ "${THREADS}" =~ ^[0-9]+$ ]] && [[ "${THREADS}" -ge 1 ]]; then
  export OMP_NUM_THREADS="${THREADS}"
  export OPENBLAS_NUM_THREADS="${THREADS}"
  export MKL_NUM_THREADS="${THREADS}"
  export NUMEXPR_NUM_THREADS="${THREADS}"
fi

echo "[wp_score] threads=${THREADS} season_oof_jobs=${SEASON_OOF_JOBS} bam_nthreads=${BAM_NTHREADS}"
echo "[wp_score] R_LIBS_USER=${R_LIBS_USER:-<unset>}"
echo "[wp_score] oof_template_protocol_m0=${OOF_TEMPLATE_PROTOCOL_M0}"
echo "[wp_score] wp_score_script=${WP_SCORE_SCRIPT}"
echo "[wp_score] late_tail_variant=${LATE_TAIL_VARIANT} time_sec=${LATE_TAIL_TIME_SEC} score_abs=${LATE_TAIL_SCORE_ABS}"

if [[ "${NO_ELO}" != "1" ]]; then
  echo "[error] This WP scoring script is no-Elo flow only. Set NO_ELO=1." >&2
  exit 1
fi

if [[ "${OOF_TEMPLATE_PROTOCOL_M0}" == "1" && "${TRAIN_MODEL}" == "1" ]]; then
  echo "[error] OOF_TEMPLATE_PROTOCOL_M0=1 is incompatible with TRAIN_MODEL=1." >&2
  exit 1
fi

if [[ "${TRAIN_MODEL}" == "1" ]]; then
  echo "[wp_score] train M0-style model on RS states: ${TRAIN_PATH}"
  "${RSCRIPT_BIN}" scripts/core/fit_wp_model_late45.r \
    --start-season "${START_SEASON}" \
    --end-season "${END_SEASON}" \
    --seasontype "${SEASONTYPE}" \
    --train-path "${TRAIN_PATH}" \
    --model-out "${MODEL_OUT}" \
    --no-era-smooth-interaction \
    --protocol-m0-spec \
    --late-tail-variant "${LATE_TAIL_VARIANT}" \
    --late-tail-time-sec "${LATE_TAIL_TIME_SEC}" \
    --late-tail-score-abs "${LATE_TAIL_SCORE_ABS}"
  MODEL_IN="${MODEL_OUT}"
fi

if [[ "${OOF_TEMPLATE_PROTOCOL_M0}" != "1" ]] && [[ "${ALLOW_NON_ADOPTED_MODEL}" != "1" ]] && [[ "${MODEL_IN}" != "${ADOPTED_MODEL_IN}" ]] && [[ "${TRAIN_MODEL}" != "1" ]]; then
  echo "[error] MODEL_IN is not the adopted model: ${MODEL_IN}" >&2
  exit 1
fi

if [[ "${OOF_TEMPLATE_PROTOCOL_M0}" != "1" && ! -f "${MODEL_IN}" ]]; then
  if [[ "${TRAIN_IF_MISSING}" != "1" ]]; then
    echo "[error] model not found: ${MODEL_IN}" >&2
    exit 1
  fi
  echo "[wp_score] model not found, auto-training: ${MODEL_IN}"
  "${RSCRIPT_BIN}" scripts/core/fit_wp_model_late45.r \
    --start-season "${START_SEASON}" \
    --end-season "${END_SEASON}" \
    --seasontype "${SEASONTYPE}" \
    --train-path "${TRAIN_PATH}" \
    --model-out "${MODEL_OUT}" \
    --no-era-smooth-interaction \
    --protocol-m0-spec \
    --late-tail-variant "${LATE_TAIL_VARIANT}" \
    --late-tail-time-sec "${LATE_TAIL_TIME_SEC}" \
    --late-tail-score-abs "${LATE_TAIL_SCORE_ABS}"
  MODEL_IN="${MODEL_OUT}"
fi

if [[ ! -f "${SHOT_PATH}" ]]; then
  echo "[error] shot state not found: ${SHOT_PATH}" >&2
  exit 1
fi

WP_SCORE_ARGS=(
  --start-season "${START_SEASON}"
  --end-season "${END_SEASON}"
  --seasontype "${SEASONTYPE}"
  --shot-path "${SHOT_PATH}"
  --season-oof-jobs "${SEASON_OOF_JOBS}"
  --bam-nthreads "${BAM_NTHREADS}"
)
if [[ "${OOF_TEMPLATE_PROTOCOL_M0}" == "1" ]]; then
  WP_SCORE_ARGS+=(--protocol-m0-spec --oof-template-protocol-m0-spec)
else
  WP_SCORE_ARGS+=(--model-in "${MODEL_IN}")
fi
if [[ "${LATE_TAIL_VARIANT}" != "none" ]]; then
  WP_SCORE_ARGS+=(
    --late-tail-variant "${LATE_TAIL_VARIANT}"
    --late-tail-time-sec "${LATE_TAIL_TIME_SEC}"
    --late-tail-score-abs "${LATE_TAIL_SCORE_ABS}"
  )
fi

"${RSCRIPT_BIN}" "${WP_SCORE_SCRIPT}" "${WP_SCORE_ARGS[@]}"

WP_WITH_PATH="data/wp/shot_decision_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_with_wp.csv.gz"
if [[ ! -f "${WP_WITH_PATH}" ]]; then
  echo "[error] expected scored shot file not found: ${WP_WITH_PATH}" >&2
  exit 1
fi
echo "[done] scored shot states: ${WP_WITH_PATH}"
