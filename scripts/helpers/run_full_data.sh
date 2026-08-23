#!/bin/bash
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

# Full-data train wrapper (reuse nested-wf internals)

INPUT="${INPUT:-data/analysis/shotchoice_panel_clutch_rs.parquet}"
OUTDIR="${OUTDIR:-results/full_data_catboost_state_selected}"

TRAIN_START="${TRAIN_START:-2000}"
MAX_SEASON="${MAX_SEASON:-}"

INNER_TRAIN_INIT_SPAN="${INNER_TRAIN_INIT_SPAN:-10}"
INNER_BLOCK_SPAN="${INNER_BLOCK_SPAN:-3}"
INNER_STEP="${INNER_STEP:-3}"

TREAT_COL="${TREAT_COL:-shot_zone_choice}"
OUTCOME_COL="${OUTCOME_COL:-delta_wp}"
SEASON_COL="${SEASON_COL:-season}"
TREAT_A="${TREAT_A:-three-point}"
TREAT_B="${TREAT_B:-two-point}"
TREATMENT_SCHEME="${TREATMENT_SCHEME:-binary}"

PROP_MODEL="${PROP_MODEL:-catboost}"
OUTCOME_MODEL="${OUTCOME_MODEL:-catboost}"
TAU_MODEL="${TAU_MODEL:-catboost}"
USE_FIXED_HPARAMS="${USE_FIXED_HPARAMS:-1}"
FIXED_PROP_PARAMS_JSON="${FIXED_PROP_PARAMS_JSON:-}"
FIXED_OUTCOME_PARAMS_JSON="${FIXED_OUTCOME_PARAMS_JSON:-}"
FIXED_TAU_PARAMS_JSON="${FIXED_TAU_PARAMS_JSON:-}"

RANDOM_STATE="${RANDOM_STATE:-123}"
CLUSTER_COL="${CLUSTER_COL:-GAME_ID}"
MIN_SAMPLES_PER_TREAT="${MIN_SAMPLES_PER_TREAT:-200}"
MIN_PROP="${MIN_PROP:-1e-2}"
MAX_PROP="${MAX_PROP:-1.0}"
ES_ROUNDS="${ES_ROUNDS:-200}"
FINAL_ES_TAIL_SPAN="${FINAL_ES_TAIL_SPAN:-3}"
DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING:-0}"
RUN_CATE_SURFACE="${RUN_CATE_SURFACE:-1}"
CATE_PREFIX="${CATE_PREFIX:-full_data_}"
CATE_BOOTSTRAP="${CATE_BOOTSTRAP:-1000}"
CATE_N_SAMPLE="${CATE_N_SAMPLE:-0}"
CATE_SPLIT_BY_TIME="${CATE_SPLIT_BY_TIME:-1}"
CATE_TIME_LONG_LO="${CATE_TIME_LONG_LO:-15}"
CATE_TIME_LONG_HI="${CATE_TIME_LONG_HI:-300}"
CATE_TIME_SHORT_LO="${CATE_TIME_SHORT_LO:-0}"
CATE_TIME_SHORT_HI="${CATE_TIME_SHORT_HI:-15}"
CATE_N_TIME_LONG="${CATE_N_TIME_LONG:-25}"
CATE_N_TIME_SHORT="${CATE_N_TIME_SHORT:-16}"
APPLY_TAU_CALIBRATION_TO_CATE="${APPLY_TAU_CALIBRATION_TO_CATE:-1}"
TAU_CALIBRATION_AUTO_DECIDE="${TAU_CALIBRATION_AUTO_DECIDE:-1}"
TAU_CALIBRATION_BETA_GAP_THRESHOLD="${TAU_CALIBRATION_BETA_GAP_THRESHOLD:-0.1}"
TAU_CALIBRATION_ALPHA_ABS_THRESHOLD="${TAU_CALIBRATION_ALPHA_ABS_THRESHOLD:-0.001}"
RUN_CALIBRATION_BLP="${RUN_CALIBRATION_BLP:-1}"
CALIB_PREFIX="${CALIB_PREFIX:-full_data_}"
CALIB_N_BUCKETS="${CALIB_N_BUCKETS:-10}"
RUN_APPLY_TAU_CALIBRATION="${RUN_APPLY_TAU_CALIBRATION:-1}"
RUN_PROPENSITY_CONTEXT="${RUN_PROPENSITY_CONTEXT:-1}"
PROPENSITY_PREFIX="${PROPENSITY_PREFIX:-full_data_}"
PROPENSITY_E_COL="${PROPENSITY_E_COL:-e_hat_${TREAT_A}}"
PROPENSITY_EXTREME_THRESHOLD="${PROPENSITY_EXTREME_THRESHOLD:-0.05}"
PROPENSITY_OVERLAP_THRESHOLD="${PROPENSITY_OVERLAP_THRESHOLD:-0.05}"
PROPENSITY_CALIBRATION_BINS="${PROPENSITY_CALIBRATION_BINS:-10}"

THREADS="${THREADS:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --input)
      INPUT="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

AVAILABLE_CPUS="$(nproc 2>/dev/null || echo 1)"
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-}"
if [[ -z "${THREADS}" ]]; then
  if [[ -n "${SLURM_CPUS}" ]]; then
    THREADS="${SLURM_CPUS}"
  else
    THREADS="${AVAILABLE_CPUS}"
  fi
fi
if [[ "${THREADS}" -lt 1 ]]; then
  THREADS="1"
fi

mkdir -p logs
mkdir -p "${OUTDIR}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

echo "================================================================================"
echo "Full-data train"
echo "================================================================================"
echo "input:            ${INPUT}"
echo "outdir:           ${OUTDIR}"
echo "models:           prop=${PROP_MODEL}, outcome=${OUTCOME_MODEL}, tau=${TAU_MODEL}"
echo "use_fixed_hparams:${USE_FIXED_HPARAMS}"
echo "run_cate:         ${RUN_CATE_SURFACE}"
echo "apply_calib_cate: ${APPLY_TAU_CALIBRATION_TO_CATE}"
echo "calib_auto_decide:${TAU_CALIBRATION_AUTO_DECIDE} (|alpha|<=${TAU_CALIBRATION_ALPHA_ABS_THRESHOLD}, beta_gap<=${TAU_CALIBRATION_BETA_GAP_THRESHOLD}, CI covers 1)"
echo "run_calib_blp:    ${RUN_CALIBRATION_BLP}"
echo "run_apply_calib:  ${RUN_APPLY_TAU_CALIBRATION}"
echo "run_propensity:   ${RUN_PROPENSITY_CONTEXT}"
echo "prop_extreme_thr: ${PROPENSITY_EXTREME_THRESHOLD}"
echo "prop_overlap_thr: ${PROPENSITY_OVERLAP_THRESHOLD}"
echo "threads:          ${THREADS}"
echo "extra args:       ${EXTRA_ARGS[*]:-(none)}"

START_TIME=$(date +%s)
LOG_PATH="logs/full_data_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  python3 "${PROJECT_DIR}/scripts/core/train_full_data_selected_model.py"
  --input "${INPUT}"
  --outdir "${OUTDIR}"
  --train-start "${TRAIN_START}"
  --inner-train-init-span "${INNER_TRAIN_INIT_SPAN}"
  --inner-block-span "${INNER_BLOCK_SPAN}"
  --inner-step "${INNER_STEP}"
  --treat-col "${TREAT_COL}"
  --outcome-col "${OUTCOME_COL}"
  --season-col "${SEASON_COL}"
  --treat-a "${TREAT_A}"
  --treat-b "${TREAT_B}"
  --treatment-scheme "${TREATMENT_SCHEME}"
  --prop-model "${PROP_MODEL}"
  --outcome-model "${OUTCOME_MODEL}"
  --tau-model "${TAU_MODEL}"
  --random-state "${RANDOM_STATE}"
  --min-samples-per-treat "${MIN_SAMPLES_PER_TREAT}"
  --min-prop "${MIN_PROP}"
  --max-prop "${MAX_PROP}"
  --es-rounds "${ES_ROUNDS}"
  --final-es-tail-span "${FINAL_ES_TAIL_SPAN}"
)

if [[ -n "${MAX_SEASON}" ]]; then
  CMD+=(--max-season "${MAX_SEASON}")
fi
if [[ "${DISABLE_EARLY_STOPPING}" == "1" ]]; then
  CMD+=(--disable-early-stopping)
fi
if [[ "${USE_FIXED_HPARAMS}" == "1" ]]; then
  CMD+=(--use-fixed-hparams)
  if [[ -n "${FIXED_PROP_PARAMS_JSON}" ]]; then
    CMD+=(--fixed-prop-params-json "${FIXED_PROP_PARAMS_JSON}")
  fi
  if [[ -n "${FIXED_OUTCOME_PARAMS_JSON}" ]]; then
    CMD+=(--fixed-outcome-params-json "${FIXED_OUTCOME_PARAMS_JSON}")
  fi
  if [[ -n "${FIXED_TAU_PARAMS_JSON}" ]]; then
    CMD+=(--fixed-tau-params-json "${FIXED_TAU_PARAMS_JSON}")
  fi
fi

echo "Running: ${CMD[*]} ${EXTRA_ARGS[*]}"
"${CMD[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "Completed successfully."
echo "Elapsed: ${ELAPSED} sec ($((ELAPSED / 60)) min)"
echo "Log: ${LOG_PATH}"

if [[ "${RUN_CALIBRATION_BLP}" == "1" ]]; then
  NUISANCE_PATH="${OUTDIR}/nuisance_oos_train.parquet"
  TAU_PATH="${OUTDIR}/tau_oos_train.parquet"
  if [[ ! -f "${NUISANCE_PATH}" ]]; then
    echo "[error] calibration skipped: nuisance file not found: ${NUISANCE_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${TAU_PATH}" ]]; then
    echo "[error] calibration skipped: tau file not found: ${TAU_PATH}" >&2
    exit 1
  fi

  echo "Running full-data calibration + BLP"
  python3 "${PROJECT_DIR}/scripts/core/calibrate_tau_cate.py" \
    --nuisance "${NUISANCE_PATH}" \
    --oof-preds "${TAU_PATH}" \
    --tau-col tau_hat \
    --treat-col "${TREAT_COL}" \
    --treat-a-label "${TREAT_A}" \
    --treat-b-label "${TREAT_B}" \
    --outcome-col "${OUTCOME_COL}" \
    --cluster-col GAME_ID \
    --n-buckets "${CALIB_N_BUCKETS}" \
    --outdir "${OUTDIR}" \
    --prefix "${CALIB_PREFIX}" \
    --plot
fi

CALIB_JSON_PATH="${OUTDIR}/${CALIB_PREFIX}blp.json"
NEED_TAU_CALIBRATION="1"
if [[ "${TAU_CALIBRATION_AUTO_DECIDE}" == "1" && -f "${CALIB_JSON_PATH}" ]]; then
  NEED_TAU_CALIBRATION="$(python3 - <<PY
import json, math
p=r"""${CALIB_JSON_PATH}"""
thr=float(r"""${TAU_CALIBRATION_BETA_GAP_THRESHOLD}""")
athr=float(r"""${TAU_CALIBRATION_ALPHA_ABS_THRESHOLD}""")
try:
    d=json.load(open(p, "r", encoding="utf-8"))
    alpha=float(d.get("alpha", float("nan")))
    beta=float(d.get("beta", float("nan")))
    ci=d.get("beta_ci", [float("nan"), float("nan")])
    lo=float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) >= 2 else float("nan")
    hi=float(ci[1]) if isinstance(ci, (list, tuple)) and len(ci) >= 2 else float("nan")
    alpha_ok=math.isfinite(alpha) and abs(alpha) <= athr
    beta_gap_ok=math.isfinite(beta) and abs(beta-1.0) <= thr
    ci_covers_one=math.isfinite(lo) and math.isfinite(hi) and lo <= 1.0 <= hi
    print("0" if (alpha_ok and beta_gap_ok and ci_covers_one) else "1")
except Exception:
    print("1")
PY
)"
  if [[ "${NEED_TAU_CALIBRATION}" == "0" ]]; then
    echo "Auto-check: BLP indicates well-calibrated tau; skip calibration application."
  else
    echo "Auto-check: BLP indicates calibration needed; apply calibration."
  fi
fi

if [[ "${RUN_APPLY_TAU_CALIBRATION}" == "1" ]]; then
  NUISANCE_OOS_PATH="${OUTDIR}/nuisance_oos_train.parquet"
  TAU_OOS_PATH="${OUTDIR}/tau_oos_train.parquet"
  TAU_FULL_PATH="${OUTDIR}/tau_full_train.parquet"
  CALIB_JSON_PATH="${OUTDIR}/${CALIB_PREFIX}blp.json"

  for required in "${NUISANCE_OOS_PATH}" "${TAU_OOS_PATH}" "${TAU_FULL_PATH}"; do
    if [[ ! -f "${required}" ]]; then
      echo "[error] tau calibration apply skipped: required file not found: ${required}" >&2
      exit 1
    fi
  done

  APPLY_ARGS=(
    python3 "${PROJECT_DIR}/scripts/core/apply_tau_calibration_and_compare.py"
    --nuisance-oos "${NUISANCE_OOS_PATH}"
    --tau-oos "${TAU_OOS_PATH}"
    --tau-full "${TAU_FULL_PATH}"
    --tau-col tau_hat
    --cluster-col GAME_ID
    --treat-col "${TREAT_COL}"
    --treat-a-label "${TREAT_A}"
    --treat-b-label "${TREAT_B}"
    --outcome-col "${OUTCOME_COL}"
    --min-prop "${MIN_PROP}"
    --max-prop "${MAX_PROP}"
    --n-buckets "${CALIB_N_BUCKETS}"
    --outdir "${OUTDIR}"
    --prefix "${CALIB_PREFIX}"
  )

  if [[ "${NEED_TAU_CALIBRATION}" != "1" ]]; then
    echo "Skipping tau calibration apply/compare (auto-check not needed)."
  elif [[ -f "${CALIB_JSON_PATH}" ]]; then
    APPLY_ARGS+=(--calib-json "${CALIB_JSON_PATH}")
    echo "Running tau calibration apply/compare (using ${CALIB_JSON_PATH})"
    "${APPLY_ARGS[@]}"
  else
    echo "Running tau calibration apply/compare (fit alpha/beta from OOS BLP)"
    "${APPLY_ARGS[@]}"
  fi
fi

if [[ "${RUN_CATE_SURFACE}" == "1" ]]; then
  DATA_PATH="${INPUT}"
  DR_MODEL_PATH="${OUTDIR}/tau_model.joblib"
  CALIB_JSON_PATH="${OUTDIR}/${CALIB_PREFIX}blp.json"
  if [[ ! -f "${DATA_PATH}" ]]; then
    echo "[error] CATE surface skipped: input data not found: ${DATA_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${DR_MODEL_PATH}" ]]; then
    echo "[error] CATE surface skipped: DR model not found: ${DR_MODEL_PATH}" >&2
    exit 1
  fi

  CATE_CALIB_ARGS=()
  if [[ "${APPLY_TAU_CALIBRATION_TO_CATE}" == "1" && "${NEED_TAU_CALIBRATION}" == "1" && -f "${CALIB_JSON_PATH}" ]]; then
    CATE_CALIB_ARGS=(--tau-calib-json "${CALIB_JSON_PATH}")
    echo "CATE surface will apply tau calibration from ${CALIB_JSON_PATH}"
  elif [[ "${APPLY_TAU_CALIBRATION_TO_CATE}" == "1" && "${NEED_TAU_CALIBRATION}" != "1" ]]; then
    echo "CATE surface uses raw tau (auto-check judged calibration unnecessary)."
  fi

  if [[ "${CATE_SPLIT_BY_TIME}" == "1" ]]; then
    echo "Running CATE surface plot (time range: ${CATE_TIME_LONG_LO}-${CATE_TIME_LONG_HI})"
    python3 "${PROJECT_DIR}/scripts/core/plot_cate_surface_gcomp.py" \
      --input "${DATA_PATH}" \
      --dr-model "${DR_MODEL_PATH}" \
      --outdir "${OUTDIR}" \
      --prefix "${CATE_PREFIX}t15_300_" \
      --treat-a "${TREAT_A}" \
      --treat-b "${TREAT_B}" \
      --time-lo "${CATE_TIME_LONG_LO}" \
      --time-hi "${CATE_TIME_LONG_HI}" \
      --score-lo -10 \
      --score-hi 10 \
      --n-time "${CATE_N_TIME_LONG}" \
      --n-sample "${CATE_N_SAMPLE}" \
      --bootstrap "${CATE_BOOTSTRAP}" \
      "${CATE_CALIB_ARGS[@]}" \
      --plot \
      --plot-sig-mask

    echo "Running CATE surface plot (time range: ${CATE_TIME_SHORT_LO}-${CATE_TIME_SHORT_HI})"
    python3 "${PROJECT_DIR}/scripts/core/plot_cate_surface_gcomp.py" \
      --input "${DATA_PATH}" \
      --dr-model "${DR_MODEL_PATH}" \
      --outdir "${OUTDIR}" \
      --prefix "${CATE_PREFIX}t0_15_" \
      --treat-a "${TREAT_A}" \
      --treat-b "${TREAT_B}" \
      --time-lo "${CATE_TIME_SHORT_LO}" \
      --time-hi "${CATE_TIME_SHORT_HI}" \
      --score-lo -10 \
      --score-hi 10 \
      --n-time "${CATE_N_TIME_SHORT}" \
      --n-sample "${CATE_N_SAMPLE}" \
      --bootstrap "${CATE_BOOTSTRAP}" \
      "${CATE_CALIB_ARGS[@]}" \
      --plot \
      --plot-sig-mask
  else
    echo "Running CATE surface plot (single range)"
    python3 "${PROJECT_DIR}/scripts/core/plot_cate_surface_gcomp.py" \
      --input "${DATA_PATH}" \
      --dr-model "${DR_MODEL_PATH}" \
      --outdir "${OUTDIR}" \
      --prefix "${CATE_PREFIX}" \
      --treat-a "${TREAT_A}" \
      --treat-b "${TREAT_B}" \
      --score-lo -10 \
      --score-hi 10 \
      --n-sample "${CATE_N_SAMPLE}" \
      --bootstrap "${CATE_BOOTSTRAP}" \
      "${CATE_CALIB_ARGS[@]}" \
      --plot \
      --plot-sig-mask
  fi
fi

if [[ "${RUN_PROPENSITY_CONTEXT}" == "1" ]]; then
  NUISANCE_PATH="${OUTDIR}/nuisance_full_train.parquet"
  if [[ ! -f "${NUISANCE_PATH}" ]]; then
    echo "[error] propensity context plot skipped: nuisance file not found: ${NUISANCE_PATH}" >&2
    exit 1
  fi

  echo "Running propensity context plot"
  python3 "${PROJECT_DIR}/scripts/core/plot_propensity_context.py" \
    --input "${NUISANCE_PATH}" \
    --e-col "${PROPENSITY_E_COL}" \
    --treat-col "${TREAT_COL}" \
    --treat-a "${TREAT_A}" \
    --treat-b "${TREAT_B}" \
    --min-prop "${MIN_PROP}" \
    --max-prop "${MAX_PROP}" \
    --extreme-threshold "${PROPENSITY_EXTREME_THRESHOLD}" \
    --overlap-threshold "${PROPENSITY_OVERLAP_THRESHOLD}" \
    --calibration-bins "${PROPENSITY_CALIBRATION_BINS}" \
    --time-col time_left_game \
    --score-col score_diff \
    --outdir "${OUTDIR}" \
    --prefix "${PROPENSITY_PREFIX}"
fi
