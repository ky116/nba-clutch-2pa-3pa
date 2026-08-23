#!/bin/bash
set -euo pipefail
THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  for cand in "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${PWD}/../.." "${THIS_SCRIPT_DIR}/../.."; do
    if [[ -n "${cand}" && -f "${cand}/scripts/core/fit_wp_model.r" ]]; then
      PROJECT_DIR="$(cd "${cand}" && pwd -P)"
      break
    fi
  done
fi
if [[ -z "${PROJECT_DIR:-}" || ! -f "${PROJECT_DIR}/scripts/core/fit_wp_model.r" ]]; then
  echo "[error] cannot resolve PROJECT_DIR from PWD=${PWD} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} THIS_SCRIPT_DIR=${THIS_SCRIPT_DIR}" >&2
  exit 1
fi
cd "${PROJECT_DIR}"

# Common settings across 3 model runs
INPUT="${INPUT:-data/analysis/shotchoice_panel_clutch_rs.parquet}"
TRAIN_START="${TRAIN_START:-2000}"
MAX_SEASON="${MAX_SEASON:-}"
INNER_TRAIN_INIT_SPAN="${INNER_TRAIN_INIT_SPAN:-10}"
INNER_BLOCK_SPAN="${INNER_BLOCK_SPAN:-3}"
INNER_STEP="${INNER_STEP:-3}"

TREAT_COL="${TREAT_COL:-shot_zone_choice}"
OUTCOME_COL="${OUTCOME_COL:-delta_wp}"
SEASON_COL="${SEASON_COL:-season}"
TREATMENT_SCHEME="${TREATMENT_SCHEME:-binary}"
TREAT_A="${TREAT_A:-three-point}"
TREAT_B="${TREAT_B:-two-point}"
OOF_SCHEME="${OOF_SCHEME:-season_loso}"

RANDOM_STATE="${RANDOM_STATE:-123}"
MIN_SAMPLES_PER_TREAT="${MIN_SAMPLES_PER_TREAT:-200}"
MIN_PROP="${MIN_PROP:-1e-2}"
MAX_PROP="${MAX_PROP:-1.0}"
ES_ROUNDS="${ES_ROUNDS:-200}"
FINAL_ES_TAIL_SPAN="${FINAL_ES_TAIL_SPAN:-3}"

USE_FIXED_HPARAMS="${USE_FIXED_HPARAMS:-1}"
AUTO_EXPORT_FIXED_HPARAMS="${AUTO_EXPORT_FIXED_HPARAMS:-1}"
AUTO_SUBMIT_ENSEMBLE_OOS="${AUTO_SUBMIT_ENSEMBLE_OOS:-1}"

CPU_THREADS="${CPU_THREADS:-32}"
CPU_MEM="${CPU_MEM:-96G}"
CPU_TIME="${CPU_TIME:-144:00:00}"
CPU_EXCLUDE="${CPU_EXCLUDE:-}"

OUTDIR_CAT="${OUTDIR_CAT:-results/full_data_catboost_state_fixed_loso}"
OUTDIR_XGB="${OUTDIR_XGB:-results/full_data_xgb_state_fixed_loso}"
OUTDIR_LGBM="${OUTDIR_LGBM:-results/full_data_lgbm_state_fixed_loso}"

FIXED_PROP_CAT="${FIXED_PROP_CAT:-results/fixed_hparams_catboost_majority/fixed_prop_params.json}"
FIXED_OUTCOME_CAT="${FIXED_OUTCOME_CAT:-results/fixed_hparams_catboost_majority/fixed_outcome_params.json}"
FIXED_TAU_CAT="${FIXED_TAU_CAT:-results/fixed_hparams_catboost_majority/fixed_tau_params.json}"

FIXED_PROP_XGB="${FIXED_PROP_XGB:-results/fixed_hparams_xgb_majority/fixed_prop_params.json}"
FIXED_OUTCOME_XGB="${FIXED_OUTCOME_XGB:-results/fixed_hparams_xgb_majority/fixed_outcome_params.json}"
FIXED_TAU_XGB="${FIXED_TAU_XGB:-results/fixed_hparams_xgb_majority/fixed_tau_params.json}"

FIXED_PROP_LGBM="${FIXED_PROP_LGBM:-results/fixed_hparams_lgbm_majority/fixed_prop_params.json}"
FIXED_OUTCOME_LGBM="${FIXED_OUTCOME_LGBM:-results/fixed_hparams_lgbm_majority/fixed_outcome_params.json}"
FIXED_TAU_LGBM="${FIXED_TAU_LGBM:-results/fixed_hparams_lgbm_majority/fixed_tau_params.json}"

WF_ROOT_CAT="${WF_ROOT_CAT:-results/nested_wf_catboost_gpu}"
WF_ROOT_XGB="${WF_ROOT_XGB:-results/nested_wf_xgb}"
WF_ROOT_LGBM="${WF_ROOT_LGBM:-results/nested_wf_lgbm}"
ENSEMBLE_OUTDIR="${ENSEMBLE_OUTDIR:-results/full_data_ensemble_state_fixed_loso}"
mkdir -p "${PROJECT_DIR}/logs"
FULL_DATA_SLURM_SCRIPT="${PROJECT_DIR}/scripts/slurm/run_full_data_slurm.sh"
FULL_DATA_ENSEMBLE_SLURM_SCRIPT="${PROJECT_DIR}/scripts/slurm/run_full_data_ensemble_oos_slurm.sh"

submit_gpu_catboost() {
  echo "[1/3] Submit CatBoost (GPU)" >&2
  local out
  out=$(
    INPUT="$INPUT" \
  OUTDIR="$OUTDIR_CAT" \
  TRAIN_START="$TRAIN_START" \
  MAX_SEASON="$MAX_SEASON" \
  INNER_TRAIN_INIT_SPAN="$INNER_TRAIN_INIT_SPAN" \
  INNER_BLOCK_SPAN="$INNER_BLOCK_SPAN" \
  INNER_STEP="$INNER_STEP" \
  TREAT_COL="$TREAT_COL" \
  OUTCOME_COL="$OUTCOME_COL" \
  SEASON_COL="$SEASON_COL" \
  TREATMENT_SCHEME="$TREATMENT_SCHEME" \
  TREAT_A="$TREAT_A" \
  TREAT_B="$TREAT_B" \
  PROP_MODEL="catboost" \
  OUTCOME_MODEL="catboost" \
  TAU_MODEL="catboost" \
  OOF_SCHEME="$OOF_SCHEME" \
  RANDOM_STATE="$RANDOM_STATE" \
  MIN_SAMPLES_PER_TREAT="$MIN_SAMPLES_PER_TREAT" \
  MIN_PROP="$MIN_PROP" \
  MAX_PROP="$MAX_PROP" \
  ES_ROUNDS="$ES_ROUNDS" \
  FINAL_ES_TAIL_SPAN="$FINAL_ES_TAIL_SPAN" \
  USE_FIXED_HPARAMS="$USE_FIXED_HPARAMS" \
  FIXED_PROP_PARAMS_JSON="$FIXED_PROP_CAT" \
  FIXED_OUTCOME_PARAMS_JSON="$FIXED_OUTCOME_CAT" \
  FIXED_TAU_PARAMS_JSON="$FIXED_TAU_CAT" \
  sbatch "${FULL_DATA_SLURM_SCRIPT}" --oof-scheme "$OOF_SCHEME"
  )
  echo "$out" >&2
  printf '%s\n' "$out" | awk '/Submitted batch job/ {print $4}' | tail -n 1
}

submit_cpu_model() {
  local model="$1"
  local outdir="$2"
  local fixed_prop="$3"
  local fixed_outcome="$4"
  local fixed_tau="$5"
  local out
  local sbatch_args=(
    --job-name="nba_full_data_${model}"
    --cpus-per-task="$CPU_THREADS"
    --mem="$CPU_MEM"
    --time="$CPU_TIME"
    --chdir="$PROJECT_DIR"
    -o "$PROJECT_DIR/logs/%j.txt"
  )
  if [[ -n "$CPU_EXCLUDE" ]]; then
    sbatch_args+=(--exclude="$CPU_EXCLUDE")
  fi

  echo "[${6}/3] Submit ${model} (CPU x${CPU_THREADS})" >&2
  out=$(
    INPUT="$INPUT" \
  OUTDIR="$outdir" \
  TRAIN_START="$TRAIN_START" \
  MAX_SEASON="$MAX_SEASON" \
  INNER_TRAIN_INIT_SPAN="$INNER_TRAIN_INIT_SPAN" \
  INNER_BLOCK_SPAN="$INNER_BLOCK_SPAN" \
  INNER_STEP="$INNER_STEP" \
  TREAT_COL="$TREAT_COL" \
  OUTCOME_COL="$OUTCOME_COL" \
  SEASON_COL="$SEASON_COL" \
  TREATMENT_SCHEME="$TREATMENT_SCHEME" \
  TREAT_A="$TREAT_A" \
  TREAT_B="$TREAT_B" \
  PROP_MODEL="$model" \
  OUTCOME_MODEL="$model" \
  TAU_MODEL="$model" \
  OOF_SCHEME="$OOF_SCHEME" \
  RANDOM_STATE="$RANDOM_STATE" \
  MIN_SAMPLES_PER_TREAT="$MIN_SAMPLES_PER_TREAT" \
  MIN_PROP="$MIN_PROP" \
  MAX_PROP="$MAX_PROP" \
  ES_ROUNDS="$ES_ROUNDS" \
  FINAL_ES_TAIL_SPAN="$FINAL_ES_TAIL_SPAN" \
  USE_FIXED_HPARAMS="$USE_FIXED_HPARAMS" \
  FIXED_PROP_PARAMS_JSON="$fixed_prop" \
  FIXED_OUTCOME_PARAMS_JSON="$fixed_outcome" \
  FIXED_TAU_PARAMS_JSON="$fixed_tau" \
  THREADS="$CPU_THREADS" \
  CATBOOST_USE_GPU=0 \
  sbatch "${sbatch_args[@]}" \
    --wrap "cd '$PROJECT_DIR' && bash '$PROJECT_DIR/scripts/helpers/run_full_data.sh' --threads '$CPU_THREADS' --oof-scheme '$OOF_SCHEME'"
  )
  echo "$out" >&2
  printf '%s\n' "$out" | awk '/Submitted batch job/ {print $4}' | tail -n 1
}

submit_ensemble_oos() {
  local dep="$1"
  local out
  echo "[4/4] Submit ensemble OOS (afterok:${dep})" >&2
  out=$(
    PROJECT_DIR="$PROJECT_DIR" \
    CAT_DIR="$OUTDIR_CAT" \
    XGB_DIR="$OUTDIR_XGB" \
    LGBM_DIR="$OUTDIR_LGBM" \
    OUTDIR="$ENSEMBLE_OUTDIR" \
    sbatch --dependency="afterok:${dep}" "${FULL_DATA_ENSEMBLE_SLURM_SCRIPT}"
  )
  echo "$out" >&2
  printf '%s\n' "$out" | awk '/Submitted batch job/ {print $4}' | tail -n 1
}

if [[ "$USE_FIXED_HPARAMS" == "1" ]]; then
  if [[ "${AUTO_EXPORT_FIXED_HPARAMS}" == "1" ]]; then
    echo "[prep] Export fixed hyperparameters from WF outputs"
    python3 "${PROJECT_DIR}/scripts/core/export_fixed_hparams_majority_from_wf.py" \
      --wf-root "$WF_ROOT_CAT" \
      --model catboost \
      --write-json-dir "$(dirname "$FIXED_PROP_CAT")"
    python3 "${PROJECT_DIR}/scripts/core/export_fixed_hparams_majority_from_wf.py" \
      --wf-root "$WF_ROOT_XGB" \
      --model xgb \
      --write-json-dir "$(dirname "$FIXED_PROP_XGB")"
    python3 "${PROJECT_DIR}/scripts/core/export_fixed_hparams_majority_from_wf.py" \
      --wf-root "$WF_ROOT_LGBM" \
      --model lgbm \
      --write-json-dir "$(dirname "$FIXED_PROP_LGBM")"
  fi

  for p in "$FIXED_PROP_CAT" "$FIXED_OUTCOME_CAT" "$FIXED_TAU_CAT" \
           "$FIXED_PROP_XGB" "$FIXED_OUTCOME_XGB" "$FIXED_TAU_XGB" \
           "$FIXED_PROP_LGBM" "$FIXED_OUTCOME_LGBM" "$FIXED_TAU_LGBM"; do
    if [[ ! -f "$p" ]]; then
      echo "[error] fixed params not found: $p" >&2
      exit 1
    fi
  done
fi

JOB_CAT="$(submit_gpu_catboost)"
JOB_XGB="$(submit_cpu_model "xgb" "$OUTDIR_XGB" "$FIXED_PROP_XGB" "$FIXED_OUTCOME_XGB" "$FIXED_TAU_XGB" "2")"
JOB_LGBM="$(submit_cpu_model "lgbm" "$OUTDIR_LGBM" "$FIXED_PROP_LGBM" "$FIXED_OUTCOME_LGBM" "$FIXED_TAU_LGBM" "3")"

JOB_DEP_LIST="$(printf '%s:%s:%s' "$JOB_CAT" "$JOB_XGB" "$JOB_LGBM")"
JOB_ENSEMBLE=""
if [[ "$AUTO_SUBMIT_ENSEMBLE_OOS" == "1" ]]; then
  JOB_ENSEMBLE="$(submit_ensemble_oos "$JOB_DEP_LIST")"
fi

cat <<EOF

Submitted 3 full-data jobs:
  catboost(GPU): ${OUTDIR_CAT}
  xgb(CPU):      ${OUTDIR_XGB}
  lgbm(CPU):     ${OUTDIR_LGBM}

Common settings:
  oof_scheme=${OOF_SCHEME}
  use_fixed_hparams=${USE_FIXED_HPARAMS}
  auto_export_fixed_hparams=${AUTO_EXPORT_FIXED_HPARAMS}
  auto_submit_ensemble_oos=${AUTO_SUBMIT_ENSEMBLE_OOS}

Job IDs:
  catboost=${JOB_CAT}
  xgb=${JOB_XGB}
  lgbm=${JOB_LGBM}

EOF

if [[ -n "${JOB_ENSEMBLE}" ]]; then
cat <<EOF
Ensemble OOS job:
  ensemble=${JOB_ENSEMBLE}
  outdir=${ENSEMBLE_OUTDIR}

EOF
fi
