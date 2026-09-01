#!/bin/bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${THIS_SCRIPT_DIR}/../.." && pwd -P)}"
cd "${PROJECT_DIR}"

MODEL="${MODEL:?MODEL must be catboost, xgb, or lgbm}"
INPUT="${INPUT:-results/wp_calibration/model_dependence_refit/wp_residual_analysis_panel.parquet}"
OUTDIR="${OUTDIR:-results/wp_calibration/model_dependence_refit/${MODEL}}"
FIXED_PROP_PARAMS_JSON="${FIXED_PROP_PARAMS_JSON:?fixed propensity parameters are required}"
FIXED_OUTCOME_PARAMS_JSON="${FIXED_OUTCOME_PARAMS_JSON:?fixed outcome parameters are required}"
FIXED_TAU_PARAMS_JSON="${FIXED_TAU_PARAMS_JSON:?fixed tau parameters are required}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

if [[ ! " catboost xgb lgbm " =~ " ${MODEL} " ]]; then
  echo "[error] unsupported MODEL=${MODEL}" >&2
  exit 2
fi

export INPUT OUTDIR THREADS
export TREAT_COL=shot_zone_choice OUTCOME_COL=wp_residual_offense SEASON_COL=season
export TREATMENT_SCHEME=binary TREAT_A=three-point TREAT_B=two-point
export PROP_MODEL="${MODEL}" OUTCOME_MODEL="${MODEL}" TAU_MODEL="${MODEL}"
export OOF_SCHEME=season_loso USE_FIXED_HPARAMS=1
export FIXED_PROP_PARAMS_JSON FIXED_OUTCOME_PARAMS_JSON FIXED_TAU_PARAMS_JSON
export RUN_CATE_SURFACE=0 RUN_PROPENSITY_CONTEXT=0
export RUN_CALIBRATION_BLP=1 RUN_APPLY_TAU_CALIBRATION=1
export CALIB_PREFIX=wp_residual_ TAU_CALIBRATION_AUTO_DECIDE=0

bash scripts/helpers/run_full_data.sh --threads "${THREADS}" --oof-scheme season_loso

CALIB_JSON="${OUTDIR}/wp_residual_blp.json"
test -f "${CALIB_JSON}"
python3 scripts/core/plot_cate_surface_gcomp.py \
  --input "${INPUT}" \
  --dr-model "${OUTDIR}/tau_model.joblib" \
  --outdir "${OUTDIR}" \
  --prefix wp_residual_t30_300_ \
  --treat-a three-point \
  --treat-b two-point \
  --time-lo 30 \
  --time-hi 300 \
  --score-lo -10 \
  --score-hi 10 \
  --n-time 25 \
  --n-score 21 \
  --n-sample 0 \
  --bootstrap 0 \
  --tau-calib-json "${CALIB_JSON}"
