#!/bin/bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${THIS_SCRIPT_DIR}/../.." && pwd -P)}"
cd "${PROJECT_DIR}"

ROOT="${ROOT:-results/wp_calibration/model_dependence_refit}"
ENSEMBLE_DIR="${ENSEMBLE_DIR:-${ROOT}/ensemble}"
SENSITIVITY_DIR="${SENSITIVITY_DIR:-results/wp_calibration/model_dependence_sensitivity}"
MAIN_SURFACE="${MAIN_SURFACE:-results/full_data_ensemble_state_fixed_loso/full_data_t30_300_cate_surface_equal_weight.csv}"
SUPPORT_SURFACE="${SUPPORT_SURFACE:-results/full_data_ensemble_state_fixed_loso/cate_surface_support/full_data_t30_300_cate_surface_cell_counts.csv}"

python3 scripts/core/average_cate_surfaces.py \
  --surface "${ROOT}/catboost/wp_residual_t30_300_tau_surface_three-point_vs_two-point.parquet" \
  --surface "${ROOT}/xgb/wp_residual_t30_300_tau_surface_three-point_vs_two-point.parquet" \
  --surface "${ROOT}/lgbm/wp_residual_t30_300_tau_surface_three-point_vs_two-point.parquet" \
  --label catboost \
  --label xgb \
  --label lgbm \
  --outdir "${ENSEMBLE_DIR}" \
  --prefix wp_residual_t30_300_ \
  --score-lo -10 \
  --score-hi 10

python3 scripts/helpers/summarize_wp_model_dependence_sensitivity.py \
  --cate-surface "${MAIN_SURFACE}" \
  --bias-surface "${ENSEMBLE_DIR}/wp_residual_t30_300_cate_surface_equal_weight.csv" \
  --bias-col tau_mean_ensemble \
  --support "${SUPPORT_SURFACE}" \
  --outdir "${SENSITIVITY_DIR}" \
  --score-lo -10 \
  --score-hi 10 \
  --time-lo 30 \
  --time-hi 300

python3 scripts/helpers/validate_wp_model_dependence_refit.py \
  --main-surface "${MAIN_SURFACE}" \
  --bias-surface "${ENSEMBLE_DIR}/wp_residual_t30_300_cate_surface_equal_weight.csv" \
  --sensitivity-surface "${SENSITIVITY_DIR}/wp_model_dependence_sensitivity_surface.csv" \
  --output "${ROOT}/validation.json"
