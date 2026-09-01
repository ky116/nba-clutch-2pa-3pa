#!/bin/bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${THIS_SCRIPT_DIR}/../.." && pwd -P)}"
cd "${PROJECT_DIR}"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"
ROOT="${ROOT:-results/wp_calibration/model_dependence_refit}"
PANEL="${PANEL:-data/analysis/shotchoice_panel_clutch_rs.parquet}"
WITH_WP="${WITH_WP:-data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz}"
RESIDUAL_PANEL="${RESIDUAL_PANEL:-${ROOT}/wp_residual_analysis_panel.parquet}"
CPU_THREADS="${CPU_THREADS:-32}"
CPU_MEM="${CPU_MEM:-96G}"
CPU_TIME="${CPU_TIME:-144:00:00}"

for path in \
  "${PANEL}" "${WITH_WP}" \
  results/fixed_hparams_catboost_majority/fixed_prop_params.json \
  results/fixed_hparams_catboost_majority/fixed_outcome_params.json \
  results/fixed_hparams_catboost_majority/fixed_tau_params.json \
  results/fixed_hparams_xgb_majority/fixed_prop_params.json \
  results/fixed_hparams_xgb_majority/fixed_outcome_params.json \
  results/fixed_hparams_xgb_majority/fixed_tau_params.json \
  results/fixed_hparams_lgbm_majority/fixed_prop_params.json \
  results/fixed_hparams_lgbm_majority/fixed_outcome_params.json \
  results/fixed_hparams_lgbm_majority/fixed_tau_params.json; do
  if [[ ! -f "${path}" ]]; then
    echo "[error] required input not found: ${path}" >&2
    exit 1
  fi
done

submit_and_get_id() {
  local output
  output="$("$@")"
  echo "${output}" >&2
  printf '%s\n' "${output}" | awk '/Submitted batch job/ {print $4}' | tail -n 1
}

PREP_JOB="$(submit_and_get_id sbatch \
  --job-name=nba_wp_residual_prep \
  --cpus-per-task=4 \
  --mem=32G \
  --time=12:00:00 \
  --chdir="${PROJECT_DIR}" \
  -o "${PROJECT_DIR}/logs/%j.txt" \
  --wrap "cd '${PROJECT_DIR}' && '${PYTHON_BIN}' scripts/helpers/prepare_wp_residual_analysis_panel.py --panel '${PANEL}' --with-wp '${WITH_WP}' --output '${RESIDUAL_PANEL}'")"

submit_learner() {
  local model="$1"
  local threads="$2"
  local mem="$3"
  local fixed_root="results/fixed_hparams_${model}_majority"
  local model_job_name="nba_wp_residual_${model}"
  local sbatch_args=(
    --dependency="afterok:${PREP_JOB}"
    --job-name="${model_job_name}"
    --cpus-per-task="${threads}"
    --mem="${mem}"
    --time="${CPU_TIME}"
    --chdir="${PROJECT_DIR}"
    -o "${PROJECT_DIR}/logs/%j.txt"
  )
  local gpu_flag=0
  if [[ "${model}" == "catboost" ]]; then
    sbatch_args+=(--gres=gpu:1)
    gpu_flag=1
  fi
  submit_and_get_id sbatch "${sbatch_args[@]}" \
    --wrap "cd '${PROJECT_DIR}' && MODEL='${model}' INPUT='${RESIDUAL_PANEL}' OUTDIR='${ROOT}/${model}' THREADS='${threads}' CATBOOST_USE_GPU='${gpu_flag}' XGB_NUM_THREADS='${threads}' LGBM_NUM_THREADS='${threads}' FIXED_PROP_PARAMS_JSON='${fixed_root}/fixed_prop_params.json' FIXED_OUTCOME_PARAMS_JSON='${fixed_root}/fixed_outcome_params.json' FIXED_TAU_PARAMS_JSON='${fixed_root}/fixed_tau_params.json' bash scripts/helpers/run_wp_residual_cate_learner.sh"
}

CAT_JOB="$(submit_learner catboost 8 48G)"
XGB_JOB="$(submit_learner xgb "${CPU_THREADS}" "${CPU_MEM}")"
LGBM_JOB="$(submit_learner lgbm "${CPU_THREADS}" "${CPU_MEM}")"

ASSEMBLY_JOB="$(submit_and_get_id sbatch \
  --dependency="afterok:${CAT_JOB}:${XGB_JOB}:${LGBM_JOB}" \
  --job-name=nba_wp_sensitivity_assemble \
  --cpus-per-task=4 \
  --mem=32G \
  --time=12:00:00 \
  --chdir="${PROJECT_DIR}" \
  -o "${PROJECT_DIR}/logs/%j.txt" \
  --wrap "cd '${PROJECT_DIR}' && ROOT='${ROOT}' bash scripts/helpers/assemble_wp_model_dependence_refit.sh")"

cat <<EOF
Submitted WP model-dependence refit:
  prepare=${PREP_JOB}
  catboost=${CAT_JOB}
  xgb=${XGB_JOB}
  lgbm=${LGBM_JOB}
  assembly=${ASSEMBLY_JOB}

Specification:
  outcome=wp_next_offense-final_win_offense
  oof_scheme=season_loso
  learners=catboost,xgb,lgbm
  hyperparameters=fixed main pooled values
  recalibration=learner-specific OOF BLP
  ensemble=equal weight
  grid=time 30..300 (25 points) x score -10..10 (21 points) = 525 cells
  interpolation=none
EOF
