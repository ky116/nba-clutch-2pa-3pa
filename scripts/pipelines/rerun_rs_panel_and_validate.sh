#!/usr/bin/env bash
set -euo pipefail

# Rebuild the RS m0/k0 WP-scored shot states and shotchoice panel, then run
# targeted checks for the terminal-shot delta_wp fix.
#
# Server use:
#   sbatch scripts/slurm/run_rs_m0_k0_shot_panel_slurm.sh
#   python3 scripts/helpers/validate_wp_scored_shots.py
#
# Local/direct use:
#   bash scripts/pipelines/rerun_rs_panel_and_validate.sh

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
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_DIR}"

echo "[rerun] step 1/2: rebuild RS m0/k0 shot panel"
OOF_TEMPLATE_PROTOCOL_M0="${OOF_TEMPLATE_PROTOCOL_M0:-1}" \
  bash scripts/pipelines/run_rs_m0_k0_shot_panel.sh

echo "[rerun] step 2/2: validate scored shots and panel"
"${PYTHON_BIN}" scripts/helpers/validate_wp_scored_shots.py \
  --with-wp "data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz" \
  --panel "data/analysis/shotchoice_panel_clutch_rs.parquet"
