#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper. Prefer the separated scripts:
#   bash scripts/pipelines/run_rs_m0_k0_wp_scoring.sh
#   bash scripts/pipelines/run_rs_m0_k0_panel_from_wp.sh

REUSE_WP_SCORING="${REUSE_WP_SCORING:-0}"

if [[ "${REUSE_WP_SCORING}" != "1" ]]; then
  bash scripts/pipelines/run_rs_m0_k0_wp_scoring.sh
else
  echo "[reuse] skipping WP scoring; building panel from existing WP-scored file"
fi

bash scripts/pipelines/run_rs_m0_k0_panel_from_wp.sh
