#!/bin/bash
#SBATCH --job-name=rs_m0_k0_panel
#SBATCH --output=logs/%j.txt
#SBATCH --error=logs/%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  for cand in "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${PWD}/../.." "${THIS_SCRIPT_DIR}/../.."; do
    if [[ -n "${cand}" && -f "${cand}/scripts/pipelines/run_rs_m0_k0_panel_from_wp.sh" ]]; then
      PROJECT_DIR="$(cd "${cand}" && pwd)"
      break
    fi
  done
fi
if [[ -z "${PROJECT_DIR:-}" || ! -f "${PROJECT_DIR}/scripts/pipelines/run_rs_m0_k0_panel_from_wp.sh" ]]; then
  echo "[error] cannot resolve PROJECT_DIR from PWD=${PWD} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} THIS_SCRIPT_DIR=${THIS_SCRIPT_DIR}" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

export THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

exec bash "${PROJECT_DIR}/scripts/pipelines/run_rs_m0_k0_panel_from_wp.sh"
