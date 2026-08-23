#!/bin/bash
#SBATCH --job-name=rs_m0_k0_panel
#SBATCH --output=logs/%j.txt
#SBATCH --error=logs/%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

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
mkdir -p logs

export THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
# Strict season-OOF refits are memory/thread heavy. Keep the Slurm default
# sequential to avoid forked mgcv::bam workers stalling; override explicitly
# after testing on the target node.
export SEASON_OOF_JOBS="${SEASON_OOF_JOBS:-1}"
export BAM_NTHREADS="${BAM_NTHREADS:-8}"
# Slurm already writes stdout/stderr to logs/%j.txt; do not tee back into the same file.
export LOG_TO_FILE="0"

exec bash "${PROJECT_DIR}/scripts/pipelines/run_rs_m0_k0_shot_panel.sh"
