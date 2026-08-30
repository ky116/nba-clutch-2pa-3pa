#!/bin/bash
#SBATCH --job-name=rs_m0_k0_wp_score
#SBATCH --output=logs/%j.txt
#SBATCH --error=logs/%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --exclude=chiron,nevera,nsx

set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  for cand in "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${PWD}/../.." "${THIS_SCRIPT_DIR}/../.."; do
    if [[ -n "${cand}" && -f "${cand}/scripts/pipelines/run_rs_m0_k0_wp_scoring.sh" ]]; then
      PROJECT_DIR="$(cd "${cand}" && pwd)"
      break
    fi
  done
fi
if [[ -z "${PROJECT_DIR:-}" || ! -f "${PROJECT_DIR}/scripts/pipelines/run_rs_m0_k0_wp_scoring.sh" ]]; then
  echo "[error] cannot resolve PROJECT_DIR from PWD=${PWD} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} THIS_SCRIPT_DIR=${THIS_SCRIPT_DIR}" >&2
  exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

export THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export SEASON_OOF_JOBS="${SEASON_OOF_JOBS:-1}"
export BAM_NTHREADS="${BAM_NTHREADS:-8}"
export OOF_TEMPLATE_PROTOCOL_M0="${OOF_TEMPLATE_PROTOCOL_M0:-1}"
export RSCRIPT_BIN="${RSCRIPT_BIN:-/usr/bin/Rscript}"
if [[ -d "${PROJECT_DIR}/.rlib/4.5" ]]; then
  export R_LIBS="${PROJECT_DIR}/.rlib/4.5"
  export R_LIBS_USER="${PROJECT_DIR}/.rlib/4.5"
fi

exec bash "${PROJECT_DIR}/scripts/pipelines/run_rs_m0_k0_wp_scoring.sh"
