#!/usr/bin/env bash
set -euo pipefail

# Build RS shot panel from shot_decision_states using the adopted WP model.
# For downstream DML panel, this flow does not use Elo (elo_k=0).
#
# Default flow:
#  1) (optional) train model on data/wp/wp_states_2000_2024_rs.csv.gz
#  2) infer wp_before/wp_next/delta_wp for shot_decision_states
#  3) build shotchoice panels (clutch/strict/extended) and DML csv
#
# Usage examples:
#   bash scripts/pipelines/run_rs_m0_k0_shot_panel.sh
#   TRAIN_MODEL=1 bash scripts/pipelines/run_rs_m0_k0_shot_panel.sh
#   MODEL_IN=models/wp_gam_m0_elo_k0_2000_2024_rs.rds bash scripts/pipelines/run_rs_m0_k0_shot_panel.sh

START_SEASON="${START_SEASON:-2000}"
END_SEASON="${END_SEASON:-2024}"
SEASONTYPE="${SEASONTYPE:-rs}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
LOG_DIR="${LOG_DIR:-logs}"

if [[ "${LOG_TO_FILE}" == "1" && -z "${RUN_RS_LOG_REDIRECTED:-}" ]]; then
  mkdir -p "${LOG_DIR}"
  job_id="${SLURM_JOB_ID:-local}"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LOG_PATH="${LOG_PATH:-${LOG_DIR}/${job_id}.txt}"
  else
    ts="$(date +%Y%m%d_%H%M%S)"
    LOG_PATH="${LOG_PATH:-${LOG_DIR}/run_rs_m0_k0_shot_panel_${job_id}_${ts}.log}"
  fi
  export RUN_RS_LOG_REDIRECTED=1
  exec > >(tee -a "${LOG_PATH}") 2>&1
  echo "[run] log_path=${LOG_PATH}"
fi

TRAIN_PATH="${TRAIN_PATH:-data/wp/wp_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.csv.gz}"
SHOT_PATH="${SHOT_PATH:-data/wp/shot_decision_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.csv.gz}"

ADOPTED_MODEL_IN="${ADOPTED_MODEL_IN:-models/wp_gam_m0_elo_k0_2000_2024_rs.rds}"
MODEL_IN="${MODEL_IN:-${ADOPTED_MODEL_IN}}"
MODEL_OUT="${MODEL_OUT:-models/wp_gam_m0_elo_k0_${START_SEASON}_${END_SEASON}_${SEASONTYPE}.rds}"
TRAIN_MODEL="${TRAIN_MODEL:-0}"  # 1: retrain with scripts/core/fit_wp_model.r
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}"  # 1: auto-train when MODEL_IN does not exist
NO_ELO="${NO_ELO:-1}"
ALLOW_NON_ADOPTED_MODEL="${ALLOW_NON_ADOPTED_MODEL:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-data/analysis}"
TEAM_STATS="${TEAM_STATS:-data/analysis/team_shot_stats_2000_2024.parquet}"
TEAM_FOULS="${TEAM_FOULS:-data/analysis/cumulative_team_fouls_2000_2024_rs.parquet}"

RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
SEASON_OOF_JOBS="${SEASON_OOF_JOBS:-1}"
BAM_NTHREADS="${BAM_NTHREADS:-1}"
if [[ -z "${R_LIBS_USER:-}" ]]; then
  cand="$(find "${HOME}/R" -maxdepth 3 -type d -name "*-linux-gnu-library" 2>/dev/null | sort | tail -n1 || true)"
  if [[ -n "${cand}" ]]; then
    ver="$(find "${cand}" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n1 || true)"
    if [[ -n "${ver}" ]]; then
      export R_LIBS_USER="${ver}"
    fi
  fi
fi
if [[ "${THREADS}" =~ ^[0-9]+$ ]] && [[ "${THREADS}" -ge 1 ]]; then
  export OMP_NUM_THREADS="${THREADS}"
  export OPENBLAS_NUM_THREADS="${THREADS}"
  export MKL_NUM_THREADS="${THREADS}"
  export NUMEXPR_NUM_THREADS="${THREADS}"
fi
echo "[run] threads=${THREADS} (OMP/OPENBLAS/MKL/NUMEXPR)"
echo "[run] R_LIBS_USER=${R_LIBS_USER:-<unset>}"
echo "[run] no_elo=${NO_ELO} (elo_k fixed to 0)"
echo "[run] season_oof_jobs=${SEASON_OOF_JOBS} bam_nthreads=${BAM_NTHREADS}"

if [[ "${NO_ELO}" != "1" ]]; then
  echo "[error] This script is no-Elo flow only. Set NO_ELO=1." >&2
  exit 1
fi

if [[ "${TRAIN_MODEL}" == "1" ]]; then
  echo "[run] train M0-style model on RS states (train path: ${TRAIN_PATH})"
  if [[ "${TRAIN_PATH}" == *"_elo_k20_"* ]] || [[ "${TRAIN_PATH}" == *"_elo_k40_"* ]]; then
    echo "[error] TRAIN_PATH must be elo_k0/base states for this M0 k0 flow: ${TRAIN_PATH}" >&2
    exit 1
  fi
  "${RSCRIPT_BIN}" scripts/core/fit_wp_model.r \
    --start-season "${START_SEASON}" \
    --end-season "${END_SEASON}" \
    --seasontype "${SEASONTYPE}" \
    --train-path "${TRAIN_PATH}" \
    --model-out "${MODEL_OUT}" \
    --no-era-smooth-interaction \
    --protocol-m0-spec
  MODEL_IN="${MODEL_OUT}"
fi

if [[ "${ALLOW_NON_ADOPTED_MODEL}" != "1" ]] && [[ "${MODEL_IN}" != "${ADOPTED_MODEL_IN}" ]] && [[ "${TRAIN_MODEL}" != "1" ]]; then
  echo "[error] MODEL_IN is not the adopted model: ${MODEL_IN}" >&2
  echo "[hint] use MODEL_IN=${ADOPTED_MODEL_IN} (or set ALLOW_NON_ADOPTED_MODEL=1 intentionally)" >&2
  exit 1
fi

if [[ ! -f "${MODEL_IN}" ]]; then
  if [[ "${TRAIN_IF_MISSING}" == "1" ]]; then
    echo "[warn] model not found, auto-training: ${MODEL_IN}"
    "${RSCRIPT_BIN}" scripts/core/fit_wp_model.r \
      --start-season "${START_SEASON}" \
      --end-season "${END_SEASON}" \
      --seasontype "${SEASONTYPE}" \
      --train-path "${TRAIN_PATH}" \
      --model-out "${MODEL_OUT}" \
      --no-era-smooth-interaction \
      --protocol-m0-spec
    MODEL_IN="${MODEL_OUT}"
  else
    echo "[error] model not found: ${MODEL_IN}" >&2
    echo "[hint] set TRAIN_MODEL=1, TRAIN_IF_MISSING=1, or MODEL_IN=<model.rds>" >&2
    exit 1
  fi
fi
if [[ ! -f "${SHOT_PATH}" ]]; then
  echo "[error] shot state not found: ${SHOT_PATH}" >&2
  exit 1
fi

echo "[run] model_in=${MODEL_IN}"
echo "[run] shot_path=${SHOT_PATH}"

"${RSCRIPT_BIN}" scripts/core/fit_wp_and_score_shots.r \
  --start-season "${START_SEASON}" \
  --end-season "${END_SEASON}" \
  --seasontype "${SEASONTYPE}" \
  --model-in "${MODEL_IN}" \
  --shot-path "${SHOT_PATH}" \
  --season-oof-jobs "${SEASON_OOF_JOBS}" \
  --bam-nthreads "${BAM_NTHREADS}"

WP_WITH_PATH="data/wp/shot_decision_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_with_wp.csv.gz"
if [[ ! -f "${WP_WITH_PATH}" ]]; then
  echo "[error] expected scored shot file not found: ${WP_WITH_PATH}" >&2
  exit 1
fi

echo "[check] quick delta_wp sanity checks"
"${PYTHON_BIN}" - <<'PY' "${WP_WITH_PATH}"
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
usecols = ["wp_before", "wp_next", "delta_wp", "shot_made"]
chunks = pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=500000)

n = 0
n_nan = 0
n_wp_before_oob = 0
n_wp_next_oob = 0
n_delta_oob = 0
sum_delta = 0.0
sum_abs_delta = 0.0
sum_delta_made = 0.0
sum_delta_miss = 0.0
n_made = 0
n_miss = 0
vals = []

for ch in chunks:
    n += len(ch)
    wb = pd.to_numeric(ch["wp_before"], errors="coerce")
    wn = pd.to_numeric(ch["wp_next"], errors="coerce")
    dw = pd.to_numeric(ch["delta_wp"], errors="coerce")
    sm = pd.to_numeric(ch["shot_made"], errors="coerce")

    n_nan += int(dw.isna().sum())
    n_wp_before_oob += int(((wb < -1e-9) | (wb > 1 + 1e-9)).sum())
    n_wp_next_oob += int(((wn < -1e-9) | (wn > 1 + 1e-9)).sum())
    n_delta_oob += int(((dw < -1 - 1e-9) | (dw > 1 + 1e-9)).sum())

    dw_valid = dw.dropna()
    if not dw_valid.empty:
        sum_delta += float(dw_valid.sum())
        sum_abs_delta += float(dw_valid.abs().sum())
        vals.append(dw_valid.to_numpy())

    made_mask = (sm == 1) & dw.notna()
    miss_mask = (sm == 0) & dw.notna()
    n_made += int(made_mask.sum())
    n_miss += int(miss_mask.sum())
    sum_delta_made += float(dw[made_mask].sum())
    sum_delta_miss += float(dw[miss_mask].sum())

if n == 0:
    raise SystemExit("[error] sanity check failed: no rows found")

dw_all = np.concatenate(vals) if vals else np.array([], dtype=float)
mean_delta = sum_delta / max(len(dw_all), 1)
mean_abs_delta = sum_abs_delta / max(len(dw_all), 1)
q = np.quantile(dw_all, [0.01, 0.05, 0.5, 0.95, 0.99]) if len(dw_all) else [np.nan] * 5
mean_made = (sum_delta_made / n_made) if n_made > 0 else np.nan
mean_miss = (sum_delta_miss / n_miss) if n_miss > 0 else np.nan

print(f"[check] rows={n:,} delta_nan={n_nan:,}")
print(f"[check] wp_before_oob={n_wp_before_oob:,} wp_next_oob={n_wp_next_oob:,} delta_oob={n_delta_oob:,}")
print(f"[check] delta mean={mean_delta:.6f} mean_abs={mean_abs_delta:.6f}")
print(f"[check] delta q01/q05/q50/q95/q99 = {q[0]:.6f} / {q[1]:.6f} / {q[2]:.6f} / {q[3]:.6f} / {q[4]:.6f}")
print(f"[check] delta by shot_made: made={mean_made:.6f} miss={mean_miss:.6f}")

if n_wp_before_oob > 0 or n_wp_next_oob > 0 or n_delta_oob > 0:
    raise SystemExit("[error] sanity check failed: wp/delta_wp out-of-bounds values found")
if np.isfinite(mean_made) and np.isfinite(mean_miss) and mean_made <= mean_miss:
    print("[warn] unexpected pattern: mean(delta_wp | made) <= mean(delta_wp | miss)")
PY

"${PYTHON_BIN}" scripts/core/build_shotchoice_panel_from_wp.py \
  --input "${WP_WITH_PATH}" \
  --seasontype "${SEASONTYPE}" \
  --elo-k 0 \
  --team-stats "${TEAM_STATS}" \
  --team-fouls "${TEAM_FOULS}" \
  --output-dir "${OUTPUT_DIR}"

echo "[done] scored shot states: ${WP_WITH_PATH}"
echo "[done] panel outputs (parquet): ${OUTPUT_DIR}/shotchoice_panel_*_${SEASONTYPE}.parquet"
