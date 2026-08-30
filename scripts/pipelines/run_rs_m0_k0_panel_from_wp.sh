#!/usr/bin/env bash
set -euo pipefail

# Build shot-choice panels from an existing WP-scored shot-state file.
# This script does not fit or score any WP model.

START_SEASON="${START_SEASON:-2000}"
END_SEASON="${END_SEASON:-2024}"
SEASONTYPE="${SEASONTYPE:-rs}"
WP_WITH_PATH="${WP_WITH_PATH:-data/wp/shot_decision_states_${START_SEASON}_${END_SEASON}_${SEASONTYPE}_with_wp.csv.gz}"
PANEL_ELO_K="${PANEL_ELO_K:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-data/analysis}"
TEAM_STATS="${TEAM_STATS:-data/analysis/team_shot_stats_2000_2024.parquet}"
TEAM_FOULS="${TEAM_FOULS:-data/analysis/cumulative_team_fouls_2000_2024_rs.parquet}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ ! -f "${WP_WITH_PATH}" ]]; then
  echo "[error] WP-scored shot file not found: ${WP_WITH_PATH}" >&2
  exit 1
fi

echo "[panel] with_wp=${WP_WITH_PATH}"
echo "[panel] output_dir=${OUTPUT_DIR}"
echo "[panel] panel_elo_k=${PANEL_ELO_K}"

echo "[panel] quick delta_wp sanity checks"
"${PYTHON_BIN}" - <<'PY' "${WP_WITH_PATH}"
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
usecols = ["wp_before", "wp_next", "delta_wp", "shot_made"]
chunks = pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=500000)

n = n_nan = n_wp_before_oob = n_wp_next_oob = n_delta_oob = 0
sum_delta = sum_abs_delta = sum_delta_made = sum_delta_miss = 0.0
n_made = n_miss = 0
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
q = np.quantile(dw_all, [0.01, 0.05, 0.5, 0.95, 0.99]) if len(dw_all) else [np.nan] * 5
mean_delta = sum_delta / max(len(dw_all), 1)
mean_abs_delta = sum_abs_delta / max(len(dw_all), 1)
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
  --elo-k "${PANEL_ELO_K}" \
  --team-stats "${TEAM_STATS}" \
  --team-fouls "${TEAM_FOULS}" \
  --output-dir "${OUTPUT_DIR}"

echo "[done] panel outputs (parquet): ${OUTPUT_DIR}/shotchoice_panel_*_${SEASONTYPE}.parquet"
