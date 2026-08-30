#!/bin/bash
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
cd "$PROJECT_DIR"

CAT_DIR="${CAT_DIR:-results/full_data_catboost_state_fixed_loso}"
XGB_DIR="${XGB_DIR:-results/full_data_xgb_state_fixed_loso}"
LGBM_DIR="${LGBM_DIR:-results/full_data_lgbm_state_fixed_loso}"
OUTDIR="${OUTDIR:-results/full_data_ensemble_state_fixed_loso}"

TREAT_COL="${TREAT_COL:-shot_zone_choice}"
OUTCOME_COL="${OUTCOME_COL:-delta_wp}"
TREAT_A="${TREAT_A:-three-point}"
TREAT_B="${TREAT_B:-two-point}"

THREADS="${THREADS:-8}"
RUN_CATE_ENSEMBLE="${RUN_CATE_ENSEMBLE:-1}"
CATE_SURFACE_GLOB="${CATE_SURFACE_GLOB:-*tau_surface_${TREAT_A}_vs_${TREAT_B}.parquet}"
CATE_SURFACE_PREFIXES="${CATE_SURFACE_PREFIXES:-full_data_t30_300_ full_data_t0_30_}"
CATE_ENSEMBLE_PREFIX="${CATE_ENSEMBLE_PREFIX:-ensemble_oos_}"
RUN_CATE_CELL_SUPPORT="${RUN_CATE_CELL_SUPPORT:-1}"
CATE_CELL_SUPPORT_DATA="${CATE_CELL_SUPPORT_DATA:-$OUTDIR/nuisance_oos_train.parquet}"
CATE_CELL_SUPPORT_OUTDIR="${CATE_CELL_SUPPORT_OUTDIR:-$OUTDIR/cate_surface_support}"
RUN_CATE_SIGN_AGREEMENT="${RUN_CATE_SIGN_AGREEMENT:-1}"
CATE_SIGN_AGREEMENT_OUTDIR="${CATE_SIGN_AGREEMENT_OUTDIR:-$OUTDIR/cate_surface_sign_agreement}"
CAT_CATE_SURFACE="${CAT_CATE_SURFACE:-}"
XGB_CATE_SURFACE="${XGB_CATE_SURFACE:-}"
LGBM_CATE_SURFACE="${LGBM_CATE_SURFACE:-}"
ENSEMBLE_USE_CALIBRATED_TAU="${ENSEMBLE_USE_CALIBRATED_TAU:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS="$2"
      shift 2
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

mkdir -p "$OUTDIR" logs

PY_BIN="${PY_BIN:-.venv/bin/python}"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

echo "========================================"
echo "Ensemble CATE pipeline"
echo "========================================"
echo "cat_dir:   $CAT_DIR"
echo "xgb_dir:   $XGB_DIR"
echo "lgbm_dir:  $LGBM_DIR"
echo "outdir:    $OUTDIR"
echo "threads:   $THREADS"
echo "run_cate_ensemble: $RUN_CATE_ENSEMBLE"
echo "run_cate_cell_support: $RUN_CATE_CELL_SUPPORT"
echo "cate_cell_support_outdir: $CATE_CELL_SUPPORT_OUTDIR"
echo "run_cate_sign_agreement: $RUN_CATE_SIGN_AGREEMENT"
echo "ensemble_use_calibrated_tau: $ENSEMBLE_USE_CALIBRATED_TAU"
echo "cate_surface_prefixes: $CATE_SURFACE_PREFIXES"

for required in \
  "$CAT_DIR/nuisance_oos_train.parquet" "$CAT_DIR/tau_oos_train.parquet" \
  "$XGB_DIR/nuisance_oos_train.parquet" "$XGB_DIR/tau_oos_train.parquet" \
  "$LGBM_DIR/nuisance_oos_train.parquet" "$LGBM_DIR/tau_oos_train.parquet"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] missing required file: $required" >&2
    exit 1
  fi
done

"$PY_BIN" - <<PY
from pathlib import Path
import json
import numpy as np
import pandas as pd

cat_dir = Path(r"""$CAT_DIR""")
xgb_dir = Path(r"""$XGB_DIR""")
lgbm_dir = Path(r"""$LGBM_DIR""")
outdir = Path(r"""$OUTDIR""")
use_calibrated_tau = str(r"""$ENSEMBLE_USE_CALIBRATED_TAU""") == "1"
outdir.mkdir(parents=True, exist_ok=True)

def read_tau_oos(model_dir: Path) -> tuple[pd.DataFrame, str, Path]:
    if use_calibrated_tau:
        path = model_dir / "full_data_tau_oos_train_calibrated.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"calibrated tau requested but missing: {path}. "
                "Set ENSEMBLE_USE_CALIBRATED_TAU=0 to use raw tau."
            )
        df = pd.read_parquet(path).reset_index(drop=True)
        if "tau_hat_cal" not in df.columns:
            raise KeyError(f"{path} is missing tau_hat_cal")
        return df, "tau_hat_cal", path
    path = model_dir / "tau_oos_train.parquet"
    df = pd.read_parquet(path).reset_index(drop=True)
    if "tau_hat" not in df.columns:
        raise KeyError(f"{path} is missing tau_hat")
    return df, "tau_hat", path

nuis = [
    pd.read_parquet(cat_dir / "nuisance_oos_train.parquet").reset_index(drop=True),
    pd.read_parquet(xgb_dir / "nuisance_oos_train.parquet").reset_index(drop=True),
    pd.read_parquet(lgbm_dir / "nuisance_oos_train.parquet").reset_index(drop=True),
]
tau_pairs = [read_tau_oos(d) for d in [cat_dir, xgb_dir, lgbm_dir]]
tau = [p[0] for p in tau_pairs]
tau_cols = [p[1] for p in tau_pairs]
tau_paths = [p[2] for p in tau_pairs]

n = len(nuis[0])
if any(len(df) != n for df in nuis + tau):
    raise ValueError("row count mismatch among model artifacts")

keys = ["GAME_ID", "GAME_EVENT_ID", "season", "time_left_game", "score_diff", "shot_zone_choice", "delta_wp"]
for k in keys:
    if k in nuis[0].columns:
        b = nuis[0][k].to_numpy()
        for i in (1, 2):
            if not np.array_equal(b, nuis[i][k].to_numpy()):
                raise ValueError(f"key mismatch: {k}")

out_n = nuis[0].copy()
for c in [c for c in out_n.columns if c.startswith("m_hat_") or c.startswith("e_hat_")]:
    out_n[c] = np.mean([df[c].to_numpy(dtype=float) for df in nuis], axis=0)

out_t = tau[0].copy()
raw_tau_cols = ["tau_hat_raw" if "tau_hat_raw" in df.columns else "tau_hat" for df in tau]
out_t["tau_hat_raw"] = np.mean([df[c].to_numpy(dtype=float) for df, c in zip(tau, raw_tau_cols)], axis=0)
out_t["tau_hat_cal"] = np.mean([df[c].to_numpy(dtype=float) for df, c in zip(tau, tau_cols)], axis=0)
out_t["tau_hat"] = out_t["tau_hat_cal"].to_numpy(dtype=float)
out_t["tau_calibration_applied"] = bool(use_calibrated_tau)

out_n_path = outdir / "nuisance_oos_train.parquet"
out_t_path = outdir / "tau_oos_train.parquet"
tau_meta_path = outdir / "ensemble_tau_metadata.json"
out_n.to_parquet(out_n_path, index=False)
out_t.to_parquet(out_t_path, index=False)
tau_meta_path.write_text(
    json.dumps(
        {
            "tau_calibration_applied": bool(use_calibrated_tau),
            "ensemble_tau_column": "tau_hat",
            "model_tau_source_column": "tau_hat_cal" if use_calibrated_tau else "tau_hat",
            "model_tau_source_paths": [str(p) for p in tau_paths],
            "raw_tau_column": "tau_hat_raw",
            "calibrated_tau_column": "tau_hat_cal",
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"[saved] {out_n_path}")
print(f"[saved] {out_t_path}")
print(f"[saved] {tau_meta_path}")
print(f"[info] ensemble tau source column: {'tau_hat_cal' if use_calibrated_tau else 'tau_hat'}")
PY

if [[ "$RUN_CATE_ENSEMBLE" == "1" ]]; then
  derive_surface_prefix() {
    local surface_path="$1"
    local base
    base="$(basename "$surface_path" .parquet)"
    base="${base%tau_surface_${TREAT_A}_vs_${TREAT_B}}"
    printf '%s' "$base"
  }

  collect_surface_pairs() {
    local model_dir="$1"
    local explicit_path="$2"
    local prefix
    local path
    if [[ -n "$explicit_path" ]]; then
      if [[ ! -f "$explicit_path" ]]; then
        echo "[error] specified CATE surface not found: $explicit_path" >&2
        exit 1
      fi
      prefix="$(derive_surface_prefix "$explicit_path")"
      printf '%s\t%s\n' "$prefix" "$explicit_path"
      return
    fi
    mapfile -t _matches < <(find "$model_dir" -maxdepth 1 -type f -name "$CATE_SURFACE_GLOB" | sort)
    if [[ "${#_matches[@]}" -eq 0 ]]; then
      echo "[error] no CATE surface matched in $model_dir with glob: $CATE_SURFACE_GLOB" >&2
      exit 1
    fi
    for path in "${_matches[@]}"; do
      prefix="$(derive_surface_prefix "$path")"
      printf '%s\t%s\n' "$prefix" "$path"
    done
  }

  declare -A CAT_SURFACE_MAP=()
  declare -A XGB_SURFACE_MAP=()
  declare -A LGBM_SURFACE_MAP=()

  while IFS=$'\t' read -r prefix path; do
    [[ -n "$prefix" ]] || continue
    CAT_SURFACE_MAP["$prefix"]="$path"
  done < <(collect_surface_pairs "$CAT_DIR" "$CAT_CATE_SURFACE")

  while IFS=$'\t' read -r prefix path; do
    [[ -n "$prefix" ]] || continue
    XGB_SURFACE_MAP["$prefix"]="$path"
  done < <(collect_surface_pairs "$XGB_DIR" "$XGB_CATE_SURFACE")

  while IFS=$'\t' read -r prefix path; do
    [[ -n "$prefix" ]] || continue
    LGBM_SURFACE_MAP["$prefix"]="$path"
  done < <(collect_surface_pairs "$LGBM_DIR" "$LGBM_CATE_SURFACE")

  mapfile -t COMMON_CATE_PREFIXES < <(
    for prefix in "${!CAT_SURFACE_MAP[@]}"; do
      if [[ -n "${XGB_SURFACE_MAP[$prefix]:-}" && -n "${LGBM_SURFACE_MAP[$prefix]:-}" ]]; then
        printf '%s\n' "$prefix"
      fi
    done | sort
  )

  if [[ "${#COMMON_CATE_PREFIXES[@]}" -eq 0 ]]; then
    echo "[error] no common CATE surface prefixes found across catboost/xgb/lgbm" >&2
    exit 1
  fi

  if [[ -n "$CATE_SURFACE_PREFIXES" ]]; then
    declare -A CATE_PREFIX_ALLOW=()
    for prefix in $CATE_SURFACE_PREFIXES; do
      CATE_PREFIX_ALLOW["$prefix"]=1
    done
    FILTERED_CATE_PREFIXES=()
    for prefix in "${COMMON_CATE_PREFIXES[@]}"; do
      if [[ -n "${CATE_PREFIX_ALLOW[$prefix]:-}" ]]; then
        FILTERED_CATE_PREFIXES+=("$prefix")
      fi
    done
    COMMON_CATE_PREFIXES=("${FILTERED_CATE_PREFIXES[@]}")
    if [[ "${#COMMON_CATE_PREFIXES[@]}" -eq 0 ]]; then
      echo "[error] no common CATE surface prefixes remain after CATE_SURFACE_PREFIXES filter: $CATE_SURFACE_PREFIXES" >&2
      exit 1
    fi
  fi

  for prefix in "${COMMON_CATE_PREFIXES[@]}"; do
    CAT_CATE_SURFACE_RESOLVED="${CAT_SURFACE_MAP[$prefix]}"
    XGB_CATE_SURFACE_RESOLVED="${XGB_SURFACE_MAP[$prefix]}"
    LGBM_CATE_SURFACE_RESOLVED="${LGBM_SURFACE_MAP[$prefix]}"

    echo "Running CATE surface equal-weight ensemble"
    echo "  prefix:   $prefix"
    echo "  catboost: $CAT_CATE_SURFACE_RESOLVED"
    echo "  xgb:      $XGB_CATE_SURFACE_RESOLVED"
    echo "  lgbm:     $LGBM_CATE_SURFACE_RESOLVED"
    "$PY_BIN" scripts/core/average_cate_surfaces.py \
      --surface "$CAT_CATE_SURFACE_RESOLVED" \
      --label "catboost" \
      --surface "$XGB_CATE_SURFACE_RESOLVED" \
      --label "xgb" \
      --surface "$LGBM_CATE_SURFACE_RESOLVED" \
      --label "lgbm" \
      --outdir "$OUTDIR" \
      --prefix "$prefix"

    if [[ "$RUN_CATE_CELL_SUPPORT" == "1" ]]; then
      surface_csv="${OUTDIR}/${prefix}cate_surface_equal_weight.csv"
      if [[ -f "$CATE_CELL_SUPPORT_DATA" && -f "$surface_csv" ]]; then
        echo "Running CATE surface cell-support summary"
        echo "  data:     $CATE_CELL_SUPPORT_DATA"
        echo "  surface:  $surface_csv"
        echo "  outdir:   $CATE_CELL_SUPPORT_OUTDIR"
        "$PY_BIN" scripts/core/plot_cate_cell_counts.py \
          --data "$CATE_CELL_SUPPORT_DATA" \
          --surface-csv "$surface_csv" \
          --outdir "$CATE_CELL_SUPPORT_OUTDIR" \
          --prefix "$prefix"
      else
        echo "[warn] skip CATE cell-support summary: missing data or surface csv"
        echo "       data=$CATE_CELL_SUPPORT_DATA"
        echo "       surface=$surface_csv"
      fi
    fi

    if [[ "$RUN_CATE_SIGN_AGREEMENT" == "1" ]]; then
      echo "Running CATE surface sign agreement summary"
      echo "  outdir:   $CATE_SIGN_AGREEMENT_OUTDIR"
      echo "  prefix:   $prefix"
      "$PY_BIN" scripts/core/summarize_cate_surface_sign_agreement.py \
        --surface "$CAT_CATE_SURFACE_RESOLVED" \
        --label "catboost" \
        --surface "$XGB_CATE_SURFACE_RESOLVED" \
        --label "xgb" \
        --surface "$LGBM_CATE_SURFACE_RESOLVED" \
        --label "lgbm" \
        --outdir "$CATE_SIGN_AGREEMENT_OUTDIR" \
        --prefix "$prefix"
    fi
  done
fi

echo "Completed ensemble OOS CATE pipeline."
echo "OUTDIR: $OUTDIR"
