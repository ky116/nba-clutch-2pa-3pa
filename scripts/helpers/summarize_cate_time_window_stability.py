#!/usr/bin/env python3
"""Summarize CATE stability across late-clock time windows.

This is intended as a post-WP-freeze diagnostic. It does not select a new WP
specification; it compares whether the main CATE structure is stable in
0-30s, 15-30s, 30-60s, and the primary 30-300s window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS: list[tuple[str, float, float]] = [
    ("0_30", 0.0, 30.0),
    ("15_30", 15.0, 30.0),
    ("30_60", 30.0, 60.0),
    ("30_300", 30.0, 300.0),
]

SCORE_REGIONS: list[tuple[str, int, int]] = [
    ("trailing", -10, -1),
    ("tied", 0, 0),
    ("leading", 1, 10),
    ("all_score", -10, 10),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rows",
        type=Path,
        default=Path("results/wf_cate_surfaces/outer_fold_ensemble_tau_test_rows.parquet"),
        help="Row-level held-out CATE predictions from rebuild_wf_cate_surfaces_recalibrated.py.",
    )
    p.add_argument("--outdir", type=Path, default=Path("results/wf_cate_surfaces/time_window_stability"))
    p.add_argument("--score-lo", type=int, default=-10)
    p.add_argument("--score-hi", type=int, default=10)
    p.add_argument(
        "--tau-col",
        default="tau_mean_ensemble",
        help="CATE column to summarize. Positive means 3PA-favoring.",
    )
    return p.parse_args()


def summarize_group(g: pd.DataFrame, tau_col: str) -> pd.Series:
    tau = pd.to_numeric(g[tau_col], errors="coerce").dropna()
    if tau.empty:
        return pd.Series(
            {
                "n": 0,
                "tau_mean": np.nan,
                "tau_median": np.nan,
                "tau_sd": np.nan,
                "share_three_pref": np.nan,
                "share_two_pref": np.nan,
                "share_near_zero_abs_0_001": np.nan,
                "mean_sign_agree_ratio": np.nan,
            }
        )
    out = {
        "n": int(tau.size),
        "tau_mean": float(tau.mean()),
        "tau_median": float(tau.median()),
        "tau_sd": float(tau.std(ddof=1)) if tau.size > 1 else 0.0,
        "share_three_pref": float((tau > 0).mean()),
        "share_two_pref": float((tau < 0).mean()),
        "share_near_zero_abs_0_001": float((tau.abs() <= 0.001).mean()),
    }
    if "tau_sign_agree_ratio" in g.columns:
        out["mean_sign_agree_ratio"] = float(pd.to_numeric(g["tau_sign_agree_ratio"], errors="coerce").mean())
    else:
        out["mean_sign_agree_ratio"] = np.nan
    return pd.Series(out)


def add_windows(df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    time = pd.to_numeric(df["time_left_game"], errors="coerce")
    for name, lo, hi in WINDOWS:
        sub = df[time.ge(lo) & time.lt(hi)].copy()
        sub["time_window"] = name
        sub["time_lo"] = lo
        sub["time_hi"] = hi
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def add_score_regions(df: pd.DataFrame, score_lo: int, score_hi: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    score = pd.to_numeric(df["score_diff"], errors="coerce")
    base = df[score.ge(score_lo) & score.le(score_hi)].copy()
    for name, lo, hi in SCORE_REGIONS:
        sub = base[score.loc[base.index].ge(lo) & score.loc[base.index].le(hi)].copy()
        sub["score_region"] = name
        sub["score_lo"] = lo
        sub["score_hi"] = hi
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def fold_summary(df: pd.DataFrame, tau_col: str) -> pd.DataFrame:
    group_cols = ["time_window", "time_lo", "time_hi", "score_region", "score_lo", "score_hi", "outer_fold", "test_years"]
    return df.groupby(group_cols, sort=True).apply(summarize_group, tau_col=tau_col, include_groups=False).reset_index()


def stability_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in folds.groupby(["time_window", "time_lo", "time_hi", "score_region", "score_lo", "score_hi"], sort=True):
        tau = pd.to_numeric(g["tau_mean"], errors="coerce").dropna()
        if tau.empty:
            continue
        signs = np.sign(tau.to_numpy(dtype=float))
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())
        n_zero = int((signs == 0).sum())
        majority = "three_pref" if n_pos > n_neg else "two_pref" if n_neg > n_pos else "mixed"
        rows.append(
            {
                "time_window": keys[0],
                "time_lo": keys[1],
                "time_hi": keys[2],
                "score_region": keys[3],
                "score_lo": keys[4],
                "score_hi": keys[5],
                "n_folds": int(tau.size),
                "folds_three_pref": n_pos,
                "folds_two_pref": n_neg,
                "folds_zero": n_zero,
                "majority_direction": majority,
                "sign_consistency": float(max(n_pos, n_neg, n_zero) / tau.size),
                "tau_fold_mean": float(tau.mean()),
                "tau_fold_median": float(tau.median()),
                "tau_fold_sd": float(tau.std(ddof=1)) if tau.size > 1 else 0.0,
                "tau_fold_min": float(tau.min()),
                "tau_fold_max": float(tau.max()),
                "row_n_total": int(pd.to_numeric(g["n"], errors="coerce").sum()),
                "row_n_min_fold": int(pd.to_numeric(g["n"], errors="coerce").min()),
                "row_n_max_fold": int(pd.to_numeric(g["n"], errors="coerce").max()),
                "mean_model_sign_agree_ratio": float(pd.to_numeric(g["mean_sign_agree_ratio"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def candidate_comparison(stability: pd.DataFrame) -> pd.DataFrame:
    keep = stability[
        stability["time_window"].isin(["0_30", "15_30", "30_60", "30_300"])
        & stability["score_region"].isin(["trailing", "leading", "all_score"])
    ].copy()
    keep = keep.sort_values(["score_region", "time_lo"])
    return keep[
        [
            "time_window",
            "score_region",
            "n_folds",
            "row_n_total",
            "majority_direction",
            "sign_consistency",
            "tau_fold_mean",
            "tau_fold_sd",
            "tau_fold_min",
            "tau_fold_max",
            "mean_model_sign_agree_ratio",
        ]
    ]


def main() -> None:
    args = parse_args()
    if not args.rows.exists():
        raise FileNotFoundError(args.rows)
    df = pd.read_parquet(args.rows)
    needed = {"outer_fold", "test_years", "time_left_game", "score_diff", args.tau_col}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"{args.rows} missing columns: {missing}")

    with_windows = add_windows(df)
    expanded = add_score_regions(with_windows, args.score_lo, args.score_hi)
    folds = fold_summary(expanded, args.tau_col)
    stability = stability_summary(folds)
    comparison = candidate_comparison(stability)

    args.outdir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.outdir / "cate_time_window_stability_by_fold.csv", index=False)
    stability.to_csv(args.outdir / "cate_time_window_stability_summary.csv", index=False)
    comparison.to_csv(args.outdir / "cate_time_window_primary_candidate_comparison.csv", index=False)

    readme = args.outdir / "README.md"
    readme.write_text(
        """# CATE Time-Window Stability Diagnostic

Purpose: summarize late-clock validity after setting the primary CATE surface
to 30-300s and treating 0-30s as a separate regime.

Key comparison:
- 0-30s: late-clock validity / separate-regime diagnostic region.
- 15-30s: boundary-adjacent subregion retained for diagnostics.
- 30-60s: near-late reference region.
- 30-300s: primary CATE window.

Interpretation target:
- trailing states should remain 3PA-favoring;
- leading states should remain weak or negative;
- signs should be stable across outer folds rather than driven by one test era.
""",
        encoding="utf-8",
    )
    print(f"[saved] {args.outdir / 'cate_time_window_stability_by_fold.csv'}")
    print(f"[saved] {args.outdir / 'cate_time_window_stability_summary.csv'}")
    print(f"[saved] {args.outdir / 'cate_time_window_primary_candidate_comparison.csv'}")


if __name__ == "__main__":
    main()
