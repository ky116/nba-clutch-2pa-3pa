#!/usr/bin/env python3
"""Summarize WP model-dependence sensitivity for the final CATE surface.

The diagnostic compares tau_wp(X) with tau_wp(X) - b(X), where b(X) is the
conditional differential WP calibration residual for 3PA versus 2PA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TIME_BANDS: list[tuple[str, float, float]] = [
    ("15_30", 15.0, 30.0),
    ("30_60", 30.0, 60.0),
    ("60_120", 60.0, 120.0),
    ("120_180", 120.0, 180.0),
    ("180_240", 180.0, 240.0),
    ("240_300", 240.0, 300.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cate-surface",
        type=Path,
        default=Path("results/full_data_ensemble_state_fixed_loso/full_data_t30_300_cate_surface_equal_weight.csv"),
    )
    p.add_argument(
        "--bias-surface",
        type=Path,
        default=Path("results/wp_calibration/differential_surface/wp_residual_differential_calibration_surface.csv"),
    )
    p.add_argument(
        "--support",
        type=Path,
        default=Path("results/full_data_ensemble_state_fixed_loso/cate_surface_support/full_data_t30_300_cate_surface_cell_counts.csv"),
    )
    p.add_argument("--outdir", type=Path, default=Path("results/wp_calibration/model_dependence_sensitivity"))
    p.add_argument("--score-lo", type=float, default=-7)
    p.add_argument("--score-hi", type=float, default=7)
    p.add_argument("--time-lo", type=float, default=30)
    p.add_argument("--time-hi", type=float, default=300)
    p.add_argument("--tau-col", default="tau_mean_ensemble")
    p.add_argument("--bias-col", default="b_hat")
    p.add_argument("--support-threshold", type=int, default=20)
    p.add_argument("--extreme-top-n", type=int, default=30)
    return p.parse_args()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def filter_window(df: pd.DataFrame, score_lo: float, score_hi: float, time_lo: float, time_hi: float) -> pd.DataFrame:
    score = pd.to_numeric(df["score_diff"], errors="coerce")
    time = pd.to_numeric(df["time_left_game"], errors="coerce")
    return df[score.ge(score_lo) & score.le(score_hi) & time.ge(time_lo) & time.le(time_hi)].copy()


def interpolate_bias_to_cate_grid(cate: pd.DataFrame, bias: pd.DataFrame, bias_col: str) -> pd.DataFrame:
    out = cate[["time_left_game", "score_diff"]].copy()
    out["b_hat"] = np.nan
    out["b_hat_se"] = np.nan
    for score, idx in out.groupby("score_diff", sort=False).groups.items():
        cur = bias[pd.to_numeric(bias["score_diff"], errors="coerce").eq(score)].copy()
        cur = cur.sort_values("time_left_game")
        if cur.empty:
            continue
        x = pd.to_numeric(cur["time_left_game"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(cur[bias_col], errors="coerce").to_numpy(dtype=float)
        targets = pd.to_numeric(out.loc[idx, "time_left_game"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue
        out.loc[idx, "b_hat"] = np.interp(targets, x[valid], y[valid])
        if "b_hat_se" in cur.columns:
            se = pd.to_numeric(cur["b_hat_se"], errors="coerce").to_numpy(dtype=float)
            valid_se = np.isfinite(x) & np.isfinite(se)
            if valid_se.sum() >= 2:
                out.loc[idx, "b_hat_se"] = np.interp(targets, x[valid_se], se[valid_se])
    return out


def sign_label(values: pd.Series) -> pd.Series:
    signs = np.sign(pd.to_numeric(values, errors="coerce"))
    return signs.map({-1.0: "negative", 0.0: "zero", 1.0: "positive"})


def summarize_subset(label: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    valid = df["tau_wp"].notna() & df["b_hat"].notna() & df["tau_sensitivity"].notna()
    cur = df.loc[valid].copy()
    if cur.empty:
        return {
            "subset": label,
            "n_cells": 0,
            "n_supported_cells": 0,
            "sign_flip_share": np.nan,
            "mean_abs_difference": np.nan,
            "median_abs_difference": np.nan,
            "p90_abs_difference": np.nan,
            "p95_abs_difference": np.nan,
            "max_abs_difference": np.nan,
            "mean_tau_wp": np.nan,
            "mean_tau_sensitivity": np.nan,
            "share_tau_wp_positive": np.nan,
            "share_tau_sensitivity_positive": np.nan,
        }
    abs_diff = cur["abs_difference"]
    return {
        "subset": label,
        "n_cells": int(len(cur)),
        "n_supported_cells": int(cur["supported"].sum()) if "supported" in cur.columns else 0,
        "sign_flip_share": float(cur["sign_flip"].mean()),
        "mean_abs_difference": float(abs_diff.mean()),
        "median_abs_difference": float(abs_diff.median()),
        "p90_abs_difference": float(abs_diff.quantile(0.90)),
        "p95_abs_difference": float(abs_diff.quantile(0.95)),
        "max_abs_difference": float(abs_diff.max()),
        "mean_tau_wp": float(cur["tau_wp"].mean()),
        "mean_tau_sensitivity": float(cur["tau_sensitivity"].mean()),
        "share_tau_wp_positive": float((cur["tau_wp"] > 0).mean()),
        "share_tau_sensitivity_positive": float((cur["tau_sensitivity"] > 0).mean()),
    }


def add_time_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_band"] = pd.NA
    t = pd.to_numeric(out["time_left_game"], errors="coerce")
    for name, lo, hi in TIME_BANDS:
        upper = t.le(hi) if hi == TIME_BANDS[-1][2] else t.lt(hi)
        out.loc[t.ge(lo) & upper, "time_band"] = name
    return out


def main() -> None:
    args = parse_args()
    cate = read_csv(args.cate_surface, "CATE surface")
    bias = read_csv(args.bias_surface, "bias surface")
    support = read_csv(args.support, "support surface")
    require_columns(cate, ["time_left_game", "score_diff", args.tau_col], "CATE surface")
    require_columns(bias, ["time_left_game", "score_diff", args.bias_col], "bias surface")
    require_columns(support, ["time_left_game", "score_diff", "cell_count"], "support surface")

    cate = filter_window(cate, args.score_lo, args.score_hi, args.time_lo, args.time_hi)
    bias = filter_window(bias, args.score_lo, args.score_hi, args.time_lo, args.time_hi)
    support = filter_window(support, args.score_lo, args.score_hi, args.time_lo, args.time_hi)

    surface = cate[["time_left_game", "score_diff", args.tau_col]].rename(columns={args.tau_col: "tau_wp"})
    bias_interp = interpolate_bias_to_cate_grid(surface, bias, args.bias_col)
    surface = surface.merge(bias_interp, on=["time_left_game", "score_diff"], how="left")
    support_cols = ["time_left_game", "score_diff", "cell_count"]
    if "cell_density" in support.columns:
        support_cols.append("cell_density")
    surface = surface.merge(support[support_cols], on=["time_left_game", "score_diff"], how="left")
    if "cell_density" not in surface.columns:
        surface["cell_density"] = np.nan
    surface["tau_sensitivity"] = surface["tau_wp"] - surface["b_hat"]
    surface["abs_difference"] = (surface["tau_wp"] - surface["tau_sensitivity"]).abs()
    surface["supported"] = pd.to_numeric(surface["cell_count"], errors="coerce").ge(args.support_threshold)
    surface["tau_wp_sign"] = sign_label(surface["tau_wp"])
    surface["tau_sensitivity_sign"] = sign_label(surface["tau_sensitivity"])
    valid_sign = surface["tau_wp"].notna() & surface["tau_sensitivity"].notna()
    surface["sign_flip"] = pd.Series(pd.NA, index=surface.index, dtype="boolean")
    surface.loc[valid_sign, "sign_flip"] = (
        np.sign(surface.loc[valid_sign, "tau_wp"]) != np.sign(surface.loc[valid_sign, "tau_sensitivity"])
    )
    surface = add_time_band(surface)

    rows = [
        summarize_subset("all_cells", surface),
        summarize_subset(f"supported_cell_count_ge_{args.support_threshold}", surface[surface["supported"]]),
        summarize_subset("trailing_all", surface[surface["score_diff"] < 0]),
        summarize_subset("trailing_supported", surface[(surface["score_diff"] < 0) & surface["supported"]]),
        summarize_subset("tied_all", surface[surface["score_diff"].eq(0)]),
        summarize_subset("leading_all", surface[surface["score_diff"] > 0]),
        summarize_subset("leading_supported", surface[(surface["score_diff"] > 0) & surface["supported"]]),
        summarize_subset("late_240_300_all", surface[surface["time_left_game"].ge(240)]),
        summarize_subset("late_240_300_supported", surface[surface["time_left_game"].ge(240) & surface["supported"]]),
    ]
    summary = pd.DataFrame(rows)

    by_score = pd.DataFrame(
        summarize_subset(f"score_{score:g}", group)
        | {"score_diff": score}
        for score, group in surface.groupby("score_diff", sort=True)
    )
    by_time_band = pd.DataFrame(
        summarize_subset(str(band), group)
        | {"time_band": band}
        for band, group in surface.dropna(subset=["time_band"]).groupby("time_band", sort=False)
    )
    extremes = surface.sort_values("abs_difference", ascending=False).head(args.extreme_top_n)
    late_extremes = surface[surface["time_left_game"].ge(240)].sort_values("abs_difference", ascending=False).head(args.extreme_top_n)

    args.outdir.mkdir(parents=True, exist_ok=True)
    surface.to_csv(args.outdir / "wp_model_dependence_sensitivity_surface.csv", index=False)
    summary.to_csv(args.outdir / "wp_model_dependence_sensitivity_summary.csv", index=False)
    by_score.to_csv(args.outdir / "wp_model_dependence_sensitivity_by_score.csv", index=False)
    by_time_band.to_csv(args.outdir / "wp_model_dependence_sensitivity_by_time_band.csv", index=False)
    extremes.to_csv(args.outdir / "wp_model_dependence_sensitivity_extreme_cells.csv", index=False)
    late_extremes.to_csv(args.outdir / "wp_model_dependence_sensitivity_late_240_300_extreme_cells.csv", index=False)
    (args.outdir / "README.md").write_text(
        f"""# WP Model-Dependence Sensitivity

Compares `tau_wp` from the final CATE surface with `tau_sensitivity = tau_wp - b_hat`.
`b_hat` is linearly interpolated from the WP residual differential calibration
surface onto the CATE surface grid.

Inputs:
- CATE surface: `{args.cate_surface}`
- WP residual differential surface: `{args.bias_surface}`
- Support surface: `{args.support}`

Scope:
- score differential: {args.score_lo:g} to {args.score_hi:g}
- time remaining: {args.time_lo:g} to {args.time_hi:g}
- supported cell threshold: cell_count >= {args.support_threshold}
""",
        encoding="utf-8",
    )

    print(f"[saved] {args.outdir / 'wp_model_dependence_sensitivity_surface.csv'}")
    print(f"[saved] {args.outdir / 'wp_model_dependence_sensitivity_summary.csv'}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
