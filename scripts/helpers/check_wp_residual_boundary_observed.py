#!/usr/bin/env python3
"""Direct observed residual check for the 300-second WP bias surface boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
CORE_DIR = PROJECT_DIR / "scripts" / "core"
HELPER_DIR = PROJECT_DIR / "scripts" / "helpers"
for path in (CORE_DIR, HELPER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fit_wp_residual_differential_surface import attach_residual, load_panel, load_residuals  # noqa: E402
from treatment_utils import apply_treatment_scheme  # noqa: E402


WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("285_300", 285.0, 300.0),
    ("295_300", 295.0, 300.0),
    ("exact_300", 300.0, 300.0),
    ("270_285", 270.0, 285.0),
    ("240_285", 240.0, 285.0),
)

SCORE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("trailing_-7_-1", -7.0, -1.0),
    ("tied_0", 0.0, 0.0),
    ("leading_1_7", 1.0, 7.0),
    ("all_-7_7", -7.0, 7.0),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=Path("data/analysis/shotchoice_panel_clutch_rs.parquet"))
    p.add_argument("--with-wp", type=Path, default=Path("data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz"))
    p.add_argument(
        "--bias-surface",
        type=Path,
        default=Path("results/wp_calibration/differential_surface/wp_residual_differential_calibration_surface.csv"),
    )
    p.add_argument("--outdir", type=Path, default=Path("results/wp_calibration/model_dependence_sensitivity"))
    p.add_argument("--score-lo", type=float, default=-7.0)
    p.add_argument("--score-hi", type=float, default=7.0)
    return p.parse_args()


def observed_diff(rows: pd.DataFrame) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    means: dict[str, float] = {}
    for label in ("three-point", "two-point"):
        residual = rows.loc[rows["shot_zone_choice"].astype(str).eq(label), "wp_residual_offense"]
        short = label.replace("-", "_")
        out[f"n_{short}"] = int(residual.notna().sum())
        means[label] = float(residual.mean()) if residual.notna().sum() else np.nan
        out[f"mean_resid_{short}"] = means[label]
    out["observed_resid_diff_3pa_minus_2pa"] = (
        means["three-point"] - means["two-point"]
        if np.isfinite(means["three-point"]) and np.isfinite(means["two-point"])
        else np.nan
    )
    return out


def add_surface_comparison(out: pd.DataFrame, surface_300: pd.DataFrame, value_col: str) -> pd.DataFrame:
    merged = out.merge(surface_300[["score_diff", value_col]], on="score_diff", how="left")
    obs = merged["observed_resid_diff_3pa_minus_2pa"]
    b_hat = merged[value_col]
    valid = obs.notna() & b_hat.notna()
    merged["same_sign_as_b_hat_300"] = pd.Series(pd.NA, index=merged.index, dtype="boolean")
    merged.loc[valid, "same_sign_as_b_hat_300"] = np.sign(obs.loc[valid]) == np.sign(b_hat.loc[valid])
    merged["abs_ratio_observed_to_b_hat_300"] = obs.abs() / b_hat.abs()
    return merged


def main() -> None:
    args = parse_args()
    panel = load_panel(args.panel)
    residuals = load_residuals(args.with_wp)
    df, meta = attach_residual(panel, residuals)
    df = apply_treatment_scheme(
        df,
        treat_col="shot_zone_choice",
        scheme="binary",
        out_col="shot_zone_choice",
        drop_unknown=True,
    )
    df = df[df["shot_zone_choice"].astype(str).isin(["three-point", "two-point"])].copy()
    df = df.dropna(subset=["wp_residual_offense", "time_left_game", "score_diff"]).copy()
    df["time_left_game"] = pd.to_numeric(df["time_left_game"], errors="coerce")
    df["score_diff"] = pd.to_numeric(df["score_diff"], errors="coerce")

    bias = pd.read_csv(args.bias_surface)
    surface_300 = bias[
        pd.to_numeric(bias["time_left_game"], errors="coerce").eq(300)
        & pd.to_numeric(bias["score_diff"], errors="coerce").between(args.score_lo, args.score_hi)
    ][["score_diff", "b_hat"]].copy()

    score_rows: list[dict[str, float | int | str]] = []
    band_rows: list[dict[str, float | int | str]] = []
    for window, lo, hi in WINDOWS:
        cur = df[
            df["time_left_game"].ge(lo)
            & df["time_left_game"].le(hi)
            & df["score_diff"].between(args.score_lo, args.score_hi)
        ].copy()
        for score in range(int(args.score_lo), int(args.score_hi) + 1):
            cell = cur[cur["score_diff"].eq(score)]
            score_rows.append(
                {"window": window, "time_lo": lo, "time_hi": hi, "score_diff": float(score)}
                | observed_diff(cell)
            )
        for band, score_lo, score_hi in SCORE_BANDS:
            cell = cur[cur["score_diff"].between(score_lo, score_hi)]
            band = {
                "window": window,
                "time_lo": lo,
                "time_hi": hi,
                "score_band": band,
                "score_lo": score_lo,
                "score_hi": score_hi,
            } | observed_diff(cell)
            b_cur = surface_300[surface_300["score_diff"].between(score_lo, score_hi)]["b_hat"]
            band["mean_b_hat_300"] = float(b_cur.mean()) if b_cur.notna().sum() else np.nan
            obs = band["observed_resid_diff_3pa_minus_2pa"]
            b_hat = band["mean_b_hat_300"]
            band["same_sign_as_mean_b_hat_300"] = bool(np.sign(obs) == np.sign(b_hat)) if np.isfinite(obs) and np.isfinite(b_hat) else pd.NA
            band["abs_ratio_observed_to_mean_b_hat_300"] = abs(obs) / abs(b_hat) if np.isfinite(obs) and b_hat else np.nan
            band_rows.append(band)

    score_out = add_surface_comparison(pd.DataFrame(score_rows), surface_300, "b_hat")
    band_out = pd.DataFrame(band_rows)

    summary = []
    for window, group in score_out.groupby("window", sort=False):
        valid = group["observed_resid_diff_3pa_minus_2pa"].notna() & group["b_hat"].notna()
        summary.append(
            {
                "window": window,
                "n_score_cells": int(valid.sum()),
                "same_sign_share_score_cells": float(group.loc[valid, "same_sign_as_b_hat_300"].mean()) if valid.sum() else np.nan,
                "mean_observed_diff_score_cells": float(group.loc[valid, "observed_resid_diff_3pa_minus_2pa"].mean()) if valid.sum() else np.nan,
                "mean_b_hat_300_score_cells": float(group.loc[valid, "b_hat"].mean()) if valid.sum() else np.nan,
            }
        )
    summary_out = pd.DataFrame(summary)

    args.outdir.mkdir(parents=True, exist_ok=True)
    score_path = args.outdir / "wp_residual_boundary_observed_check_by_score.csv"
    band_path = args.outdir / "wp_residual_boundary_observed_check_by_band.csv"
    summary_path = args.outdir / "wp_residual_boundary_observed_check_summary.csv"
    score_out.to_csv(score_path, index=False)
    band_out.to_csv(band_path, index=False)
    summary_out.to_csv(summary_path, index=False)

    readme = args.outdir / "wp_residual_boundary_observed_check.md"
    readme.write_text(
        "# WP Residual Boundary Observed Check\n\n"
        "Direct observed check for whether the exact-300s `b_hat` row is supported by nearby data.\n"
        "`observed_resid_diff_3pa_minus_2pa` is the unadjusted mean offense residual for 3PA minus 2PA,\n"
        "where the residual is `wp_next_offense - final_win_offense`. The join and treatment mapping match\n"
        "`fit_wp_residual_differential_surface.py`.\n\n"
        f"Merged rows: {meta['merged_rows']:,}; rows after binary treatment/drop: {len(df):,}.\n\n"
        f"- By score: `{score_path}`\n"
        f"- By score band: `{band_path}`\n"
        f"- Summary: `{summary_path}`\n",
        encoding="utf-8",
    )

    print(f"[saved] {score_path}")
    print(f"[saved] {band_path}")
    print(f"[saved] {summary_path}")
    print(summary_out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(band_out[band_out["window"].isin(["285_300", "295_300", "exact_300"])].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
