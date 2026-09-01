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

SCORE_MARGIN_REGIONS: list[tuple[str, int, int]] = [
    ("trailing_3plus_possessions", -10, -7),
    ("trailing_2_possessions", -6, -4),
    ("trailing_1_possession", -3, -1),
    ("tied", 0, 0),
    ("leading_1_possession", 1, 3),
    ("leading_2_possessions", 4, 6),
    ("leading_3plus_possessions", 7, 10),
]

REGION_SUMMARY_COLUMNS = [
    "region",
    "n_cells",
    "mean_tau_wp",
    "mean_tau_sensitivity",
    "mean_difference",
    "sign_flip_share",
    "share_tau_wp_positive",
    "share_tau_sensitivity_positive",
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
        default=Path(
            "results/wp_calibration/model_dependence_refit/ensemble/"
            "wp_residual_t30_300_cate_surface_equal_weight.csv"
        ),
    )
    p.add_argument(
        "--support",
        type=Path,
        default=Path("results/full_data_ensemble_state_fixed_loso/cate_surface_support/full_data_t30_300_cate_surface_cell_counts.csv"),
    )
    p.add_argument("--outdir", type=Path, default=Path("results/wp_calibration/model_dependence_sensitivity"))
    p.add_argument("--score-lo", type=float, default=-10)
    p.add_argument("--score-hi", type=float, default=10)
    p.add_argument("--time-lo", type=float, default=30)
    p.add_argument("--time-hi", type=float, default=300)
    p.add_argument("--tau-col", default="tau_mean_ensemble")
    p.add_argument("--bias-col", default="tau_mean_ensemble")
    p.add_argument("--support-threshold", type=int, default=20)
    p.add_argument("--extreme-top-n", type=int, default=30)
    p.add_argument(
        "--existing-sensitivity-surface",
        type=Path,
        help="Reaggregate an existing 525-cell sensitivity surface without rebuilding it.",
    )
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


def align_bias_to_cate_grid(cate: pd.DataFrame, bias: pd.DataFrame, bias_col: str) -> pd.DataFrame:
    """Require an exact one-to-one match; interpolation is intentionally forbidden."""
    keys = ["time_left_game", "score_diff"]
    if cate.duplicated(keys).any():
        raise ValueError("CATE surface has duplicate grid cells.")
    if bias.duplicated(keys).any():
        raise ValueError("bias surface has duplicate grid cells.")

    keep = keys + [bias_col]
    if "b_hat_se" in bias.columns and "b_hat_se" not in keep:
        keep.append("b_hat_se")
    aligned = cate[keys].merge(bias[keep], on=keys, how="left", validate="one_to_one", indicator=True)
    missing = aligned["_merge"].ne("both")
    if missing.any():
        examples = aligned.loc[missing, keys].head(5).to_dict("records")
        raise ValueError(f"bias surface is missing {int(missing.sum())} exact CATE grid cells; examples={examples}")
    aligned = aligned.drop(columns="_merge").rename(columns={bias_col: "b_hat"})
    if aligned["b_hat"].isna().any():
        raise ValueError("bias surface contains nonnumeric or missing b(x) values on the CATE grid.")
    return aligned


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


def validate_full_sensitivity_grid(surface: pd.DataFrame) -> None:
    """Require the canonical 25-time by 21-score grid used for region summaries."""
    required = ["time_left_game", "score_diff", "tau_wp", "tau_sensitivity", "sign_flip"]
    require_columns(surface, required, "sensitivity surface")
    keys = ["time_left_game", "score_diff"]
    if surface.duplicated(keys).any():
        raise ValueError("sensitivity surface has duplicate grid cells.")

    time = np.sort(pd.to_numeric(surface["time_left_game"], errors="coerce").dropna().unique())
    score = np.sort(pd.to_numeric(surface["score_diff"], errors="coerce").dropna().unique())
    expected_time = np.linspace(30.0, 300.0, 25)
    expected_score = np.arange(-10.0, 11.0)
    if len(surface) != 525 or not np.array_equal(time, expected_time) or not np.array_equal(score, expected_score):
        raise ValueError(
            "region summaries require the complete 525-cell grid: "
            "25 time points from 30 to 300 s and integer score differentials -10 to 10."
        )


def summarize_score_margin_regions(surface: pd.DataFrame) -> pd.DataFrame:
    validate_full_sensitivity_grid(surface)
    rows: list[dict[str, float | int | str]] = []
    score = pd.to_numeric(surface["score_diff"], errors="coerce")
    for label, lo, hi in SCORE_MARGIN_REGIONS:
        base = summarize_subset(label, surface[score.between(lo, hi)])
        rows.append(
            {
                "region": label,
                "n_cells": base["n_cells"],
                "mean_tau_wp": base["mean_tau_wp"],
                "mean_tau_sensitivity": base["mean_tau_sensitivity"],
                "mean_difference": float(base["mean_tau_sensitivity"] - base["mean_tau_wp"]),
                "sign_flip_share": base["sign_flip_share"],
                "share_tau_wp_positive": base["share_tau_wp_positive"],
                "share_tau_sensitivity_positive": base["share_tau_sensitivity_positive"],
            }
        )
    summary = pd.DataFrame(rows, columns=REGION_SUMMARY_COLUMNS)
    if int(summary["n_cells"].sum()) != 525:
        raise ValueError("score-margin regions do not partition all 525 sensitivity cells.")
    return summary


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
    if args.existing_sensitivity_surface is not None:
        surface = read_csv(args.existing_sensitivity_surface, "existing sensitivity surface")
        require_columns(
            surface,
            [
                "time_left_game",
                "score_diff",
                "tau_wp",
                "b_hat",
                "tau_sensitivity",
                "abs_difference",
                "supported",
                "sign_flip",
            ],
            "existing sensitivity surface",
        )
        if "time_band" not in surface.columns:
            surface = add_time_band(surface)
    else:
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
        bias_aligned = align_bias_to_cate_grid(surface, bias, args.bias_col)
        surface = surface.merge(
            bias_aligned,
            on=["time_left_game", "score_diff"],
            how="left",
            validate="one_to_one",
        )
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
        summarize_subset("late_240_300_all", surface[surface["time_left_game"].ge(240)]),
        summarize_subset("late_240_300_supported", surface[surface["time_left_game"].ge(240) & surface["supported"]]),
    ]
    summary = pd.DataFrame(rows)
    region_summary = summarize_score_margin_regions(surface)

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
    if args.existing_sensitivity_surface is None:
        surface.to_csv(args.outdir / "wp_model_dependence_sensitivity_surface.csv", index=False)
    summary.to_csv(args.outdir / "wp_model_dependence_sensitivity_summary.csv", index=False)
    region_summary.to_csv(args.outdir / "wp_model_dependence_sensitivity_by_score_margin_region.csv", index=False)
    if args.existing_sensitivity_surface is None:
        by_score.to_csv(args.outdir / "wp_model_dependence_sensitivity_by_score.csv", index=False)
        by_time_band.to_csv(args.outdir / "wp_model_dependence_sensitivity_by_time_band.csv", index=False)
        extremes.to_csv(args.outdir / "wp_model_dependence_sensitivity_extreme_cells.csv", index=False)
        late_extremes.to_csv(args.outdir / "wp_model_dependence_sensitivity_late_240_300_extreme_cells.csv", index=False)
    (args.outdir / "README.md").write_text(
        f"""# WP Model-Dependence Sensitivity

Compares `tau_wp` from the final CATE surface with `tau_sensitivity = tau_wp - b_hat`.
`b_hat` is estimated directly on the same 25-point time grid and 21-point
score grid as `tau_wp`; the two surfaces are joined exactly without interpolation.

Inputs:
- CATE surface: `{args.cate_surface}`
- WP residual differential surface: `{args.bias_surface}`
- Support surface: `{args.support}`

Scope:
- score differential: {args.score_lo:g} to {args.score_hi:g}
- time remaining: {args.time_lo:g} to {args.time_hi:g}
- expected full grid: 25 x 21 = 525 cells
- supported cell threshold: cell_count >= {args.support_threshold}

Score-margin regions (all 25 time points from 30 to 300 s):
- trailing 3+ possessions: -10 to -7
- trailing 2 possessions: -6 to -4
- trailing 1 possession: -3 to -1
- tied: 0
- leading 1 possession: 1 to 3
- leading 2 possessions: 4 to 6
- leading 3+ possessions: 7 to 10
""",
        encoding="utf-8",
    )

    if args.existing_sensitivity_surface is None:
        print(f"[saved] {args.outdir / 'wp_model_dependence_sensitivity_surface.csv'}")
    print(f"[saved] {args.outdir / 'wp_model_dependence_sensitivity_summary.csv'}")
    print(f"[saved] {args.outdir / 'wp_model_dependence_sensitivity_by_score_margin_region.csv'}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(region_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
