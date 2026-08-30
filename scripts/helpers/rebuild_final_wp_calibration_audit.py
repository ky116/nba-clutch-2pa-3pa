#!/usr/bin/env python
"""Rebuild final late-clock WP calibration diagnostics.

The primary audit compares the preserved baseline M0 WP against the frozen
late45 surface WP on broad late-tail next states. Narrow down-3 made-shot cells
are retained as targeted and stress diagnostics, not as the main conclusion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-6
SHOT_TYPES = ("2PA", "3PA")
TIME_BINS = [(0.0, 15.0), (15.0, 30.0), (30.0, 45.0), (45.0, 60.0)]
COARSE_SANITY_BINS = [
    (45.0, 60.0),
    (60.0, 120.0),
    (120.0, 180.0),
    (180.0, 240.0),
    (240.0, 300.0),
]


@dataclass(frozen=True)
class ModelInput:
    label: str
    path: Path


def clip_prob(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").clip(EPS, 1 - EPS)


def logit(p: pd.Series) -> pd.Series:
    p = clip_prob(p)
    return np.log(p / (1 - p))


def logistic_calibration(y: pd.Series, p: pd.Series) -> tuple[float, float]:
    try:
        import statsmodels.api as sm
    except Exception:
        return np.nan, np.nan

    yv = pd.to_numeric(y, errors="coerce")
    eta = logit(p)
    keep = yv.isin([0, 1]) & eta.notna()
    yv = yv[keep].astype(float)
    eta = eta[keep].astype(float)
    if len(yv) < 50 or yv.nunique() < 2:
        return np.nan, np.nan
    x = sm.add_constant(eta.to_numpy())
    try:
        fit = sm.GLM(yv.to_numpy(), x, family=sm.families.Binomial()).fit()
    except Exception:
        return np.nan, np.nan
    return float(fit.params[0]), float(fit.params[1])


def ece_metrics(y: pd.Series, p: pd.Series, n_bins: int) -> tuple[float, float]:
    df = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "p": clip_prob(p)}).dropna()
    df = df[df["y"].isin([0, 1])]
    if df.empty:
        return np.nan, np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(df["p"], bins, right=True) - 1, 0, n_bins - 1)
    df["_bin"] = idx
    grouped = df.groupby("_bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), obs=("y", "mean")
    )
    gap = (grouped["pred"] - grouped["obs"]).abs()
    return float((gap * grouped["n"]).sum() / grouped["n"].sum()), float(gap.max())


def metric_row(y: pd.Series, p: pd.Series) -> dict[str, float | int]:
    df = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "p": clip_prob(p)}).dropna()
    df = df[df["y"].isin([0, 1])]
    n = int(len(df))
    if n == 0:
        return {
            "n": 0,
            "mean_fitted_wp": np.nan,
            "empirical_win_rate": np.nan,
            "gap": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "ece": np.nan,
            "mce": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }
    ece, mce = ece_metrics(df["y"], df["p"], n_bins=20)
    intercept, slope = logistic_calibration(df["y"], df["p"])
    mean_pred = float(df["p"].mean())
    empirical = float(df["y"].mean())
    return {
        "n": n,
        "mean_fitted_wp": mean_pred,
        "empirical_win_rate": empirical,
        "gap": mean_pred - empirical,
        "brier": float(np.mean((df["p"] - df["y"]) ** 2)),
        "log_loss": float(-np.mean(df["y"] * np.log(df["p"]) + (1 - df["y"]) * np.log(1 - df["p"]))),
        "ece": ece,
        "mce": mce,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["GAME_ID_norm"] = (
        out["GAME_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    )
    out["GAME_EVENT_ID_norm"] = pd.to_numeric(
        out["GAME_EVENT_ID"], errors="coerce"
    ).astype("Int64")
    return out


def add_offense_orientation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    home_poss = pd.to_numeric(out["before_home_possession"], errors="coerce")
    home_margin = pd.to_numeric(out["before_score_diff"], errors="coerce")
    home_wp = pd.to_numeric(out["wp_next"], errors="coerce")
    home_win = pd.to_numeric(out["final_home_win"], errors="coerce")
    out["offense_score_diff_before"] = np.where(
        home_poss.eq(1), home_margin, np.where(home_poss.eq(0), -home_margin, np.nan)
    )
    out["offense_wp_next"] = np.where(
        home_poss.eq(1), home_wp, np.where(home_poss.eq(0), 1 - home_wp, np.nan)
    )
    out["offense_win"] = np.where(
        home_poss.eq(1), home_win, np.where(home_poss.eq(0), 1 - home_win, np.nan)
    )
    return out


def read_audit_rows(model: ModelInput) -> pd.DataFrame:
    usecols = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "shot_made",
        "next_is_terminal",
        "before_score_diff",
        "before_time_left_game",
        "before_home_possession",
        "next_score_diff",
        "next_time_left_game",
        "wp_next",
        "final_home_win",
        "season",
    ]
    parts = []
    for chunk in pd.read_csv(model.path, usecols=usecols, chunksize=500_000):
        terminal = pd.to_numeric(chunk["next_is_terminal"], errors="coerce").eq(1)
        next_time = pd.to_numeric(chunk["next_time_left_game"], errors="coerce")
        next_score = pd.to_numeric(chunk["next_score_diff"], errors="coerce")
        before_time = pd.to_numeric(chunk["before_time_left_game"], errors="coerce")
        before_score = pd.to_numeric(chunk["before_score_diff"], errors="coerce")
        late_tail = next_time.ge(0) & next_time.le(45) & next_score.abs().le(7) & ~terminal
        shot_context = before_time.gt(0) & before_time.le(60) & before_score.abs().le(10) & ~terminal
        coarse_sanity = next_time.gt(45) & next_time.le(300) & ~terminal
        keep = late_tail | shot_context | coarse_sanity
        if keep.any():
            sub = normalize_ids(add_offense_orientation(chunk.loc[keep]))
            sub["model"] = model.label
            parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach_shot_type(rows: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    if rows.empty:
        return rows.assign(shot_type=pd.Series(dtype="object"))
    maps = []
    for season in sorted(rows["season"].dropna().astype(int).unique()):
        path = raw_dir / f"shotdetail_{season}.csv"
        if not path.exists():
            continue
        ids = rows.loc[
            rows["season"].eq(season), ["GAME_ID_norm", "GAME_EVENT_ID_norm"]
        ].drop_duplicates()
        shots = pd.read_csv(
            path,
            usecols=["GAME_ID", "GAME_EVENT_ID", "SHOT_TYPE", "SHOT_ATTEMPTED_FLAG"],
            low_memory=False,
        )
        shots = shots[pd.to_numeric(shots["SHOT_ATTEMPTED_FLAG"], errors="coerce").eq(1)].copy()
        shots = normalize_ids(shots)
        shots = shots.merge(ids, on=["GAME_ID_norm", "GAME_EVENT_ID_norm"], how="inner")
        maps.append(shots[["GAME_ID_norm", "GAME_EVENT_ID_norm", "SHOT_TYPE"]].drop_duplicates())
    if not maps:
        raise SystemExit("No shotdetail rows matched the calibration sample.")
    shotmap = pd.concat(maps, ignore_index=True).drop_duplicates(
        ["GAME_ID_norm", "GAME_EVENT_ID_norm"]
    )
    out = rows.merge(shotmap, on=["GAME_ID_norm", "GAME_EVENT_ID_norm"], how="left")
    shot_text = out["SHOT_TYPE"].astype(str)
    out["shot_type"] = np.where(
        shot_text.str.contains("3PT", na=False),
        "3PA",
        np.where(shot_text.str.contains("2PT", na=False), "2PA", np.nan),
    )
    return out


def overall_late_tail(rows: pd.DataFrame) -> pd.DataFrame:
    mask = (
        pd.to_numeric(rows["next_time_left_game"], errors="coerce").ge(0)
        & pd.to_numeric(rows["next_time_left_game"], errors="coerce").le(45)
        & pd.to_numeric(rows["next_score_diff"], errors="coerce").abs().le(7)
        & ~pd.to_numeric(rows["next_is_terminal"], errors="coerce").eq(1)
    )
    out = []
    for model, g in rows.loc[mask].groupby("model", sort=False):
        out.append(
            {
                "diagnostic": "overall_late_tail_next_state",
                "model": model,
                **metric_row(g["final_home_win"], g["wp_next"]),
            }
        )
    return pd.DataFrame(out)


def overall_late_tail_by_season(rows: pd.DataFrame) -> pd.DataFrame:
    mask = (
        pd.to_numeric(rows["next_time_left_game"], errors="coerce").ge(0)
        & pd.to_numeric(rows["next_time_left_game"], errors="coerce").le(45)
        & pd.to_numeric(rows["next_score_diff"], errors="coerce").abs().le(7)
        & ~pd.to_numeric(rows["next_is_terminal"], errors="coerce").eq(1)
    )
    out = []
    for (model, season), g in rows.loc[mask].groupby(["model", "season"], sort=True):
        out.append(
            {
                "diagnostic": "overall_late_tail_next_state",
                "model": model,
                "season": int(season),
                **metric_row(g["final_home_win"], g["wp_next"]),
            }
        )
    return pd.DataFrame(out)


def final_surface_coarse_sanity(rows: pd.DataFrame) -> pd.DataFrame:
    surface = rows[rows["model"].eq("final_surface")].copy()
    terminal = pd.to_numeric(surface["next_is_terminal"], errors="coerce").eq(1)
    time = pd.to_numeric(surface["next_time_left_game"], errors="coerce")
    out = []
    for lo, hi in COARSE_SANITY_BINS:
        mask = time.gt(lo) & time.le(hi) & ~terminal
        g = surface.loc[mask]
        row = metric_row(g["final_home_win"], g["wp_next"])
        row.update(
            {
                "diagnostic": "coarse_45_300_sanity_next_state",
                "model": "final_surface",
                "time_bin": f"{int(lo)}-{int(hi)}s",
                "publication_role": "reviewer_response_holdout",
            }
        )
        out.append(row)
    return pd.DataFrame(out)


def final_surface_coarse_by_shot_type(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    surface = rows[rows["model"].eq("final_surface") & rows["shot_type"].isin(SHOT_TYPES)].copy()
    terminal = pd.to_numeric(surface["next_is_terminal"], errors="coerce").eq(1)
    time = pd.to_numeric(surface["next_time_left_game"], errors="coerce")
    long_rows = []
    for bin_order, (lo, hi) in enumerate(COARSE_SANITY_BINS, start=1):
        mask = time.gt(lo) & time.le(hi) & ~terminal
        for shot_type, g in surface.loc[mask].groupby("shot_type", sort=True):
            row = metric_row(g["offense_win"], g["offense_wp_next"])
            row.update(
                {
                    "diagnostic": "coarse_45_300_sanity_next_state_by_shot_type",
                    "model": "final_surface",
                    "orientation": "offense",
                    "bin_order": bin_order,
                    "time_bin": f"{int(lo)}-{int(hi)}s",
                    "shot_type": shot_type,
                    "publication_role": "reviewer_response_holdout",
                }
            )
            long_rows.append(row)
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, long_df

    wide = long_df.pivot_table(
        index=["diagnostic", "model", "orientation", "bin_order", "time_bin", "publication_role"],
        columns="shot_type",
        values=["n", "mean_fitted_wp", "empirical_win_rate", "gap"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{shot_type}" for metric, shot_type in wide.columns]
    wide = wide.reset_index().sort_values("bin_order")
    if "gap_3PA" in wide.columns and "gap_2PA" in wide.columns:
        wide["delta_cal_gap_3pa_minus_2pa"] = wide["gap_3PA"] - wide["gap_2PA"]
        wide["abs_delta_cal_gap"] = wide["delta_cal_gap_3pa_minus_2pa"].abs()
    return long_df, wide


def down3_made(rows: pd.DataFrame) -> pd.DataFrame:
    mask = (
        pd.to_numeric(rows["before_time_left_game"], errors="coerce").gt(0)
        & pd.to_numeric(rows["before_time_left_game"], errors="coerce").le(45)
        & pd.to_numeric(rows["offense_score_diff_before"], errors="coerce").eq(-3)
        & pd.to_numeric(rows["shot_made"], errors="coerce").eq(1)
        & ~pd.to_numeric(rows["next_is_terminal"], errors="coerce").eq(1)
        & rows["shot_type"].isin(SHOT_TYPES)
    )
    out = []
    for (model, shot_type), g in rows.loc[mask].groupby(["model", "shot_type"], sort=True):
        row = metric_row(g["offense_win"], g["offense_wp_next"])
        row.update(
            {
                "diagnostic": "targeted_down3_made_0_45",
                "model": model,
                "shot_type": shot_type,
                "n_games": int(g["GAME_ID_norm"].nunique()),
            }
        )
        out.append(row)
    return pd.DataFrame(out)


def down3_made_time(rows: pd.DataFrame) -> pd.DataFrame:
    base = (
        pd.to_numeric(rows["before_time_left_game"], errors="coerce").gt(0)
        & pd.to_numeric(rows["before_time_left_game"], errors="coerce").le(60)
        & pd.to_numeric(rows["offense_score_diff_before"], errors="coerce").eq(-3)
        & pd.to_numeric(rows["shot_made"], errors="coerce").eq(1)
        & ~pd.to_numeric(rows["next_is_terminal"], errors="coerce").eq(1)
        & rows["shot_type"].isin(SHOT_TYPES)
    )
    out = []
    time = pd.to_numeric(rows["before_time_left_game"], errors="coerce")
    for lo, hi in TIME_BINS:
        if lo == 0:
            time_mask = time.gt(lo) & time.le(hi)
        else:
            time_mask = time.gt(lo) & time.le(hi)
        for (model, shot_type), g in rows.loc[base & time_mask].groupby(
            ["model", "shot_type"], sort=True
        ):
            row = metric_row(g["offense_win"], g["offense_wp_next"])
            row.update(
                {
                    "diagnostic": "targeted_down3_made_time_band",
                    "model": model,
                    "time_bin": f"{int(lo)}-{int(hi)}s",
                    "negative_control": bool(lo >= 45),
                    "shot_type": shot_type,
                    "n_games": int(g["GAME_ID_norm"].nunique()),
                }
            )
            out.append(row)
    return pd.DataFrame(out)


def down3_le5_stress(rows: pd.DataFrame) -> pd.DataFrame:
    mask = (
        pd.to_numeric(rows["before_time_left_game"], errors="coerce").gt(0)
        & pd.to_numeric(rows["before_time_left_game"], errors="coerce").le(5)
        & pd.to_numeric(rows["offense_score_diff_before"], errors="coerce").eq(-3)
        & pd.to_numeric(rows["shot_made"], errors="coerce").eq(1)
        & ~pd.to_numeric(rows["next_is_terminal"], errors="coerce").eq(1)
        & rows["shot_type"].isin(SHOT_TYPES)
    )
    out = []
    for (model, shot_type), g in rows.loc[mask].groupby(["model", "shot_type"], sort=True):
        row = metric_row(g["offense_win"], g["offense_wp_next"])
        row.update(
            {
                "diagnostic": "stress_down3_made_le5",
                "model": model,
                "shot_type": shot_type,
                "n_games": int(g["GAME_ID_norm"].nunique()),
            }
        )
        out.append(row)
    return pd.DataFrame(out)


def add_surface_improvement(long_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    keep_metrics = [
        "n",
        "mean_fitted_wp",
        "empirical_win_rate",
        "gap",
        "brier",
        "log_loss",
        "ece",
        "mce",
        "calibration_intercept",
        "calibration_slope",
    ]
    wide = long_df.pivot_table(index=keys, columns="model", values=keep_metrics, aggfunc="first")
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    if "gap_baseline_m0" in wide.columns and "gap_final_surface" in wide.columns:
        wide["abs_gap_change_surface_minus_baseline"] = (
            wide["gap_final_surface"].abs() - wide["gap_baseline_m0"].abs()
        )
        wide["gap_change_surface_minus_baseline"] = wide["gap_final_surface"] - wide["gap_baseline_m0"]
    if "brier_baseline_m0" in wide.columns and "brier_final_surface" in wide.columns:
        wide["brier_change_surface_minus_baseline"] = (
            wide["brier_final_surface"] - wide["brier_baseline_m0"]
        )
    if "log_loss_baseline_m0" in wide.columns and "log_loss_final_surface" in wide.columns:
        wide["log_loss_change_surface_minus_baseline"] = (
            wide["log_loss_final_surface"] - wide["log_loss_baseline_m0"]
        )
    return wide


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote: {path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-wp",
        type=Path,
        default=Path("data/wp/shot_decision_states_2000_2024_rs_with_wp_baseline_m0.csv.gz"),
    )
    parser.add_argument(
        "--surface-wp",
        type=Path,
        default=Path("data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz"),
    )
    parser.add_argument("--shotdetail-dir", type=Path, default=Path("data/nba_raw"))
    parser.add_argument("--outdir", type=Path, default=Path("results/wp_calibration/final_wp_audit"))
    args = parser.parse_args()

    models = [
        ModelInput("baseline_m0", args.baseline_wp),
        ModelInput("final_surface", args.surface_wp),
    ]
    rows = []
    for model in models:
        if not model.path.exists():
            raise SystemExit(f"Missing WP-scored input for {model.label}: {model.path}")
        rows.append(read_audit_rows(model))
    audit_rows = pd.concat(rows, ignore_index=True)
    audit_rows = attach_shot_type(audit_rows, args.shotdetail_dir)

    overall = overall_late_tail(audit_rows)
    overall_by_season = overall_late_tail_by_season(audit_rows)
    coarse_sanity = final_surface_coarse_sanity(audit_rows)
    coarse_by_shot, coarse_by_shot_diff = final_surface_coarse_by_shot_type(audit_rows)
    targeted = down3_made(audit_rows)
    time_band = down3_made_time(audit_rows)
    stress = down3_le5_stress(audit_rows)

    write_csv(overall, args.outdir / "overall_late_tail_calibration_long.csv")
    write_csv(
        add_surface_improvement(overall, ["diagnostic"]),
        args.outdir / "overall_late_tail_calibration_comparison.csv",
    )
    write_csv(overall_by_season, args.outdir / "overall_late_tail_calibration_by_season_long.csv")
    write_csv(
        add_surface_improvement(overall_by_season, ["diagnostic", "season"]),
        args.outdir / "overall_late_tail_calibration_by_season_comparison.csv",
    )
    write_csv(
        coarse_sanity,
        args.outdir / "final_surface_coarse_45_300_sanity_next_state.csv",
    )
    write_csv(
        coarse_by_shot,
        args.outdir / "final_surface_coarse_45_300_sanity_next_state_by_shot_type_long.csv",
    )
    write_csv(
        coarse_by_shot_diff,
        args.outdir / "final_surface_coarse_45_300_sanity_next_state_by_shot_type_diff.csv",
    )
    write_csv(targeted, args.outdir / "targeted_down3_made_calibration_long.csv")
    write_csv(
        add_surface_improvement(targeted, ["diagnostic", "shot_type"]),
        args.outdir / "targeted_down3_made_calibration_comparison.csv",
    )
    write_csv(time_band, args.outdir / "targeted_down3_made_time_band_long.csv")
    write_csv(
        add_surface_improvement(time_band, ["diagnostic", "time_bin", "negative_control", "shot_type"]),
        args.outdir / "targeted_down3_made_time_band_comparison.csv",
    )
    write_csv(stress, args.outdir / "stress_down3_made_le5_long.csv")
    write_csv(
        add_surface_improvement(stress, ["diagnostic", "shot_type"]),
        args.outdir / "stress_down3_made_le5_comparison.csv",
    )

    print("\nOverall late-tail calibration:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nFinal-surface coarse 45-300s sanity check:")
    print(coarse_sanity.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nFinal-surface coarse 45-300s differential sanity check:")
    print(coarse_by_shot_diff.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTargeted down-3 made-shot calibration:")
    print(targeted.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
