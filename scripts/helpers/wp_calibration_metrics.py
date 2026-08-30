#!/usr/bin/env python
"""Compute WP calibration metrics from WP-scored shot-state rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-6


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
    grouped = df.groupby("_bin", observed=True).agg(n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
    gap = (grouped["pred"] - grouped["obs"]).abs()
    ece = float((gap * grouped["n"]).sum() / grouped["n"].sum())
    mce = float(gap.max())
    return ece, mce


def summarize(y: pd.Series, p: pd.Series, n_bins: int) -> dict[str, float | int]:
    df = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "p": clip_prob(p)}).dropna()
    df = df[df["y"].isin([0, 1])]
    n = int(len(df))
    if n == 0:
        return {
            "n": 0,
            "mean_pred": np.nan,
            "empirical_win_rate": np.nan,
            "bias": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "ece": np.nan,
            "mce": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }
    brier = float(np.mean((df["p"] - df["y"]) ** 2))
    log_loss = float(-np.mean(df["y"] * np.log(df["p"]) + (1 - df["y"]) * np.log(1 - df["p"])))
    ece, mce = ece_metrics(df["y"], df["p"], n_bins)
    intercept, slope = logistic_calibration(df["y"], df["p"])
    mean_pred = float(df["p"].mean())
    empirical = float(df["y"].mean())
    return {
        "n": n,
        "mean_pred": mean_pred,
        "empirical_win_rate": empirical,
        "bias": mean_pred - empirical,
        "brier": brier,
        "log_loss": log_loss,
        "ece": ece,
        "mce": mce,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def add_orientation(df: pd.DataFrame, target: str, offense_oriented: bool) -> tuple[pd.Series, pd.Series]:
    pred_col = "wp_before" if target == "before" else "wp_next"
    p = clip_prob(df[pred_col])
    y = pd.to_numeric(df["final_home_win"], errors="coerce")
    if not offense_oriented:
        return y, p
    poss_col = "before_home_possession" if target == "before" else "next_home_possession"
    home_poss = pd.to_numeric(df[poss_col], errors="coerce")
    y_off = pd.Series(np.where(home_poss.eq(1), y, np.where(home_poss.eq(0), 1 - y, np.nan)), index=df.index)
    p_off = pd.Series(np.where(home_poss.eq(1), p, np.where(home_poss.eq(0), 1 - p, np.nan)), index=df.index)
    return y_off, p_off


def filter_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df
    if args.start_season is not None:
        out = out[pd.to_numeric(out["season"], errors="coerce").ge(args.start_season)]
    if args.end_season is not None:
        out = out[pd.to_numeric(out["season"], errors="coerce").le(args.end_season)]
    if args.target == "next" and args.exclude_terminal_next:
        out = out[~pd.to_numeric(out["next_is_terminal"], errors="coerce").eq(1)]
    time_col = "before_time_left_game" if args.target == "before" else "next_time_left_game"
    score_col = "before_score_diff" if args.target == "before" else "next_score_diff"
    if args.max_time is not None:
        out = out[pd.to_numeric(out[time_col], errors="coerce").le(args.max_time)]
    if args.min_time is not None:
        out = out[pd.to_numeric(out[time_col], errors="coerce").ge(args.min_time)]
    if args.score_abs is not None:
        out = out[pd.to_numeric(out[score_col], errors="coerce").abs().le(args.score_abs)]
    return out.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-wp", type=Path, default=Path("data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz"))
    parser.add_argument("--target", choices=["before", "next"], default="next")
    parser.add_argument("--offense-oriented", action="store_true")
    parser.add_argument("--exclude-terminal-next", action="store_true", default=True)
    parser.add_argument("--include-terminal-next", dest="exclude_terminal_next", action="store_false")
    parser.add_argument("--min-time", type=float, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--score-abs", type=float, default=None)
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/wp_calibration_metrics.csv"))
    args = parser.parse_args()

    usecols = [
        "season",
        "final_home_win",
        "wp_before",
        "wp_next",
        "next_is_terminal",
        "before_home_possession",
        "next_home_possession",
        "before_time_left_game",
        "next_time_left_game",
        "before_score_diff",
        "next_score_diff",
    ]
    df = pd.read_csv(args.with_wp, usecols=usecols)
    df = filter_rows(df, args)
    y, p = add_orientation(df, args.target, args.offense_oriented)
    row = {
        "with_wp": str(args.with_wp),
        "target": args.target,
        "orientation": "offense" if args.offense_oriented else "home",
        "exclude_terminal_next": bool(args.exclude_terminal_next) if args.target == "next" else np.nan,
        "min_time": args.min_time,
        "max_time": args.max_time,
        "score_abs": args.score_abs,
        "start_season": args.start_season,
        "end_season": args.end_season,
        **summarize(y, p, args.bins),
    }
    result = pd.DataFrame([row])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
