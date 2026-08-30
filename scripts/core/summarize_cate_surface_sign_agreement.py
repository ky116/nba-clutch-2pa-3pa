#!/usr/bin/env python3
"""Summarize sign agreement across multiple CATE surface parquet files.

Example:
  .venv/bin/python summarize_cate_surface_sign_agreement.py \
    --surface results/full_data_catboost_state_fixed_loso/full_data_t30_300_tau_surface_three-point_vs_two-point.parquet \
    --surface results/full_data_xgb_state_fixed_loso/full_data_t30_300_tau_surface_three-point_vs_two-point.parquet \
    --surface results/full_data_lgbm_state_fixed_loso/full_data_t30_300_tau_surface_three-point_vs_two-point.parquet \
    --label catboost --label xgb --label lgbm \
    --outdir results/cate_surface_agreement \
    --prefix full_data_t30_300_
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CATE surface sign-agreement summary.")
    p.add_argument("--surface", action="append", required=True, help="Input tau-surface parquet path. Repeat for each model.")
    p.add_argument("--label", action="append", default=None, help="Model label for each --surface. Must match count if provided.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--prefix", default="", help="Output file prefix.")
    p.add_argument("--time-col", default="time_left_game", help="Time column name.")
    p.add_argument("--score-col", default="score_diff_offense", help="Score-diff column name.")
    p.add_argument("--tau-col", default="tau_mean", help="Tau estimate column.")
    p.add_argument("--eps", type=float, default=0.0, help="Neutral zone threshold for sign: [-eps, eps] -> 0.")
    p.add_argument("--score-lo", type=float, default=-10.0, help="Fixed lower x-axis bound for score_diff plots.")
    p.add_argument("--score-hi", type=float, default=10.0, help="Fixed upper x-axis bound for score_diff plots.")
    p.add_argument("--score-tick-step", type=float, default=2.0, help="Tick spacing for score_diff plots.")
    p.add_argument("--plot", action="store_true", help="Also save sign-agreement diagnostic PNGs.")
    return p.parse_args()


def _to_sign(values: np.ndarray, eps: float) -> np.ndarray:
    return np.where(values > eps, 1, np.where(values < -eps, -1, 0))


def _resolve_score_col(df: pd.DataFrame, requested_score_col: str) -> pd.DataFrame:
    if requested_score_col in df.columns:
        return df
    if requested_score_col == "score_diff_offense" and "score_diff" in df.columns:
        out = df.copy()
        out["score_diff_offense"] = out["score_diff"]
        return out
    return df


def _pivot_for_plot(df: pd.DataFrame, value_col: str, time_col: str, score_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Match plot_cate_surface_gcomp.py: x=score_diff, y=time_left_game
    time_vals = np.sort(df[time_col].unique())
    score_vals = np.sort(df[score_col].unique())
    piv = df.pivot(index=time_col, columns=score_col, values=value_col).reindex(index=time_vals, columns=score_vals)
    return piv.to_numpy(dtype=float), time_vals, score_vals


def _plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    time_col: str,
    score_col: str,
    vmin: float,
    vmax: float,
    score_lo: float | None = None,
    score_hi: float | None = None,
    score_tick_step: float | None = None,
) -> None:
    arr, time_vals, score_vals = _pivot_for_plot(df, value_col, time_col, score_col)
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        arr,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=[score_vals.min(), score_vals.max(), time_vals.min(), time_vals.max()],
    )
    plt.colorbar(im, label=value_col)
    if score_lo is not None and score_hi is not None:
        plt.xlim(float(score_lo), float(score_hi))
        score_tick_min = int(np.ceil(float(score_lo)))
        score_tick_max = int(np.floor(float(score_hi)))
    else:
        score_tick_min = int(np.floor(float(score_vals.min())))
        score_tick_max = int(np.ceil(float(score_vals.max())))
    tick_step = int(score_tick_step) if score_tick_step is not None and score_tick_step > 0 else 2
    plt.xticks(np.arange(score_tick_min, score_tick_max + 1, tick_step, dtype=int))
    plt.xlabel(score_col)
    plt.ylabel(time_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    surfaces = [Path(s) for s in args.surface]
    labels = list(args.label) if args.label else [p.parent.name for p in surfaces]
    if len(labels) != len(surfaces):
        raise ValueError("--label count must match --surface count.")
    if len(set(labels)) != len(labels):
        raise ValueError("--label values must be unique.")
    if len(surfaces) < 2:
        raise ValueError("At least 2 --surface inputs are required.")

    key_cols = [args.time_col, args.score_col]
    merged: pd.DataFrame | None = None
    tau_cols: list[str] = []
    sign_cols: list[str] = []

    for path, label in zip(surfaces, labels):
        if not path.exists():
            raise FileNotFoundError(f"surface not found: {path}")
        df = pd.read_parquet(path)
        df = _resolve_score_col(df, args.score_col)
        missing = [c for c in key_cols + [args.tau_col] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        tau_col = f"tau_{label}"
        sign_col = f"sign_{label}"
        cur = df[key_cols + [args.tau_col]].copy()
        cur = cur.rename(columns={args.tau_col: tau_col})
        cur[sign_col] = _to_sign(cur[tau_col].to_numpy(dtype=float), eps=float(args.eps))
        tau_cols.append(tau_col)
        sign_cols.append(sign_col)
        merged = cur if merged is None else merged.merge(cur, on=key_cols, how="inner")

    assert merged is not None
    n_models = len(sign_cols)
    sign_mat = merged[sign_cols].to_numpy(dtype=int)

    merged["positive_share"] = (sign_mat == 1).mean(axis=1)
    merged["negative_share"] = (sign_mat == -1).mean(axis=1)
    merged["neutral_share"] = (sign_mat == 0).mean(axis=1)
    merged["agreement_ratio"] = merged[["positive_share", "negative_share", "neutral_share"]].max(axis=1)
    merged["is_unanimous"] = merged["agreement_ratio"] == 1.0
    merged["has_pos_neg_conflict"] = ((sign_mat == 1).any(axis=1)) & ((sign_mat == -1).any(axis=1))
    merged["mean_tau"] = merged[tau_cols].mean(axis=1)
    merged["std_tau"] = merged[tau_cols].std(axis=1, ddof=0)
    merged["majority_positive"] = merged["positive_share"] > 0.5
    merged["majority_negative"] = merged["negative_share"] > 0.5
    merged["majority_neutral"] = merged["neutral_share"] > 0.5
    merged["majority_sign"] = np.select(
        [merged["majority_positive"], merged["majority_negative"], merged["majority_neutral"]],
        [1, -1, 0],
        default=0,
    ).astype(int)

    grid_out = outdir / f"{args.prefix}cate_surface_sign_agreement.csv"
    grid_parquet_out = outdir / f"{args.prefix}cate_surface_sign_agreement.parquet"
    merged.sort_values(key_cols).to_csv(grid_out, index=False)
    merged.sort_values(key_cols).to_parquet(grid_parquet_out, index=False)

    summary = {
        "n_models": n_models,
        "n_cells": int(len(merged)),
        "mean_agreement_ratio": float(merged["agreement_ratio"].mean()),
        "median_agreement_ratio": float(merged["agreement_ratio"].median()),
        "unanimous_ratio": float(merged["is_unanimous"].mean()),
        "pos_neg_conflict_ratio": float(merged["has_pos_neg_conflict"].mean()),
        "majority_positive_ratio": float((merged["majority_sign"] == 1).mean()),
        "majority_negative_ratio": float((merged["majority_sign"] == -1).mean()),
        "majority_neutral_ratio": float((merged["majority_sign"] == 0).mean()),
        "eps": float(args.eps),
    }
    summary_df = pd.DataFrame([summary])
    summary_csv = outdir / f"{args.prefix}cate_surface_sign_agreement_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_txt = outdir / f"{args.prefix}cate_surface_sign_agreement_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("CATE Surface Sign Agreement Summary\n")
        f.write(f"n_models: {summary['n_models']}\n")
        f.write(f"n_cells: {summary['n_cells']}\n")
        f.write(f"mean_agreement_ratio: {summary['mean_agreement_ratio']:.6f}\n")
        f.write(f"median_agreement_ratio: {summary['median_agreement_ratio']:.6f}\n")
        f.write(f"unanimous_ratio: {summary['unanimous_ratio']:.6f}\n")
        f.write(f"pos_neg_conflict_ratio: {summary['pos_neg_conflict_ratio']:.6f}\n")
        f.write(f"majority_positive_ratio: {summary['majority_positive_ratio']:.6f}\n")
        f.write(f"majority_negative_ratio: {summary['majority_negative_ratio']:.6f}\n")
        f.write(f"majority_neutral_ratio: {summary['majority_neutral_ratio']:.6f}\n")
        f.write(f"eps: {summary['eps']:.6g}\n")
        f.write("\nlabels:\n")
        for label, path in zip(labels, surfaces):
            f.write(f"- {label}: {path}\n")

    if args.plot:
        agree_png = outdir / f"{args.prefix}cate_surface_agreement_ratio.png"
        pos_png = outdir / f"{args.prefix}cate_surface_positive_share.png"
        _plot_heatmap(
            merged,
            value_col="agreement_ratio",
            title=f"CATE Sign Agreement Ratio ({n_models} models)",
            out_path=agree_png,
            time_col=args.time_col,
            score_col=args.score_col,
            vmin=1.0 / n_models,
            vmax=1.0,
            score_lo=args.score_lo,
            score_hi=args.score_hi,
            score_tick_step=args.score_tick_step,
        )
        _plot_heatmap(
            merged,
            value_col="positive_share",
            title=f"CATE Positive Share ({n_models} models)",
            out_path=pos_png,
            time_col=args.time_col,
            score_col=args.score_col,
            vmin=0.0,
            vmax=1.0,
            score_lo=args.score_lo,
            score_hi=args.score_hi,
            score_tick_step=args.score_tick_step,
        )

    print(f"[saved] {grid_out}")
    print(f"[saved] {grid_parquet_out}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {summary_txt}")
    if args.plot:
        print(f"[saved] {agree_png}")
        print(f"[saved] {pos_png}")


if __name__ == "__main__":
    main()
