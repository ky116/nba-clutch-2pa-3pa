#!/usr/bin/env python3
"""Create equal-weight averaged CATE surface from multiple parquet surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Equal-weight average for CATE surfaces.")
    p.add_argument("--surface", action="append", required=True, help="Input tau surface parquet. Repeat for each model.")
    p.add_argument("--label", action="append", default=None, help="Optional label for each input surface.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--prefix", default="", help="Output file prefix.")
    p.add_argument("--time-col", default="time_left_game")
    p.add_argument("--score-col", default="score_diff")
    p.add_argument("--score-label", default=None, help="Optional x-axis label override.")
    p.add_argument("--tau-col", default="tau_mean")
    p.add_argument("--cell-count-csv", default=None, help="Optional cell-count csv from plot_cate_cell_counts.py")
    p.add_argument("--min-cell-count", type=int, default=0, help="Mask cells with cell_count below this threshold in plots.")
    p.add_argument("--score-lo", type=float, default=-10.0, help="Fixed lower x-axis bound for score_diff plots.")
    p.add_argument("--score-hi", type=float, default=10.0, help="Fixed upper x-axis bound for score_diff plots.")
    p.add_argument("--score-tick-step", type=float, default=2.0, help="Tick spacing for score_diff plots.")
    p.add_argument("--plot", action="store_true", help="Also save intermediate tau mean/std diagnostic PNGs.")
    return p.parse_args()


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    title: str | None,
    time_col: str,
    score_col: str,
    cmap_name: str = "viridis",
    symmetric: bool = False,
    min_cell_count: int = 0,
    cell_count_col: str = "cell_count",
    figsize: tuple[float, float] = (10, 6),
    score_lo: float | None = None,
    score_hi: float | None = None,
    score_tick_step: float | None = None,
    score_label: str | None = None,
    colorbar_label: str | None = None,
) -> None:
    time_vals = np.sort(df[time_col].unique())
    score_vals = np.sort(df[score_col].unique())
    plot_df = df.copy()
    if min_cell_count > 0 and cell_count_col in plot_df.columns:
        plot_df.loc[plot_df[cell_count_col] < min_cell_count, value_col] = np.nan
    arr = (
        plot_df.pivot(index=time_col, columns=score_col, values=value_col)
        .reindex(index=time_vals, columns=score_vals)
        .to_numpy(dtype=float)
    )
    arr = np.ma.masked_invalid(arr)
    plt.figure(figsize=figsize)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")
    im_kwargs = dict(
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[float(score_vals.min()), float(score_vals.max()), float(time_vals.min()), float(time_vals.max())],
        cmap=cmap,
    )
    if symmetric:
        vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(np.nanmax(np.abs(arr))) else 1.0
        vmax = max(vmax, 1e-8)
        im_kwargs["vmin"] = -vmax
        im_kwargs["vmax"] = vmax

    im = plt.imshow(
        arr,
        **im_kwargs,
    )
    plt.colorbar(im, label=colorbar_label or value_col)
    if score_lo is not None and score_hi is not None:
        plt.xlim(float(score_lo), float(score_hi))
        score_tick_min = int(np.ceil(float(score_lo)))
        score_tick_max = int(np.floor(float(score_hi)))
    else:
        score_tick_min = int(np.floor(float(score_vals.min())))
        score_tick_max = int(np.ceil(float(score_vals.max())))
    tick_step = int(score_tick_step) if score_tick_step is not None and score_tick_step > 0 else 2
    score_ticks = np.arange(score_tick_min, score_tick_max + 1, tick_step, dtype=int)
    plt.xticks(score_ticks)
    if score_label is None and score_col == "score_diff":
        score_label = "score_diff_offense"
    plt.xlabel(score_label or score_col)
    plt.ylabel(time_col)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    surfaces = [Path(s) for s in args.surface]
    labels = list(args.label) if args.label else [p.parent.name for p in surfaces]
    if len(labels) != len(surfaces):
        raise ValueError("--label count must match --surface count.")
    if len(set(labels)) != len(labels):
        raise ValueError("--label must be unique.")
    if len(surfaces) < 2:
        raise ValueError("Need at least 2 surfaces.")

    key_cols = [args.time_col, args.score_col]
    merged: pd.DataFrame | None = None
    tau_cols: list[str] = []

    for path, label in zip(surfaces, labels):
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path)
        for col in key_cols + [args.tau_col]:
            if col not in df.columns:
                raise ValueError(f"{path} missing column: {col}")
        colname = f"tau_{label}"
        cur = df[key_cols + [args.tau_col]].rename(columns={args.tau_col: colname})
        merged = cur if merged is None else merged.merge(cur, on=key_cols, how="inner")
        tau_cols.append(colname)

    assert merged is not None
    if args.cell_count_csv:
        count_df = pd.read_csv(args.cell_count_csv)
        need_cols = key_cols + ["cell_count"]
        missing_cols = [c for c in need_cols if c not in count_df.columns]
        if missing_cols:
            raise ValueError(f"{args.cell_count_csv} missing column(s): {missing_cols}")
        merged = merged.merge(count_df[need_cols], on=key_cols, how="left")

    merged["tau_mean_ensemble"] = merged[tau_cols].mean(axis=1)
    merged["tau_std_across_models"] = merged[tau_cols].std(axis=1, ddof=0)
    merged["tau_sign_agree_ratio"] = (
        np.vstack([(merged[c].to_numpy() > 0).astype(float) for c in tau_cols]).T.mean(axis=1)
    )
    merged["tau_sign_disagree"] = ((merged["tau_sign_agree_ratio"] > 0.0) & (merged["tau_sign_agree_ratio"] < 1.0)).astype(int)

    out_csv = outdir / f"{args.prefix}cate_surface_equal_weight.csv"
    out_parquet = outdir / f"{args.prefix}cate_surface_equal_weight.parquet"
    merged.sort_values(key_cols).to_csv(out_csv, index=False)
    merged.sort_values(key_cols).to_parquet(out_parquet, index=False)

    if args.plot:
        out_png_tau = outdir / f"{args.prefix}cate_surface_equal_weight_tau_mean.png"
        out_png_tau_portrait = outdir / f"{args.prefix}cate_surface_equal_weight_tau_mean_portrait.png"
        out_png_std = outdir / f"{args.prefix}cate_surface_equal_weight_tau_std_across_models.png"
        out_png_std_portrait = outdir / f"{args.prefix}cate_surface_equal_weight_tau_std_across_models_portrait.png"
        plot_heatmap(
            merged,
            value_col="tau_mean_ensemble",
            out_path=out_png_tau,
            title=None,
            time_col=args.time_col,
            score_col=args.score_col,
            cmap_name="viridis",
            symmetric=False,
            min_cell_count=args.min_cell_count,
            score_lo=args.score_lo,
            score_hi=args.score_hi,
            score_tick_step=args.score_tick_step,
            score_label=args.score_label,
            colorbar_label="Recalibrated CATE",
        )
        plot_heatmap(
            merged,
            value_col="tau_mean_ensemble",
            out_path=out_png_tau_portrait,
            title=None,
            time_col=args.time_col,
            score_col=args.score_col,
            cmap_name="viridis",
            symmetric=False,
            min_cell_count=args.min_cell_count,
            figsize=(6, 10),
            score_lo=args.score_lo,
            score_hi=args.score_hi,
            score_tick_step=args.score_tick_step,
            score_label=args.score_label,
            colorbar_label="Recalibrated CATE",
        )
        plot_heatmap(
            merged,
            value_col="tau_std_across_models",
            out_path=out_png_std,
            title=f"Tau Std Across Models ({len(tau_cols)} models)",
            time_col=args.time_col,
            score_col=args.score_col,
            cmap_name="magma",
            symmetric=False,
            min_cell_count=args.min_cell_count,
            score_lo=args.score_lo,
            score_hi=args.score_hi,
            score_tick_step=args.score_tick_step,
            score_label=args.score_label,
        )
        plot_heatmap(
            merged,
            value_col="tau_std_across_models",
            out_path=out_png_std_portrait,
            title=f"Tau Std Across Models ({len(tau_cols)} models)",
            time_col=args.time_col,
            score_col=args.score_col,
            cmap_name="magma",
            symmetric=False,
            min_cell_count=args.min_cell_count,
            figsize=(6, 10),
            score_lo=args.score_lo,
            score_hi=args.score_hi,
            score_tick_step=args.score_tick_step,
            score_label=args.score_label,
        )

    summary = {
        "n_models": len(tau_cols),
        "n_cells": int(len(merged)),
        "min_cell_count_threshold_for_plot": int(args.min_cell_count),
        "tau_mean_abs_mean": float(np.nanmean(np.abs(merged["tau_mean_ensemble"]))),
        "tau_std_mean": float(np.nanmean(merged["tau_std_across_models"])),
        "sign_disagree_ratio": float(np.nanmean(merged["tau_sign_disagree"])),
    }
    summary_path = outdir / f"{args.prefix}cate_surface_equal_weight_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_parquet}")
    if args.plot:
        print(f"[saved] {out_png_tau}")
        print(f"[saved] {out_png_tau_portrait}")
        print(f"[saved] {out_png_std}")
        print(f"[saved] {out_png_std_portrait}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
