#!/usr/bin/env python3
"""Plot per-cell sample counts on an existing CATE surface grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot sample counts on an existing CATE surface grid.")
    p.add_argument("--data", required=True, help="Input parquet/csv with raw rows.")
    p.add_argument("--surface-csv", required=True, help="Existing surface csv defining time/score grid.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--prefix", default="", help="Output file prefix.")
    p.add_argument("--time-col", default="time_left_game")
    p.add_argument("--score-col", default="score_diff")
    p.add_argument("--title", default=None)
    p.add_argument("--plot", action="store_true", help="Also save cell-count diagnostic PNGs.")
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def assign_nearest(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.abs(values[:, None] - grid[None, :]).argmin(axis=1)
    return grid[idx]


def cell_widths(grid: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("grid must be 1D with at least 2 points")
    mids = (grid[:-1] + grid[1:]) / 2.0
    edges = np.empty(len(grid) + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = float(grid[0] if lo is None else lo)
    edges[-1] = float(grid[-1] if hi is None else hi)
    return np.diff(edges)


def plot_heatmap(df: pd.DataFrame, value_col: str, out_path: Path, title: str, time_col: str, score_col: str) -> None:
    time_vals = np.sort(df[time_col].unique())
    score_vals = np.sort(df[score_col].unique())
    arr = (
        df.pivot(index=time_col, columns=score_col, values=value_col)
        .reindex(index=time_vals, columns=score_vals)
        .to_numpy(dtype=float)
    )
    arr = np.ma.masked_invalid(arr)

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(color="white")
    im = plt.imshow(
        arr,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[float(score_vals.min()), float(score_vals.max()), float(time_vals.min()), float(time_vals.max())],
        cmap=cmap,
    )
    plt.colorbar(im, label=value_col)
    plt.xlabel(score_col)
    plt.ylabel(time_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    surface_path = Path(args.surface_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_table(data_path)
    surface = pd.read_csv(surface_path)

    for col in [args.time_col, args.score_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column in data: {col}")
        if col not in surface.columns:
            raise ValueError(f"Missing column in surface csv: {col}")

    time_grid = np.sort(surface[args.time_col].dropna().unique().astype(float))
    score_grid = np.sort(surface[args.score_col].dropna().unique().astype(float))
    time_lo = float(time_grid.min())
    time_hi = float(time_grid.max())
    score_lo = float(score_grid.min())
    score_hi = float(score_grid.max())

    use = df[
        df[args.time_col].between(time_lo, time_hi, inclusive="both")
        & df[args.score_col].between(score_lo, score_hi, inclusive="both")
    ][[args.time_col, args.score_col]].copy()
    if use.empty:
        raise ValueError("No rows fall within the selected surface bounds.")

    use[args.time_col] = assign_nearest(use[args.time_col].to_numpy(dtype=float), time_grid)
    use[args.score_col] = use[args.score_col].round().astype(float)

    counts = (
        use.groupby([args.time_col, args.score_col], as_index=False)
        .size()
        .rename(columns={"size": "cell_count"})
    )

    out = surface[[args.time_col, args.score_col]].drop_duplicates().copy()
    out = out.merge(counts, on=[args.time_col, args.score_col], how="left")
    out["cell_count"] = out["cell_count"].fillna(0).astype(int)
    out["cell_count_log1p"] = np.log1p(out["cell_count"].astype(float))
    time_width_map = dict(zip(time_grid.tolist(), cell_widths(time_grid, lo=time_lo, hi=time_hi).tolist()))
    out["time_cell_width"] = out[args.time_col].map(time_width_map).astype(float)
    out["cell_density"] = out["cell_count"] / out["time_cell_width"]
    out["cell_density_log1p"] = np.log1p(out["cell_density"])

    csv_path = outdir / f"{args.prefix}cate_surface_cell_counts.csv"
    png_path = outdir / f"{args.prefix}cate_surface_cell_counts_log1p.png"
    density_png_path = outdir / f"{args.prefix}cate_surface_cell_density_log1p.png"
    summary_path = outdir / f"{args.prefix}cate_surface_cell_counts_summary.csv"
    out.sort_values([args.time_col, args.score_col]).to_csv(csv_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "n_rows_in_window": int(len(use)),
                "n_cells": int(len(out)),
                "n_nonzero_cells": int((out["cell_count"] > 0).sum()),
                "zero_cell_ratio": float((out["cell_count"] == 0).mean()),
                "min_nonzero_count": int(out.loc[out["cell_count"] > 0, "cell_count"].min()) if (out["cell_count"] > 0).any() else 0,
                "median_nonzero_count": float(out.loc[out["cell_count"] > 0, "cell_count"].median()) if (out["cell_count"] > 0).any() else 0.0,
                "max_count": int(out["cell_count"].max()),
                "median_nonzero_density": float(out.loc[out["cell_count"] > 0, "cell_density"].median()) if (out["cell_count"] > 0).any() else 0.0,
                "max_density": float(out["cell_density"].max()),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    if args.plot:
        title = args.title or "CATE Surface Cell Counts (log1p)"
        plot_heatmap(out, "cell_count_log1p", png_path, title, args.time_col, args.score_col)
        plot_heatmap(
            out,
            "cell_density_log1p",
            density_png_path,
            title.replace("Counts", "Density") if "Counts" in title else f"{title} Density",
            args.time_col,
            args.score_col,
        )

    print(f"[saved] {csv_path}")
    if args.plot:
        print(f"[saved] {png_path}")
        print(f"[saved] {density_png_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
