#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTDIR = Path("results/figure_source_data")
X_LABEL = "Score differential (offense perspective)"
Y_LABEL = "Time remaining (seconds)"
COLORBAR_LABEL = "Recalibrated CATE"


def draw_heatmap(source_path: Path, out_path: Path) -> None:
    df = pd.read_csv(source_path)
    plot_df = df.copy()
    if "shown_in_figure" in plot_df.columns:
        shown = plot_df["shown_in_figure"]
        if shown.dtype == object:
            shown = shown.astype(str).str.lower().map({"true": True, "false": False})
        else:
            shown = shown.astype(bool)
        plot_df.loc[~shown.fillna(False), "cate_value"] = np.nan
    if "masked_n50" in out_path.name and "cell_count" in plot_df.columns:
        plot_df.loc[plot_df["cell_count"] < 50, "cate_value"] = np.nan

    time_vals = np.sort(plot_df["time_left_game"].unique())
    score_vals = np.sort(plot_df["score_diff"].unique())
    arr = (
        plot_df.pivot(index="time_left_game", columns="score_diff", values="cate_value")
        .reindex(index=time_vals, columns=score_vals)
        .to_numpy(dtype=float)
    )
    arr = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            float(score_vals.min()),
            float(score_vals.max()),
            float(time_vals.min()),
            float(time_vals.max()),
        ],
        cmap=cmap,
    )
    ax.set_xlim(-10, 10)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    fig.colorbar(im, ax=ax, label=COLORBAR_LABEL)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw manuscript CATE surface figures from source CSVs.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory containing figure2/figure3 CATE source CSVs and receiving PNGs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    figure_specs = [
        (
            outdir / "figure2_full_data_t30_300_cate_surface_source_data.csv",
            outdir / "figure2_full_data_t30_300_cate_surface.png",
        ),
        (
            outdir / "figures1_cate_surface_0_30s_masked_n50_source_data.csv",
            outdir / "figures1_cate_surface_0_30s_masked_n50.png",
        ),
    ]
    for source_path, out_path in figure_specs:
        draw_heatmap(source_path, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
