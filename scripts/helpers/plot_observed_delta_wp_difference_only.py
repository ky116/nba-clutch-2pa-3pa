#!/usr/bin/env python3
"""Plot only the observed mean delta-WP difference: 3P minus 2P."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="results/late_clock_diagnostics/shot_level_delta_wp_distribution/observed_delta_wp_surface_t0_30_binned_3sec_1point.csv",
    )
    p.add_argument(
        "--out",
        default="results/late_clock_diagnostics/shot_level_delta_wp_distribution/observed_delta_wp_difference_3p_minus_2p_t0_30.png",
    )
    p.add_argument(
        "--out-highlighted",
        default="results/late_clock_diagnostics/shot_level_delta_wp_distribution/observed_delta_wp_difference_3p_minus_2p_t0_30_forced_rule_highlight.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def draw(data: pd.DataFrame, out: Path, highlighted: bool, dpi: int) -> None:
    time_order = ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-30"]
    score_order = list(range(-10, 11))
    shown = data.copy()
    shown.loc[~shown["supported_diff_n30"].astype(bool), "mean_delta_wp_diff_3p_minus_2p"] = np.nan
    matrix = (
        shown.pivot(index="time_bin", columns="score", values="mean_delta_wp_diff_3p_minus_2p")
        .reindex(index=time_order, columns=score_order)
        .to_numpy(dtype=float)
        * 100.0
    )

    fig, ax = plt.subplots(figsize=(15.5, 5.8), dpi=180)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    limit = 6.0
    image = ax.imshow(
        matrix,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.018)
    cbar.set_label("Observed mean ΔWP difference (3P − 2P), percentage points", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(len(score_order)))
    ax.set_xticklabels(score_order, fontsize=11)
    ax.set_yticks(np.arange(len(time_order)))
    ax.set_yticklabels(time_order, fontsize=12, fontweight="bold")
    ax.set_xlabel("Score difference from offensive-team perspective", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time remaining (seconds)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Observed ΔWP difference: 3P minus 2P",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )

    ax.set_xticks(np.arange(-0.5, len(score_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(time_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if highlighted:
        # Forced-rule region: time <= 5 seconds and score_diff <= -3.
        score_last = score_order.index(-3)
        rect = Rectangle(
            (-0.5, -0.5),
            score_last + 1,
            2,
            fill=False,
            edgecolor="#ffd43b",
            linewidth=4,
            linestyle="-",
            zorder=5,
        )
        ax.add_patch(rect)
        ax.text(
            score_last + 0.45,
            0.5,
            "Forced-3P rule region",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#111111",
            bbox={"facecolor": "#fff3bf", "edgecolor": "#e0a800", "boxstyle": "round,pad=0.3"},
            zorder=6,
        )

    ax.text(
        0.0,
        -0.16,
        "Gray cells: fewer than 30 observed attempts in either shot category.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#555555",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    draw(data, Path(args.out), highlighted=False, dpi=args.dpi)
    draw(data, Path(args.out_highlighted), highlighted=True, dpi=args.dpi)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_highlighted}")


if __name__ == "__main__":
    main()
