#!/usr/bin/env python3
"""Plot exact-second observed 3P-minus-2P ΔWP where both groups have n>=30."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path(
    "results/late_clock_diagnostics/shot_level_delta_wp_distribution/"
    "observed_delta_wp_surface_t0_15.csv"
)
OUTPUT = Path(
    "results/late_clock_diagnostics/shot_level_delta_wp_distribution/"
    "observed_delta_wp_difference_3p_minus_2p_t0_15_exact_n30.png"
)


def main() -> None:
    data = pd.read_csv(INPUT)
    times = np.arange(0, 16)
    scores = np.arange(-10, 11)
    supported = (
        data["count_three-point"].ge(30)
        & data["count_two-point"].ge(30)
    )
    shown = data.copy()
    shown.loc[~supported, "mean_delta_wp_diff_3p_minus_2p"] = np.nan
    matrix = (
        shown.pivot(
            index="time_sec",
            columns="score",
            values="mean_delta_wp_diff_3p_minus_2p",
        )
        .reindex(index=times, columns=scores)
        .to_numpy(dtype=float)
    )

    fig, ax = plt.subplots(figsize=(9.4, 6.2), dpi=180)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[-10.5, 10.5, -0.5, 15.5],
        cmap=cmap,
        vmin=-0.2,
        vmax=0.2,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("ΔWP", fontsize=14)
    colorbar.ax.tick_params(labelsize=11)

    ax.set_title("Observed difference: 3P − 2P", fontsize=18, pad=12)
    ax.set_xlabel("Score differential (offense perspective)", fontsize=14)
    ax.set_ylabel("Seconds remaining", fontsize=14)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_yticks(np.arange(0, 16, 3))
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")
    print(f"Displayed cells: {int(supported.sum())}/{len(data)} (3P n>=30 and 2P n>=30)")


if __name__ == "__main__":
    main()
