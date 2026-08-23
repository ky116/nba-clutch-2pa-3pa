#!/usr/bin/env python3
"""Summarize and plot shot-level delta-WP distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUANTILES = [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-wp",
        default="data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz",
    )
    parser.add_argument(
        "--panel",
        default="data/analysis/shotchoice_panel_clutch_rs.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="results/late_clock_diagnostics/shot_level_delta_wp_distribution",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--bins", type=int, default=200)
    return parser.parse_args()


def read_all_shots(path: str, chunksize: int) -> pd.DataFrame:
    parts = []
    usecols = ["delta_wp", "next_is_terminal", "shot_made"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        parts.append(chunk)
    result = pd.concat(parts, ignore_index=True)
    result["sample"] = "All RS shots"
    return result


def summarize(values: pd.Series, sample: str) -> dict[str, float | int | str]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    out: dict[str, float | int | str] = {
        "sample": sample,
        "n": len(values),
        "mean": values.mean(),
        "std": values.std(),
    }
    out.update({f"q{q:g}": value for q, value in values.quantile(QUANTILES).items()})
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_shots = read_all_shots(args.with_wp, args.chunksize)
    panel = pd.read_parquet(args.panel, columns=["delta_wp"])

    terminal = all_shots.loc[all_shots["next_is_terminal"].eq(1), "delta_wp"]
    nonterminal = all_shots.loc[all_shots["next_is_terminal"].eq(0), "delta_wp"]
    samples = {
        "All RS shots": all_shots["delta_wp"],
        "Terminal RS shots": terminal,
        "Nonterminal RS shots": nonterminal,
        "Clutch analysis panel": panel["delta_wp"],
    }

    summary = pd.DataFrame([summarize(values, name) for name, values in samples.items()])
    summary.to_csv(out_dir / "delta_wp_summary.csv", index=False)

    edges = np.linspace(-1, 1, args.bins + 1)
    histogram_rows = []
    for name, values in samples.items():
        clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
        counts, _ = np.histogram(clean, bins=edges)
        histogram_rows.extend(
            {
                "sample": name,
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "share": count / len(clean),
            }
            for left, right, count in zip(edges[:-1], edges[1:], counts)
        )
    pd.DataFrame(histogram_rows).to_csv(out_dir / "delta_wp_histogram.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    plot_order = [
        "All RS shots",
        "Clutch analysis panel",
        "Terminal RS shots",
        "Nonterminal RS shots",
    ]
    for ax, name in zip(axes.flat, plot_order):
        clean = pd.to_numeric(samples[name], errors="coerce").dropna()
        ax.hist(clean, bins=edges, density=True, color="#3568A8", alpha=0.85)
        ax.axvline(clean.mean(), color="#C23B22", linewidth=1.5, label=f"Mean: {clean.mean():.4f}")
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{name} (n={len(clean):,})")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Shot-level delta WP")
    fig.suptitle("Shot-level delta WP distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "shot_level_delta_wp_distribution.png", dpi=300)
    fig.savefig(out_dir / "shot_level_delta_wp_distribution.pdf")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
