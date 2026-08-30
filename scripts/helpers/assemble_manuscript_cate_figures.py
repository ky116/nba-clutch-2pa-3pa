#!/usr/bin/env python3
"""Assemble the retained manuscript CATE PNGs from pipeline outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_mplconfig = Path(f"/tmp/matplotlib-{os.getuid()}")
_mplconfig.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE2_NAME = "figure2_full_data_t30_300_cate_surface"
FIGURE3_NAME = "figures1_cate_surface_0_30s_masked_n50"
WF_NAME = "figure3_outer_fold_t30_300_cate_surface"
WF_SOURCE_NAME = "outer_fold_ensemble_cate_surface_30_300"
X_AXIS_LABEL = "Score differential (offense perspective)"
Y_AXIS_LABEL = "Time remaining (seconds)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble-dir", type=Path, default=Path("results/full_data_ensemble_state_fixed_loso"))
    p.add_argument("--wf-dir", type=Path, default=Path("results/wf_cate_surfaces"))
    p.add_argument("--figure-source-dir", type=Path, default=Path("results/figure_source_data"))
    p.add_argument("--panel", type=Path, default=Path("data/analysis/shotchoice_panel_clutch_rs.parquet"))
    p.add_argument("--outdir", type=Path, default=Path("."))
    p.add_argument("--min-figure3-cell-count", type=int, default=50)
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def choose_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n  ".join(str(p) for p in paths)
    raise FileNotFoundError(f"{label} not found. Tried:\n  {joined}")


def assign_nearest(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.abs(values[:, None] - grid[None, :]).argmin(axis=1)
    return grid[idx]


def cell_widths(grid: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 2:
        return np.ones_like(grid, dtype=float)
    mids = (grid[:-1] + grid[1:]) / 2.0
    edges = np.empty(len(grid) + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = float(grid[0] if lo is None else lo)
    edges[-1] = float(grid[-1] if hi is None else hi)
    return np.diff(edges)


def add_cell_counts(surface: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    if "cell_count" in surface.columns:
        out = surface.copy()
        out["cell_count"] = pd.to_numeric(out["cell_count"], errors="coerce").fillna(0).astype(int)
        return out
    if not panel_path.exists():
        raise FileNotFoundError(
            f"cell_count is absent from the surface CSV and panel data was not found: {panel_path}"
        )

    panel = read_table(panel_path)
    for col in ["time_left_game", "score_diff"]:
        if col not in surface.columns:
            raise ValueError(f"surface missing column: {col}")
        if col not in panel.columns:
            raise ValueError(f"panel missing column: {col}")

    time_grid = np.sort(surface["time_left_game"].dropna().unique().astype(float))
    score_grid = np.sort(surface["score_diff"].dropna().unique().astype(float))
    time_lo = float(time_grid.min())
    time_hi = float(time_grid.max())
    score_lo = float(score_grid.min())
    score_hi = float(score_grid.max())

    use = panel[
        panel["time_left_game"].between(time_lo, time_hi, inclusive="both")
        & panel["score_diff"].between(score_lo, score_hi, inclusive="both")
    ][["time_left_game", "score_diff"]].copy()
    if use.empty:
        raise ValueError(f"No panel rows fall within surface bounds for {panel_path}")

    use["time_left_game"] = assign_nearest(use["time_left_game"].to_numpy(dtype=float), time_grid)
    use["score_diff"] = use["score_diff"].round().astype(float)

    counts = (
        use.groupby(["time_left_game", "score_diff"], as_index=False)
        .size()
        .rename(columns={"size": "cell_count"})
    )

    out = surface.copy()
    out = out.merge(counts, on=["time_left_game", "score_diff"], how="left")
    out["cell_count"] = out["cell_count"].fillna(0).astype(int)
    out["cell_count_log1p"] = np.log1p(out["cell_count"].astype(float))
    width_map = dict(zip(time_grid.tolist(), cell_widths(time_grid, lo=time_lo, hi=time_hi).tolist()))
    out["time_cell_width"] = out["time_left_game"].map(width_map).astype(float)
    out["cell_density"] = out["cell_count"] / out["time_cell_width"]
    out["cell_density_log1p"] = np.log1p(out["cell_density"])
    return out


def make_source(surface_path: Path, out_path: Path, panel_path: Path, min_count: int | None) -> None:
    surface = pd.read_csv(surface_path)
    if "score_diff" not in surface.columns and "score_diff_offense" in surface.columns:
        surface = surface.rename(columns={"score_diff_offense": "score_diff"})
    if "tau_mean_ensemble" not in surface.columns:
        raise ValueError(f"{surface_path} missing tau_mean_ensemble")
    surface = add_cell_counts(surface, panel_path)
    surface["cate_value"] = surface["tau_mean_ensemble"]
    surface["n_in_cell"] = surface["cell_count"]
    if min_count is None:
        surface["shown_in_figure"] = True
    else:
        surface["shown_in_figure"] = surface["cell_count"] >= int(min_count)

    preferred = [
        "time_left_game",
        "score_diff",
        "cate_value",
        "tau_mean_ensemble",
        "tau_catboost",
        "tau_xgb",
        "tau_lgbm",
        "tau_std_across_models",
        "tau_sign_agree_ratio",
        "tau_sign_disagree",
        "n_in_cell",
        "cell_count",
        "cell_count_log1p",
        "time_cell_width",
        "cell_density",
        "cell_density_log1p",
        "shown_in_figure",
    ]
    cols = [c for c in preferred if c in surface.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    surface[cols].sort_values(["time_left_game", "score_diff"]).to_csv(out_path, index=False)
    print(f"[saved] {out_path}")


def draw_heatmap(source_path: Path, out_path: Path, min_count: int | None = None) -> None:
    df = pd.read_csv(source_path)
    plot_df = df.copy()
    if "shown_in_figure" in plot_df.columns:
        plot_df.loc[~plot_df["shown_in_figure"].astype(bool), "cate_value"] = np.nan
    if min_count is not None and "cell_count" in plot_df.columns:
        plot_df.loc[plot_df["cell_count"] < int(min_count), "cate_value"] = np.nan

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
        extent=[float(score_vals.min()), float(score_vals.max()), float(time_vals.min()), float(time_vals.max())],
        cmap=cmap,
    )
    ax.set_xlim(-10, 10)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)
    fig.colorbar(im, ax=ax, label="Recalibrated CATE")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


def draw_wf_heatmap(source_path: Path, out_path: Path, min_count: int = 20) -> None:
    surface = pd.read_csv(source_path)
    folds = list(surface["outer_fold"].drop_duplicates())
    fig, axes = plt.subplots(1, len(folds), figsize=(4.2 * len(folds), 4.2), sharey=True)
    if len(folds) == 1:
        axes = [axes]
    vals = surface.loc[surface["shown_in_figure"].astype(bool), "cate_value"].to_numpy(dtype=float)
    vmax = max(float(np.nanmax(np.abs(vals))) if vals.size else 0.01, 0.01)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    im = None
    for ax, fold in zip(axes, folds):
        cur = surface[surface["outer_fold"] == fold].copy()
        cur.loc[~cur["shown_in_figure"].astype(bool), "cate_value"] = np.nan
        pivot = cur.pivot(index="time_bin_mid", columns="score_diff_cell", values="cate_value").sort_index()
        arr = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            extent=[
                float(pivot.columns.min()) - 0.5,
                float(pivot.columns.max()) + 0.5,
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
        )
        ax.set_title(str(cur["test_years"].iloc[0]))
        ax.set_xticks(np.arange(int(pivot.columns.min()), int(pivot.columns.max()) + 1, 5))
    axes[0].set_ylabel(Y_AXIS_LABEL)
    fig.supxlabel(X_AXIS_LABEL, y=0.02)
    if im is not None:
        fig.colorbar(im, ax=axes, label="Recalibrated CATE", shrink=0.82)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    args = parse_args()

    figure2_surface = choose_existing(
        [args.ensemble_dir / "full_data_t30_300_cate_surface_equal_weight.csv"],
        "Figure 2 ensemble surface CSV",
    )
    figure3_surface = choose_existing(
        [
            args.ensemble_dir / "masked_n50" / "full_data_t0_30_cate_surface_equal_weight.csv",
            args.ensemble_dir / "full_data_t0_30_cate_surface_equal_weight.csv",
        ],
        "Figure 3 ensemble surface CSV",
    )
    wf_surface = choose_existing(
        [args.wf_dir / f"{WF_SOURCE_NAME}.csv"],
        "walk-forward CATE surface CSV",
    )

    figure2_source = args.figure_source_dir / f"{FIGURE2_NAME}_source_data.csv"
    figure3_source = args.figure_source_dir / f"{FIGURE3_NAME}_source_data.csv"
    make_source(figure2_surface, figure2_source, args.panel, min_count=None)
    make_source(figure3_surface, figure3_source, args.panel, min_count=args.min_figure3_cell_count)

    draw_heatmap(figure2_source, args.outdir / f"{FIGURE2_NAME}.png")
    draw_heatmap(
        figure3_source,
        args.outdir / f"{FIGURE3_NAME}.png",
        min_count=args.min_figure3_cell_count,
    )
    draw_wf_heatmap(wf_surface, args.outdir / f"{WF_NAME}.png")


if __name__ == "__main__":
    main()
