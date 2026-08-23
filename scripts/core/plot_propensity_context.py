#!/usr/bin/env python3
"""Plot propensity distribution and context heatmap.

Outputs:
  - histogram of e_hat (overall)
  - heatmap of mean e_hat over (time_left_game, score_diff) grid
  - propensity diagnostics summary (overlap/positivity, extreme_rate, ipw_var, calibration, logloss, brier)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="parquet with e_hat columns")
    p.add_argument("--e-col", type=str, required=True, help="propensity column (e.g., e_hat_three-point)")
    p.add_argument("--time-col", type=str, default="time_left_game")
    p.add_argument("--score-col", type=str, default="score_diff")
    p.add_argument("--time-min", type=float, default=0)
    p.add_argument("--time-max", type=float, default=300)
    p.add_argument("--time-step", type=float, default=15)
    p.add_argument("--score-min", type=float, default=-10)
    p.add_argument("--score-max", type=float, default=10)
    p.add_argument("--score-step", type=float, default=2)
    p.add_argument("--outdir", type=str, default="data/analysis")
    p.add_argument("--prefix", type=str, default="propensity_")
    p.add_argument("--treat-col", type=str, default=None, help="observed treatment column")
    p.add_argument("--treat-a", type=str, default=None, help="treatment label for e-col")
    p.add_argument("--treat-b", type=str, default=None, help="optional treatment-B label")
    p.add_argument("--min-prop", type=float, default=1e-2, help="clip lower bound for IPW diagnostics")
    p.add_argument("--max-prop", type=float, default=1.0, help="clip upper bound for IPW diagnostics")
    p.add_argument(
        "--extreme-threshold",
        type=float,
        default=0.05,
        help="threshold for extreme_rate on observed propensity",
    )
    p.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.05,
        help="threshold for overlap rate: e in [thr, 1-thr]",
    )
    p.add_argument("--calibration-bins", type=int, default=10)
    return p.parse_args()


def _safe_brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    err = y_true - prob
    return float(np.mean(err * err))


def _compute_propensity_diagnostics(df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    e_raw = pd.to_numeric(df[args.e_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(e_raw)
    e_raw = e_raw[mask]
    if e_raw.size == 0:
        raise ValueError(f"No finite rows in {args.e_col}.")
    e = np.clip(e_raw, 1e-9, 1.0 - 1e-9)

    q = np.quantile(e, [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    overlap_thr = float(args.overlap_threshold)
    diagnostics: dict[str, float | int | None] = {
        "n_rows": int(e.size),
        "propensity_mean": float(np.mean(e)),
        "propensity_std": float(np.std(e)),
        "propensity_min": float(np.min(e)),
        "propensity_max": float(np.max(e)),
        "propensity_q01": float(q[0]),
        "propensity_q05": float(q[1]),
        "propensity_q10": float(q[2]),
        "propensity_q50": float(q[3]),
        "propensity_q90": float(q[4]),
        "propensity_q95": float(q[5]),
        "propensity_q99": float(q[6]),
        "overlap_threshold": overlap_thr,
        "overlap_rate": float(np.mean((e >= overlap_thr) & (e <= 1.0 - overlap_thr))),
        "overlap_low_rate": float(np.mean(e < overlap_thr)),
        "overlap_high_rate": float(np.mean(e > 1.0 - overlap_thr)),
    }

    calibration_df = pd.DataFrame()
    if args.treat_col and args.treat_a:
        z = df.loc[mask, args.treat_col].astype(str).to_numpy()
        keep = (z == str(args.treat_a))
        if args.treat_b:
            keep = keep | (z == str(args.treat_b))
        z = z[keep]
        e = e[keep]
        y = (z == str(args.treat_a)).astype(int)
        if y.size == 0:
            raise ValueError("No valid rows for propensity diagnostics after treatment filtering.")

        e_clip = np.clip(e, float(args.min_prop), float(args.max_prop))
        obs_e = np.where(y == 1, e, 1.0 - e)
        obs_e_clip = np.clip(obs_e, float(args.min_prop), float(args.max_prop))

        extreme_thr = float(args.extreme_threshold)
        diagnostics.update(
            {
                "n_rows_binary_treatment": int(y.size),
                "treat_a_rate": float(np.mean(y)),
                "logloss": float(log_loss(y, e_clip, labels=[0, 1])),
                "brier": _safe_brier(y, e_clip),
                "extreme_threshold": extreme_thr,
                "extreme_rate": float(np.mean(obs_e < extreme_thr)),
                "ipw_var": float(np.var(1.0 / obs_e_clip)),
            }
        )

        n_bins = int(args.calibration_bins)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.clip(np.digitize(e, edges[1:-1], right=False), 0, n_bins - 1)
        calib_rows = []
        for b in range(n_bins):
            idx = bin_ids == b
            if not np.any(idx):
                continue
            pred_mean = float(np.mean(e[idx]))
            obs_rate = float(np.mean(y[idx]))
            cnt = int(np.sum(idx))
            calib_rows.append(
                {
                    "bin": int(b),
                    "count": cnt,
                    "pred_mean": pred_mean,
                    "obs_rate": obs_rate,
                    "abs_gap": abs(obs_rate - pred_mean),
                    "bin_left": float(edges[b]),
                    "bin_right": float(edges[b + 1]),
                }
            )
        calibration_df = pd.DataFrame(calib_rows)
        if not calibration_df.empty:
            diagnostics["calibration_ece"] = float(
                np.sum(calibration_df["abs_gap"] * calibration_df["count"]) / np.sum(calibration_df["count"])
            )
            diagnostics["calibration_mce"] = float(calibration_df["abs_gap"].max())
        else:
            diagnostics["calibration_ece"] = None
            diagnostics["calibration_mce"] = None
    else:
        diagnostics.update(
            {
                "n_rows_binary_treatment": None,
                "treat_a_rate": None,
                "logloss": None,
                "brier": None,
                "extreme_threshold": float(args.extreme_threshold),
                "extreme_rate": None,
                "ipw_var": None,
                "calibration_ece": None,
                "calibration_mce": None,
            }
        )
    return diagnostics, calibration_df


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)

    for col in [args.e_col, args.time_col, args.score_col]:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Histogram
    import matplotlib.pyplot as plt

    e = df[args.e_col].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(e, bins=50, color="#2b6cb0", alpha=0.8)
    ax.set_xlabel(args.e_col)
    ax.set_ylabel("count")
    ax.set_title(f"Propensity distribution ({args.e_col})")
    fig.tight_layout()
    hist_path = outdir / f"{args.prefix}{args.e_col}_hist.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)

    # Heatmap grid (bin by rounding to desired step and clipping to range)
    time_grid = np.arange(args.time_min, args.time_max + args.time_step, args.time_step)
    score_grid = np.arange(args.score_min, args.score_max + args.score_step, args.score_step)

    t = df[args.time_col].astype(float).to_numpy()
    s = df[args.score_col].astype(float).to_numpy()
    t_bin = np.clip(np.round(t / args.time_step) * args.time_step, args.time_min, args.time_max)
    s_bin = np.clip(np.round(s / args.score_step) * args.score_step, args.score_min, args.score_max)

    tmp = pd.DataFrame({"time_bin": t_bin, "score_bin": s_bin, "e": df[args.e_col].astype(float).to_numpy()})
    g = tmp.groupby(["time_bin", "score_bin"], observed=True)["e"].mean().reset_index()

    mat = g.pivot(index="time_bin", columns="score_bin", values="e")
    mat = mat.reindex(index=time_grid, columns=score_grid)
    mat = mat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=[
            float(score_grid.min()),
            float(score_grid.max()),
            float(time_grid.min()),
            float(time_grid.max()),
        ],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label=f"mean {args.e_col}")
    ax.set_xlabel(args.score_col)
    ax.set_ylabel(args.time_col)
    ax.set_title(f"Mean {args.e_col} by {args.time_col} × {args.score_col}")
    ax.set_xticks(score_grid)
    fig.tight_layout()
    heat_path = outdir / f"{args.prefix}{args.e_col}_heatmap.png"
    fig.savefig(heat_path, dpi=200)
    plt.close(fig)

    diagnostics, calibration_df = _compute_propensity_diagnostics(df, args)
    diag_json_path = outdir / f"{args.prefix}{args.e_col}_diagnostics.json"
    diag_csv_path = outdir / f"{args.prefix}{args.e_col}_diagnostics.csv"
    pd.DataFrame([diagnostics]).to_csv(diag_csv_path, index=False)
    diag_json_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    calib_csv_path = outdir / f"{args.prefix}{args.e_col}_calibration.csv"
    if not calibration_df.empty:
        calibration_df.to_csv(calib_csv_path, index=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1)
        ax.scatter(
            calibration_df["pred_mean"].to_numpy(dtype=float),
            calibration_df["obs_rate"].to_numpy(dtype=float),
            s=np.maximum(20, 200 * calibration_df["count"].to_numpy(dtype=float) / calibration_df["count"].max()),
            alpha=0.8,
            color="#2b6cb0",
        )
        ax.set_xlabel(f"Predicted P({args.treat_a})")
        ax.set_ylabel(f"Observed rate ({args.treat_a})")
        ax.set_title("Propensity calibration")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        calib_png_path = outdir / f"{args.prefix}{args.e_col}_calibration.png"
        fig.savefig(calib_png_path, dpi=200)
        plt.close(fig)
    else:
        calib_png_path = None

    print(f"[saved] {hist_path}")
    print(f"[saved] {heat_path}")
    print(f"[saved] {diag_csv_path}")
    print(f"[saved] {diag_json_path}")
    if not calibration_df.empty:
        print(f"[saved] {calib_csv_path}")
        print(f"[saved] {calib_png_path}")


if __name__ == "__main__":
    main()
