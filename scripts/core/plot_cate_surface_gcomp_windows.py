#!/usr/bin/env python3
"""Generate the primary and late-clock CATE surfaces in one process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from plot_cate_surface_gcomp import (
    compute_surface,
    maybe_plot,
    maybe_plot_ci_width,
    maybe_plot_sig_mask,
    _grid_from_bounds,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--dr-model", required=True)
    p.add_argument("--outdir", default="data/analysis")
    p.add_argument("--prefix", default="")
    p.add_argument("--treat-a", required=True)
    p.add_argument("--treat-b", required=True)
    p.add_argument("--time-col", default="time_left_game")
    p.add_argument("--score-col", default="score_diff")
    p.add_argument("--score-perspective", default="home", choices=["home"])
    p.add_argument("--score-lo", type=float, default=-10.0)
    p.add_argument("--score-hi", type=float, default=10.0)
    p.add_argument("--n-score", type=int, default=21)
    p.add_argument("--n-sample", type=int, default=20000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--tau-threshold", type=float, default=0.0)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--tau-calib-json", default=None)
    p.add_argument("--tau-calib-alpha", type=float, default=None)
    p.add_argument("--tau-calib-beta", type=float, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-ci-width", action="store_true")
    p.add_argument("--plot-sig-mask", action="store_true")
    p.add_argument("--plot-sig-contour", action="store_true")
    p.add_argument("--score-tick-step", type=float, default=None)
    p.add_argument("--long-name", default="t30_300")
    p.add_argument("--long-time-lo", type=float, default=30.0)
    p.add_argument("--long-time-hi", type=float, default=300.0)
    p.add_argument("--long-n-time", type=int, default=25)
    p.add_argument("--short-name", default="t0_30")
    p.add_argument("--short-time-lo", type=float, default=0.0)
    p.add_argument("--short-time-hi", type=float, default=30.0)
    p.add_argument("--short-n-time", type=int, default=31)
    return p.parse_args()


def resolve_calibration(args: argparse.Namespace) -> tuple[float | None, float | None]:
    alpha = args.tau_calib_alpha
    beta = args.tau_calib_beta
    if args.tau_calib_json:
        calib_path = Path(args.tau_calib_json)
        if not calib_path.exists():
            raise FileNotFoundError(f"tau calibration json not found: {calib_path}")
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        alpha = float(calib["alpha"]) if alpha is None else alpha
        beta = float(calib["beta"]) if beta is None else beta
    if (alpha is None) ^ (beta is None):
        raise ValueError("Specify both --tau-calib-alpha and --tau-calib-beta, or provide --tau-calib-json.")
    return alpha, beta


def write_surface(
    *,
    args: argparse.Namespace,
    combined: pd.DataFrame,
    score_grid: np.ndarray,
    window_name: str,
    time_lo: float,
    time_hi: float,
) -> None:
    outdir = Path(args.outdir)
    score_out_col = args.score_col
    time_grid = _grid_from_bounds(np.array([], dtype=float), lo=time_lo, hi=time_hi, n=len(combined_times(time_lo, time_hi, combined, args.time_col)), round_int=False)
    wanted = np.round(time_grid.astype(float), 8)
    out_df = combined[np.round(combined[args.time_col].astype(float), 8).isin(wanted)].copy()
    out_df = out_df.sort_values([args.time_col, score_out_col]).reset_index(drop=True)

    stem = f"{args.prefix}{window_name}_tau_surface_{args.treat_a}_vs_{args.treat_b}"
    out_path = outdir / f"{stem}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} rows={len(out_df):,}")

    has_ci = "tau_q025" in out_df.columns and "tau_q975" in out_df.columns
    if args.plot:
        out_png = outdir / f"{stem}.png"
        maybe_plot(
            out=out_df,
            time_grid=time_grid,
            score_grid=score_grid,
            time_col=args.time_col,
            score_col=score_out_col,
            score_label=args.score_col,
            out_png=out_png,
            title=f"G-comp tau: {args.treat_a} - {args.treat_b}, {time_lo:g}-{time_hi:g}s",
            threshold_shade=None,
            score_tick_step=args.score_tick_step,
            sig_contour=args.plot_sig_contour and has_ci,
        )
        print(f"[saved] {out_png}")
    if args.plot_ci_width and has_ci:
        out_png = outdir / f"{stem}_ci_width.png"
        maybe_plot_ci_width(out_df, time_grid, score_grid, args.time_col, score_out_col, args.score_col, out_png, f"CI width, {time_lo:g}-{time_hi:g}s", args.score_tick_step)
        print(f"[saved] {out_png}")
    if args.plot_sig_mask and has_ci:
        out_png = outdir / f"{stem}_sig_mask.png"
        maybe_plot_sig_mask(out_df, time_grid, score_grid, args.time_col, score_out_col, args.score_col, out_png, f"Sign certainty, {time_lo:g}-{time_hi:g}s", args.score_tick_step)
        print(f"[saved] {out_png}")
    elif args.plot_sig_mask and not has_ci:
        print(f"[warn] skip sign mask for {window_name}: bootstrap CI columns are unavailable")


def window_grid(time_lo: float, time_hi: float, n_time: int) -> np.ndarray:
    return _grid_from_bounds(np.array([], dtype=float), lo=time_lo, hi=time_hi, n=n_time, round_int=False)


def combined_times(time_lo: float, time_hi: float, combined: pd.DataFrame, time_col: str) -> np.ndarray:
    values = combined[time_col].to_numpy(dtype=float)
    mask = (values >= time_lo - 1e-8) & (values <= time_hi + 1e-8)
    return np.sort(np.unique(np.round(values[mask], 8))).astype(np.float32)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    payload = load(args.dr_model)
    calib_alpha, calib_beta = resolve_calibration(args)
    if calib_alpha is not None and calib_beta is not None:
        print(f"[info] apply tau calibration: alpha={float(calib_alpha):.6g} beta={float(calib_beta):.6g}")

    long_grid = window_grid(args.long_time_lo, args.long_time_hi, args.long_n_time)
    short_grid = window_grid(args.short_time_lo, args.short_time_hi, args.short_n_time)
    time_grid = np.sort(np.unique(np.concatenate([long_grid, short_grid]))).astype(np.float32)
    score_grid = _grid_from_bounds(
        df[args.score_col].to_numpy(),
        lo=args.score_lo,
        hi=args.score_hi,
        n=args.n_score,
        round_int=True,
    )
    print(
        f"[info] compute combined CATE surface once: "
        f"time_points={len(time_grid)} score_points={len(score_grid)} cells={len(time_grid) * len(score_grid)}"
    )
    combined, _, _ = compute_surface(
        payload=payload,
        df=df,
        treat_a=args.treat_a,
        treat_b=args.treat_b,
        time_col=args.time_col,
        score_col=args.score_col,
        score_out_col=args.score_col,
        score_perspective=args.score_perspective,
        n_time=len(time_grid),
        n_score=len(score_grid),
        qlo=0.05,
        qhi=0.95,
        time_lo=None,
        time_hi=None,
        score_lo=None,
        score_hi=None,
        n_sample=args.n_sample,
        seed=args.seed,
        tau_threshold=args.tau_threshold,
        bootstrap=args.bootstrap,
        tau_calib_alpha=calib_alpha,
        tau_calib_beta=calib_beta,
        time_grid_override=time_grid,
        score_grid_override=score_grid,
    )

    score_base = df[args.score_col].to_numpy()
    if not np.any(np.abs(score_base) <= 5):
        mask = combined[args.score_col].abs() <= 5
        if mask.any():
            combined.loc[mask, ["tau_mean", "tau_se", "tau_p_abs_ge_threshold"]] = np.nan

    write_surface(
        args=args,
        combined=combined,
        score_grid=score_grid,
        window_name=args.long_name,
        time_lo=args.long_time_lo,
        time_hi=args.long_time_hi,
    )
    write_surface(
        args=args,
        combined=combined,
        score_grid=score_grid,
        window_name=args.short_name,
        time_lo=args.short_time_lo,
        time_hi=args.short_time_hi,
    )


if __name__ == "__main__":
    main()
