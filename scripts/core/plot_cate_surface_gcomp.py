#!/usr/bin/env python3
"""plot_cate_surface_gcomp.py

G-computation (marginalized) CATE surface for a DR-learner.

Works with the artifact produced by train_dr_learner.py:
  payload = joblib.load(<dr_learner_*.joblib>)
  payload['models'] is a dict: level(str) -> estimator
    - xgb/lgbm: sklearn Pipeline with OneHotEncoder
    - catboost: CatBoostRegressor trained on DataFrame (cat_features by name)

We compute a 2D surface over (time_left_game, score_diff) via:
  tau(g) = mean_i [ mu_a(X_i(g)) - mu_b(X_i(g)) ]
where X_i(g) is the i-th row with only time_left_game and score_diff overwritten.

Binary treatment:
  choose (a,b) as treatment level strings, e.g. a='1', b='0'.

Outputs:
  - <outdir>/<prefix>tau_surface.parquet (grid + tau_mean + tau_se)
  - <outdir>/<prefix>tau_surface.png      (optional)

Notes:
  * This is "marginalized" CATE (partial dependence-like). It avoids picking
    arbitrary fixed values for other covariates.
  * To reduce extrapolation, default grid range uses data quantiles (5%-95%).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from joblib import load


def _as_category(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _predict_mu(payload: Dict, X: pd.DataFrame, level: str) -> np.ndarray:
    model_name = payload.get("model_name")
    mdl = payload["models"][level]
    pre_map = payload.get("preprocessors") or {}
    pre = pre_map.get(level)

    if model_name == "catboost":
        cat_cols = payload.get("categorical_cols") or []
        Xcb = X.copy()
        Xcb = _as_category(Xcb, cat_cols)
        pred = mdl.predict(Xcb)
        return np.asarray(pred, dtype=np.float32)

    # Nested walk-forward payload (xgb/lgbm): estimator + external preprocessor
    if pre is not None:
        Xenc = pre.transform(X)
        pred = mdl.predict(Xenc)
        return np.asarray(pred, dtype=np.float32)

    # train_dr_learner payload (xgb/lgbm): pipeline expects DataFrame
    pred = mdl.predict(X)
    return np.asarray(pred, dtype=np.float32)


def _resolve_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    missing: list[str] = []
    for col in feature_cols:
        if col in df.columns:
            data[col] = df[col]
            continue
        if col.endswith("_eb"):
            alt = col[:-3]
            if alt in df.columns:
                data[col] = df[alt]
                continue
        missing.append(col)
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return pd.DataFrame(data, index=df.index)


def _grid_from_quantiles(x: np.ndarray, qlo: float, qhi: float, n: int, round_int: bool = False) -> np.ndarray:
    lo = float(np.nanquantile(x, qlo))
    hi = float(np.nanquantile(x, qhi))
    if round_int:
        lo = float(np.floor(lo))
        hi = float(np.ceil(hi))
    if lo == hi:
        return np.array([lo], dtype=np.float32)
    g = np.linspace(lo, hi, n, dtype=np.float32)
    if round_int:
        g = np.unique(np.round(g)).astype(np.float32)
    return g


def _grid_from_bounds(x: np.ndarray, lo: float, hi: float, n: int, round_int: bool = False) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    if round_int:
        lo = float(np.floor(lo))
        hi = float(np.ceil(hi))
    if lo == hi:
        return np.array([lo], dtype=np.float32)
    g = np.linspace(lo, hi, n, dtype=np.float32)
    if round_int:
        g = np.unique(np.round(g)).astype(np.float32)
    return g


def compute_surface(
    payload: Dict,
    df: pd.DataFrame,
    treat_a: str,
    treat_b: str,
    time_col: str,
    score_col: str,
    score_out_col: str,
    score_perspective: str,
    n_time: int,
    n_score: int,
    qlo: float,
    qhi: float,
    time_lo: Optional[float],
    time_hi: Optional[float],
    score_lo: Optional[float],
    score_hi: Optional[float],
    n_sample: int,
    seed: int,
    tau_threshold: float,
    bootstrap: int,
    tau_calib_alpha: Optional[float],
    tau_calib_beta: Optional[float],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feat_cols = payload.get("feature_cols")
    if not feat_cols:
        raise ValueError("payload does not contain feature_cols")
    for c in [time_col, score_col]:
        if c not in feat_cols:
            raise ValueError(f"{c} is not in feature_cols of the DR-learner artifact")

    X_all = _resolve_feature_frame(df, feat_cols)

    # Sample rows for marginalization (speed)
    if n_sample is not None and n_sample > 0 and len(X_all) > n_sample:
        X = X_all.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    else:
        X = X_all.reset_index(drop=True)

    # Build grid (use quantiles by default)
    if time_lo is not None and time_hi is not None:
        time_grid = _grid_from_bounds(X_all[time_col].to_numpy(), lo=time_lo, hi=time_hi, n=n_time, round_int=False)
    else:
        time_grid = _grid_from_quantiles(X_all[time_col].to_numpy(), qlo=qlo, qhi=qhi, n=n_time, round_int=False)
    score_base = X_all[score_col].to_numpy()
    if score_lo is not None and score_hi is not None:
        score_grid = _grid_from_bounds(score_base, lo=score_lo, hi=score_hi, n=n_score, round_int=True)
    else:
        score_grid = _grid_from_quantiles(score_base, qlo=qlo, qhi=qhi, n=n_score, round_int=True)

    # Pre-allocate
    tau_mean = np.zeros((len(time_grid), len(score_grid)), dtype=np.float32)
    tau_se = np.zeros_like(tau_mean)
    tau_p = np.zeros_like(tau_mean)  # proportion(|tau| >= threshold) as a simple "robustness" summary

    # For bootstrap, we resample rows and re-average (no refit)
    rng = np.random.default_rng(seed)

    # Prepare a working copy once to reduce overhead
    X_work = X.copy()

    for i, t in enumerate(time_grid):
        X_work[time_col] = t
        for j, s in enumerate(score_grid):
            X_work[score_col] = s

            mu_a = _predict_mu(payload, X_work, treat_a)
            mu_b = _predict_mu(payload, X_work, treat_b)
            tau = (mu_a - mu_b).astype(np.float32)
            if tau_calib_alpha is not None and tau_calib_beta is not None:
                tau = (float(tau_calib_alpha) + float(tau_calib_beta) * tau).astype(np.float32)

            tau_mean[i, j] = float(np.mean(tau))
            tau_se[i, j] = float(np.std(tau, ddof=1) / np.sqrt(len(tau))) if len(tau) > 1 else 0.0
            tau_p[i, j] = float(np.mean(np.abs(tau) >= tau_threshold)) if tau_threshold > 0 else np.nan

            # Optional bootstrap CI (stored in wide df later if requested)

    # Long-form dataframe
    out_rows = []
    for i, t in enumerate(time_grid):
        for j, s in enumerate(score_grid):
            out_rows.append(
                {
                    time_col: float(t),
                    score_out_col: float(s),
                    "tau_mean": float(tau_mean[i, j]),
                    "tau_se": float(tau_se[i, j]),
                    "tau_p_abs_ge_threshold": float(tau_p[i, j]) if not np.isnan(tau_p[i, j]) else None,
                }
            )
    out = pd.DataFrame(out_rows)

    # Bootstrap CI (adds columns tau_q025, tau_q975)
    if bootstrap and bootstrap > 0:
        # We'll compute CI on the fly in a second pass to avoid storing huge arrays.
        # This is still fairly quick because we only resample row indices.
        qlo_col, qhi_col = [], []
        for i, t in enumerate(time_grid):
            X_work[time_col] = t
            for j, s in enumerate(score_grid):
                X_work[score_col] = s
                mu_a = _predict_mu(payload, X_work, treat_a)
                mu_b = _predict_mu(payload, X_work, treat_b)
                tau = (mu_a - mu_b).astype(np.float32)
                if tau_calib_alpha is not None and tau_calib_beta is not None:
                    tau = (float(tau_calib_alpha) + float(tau_calib_beta) * tau).astype(np.float32)

                if len(tau) == 0:
                    qlo_col.append(np.nan)
                    qhi_col.append(np.nan)
                    continue

                boot_means = []
                for _ in range(bootstrap):
                    idx = rng.integers(0, len(tau), size=len(tau))
                    boot_means.append(float(np.mean(tau[idx])))
                boot_means = np.asarray(boot_means, dtype=np.float32)
                qlo_col.append(float(np.quantile(boot_means, 0.025)))
                qhi_col.append(float(np.quantile(boot_means, 0.975)))

        out["tau_q025"] = qlo_col
        out["tau_q975"] = qhi_col

    return out, time_grid, score_grid


def maybe_plot(
    out: pd.DataFrame,
    time_grid: np.ndarray,
    score_grid: np.ndarray,
    time_col: str,
    score_col: str,
    score_label: str,
    out_png: Path,
    title: str,
    threshold_shade: Optional[float],
    score_tick_step: Optional[float] = None,
    sig_contour: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    # Pivot to matrix
    mat = out.pivot(index=time_col, columns=score_col, values="tau_mean").to_numpy(dtype=float)
    mat = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=[float(score_grid.min()), float(score_grid.max()), float(time_grid.min()), float(time_grid.max())],
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, label="tau_mean")

    # 0-contour for sign flip (tau_mean)
    try:
        Xs, Xt = np.meshgrid(score_grid, time_grid)
        ax.contour(Xs, Xt, mat, levels=[0.0])
    except Exception:
        pass

    # Optional CI sign certainty contours
    if sig_contour:
        if "tau_q025" not in out.columns or "tau_q975" not in out.columns:
            raise ValueError("tau_q025/tau_q975 not found. Run with --bootstrap > 0.")
        q025 = out.pivot(index=time_col, columns=score_col, values="tau_q025").to_numpy(dtype=float)
        q975 = out.pivot(index=time_col, columns=score_col, values="tau_q975").to_numpy(dtype=float)
        q025 = np.ma.masked_invalid(q025)
        q975 = np.ma.masked_invalid(q975)
        Xs, Xt = np.meshgrid(score_grid, time_grid)
        try:
            ax.contour(Xs, Xt, q025, levels=[0.0], colors="white", linewidths=1.2, linestyles="--")
            ax.contour(Xs, Xt, q975, levels=[0.0], colors="black", linewidths=1.2, linestyles="--")
        except Exception:
            pass

    # Optional shading of "near-zero" region
    if threshold_shade is not None and threshold_shade > 0:
        near = (np.abs(mat) < threshold_shade).astype(float)
        ax.contourf(
            np.meshgrid(score_grid, time_grid)[0],
            np.meshgrid(score_grid, time_grid)[1],
            near,
            levels=[0.5, 1.5],
            alpha=0.2,
        )

    ax.set_xlabel(score_label)
    ax.set_ylabel(time_col)
    ax.set_title(title)
    if score_tick_step is not None and score_tick_step > 0:
        xmin = float(score_grid.min())
        xmax = float(score_grid.max())
        start = np.ceil(xmin / score_tick_step) * score_tick_step
        ticks = np.arange(start, xmax + 0.5 * score_tick_step, score_tick_step)
        ax.set_xticks(ticks)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def maybe_plot_ci_width(
    out: pd.DataFrame,
    time_grid: np.ndarray,
    score_grid: np.ndarray,
    time_col: str,
    score_col: str,
    score_label: str,
    out_png: Path,
    title: str,
    score_tick_step: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt

    if "tau_q025" not in out.columns or "tau_q975" not in out.columns:
        raise ValueError("tau_q025/tau_q975 not found. Run with --bootstrap > 0.")

    out = out.copy()
    out["tau_ci_width"] = out["tau_q975"] - out["tau_q025"]

    mat = out.pivot(index=time_col, columns=score_col, values="tau_ci_width").to_numpy(dtype=float)
    mat = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=[float(score_grid.min()), float(score_grid.max()), float(time_grid.min()), float(time_grid.max())],
        cmap=cmap,
    )
    fig.colorbar(im, ax=ax, label="CI width (tau_q975 - tau_q025)")

    ax.set_xlabel(score_label)
    ax.set_ylabel(time_col)
    ax.set_title(title)
    if score_tick_step is not None and score_tick_step > 0:
        xmin = float(score_grid.min())
        xmax = float(score_grid.max())
        start = np.ceil(xmin / score_tick_step) * score_tick_step
        ticks = np.arange(start, xmax + 0.5 * score_tick_step, score_tick_step)
        ax.set_xticks(ticks)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def maybe_plot_sig_mask(
    out: pd.DataFrame,
    time_grid: np.ndarray,
    score_grid: np.ndarray,
    time_col: str,
    score_col: str,
    score_label: str,
    out_png: Path,
    title: str,
    score_tick_step: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    if "tau_q025" not in out.columns or "tau_q975" not in out.columns:
        raise ValueError("tau_q025/tau_q975 not found. Run with --bootstrap > 0.")

    q025 = out.pivot(index=time_col, columns=score_col, values="tau_q025").to_numpy(dtype=float)
    q975 = out.pivot(index=time_col, columns=score_col, values="tau_q975").to_numpy(dtype=float)
    q025 = np.ma.masked_invalid(q025)
    q975 = np.ma.masked_invalid(q975)

    # -1: significantly negative, 0: crosses zero, +1: significantly positive
    sig = np.zeros_like(q025, dtype=float)
    sig[q975 < 0] = -1.0
    sig[q025 > 0] = 1.0
    sig = np.ma.masked_invalid(sig)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = ListedColormap(["#2b6cb0", "#e2e8f0", "#c53030"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    im = ax.imshow(
        sig,
        aspect="auto",
        origin="lower",
        extent=[float(score_grid.min()), float(score_grid.max()), float(time_grid.min()), float(time_grid.max())],
        cmap=cmap,
        norm=norm,
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["tau_q975 < 0", "crosses 0", "tau_q025 > 0"])

    ax.set_xlabel(score_label)
    ax.set_ylabel(time_col)
    ax.set_title(title)
    if score_tick_step is not None and score_tick_step > 0:
        xmin = float(score_grid.min())
        xmax = float(score_grid.max())
        start = np.ceil(xmin / score_tick_step) * score_tick_step
        ticks = np.arange(start, xmax + 0.5 * score_tick_step, score_tick_step)
        ax.set_xticks(ticks)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="panel parquet used for features")
    p.add_argument("--dr-model", type=str, required=True, help="DR-learner joblib path")
    p.add_argument("--outdir", type=str, default="data/analysis", help="output directory")
    p.add_argument("--prefix", type=str, default="", help="output file prefix")

    p.add_argument("--treat-a", type=str, default=None, help="treatment level A (string)")
    p.add_argument("--treat-b", type=str, default=None, help="treatment level B (string)")

    p.add_argument("--time-col", type=str, default="time_left_game")
    p.add_argument("--score-col", type=str, default="score_diff")
    p.add_argument(
        "--score-perspective",
        type=str,
        default="home",
        choices=["home"],
        help="use score_diff as-is (lead=positive)",
    )

    p.add_argument("--n-time", type=int, default=25)
    p.add_argument("--n-score", type=int, default=31)
    p.add_argument("--qlo", type=float, default=0.05)
    p.add_argument("--qhi", type=float, default=0.95)
    p.add_argument("--time-lo", type=float, default=None, help="Optional fixed min for time_left_game grid")
    p.add_argument("--time-hi", type=float, default=None, help="Optional fixed max for time_left_game grid")
    p.add_argument("--score-lo", type=float, default=-10.0, help="Fixed min for score_diff grid (default: -10)")
    p.add_argument("--score-hi", type=float, default=10.0, help="Fixed max for score_diff grid (default: 10)")

    p.add_argument("--n-sample", type=int, default=20000, help="rows to average over (speed/variance tradeoff)")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--tau-threshold", type=float, default=0.0, help="report P(|tau_i|>=threshold) on each grid")
    p.add_argument("--bootstrap", type=int, default=0, help="bootstrap replicates for CI of tau_mean")
    p.add_argument(
        "--tau-calib-json",
        type=str,
        default=None,
        help="Optional JSON containing linear tau calibration coefficients alpha/beta.",
    )
    p.add_argument("--tau-calib-alpha", type=float, default=None, help="Optional tau calibration intercept alpha.")
    p.add_argument("--tau-calib-beta", type=float, default=None, help="Optional tau calibration slope beta.")

    p.add_argument("--plot", action="store_true", help="save PNG heatmap")
    p.add_argument("--shade-near-zero", type=float, default=None, help="shade |tau_mean|<x region")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--score-tick-step", type=float, default=None, help="x-axis tick step for score_diff")
    p.add_argument("--plot-ci-width", action="store_true", help="save PNG heatmap of CI width (requires --bootstrap)")
    p.add_argument("--plot-sig-mask", action="store_true", help="save PNG with sign-certainty mask (requires --bootstrap)")
    p.add_argument("--plot-sig-contour", action="store_true", help="overlay tau_q025/tau_q975 zero contours (requires --bootstrap)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.input)
    payload = load(args.dr_model)

    treat_levels = payload.get("treat_levels") or list(payload["models"].keys())
    baseline = payload.get("baseline")

    # Default pair: (non-baseline vs baseline)
    if args.treat_a is None or args.treat_b is None:
        if baseline is None:
            # fall back to first/second
            if len(treat_levels) < 2:
                raise ValueError("Need at least two treatment levels")
            b = str(treat_levels[0])
            a = str(treat_levels[1])
        else:
            b = str(baseline)
            others = [str(x) for x in treat_levels if str(x) != b]
            if not others:
                raise ValueError("No non-baseline treatment level found")
            a = others[0]
        treat_a, treat_b = a, b
    else:
        treat_a, treat_b = str(args.treat_a), str(args.treat_b)

    if treat_a not in payload["models"]:
        raise ValueError(f"treat-a '{treat_a}' not found. available={list(payload['models'].keys())}")
    if treat_b not in payload["models"]:
        raise ValueError(f"treat-b '{treat_b}' not found. available={list(payload['models'].keys())}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    calib_alpha = args.tau_calib_alpha
    calib_beta = args.tau_calib_beta
    if args.tau_calib_json:
        calib_path = Path(args.tau_calib_json)
        if not calib_path.exists():
            raise FileNotFoundError(f"tau calibration json not found: {calib_path}")
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        if calib_alpha is None:
            calib_alpha = float(calib["alpha"])
        if calib_beta is None:
            calib_beta = float(calib["beta"])
    if (calib_alpha is None) ^ (calib_beta is None):
        raise ValueError("Specify both --tau-calib-alpha and --tau-calib-beta, or provide --tau-calib-json.")
    if calib_alpha is not None and calib_beta is not None:
        print(f"[info] apply tau calibration: alpha={float(calib_alpha):.6g} beta={float(calib_beta):.6g}")

    score_out_col = args.score_col
    score_label = args.score_col
    out_df, time_grid, score_grid = compute_surface(
        payload=payload,
        df=df,
        treat_a=treat_a,
        treat_b=treat_b,
        time_col=args.time_col,
        score_col=args.score_col,
        score_out_col=score_out_col,
        score_perspective=args.score_perspective,
        n_time=args.n_time,
        n_score=args.n_score,
        qlo=args.qlo,
        qhi=args.qhi,
        time_lo=args.time_lo,
        time_hi=args.time_hi,
        score_lo=args.score_lo,
        score_hi=args.score_hi,
        n_sample=args.n_sample,
        seed=args.seed,
        tau_threshold=args.tau_threshold,
        bootstrap=args.bootstrap,
        tau_calib_alpha=calib_alpha,
        tau_calib_beta=calib_beta,
    )

    # Mask no-data band (e.g., extended clutch has no |score_diff|<=5)
    score_base = df[args.score_col].to_numpy()
    if not np.any(np.abs(score_base) <= 5):
        mask = out_df[score_out_col].abs() <= 5
        if mask.any():
            out_df.loc[mask, ["tau_mean", "tau_se", "tau_p_abs_ge_threshold"]] = np.nan

    stem = f"{args.prefix}tau_surface_{treat_a}_vs_{treat_b}"
    out_path = outdir / f"{stem}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} rows={len(out_df):,}")

    has_bootstrap_ci = "tau_q025" in out_df.columns and "tau_q975" in out_df.columns

    if args.plot:
        out_png = outdir / f"{stem}.png"
        title = args.title or f"G-comp tau: {treat_a} - {treat_b}"
        if args.plot_sig_contour and not has_bootstrap_ci:
            print("[warn] skip --plot-sig-contour because bootstrap CI columns are unavailable; run with --bootstrap > 0")
        maybe_plot(
            out=out_df,
            time_grid=time_grid,
            score_grid=score_grid,
            time_col=args.time_col,
            score_col=score_out_col,
            score_label=score_label,
            out_png=out_png,
            title=title,
            threshold_shade=args.shade_near_zero,
            score_tick_step=args.score_tick_step,
            sig_contour=args.plot_sig_contour and has_bootstrap_ci,
        )
        print(f"[saved] {out_png}")

    if args.plot_ci_width:
        if not has_bootstrap_ci:
            print("[warn] skip --plot-ci-width because bootstrap CI columns are unavailable; run with --bootstrap > 0")
            return
        out_png = outdir / f"{stem}_ci_width.png"
        title = args.title or f"G-comp tau CI width: {treat_a} - {treat_b}"
        maybe_plot_ci_width(
            out=out_df,
            time_grid=time_grid,
            score_grid=score_grid,
            time_col=args.time_col,
            score_col=score_out_col,
            score_label=score_label,
            out_png=out_png,
            title=title,
            score_tick_step=args.score_tick_step,
        )
        print(f"[saved] {out_png}")

    if args.plot_sig_mask:
        if not has_bootstrap_ci:
            print("[warn] skip --plot-sig-mask because bootstrap CI columns are unavailable; run with --bootstrap > 0")
            return
        out_png = outdir / f"{stem}_sig_mask.png"
        title = args.title or f"G-comp tau sign certainty: {treat_a} - {treat_b}"
        maybe_plot_sig_mask(
            out=out_df,
            time_grid=time_grid,
            score_grid=score_grid,
            time_col=args.time_col,
            score_col=score_out_col,
            score_label=score_label,
            out_png=out_png,
            title=title,
            score_tick_step=args.score_tick_step,
        )
        print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
