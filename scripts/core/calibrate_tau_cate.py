#!/usr/bin/env python3
r"""calibrate_tau_cate.py

Calibration diagnostics for binary-treatment CATE scores.

This script implements the two minimal diagnostics typically expected for
JSA/JQAS-style heterogeneous treatment effect analyses:

1) Quantile-bucket calibration:
   - Bucket observations by predicted CATE \hat\tau(x) (e.g., deciles).
   - Within each bucket, re-estimate effect using an AIPW/DR score (psi_tau).
   - Check whether bucket-mean AIPW effect is monotone in bucket-mean \hat\tau.

2) BLP (Best Linear Predictor):
   - Regress psi_tau on tau_hat: psi_tau = alpha + beta * tau_hat + error.
   - Well-calibrated scores tend to have alpha \approx 0 and beta \approx 1.

Assumptions:
- You already have OOF nuisance estimates m_hat_* and e_hat_* from DML.
- You already have OOF tau_hat from your DR-learner.

Notes:
- If your binary treatment uses labels (e.g., 'three-point', 'two-point'),
  pass --treat-a-label and --treat-b-label.
- If you only have e_hat for treatment A, set --e-a-col and the script will
  set e_b = 1 - e_a.

Outputs are saved under --outdir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibration: bucket AIPW + BLP for tau_hat")

    p.add_argument(
        "--nuisance",
        type=str,
        required=True,
        help="Parquet containing outcome, treatment, and nuisance columns (m_hat_*, e_hat_*)",
    )
    p.add_argument(
        "--oof-preds",
        type=str,
        required=True,
        help="Parquet containing OOF tau_hat (same row order as nuisance)",
    )
    p.add_argument(
        "--tau-col",
        type=str,
        required=True,
        help="Column in --oof-preds to use as tau_hat",
    )

    p.add_argument("--treat-col", type=str, default="shot_zone_choice")
    p.add_argument("--treat-a-label", type=str, default="three-point")
    p.add_argument("--treat-b-label", type=str, default="two-point")

    p.add_argument("--outcome-col", type=str, default="delta_wp")

    p.add_argument(
        "--m-a-col",
        type=str,
        default=None,
        help="Outcome model pred for treatment A (default: m_hat_<treat-a-label>)",
    )
    p.add_argument(
        "--m-b-col",
        type=str,
        default=None,
        help="Outcome model pred for treatment B (default: m_hat_<treat-b-label>)",
    )
    p.add_argument(
        "--e-a-col",
        type=str,
        default=None,
        help="Propensity for treatment A (default: e_hat_<treat-a-label>)",
    )
    p.add_argument(
        "--e-b-col",
        type=str,
        default=None,
        help="Propensity for treatment B (default: e_hat_<treat-b-label>; if missing use 1-e_a)",
    )

    p.add_argument("--min-prop", type=float, default=1e-3, help="Clip lower bound for propensity")
    p.add_argument("--max-prop", type=float, default=1 - 1e-3, help="Clip upper bound for propensity")

    p.add_argument("--n-buckets", type=int, default=10, help="Number of tau_hat quantile buckets")
    p.add_argument(
        "--bucket-by-abs",
        action="store_true",
        help="Bucket by |tau_hat| instead of tau_hat (not typical for monotone check)",
    )

    p.add_argument(
        "--cluster-col",
        type=str,
        default=None,
        help="Optional cluster id column (e.g., GAME_ID). If provided, BLP uses cluster-robust SE.",
    )

    p.add_argument("--outdir", type=str, default="data/analysis")
    p.add_argument("--prefix", type=str, default="calib_")

    p.add_argument("--plot", action="store_true", help="Save calibration plot PNG")
    p.add_argument(
        "--blp-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap replicates for BLP CI (cluster bootstrap if --cluster-col is provided).",
    )
    p.add_argument("--seed", type=int, default=123, help="Random seed for bootstrap.")

    return p


def _default_col(prefix: str, label: str) -> str:
    return f"{prefix}{label}"


def resolve_cols(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[str, str, str, Optional[str]]:
    m_a = args.m_a_col or _default_col("m_hat_", args.treat_a_label)
    m_b = args.m_b_col or _default_col("m_hat_", args.treat_b_label)
    e_a = args.e_a_col or _default_col("e_hat_", args.treat_a_label)
    e_b = args.e_b_col or _default_col("e_hat_", args.treat_b_label)

    missing = [c for c in [m_a, m_b, e_a] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required nuisance columns: {missing}")

    if e_b not in df.columns:
        e_b = None

    return m_a, m_b, e_a, e_b


def compute_psi_tau(
    df: pd.DataFrame,
    treat_col: str,
    outcome_col: str,
    treat_a_label: str,
    treat_b_label: str,
    m_a_col: str,
    m_b_col: str,
    e_a_col: str,
    e_b_col: Optional[str],
    min_prop: float,
    max_prop: float,
) -> pd.Series:
    # filter to the binary support, if needed
    mask = df[treat_col].astype(str).isin([treat_a_label, treat_b_label])
    if not bool(mask.all()):
        df = df.loc[mask].copy()

    t = (df[treat_col].astype(str) == treat_a_label).astype(np.float32).to_numpy()
    y = df[outcome_col].to_numpy(dtype=np.float32)

    m_a = df[m_a_col].to_numpy(dtype=np.float32)
    m_b = df[m_b_col].to_numpy(dtype=np.float32)

    e_a = df[e_a_col].to_numpy(dtype=np.float32)
    e_a = np.clip(e_a, min_prop, max_prop).astype(np.float32)

    if e_b_col is not None:
        e_b = df[e_b_col].to_numpy(dtype=np.float32)
        e_b = np.clip(e_b, min_prop, max_prop).astype(np.float32)
    else:
        e_b = (1.0 - e_a).astype(np.float32)
        e_b = np.clip(e_b, min_prop, max_prop).astype(np.float32)

    psi_tau = (m_a - m_b) + t * (y - m_a) / e_a - (1.0 - t) * (y - m_b) / e_b
    return pd.Series(psi_tau, index=df.index, name="psi_tau")


def bucket_table(df: pd.DataFrame, tau_col: str, psi_col: str, n_buckets: int, bucket_by_abs: bool) -> pd.DataFrame:
    tau = df[tau_col].astype(float)
    key = tau.abs() if bucket_by_abs else tau

    # qcut can drop bins if there are ties; handle gracefully
    buckets = pd.qcut(key, q=n_buckets, labels=False, duplicates="drop")
    df = df.copy()
    df["bucket"] = buckets

    g = df.groupby("bucket", observed=True)
    out = g.agg(
        n=(psi_col, "size"),
        tau_mean=(tau_col, "mean"),
        tau_p10=(tau_col, lambda x: np.nanpercentile(x, 10)),
        tau_p90=(tau_col, lambda x: np.nanpercentile(x, 90)),
        psi_mean=(psi_col, "mean"),
        psi_std=(psi_col, "std"),
    ).reset_index()

    out["psi_se"] = out["psi_std"] / np.sqrt(out["n"].clip(lower=1))
    out["psi_ci_low"] = out["psi_mean"] - 1.96 * out["psi_se"]
    out["psi_ci_high"] = out["psi_mean"] + 1.96 * out["psi_se"]

    # order buckets from low to high key
    out = out.sort_values("bucket").reset_index(drop=True)
    out["bucket"] = out["bucket"].astype(int)

    return out


def blp_regression(
    df: pd.DataFrame,
    tau_col: str,
    psi_col: str,
    cluster_col: Optional[str],
    n_boot: int,
    seed: int,
) -> dict:
    try:
        import statsmodels.api as sm  # type: ignore
    except ModuleNotFoundError:
        sm = None

    x = df[tau_col].astype(float).to_numpy()
    y = df[psi_col].astype(float).to_numpy()
    effective_cluster_col = cluster_col if cluster_col is not None and cluster_col in df.columns else None

    def _ols_fit_numpy(xv: np.ndarray, yv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        Xv = np.column_stack([np.ones(len(xv), dtype=float), xv.astype(float)])
        params, *_ = np.linalg.lstsq(Xv, yv.astype(float), rcond=None)
        yhat = Xv @ params
        resid = yv - yhat
        sse = float(np.sum(resid**2))
        sst = float(np.sum((yv - float(np.mean(yv))) ** 2))
        r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")

        xtx_inv = np.linalg.pinv(Xv.T @ Xv)
        h = np.sum((Xv @ xtx_inv) * Xv, axis=1)
        denom = np.clip(1.0 - h, 1e-12, None)
        w = (resid / denom) ** 2
        cov_hc3 = xtx_inv @ (Xv.T @ (Xv * w[:, None])) @ xtx_inv
        se = np.sqrt(np.clip(np.diag(cov_hc3), 0.0, None))
        return params.astype(float), se.astype(float), r2

    if sm is not None:
        X = sm.add_constant(x, has_constant="add")

        if effective_cluster_col is None:
            res = sm.OLS(y, X).fit(cov_type="HC3")
            clusters = None
        else:
            clusters = df[effective_cluster_col].to_numpy()
            res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})

        alpha, beta = res.params.tolist()
        se_alpha, se_beta = res.bse.tolist()
        r2 = float(res.rsquared)
        nobs = int(res.nobs)
        cov_type = str(res.cov_type)
        summary_text = res.summary().as_text()
    else:
        clusters = df[effective_cluster_col].to_numpy() if effective_cluster_col is not None else None
        params, se, r2 = _ols_fit_numpy(x, y)
        alpha, beta = float(params[0]), float(params[1])
        nobs = int(len(df))
        if clusters is None:
            se_alpha, se_beta = float(se[0]), float(se[1])
            cov_type = "HC3_numpy"
        else:
            se_alpha, se_beta = float("nan"), float("nan")
            cov_type = "cluster_bootstrap"
        summary_text = (
            "BLP regression fallback (statsmodels unavailable)\n"
            f"n={nobs}, r2={r2:.6g}, alpha={alpha:.6g}, beta={beta:.6g}, cov={cov_type}\n"
        )

    # Default CI: bootstrap percentiles (cluster bootstrap if cluster IDs are available).
    if np.isfinite(se_alpha) and np.isfinite(se_beta):
        alpha_ci = [float(alpha - 1.96 * se_alpha), float(alpha + 1.96 * se_alpha)]
        beta_ci = [float(beta - 1.96 * se_beta), float(beta + 1.96 * se_beta)]
    else:
        alpha_ci = [float("nan"), float("nan")]
        beta_ci = [float("nan"), float("nan")]
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        ab = np.full((int(n_boot), 2), np.nan, dtype=float)
        n = len(df)
        if clusters is None and sm is not None:
            for b in range(int(n_boot)):
                idx = rng.integers(0, n, size=n)
                Xb = sm.add_constant(x[idx], has_constant="add")
                yb = y[idx]
                rb = sm.OLS(yb, Xb).fit()
                ab[b, 0] = float(rb.params[0])
                ab[b, 1] = float(rb.params[1])
        elif clusters is None:
            for b in range(int(n_boot)):
                idx = rng.integers(0, n, size=n)
                pb, _, _ = _ols_fit_numpy(x[idx], y[idx])
                ab[b, 0] = float(pb[0])
                ab[b, 1] = float(pb[1])
        else:
            uniq = pd.unique(clusters[pd.notna(clusters)])
            if len(uniq) > 1:
                idx_by_cluster = {g: np.flatnonzero(clusters == g) for g in uniq}
                g = len(uniq)
                for b in range(int(n_boot)):
                    draw = rng.integers(0, g, size=g)
                    idx = np.concatenate([idx_by_cluster[uniq[k]] for k in draw], axis=0)
                    if sm is not None:
                        Xb = sm.add_constant(x[idx], has_constant="add")
                        yb = y[idx]
                        rb = sm.OLS(yb, Xb).fit()
                        ab[b, 0] = float(rb.params[0])
                        ab[b, 1] = float(rb.params[1])
                    else:
                        pb, _, _ = _ols_fit_numpy(x[idx], y[idx])
                        ab[b, 0] = float(pb[0])
                        ab[b, 1] = float(pb[1])
        ab = ab[np.isfinite(ab).all(axis=1)]
        if len(ab) >= 20:
            alpha_ci = [float(np.quantile(ab[:, 0], 0.025)), float(np.quantile(ab[:, 0], 0.975))]
            beta_ci = [float(np.quantile(ab[:, 1], 0.025)), float(np.quantile(ab[:, 1], 0.975))]
            se_alpha = float(np.std(ab[:, 0], ddof=1))
            se_beta = float(np.std(ab[:, 1], ddof=1))
            cov_type = "cluster_bootstrap" if clusters is not None else "bootstrap"

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "se_alpha": float(se_alpha),
        "se_beta": float(se_beta),
        "alpha_ci": alpha_ci,
        "beta_ci": beta_ci,
        "r2": float(r2),
        "n": int(nobs),
        "cov_type": cov_type,
        "summary": summary_text,
    }


def save_plot(outdir: Path, prefix: str, bkt: pd.DataFrame, blp: dict) -> Path:
    # Helper separated to keep imports local
    import matplotlib.pyplot as plt

    x = bkt["tau_mean"].to_numpy()
    y = bkt["psi_mean"].to_numpy()
    yerr = 1.96 * bkt["psi_se"].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=yerr, fmt="o", label="Binned pseudo-outcome mean ±95% CI")

    # y=x reference
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.5", label="Perfect calibration (y=x)")

    # BLP line
    alpha = blp.get("alpha", 0.0)
    beta = blp.get("beta", 1.0)
    xs = np.array([lo, hi])
    ax.plot(xs, alpha + beta * xs, color="C1", label=f"BLP fit (y={alpha:.3g}+{beta:.3g}x)")

    ax.set_xlabel("Bucket mean of tau_hat")
    ax.set_ylabel("Bucket mean of pseudo-outcome")
    ax.set_title("Calibration (bucket pseudo-outcome) + BLP")
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    path = outdir / f"{prefix}calibration_plot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    args = build_argparser().parse_args()

    nuisance = pd.read_parquet(args.nuisance).reset_index(drop=True)
    preds = pd.read_parquet(args.oof_preds).reset_index(drop=True)

    if len(nuisance) != len(preds):
        raise ValueError(f"Row count mismatch: nuisance={len(nuisance)}, preds={len(preds)}")

    if args.tau_col not in preds.columns:
        raise ValueError(f"tau column not found in oof-preds: {args.tau_col}")

    nuisance["tau_hat"] = preds[args.tau_col].to_numpy(dtype=float)
    if args.cluster_col is not None and args.cluster_col not in nuisance.columns and args.cluster_col in preds.columns:
        nuisance[args.cluster_col] = preds[args.cluster_col].to_numpy()

    if args.treat_col not in nuisance.columns:
        raise ValueError(f"treat_col not found: {args.treat_col}")
    if args.outcome_col not in nuisance.columns:
        raise ValueError(f"outcome_col not found: {args.outcome_col}")

    m_a_col, m_b_col, e_a_col, e_b_col = resolve_cols(nuisance, args)

    psi = compute_psi_tau(
        nuisance,
        treat_col=args.treat_col,
        outcome_col=args.outcome_col,
        treat_a_label=args.treat_a_label,
        treat_b_label=args.treat_b_label,
        m_a_col=m_a_col,
        m_b_col=m_b_col,
        e_a_col=e_a_col,
        e_b_col=e_b_col,
        min_prop=args.min_prop,
        max_prop=args.max_prop,
    )

    df = nuisance.loc[psi.index].copy()
    df["psi_tau"] = psi
    effective_cluster_col = args.cluster_col if args.cluster_col in df.columns else None
    if args.cluster_col is not None and effective_cluster_col is None:
        print(f"[warn] cluster_col '{args.cluster_col}' not found; falling back to HC3 / non-cluster bootstrap.")
    keep = np.isfinite(pd.to_numeric(df["tau_hat"], errors="coerce").to_numpy(dtype=float))
    keep &= np.isfinite(pd.to_numeric(df["psi_tau"], errors="coerce").to_numpy(dtype=float))
    if effective_cluster_col is not None:
        keep &= pd.notna(df[effective_cluster_col]).to_numpy()
    n_before = len(df)
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError("No valid rows after filtering finite tau_hat/psi_tau for calibration.")
    if len(df) < n_before:
        print(f"[info] dropped rows with invalid tau/psi for calibration: {n_before - len(df):,}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    bkt = bucket_table(df, tau_col="tau_hat", psi_col="psi_tau", n_buckets=args.n_buckets, bucket_by_abs=args.bucket_by_abs)

    bkt_path = outdir / f"{prefix}bucket_table.csv"
    bkt.to_csv(bkt_path, index=False)
    print(f"saved {bkt_path}")

    blp = blp_regression(
        df,
        tau_col="tau_hat",
        psi_col="psi_tau",
        cluster_col=effective_cluster_col,
        n_boot=int(args.blp_bootstrap),
        seed=int(args.seed),
    )

    blp_json = outdir / f"{prefix}blp.json"
    blp_txt = outdir / f"{prefix}blp.txt"

    with blp_json.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in blp.items() if k != "summary"}, f, ensure_ascii=False, indent=2)

    with blp_txt.open("w", encoding="utf-8") as f:
        f.write(blp["summary"])

    print(f"saved {blp_json}")
    print(f"saved {blp_txt}")

    if args.plot:
        try:
            plot_path = save_plot(outdir, prefix, bkt, blp)
            print(f"saved {plot_path}")
        except Exception as e:
            print(f"plot failed: {e}")

    # Quick console summary
    print("\n=== BLP quick check ===")
    print(f"alpha = {blp['alpha']:.4g}  (95% CI: [{blp['alpha_ci'][0]:.4g}, {blp['alpha_ci'][1]:.4g}])")
    print(f"beta  = {blp['beta']:.4g}  (95% CI: [{blp['beta_ci'][0]:.4g}, {blp['beta_ci'][1]:.4g}])")
    print(f"R^2   = {blp['r2']:.4g}  n={blp['n']}  cov={blp['cov_type']}")


if __name__ == "__main__":
    main()
