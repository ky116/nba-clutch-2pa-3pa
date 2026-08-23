#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from run_nested_walk_forward import _compute_blp_metrics_robust


def _default_col(prefix: str, label: str) -> str:
    return f"{prefix}{label}"


def _resolve_cols(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str, str, Optional[str]]:
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


def _compute_psi_tau(
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
    mask = df[treat_col].astype(str).isin([treat_a_label, treat_b_label])
    if not bool(mask.all()):
        df = df.loc[mask].copy()

    t = (df[treat_col].astype(str) == treat_a_label).astype(np.float32).to_numpy()
    y = df[outcome_col].to_numpy(dtype=np.float32)
    m_a = df[m_a_col].to_numpy(dtype=np.float32)
    m_b = df[m_b_col].to_numpy(dtype=np.float32)
    e_a = np.clip(df[e_a_col].to_numpy(dtype=np.float32), min_prop, max_prop)
    if e_b_col is not None:
        e_b = np.clip(df[e_b_col].to_numpy(dtype=np.float32), min_prop, max_prop)
    else:
        e_b = np.clip(1.0 - e_a, min_prop, max_prop)

    psi_tau = (m_a - m_b) + t * (y - m_a) / e_a - (1.0 - t) * (y - m_b) / e_b
    return pd.Series(psi_tau, index=df.index, name="psi_tau")


def _bucket_calibration_mae(
    tau: np.ndarray,
    psi: np.ndarray,
    n_buckets: int,
) -> float:
    tmp = pd.DataFrame({"tau": tau, "psi": psi})
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["tau", "psi"])
    if tmp.empty:
        return float("nan")
    b = pd.qcut(tmp["tau"], q=n_buckets, labels=False, duplicates="drop")
    tmp["bucket"] = b
    g = tmp.groupby("bucket", observed=True)[["tau", "psi"]].mean()
    if g.empty:
        return float("nan")
    return float(np.mean(np.abs(g["psi"].to_numpy() - g["tau"].to_numpy())))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply linear tau calibration and quantify improvement.")
    p.add_argument("--nuisance-oos", type=str, required=True)
    p.add_argument("--tau-oos", type=str, required=True)
    p.add_argument("--tau-full", type=str, default=None)
    p.add_argument("--tau-col", type=str, default="tau_hat")
    p.add_argument("--calib-json", type=str, default=None, help="Optional JSON containing alpha/beta.")
    p.add_argument("--cluster-col", type=str, default="GAME_ID")
    p.add_argument("--treat-col", type=str, default="shot_zone_choice")
    p.add_argument("--treat-a-label", type=str, default="three-point")
    p.add_argument("--treat-b-label", type=str, default="two-point")
    p.add_argument("--outcome-col", type=str, default="delta_wp")
    p.add_argument("--m-a-col", type=str, default=None)
    p.add_argument("--m-b-col", type=str, default=None)
    p.add_argument("--e-a-col", type=str, default=None)
    p.add_argument("--e-b-col", type=str, default=None)
    p.add_argument("--min-prop", type=float, default=1e-3)
    p.add_argument("--max-prop", type=float, default=1.0 - 1e-3)
    p.add_argument("--n-buckets", type=int, default=10)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--prefix", type=str, default="full_data_")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    nuisance = pd.read_parquet(args.nuisance_oos).reset_index(drop=True)
    tau_oos = pd.read_parquet(args.tau_oos).reset_index(drop=True)
    if args.tau_col not in tau_oos.columns:
        raise ValueError(f"tau column not found: {args.tau_col}")
    if len(nuisance) != len(tau_oos):
        raise ValueError(f"Row count mismatch: nuisance={len(nuisance)}, tau_oos={len(tau_oos)}")

    m_a_col, m_b_col, e_a_col, e_b_col = _resolve_cols(nuisance, args)
    nuisance["tau_hat_raw"] = tau_oos[args.tau_col].to_numpy(dtype=float)
    nuisance["psi_tau"] = _compute_psi_tau(
        nuisance,
        treat_col=args.treat_col,
        outcome_col=args.outcome_col,
        treat_a_label=args.treat_a_label,
        treat_b_label=args.treat_b_label,
        m_a_col=m_a_col,
        m_b_col=m_b_col,
        e_a_col=e_a_col,
        e_b_col=e_b_col,
        min_prop=float(args.min_prop),
        max_prop=float(args.max_prop),
    )

    keep = np.isfinite(pd.to_numeric(nuisance["tau_hat_raw"], errors="coerce").to_numpy(dtype=float))
    keep &= np.isfinite(pd.to_numeric(nuisance["psi_tau"], errors="coerce").to_numpy(dtype=float))
    if args.cluster_col and args.cluster_col in nuisance.columns:
        keep &= pd.notna(nuisance[args.cluster_col]).to_numpy()
    df_eval = nuisance.loc[keep].copy()
    if df_eval.empty:
        raise ValueError("No valid OOS rows for calibration apply/compare.")

    if args.calib_json:
        calib = json.loads(Path(args.calib_json).read_text(encoding="utf-8"))
        alpha = float(calib["alpha"])
        beta = float(calib["beta"])
    else:
        c = df_eval[args.cluster_col].to_numpy() if args.cluster_col in df_eval.columns else None
        fit_raw = _compute_blp_metrics_robust(
            tau_hat=df_eval["tau_hat_raw"].to_numpy(dtype=float),
            psi_tau=df_eval["psi_tau"].to_numpy(dtype=float),
            cluster=c,
        )
        alpha = float(fit_raw["blp_alpha"])
        beta = float(fit_raw["blp_beta"])

    df_eval["tau_hat_cal"] = alpha + beta * df_eval["tau_hat_raw"].to_numpy(dtype=float)
    c_eval = df_eval[args.cluster_col].to_numpy() if args.cluster_col in df_eval.columns else None
    blp_raw = _compute_blp_metrics_robust(
        tau_hat=df_eval["tau_hat_raw"].to_numpy(dtype=float),
        psi_tau=df_eval["psi_tau"].to_numpy(dtype=float),
        cluster=c_eval,
    )
    blp_cal = _compute_blp_metrics_robust(
        tau_hat=df_eval["tau_hat_cal"].to_numpy(dtype=float),
        psi_tau=df_eval["psi_tau"].to_numpy(dtype=float),
        cluster=c_eval,
    )

    bucket_mae_raw = _bucket_calibration_mae(
        tau=df_eval["tau_hat_raw"].to_numpy(dtype=float),
        psi=df_eval["psi_tau"].to_numpy(dtype=float),
        n_buckets=int(args.n_buckets),
    )
    bucket_mae_cal = _bucket_calibration_mae(
        tau=df_eval["tau_hat_cal"].to_numpy(dtype=float),
        psi=df_eval["psi_tau"].to_numpy(dtype=float),
        n_buckets=int(args.n_buckets),
    )

    tau_oos_out = tau_oos.copy()
    tau_oos_out[f"{args.tau_col}_raw"] = tau_oos[args.tau_col].to_numpy(dtype=float)
    tau_oos_out[f"{args.tau_col}_cal"] = alpha + beta * tau_oos[args.tau_col].to_numpy(dtype=float)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tau_oos_out_path = outdir / f"{args.prefix}tau_oos_train_calibrated.parquet"
    tau_oos_out.to_parquet(tau_oos_out_path, index=False)
    print(f"saved {tau_oos_out_path}")

    tau_full_out_path = None
    if args.tau_full:
        tau_full = pd.read_parquet(args.tau_full).reset_index(drop=True)
        if args.tau_col not in tau_full.columns:
            raise ValueError(f"tau column not found in tau-full: {args.tau_col}")
        tau_full_out = tau_full.copy()
        tau_full_out[f"{args.tau_col}_raw"] = tau_full[args.tau_col].to_numpy(dtype=float)
        tau_full_out[f"{args.tau_col}_cal"] = alpha + beta * tau_full[args.tau_col].to_numpy(dtype=float)
        tau_full_out_path = outdir / f"{args.prefix}tau_full_train_calibrated.parquet"
        tau_full_out.to_parquet(tau_full_out_path, index=False)
        print(f"saved {tau_full_out_path}")

    summary = {
        "alpha": alpha,
        "beta": beta,
        "n_eval_oos": int(len(df_eval)),
        "blp_raw": blp_raw,
        "blp_cal": blp_cal,
        "bucket_mae_raw": bucket_mae_raw,
        "bucket_mae_cal": bucket_mae_cal,
        "delta_abs_beta_gap": float(abs(blp_raw["blp_beta"] - 1.0) - abs(blp_cal["blp_beta"] - 1.0)),
        "delta_abs_alpha": float(abs(blp_raw["blp_alpha"]) - abs(blp_cal["blp_alpha"])),
        "delta_bucket_mae": float(bucket_mae_raw - bucket_mae_cal),
        "tau_oos_calibrated_path": str(tau_oos_out_path),
        "tau_full_calibrated_path": str(tau_full_out_path) if tau_full_out_path is not None else None,
    }

    metrics_path = outdir / f"{args.prefix}tau_calibration_apply_metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {metrics_path}")

    print("\n=== Calibration Apply Summary ===")
    print(f"alpha={alpha:.6g} beta={beta:.6g}")
    print(f"abs(beta-1): raw={abs(blp_raw['blp_beta']-1.0):.6g} cal={abs(blp_cal['blp_beta']-1.0):.6g}")
    print(f"abs(alpha):  raw={abs(blp_raw['blp_alpha']):.6g} cal={abs(blp_cal['blp_alpha']):.6g}")
    print(f"bucket_mae:  raw={bucket_mae_raw:.6g} cal={bucket_mae_cal:.6g}")


if __name__ == "__main__":
    main()
