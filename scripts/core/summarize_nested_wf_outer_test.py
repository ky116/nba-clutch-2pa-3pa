#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _model_name_from_dir(path: Path) -> str:
    name = path.name
    if name.startswith("nested_wf_"):
        parts = name.split("_")
        if len(parts) >= 3:
            return parts[2]
    return name


def _outer_dirs(base_dir: Path) -> list[Path]:
    return sorted([p for p in base_dir.glob("train*_test*") if p.is_dir()])


def _load_outer_split(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["outer_split"]


def _load_tau_test(outer_dir: Path) -> pd.DataFrame:
    tau_test_path = outer_dir / "tau_test.parquet"
    if not tau_test_path.exists():
        raise FileNotFoundError(f"Missing required outer-test artifact under {outer_dir}")
    return pd.read_parquet(tau_test_path)


def _load_blp_alpha_beta(outer_dir: Path) -> tuple[float, float]:
    path = outer_dir / "blp_metrics_oos_train.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing BLP metrics: {path}")
    with path.open("r", encoding="utf-8") as f:
        blp = json.load(f)
    alpha = float(blp["blp_alpha"])
    beta = float(blp["blp_beta"])
    if not (np.isfinite(alpha) and np.isfinite(beta)):
        raise ValueError(f"Non-finite BLP calibration in {path}: alpha={alpha}, beta={beta}")
    return alpha, beta


def _tau_for_summary(
    df: pd.DataFrame,
    outer_dir: Path,
    recalibrate_tau_blp: bool,
) -> tuple[np.ndarray, float | None, float | None]:
    tau_raw = df["tau_hat"].to_numpy(dtype=float)
    if not recalibrate_tau_blp:
        return tau_raw, None, None
    alpha, beta = _load_blp_alpha_beta(outer_dir)
    return alpha + beta * tau_raw, alpha, beta


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    random_state: int,
    cluster: np.ndarray | None = None,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(random_state))
    vals = np.empty(int(n_bootstrap), dtype=float)
    if cluster is None:
        n = len(values)
        for b in range(int(n_bootstrap)):
            idx = rng.integers(0, n, size=n)
            vals[b] = float(np.mean(values[idx]))
    else:
        cluster = np.asarray(cluster)
        uniq = pd.unique(cluster)
        grouped = {c: np.flatnonzero(cluster == c) for c in uniq}
        for b in range(int(n_bootstrap)):
            sampled = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([grouped[c] for c in sampled])
            vals[b] = float(np.mean(values[idx]))
    return (
        float(np.quantile(vals, alpha / 2.0)),
        float(np.quantile(vals, 1.0 - alpha / 2.0)),
    )


def _summarize_outer_dir(
    outer_dir: Path,
    model: str,
    recalibrate_tau_blp: bool,
) -> dict:
    meta_path = outer_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing required outer-test artifact under {outer_dir}")

    outer = _load_outer_split(meta_path)
    df = _load_tau_test(outer_dir)
    tau_hat, alpha, beta = _tau_for_summary(df, outer_dir, recalibrate_tau_blp)

    return {
        "outer_tag": outer_dir.name,
        "model": model,
        "train_start": int(outer["train_start"]),
        "train_end": int(outer["train_end"]),
        "test_start": int(outer["test_start"]),
        "test_end": int(outer["test_end"]),
        "n_test": int(len(df)),
        "tau_mean": float(np.mean(tau_hat)),
        "tau_std": float(np.std(tau_hat, ddof=1)) if len(tau_hat) > 1 else np.nan,
        "tau_q05": float(np.quantile(tau_hat, 0.05)),
        "tau_q50": float(np.quantile(tau_hat, 0.50)),
        "tau_q95": float(np.quantile(tau_hat, 0.95)),
        "share_tau_positive": float(np.mean(tau_hat > 0.0)),
        "tau_calibration_applied": bool(recalibrate_tau_blp),
        "tau_calibration_alpha": alpha,
        "tau_calibration_beta": beta,
    }


def _compute_model_ci(
    base_dir: Path,
    model: str,
    n_bootstrap: int,
    bootstrap_alpha: float,
    cluster_col: str,
    random_state: int,
    recalibrate_tau_blp: bool,
) -> dict:
    dfs = []
    for outer_dir in _outer_dirs(base_dir):
        df = _load_tau_test(outer_dir).copy()
        tau_hat, _, _ = _tau_for_summary(df, outer_dir, recalibrate_tau_blp)
        df["_tau_hat_summary"] = tau_hat
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    tau_hat = df["_tau_hat_summary"].to_numpy(dtype=float)
    positive = (tau_hat > 0.0).astype(float)
    cluster = df[cluster_col].to_numpy() if cluster_col in df.columns else None
    tau_ci_lo, tau_ci_hi = _bootstrap_mean_ci(
        tau_hat,
        n_bootstrap=n_bootstrap,
        alpha=bootstrap_alpha,
        random_state=random_state,
        cluster=cluster,
    )
    share_ci_lo, share_ci_hi = _bootstrap_mean_ci(
        positive,
        n_bootstrap=n_bootstrap,
        alpha=bootstrap_alpha,
        random_state=random_state + 1003,
        cluster=cluster,
    )
    return {
        "model": model,
        "n_test_total": int(len(df)),
        "n_clusters": int(pd.unique(cluster).shape[0]) if cluster is not None else int(len(df)),
        "cluster_col": cluster_col if cluster is not None else "",
        "tau_mean": float(np.mean(tau_hat)),
        "tau_mean_ci_lo": tau_ci_lo,
        "tau_mean_ci_hi": tau_ci_hi,
        "share_tau_positive": float(np.mean(positive)),
        "share_tau_positive_ci_lo": share_ci_lo,
        "share_tau_positive_ci_hi": share_ci_hi,
        "tau_calibration_applied": bool(recalibrate_tau_blp),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_alpha": float(bootstrap_alpha),
    }


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model", sort=True):
        row = {
            "model": model,
            "n_outer": int(len(g)),
            "n_test_total": int(g["n_test"].sum()),
            "n_test_mean": float(g["n_test"].mean()),
            "tau_mean": float(g["tau_mean"].mean()),
            "tau_mean_std_across_outer": float(g["tau_mean"].std(ddof=1)) if len(g) > 1 else np.nan,
            "tau_q05_mean": float(g["tau_q05"].mean()),
            "tau_q50_mean": float(g["tau_q50"].mean()),
            "tau_q95_mean": float(g["tau_q95"].mean()),
            "share_tau_positive_mean": float(g["share_tau_positive"].mean()),
            "weighted_tau_mean": float(np.average(g["tau_mean"], weights=g["n_test"])),
            "weighted_share_tau_positive": float(np.average(g["share_tau_positive"], weights=g["n_test"])),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize nested walk-forward outer-test CATE estimates.")
    ap.add_argument("--catboost-dir", default="results/nested_wf_catboost_gpu")
    ap.add_argument("--lgbm-dir", default="results/nested_wf_lgbm")
    ap.add_argument("--xgb-dir", default="results/nested_wf_xgb")
    ap.add_argument("--outdir", default="results/wf_outer_test_main")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-alpha", type=float, default=0.05)
    ap.add_argument("--cluster-col", default="GAME_ID")
    ap.add_argument("--random-state", type=int, default=123)
    ap.add_argument(
        "--recalibrate-tau-blp",
        action="store_true",
        help="Use each outer fold's train-side BLP alpha/beta to recalibrate tau_test before summarizing CATE.",
    )
    args = ap.parse_args()

    base_dirs = [Path(args.catboost_dir), Path(args.lgbm_dir), Path(args.xgb_dir)]
    rows = []
    ci_rows = []
    for base_dir in base_dirs:
        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")
        model = _model_name_from_dir(base_dir)
        for outer_dir in _outer_dirs(base_dir):
            rows.append(
                _summarize_outer_dir(
                    outer_dir=outer_dir,
                    model=model,
                    recalibrate_tau_blp=bool(args.recalibrate_tau_blp),
                )
            )
        ci_rows.append(
            _compute_model_ci(
                base_dir=base_dir,
                model=model,
                n_bootstrap=int(args.n_bootstrap),
                bootstrap_alpha=float(args.bootstrap_alpha),
                cluster_col=args.cluster_col,
                random_state=int(args.random_state),
                recalibrate_tau_blp=bool(args.recalibrate_tau_blp),
            )
        )

    per_outer = pd.DataFrame(rows).sort_values(["test_start", "model"]).reset_index(drop=True)
    agg = _aggregate(per_outer)
    ci_df = pd.DataFrame(ci_rows).sort_values("model").reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_outer_path = outdir / "wf_outer_test_cate_by_outer.csv"
    agg_path = outdir / "wf_outer_test_cate_aggregate.csv"
    ci_path = outdir / "wf_outer_test_cate_ci.csv"
    per_outer.to_csv(per_outer_path, index=False)
    agg.to_csv(agg_path, index=False)
    ci_df.to_csv(ci_path, index=False)

    print(f"Wrote {per_outer_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {ci_path}")
    print("\nPer-outer CATE summary:")
    print(per_outer.to_string(index=False))
    print("\nAggregate summary:")
    print(agg.to_string(index=False))
    print("\nCluster-bootstrap CATE CI summary:")
    print(ci_df.to_string(index=False))


if __name__ == "__main__":
    main()
