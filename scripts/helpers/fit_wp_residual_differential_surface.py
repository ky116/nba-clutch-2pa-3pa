#!/usr/bin/env python
"""Fit a differential WP calibration residual surface.

This is a reviewer-response diagnostic, not a WP model refit. It treats
offense-oriented next-state WP residuals as the outcome and estimates

    b(x) = E[R(3PA) - R(2PA) | X=x]

with the same DR pseudo-outcome machinery and CATE surface marginalization used
for the main 3PA-vs-2PA analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

PROJECT_DIR = Path(__file__).resolve().parents[2]
CORE_DIR = PROJECT_DIR / "scripts" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from dml_dr_core import DMLConfig, crossfit_nuisance  # noqa: E402
from run_nested_walk_forward import (  # noqa: E402
    default_feature_cols,
    fit_tau_full,
)
from plot_cate_surface_gcomp import compute_surface  # noqa: E402
from treatment_utils import apply_treatment_scheme  # noqa: E402


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["GAME_ID"] = (
        out["GAME_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    )
    out["GAME_EVENT_ID"] = pd.to_numeric(out["GAME_EVENT_ID"], errors="coerce").astype("Int64")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    return out


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"GAME_ID", "GAME_EVENT_ID", "season", "shot_zone_choice", "time_left_game", "score_diff"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")
    return normalize_ids(df)


def load_residuals(with_wp: Path) -> pd.DataFrame:
    usecols = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "season",
        "next_is_terminal",
        "before_home_possession",
        "wp_next",
        "final_home_win",
    ]
    df = pd.read_csv(with_wp, usecols=usecols)
    df = normalize_ids(df)
    home_poss = pd.to_numeric(df["before_home_possession"], errors="coerce")
    home_wp = pd.to_numeric(df["wp_next"], errors="coerce")
    home_win = pd.to_numeric(df["final_home_win"], errors="coerce")
    df["wp_next_offense"] = np.where(home_poss.eq(1), home_wp, np.where(home_poss.eq(0), 1 - home_wp, np.nan))
    df["final_win_offense"] = np.where(home_poss.eq(1), home_win, np.where(home_poss.eq(0), 1 - home_win, np.nan))
    df["wp_residual_offense"] = df["wp_next_offense"] - df["final_win_offense"]
    df = df[~pd.to_numeric(df["next_is_terminal"], errors="coerce").eq(1)].copy()
    return df[["GAME_ID", "GAME_EVENT_ID", "season", "wp_residual_offense"]]


def attach_residual(panel: pd.DataFrame, residuals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    keys = ["GAME_ID", "GAME_EVENT_ID", "season"]
    panel_dup = panel.duplicated(keys, keep=False)
    resid_dup = residuals.duplicated(keys, keep=False)
    meta = {
        "panel_rows_input": int(len(panel)),
        "panel_rows_with_duplicate_event_key": int(panel_dup.sum()),
        "wp_rows_input_nonterminal": int(len(residuals)),
        "wp_rows_with_duplicate_event_key": int(resid_dup.sum()),
    }
    panel_clean = panel.loc[~panel_dup].copy()
    resid_clean = residuals.loc[~resid_dup].copy()
    out = panel_clean.merge(resid_clean, on=keys, how="inner")
    meta["panel_rows_after_duplicate_drop"] = int(len(panel_clean))
    meta["merged_rows"] = int(len(out))
    meta["panel_rows_without_residual_match"] = int(len(panel_clean) - len(out))
    return out, meta


def load_params(path: Path | None, key: str | None) -> dict:
    if path is None:
        return {}
    obj = json.loads(path.read_text())
    if key:
        obj = obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict params in {path}")
    return obj


def cap_estimators(params: dict, cap: int | None) -> dict:
    out = dict(params)
    if cap is None or cap <= 0:
        return out
    if "n_estimators" in out:
        out["n_estimators"] = min(int(out["n_estimators"]), int(cap))
    if "iterations" in out:
        out["iterations"] = min(int(out["iterations"]), int(cap))
    return out


def read_reference_tau_surface(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def add_adjusted_tau(
    bias_surface: pd.DataFrame,
    reference: pd.DataFrame | None,
    time_col: str,
    score_col: str,
) -> pd.DataFrame:
    out = bias_surface.rename(columns={"tau_mean": "b_hat", "tau_se": "b_hat_se"}).copy()
    if reference is None:
        return out
    ref = reference.copy()
    if "score_diff_offense" in ref.columns and score_col not in ref.columns:
        ref = ref.rename(columns={"score_diff_offense": score_col})
    tau_col = "tau_mean_ensemble" if "tau_mean_ensemble" in ref.columns else "tau_mean"
    if tau_col not in ref.columns:
        return out
    keep = [time_col, score_col, tau_col]
    ref = ref[keep].rename(columns={tau_col: "tau_wp"})
    out = out.merge(ref, on=[time_col, score_col], how="left")
    out["relative_abs_b_to_tau"] = np.where(
        out["tau_wp"].abs() > 0,
        out["b_hat"].abs() / out["tau_wp"].abs(),
        np.nan,
    )
    out["tau_adj"] = out["tau_wp"] - out["b_hat"]
    out["sign_preserved_after_adjustment"] = np.sign(out["tau_wp"]) == np.sign(out["tau_adj"])
    return out


def summarize_adjusted(surface: pd.DataFrame) -> pd.DataFrame:
    row: dict[str, float | int] = {
        "n_surface_cells": int(len(surface)),
        "mean_b_hat": float(surface["b_hat"].mean()),
        "median_b_hat": float(surface["b_hat"].median()),
        "max_abs_b_hat": float(surface["b_hat"].abs().max()),
    }
    if "tau_wp" in surface.columns:
        valid = surface["tau_wp"].notna() & surface["tau_adj"].notna()
        row.update(
            {
                "n_cells_with_tau_wp": int(valid.sum()),
                "mean_abs_b_to_tau": float(surface.loc[valid, "relative_abs_b_to_tau"].replace([np.inf, -np.inf], np.nan).mean()),
                "median_abs_b_to_tau": float(surface.loc[valid, "relative_abs_b_to_tau"].replace([np.inf, -np.inf], np.nan).median()),
                "max_abs_b_to_tau": float(surface.loc[valid, "relative_abs_b_to_tau"].replace([np.inf, -np.inf], np.nan).max()),
                "sign_preserved_share": float(surface.loc[valid, "sign_preserved_after_adjustment"].mean()),
                "n_sign_flips": int((~surface.loc[valid, "sign_preserved_after_adjustment"]).sum()),
            }
        )
    return pd.DataFrame([row])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/analysis/shotchoice_panel_clutch_rs.parquet"))
    p.add_argument("--with-wp", type=Path, default=Path("data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz"))
    p.add_argument("--outdir", type=Path, default=Path("results/wp_calibration/differential_surface"))
    p.add_argument("--treat-a", default="three-point")
    p.add_argument("--treat-b", default="two-point")
    p.add_argument("--treat-col", default="shot_zone_choice")
    p.add_argument("--treatment-scheme", default="binary", choices=["binary", "multi"])
    p.add_argument("--outcome-model", default="lgbm", choices=["xgb", "lgbm", "catboost"])
    p.add_argument("--prop-model", default="lgbm", choices=["xgb", "lgbm", "catboost"])
    p.add_argument("--tau-model", default="lgbm", choices=["xgb", "lgbm", "catboost"])
    p.add_argument("--params-json", type=Path, default=None)
    p.add_argument("--prop-params-key", default="nuisance_best_prop")
    p.add_argument("--outcome-params-key", default="nuisance_best_outcome")
    p.add_argument("--tau-params-key", default="tau_best_params")
    p.add_argument("--max-estimators", type=int, default=1200)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--min-samples-per-treat", type=int, default=200)
    p.add_argument("--min-prop", type=float, default=0.01)
    p.add_argument("--max-prop", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-time", type=int, default=60)
    p.add_argument("--n-score", type=int, default=21)
    p.add_argument("--time-lo", type=float, default=15)
    p.add_argument("--time-hi", type=float, default=300)
    p.add_argument("--score-lo", type=float, default=-10)
    p.add_argument("--score-hi", type=float, default=10)
    p.add_argument("--n-sample", type=int, default=100_000)
    p.add_argument("--reference-tau-surface", type=Path, default=None)
    p.add_argument("--no-early-stopping", action="store_true")
    args = p.parse_args()

    panel = load_panel(args.panel)
    residuals = load_residuals(args.with_wp)
    df, merge_meta = attach_residual(panel, residuals)
    df = apply_treatment_scheme(
        df,
        treat_col=args.treat_col,
        scheme=args.treatment_scheme,
        out_col=args.treat_col,
        drop_unknown=True,
    )
    df = df[df[args.treat_col].astype(str).isin([args.treat_a, args.treat_b])].copy()
    df = df.dropna(subset=["wp_residual_offense"]).reset_index(drop=True)
    df[args.treat_col] = df[args.treat_col].astype("category").cat.set_categories([args.treat_b, args.treat_a])
    df = df.dropna(subset=[args.treat_col]).copy()

    feature_cols = default_feature_cols(df)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    prop_params = cap_estimators(load_params(args.params_json, args.prop_params_key), args.max_estimators)
    outcome_params = cap_estimators(load_params(args.params_json, args.outcome_params_key), args.max_estimators)
    tau_params = cap_estimators(load_params(args.params_json, args.tau_params_key), args.max_estimators)

    cfg = DMLConfig(
        n_splits=args.n_splits,
        outcome_model=args.outcome_model,
        prop_model=args.prop_model,
        random_state=args.seed,
        outcome_params=outcome_params,
        prop_params=prop_params,
        min_samples_per_treat=args.min_samples_per_treat,
        group_col="GAME_ID" if "GAME_ID" in df.columns else None,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"[info] residual diagnostic rows={len(df):,}, features={len(feature_cols)}")
    nuisance = crossfit_nuisance(
        df=df,
        feature_cols=feature_cols,
        treat_col=args.treat_col,
        outcome_col="wp_residual_offense",
        config=cfg,
    )
    nuisance_path = args.outdir / "wp_residual_nuisance_oof.parquet"
    nuisance.to_parquet(nuisance_path, index=False)
    print(f"[info] wrote {nuisance_path}")

    tau_full, payload = fit_tau_full(
        df_train=nuisance,
        df_apply=df,
        feature_cols=feature_cols,
        treat_col=args.treat_col,
        outcome_col="wp_residual_offense",
        tau_model=args.tau_model,
        tau_params=tau_params,
        min_prop=args.min_prop,
        max_prop=args.max_prop,
        treat_a=args.treat_a,
        treat_b=args.treat_b,
        random_state=args.seed,
        enable_early_stopping=not args.no_early_stopping,
    )
    tau_path = args.outdir / "wp_residual_bias_tau_full.parquet"
    model_path = args.outdir / "wp_residual_bias_tau_model.joblib"
    tau_full.to_parquet(tau_path, index=False)
    dump(payload, model_path)
    print(f"[info] wrote {tau_path}")
    print(f"[info] wrote {model_path}")

    surface, _, _ = compute_surface(
        payload=payload,
        df=df,
        treat_a=args.treat_a,
        treat_b=args.treat_b,
        time_col="time_left_game",
        score_col="score_diff",
        score_out_col="score_diff",
        score_perspective="offense",
        n_time=args.n_time,
        n_score=args.n_score,
        qlo=0.05,
        qhi=0.95,
        time_lo=args.time_lo,
        time_hi=args.time_hi,
        score_lo=args.score_lo,
        score_hi=args.score_hi,
        n_sample=args.n_sample,
        seed=args.seed,
        tau_threshold=0.001,
        bootstrap=0,
        tau_calib_alpha=None,
        tau_calib_beta=None,
    )
    ref = read_reference_tau_surface(args.reference_tau_surface)
    adjusted = add_adjusted_tau(surface, ref, time_col="time_left_game", score_col="score_diff")
    surface_path = args.outdir / "wp_residual_differential_calibration_surface.csv"
    summary_path = args.outdir / "wp_residual_differential_calibration_surface_summary.csv"
    adjusted.to_csv(surface_path, index=False)
    summarize_adjusted(adjusted).to_csv(summary_path, index=False)

    meta = {
        **merge_meta,
        "rows_used": int(len(df)),
        "feature_cols": feature_cols,
        "treat_a": args.treat_a,
        "treat_b": args.treat_b,
        "outcome": "wp_residual_offense",
        "residual_definition": "offense_wp_next - offense_final_win",
        "params_json": str(args.params_json) if args.params_json else None,
        "prop_params": prop_params,
        "outcome_params": outcome_params,
        "tau_params": tau_params,
        "max_estimators": args.max_estimators,
        "reference_tau_surface": str(args.reference_tau_surface) if args.reference_tau_surface else None,
    }
    meta_path = args.outdir / "wp_residual_differential_calibration_surface_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[info] wrote {surface_path}")
    print(f"[info] wrote {summary_path}")
    print(f"[info] wrote {meta_path}")
    print(summarize_adjusted(adjusted).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
