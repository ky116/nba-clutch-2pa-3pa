#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from dml_dr_core import DMLConfig
from run_nested_walk_forward import (
    _collect_nuisance_oos,
    _compute_blp_metrics_robust,
    _default_outcome_grid,
    _default_prop_grid,
    _default_tau_grid,
    _log_progress,
    _parse_csv_list,
    default_feature_cols,
    fit_nuisance_full,
    fit_tau_full,
    iter_inner_splits,
    make_tau_oos,
    tune_nuisance_walk_forward,
    tune_tau_walk_forward,
)
from treatment_utils import apply_treatment_scheme


DEFAULT_INPUT = "data/analysis/shotchoice_panel_clutch_rs.parquet"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retune hyperparameters on full data and train final nuisance/tau CATE artifacts."
    )
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--outdir", default="results/full_data_catboost_state_retuned")

    p.add_argument("--train-start", type=int, default=2000)
    p.add_argument("--max-season", type=int, default=None)

    p.add_argument("--inner-train-init-span", type=int, default=4)
    p.add_argument("--inner-block-span", type=int, default=3)
    p.add_argument("--inner-step", type=int, default=3)

    p.add_argument("--treat-col", default="shot_zone_choice")
    p.add_argument("--outcome-col", default="delta_wp")
    p.add_argument("--season-col", default="season")
    p.add_argument("--treat-a", default="three-point")
    p.add_argument("--treat-b", default="two-point")
    p.add_argument("--treatment-scheme", default="binary")

    p.add_argument("--prop-model", default="catboost", choices=["xgb", "lgbm", "catboost"])
    p.add_argument("--outcome-model", default="catboost", choices=["xgb", "lgbm", "catboost"])
    p.add_argument("--tau-model", default="catboost", choices=["xgb", "lgbm", "catboost"])

    p.add_argument("--features", default=None, help="comma-separated feature columns (optional)")

    p.add_argument("--cluster-col", default="GAME_ID")
    p.add_argument("--random-state", type=int, default=123)
    p.add_argument("--min-samples-per-treat", type=int, default=200)
    p.add_argument("--min-prop", type=float, default=1e-2)
    p.add_argument("--max-prop", type=float, default=1.0)
    p.add_argument("--disable-early-stopping", action="store_true")
    p.add_argument("--es-rounds", type=int, default=200)
    p.add_argument("--final-es-tail-span", type=int, default=3)
    p.add_argument(
        "--oof-scheme",
        default="inner_splits",
        choices=["inner_splits", "season_loso"],
        help="How to build OOF nuisance/tau artifacts for diagnostics/calibration.",
    )

    p.add_argument("--use-fixed-hparams", action="store_true", help="Skip retuning and use provided fixed hyperparameters.")
    p.add_argument(
        "--fixed-prop-params-json",
        default=None,
        help="JSON string or JSON file path for propensity model hyperparameters.",
    )
    p.add_argument(
        "--fixed-outcome-params-json",
        default=None,
        help="JSON string or JSON file path for outcome model hyperparameters.",
    )
    p.add_argument(
        "--fixed-tau-params-json",
        default=None,
        help="JSON string or JSON file path for tau model hyperparameters.",
    )

    return p


def _save_tuning_table(path: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    if "params" in out.columns:
        out["params"] = out["params"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
        )
    out.to_csv(path, index=False)


def _load_json_arg(arg_val: Optional[str]) -> Optional[Dict[str, Any]]:
    if arg_val is None:
        return None
    s = str(arg_val).strip()
    if not s:
        return None
    p = Path(s)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(s)


def _build_season_loso_oos(
    df: pd.DataFrame,
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    treat_a: str,
    treat_b: str,
    nuisance_cfg: DMLConfig,
    tau_model: str,
    tau_params: Dict[str, Any],
    min_prop: float,
    max_prop: float,
    random_state: int,
    enable_early_stopping: bool,
    es_rounds: int,
    final_es_tail_span: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(pd.unique(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int)))
    if len(seasons) < 2:
        raise ValueError("season_loso OOF requires at least 2 seasons.")

    treat_levels = df[treat_col].astype("category").cat.categories.tolist()
    m_cols = [f"m_hat_{lvl}" for lvl in treat_levels]
    e_cols = [f"e_hat_{lvl}" for lvl in treat_levels]

    df_n_oos = df.copy()
    for c in m_cols + e_cols:
        df_n_oos[c] = np.nan

    df_tau_oos = df.copy()
    df_tau_oos["tau_hat"] = np.nan

    for i, s in enumerate(seasons, start=1):
        _log_progress(f"season-LOSO OOF: season={s} ({i}/{len(seasons)})")
        df_tr = df[df["season"].astype(int) != int(s)].copy()
        df_va = df[df["season"].astype(int) == int(s)].copy()
        if df_tr.empty or df_va.empty:
            continue

        df_tr_n = fit_nuisance_full(
            df_train=df_tr,
            df_apply=df_tr,
            feature_cols=feature_cols,
            treat_col=treat_col,
            outcome_col=outcome_col,
            config=nuisance_cfg,
            enable_early_stopping=enable_early_stopping,
            es_rounds=es_rounds,
            final_es_tail_span=final_es_tail_span,
            eval_df_for_es=df_va if enable_early_stopping else None,
        )
        df_va_n = fit_nuisance_full(
            df_train=df_tr,
            df_apply=df_va,
            feature_cols=feature_cols,
            treat_col=treat_col,
            outcome_col=outcome_col,
            config=nuisance_cfg,
            enable_early_stopping=enable_early_stopping,
            es_rounds=es_rounds,
            final_es_tail_span=final_es_tail_span,
            eval_df_for_es=df_va if enable_early_stopping else None,
        )

        va_idx = df_va.index.to_numpy()
        for c in m_cols + e_cols:
            if c in df_va_n.columns:
                df_n_oos.loc[va_idx, c] = df_va_n[c].to_numpy()

        df_va_tau, _ = fit_tau_full(
            df_train=df_tr_n,
            df_apply=df_va_n,
            feature_cols=feature_cols,
            treat_col=treat_col,
            outcome_col=outcome_col,
            tau_model=tau_model,
            tau_params=tau_params,
            min_prop=min_prop,
            max_prop=max_prop,
            treat_a=treat_a,
            treat_b=treat_b,
            random_state=random_state,
            enable_early_stopping=enable_early_stopping,
            es_rounds=es_rounds,
            final_es_tail_span=final_es_tail_span,
        )
        df_tau_oos.loc[va_idx, "tau_hat"] = df_va_tau["tau_hat"].to_numpy(dtype=float)

    return df_n_oos, df_tau_oos


def main() -> None:
    args = build_parser().parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _log_progress(f"start full-data training input={args.input} outdir={args.outdir}")
    df = pd.read_parquet(args.input)

    if args.season_col not in df.columns:
        raise KeyError(f"season column not found: {args.season_col}")
    df[args.season_col] = pd.to_numeric(df[args.season_col], errors="coerce").astype(int)
    df = df.rename(columns={args.season_col: "season"})

    if args.treat_col not in df.columns:
        raise KeyError(f"treatment column not found: {args.treat_col}")
    df = apply_treatment_scheme(
        df,
        treat_col=args.treat_col,
        scheme=args.treatment_scheme,
        out_col=args.treat_col,
        drop_unknown=True,
    )
    if df.empty:
        raise ValueError("No rows left after treatment mapping. Check --treatment-scheme and labels.")

    max_season = args.max_season or int(df["season"].max())
    df_train = df[(df["season"] >= args.train_start) & (df["season"] <= max_season)].copy()
    if df_train.empty:
        raise ValueError("No training rows in requested season range.")

    if args.features:
        feature_cols = _parse_csv_list(args.features, str)
    else:
        feature_cols = default_feature_cols(df_train)
    missing = [c for c in feature_cols if c not in df_train.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    treat_levels = df_train[args.treat_col].astype("category").cat.categories.tolist()
    if args.treat_a not in treat_levels or args.treat_b not in treat_levels:
        raise ValueError(
            f"--treat-a/--treat-b must exist after mapping. "
            f"Got treat_a={args.treat_a}, treat_b={args.treat_b}, levels={treat_levels}"
        )

    inner_splits = list(
        iter_inner_splits(
            train_start=args.train_start,
            train_end_max=max_season,
            train_init_span=args.inner_train_init_span,
            block_span=args.inner_block_span,
            step=args.inner_step,
        )
    )
    if not inner_splits:
        raise ValueError("No inner splits created. Check full-data season range and inner split args.")

    use_early_stopping = not args.disable_early_stopping

    _log_progress(
        f"prepared full-data rows={len(df_train):,} seasons={int(df_train['season'].min())}-{int(df_train['season'].max())} "
        f"features={len(feature_cols)} inner_splits={len(inner_splits)}"
    )

    fixed_prop = _load_json_arg(args.fixed_prop_params_json)
    fixed_out = _load_json_arg(args.fixed_outcome_params_json)
    fixed_tau = _load_json_arg(args.fixed_tau_params_json)
    use_fixed = bool(args.use_fixed_hparams)
    if use_fixed and (fixed_prop is None or fixed_out is None or fixed_tau is None):
        raise ValueError(
            "--use-fixed-hparams requires all of "
            "--fixed-prop-params-json, --fixed-outcome-params-json, --fixed-tau-params-json."
        )

    if use_fixed:
        best_prop = dict(fixed_prop)
        best_out = dict(fixed_out)
        best_tau = dict(fixed_tau)
        tune_df = pd.DataFrame(
            [
                dict(
                    mode="fixed",
                    model=args.prop_model,
                    params=json.dumps(best_prop, ensure_ascii=False),
                    note="retuning_skipped",
                ),
                dict(
                    mode="fixed",
                    model=args.outcome_model,
                    params=json.dumps(best_out, ensure_ascii=False),
                    note="retuning_skipped",
                ),
            ]
        )
        tau_tune_df = pd.DataFrame(
            [
                dict(
                    mode="fixed",
                    model=args.tau_model,
                    params=json.dumps(best_tau, ensure_ascii=False),
                    note="retuning_skipped",
                )
            ]
        )
        _save_tuning_table(outdir / "nuisance_tuning.csv", tune_df)
        _save_tuning_table(outdir / "tau_tuning.csv", tau_tune_df)
        _log_progress("fixed hyperparameters loaded; nuisance/tau retuning skipped")
    else:
        _log_progress(f"nuisance tuning start ({args.prop_model}/{args.outcome_model})")
        best_prop, best_out, tune_df = tune_nuisance_walk_forward(
            df=df_train,
            splits=inner_splits,
            feature_cols=feature_cols,
            treat_col=args.treat_col,
            outcome_col=args.outcome_col,
            prop_model=args.prop_model,
            outcome_model=args.outcome_model,
            random_state=args.random_state,
            min_samples_per_treat=args.min_samples_per_treat,
            prop_grid=_default_prop_grid(args.prop_model),
            outcome_grid=_default_outcome_grid(args.outcome_model),
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            group_col=args.cluster_col,
        )
        _save_tuning_table(outdir / "nuisance_tuning.csv", tune_df)
        _log_progress("nuisance tuning done")

    nuisance_cfg = DMLConfig(
        n_splits=2,
        prop_model=args.prop_model,
        outcome_model=args.outcome_model,
        random_state=args.random_state,
        prop_params=best_prop,
        outcome_params=best_out,
        min_samples_per_treat=args.min_samples_per_treat,
        group_col=args.cluster_col,
    )

    _log_progress("full-data nuisance fit start")
    df_train_full = fit_nuisance_full(
        df_train=df_train,
        df_apply=df_train,
        feature_cols=feature_cols,
        treat_col=args.treat_col,
        outcome_col=args.outcome_col,
        config=nuisance_cfg,
        enable_early_stopping=use_early_stopping,
        es_rounds=args.es_rounds,
        final_es_tail_span=args.final_es_tail_span,
    )
    df_train_full_path = outdir / "nuisance_full_train.parquet"
    df_train_full.to_parquet(df_train_full_path, index=False)
    _log_progress(f"full-data nuisance saved -> {df_train_full_path}")

    if not use_fixed:
        _log_progress(f"tau tuning start ({args.tau_model})")
        best_tau, tau_tune_df = tune_tau_walk_forward(
            splits=inner_splits,
            df=df_train,
            feature_cols=feature_cols,
            treat_col=args.treat_col,
            outcome_col=args.outcome_col,
            model_name=args.tau_model,
            random_state=args.random_state,
            nuisance_cfg=nuisance_cfg,
            param_grid=_default_tau_grid(args.tau_model),
            min_prop=args.min_prop,
            max_prop=args.max_prop,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
        _save_tuning_table(outdir / "tau_tuning.csv", tau_tune_df)
        _log_progress("tau tuning done")

    _log_progress("full-data tau fit start")
    df_train_tau_full, tau_payload = fit_tau_full(
        df_train=df_train_full,
        df_apply=df_train_full,
        feature_cols=feature_cols,
        treat_col=args.treat_col,
        outcome_col=args.outcome_col,
        tau_model=args.tau_model,
        tau_params=best_tau,
        min_prop=args.min_prop,
        max_prop=args.max_prop,
        treat_a=args.treat_a,
        treat_b=args.treat_b,
        random_state=args.random_state,
        enable_early_stopping=use_early_stopping,
        es_rounds=args.es_rounds,
        final_es_tail_span=args.final_es_tail_span,
    )
    df_train_tau_full_path = outdir / "tau_full_train.parquet"
    df_train_tau_full.to_parquet(df_train_tau_full_path, index=False)
    tau_model_path = outdir / "tau_model.joblib"
    dump(tau_payload, tau_model_path)
    _log_progress(f"full-data tau saved -> {df_train_tau_full_path}")
    _log_progress(f"tau model saved -> {tau_model_path}")

    # OOF nuisance/tau artifacts for diagnostics/calibration
    _log_progress(f"OOF nuisance/tau build start scheme={args.oof_scheme}")
    if str(args.oof_scheme) == "season_loso":
        df_train_oos, df_train_tau_oos = _build_season_loso_oos(
            df=df_train,
            feature_cols=feature_cols,
            treat_col=args.treat_col,
            outcome_col=args.outcome_col,
            treat_a=args.treat_a,
            treat_b=args.treat_b,
            nuisance_cfg=nuisance_cfg,
            tau_model=args.tau_model,
            tau_params=best_tau,
            min_prop=args.min_prop,
            max_prop=args.max_prop,
            random_state=args.random_state,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
    else:
        df_train_oos, _ = _collect_nuisance_oos(
            df=df_train,
            splits=inner_splits,
            feature_cols=feature_cols,
            treat_col=args.treat_col,
            outcome_col=args.outcome_col,
            config=nuisance_cfg,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
        )
        df_train_tau_oos = make_tau_oos(
            df=df_train,
            splits=inner_splits,
            feature_cols=feature_cols,
            treat_col=args.treat_col,
            outcome_col=args.outcome_col,
            nuisance_cfg=nuisance_cfg,
            tau_model=args.tau_model,
            tau_params=best_tau,
            min_prop=args.min_prop,
            max_prop=args.max_prop,
            treat_a=args.treat_a,
            treat_b=args.treat_b,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
    df_train_oos_path = outdir / "nuisance_oos_train.parquet"
    df_train_oos.to_parquet(df_train_oos_path, index=False)
    _log_progress(f"OOF nuisance saved -> {df_train_oos_path}")
    df_train_tau_oos_path = outdir / "tau_oos_train.parquet"
    df_train_tau_oos.to_parquet(df_train_tau_oos_path, index=False)
    _log_progress(f"OOF tau saved -> {df_train_tau_oos_path}")

    # OOS BLP diagnostics on tau_hat vs DR pseudo-outcome for tau
    df_blp_oos = df_train_oos.copy()
    df_blp_oos["tau_hat"] = df_train_tau_oos["tau_hat"].to_numpy(dtype=float)
    req_blp_numeric_cols = [
        args.outcome_col,
        f"m_hat_{args.treat_a}",
        f"m_hat_{args.treat_b}",
        f"e_hat_{args.treat_a}",
        "tau_hat",
    ]
    mask_blp = np.ones(len(df_blp_oos), dtype=bool)
    for c in req_blp_numeric_cols:
        mask_blp &= np.isfinite(pd.to_numeric(df_blp_oos[c], errors="coerce").to_numpy(dtype=float))
    mask_blp &= df_blp_oos[args.treat_col].astype(str).isin([str(args.treat_a), str(args.treat_b)]).to_numpy()
    if args.cluster_col in df_blp_oos.columns:
        mask_blp &= pd.notna(df_blp_oos[args.cluster_col]).to_numpy()
    df_blp_oos = df_blp_oos.loc[mask_blp].copy()
    if df_blp_oos.empty:
        raise ValueError("No finite OOS rows for BLP diagnostics.")

    y_blp = df_blp_oos[args.outcome_col].to_numpy(dtype=float)
    z_blp = df_blp_oos[args.treat_col].astype(str).to_numpy()
    w_blp = (z_blp == str(args.treat_a)).astype(float)
    m1_blp = df_blp_oos[f"m_hat_{args.treat_a}"].to_numpy(dtype=float)
    m0_blp = df_blp_oos[f"m_hat_{args.treat_b}"].to_numpy(dtype=float)
    e_blp = np.clip(
        df_blp_oos[f"e_hat_{args.treat_a}"].to_numpy(dtype=float),
        float(args.min_prop),
        1.0 - 1e-6,
    )
    tau_hat_blp = df_blp_oos["tau_hat"].to_numpy(dtype=float)
    psi_tau_blp = (
        (m1_blp - m0_blp)
        + w_blp * (y_blp - m1_blp) / e_blp
        - (1.0 - w_blp) * (y_blp - m0_blp) / (1.0 - e_blp)
    )
    cluster_blp = (
        df_blp_oos[args.cluster_col].to_numpy()
        if args.cluster_col in df_blp_oos.columns
        else None
    )
    blp_metrics = _compute_blp_metrics_robust(
        tau_hat=tau_hat_blp,
        psi_tau=psi_tau_blp,
        cluster=cluster_blp,
    )
    blp_metrics_path = outdir / "blp_metrics_full_data.json"
    blp_metrics_path.write_text(json.dumps(blp_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_progress(f"BLP metrics saved -> {blp_metrics_path}")

    meta: Dict[str, Any] = dict(
        mode="full_data_retune_then_train",
        train_start=int(args.train_start),
        train_end=int(max_season),
        n_rows=int(len(df_train)),
        feature_cols=feature_cols,
        selected_models=dict(
            propensity=args.prop_model,
            outcome=args.outcome_model,
            tau=args.tau_model,
        ),
        nuisance_best_prop=best_prop,
        nuisance_best_outcome=best_out,
        tau_best_params=best_tau,
        oof_scheme=str(args.oof_scheme),
        inner_split_count=len(inner_splits),
        inner_splits=[s.__dict__ for s in inner_splits],
        early_stopping=use_early_stopping,
        nuisance_oos_path=str(df_train_oos_path),
        tau_oos_path=str(df_train_tau_oos_path),
        blp_metrics=blp_metrics,
        blp_metrics_path=str(blp_metrics_path),
    )
    meta_path = outdir / "meta_full_data.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _log_progress(f"meta saved -> {meta_path}")

    print(f"[done] full-data retune/train -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
