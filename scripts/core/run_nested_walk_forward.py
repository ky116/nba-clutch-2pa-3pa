#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested walk-forward orchestration:

Outer: train/test in 3-season blocks (expanding train).
Inner (within outer train):
  1) Tune nuisance (propensity/outcome) by walk-forward (past -> next block).
  2) Generate OOS nuisance for each inner validation block.
  3) Tune DR-learner tau-model by walk-forward using the tuned nuisance.
  4) Generate OOS tau_hat for each inner validation block.

All artifacts are saved per outer split.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from dml_dr_core import DMLConfig
from dml_models import make_classifier, make_regressor
from train_dr_learner import (
    compute_pseudo_outcomes,
    infer_categorical_cols,
)
from treatment_utils import add_treatment_scheme_arg, apply_treatment_scheme

DEFAULT_RANDOM_SEARCH_TRIALS = 100


def _log_progress(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[progress {ts}] {msg}", flush=True)


def _parse_csv_list(s: str, cast=float) -> List:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def _catboost_use_gpu() -> bool:
    return str(os.getenv("CATBOOST_USE_GPU", "0")).strip() == "1"


def _make_ohe_sparse():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _infer_categorical_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    cat_cols: List[str] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        dt = df[c].dtype
        if (
            pd.api.types.is_object_dtype(dt)
            or pd.api.types.is_string_dtype(dt)
            or str(dt).startswith("category")
        ):
            cat_cols.append(c)
    return cat_cols


def _build_preprocessor(feature_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_cols = [c for c in feature_cols if c not in categorical_cols]
    ohe = _make_ohe_sparse()
    transformers = []
    if categorical_cols:
        transformers.append(("cat", ohe, categorical_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    return ColumnTransformer(transformers, remainder="drop")


def default_feature_cols(df: pd.DataFrame, include_team_ids: bool = False) -> List[str]:
    candidate = [
        "time_left_game",
        "score_diff",
        "OT_flag",
        "start_type",
        "start_type_group",
        "after_off_reb",
        "elo_diff",
        "before_home_possession",
        "era",
        "own_fouls_period",
        "opp_fouls_period",
        "timeouts_left_us",
        "timeouts_left_them",
    ]
    if include_team_ids:
        candidate.extend(["offense_team", "defense_team"])
    redundant_share_cols = {
        "own_allowed_2pa_share_eb",
        "opp_allowed_2pa_share_eb",
    }
    team_feature_cols = [
        c for c in df.columns
        if (c.startswith("own_") or c.startswith("opp_")) and c.endswith("_eb")
        and c not in redundant_share_cols
    ]
    for c in team_feature_cols:
        if c not in candidate:
            candidate.append(c)
    return [c for c in candidate if c in df.columns]


@dataclass(frozen=True)
class SplitRange:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def iter_outer_splits(
    train_start: int,
    train_end_init: int,
    test_span: int,
    step: int,
    max_season: int,
) -> Iterable[SplitRange]:
    train_end = train_end_init
    while True:
        test_start = train_end + 1
        test_end = test_start + test_span - 1
        if test_end > max_season:
            break
        yield SplitRange(train_start, train_end, test_start, test_end)
        train_end += step


def iter_inner_splits(
    train_start: int,
    train_end_max: int,
    train_init_span: int,
    block_span: int,
    step: int,
) -> Iterable[SplitRange]:
    train_end = train_start + train_init_span - 1
    while True:
        test_start = train_end + 1
        test_end = test_start + block_span - 1
        if test_end > train_end_max:
            break
        yield SplitRange(train_start, train_end, test_start, test_end)
        train_end += step


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def _sample_param_candidates(
    model_name: str,
    params_list: List[Dict[str, Any]],
    random_state: int,
    n_trials: int = DEFAULT_RANDOM_SEARCH_TRIALS,
) -> List[Dict[str, Any]]:
    # Keep CatBoost as exhaustive grid; downsample only XGB/LGBM grids.
    if model_name not in {"xgb", "lgbm"}:
        return params_list
    if n_trials <= 0 or len(params_list) <= n_trials:
        return params_list
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(len(params_list), size=int(n_trials), replace=False)
    return [params_list[int(i)] for i in idx]


def _split_tail_eval(
    df: pd.DataFrame,
    season_col: str = "season",
    tail_span: int = 3,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if season_col not in df.columns or tail_span <= 0:
        return None, None
    seasons = np.sort(df[season_col].dropna().astype(int).unique())
    if len(seasons) <= tail_span:
        return None, None
    tail = set(seasons[-tail_span:].tolist())
    df_es_val = df[df[season_col].astype(int).isin(tail)].copy()
    df_es_train = df[~df[season_col].astype(int).isin(tail)].copy()
    if df_es_train.empty or df_es_val.empty:
        return None, None
    return df_es_train, df_es_val


def _default_prop_grid(model_name: str) -> Dict[str, List[Any]]:
    if model_name == "xgb":
        return {
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_weight": [5, 20],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "reg_lambda": [1, 5, 10],
            "reg_alpha": [0.0, 0.5],
            "gamma": [0, 1],
            "n_estimators": [5000],
        }
    if model_name == "lgbm":
        return {
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [31, 63],
            "min_data_in_leaf": [50, 150, 300],
            "feature_fraction": [0.7, 0.9],
            "bagging_fraction": [0.7, 0.9],
            "bagging_freq": [1],
            "lambda_l2": [0, 5, 20],
            "lambda_l1": [0, 1],
            "min_gain_to_split": [0, 0.5],
            "n_estimators": [5000],
        }
    if model_name == "catboost":
        grid = {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "l2_leaf_reg": [3, 10, 30],
            "min_data_in_leaf": [50, 150, 300],
            "iterations": [6000],
            "od_wait": [200],
        }
        # CatBoost GPU does not support rsm for non-pairwise objectives.
        if not _catboost_use_gpu():
            grid["rsm"] = [0.7, 0.9]
        return grid
    raise ValueError(f"Unknown model_name: {model_name}")


def _default_outcome_grid(model_name: str) -> Dict[str, List[Any]]:
    return _default_prop_grid(model_name)


def _default_tau_grid(model_name: str) -> Dict[str, List[Any]]:
    if model_name == "xgb":
        return {
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05],
            "min_child_weight": [20, 50],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "reg_lambda": [5, 10, 30],
            "reg_alpha": [0.0, 0.5, 1.0],
            "gamma": [1, 5],
            "n_estimators": [8000],
        }
    if model_name == "lgbm":
        return {
            "learning_rate": [0.03, 0.05],
            "num_leaves": [31, 63],
            "min_data_in_leaf": [200, 500, 1000],
            "feature_fraction": [0.7, 0.9],
            "bagging_fraction": [0.7, 0.9],
            "bagging_freq": [1],
            "lambda_l2": [5, 20, 80],
            "lambda_l1": [0, 1],
            "min_gain_to_split": [0.5, 1.0],
            "n_estimators": [8000],
        }
    if model_name == "catboost":
        grid = {
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "l2_leaf_reg": [3, 10, 30],
            "min_data_in_leaf": [100, 200, 500],
            "iterations": [8000],
            "od_wait": [300],
        }
        # CatBoost GPU does not support rsm for non-pairwise objectives.
        if not _catboost_use_gpu():
            grid["rsm"] = [0.7, 0.9]
        return grid
    raise ValueError(f"Unknown model_name: {model_name}")


@dataclass
class FittedNuisance:
    treat_levels: List[Any]
    cat_cols: List[str]
    pre: Optional[ColumnTransformer]
    prop_model_name: str
    outcome_model_name: str
    prop_model: Optional[Any]
    prop_fallback: Optional[np.ndarray]
    outcome_models: Dict[int, Optional[Any]]
    outcome_fallback: Dict[int, float]


def _best_num_trees(model_name: str, fitted: Any) -> Optional[int]:
    if model_name == "xgb":
        best = getattr(fitted, "best_iteration", None)
        if best is None:
            return None
        return int(best) + 1
    if model_name == "lgbm":
        best = getattr(fitted, "best_iteration_", None)
        if best is None:
            return None
        return int(best)
    if model_name == "catboost":
        if hasattr(fitted, "get_best_iteration"):
            best = int(fitted.get_best_iteration())
            if best >= 0:
                return best + 1
        return None
    return None


def _set_num_trees(params: Dict[str, Any], model_name: str, n_trees: int) -> Dict[str, Any]:
    out = dict(params)
    if model_name in {"xgb", "lgbm"}:
        out["n_estimators"] = int(n_trees)
    elif model_name == "catboost":
        out["iterations"] = int(n_trees)
    return out


def _fit_classifier_with_es(
    model_name: str,
    est: Any,
    X_train: Any,
    y_train: np.ndarray,
    cat_cols: List[str],
    X_eval: Optional[Any],
    y_eval: Optional[np.ndarray],
    es_rounds: int,
) -> Any:
    if X_eval is None or y_eval is None:
        if model_name == "catboost":
            return est.fit(X_train, y_train, cat_features=cat_cols)
        return est.fit(X_train, y_train)

    if model_name == "catboost":
        return est.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_eval, y_eval),
            use_best_model=True,
            early_stopping_rounds=es_rounds,
            verbose=False,
        )
    if model_name == "lgbm":
        try:
            import lightgbm as lgb

            return est.fit(
                X_train,
                y_train,
                eval_X=X_eval,
                eval_y=y_eval,
                callbacks=[
                    lgb.early_stopping(es_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        except TypeError:
            try:
                return est.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_eval, y_eval)],
                    early_stopping_rounds=es_rounds,
                )
            except TypeError:
                return est.fit(X_train, y_train)
    try:
        return est.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
            early_stopping_rounds=es_rounds,
        )
    except TypeError:
        return est.fit(X_train, y_train)


def _fit_regressor_with_es(
    model_name: str,
    est: Any,
    X_train: Any,
    y_train: np.ndarray,
    cat_cols: List[str],
    X_eval: Optional[Any],
    y_eval: Optional[np.ndarray],
    es_rounds: int,
) -> Any:
    if X_eval is None or y_eval is None:
        if model_name == "catboost":
            return est.fit(X_train, y_train, cat_features=cat_cols)
        return est.fit(X_train, y_train)

    if model_name == "catboost":
        return est.fit(
            X_train,
            y_train,
            cat_features=cat_cols,
            eval_set=(X_eval, y_eval),
            use_best_model=True,
            early_stopping_rounds=es_rounds,
            verbose=False,
        )
    if model_name == "lgbm":
        try:
            import lightgbm as lgb

            return est.fit(
                X_train,
                y_train,
                eval_X=X_eval,
                eval_y=y_eval,
                callbacks=[
                    lgb.early_stopping(es_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        except TypeError:
            try:
                return est.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_eval, y_eval)],
                    early_stopping_rounds=es_rounds,
                )
            except TypeError:
                return est.fit(X_train, y_train)
    try:
        return est.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
            early_stopping_rounds=es_rounds,
        )
    except TypeError:
        return est.fit(X_train, y_train)


def fit_nuisance(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    config: DMLConfig,
    treat_levels: Optional[List[Any]] = None,
    df_eval: Optional[pd.DataFrame] = None,
    es_rounds: int = 200,
    refit_df: Optional[pd.DataFrame] = None,
) -> FittedNuisance:
    df0 = df_train.copy()
    df0 = df0.dropna(subset=[treat_col, outcome_col]).copy()
    df0[treat_col] = df0[treat_col].astype("category")

    if treat_levels is None:
        treat_levels = df0[treat_col].cat.categories.tolist()
    else:
        df0[treat_col] = df0[treat_col].cat.set_categories(treat_levels)

    z = df0[treat_col].cat.codes.to_numpy()
    y = df0[outcome_col].to_numpy(dtype=float)
    K = len(treat_levels)

    X_df = df0[feature_cols].copy()
    cat_cols = (
        config.categorical_cols
        if config.categorical_cols is not None
        else _infer_categorical_cols(df0, feature_cols)
    )
    cat_cols = [c for c in cat_cols if c in X_df.columns]

    uses_catboost = (config.prop_model == "catboost") or (config.outcome_model == "catboost")
    if uses_catboost:
        for c in cat_cols:
            if c in X_df.columns:
                X_df[c] = X_df[c].astype("category")

    uses_ohe = (config.prop_model in {"xgb", "lgbm"}) or (config.outcome_model in {"xgb", "lgbm"})
    pre = _build_preprocessor(feature_cols, cat_cols) if uses_ohe else None
    X_eval_df = None
    z_eval = None
    y_eval = None
    if df_eval is not None and not df_eval.empty:
        dfe = df_eval.copy()
        dfe = dfe.dropna(subset=[treat_col, outcome_col]).copy()
        dfe[treat_col] = dfe[treat_col].astype("category").cat.set_categories(treat_levels)
        z_eval_arr = dfe[treat_col].cat.codes.to_numpy()
        keep = z_eval_arr >= 0
        if np.any(keep):
            dfe = dfe.loc[keep].copy()
            z_eval = z_eval_arr[keep].astype(int)
            y_eval = dfe[outcome_col].to_numpy(dtype=float)
            X_eval_df = dfe[feature_cols].copy()
            for c in cat_cols:
                if c in X_eval_df.columns:
                    X_eval_df[c] = X_eval_df[c].astype("category")

    prop_est = make_classifier(
        config.prop_model,
        random_state=config.random_state,
        params=dict(config.prop_params),
        n_classes=K,
    )
    if config.prop_model == "xgb":
        if K == 2:
            prop_est.set_params(objective="binary:logistic")
        else:
            prop_est.set_params(objective="multi:softprob", num_class=K)
    elif config.prop_model == "lgbm":
        if K == 2:
            prop_est.set_params(objective="binary")
        else:
            prop_est.set_params(objective="multiclass", num_class=K)
    elif config.prop_model == "catboost":
        if K == 2:
            prop_est.set_params(loss_function="Logloss")
        else:
            prop_est.set_params(loss_function="MultiClass", classes_count=K)

    # Propensity
    counts = np.bincount(z, minlength=K)
    prop_fallback: Optional[np.ndarray] = None
    prop_model: Optional[Any] = None
    if (counts == 0).any():
        p = (counts + 1.0) / (counts.sum() + K)
        prop_fallback = p.astype(np.float32)
    else:
        if config.prop_model == "catboost":
            prop_model = _fit_classifier_with_es(
                config.prop_model,
                prop_est,
                X_df,
                z,
                cat_cols=cat_cols,
                X_eval=X_eval_df,
                y_eval=z_eval,
                es_rounds=es_rounds,
            )
        elif config.prop_model in {"xgb", "lgbm"}:
            if pre is None:
                raise RuntimeError("Encoded features are missing for XGB/LGBM.")
            X_enc = pre.fit_transform(X_df)
            X_eval_enc = pre.transform(X_eval_df) if X_eval_df is not None else None
            prop_model = _fit_classifier_with_es(
                config.prop_model,
                prop_est,
                X_enc,
                z,
                cat_cols=[],
                X_eval=X_eval_enc,
                y_eval=z_eval,
                es_rounds=es_rounds,
            )
        else:
            raise ValueError(f"Unknown prop_model: {config.prop_model}")

        if refit_df is not None:
            best_n = _best_num_trees(config.prop_model, prop_model)
            if best_n is not None and best_n > 0:
                ref_params = _set_num_trees(dict(config.prop_params), config.prop_model, best_n)
                ref_est = make_classifier(
                    config.prop_model,
                    random_state=config.random_state,
                    params=ref_params,
                    n_classes=K,
                )
                if config.prop_model == "xgb":
                    if K == 2:
                        ref_est.set_params(objective="binary:logistic")
                    else:
                        ref_est.set_params(objective="multi:softprob", num_class=K)
                elif config.prop_model == "lgbm":
                    if K == 2:
                        ref_est.set_params(objective="binary")
                    else:
                        ref_est.set_params(objective="multiclass", num_class=K)
                elif config.prop_model == "catboost":
                    if K == 2:
                        ref_est.set_params(loss_function="Logloss")
                    else:
                        ref_est.set_params(loss_function="MultiClass", classes_count=K)
                dfr = refit_df.dropna(subset=[treat_col, outcome_col]).copy()
                dfr[treat_col] = dfr[treat_col].astype("category").cat.set_categories(treat_levels)
                zr = dfr[treat_col].cat.codes.to_numpy()
                keep = zr >= 0
                dfr = dfr.loc[keep].copy()
                zr = zr[keep].astype(int)
                Xr = dfr[feature_cols].copy()
                for c in cat_cols:
                    if c in Xr.columns:
                        Xr[c] = Xr[c].astype("category")
                if config.prop_model == "catboost":
                    prop_model = ref_est.fit(Xr, zr, cat_features=cat_cols)
                else:
                    if pre is None:
                        raise RuntimeError("Encoded features are missing for XGB/LGBM.")
                    pre.fit(Xr)
                    Xr_enc = pre.transform(Xr)
                    prop_model = ref_est.fit(Xr_enc, zr)

    # Ensure the shared preprocessor is fitted even when propensity falls back
    # (e.g., missing treatment classes), because outcome models may still need OHE.
    if pre is not None and not hasattr(pre, "transformers_"):
        pre.fit(X_df)

    # Outcome models
    outcome_models: Dict[int, Optional[Any]] = {}
    outcome_fallback: Dict[int, float] = {}
    for k in range(K):
        mask = (z == k)
        nk = int(mask.sum())
        if nk < config.min_samples_per_treat:
            fallback = float(np.mean(y[mask])) if nk > 0 else float(np.mean(y))
            outcome_models[k] = None
            outcome_fallback[k] = fallback
            continue

        out_fold = make_regressor(
            config.outcome_model,
            random_state=config.random_state,
            params=dict(config.outcome_params),
        )
        X_eval_k = None
        y_eval_k = None
        if X_eval_df is not None and z_eval is not None and y_eval is not None:
            X_eval_k = X_eval_df
            y_eval_k = y_eval
        if config.outcome_model == "catboost":
            Xtr_y = X_df.loc[mask].copy()
            out_fold = _fit_regressor_with_es(
                config.outcome_model,
                out_fold,
                Xtr_y,
                y[mask],
                cat_cols=cat_cols,
                X_eval=X_eval_k,
                y_eval=y_eval_k,
                es_rounds=es_rounds,
            )
        elif config.outcome_model in {"xgb", "lgbm"}:
            if pre is None:
                raise RuntimeError("Encoded features are missing for XGB/LGBM.")
            Xtr_enc = pre.transform(X_df.loc[mask])
            X_eval_enc = pre.transform(X_eval_k) if X_eval_k is not None else None
            out_fold = _fit_regressor_with_es(
                config.outcome_model,
                out_fold,
                Xtr_enc,
                y[mask],
                cat_cols=[],
                X_eval=X_eval_enc,
                y_eval=y_eval_k,
                es_rounds=es_rounds,
            )
        else:
            raise ValueError(f"Unknown outcome_model: {config.outcome_model}")

        if refit_df is not None:
            best_n = _best_num_trees(config.outcome_model, out_fold)
            if best_n is not None and best_n > 0:
                ref_params = _set_num_trees(dict(config.outcome_params), config.outcome_model, best_n)
                ref_est = make_regressor(
                    config.outcome_model,
                    random_state=config.random_state,
                    params=ref_params,
                )
                dfr = refit_df.dropna(subset=[treat_col, outcome_col]).copy()
                dfr[treat_col] = dfr[treat_col].astype("category").cat.set_categories(treat_levels)
                zr = dfr[treat_col].cat.codes.to_numpy()
                keep = zr >= 0
                dfr = dfr.loc[keep].copy()
                zr = zr[keep].astype(int)
                yr = dfr[outcome_col].to_numpy(dtype=float)
                mask_r = (zr == k)
                if int(mask_r.sum()) >= config.min_samples_per_treat:
                    Xr = dfr[feature_cols].copy()
                    for c in cat_cols:
                        if c in Xr.columns:
                            Xr[c] = Xr[c].astype("category")
                    if config.outcome_model == "catboost":
                        out_fold = ref_est.fit(Xr.loc[mask_r], yr[mask_r], cat_features=cat_cols)
                    else:
                        if pre is None:
                            raise RuntimeError("Encoded features are missing for XGB/LGBM.")
                        pre.fit(Xr)
                        Xr_enc = pre.transform(Xr.loc[mask_r])
                        out_fold = ref_est.fit(Xr_enc, yr[mask_r])
        outcome_models[k] = out_fold
        outcome_fallback[k] = float(np.mean(y[mask]))

    return FittedNuisance(
        treat_levels=list(treat_levels),
        cat_cols=cat_cols,
        pre=pre,
        prop_model_name=config.prop_model,
        outcome_model_name=config.outcome_model,
        prop_model=prop_model,
        prop_fallback=prop_fallback,
        outcome_models=outcome_models,
        outcome_fallback=outcome_fallback,
    )


def predict_nuisance(
    fitted: FittedNuisance,
    df_new: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X_df = df_new[feature_cols].copy()
    if fitted.cat_cols:
        for c in fitted.cat_cols:
            if c in X_df.columns:
                X_df[c] = X_df[c].astype("category")

    K = len(fitted.treat_levels)
    n = len(df_new)
    m_hat = np.full((n, K), np.nan, dtype=np.float32)
    e_hat = np.full((n, K), np.nan, dtype=np.float32)

    # propensity
    if fitted.prop_fallback is not None:
        e_hat[:] = fitted.prop_fallback
    else:
        if fitted.prop_model is None:
            raise RuntimeError("prop_model missing with no fallback.")
        if fitted.prop_model_name in {"xgb", "lgbm"}:
            if fitted.pre is None:
                raise RuntimeError("Encoded features are missing for XGB/LGBM.")
            X_enc = fitted.pre.transform(X_df)
            prob = fitted.prop_model.predict_proba(X_enc)
        else:
            prob = fitted.prop_model.predict_proba(X_df)
        prob = np.asarray(prob, dtype=np.float32)
        prob = np.clip(prob, 1e-6, 1.0)
        prob = prob / prob.sum(axis=1, keepdims=True)
        e_hat[:] = prob

    # outcome
    for k in range(K):
        mdl = fitted.outcome_models.get(k)
        if mdl is None:
            m_hat[:, k] = np.float32(fitted.outcome_fallback.get(k, 0.0))
            continue
        if fitted.outcome_model_name in {"xgb", "lgbm"}:
            if fitted.pre is None:
                raise RuntimeError("Encoded features are missing for XGB/LGBM.")
            X_enc = fitted.pre.transform(X_df)
            pred = mdl.predict(X_enc)
        else:
            pred = mdl.predict(X_df)
        m_hat[:, k] = np.asarray(pred, dtype=np.float32)

    return m_hat, e_hat


def _align_treat_codes(df: pd.DataFrame, treat_col: str, treat_levels: List[Any]) -> np.ndarray:
    z = df[treat_col].astype("category").cat.set_categories(treat_levels)
    return z.cat.codes.to_numpy()


def score_nuisance(
    df_val: pd.DataFrame,
    m_hat: np.ndarray,
    e_hat: np.ndarray,
    treat_col: str,
    outcome_col: str,
    treat_levels: List[Any],
) -> Tuple[float, float]:
    z = _align_treat_codes(df_val, treat_col, treat_levels)
    y = df_val[outcome_col].to_numpy(dtype=float)
    keep = z >= 0
    if not np.any(keep):
        return float("nan"), float("nan")
    z = z[keep]
    y = y[keep]
    row_idx = np.where(keep)[0]
    m_hat = m_hat[row_idx, :]
    e_hat = e_hat[row_idx, :]
    e_hat = np.clip(e_hat, 1e-6, 1.0)
    e_hat = e_hat / e_hat.sum(axis=1, keepdims=True)

    # propensity log-loss
    prop_loss = float(log_loss(z, e_hat, labels=list(range(len(treat_levels)))))

    # outcome MSE for observed treatment
    y_pred = m_hat[np.arange(len(y)), z]
    out_mse = float(mean_squared_error(y, y_pred))
    return prop_loss, out_mse


def tune_nuisance_walk_forward(
    df: pd.DataFrame,
    splits: List[SplitRange],
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    prop_model: str,
    outcome_model: str,
    random_state: int,
    min_samples_per_treat: int,
    prop_grid: Dict[str, List[Any]],
    outcome_grid: Dict[str, List[Any]],
    enable_early_stopping: bool,
    es_rounds: int,
    group_col: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    treat_levels = df[treat_col].astype("category").cat.categories.tolist()

    prop_params_list = _sample_param_candidates(
        prop_model,
        _expand_grid(prop_grid),
        random_state=random_state + 101,
    )
    outcome_params_list = _sample_param_candidates(
        outcome_model,
        _expand_grid(outcome_grid),
        random_state=random_state + 211,
    )

    prop_rows = []
    for params in prop_params_list:
        losses = []
        for split in splits:
            df_tr = df[(df["season"] >= split.train_start) & (df["season"] <= split.train_end)].copy()
            df_va = df[(df["season"] >= split.test_start) & (df["season"] <= split.test_end)].copy()
            if df_tr.empty or df_va.empty:
                continue
            cfg = DMLConfig(
                n_splits=2,
                prop_model=prop_model,
                outcome_model=outcome_model,
                random_state=random_state,
                prop_params=dict(params),
                outcome_params={},
                min_samples_per_treat=min_samples_per_treat,
                group_col=group_col,
            )
            fitted = fit_nuisance(
                df_tr,
                feature_cols,
                treat_col,
                outcome_col,
                cfg,
                treat_levels=treat_levels,
                df_eval=df_va if enable_early_stopping else None,
                es_rounds=es_rounds,
            )
            m_hat, e_hat = predict_nuisance(fitted, df_va, feature_cols)
            prop_loss, _ = score_nuisance(df_va, m_hat, e_hat, treat_col, outcome_col, treat_levels)
            if np.isfinite(prop_loss):
                losses.append(prop_loss)
        if losses:
            prop_rows.append(dict(params=params, mean_log_loss=float(np.mean(losses))))

    outcome_rows = []
    for params in outcome_params_list:
        mse_losses = []
        rmse_losses = []
        mae_losses = []
        for split in splits:
            df_tr = df[(df["season"] >= split.train_start) & (df["season"] <= split.train_end)].copy()
            df_va = df[(df["season"] >= split.test_start) & (df["season"] <= split.test_end)].copy()
            if df_tr.empty or df_va.empty:
                continue
            cfg = DMLConfig(
                n_splits=2,
                prop_model=prop_model,
                outcome_model=outcome_model,
                random_state=random_state,
                prop_params={},
                outcome_params=dict(params),
                min_samples_per_treat=min_samples_per_treat,
                group_col=group_col,
            )
            fitted = fit_nuisance(
                df_tr,
                feature_cols,
                treat_col,
                outcome_col,
                cfg,
                treat_levels=treat_levels,
                df_eval=df_va if enable_early_stopping else None,
                es_rounds=es_rounds,
            )
            m_hat, e_hat = predict_nuisance(fitted, df_va, feature_cols)
            _, out_mse = score_nuisance(df_va, m_hat, e_hat, treat_col, outcome_col, treat_levels)
            if not np.isfinite(out_mse):
                continue
            z_va = _align_treat_codes(df_va, treat_col, treat_levels)
            keep = z_va >= 0
            if not np.any(keep):
                continue
            row_idx = np.where(keep)[0]
            y_true = df_va[outcome_col].to_numpy(dtype=np.float32)[keep]
            y_pred = np.asarray(m_hat[row_idx, z_va[keep]], dtype=np.float32)
            mse_losses.append(out_mse)
            rmse_losses.append(float(np.sqrt(out_mse)))
            mae_losses.append(float(mean_absolute_error(y_true, y_pred)))
        if mse_losses:
            outcome_rows.append(
                dict(
                    params=params,
                    mean_mse=float(np.mean(mse_losses)),
                    mean_rmse=float(np.mean(rmse_losses)),
                    mean_mae=float(np.mean(mae_losses)),
                )
            )

    df_prop = pd.DataFrame(prop_rows).sort_values("mean_log_loss", ascending=True)
    df_out = pd.DataFrame(outcome_rows).sort_values(["mean_rmse", "mean_mae"], ascending=[True, True])

    best_prop = dict(df_prop.iloc[0]["params"]) if not df_prop.empty else {}
    best_out = dict(df_out.iloc[0]["params"]) if not df_out.empty else {}

    df_prop["task"] = "propensity"
    df_out["task"] = "outcome"
    df_all = pd.concat([df_prop, df_out], ignore_index=True)
    return best_prop, best_out, df_all


def compute_dr_r_losses(
    df_eval: pd.DataFrame,
    treat_col: str,
    outcome_col: str,
    treat_a: str,
    treat_b: str,
    tau_hat: np.ndarray,
    min_prop: float,
) -> Dict[str, float]:
    y = df_eval[outcome_col].to_numpy(dtype=float)
    z = df_eval[treat_col].astype(str).to_numpy()
    w = (z == str(treat_a)).astype(float)

    m1 = df_eval[f"m_hat_{treat_a}"].to_numpy(dtype=float)
    m0 = df_eval[f"m_hat_{treat_b}"].to_numpy(dtype=float)
    e = np.clip(df_eval[f"e_hat_{treat_a}"].to_numpy(dtype=float), min_prop, 1.0 - 1e-6)

    m_obs = w * m1 + (1.0 - w) * m0
    pseudo_tau = (m1 - m0) + w * (y - m1) / e - (1.0 - w) * (y - m0) / (1.0 - e)

    dr_loss = float(np.mean((pseudo_tau - tau_hat) ** 2))
    r_loss = float(np.mean(((y - m_obs) - (w - e) * tau_hat) ** 2))
    return dict(dr_loss=dr_loss, r_loss=r_loss)


def compute_blp_metrics(tau_hat: np.ndarray, psi_tau: np.ndarray) -> Dict[str, float]:
    x = np.asarray(tau_hat, dtype=float)
    y = np.asarray(psi_tau, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return dict(blp_alpha=np.nan, blp_beta=np.nan, blp_r2=np.nan)

    X = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    y_hat = alpha + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return dict(blp_alpha=alpha, blp_beta=beta, blp_r2=r2)


def _compute_blp_metrics_robust(
    tau_hat: np.ndarray,
    psi_tau: np.ndarray,
    cluster: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    x = np.asarray(tau_hat, dtype=float)
    y = np.asarray(psi_tau, dtype=float)
    if cluster is not None:
        c = np.asarray(cluster)
        mask = np.isfinite(x) & np.isfinite(y) & pd.notna(c)
        c = c[mask]
    else:
        c = None
        mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return dict(
            blp_alpha=np.nan,
            blp_beta=np.nan,
            blp_r2=np.nan,
            blp_alpha_se=np.nan,
            blp_beta_se=np.nan,
            blp_alpha_ci_lo=np.nan,
            blp_alpha_ci_hi=np.nan,
            blp_beta_ci_lo=np.nan,
            blp_beta_ci_hi=np.nan,
            blp_cov_type="none",
            blp_n=float(len(x)),
            blp_n_cluster=np.nan,
        )

    X = np.column_stack([np.ones(len(x)), x])
    k = X.shape[1]
    n = X.shape[0]
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    resid = y - (X @ beta)

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    cov_type = "HC3"
    n_cluster = np.nan
    if c is None:
        h = np.clip(np.sum(X * (X @ xtx_inv), axis=1), 0.0, 1.0 - 1e-12)
        adj_u = resid / (1.0 - h)
        meat = X.T @ ((adj_u ** 2)[:, None] * X)
        cov = xtx_inv @ meat @ xtx_inv
    else:
        uniq = pd.unique(c)
        g = len(uniq)
        n_cluster = float(g)
        cov_type = "cluster"
        if g <= 1:
            cov = np.full((k, k), np.nan, dtype=float)
        else:
            meat = np.zeros((k, k), dtype=float)
            for cl in uniq:
                idx = np.flatnonzero(c == cl)
                Xg = X[idx, :]
                ug = resid[idx]
                sg = Xg.T @ ug
                meat += np.outer(sg, sg)
            corr = (g / (g - 1.0)) * ((n - 1.0) / max(n - k, 1.0))
            cov = corr * (xtx_inv @ meat @ xtx_inv)

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = 1.959963984540054
    a = float(beta[0])
    b = float(beta[1])
    a_se = float(se[0]) if np.isfinite(se[0]) else np.nan
    b_se = float(se[1]) if np.isfinite(se[1]) else np.nan

    return dict(
        blp_alpha=a,
        blp_beta=b,
        blp_r2=r2,
        blp_alpha_se=a_se,
        blp_beta_se=b_se,
        blp_alpha_ci_lo=(a - z * a_se) if np.isfinite(a_se) else np.nan,
        blp_alpha_ci_hi=(a + z * a_se) if np.isfinite(a_se) else np.nan,
        blp_beta_ci_lo=(b - z * b_se) if np.isfinite(b_se) else np.nan,
        blp_beta_ci_hi=(b + z * b_se) if np.isfinite(b_se) else np.nan,
        blp_cov_type=cov_type,
        blp_n=float(n),
        blp_n_cluster=n_cluster,
    )


def _collect_nuisance_oos(
    df: pd.DataFrame,
    splits: List[SplitRange],
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    config: DMLConfig,
    enable_early_stopping: bool,
    es_rounds: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    treat_levels = df[treat_col].astype("category").cat.categories.tolist()
    n = len(df)
    K = len(treat_levels)
    m_hat = np.full((n, K), np.nan, dtype=np.float32)
    e_hat = np.full((n, K), np.nan, dtype=np.float32)

    for split in splits:
        df_tr = df[(df["season"] >= split.train_start) & (df["season"] <= split.train_end)].copy()
        df_va = df[(df["season"] >= split.test_start) & (df["season"] <= split.test_end)].copy()
        if df_tr.empty or df_va.empty:
            continue
        fitted = fit_nuisance(
            df_tr,
            feature_cols,
            treat_col,
            outcome_col,
            config,
            treat_levels=treat_levels,
            df_eval=df_va if enable_early_stopping else None,
            es_rounds=es_rounds,
        )
        m_pred, e_pred = predict_nuisance(fitted, df_va, feature_cols)

        idx = df_va.index.to_numpy()
        m_hat[idx, :] = m_pred
        e_hat[idx, :] = e_pred

    df_out = df.copy()
    for k, lvl in enumerate(treat_levels):
        df_out[f"m_hat_{lvl}"] = m_hat[:, k]
        df_out[f"e_hat_{lvl}"] = e_hat[:, k]

    meta = dict(treat_levels=[str(x) for x in treat_levels])
    return df_out, meta


def fit_nuisance_full(
    df_train: pd.DataFrame,
    df_apply: pd.DataFrame,
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    config: DMLConfig,
    enable_early_stopping: bool = True,
    es_rounds: int = 200,
    final_es_tail_span: int = 3,
    eval_df_for_es: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    treat_levels = df_train[treat_col].astype("category").cat.categories.tolist()
    if enable_early_stopping:
        if eval_df_for_es is not None and not eval_df_for_es.empty:
            fitted = fit_nuisance(
                df_train,
                feature_cols,
                treat_col,
                outcome_col,
                config,
                treat_levels=treat_levels,
                df_eval=eval_df_for_es,
                es_rounds=es_rounds,
            )
        else:
            df_es_train, df_es_val = _split_tail_eval(df_train, season_col="season", tail_span=final_es_tail_span)
            if df_es_train is not None and df_es_val is not None:
                fitted = fit_nuisance(
                    df_es_train,
                    feature_cols,
                    treat_col,
                    outcome_col,
                    config,
                    treat_levels=treat_levels,
                    df_eval=df_es_val,
                    es_rounds=es_rounds,
                    refit_df=df_train,
                )
            else:
                fitted = fit_nuisance(df_train, feature_cols, treat_col, outcome_col, config, treat_levels=treat_levels)
    else:
        fitted = fit_nuisance(df_train, feature_cols, treat_col, outcome_col, config, treat_levels=treat_levels)
    m_hat, e_hat = predict_nuisance(fitted, df_apply, feature_cols)
    out = df_apply.copy()
    for k, lvl in enumerate(treat_levels):
        out[f"m_hat_{lvl}"] = m_hat[:, k]
        out[f"e_hat_{lvl}"] = e_hat[:, k]
    return out


def _tau_param_grid(model_name: str, override: Optional[Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
    grid = override or _default_tau_grid(model_name)
    return _expand_grid(grid)


@dataclass
class TauModelBundle:
    model_name: str
    model: Any
    pre: Optional[ColumnTransformer]
    cat_cols: List[str]


def _fit_tau_model(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    y_train: np.ndarray,
    model_name: str,
    cat_cols: List[str],
    tuned_params: Dict[str, Any],
    random_state: int,
    df_eval: Optional[pd.DataFrame] = None,
    y_eval: Optional[np.ndarray] = None,
    es_rounds: int = 200,
    refit_df: Optional[pd.DataFrame] = None,
    refit_y: Optional[np.ndarray] = None,
) -> TauModelBundle:
    Xtr_df = df_train[feature_cols].copy()
    for c in cat_cols:
        if c in Xtr_df.columns:
            Xtr_df[c] = Xtr_df[c].astype("category")

    Xev_df = None
    if df_eval is not None and y_eval is not None:
        Xev_df = df_eval[feature_cols].copy()
        for c in cat_cols:
            if c in Xev_df.columns:
                Xev_df[c] = Xev_df[c].astype("category")

    if model_name == "catboost":
        est = make_regressor(model_name, random_state=random_state, params=dict(tuned_params))
        est = _fit_regressor_with_es(
            model_name,
            est,
            Xtr_df,
            y_train,
            cat_cols=cat_cols,
            X_eval=Xev_df,
            y_eval=y_eval,
            es_rounds=es_rounds,
        )
        if refit_df is not None and refit_y is not None:
            best_n = _best_num_trees(model_name, est)
            if best_n is not None and best_n > 0:
                ref_params = _set_num_trees(dict(tuned_params), model_name, best_n)
                Xref_df = refit_df[feature_cols].copy()
                for c in cat_cols:
                    if c in Xref_df.columns:
                        Xref_df[c] = Xref_df[c].astype("category")
                est = make_regressor(model_name, random_state=random_state, params=ref_params)
                est = est.fit(Xref_df, refit_y, cat_features=cat_cols)
        return TauModelBundle(model_name=model_name, model=est, pre=None, cat_cols=cat_cols)

    pre = _build_preprocessor(feature_cols, cat_cols)
    Xtr_enc = pre.fit_transform(Xtr_df)
    Xev_enc = pre.transform(Xev_df) if Xev_df is not None else None

    est = make_regressor(model_name, random_state=random_state, params=dict(tuned_params))
    est = _fit_regressor_with_es(
        model_name,
        est,
        Xtr_enc,
        y_train,
        cat_cols=[],
        X_eval=Xev_enc,
        y_eval=y_eval,
        es_rounds=es_rounds,
    )
    if refit_df is not None and refit_y is not None:
        best_n = _best_num_trees(model_name, est)
        if best_n is not None and best_n > 0:
            ref_params = _set_num_trees(dict(tuned_params), model_name, best_n)
            Xref_df = refit_df[feature_cols].copy()
            for c in cat_cols:
                if c in Xref_df.columns:
                    Xref_df[c] = Xref_df[c].astype("category")
            pre.fit(Xref_df)
            Xref_enc = pre.transform(Xref_df)
            est = make_regressor(model_name, random_state=random_state, params=ref_params)
            est = est.fit(Xref_enc, refit_y)
    return TauModelBundle(model_name=model_name, model=est, pre=pre, cat_cols=cat_cols)


def _predict_tau_model(bundle: TauModelBundle, df_new: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df_new[feature_cols].copy()
    for c in bundle.cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    if bundle.pre is None:
        pred = bundle.model.predict(X)
    else:
        Xenc = bundle.pre.transform(X)
        pred = bundle.model.predict(Xenc)
    return np.asarray(pred, dtype=np.float32)


def tune_tau_walk_forward(
    splits: List[SplitRange],
    df: pd.DataFrame,
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    model_name: str,
    random_state: int,
    nuisance_cfg: DMLConfig,
    param_grid: Dict[str, List[Any]],
    min_prop: float,
    max_prop: float,
    enable_early_stopping: bool,
    es_rounds: int,
    final_es_tail_span: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    treat_levels = df[treat_col].astype("category").cat.categories.tolist()
    params_list = _sample_param_candidates(
        model_name,
        _tau_param_grid(model_name, param_grid),
        random_state=random_state + 503,
    )
    rows = []

    for params in params_list:
        losses = []
        for split in splits:
            df_tr = df[(df["season"] >= split.train_start) & (df["season"] <= split.train_end)].copy()
            df_va = df[(df["season"] >= split.test_start) & (df["season"] <= split.test_end)].copy()
            if df_tr.empty or df_va.empty:
                continue

            # Fit nuisance on train and get predictions for train/val
            df_tr_n = fit_nuisance_full(
                df_tr,
                df_tr,
                feature_cols,
                treat_col,
                outcome_col,
                nuisance_cfg,
                enable_early_stopping=enable_early_stopping,
                es_rounds=es_rounds,
                final_es_tail_span=final_es_tail_span,
                eval_df_for_es=df_va if enable_early_stopping else None,
            )
            df_va_n = fit_nuisance_full(
                df_tr,
                df_va,
                feature_cols,
                treat_col,
                outcome_col,
                nuisance_cfg,
                enable_early_stopping=enable_early_stopping,
                es_rounds=es_rounds,
                final_es_tail_span=final_es_tail_span,
                eval_df_for_es=df_va if enable_early_stopping else None,
            )

            psi_tr = compute_pseudo_outcomes(
                df_tr_n,
                treat_col=treat_col,
                outcome_col=outcome_col,
                treat_levels=treat_levels,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            psi_va = compute_pseudo_outcomes(
                df_va_n,
                treat_col=treat_col,
                outcome_col=outcome_col,
                treat_levels=treat_levels,
                min_prop=min_prop,
                max_prop=max_prop,
            )

            cat_cols = infer_categorical_cols(df_tr_n, feature_cols)
            mse_levels = []
            for lvl in treat_levels:
                mdl = _fit_tau_model(
                    df_train=df_tr_n,
                    feature_cols=feature_cols,
                    y_train=psi_tr[lvl],
                    model_name=model_name,
                    cat_cols=cat_cols,
                    tuned_params=params,
                    random_state=random_state,
                    df_eval=df_va_n if enable_early_stopping else None,
                    y_eval=psi_va[lvl] if enable_early_stopping else None,
                    es_rounds=es_rounds,
                )
                pred = _predict_tau_model(mdl, df_va_n, feature_cols)
                mse_levels.append(mean_squared_error(psi_va[lvl], pred))
            losses.append(float(np.mean(mse_levels)))

        if losses:
            rows.append(dict(params=params, mean_mse=float(np.mean(losses))))

    df_out = pd.DataFrame(rows).sort_values("mean_mse", ascending=True)
    best = dict(df_out.iloc[0]["params"]) if not df_out.empty else {}
    return best, df_out


def make_tau_oos(
    df: pd.DataFrame,
    splits: List[SplitRange],
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    nuisance_cfg: DMLConfig,
    tau_model: str,
    tau_params: Dict[str, Any],
    min_prop: float,
    max_prop: float,
    treat_a: str,
    treat_b: str,
    enable_early_stopping: bool,
    es_rounds: int,
    final_es_tail_span: int,
) -> pd.DataFrame:
    treat_levels = df[treat_col].astype("category").cat.categories.tolist()
    n = len(df)
    tau_hat = np.full(n, np.nan, dtype=np.float32)

    for split in splits:
        df_tr = df[(df["season"] >= split.train_start) & (df["season"] <= split.train_end)].copy()
        df_va = df[(df["season"] >= split.test_start) & (df["season"] <= split.test_end)].copy()
        if df_tr.empty or df_va.empty:
            continue

        df_tr_n = fit_nuisance_full(
            df_tr,
            df_tr,
            feature_cols,
            treat_col,
            outcome_col,
            nuisance_cfg,
            enable_early_stopping=enable_early_stopping,
            es_rounds=es_rounds,
            final_es_tail_span=final_es_tail_span,
            eval_df_for_es=df_va if enable_early_stopping else None,
        )
        df_va_n = fit_nuisance_full(
            df_tr,
            df_va,
            feature_cols,
            treat_col,
            outcome_col,
            nuisance_cfg,
            enable_early_stopping=enable_early_stopping,
            es_rounds=es_rounds,
            final_es_tail_span=final_es_tail_span,
            eval_df_for_es=df_va if enable_early_stopping else None,
        )

        psi_tr = compute_pseudo_outcomes(
            df_tr_n,
            treat_col=treat_col,
            outcome_col=outcome_col,
            treat_levels=treat_levels,
            min_prop=min_prop,
            max_prop=max_prop,
        )
        psi_va = compute_pseudo_outcomes(
            df_va_n,
            treat_col=treat_col,
            outcome_col=outcome_col,
            treat_levels=treat_levels,
            min_prop=min_prop,
            max_prop=max_prop,
        ) if enable_early_stopping else {}

        cat_cols = infer_categorical_cols(df_tr_n, feature_cols)
        models: Dict[str, TauModelBundle] = {}
        for lvl in treat_levels:
            models[str(lvl)] = _fit_tau_model(
                df_train=df_tr_n,
                feature_cols=feature_cols,
                y_train=psi_tr[lvl],
                model_name=tau_model,
                cat_cols=cat_cols,
                tuned_params=tau_params,
                random_state=nuisance_cfg.random_state,
                df_eval=df_va_n if enable_early_stopping else None,
                y_eval=psi_va[lvl] if enable_early_stopping else None,
                es_rounds=es_rounds,
            )

        mu_a = _predict_tau_model(models[str(treat_a)], df_va_n, feature_cols)
        mu_b = _predict_tau_model(models[str(treat_b)], df_va_n, feature_cols)
        tau = np.asarray(mu_a, dtype=np.float32) - np.asarray(mu_b, dtype=np.float32)
        tau_hat[df_va.index.to_numpy()] = tau

    out = df.copy()
    out["tau_hat"] = tau_hat
    return out


def fit_tau_full(
    df_train: pd.DataFrame,
    df_apply: pd.DataFrame,
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    tau_model: str,
    tau_params: Dict[str, Any],
    min_prop: float,
    max_prop: float,
    treat_a: str,
    treat_b: str,
    random_state: int = 123,
    enable_early_stopping: bool = True,
    es_rounds: int = 200,
    final_es_tail_span: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    treat_levels = df_train[treat_col].astype("category").cat.categories.tolist()
    cat_cols = infer_categorical_cols(df_train, feature_cols)

    psi_tr = compute_pseudo_outcomes(
        df_train,
        treat_col=treat_col,
        outcome_col=outcome_col,
        treat_levels=treat_levels,
        min_prop=min_prop,
        max_prop=max_prop,
    )

    models: Dict[str, TauModelBundle] = {}
    df_es_train = df_train
    df_es_val = None
    psi_es_train = psi_tr
    psi_es_val: Dict[Any, np.ndarray] = {}
    refit_df = None
    if enable_early_stopping:
        es_pair = _split_tail_eval(df_train, season_col="season", tail_span=final_es_tail_span)
        if es_pair[0] is not None and es_pair[1] is not None:
            df_es_train, df_es_val = es_pair
            psi_es_train = compute_pseudo_outcomes(
                df_es_train,
                treat_col=treat_col,
                outcome_col=outcome_col,
                treat_levels=treat_levels,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            psi_es_val = compute_pseudo_outcomes(
                df_es_val,
                treat_col=treat_col,
                outcome_col=outcome_col,
                treat_levels=treat_levels,
                min_prop=min_prop,
                max_prop=max_prop,
            )
            refit_df = df_train

    for lvl in treat_levels:
        models[str(lvl)] = _fit_tau_model(
            df_train=df_es_train,
            feature_cols=feature_cols,
            y_train=psi_es_train[lvl],
            model_name=tau_model,
            cat_cols=cat_cols,
            tuned_params=tau_params,
            random_state=random_state,
            df_eval=df_es_val if enable_early_stopping else None,
            y_eval=psi_es_val.get(lvl) if enable_early_stopping else None,
            es_rounds=es_rounds,
            refit_df=refit_df,
            refit_y=psi_tr[lvl] if refit_df is not None else None,
        )

    mu_a = _predict_tau_model(models[str(treat_a)], df_apply, feature_cols)
    mu_b = _predict_tau_model(models[str(treat_b)], df_apply, feature_cols)
    tau = np.asarray(mu_a, dtype=np.float32) - np.asarray(mu_b, dtype=np.float32)
    out = df_apply.copy()
    out["tau_hat"] = tau

    payload = dict(
        model_name=tau_model,
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
        treat_levels=[str(x) for x in treat_levels],
        treat_a=str(treat_a),
        treat_b=str(treat_b),
        baseline=str(treat_b),
        models={k: v.model for k, v in models.items()},
        preprocessors={k: v.pre for k, v in models.items()},
    )
    return out, payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nested walk-forward orchestration")
    p.add_argument("--input", required=True, help="panel parquet with season/treat/outcome/features")
    p.add_argument("--outdir", default="results/nested_wf")

    p.add_argument("--train-start", type=int, default=2000)
    p.add_argument("--train-end-init", type=int, default=2009)
    p.add_argument("--test-span", type=int, default=3)
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--max-season", type=int, default=None)

    p.add_argument("--inner-train-init-span", type=int, default=4)
    p.add_argument("--inner-block-span", type=int, default=3)
    p.add_argument("--inner-step", type=int, default=3)

    p.add_argument("--treat-col", default="shot_zone_choice")
    p.add_argument("--outcome-col", default="delta_wp")
    p.add_argument("--season-col", default="season")
    p.add_argument("--treat-a", default="three-point")
    p.add_argument("--treat-b", default="two-point")
    add_treatment_scheme_arg(p, default="binary")

    p.add_argument("--prop-model", default="catboost", choices=["xgb", "lgbm", "catboost"])
    p.add_argument("--outcome-model", default="catboost", choices=["xgb", "lgbm", "catboost"])
    p.add_argument("--tau-model", default="catboost", choices=["xgb", "lgbm", "catboost"])

    p.add_argument("--min-samples-per-treat", type=int, default=200)
    p.add_argument("--random-state", type=int, default=123)
    p.add_argument("--min-prop", type=float, default=1e-2)
    p.add_argument("--max-prop", type=float, default=1.0)
    p.add_argument("--disable-early-stopping", action="store_true")
    p.add_argument("--es-rounds", type=int, default=200)
    p.add_argument("--final-es-tail-span", type=int, default=3)
    p.add_argument(
        "--cluster-col",
        type=str,
        default="GAME_ID",
        help="Cluster id column used for cluster-robust BLP diagnostics.",
    )

    p.add_argument("--features", default=None, help="comma-separated feature columns (optional)")
    p.add_argument("--include-team-ids", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _log_progress(f"start run input={args.input} outdir={args.outdir}")
    df = pd.read_parquet(args.input)
    if args.season_col not in df.columns:
        raise KeyError(f"season column not found: {args.season_col}")

    # normalize season col
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
        raise ValueError("No rows left after treatment mapping. Check --treatment-scheme and input labels.")

    if args.features:
        feature_cols = _parse_csv_list(args.features, str)
    else:
        feature_cols = default_feature_cols(df, include_team_ids=args.include_team_ids)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    treat_levels = df[args.treat_col].astype("category").cat.categories.tolist()
    if args.treat_a not in treat_levels or args.treat_b not in treat_levels:
        raise ValueError(
            f"--treat-a/--treat-b must exist after treatment mapping. "
            f"Got treat_a={args.treat_a}, treat_b={args.treat_b}, levels={treat_levels}"
        )
    max_season = args.max_season or int(df["season"].max())
    outer_splits = list(
        iter_outer_splits(
            train_start=args.train_start,
            train_end_init=args.train_end_init,
            test_span=args.test_span,
            step=args.step,
            max_season=max_season,
        )
    )
    if not outer_splits:
        raise ValueError("No outer splits created. Check season ranges.")
    _log_progress(
        f"prepared data rows={len(df):,} features={len(feature_cols)} "
        f"outer_splits={len(outer_splits)} seasons={int(df['season'].min())}-{int(df['season'].max())}"
    )

    use_early_stopping = not args.disable_early_stopping

    for outer in outer_splits:
        tag = f"train{outer.train_start}_{outer.train_end}_test{outer.test_start}_{outer.test_end}"
        out_base = outdir / tag
        out_base.mkdir(parents=True, exist_ok=True)
        _log_progress(f"{tag}: start")

        df_train = df[(df["season"] >= outer.train_start) & (df["season"] <= outer.train_end)].copy()
        df_test = df[(df["season"] >= outer.test_start) & (df["season"] <= outer.test_end)].copy()

        inner_splits = list(
            iter_inner_splits(
                train_start=outer.train_start,
                train_end_max=outer.train_end,
                train_init_span=args.inner_train_init_span,
                block_span=args.inner_block_span,
                step=args.inner_step,
            )
        )
        if not inner_splits:
            raise ValueError(f"No inner splits created for outer={tag}.")
        _log_progress(f"{tag}: train_rows={len(df_train):,} test_rows={len(df_test):,} inner_splits={len(inner_splits)}")

        selected_prop_model = args.prop_model
        selected_outcome_model = args.outcome_model
        selected_tau_model = args.tau_model

        # -------------------------
        # 1) Tune nuisance
        # -------------------------
        _log_progress(f"{tag}: nuisance tuning start ({selected_prop_model}/{selected_outcome_model})")
        prop_grid = _default_prop_grid(selected_prop_model)
        out_grid = _default_outcome_grid(selected_outcome_model)
        best_prop, best_out, tune_df = tune_nuisance_walk_forward(
            df_train,
            inner_splits,
            feature_cols,
            args.treat_col,
            args.outcome_col,
            selected_prop_model,
            selected_outcome_model,
            args.random_state,
            args.min_samples_per_treat,
            prop_grid,
            out_grid,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            group_col=args.cluster_col,
        )
        tune_path = out_base / "nuisance_tuning.csv"
        tune_df.to_csv(tune_path, index=False)
        _log_progress(f"{tag}: nuisance tuning done -> {tune_path}")

        # 2) OOS nuisance within outer train
        _log_progress(f"{tag}: building nuisance OOS (train)")
        nuisance_cfg = DMLConfig(
            n_splits=2,
            prop_model=selected_prop_model,
            outcome_model=selected_outcome_model,
            random_state=args.random_state,
            prop_params=best_prop,
            outcome_params=best_out,
            min_samples_per_treat=args.min_samples_per_treat,
            group_col=args.cluster_col,
        )
        df_train_oos, _ = _collect_nuisance_oos(
            df_train,
            inner_splits,
            feature_cols,
            args.treat_col,
            args.outcome_col,
            nuisance_cfg,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
        )
        df_train_oos_path = out_base / "nuisance_oos_train.parquet"
        df_train_oos.to_parquet(df_train_oos_path, index=False)
        _log_progress(f"{tag}: nuisance OOS saved -> {df_train_oos_path}")

        # 3) Full nuisance for train/test (fit on full outer train)
        _log_progress(f"{tag}: fitting full nuisance (train/test)")
        df_train_full = fit_nuisance_full(
            df_train,
            df_train,
            feature_cols,
            args.treat_col,
            args.outcome_col,
            nuisance_cfg,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
        df_test_full = fit_nuisance_full(
            df_train,
            df_test,
            feature_cols,
            args.treat_col,
            args.outcome_col,
            nuisance_cfg,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
        df_train_full_path = out_base / "nuisance_full_train.parquet"
        df_test_full_path = out_base / "nuisance_test.parquet"
        df_train_full.to_parquet(df_train_full_path, index=False)
        df_test_full.to_parquet(df_test_full_path, index=False)
        _log_progress(f"{tag}: full nuisance saved -> {df_train_full_path}, {df_test_full_path}")

        # -------------------------
        # 4) Tune tau-model
        # -------------------------
        _log_progress(f"{tag}: tau tuning start ({selected_tau_model})")
        tau_grid = _default_tau_grid(selected_tau_model)
        best_tau, tau_tune_df = tune_tau_walk_forward(
            inner_splits,
            df_train,
            feature_cols,
            args.treat_col,
            args.outcome_col,
            selected_tau_model,
            args.random_state,
            nuisance_cfg,
            tau_grid,
            args.min_prop,
            args.max_prop,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
        tau_tune_path = out_base / "tau_tuning.csv"
        tau_tune_df.to_csv(tau_tune_path, index=False)
        _log_progress(f"{tag}: tau tuning done -> {tau_tune_path}")

        # 5) OOS tau_hat within outer train
        _log_progress(f"{tag}: building tau OOS (train)")
        df_train_tau_oos = make_tau_oos(
            df_train,
            inner_splits,
            feature_cols,
            args.treat_col,
            args.outcome_col,
            nuisance_cfg,
            selected_tau_model,
            best_tau,
            args.min_prop,
            args.max_prop,
            args.treat_a,
            args.treat_b,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
        df_train_tau_oos_path = out_base / "tau_oos_train.parquet"
        df_train_tau_oos.to_parquet(df_train_tau_oos_path, index=False)
        _log_progress(f"{tag}: tau OOS saved -> {df_train_tau_oos_path}")

        # 6) Full tau_hat for train/test (fit on full outer train)
        _log_progress(f"{tag}: fitting full tau (train/test)")
        df_train_tau_full, tau_payload = fit_tau_full(
            df_train_full,
            df_train_full,
            feature_cols,
            args.treat_col,
            args.outcome_col,
            selected_tau_model,
            best_tau,
            args.min_prop,
            args.max_prop,
            args.treat_a,
            args.treat_b,
            random_state=args.random_state,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
        df_test_tau_full, _ = fit_tau_full(
            df_train_full,
            df_test_full,
            feature_cols,
            args.treat_col,
            args.outcome_col,
            selected_tau_model,
            best_tau,
            args.min_prop,
            args.max_prop,
            args.treat_a,
            args.treat_b,
            random_state=args.random_state,
            enable_early_stopping=use_early_stopping,
            es_rounds=args.es_rounds,
            final_es_tail_span=args.final_es_tail_span,
        )
        df_train_tau_full_path = out_base / "tau_full_train.parquet"
        df_test_tau_full_path = out_base / "tau_test.parquet"
        df_train_tau_full.to_parquet(df_train_tau_full_path, index=False)
        df_test_tau_full.to_parquet(df_test_tau_full_path, index=False)
        _log_progress(f"{tag}: full tau saved -> {df_train_tau_full_path}, {df_test_tau_full_path}")

        tau_model_path = out_base / "tau_model.joblib"
        dump(tau_payload, tau_model_path)
        _log_progress(f"{tag}: tau model saved -> {tau_model_path}")

        # 7) Fill the initial train span in-sample so fold summaries start at train_start.
        init_span = inner_splits[0]
        df_init = df_train[
            (df_train["season"] >= init_span.train_start) & (df_train["season"] <= init_span.train_end)
        ].copy()
        if not df_init.empty:
            _log_progress(f"{tag}: filling initial span nuisance/tau")
            df_init_n = fit_nuisance_full(
                df_init,
                df_init,
                feature_cols,
                args.treat_col,
                args.outcome_col,
                nuisance_cfg,
                enable_early_stopping=use_early_stopping,
                es_rounds=args.es_rounds,
                final_es_tail_span=args.final_es_tail_span,
            )
            # fill nuisance for initial span if missing
            for c in df_init_n.columns:
                if c.startswith("m_hat_") or c.startswith("e_hat_"):
                    if c in df_train_oos.columns:
                        mask = df_train_oos.loc[df_init_n.index, c].isna()
                        df_train_oos.loc[df_init_n.index[mask], c] = df_init_n.loc[df_init_n.index[mask], c]

            # tau_hat for initial span (in-sample)
            df_init_tau, _ = fit_tau_full(
                df_init_n,
                df_init_n,
                feature_cols,
                args.treat_col,
                args.outcome_col,
                selected_tau_model,
                best_tau,
                args.min_prop,
                args.max_prop,
                args.treat_a,
                args.treat_b,
                random_state=args.random_state,
                enable_early_stopping=use_early_stopping,
                es_rounds=args.es_rounds,
                final_es_tail_span=args.final_es_tail_span,
            )
            init_tau_mask = df_train_tau_oos.loc[df_init_tau.index, "tau_hat"].isna()
            df_train_tau_oos.loc[df_init_tau.index[init_tau_mask], "tau_hat"] = df_init_tau.loc[
                df_init_tau.index[init_tau_mask], "tau_hat"
            ]

        # 8) BLP diagnostics on outer-train OOS nuisance/tau
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

        if not df_blp_oos.empty:
            y_blp = df_blp_oos[args.outcome_col].to_numpy(dtype=float)
            z_blp = df_blp_oos[args.treat_col].astype(str).to_numpy()
            w_blp = (z_blp == str(args.treat_a)).astype(float)
            m1_blp = df_blp_oos[f"m_hat_{args.treat_a}"].to_numpy(dtype=float)
            m0_blp = df_blp_oos[f"m_hat_{args.treat_b}"].to_numpy(dtype=float)
            e1_blp = np.clip(
                df_blp_oos[f"e_hat_{args.treat_a}"].to_numpy(dtype=float),
                float(args.min_prop),
                1.0 - 1e-6,
            )
            if f"e_hat_{args.treat_b}" in df_blp_oos.columns:
                e0_blp = np.clip(
                    df_blp_oos[f"e_hat_{args.treat_b}"].to_numpy(dtype=float),
                    float(args.min_prop),
                    1.0 - 1e-6,
                )
            else:
                e0_blp = np.clip(1.0 - e1_blp, float(args.min_prop), 1.0 - 1e-6)
            tau_hat_blp = df_blp_oos["tau_hat"].to_numpy(dtype=float)
            psi_tau_blp = (m1_blp - m0_blp) + w_blp * (y_blp - m1_blp) / e1_blp - (1.0 - w_blp) * (y_blp - m0_blp) / e0_blp
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
        else:
            blp_metrics = _compute_blp_metrics_robust(
                tau_hat=np.array([], dtype=float),
                psi_tau=np.array([], dtype=float),
                cluster=None,
            )

        blp_metrics_path = out_base / "blp_metrics_oos_train.json"
        blp_metrics_path.write_text(json.dumps(blp_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        _log_progress(f"{tag}: BLP metrics saved -> {blp_metrics_path}")

        meta = dict(
            outer_split=dict(
                train_start=outer.train_start,
                train_end=outer.train_end,
                test_start=outer.test_start,
                test_end=outer.test_end,
            ),
            inner_splits=[split.__dict__ for split in inner_splits],
            feature_cols=feature_cols,
            treatment_scheme=args.treatment_scheme,
            cluster_col=args.cluster_col,
            nuisance_best_prop=best_prop,
            nuisance_best_outcome=best_out,
            tau_best_params=best_tau,
            selected_models=dict(
                propensity=selected_prop_model,
                outcome=selected_outcome_model,
                tau=selected_tau_model,
            ),
            blp_metrics=blp_metrics,
            blp_metrics_path=str(blp_metrics_path),
        )
        meta_path = out_base / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        _log_progress(f"{tag}: meta saved -> {meta_path}")

        print(f"[done] {tag} -> {out_base}", flush=True)


if __name__ == "__main__":
    main()
