# dml_tuning.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from joblib import parallel_backend
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from dml_models import make_classifier, make_regressor


def _maybe_save_cv_results(
    gs: GridSearchCV,
    outdir: str,
    tag: str,
    extra_cols: Optional[Dict[str, Any]] = None,
) -> None:
    if not outdir:
        return
    path = os.path.join(outdir, f"{tag}_cv_results.csv")
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(gs.cv_results_)
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    df.to_csv(path, index=False)
    print(f"[tune] saved cv_results: {path}")


def _subsample(df: pd.DataFrame, n_max: int, random_state: int) -> pd.DataFrame:
    if len(df) <= n_max:
        return df
    return df.sample(n=n_max, random_state=random_state)


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


def _make_ohe(dense: bool):
    # sklearn version compatibility
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=(not dense))
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=(not dense))


def _make_preprocessor(feature_cols: List[str], categorical_cols: List[str], dense: bool):
    num_cols = [c for c in feature_cols if c not in categorical_cols]
    ohe = _make_ohe(dense=dense)

    transformers = []
    if categorical_cols:
        transformers.append(("cat", ohe, categorical_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    return ColumnTransformer(transformers, remainder="drop")


def tune_outcome_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    outcome_col: str,
    model_name: Literal["xgb", "lgbm", "catboost"],
    n_max: int = 100_000,
    random_state: int = 123,
    categorical_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    param_grid_override: Optional[Dict[str, List[Any]]] = None,
    cv_results_outdir: str = "data/analysis/tuning",
) -> Dict[str, Any]:
    df_sub = _subsample(df, n_max, random_state).copy()

    if categorical_cols is None:
        categorical_cols = _infer_categorical_cols(df_sub, feature_cols)
    categorical_cols = [c for c in categorical_cols if c in feature_cols and c in df_sub.columns]

    X = df_sub[feature_cols].copy()
    y = df_sub[outcome_col].to_numpy()

    # ------------------------------------------------------------
    # XGB / LGBM: 従来通り One-Hot エンコード
    # CatBoost : DataFrame をそのまま渡し、cat_features を指定
    # ------------------------------------------------------------
    if model_name == "catboost":
        # int のID等も含めて、明示的に category にしておく
        for c in categorical_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")

        from sklearn.preprocessing import FunctionTransformer
        pre = FunctionTransformer(lambda a: a, validate=False)  # identity
        est = make_regressor(model_name, random_state=random_state, params={})
        pipe = Pipeline([("pre", pre), ("model", est)])

        param_grid = (
            param_grid_override
            if param_grid_override is not None
            else {
                "model__depth": [4, 6, 8],
                "model__iterations": [300, 600],
                "model__learning_rate": [0.03, 0.1],
            }
        )

        if group_col is not None and group_col in df_sub.columns:
            cv = GroupKFold(n_splits=3)
            groups = df_sub[group_col].to_numpy()
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
            groups = None
        n_jobs_setting = 1  # CatBoost は内部並列で十分
        fit_params = {"model__cat_features": categorical_cols}

    else:
        dense = False  # XGB / LGBM は疎行列のままでOK
        pre = _make_preprocessor(feature_cols, categorical_cols, dense=dense)
        est = make_regressor(model_name, random_state=random_state, params={})
        pipe = Pipeline([("pre", pre), ("model", est)])

        if model_name == "xgb":
            param_grid = {
                "model__max_depth": [3, 4, 5],
                "model__n_estimators": [300, 500],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.7, 0.9],
            }
        elif model_name == "lgbm":
            param_grid = {
                "model__num_leaves": [31, 63],
                "model__n_estimators": [300, 600],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.7, 0.9],
            }
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        if group_col is not None and group_col in df_sub.columns:
            cv = GroupKFold(n_splits=3)
            groups = df_sub[group_col].to_numpy()
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
            groups = None
        n_jobs_setting = int(os.environ.get("DML_TUNE_JOBS", "4"))
        fit_params = {}

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs_setting,
        verbose=1,
    )
    with parallel_backend("threading"):
        if groups is None:
            gs.fit(X, y, **fit_params)
        else:
            gs.fit(X, y, groups=groups, **fit_params)
    print(f"[tune_outcome] best params for {model_name}: {gs.best_params_}")
    _maybe_save_cv_results(
        gs,
        outdir=cv_results_outdir,
        tag=f"{model_name}_outcome_{outcome_col}",
        extra_cols={
            "task": "outcome",
            "model_name": model_name,
            "outcome_col": outcome_col,
            "random_state": random_state,
        },
    )
    return {k.replace("model__", ""): v for k, v in gs.best_params_.items()}


def tune_propensity_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    treat_col: str,
    model_name: Literal["xgb", "lgbm", "catboost"],
    n_max: int = 100_000,
    random_state: int = 123,
    categorical_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    param_grid_override: Optional[Dict[str, List[Any]]] = None,
    cv_results_outdir: str = "data/analysis/tuning",
) -> Dict[str, Any]:
    df_sub = _subsample(df, n_max, random_state).copy()

    df_sub[treat_col] = df_sub[treat_col].astype("category")
    z = df_sub[treat_col].cat.codes.to_numpy()
    if (z < 0).any():
        raise ValueError("Treatment labels contain missing values after category encoding.")
    n_classes = int(np.unique(z).size)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 treatment classes, found {n_classes}.")

    if categorical_cols is None:
        categorical_cols = _infer_categorical_cols(df_sub, feature_cols)
    categorical_cols = [c for c in categorical_cols if c in feature_cols and c in df_sub.columns]

    X = df_sub[feature_cols].copy()

    # ------------------------------------------------------------
    # XGB / LGBM: One-Hot エンコード
    # CatBoost : DataFrame をそのまま渡し、cat_features を指定
    # ------------------------------------------------------------
    if model_name == "catboost":
        # カテゴリカルカラムを category 型に変換
        for c in categorical_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")

        base_params = {"loss_function": "Logloss"} if n_classes == 2 else {"loss_function": "MultiClass"}
        pre = FunctionTransformer(lambda a: a, validate=False)  # identity
        est = make_classifier(model_name, random_state=random_state, params=base_params, n_classes=n_classes)
        pipe = Pipeline([("pre", pre), ("model", est)])

        param_grid = (
            param_grid_override
            if param_grid_override is not None
            else {
                "model__depth": [4, 6, 8],
                "model__iterations": [300, 600],
                "model__learning_rate": [0.03, 0.1],
            }
        )

        if group_col is not None and group_col in df_sub.columns:
            cv = GroupKFold(n_splits=3)
            groups = df_sub[group_col].to_numpy()
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            groups = None
        n_jobs_setting = 1
        fit_params = {"model__cat_features": categorical_cols}

    else:
        dense = False
        pre = _make_preprocessor(feature_cols, categorical_cols, dense=dense)
        if model_name == "xgb":
            base_params = (
                {"objective": "binary:logistic"}
                if n_classes == 2
                else {"objective": "multi:softprob", "num_class": n_classes}
            )
        elif model_name == "lgbm":
            base_params = (
                {"objective": "binary"}
                if n_classes == 2
                else {"objective": "multiclass", "num_class": n_classes}
            )
        else:
            base_params = {}
        est = make_classifier(model_name, random_state=random_state, params=base_params, n_classes=n_classes)
        pipe = Pipeline([("pre", pre), ("model", est)])

        if model_name == "xgb":
            param_grid = {
                "model__max_depth": [3, 4, 5],
                "model__n_estimators": [300, 500],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.7, 0.9],
            }
        elif model_name == "lgbm":
            param_grid = {
                "model__num_leaves": [31, 63],
                "model__n_estimators": [300, 600],
                "model__subsample": [0.7, 0.9],
                "model__colsample_bytree": [0.7, 0.9],
            }
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        if group_col is not None and group_col in df_sub.columns:
            cv = GroupKFold(n_splits=3)
            groups = df_sub[group_col].to_numpy()
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            groups = None
        n_jobs_setting = int(os.environ.get("DML_TUNE_JOBS", "4"))
        fit_params = {}

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=n_jobs_setting,
        verbose=1,
    )
    with parallel_backend("threading"):
        if groups is None:
            gs.fit(X, z, **fit_params)
        else:
            gs.fit(X, z, groups=groups, **fit_params)
    print(f"[tune_propensity] best params for {model_name}: {gs.best_params_}")
    _maybe_save_cv_results(
        gs,
        outdir=cv_results_outdir,
        tag=f"{model_name}_propensity_{treat_col}",
        extra_cols={
            "task": "propensity",
            "model_name": model_name,
            "treat_col": treat_col,
            "random_state": random_state,
        },
    )
    return {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
