# dml_dr_core.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None
from sklearn.preprocessing import OneHotEncoder

from dml_models import make_classifier, make_regressor


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


def _to_dense_float32(X):
    # scipy sparse -> dense
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    return X.astype(np.float32, copy=False)


@dataclass
class DMLConfig:
    n_splits: int = 5
    outcome_model: Literal["xgb", "lgbm", "catboost"] = "xgb"
    prop_model: Literal["xgb", "lgbm", "catboost"] = "xgb"
    random_state: int = 123

    outcome_params: Dict[str, Any] = field(default_factory=dict)
    prop_params: Dict[str, Any] = field(default_factory=dict)

    # 「少ないクラスを回帰に使わない」ための安全弁
    min_samples_per_treat: int = 200

    # 互換用（run 側が渡しても落ちないように残す）
    cat_features_indices: Optional[List[int]] = None

    # 明示したければ名前で渡せる（Noneならdtypeから推定）
    categorical_cols: Optional[List[str]] = None
    group_col: Optional[str] = None


def crossfit_nuisance(
    df: pd.DataFrame,
    feature_cols: List[str],
    treat_col: str,
    outcome_col: str,
    config: DMLConfig,
) -> pd.DataFrame:
    """
    多値処置DMLの nuisance を cross-fitting で作る。

    CatBoost の場合はカテゴリ変数を One-Hot にせず、
    pandas.DataFrame と cat_features をそのまま CatBoost に渡す。
    XGB / LGBM の場合は従来通り One-Hot エンコードを使う。
    """
    df = df.reset_index(drop=True).copy()
    if treat_col not in df.columns:
        raise KeyError(f"treat_col not found: {treat_col}")
    if outcome_col not in df.columns:
        raise KeyError(f"outcome_col not found: {outcome_col}")

    # NA はとりあえず処置・アウトカムだけで落とす
    df0 = df.dropna(subset=[treat_col, outcome_col]).copy()

    # treatment を category に
    df0[treat_col] = df0[treat_col].astype("category")
    treat_levels = list(df0[treat_col].cat.categories)
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

    # CatBoost を使うならカテゴリ列を明示的に category に
    uses_catboost = (config.prop_model == "catboost") or (config.outcome_model == "catboost")
    if uses_catboost:
        for c in cat_cols:
            if c in X_df.columns:
                X_df[c] = X_df[c].astype("category")

    # XGB / LGBM 用の One-Hot encoder（CatBoost のみなら不要）
    uses_ohe = (config.prop_model in {"xgb", "lgbm"}) or (config.outcome_model in {"xgb", "lgbm"})
    pre = _build_preprocessor(feature_cols, cat_cols) if uses_ohe else None

    # allocate
    n = len(df0)
    m_hat = np.full((n, K), np.nan, dtype=np.float32)
    e_hat = np.full((n, K), np.nan, dtype=np.float32)

    # base estimators
    prop_est = make_classifier(
        config.prop_model,
        random_state=config.random_state,
        params=dict(config.prop_params),
        n_classes=K,
    )
    out_est_template = make_regressor(
        config.outcome_model,
        random_state=config.random_state,
        params=dict(config.outcome_params),
    )

    # propensity objective/loss
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

    if config.group_col is not None and config.group_col in df0.columns:
        groups = df0[config.group_col].to_numpy()
        split_iter = None
        if StratifiedGroupKFold is not None:
            try:
                sgkf = StratifiedGroupKFold(
                    n_splits=config.n_splits,
                    shuffle=True,
                    random_state=config.random_state,
                )
            except TypeError:
                sgkf = StratifiedGroupKFold(n_splits=config.n_splits)
            try:
                split_iter = list(sgkf.split(X_df, z, groups=groups))
                print(f"[crossfit] using StratifiedGroupKFold(n_splits={config.n_splits}) by '{config.group_col}'")
            except ValueError as e:
                print(f"[crossfit][warn] StratifiedGroupKFold failed ({e}); fallback to GroupKFold")
                split_iter = None
        if split_iter is None:
            gkf = GroupKFold(n_splits=config.n_splits)
            split_iter = gkf.split(X_df, z, groups=groups)
            print(f"[crossfit] using GroupKFold(n_splits={config.n_splits}) by '{config.group_col}'")
    else:
        splitter = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        split_iter = splitter.split(X_df, z)

    for fold, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        X_tr = X_df.iloc[tr_idx].copy()
        X_te = X_df.iloc[te_idx].copy()
        z_tr = z[tr_idx]
        y_tr = y[tr_idx]

        # OHE を使う learner があるときだけ encoder を学習
        if uses_ohe and pre is not None:
            pre_fold = clone(pre)
            pre_fold.fit(X_tr)
            Xtr_enc = pre_fold.transform(X_tr)
            Xte_enc = pre_fold.transform(X_te)
        else:
            Xtr_enc = Xte_enc = None

        # -------------------------
        # Propensity model
        # -------------------------
        counts = np.bincount(z_tr, minlength=K)
        if (counts == 0).any():
            # レアクラスがfoldに無い場合は frequency + Laplace smoothing
            p = (counts + 1.0) / (counts.sum() + K)
            e_hat[te_idx, :] = p.astype(np.float32)
        else:
            if config.prop_model == "catboost":
                prop_fold = clone(prop_est)
                # DataFrame + cat_features（列名）でそのまま入れる
                prob = prop_fold.fit(
                    X_tr, z_tr, cat_features=cat_cols
                ).predict_proba(X_te)
                prob = np.asarray(prob, dtype=np.float32)
            elif config.prop_model in {"xgb", "lgbm"}:
                if Xtr_enc is None or Xte_enc is None:
                    raise RuntimeError("Encoded features are missing for XGB/LGBM.")
                prop_fold = clone(prop_est)
                prob = prop_fold.fit(Xtr_enc, z_tr).predict_proba(Xte_enc)
                prob = np.asarray(prob, dtype=np.float32)
            else:
                raise ValueError(f"Unknown prop_model: {config.prop_model}")

            prob = np.clip(prob, 1e-6, 1.0)
            prob = prob / prob.sum(axis=1, keepdims=True)
            e_hat[te_idx, :] = prob

        # -------------------------
        # Outcome models (one per treatment)
        # -------------------------
        for k in range(K):
            mask = (z_tr == k)
            nk = int(mask.sum())
            if nk < config.min_samples_per_treat:
                # サンプルが少ないときは平均にフォールバック
                fallback = float(np.mean(y_tr[mask])) if nk > 0 else float(np.mean(y_tr))
                m_hat[te_idx, k] = np.float32(fallback)
                continue

            out_fold = clone(out_est_template)
            if config.outcome_model == "catboost":
                Xtr_y = X_tr.loc[mask].copy()
                Xte_y = X_te.copy()
                pred = out_fold.fit(
                    Xtr_y, y_tr[mask], cat_features=cat_cols
                ).predict(Xte_y)
                pred = np.asarray(pred, dtype=np.float32)
            elif config.outcome_model in {"xgb", "lgbm"}:
                if Xtr_enc is None or Xte_enc is None:
                    raise RuntimeError("Encoded features are missing for XGB/LGBM.")
                Xtr_y = Xtr_enc[mask]
                Xte_y = Xte_enc
                pred = out_fold.fit(Xtr_y, y_tr[mask]).predict(Xte_y)
                pred = np.asarray(pred, dtype=np.float32)
            else:
                raise ValueError(f"Unknown outcome_model: {config.outcome_model}")

            m_hat[te_idx, k] = pred

        print(f"[crossfit] fold {fold}/{config.n_splits} done")

    df_out = df0[[treat_col, outcome_col] + feature_cols].copy()
    for k, lvl in enumerate(treat_levels):
        df_out[f"m_hat_{lvl}"] = m_hat[:, k]
        df_out[f"e_hat_{lvl}"] = e_hat[:, k]

    return df_out
