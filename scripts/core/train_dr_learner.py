#!/usr/bin/env python3
"""
train_dr_learner.py

DML-AIPW の cross-fitting 出力（m_hat_* / e_hat_* 付き parquet）を入力として、
DR-learner（= AIPW pseudo-outcome を使った 2nd-stage 回帰）を学習し、保存する。

- 二値処置/多値処置: どちらも対応（処置水準 k ごとにモデルを学習）
- 前提: 入力に m_hat_<level> と e_hat_<level> が各水準分そろっていること
  - 例外: K=2 かつ e_hat が片側だけある場合は、もう片側を 1-e_hat で補完できる

Pseudo outcome（各水準 k）:
  psi_k = m_hat_k(X) + 1[Z=k] * (Y - m_hat_k(X)) / clip(e_hat_k(X))

2nd-stage tuning:
  --tune を指定すると dml_tuning.tune_outcome_model() によりハイパラ調整。
  - --tune-per-treat: 水準ごとに psi_k をチューニング（最も整合的だが遅い）
  - 共有チューニング（--tune かつ --tune-per-treat なし）では、--tune-target で
    チューニング対象（psi由来）を選べる:
      * observed: psi_{Z_i}(X_i)（各行の実処置に対応するpsi）
      * baseline: psi_{baseline}(X_i)
      * mean: 平均 psī(X_i) = (1/K) Σ_k psi_k(X_i)

既定の特徴量:
  run_dml_all_models.py に合わせて、
    time_left_game, score_diff, OT_flag, start_type, start_type_group,
    after_off_reb, elo_diff, before_home_possession, era, offense_team, defense_team
  に加え、存在すれば own_*_eb / opp_*_eb も自動で含める。

出力:
  <outdir>/<prefix>dr_learner_<subset>_<model>.joblib
  <outdir>/<prefix>dr_learner_<subset>_<model>_meta.json
  （任意）<outdir>/<prefix>dr_learner_<subset>_<model>_preds.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold, StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None

from dml_models import make_regressor
from dml_tuning import tune_outcome_model

DR_CATBOOST_PARAM_GRID = {
    "model__depth": [4, 6, 8, 10],
    "model__iterations": [200, 400, 800, 1200],
    "model__learning_rate": [0.01, 0.03, 0.1, 0.2],
}

def _make_ohe_sparse() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def infer_categorical_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    cat_cols: List[str] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        if (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
        ):
            cat_cols.append(c)
    # common identifiers (keep if present)
    for c in ["era", "offense_team", "defense_team"]:
        if c in feature_cols and c in df.columns and c not in cat_cols:
            cat_cols.append(c)
    return cat_cols


def build_preprocessor(feature_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_cols = [c for c in feature_cols if c not in categorical_cols]
    transformers = []
    if categorical_cols:
        transformers.append(("cat", _make_ohe_sparse(), categorical_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    return ColumnTransformer(transformers, remainder="drop")


def maybe_fill_binary_propensity(
    df: pd.DataFrame,
    treat_levels: List[Any],
    e_prefix: str = "e_hat_",
) -> bool:
    """K=2 のとき e_hat が片側だけある場合に 1-e_hat で補完する。"""
    if len(treat_levels) != 2:
        return False
    expected = [f"{e_prefix}{lvl}" for lvl in treat_levels]
    present = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    if len(present) == 1 and len(missing) == 1:
        df[missing[0]] = 1.0 - df[present[0]].astype(float)
        return True
    return False


def ensure_nuisance_columns(
    df: pd.DataFrame,
    treat_col: str,
    m_prefix: str = "m_hat_",
    e_prefix: str = "e_hat_",
) -> List[Any]:
    z = df[treat_col].astype("category")
    cats = z.cat.categories.tolist()
    missing = []
    for label in cats:
        m_col = f"{m_prefix}{label}"
        e_col = f"{e_prefix}{label}"
        if m_col not in df.columns:
            missing.append(m_col)
        if e_col not in df.columns:
            missing.append(e_col)
    if missing:
        raise ValueError(
            f"Missing nuisance columns: {missing}\n"
            f"Input must include m_hat_<level> and e_hat_<level> for all treatment levels.\n"
            f"(If you changed treatment mapping, re-run run_dml_all_models.py with the same scheme.)"
        )
    return cats


def compute_pseudo_outcomes(
    df: pd.DataFrame,
    treat_col: str,
    outcome_col: str,
    treat_levels: List[Any],
    m_prefix: str = "m_hat_",
    e_prefix: str = "e_hat_",
    min_prop: float = 1e-2,
    max_prop: float = 1.0,
) -> Dict[Any, np.ndarray]:
    df0 = df.copy()
    df0[treat_col] = df0[treat_col].astype("category")
    y = df0[outcome_col].to_numpy(dtype=np.float32)
    z = df0[treat_col]

    psi: Dict[Any, np.ndarray] = {}
    for label in treat_levels:
        m = df0[f"{m_prefix}{label}"].to_numpy(dtype=np.float32)
        e = df0[f"{e_prefix}{label}"].to_numpy(dtype=np.float32)
        e = np.clip(e, min_prop, max_prop).astype(np.float32)

        ind = (z == label).to_numpy(dtype=np.float32)
        psi_k = m + ind * (y - m) / e
        psi[label] = psi_k.astype(np.float32)
    return psi


def fit_one_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: np.ndarray,
    model_name: str,
    cat_cols: List[str],
    tuned_params: Optional[Dict[str, Any]] = None,
    random_state: int = 123,
) -> Any:
    """Fit μ_k(x) model for one treatment level."""
    X = df[feature_cols].copy()

    if model_name == "catboost":
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        est = make_regressor("catboost", random_state=random_state, params=tuned_params or {})
        est.fit(X, y, cat_features=cat_cols)
        return est

    if model_name in {"xgb", "lgbm"}:
        pre = build_preprocessor(feature_cols, cat_cols)
        est = make_regressor(model_name, random_state=random_state, params=tuned_params or {})
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X, y)
        return pipe

    raise ValueError(f"Unknown model_name: {model_name}")


def _predict_mu(
    mdl: Any,
    df_part: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    cat_cols: List[str],
) -> np.ndarray:
    """Predict μ_k(x) on df_part with the same preprocessing as fit_one_model."""
    X = df_part[feature_cols].copy()
    if model_name == "catboost":
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        pred = mdl.predict(X)
    else:
        pred = mdl.predict(X)
    return np.asarray(pred, dtype=np.float32)


def compute_second_stage_oof_preds(
    df: pd.DataFrame,
    treat_col: str,
    feature_cols: List[str],
    cat_cols: List[str],
    model_name: str,
    psi_dict: Dict[Any, np.ndarray],
    treat_levels: List[Any],
    baseline_lvl: Any,
    tuned_params_by_level: Dict[str, Dict[str, Any]],
    tune: bool,
    tune_per_treat: bool,
    tune_target: str,
    random_state: int,
    n_splits: int,
    group_col: Optional[str] = None,
    tune_in_fold: bool = False,
) -> pd.DataFrame:
    """Compute out-of-fold μ/τ predictions for the 2nd-stage models.

    Notes:
      - 1st-stage nuisances (m_hat_*, e_hat_*) are assumed cross-fitted in the input.
      - This function only cross-fits the 2nd-stage regression, returning OOF predictions.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for OOF predictions.")

    param_grid_override = DR_CATBOOST_PARAM_GRID if model_name == "catboost" else None

    n = len(df)
    fold_id = np.full(n, -1, dtype=np.int32)
    mu_oof: Dict[str, np.ndarray] = {str(lvl): np.full(n, np.nan, dtype=np.float32) for lvl in treat_levels}

    # choose splitter
    use_group = group_col is not None and group_col in df.columns
    if use_group:
        groups = df[group_col].values
        split_iter = None
        if StratifiedGroupKFold is not None:
            y_strat = df[treat_col].astype(str).values
            try:
                sgkf = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_state,
                )
            except TypeError:
                sgkf = StratifiedGroupKFold(n_splits=n_splits)
            try:
                split_iter = list(sgkf.split(df, y=y_strat, groups=groups))
                print(f"[oof] using StratifiedGroupKFold(n_splits={n_splits}) by group_col='{group_col}'")
            except ValueError as e:
                print(f"[oof][warn] StratifiedGroupKFold failed ({e}); fallback to GroupKFold")
                split_iter = None
        if split_iter is None:
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(df, groups=groups)
            print(f"[oof] using GroupKFold(n_splits={n_splits}) by group_col='{group_col}'")
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_strat = df[treat_col].astype(str).values
        split_iter = splitter.split(df, y=y_strat)
        print(f"[oof] using StratifiedKFold(n_splits={n_splits}) stratified by '{treat_col}'")

    for f, (tr_idx, va_idx) in enumerate(split_iter):
        fold_id[va_idx] = f
        df_tr = df.iloc[tr_idx].copy()
        df_va = df.iloc[va_idx].copy()

        # keep categorical levels consistent
        df_tr[treat_col] = df_tr[treat_col].astype("category").cat.set_categories(treat_levels)
        df_va[treat_col] = df_va[treat_col].astype("category").cat.set_categories(treat_levels)

        # Optionally tune within each fold (stricter but slower)
        tuned_params_shared_fold: Dict[str, Any] = {}
        tuned_params_by_level_fold: Dict[str, Dict[str, Any]] = {}
        if tune and tune_in_fold:
            if tune and not tune_per_treat:
                psi_tr = {lvl: psi_dict[lvl][tr_idx] for lvl in treat_levels}
                y_tune_tr = _tune_target_from_psi(
                    df=df_tr,
                    treat_col=treat_col,
                    treat_levels=treat_levels,
                    psi_dict=psi_tr,
                    baseline_lvl=baseline_lvl,
                    tune_target=tune_target,
                )
                tuned_params_shared_fold = maybe_tune_params(
                    df=df_tr,
                    feature_cols=feature_cols,
                    target_y=y_tune_tr,
                    model_name=model_name,
                    tune=True,
                    random_state=random_state,
                    cat_cols=cat_cols,
                    param_grid_override=param_grid_override,
                )
            # per-treat tuning inside fold is very expensive; do it only if explicitly requested
            if tune_per_treat:
                for lvl in treat_levels:
                    psi_tr_k = psi_dict[lvl][tr_idx]
                    tuned_params_by_level_fold[str(lvl)] = maybe_tune_params(
                        df=df_tr,
                        feature_cols=feature_cols,
                        target_y=psi_tr_k,
                        model_name=model_name,
                        tune=True,
                        random_state=random_state,
                        cat_cols=cat_cols,
                        param_grid_override=param_grid_override,
                    )

        for lvl in treat_levels:
            y_tr = psi_dict[lvl][tr_idx]
            if tune and tune_in_fold:
                if tune_per_treat:
                    params = tuned_params_by_level_fold.get(str(lvl), {})
                else:
                    params = tuned_params_shared_fold
            else:
                # fixed params learned on full data (recommended for speed)
                params = tuned_params_by_level.get(str(lvl), {}) if tune else {}

            mdl = fit_one_model(
                df=df_tr,
                feature_cols=feature_cols,
                y=y_tr,
                model_name=model_name,
                cat_cols=cat_cols,
                tuned_params=params,
                random_state=random_state,
            )
            mu_oof[str(lvl)][va_idx] = _predict_mu(mdl, df_va, feature_cols, model_name, cat_cols)

        print(f"[oof] fold={f} done  train={len(tr_idx):,}  valid={len(va_idx):,}")

    # build output
    out = pd.DataFrame(
        {
            treat_col: df[treat_col].astype(str).values,
            "fold_id": fold_id,
        }
    )
    base = mu_oof[str(baseline_lvl)]
    for lvl_str, pred in mu_oof.items():
        out[f"mu_hat_{lvl_str}_oof"] = pred
        out[f"tau_hat_{lvl_str}_vs_base_oof"] = pred - base

    # sanity check
    if np.any(fold_id < 0):
        raise RuntimeError("OOF fold assignment incomplete (some rows never validated).")
    for lvl_str in mu_oof.keys():
        if np.any(~np.isfinite(out[f"mu_hat_{lvl_str}_oof"].to_numpy())):
            raise RuntimeError(f"OOF prediction contains NaN/inf for level {lvl_str}.")

    return out


def maybe_tune_params(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_y: np.ndarray,
    model_name: str,
    tune: bool,
    random_state: int = 123,
    cat_cols: Optional[List[str]] = None,
    param_grid_override: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    """Return tuned params dict (possibly empty). Uses dml_tuning.tune_outcome_model on target_y."""
    if not tune:
        return {}
    df_tmp = df.copy()
    df_tmp["_dr_tmp_target"] = target_y
    params = tune_outcome_model(
        df_tmp,
        feature_cols=feature_cols,
        outcome_col="_dr_tmp_target",
        model_name=model_name,
        n_max=len(df_tmp),
        random_state=random_state,
        categorical_cols=cat_cols,
        param_grid_override=param_grid_override,
    )
    return dict(params or {})


def default_feature_cols(df: pd.DataFrame, include_team_ids: bool = False) -> List[str]:
    # Same as run_dml_all_models.py (plus own_*/opp_*_eb if present).
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


def parse_cols_arg(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _choose_baseline(df: pd.DataFrame, treat_col: str) -> Any:
    vc = df[treat_col].astype("category").value_counts()
    return vc.index[0]


def _tune_target_from_psi(
    df: pd.DataFrame,
    treat_col: str,
    treat_levels: List[Any],
    psi_dict: Dict[Any, np.ndarray],
    baseline_lvl: Any,
    tune_target: str,
) -> np.ndarray:
    tune_target = tune_target.lower()
    if tune_target == "baseline":
        return psi_dict[baseline_lvl]
    if tune_target == "mean":
        arr = np.stack([psi_dict[lvl] for lvl in treat_levels], axis=0)
        return arr.mean(axis=0).astype(np.float32)
    if tune_target == "observed":
        z = df[treat_col].astype("category")
        out = np.empty(len(df), dtype=np.float32)
        for lvl in treat_levels:
            mask = (z == lvl).to_numpy()
            out[mask] = psi_dict[lvl][mask]
        return out
    raise ValueError(f"Unknown tune_target: {tune_target}. Choose from observed|baseline|mean.")


def run_one_subset(
    name: str,
    path: Path,
    outdir: Path,
    model_name: str,
    treat_col: str,
    outcome_col: str,
    feature_cols: List[str],
    categorical_cols: Optional[List[str]],
    baseline: Optional[str],
    min_prop: float,
    max_prop: float,
    tune: bool,
    tune_per_treat: bool,
    tune_target: str,
    random_state: int,
    save_preds: bool,
    prefix: str,
    oof_folds: int,
    oof_group_col: Optional[str],
    save_oof_preds: bool,
    oof_tune_in_fold: bool,
    include_team_ids: bool,
) -> None:
    print(f"\n=== DR-learner: subset={name}  input={path} ===")
    df = pd.read_parquet(path)
    param_grid_override = DR_CATBOOST_PARAM_GRID if model_name == "catboost" else None

    if treat_col not in df.columns:
        raise ValueError(f"{treat_col=} not found in {path}")
    if outcome_col not in df.columns:
        raise ValueError(f"{outcome_col=} not found in {path}")

    if not feature_cols:
        feature_cols = default_feature_cols(df, include_team_ids=include_team_ids)
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in input: {missing_features}")

    if categorical_cols is None:
        cat_cols = infer_categorical_cols(df, feature_cols)
    else:
        cat_cols = [c for c in categorical_cols if c in feature_cols]

    df[treat_col] = df[treat_col].astype("category")
    treat_levels = df[treat_col].cat.categories.tolist()

    filled = maybe_fill_binary_propensity(df, treat_levels=treat_levels)
    if filled:
        print("[WARN] binary propensity column missing for one level; filled with 1 - other e_hat.")

    _ = ensure_nuisance_columns(df, treat_col=treat_col)
    treat_levels = df[treat_col].cat.categories.tolist()

    if baseline is not None:
        if baseline not in [str(x) for x in treat_levels]:
            raise ValueError(f"baseline='{baseline}' not found in treatment levels: {treat_levels}")
        for lvl in treat_levels:
            if str(lvl) == baseline:
                baseline_lvl = lvl
                break
    else:
        baseline_lvl = _choose_baseline(df, treat_col)
        print(f"[WARN] --baseline not specified; using most frequent level as baseline: {baseline_lvl}")

    psi_dict = compute_pseudo_outcomes(
        df,
        treat_col=treat_col,
        outcome_col=outcome_col,
        treat_levels=treat_levels,
        min_prop=min_prop,
        max_prop=max_prop,
    )

    tuned_params_shared: Dict[str, Any] = {}
    if tune and not tune_per_treat:
        y_tune = _tune_target_from_psi(
            df=df,
            treat_col=treat_col,
            treat_levels=treat_levels,
            psi_dict=psi_dict,
            baseline_lvl=baseline_lvl,
            tune_target=tune_target,
        )
        tuned_params_shared = maybe_tune_params(
            df=df,
            feature_cols=feature_cols,
            target_y=y_tune,
            model_name=model_name,
            tune=True,
            random_state=random_state,
            cat_cols=cat_cols,
            param_grid_override=param_grid_override,
        )
        print(f"[tune] shared params for model={model_name} (target={tune_target}): {tuned_params_shared}")

    models: Dict[str, Any] = {}
    tuned_params_by_level: Dict[str, Dict[str, Any]] = {}

    for lvl in treat_levels:
        y_pseudo = psi_dict[lvl]
        if tune and tune_per_treat:
            params = maybe_tune_params(
                df=df,
                feature_cols=feature_cols,
                target_y=y_pseudo,
                model_name=model_name,
                tune=True,
                random_state=random_state,
                cat_cols=cat_cols,
                param_grid_override=param_grid_override,
            )
        else:
            params = tuned_params_shared

        tuned_params_by_level[str(lvl)] = dict(params or {})
        models[str(lvl)] = fit_one_model(
            df=df,
            feature_cols=feature_cols,
            y=y_pseudo,
            model_name=model_name,
            cat_cols=cat_cols,
            tuned_params=params,
            random_state=random_state,
        )
        print(f"[fit] level={lvl} done")

    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}dr_learner_{name}_{model_name}"
    model_path = outdir / f"{stem}.joblib"
    meta_path = outdir / f"{stem}_meta.json"

    payload = dict(
        model_name=model_name,
        subset=name,
        treat_col=treat_col,
        outcome_col=outcome_col,
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
        treat_levels=[str(x) for x in treat_levels],
        baseline=str(baseline_lvl),
        min_prop=min_prop,
        max_prop=max_prop,
        tune=tune,
        tune_per_treat=tune_per_treat,
        tune_target=tune_target,
        tuned_params_by_level=tuned_params_by_level,
        oof_folds=oof_folds,
        oof_group_col=oof_group_col,
        save_oof_preds=save_oof_preds,
        oof_tune_in_fold=oof_tune_in_fold,
        models=models,
    )
    dump(payload, model_path)
    print(f"[saved] {model_path}")

    meta = payload.copy()
    meta.pop("models", None)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[saved] {meta_path}")

    if save_preds:
        X = df[feature_cols].copy()
        mu: Dict[str, np.ndarray] = {}
        for lvl_str, mdl in models.items():
            if model_name == "catboost":
                Xcb = X.copy()
                for c in cat_cols:
                    if c in Xcb.columns:
                        Xcb[c] = Xcb[c].astype("category")
                pred = mdl.predict(Xcb)
            else:
                pred = mdl.predict(X)
            mu[lvl_str] = np.asarray(pred, dtype=np.float32)

        base = mu[str(baseline_lvl)]
        out = pd.DataFrame({treat_col: df[treat_col].astype(str).values})
        for lvl_str, pred in mu.items():
            out[f"mu_hat_{lvl_str}"] = pred
            out[f"tau_hat_{lvl_str}_vs_base"] = pred - base

        preds_path = outdir / f"{stem}_preds.parquet"
        out.to_parquet(preds_path, index=False)
        print(f"[saved] {preds_path}  cols={len(out.columns)}  rows={len(out):,}")


    if save_oof_preds and oof_folds and oof_folds >= 2:
        oof_out = compute_second_stage_oof_preds(
            df=df,
            treat_col=treat_col,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            model_name=model_name,
            psi_dict=psi_dict,
            treat_levels=treat_levels,
            baseline_lvl=baseline_lvl,
            tuned_params_by_level=tuned_params_by_level,
            tune=tune,
            tune_per_treat=tune_per_treat,
            tune_target=tune_target,
            random_state=random_state,
            n_splits=oof_folds,
            group_col=oof_group_col,
            tune_in_fold=oof_tune_in_fold,
        )
        
        oof_path = outdir / f"{stem}_oof_preds.parquet"
        oof_out.to_parquet(oof_path, index=False)
        print(f"[saved] {oof_path}  cols={len(oof_out.columns)}  rows={len(oof_out):,}")



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DR-learner from DML nuisance outputs.")
    p.add_argument("--input", type=str, default=None, help="clutch nuisance parquet (unified)")
    p.add_argument("--input-strict", type=str, default=None, help="(deprecated) strict clutch nuisance parquet")
    p.add_argument("--input-extended", type=str, default=None, help="(deprecated) extended clutch nuisance parquet")
    p.add_argument("--outdir", type=str, default="models/dr_learner", help="output directory")
    p.add_argument("--prefix", type=str, default="", help="prefix for output filenames (e.g., 'rs_')")
    p.add_argument("--model", type=str, choices=["xgb", "lgbm", "catboost"], required=True, help="final-stage learner")
    p.add_argument("--treat-col", type=str, default="shot_zone_choice")
    p.add_argument("--outcome-col", type=str, default="delta_wp")

    p.add_argument("--feature-cols", type=str, default=None, help="comma-separated feature columns")
    p.add_argument("--categorical-cols", type=str, default=None, help="comma-separated categorical cols (subset of features)")
    p.add_argument("--baseline", type=str, default=None, help="baseline treatment level (string match; recommended)")
    p.add_argument(
        "--include-team-ids",
        action="store_true",
        help="include offense_team/defense_team in default feature set",
    )

    p.add_argument("--min-prop", type=float, default=1e-2, help="clip lower bound for e_hat")
    p.add_argument("--max-prop", type=float, default=1.0, help="clip upper bound for e_hat")

    p.add_argument("--tune", action="store_true", help="enable hyperparameter tuning (GridSearchCV)")
    p.add_argument("--tune-per-treat", action="store_true", help="tune separately for each treatment level (slower)")
    p.add_argument(
        "--tune-target",
        type=str,
        default="observed",
        choices=["observed", "baseline", "mean"],
        help="target for shared tuning when --tune and not --tune-per-treat",
    )
    p.add_argument("--random-state", type=int, default=123)
    p.add_argument("--save-preds", action="store_true", help="save in-sample mu/tau predictions parquet")
    p.add_argument("--oof-folds", type=int, default=5, help="if >=2, compute 2nd-stage out-of-fold predictions (K-fold)")
    p.add_argument("--oof-group-col", type=str, default=None, help="optional group column for GroupKFold (e.g., GAME_ID)")
    p.add_argument("--save-oof-preds", action="store_true", help="save out-of-fold mu/tau predictions parquet")
    p.add_argument("--oof-tune-in-fold", action="store_true", help="(slow) tune 2nd-stage hyperparams within each fold")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    input_paths: List[Path] = []
    if args.input:
        input_paths.append(Path(args.input))
    else:
        if args.input_strict or args.input_extended:
            print("[WARN] --input-strict/--input-extended are deprecated; concatenating for unified training.")
        if args.input_strict:
            input_paths.append(Path(args.input_strict))
        if args.input_extended:
            input_paths.append(Path(args.input_extended))
    if not input_paths:
        raise SystemExit("Please specify --input (or deprecated --input-strict/--input-extended).")

    outdir = Path(args.outdir)
    feature_cols = parse_cols_arg(args.feature_cols) or []
    categorical_cols = parse_cols_arg(args.categorical_cols)
    if len(input_paths) == 1:
        input_path = input_paths[0]
    else:
        dfs = []
        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(f"Input not found: {p}")
            dfs.append(pd.read_parquet(p))
        tmp = pd.concat(dfs, ignore_index=True)
        input_path = outdir / f"{args.prefix}clutch_nuisance_merged.parquet"
        tmp.to_parquet(input_path, index=False)
        print(f"[WARN] merged nuisance saved to {input_path}")

    run_one_subset(
        name="clutch",
        path=input_path,
        outdir=outdir,
        model_name=args.model,
        treat_col=args.treat_col,
        outcome_col=args.outcome_col,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        baseline=args.baseline,
        min_prop=args.min_prop,
        max_prop=args.max_prop,
        tune=args.tune,
        tune_per_treat=args.tune_per_treat,
        tune_target=args.tune_target,
        random_state=args.random_state,
        save_preds=args.save_preds,
        prefix=args.prefix,
        oof_folds=args.oof_folds,
        oof_group_col=args.oof_group_col,
        save_oof_preds=args.save_oof_preds,
        oof_tune_in_fold=args.oof_tune_in_fold,
        include_team_ids=args.include_team_ids,
    )


if __name__ == "__main__":
    main()
