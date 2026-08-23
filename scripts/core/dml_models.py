# dml_models.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional
import os


def _catboost_gpu_kwarss() -> Dict[str, Any]:
    """
    CATBOOST_USE_GPU=1 のときだけGPUを使う。
    使うGPUは CUDA_VISIBLE_DEVICES / CATBOOST_DEVICES に従う。
    """
    if os.getenv("CATBOOST_USE_GPU", "0") != "1":
        return {}
    devices = os.getenv("CATBOOST_DEVICES", "0")  # "0" / "0,1" 形式
    return {"task_type": "GPU", "devices": devices}


def make_regressor(
    model: Literal["xgb", "lgbm", "catboost"] = "xgb",
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None,
):
    params = dict(params or {})  # copy

    if model == "xgb":
        from xgboost import XGBRegressor

        xgb_threads = int(os.getenv("XGB_NUM_THREADS", "-1"))
        base = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=xgb_threads,
            random_state=random_state,
        )
        base.update(params)
        return XGBRegressor(**base)

    if model == "lgbm":
        import lightgbm as lgb

        lgbm_threads = int(os.getenv("LGBM_NUM_THREADS", "1"))
        base = dict(
            n_estimators=500,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            n_jobs=lgbm_threads,  # 外側で並列化する前提
            random_state=random_state,
            verbose=-1,
            verbosity=-1,
            force_row_wise=True,
        )
        base.update(params)
        return lgb.LGBMRegressor(**base)

    if model == "catboost":
        from catboost import CatBoostRegressor

        base = dict(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            thread_count=4,  # GPUでもCPU側前処理に使う
            allow_writing_files=False,
        )
        base.update(_catboost_gpu_kwarss())
        base.update(params)  # ユーザー指定が最優先
        return CatBoostRegressor(**base)

    raise ValueError(f"Unknown regressor model: {model}")


def make_classifier(
    model: Literal["xgb", "lgbm", "catboost"] = "xgb",
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None,
    n_classes: Optional[int] = None,
):
    params = dict(params or {})

    if model == "xgb":
        from xgboost import XGBClassifier

        xgb_threads = int(os.getenv("XGB_NUM_THREADS", "-1"))
        if n_classes == 2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"
        base = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=objective,
            n_jobs=xgb_threads,
            random_state=random_state,
        )
        if n_classes is not None and n_classes > 2:
            base["num_class"] = int(n_classes)
        base.update(params)
        return XGBClassifier(**base)

    if model == "lgbm":
        import lightgbm as lgb

        lgbm_threads = int(os.getenv("LGBM_NUM_THREADS", "1"))
        if n_classes == 2:
            objective = "binary"
        else:
            objective = "multiclass"
        base = dict(
            n_estimators=500,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective=objective,
            n_jobs=lgbm_threads,
            random_state=random_state,
            verbose=-1,
            verbosity=-1,
            force_row_wise=True,
        )
        if n_classes is not None and n_classes > 2:
            base["num_class"] = int(n_classes)
        base.update(params)
        return lgb.LGBMClassifier(**base)

    if model == "catboost":
        from catboost import CatBoostClassifier

        if n_classes == 2:
            loss_function = "Logloss"
        else:
            loss_function = "MultiClass"
        base = dict(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function=loss_function,
            random_seed=random_state,
            verbose=False,
            thread_count=4,
            allow_writing_files=False,
        )
        if n_classes is not None and n_classes > 2:
            base["classes_count"] = int(n_classes)
        base.update(_catboost_gpu_kwarss())
        base.update(params)
        return CatBoostClassifier(**base)

    raise ValueError(f"Unknown classifier model: {model}")
