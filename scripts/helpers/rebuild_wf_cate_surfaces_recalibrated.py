#!/usr/bin/env python3
"""Rebuild walk-forward outer-test CATE surfaces with BLP-calibrated tau."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_DIRS = {
    "catboost": "results/nested_wf_catboost_gpu",
    "xgb": "results/nested_wf_xgb",
    "lgbm": "results/nested_wf_lgbm",
}

KEY_COLS = ["GAME_ID", "GAME_EVENT_ID", "season"]
MERGE_COLS = ["__row_id"]
ROW_COLS = [
    "GAME_ID",
    "GAME_EVENT_ID",
    "season",
    "time_left_game",
    "score_diff",
    "shot_zone_choice",
    "delta_wp",
]
FOLD_RE = re.compile(r"train(?P<train_start>\d{4})_(?P<train_end>\d{4})_test(?P<test_start>\d{4})_(?P<test_end>\d{4})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        default="results/wf_cate_surfaces",
    )
    p.add_argument("--catboost-dir", default=MODEL_DIRS["catboost"])
    p.add_argument("--xgb-dir", default=MODEL_DIRS["xgb"])
    p.add_argument("--lgbm-dir", default=MODEL_DIRS["lgbm"])
    p.add_argument("--score-lo", type=int, default=-10)
    p.add_argument("--score-hi", type=int, default=10)
    return p.parse_args()


def fold_years(outer_fold: str) -> tuple[str, str]:
    m = FOLD_RE.fullmatch(outer_fold)
    if not m:
        raise ValueError(f"Cannot parse outer fold name: {outer_fold}")
    return (
        f"{m.group('train_start')}-{m.group('train_end')}",
        f"{m.group('test_start')}-{m.group('test_end')}",
    )


def read_blp(path: Path) -> tuple[float, float]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return float(metrics["blp_alpha"]), float(metrics["blp_beta"])


def read_fold_model(outer_dir: Path, label: str) -> pd.DataFrame:
    tau_path = outer_dir / "tau_test.parquet"
    blp_path = outer_dir / "blp_metrics_oos_train.json"
    if not tau_path.exists():
        raise FileNotFoundError(tau_path)
    if not blp_path.exists():
        raise FileNotFoundError(blp_path)
    alpha, beta = read_blp(blp_path)
    df = pd.read_parquet(tau_path)
    needed = ROW_COLS + ["tau_hat"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{tau_path} missing columns: {missing}")
    out = df[ROW_COLS + ["tau_hat"]].copy()
    out["__row_id"] = np.arange(len(out), dtype=np.int64)
    out = out.rename(columns={"tau_hat": f"tau_{label}_raw"})
    out[f"tau_{label}"] = alpha + beta * out[f"tau_{label}_raw"].astype(float)
    out[f"tau_{label}_calibration_alpha"] = alpha
    out[f"tau_{label}_calibration_beta"] = beta
    return out


def build_rows(model_dirs: dict[str, Path]) -> pd.DataFrame:
    outer_folds = sorted(p.name for p in model_dirs["catboost"].iterdir() if p.is_dir() and p.name.startswith("train"))
    rows: list[pd.DataFrame] = []
    for outer_fold in outer_folds:
        cur: pd.DataFrame | None = None
        for label, base_dir in model_dirs.items():
            model_df = read_fold_model(base_dir / outer_fold, label)
            cols = MERGE_COLS + [c for c in model_df.columns if c not in MERGE_COLS and c not in ROW_COLS]
            if cur is None:
                cur = model_df.copy()
            else:
                if len(cur) != len(model_df):
                    raise ValueError(f"Row count mismatch for {outer_fold}: {len(cur)} vs {len(model_df)}")
                cur = cur.merge(model_df[cols], on=MERGE_COLS, how="inner", validate="one_to_one")
                if len(cur) != len(model_df):
                    raise ValueError(f"Row alignment mismatch for {outer_fold}")
        assert cur is not None
        train_years, test_years = fold_years(outer_fold)
        tau_cols = [f"tau_{m}" for m in model_dirs]
        cur["tau_mean_ensemble"] = cur[tau_cols].mean(axis=1)
        signs = np.sign(cur[tau_cols].to_numpy(dtype=float))
        n_pos = (signs > 0).sum(axis=1)
        n_neg = (signs < 0).sum(axis=1)
        cur["tau_sign_agree_ratio"] = np.maximum(n_pos, n_neg) / len(tau_cols)
        cur["tau_calibration_applied"] = True
        cur["outer_fold"] = outer_fold
        cur["train_years"] = train_years
        cur["test_years"] = test_years
        rows.append(cur)
    out = pd.concat(rows, ignore_index=True)
    return out.drop(columns=["__row_id"])


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = (
        rows.groupby(["outer_fold", "train_years", "test_years"], sort=False)
        .agg(
            n=("tau_mean_ensemble", "size"),
            tau_mean=("tau_mean_ensemble", "mean"),
            tau_sd=("tau_mean_ensemble", "std"),
            share_three_pref=("tau_mean_ensemble", lambda s: float((s > 0).mean())),
            share_two_pref=("tau_mean_ensemble", lambda s: float((s < 0).mean())),
        )
        .reset_index()
    )
    return out


def aggregate_surface(
    rows: pd.DataFrame,
    window: str,
    time_lo: int,
    time_hi: int,
    bin_width: int,
    min_n: int,
    score_lo: int,
    score_hi: int,
) -> pd.DataFrame:
    df = rows[
        rows["time_left_game"].between(time_lo, time_hi)
        & rows["score_diff"].between(score_lo, score_hi)
    ].copy()
    df["time_bin_left"] = (np.floor((df["time_left_game"].astype(float) - time_lo) / bin_width) * bin_width + time_lo).astype(int)
    df["time_bin_left"] = df["time_bin_left"].clip(lower=time_lo, upper=time_hi - bin_width if time_hi > time_lo else time_hi)
    df["time_bin_right"] = df["time_bin_left"] + bin_width
    df.loc[df["time_bin_right"] > time_hi, "time_bin_right"] = time_hi
    df["time_bin_mid"] = (df["time_bin_left"] + df["time_bin_right"]) / 2.0
    df["score_diff_cell"] = df["score_diff"].round().astype(int)
    tau_cols = ["tau_catboost", "tau_xgb", "tau_lgbm"]
    raw_cols = ["tau_catboost_raw", "tau_xgb_raw", "tau_lgbm_raw"]
    grp_cols = ["outer_fold", "train_years", "test_years", "time_bin_left", "time_bin_right", "time_bin_mid", "score_diff_cell"]
    agg = (
        df.groupby(grp_cols, sort=True)
        .agg(
            cate_value=("tau_mean_ensemble", "mean"),
            tau_mean_ensemble=("tau_mean_ensemble", "mean"),
            tau_catboost=("tau_catboost", "mean"),
            tau_xgb=("tau_xgb", "mean"),
            tau_lgbm=("tau_lgbm", "mean"),
            tau_catboost_raw=("tau_catboost_raw", "mean"),
            tau_xgb_raw=("tau_xgb_raw", "mean"),
            tau_lgbm_raw=("tau_lgbm_raw", "mean"),
            tau_std_across_models=("tau_mean_ensemble", "std"),
            tau_sign_agree_ratio=("tau_sign_agree_ratio", "mean"),
            n=("tau_mean_ensemble", "size"),
        )
        .reset_index()
    )
    agg["window"] = window
    agg["shown_in_figure"] = agg["n"] >= min_n
    return agg[
        [
            "window",
            "outer_fold",
            "train_years",
            "test_years",
            "time_bin_left",
            "time_bin_right",
            "time_bin_mid",
            "score_diff_cell",
            "cate_value",
            "tau_mean_ensemble",
            "tau_catboost",
            "tau_xgb",
            "tau_lgbm",
            "tau_catboost_raw",
            "tau_xgb_raw",
            "tau_lgbm_raw",
            "tau_std_across_models",
            "tau_sign_agree_ratio",
            "n",
            "shown_in_figure",
        ]
    ]


def plot_surface(surface: pd.DataFrame, out_path: Path, title: str, min_n: int) -> None:
    folds = list(surface["outer_fold"].drop_duplicates())
    fig, axes = plt.subplots(1, len(folds), figsize=(4.2 * len(folds), 4.2), sharey=True)
    if len(folds) == 1:
        axes = [axes]
    vals = surface.loc[surface["shown_in_figure"], "cate_value"].to_numpy(dtype=float)
    vmax = max(float(np.nanmax(np.abs(vals))) if vals.size else 0.01, 0.01)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    im = None
    for ax, fold in zip(axes, folds):
        cur = surface[surface["outer_fold"] == fold].copy()
        cur.loc[~cur["shown_in_figure"], "cate_value"] = np.nan
        pivot = cur.pivot(index="time_bin_mid", columns="score_diff_cell", values="cate_value").sort_index()
        arr = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            extent=[
                float(pivot.columns.min()) - 0.5,
                float(pivot.columns.max()) + 0.5,
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
        )
        test_years = str(cur["test_years"].iloc[0])
        ax.set_title(test_years)
        ax.set_xticks(np.arange(int(pivot.columns.min()), int(pivot.columns.max()) + 1, 5))
    axes[0].set_ylabel("Time remaining (seconds)")
    fig.supxlabel("Score differential (offense perspective)", y=0.02)
    if im is not None:
        fig.colorbar(im, ax=axes, label="Recalibrated CATE", shrink=0.82)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dirs = {
        "catboost": Path(args.catboost_dir),
        "xgb": Path(args.xgb_dir),
        "lgbm": Path(args.lgbm_dir),
    }
    rows = build_rows(model_dirs)
    rows.to_parquet(outdir / "outer_fold_ensemble_tau_test_rows.parquet", index=False)
    summarize_rows(rows).to_csv(outdir / "outer_fold_ensemble_cate_surface_summary.csv", index=False)

    surface_30_300 = aggregate_surface(rows, "30--300s", 30, 300, 15, 20, args.score_lo, args.score_hi)
    surface_0_30 = aggregate_surface(rows, "0--30s", 0, 30, 3, 10, args.score_lo, args.score_hi)
    surface_30_300.to_csv(outdir / "outer_fold_ensemble_cate_surface_30_300.csv", index=False)
    surface_0_30.to_csv(outdir / "outer_fold_ensemble_cate_surface_0_30.csv", index=False)

    readme = outdir / "README.md"
    readme.write_text(
        """# Walk-forward outer-fold ensemble CATE surfaces

Purpose: show what the nested walk-forward models learned in each held-out outer test period.

Construction:
- For each outer fold, read aligned `tau_test.parquet` rows from CatBoost, XGBoost, and LightGBM terminal-fix nested-WF runs.
- Apply each model/fold's outer-training BLP recalibration from `blp_metrics_oos_train.json`: `tau_cal = blp_alpha + blp_beta * tau_hat`.
- Average the three model-specific calibrated tau values row-wise to form `tau_mean_ensemble`.
- Aggregate held-out rows into `time_left_game x score_diff` cells.
- Figure values are percentage points of recalibrated CATE (`3P - 2P` fitted delta WP); positive means 3P-favoring, negative means 2P-favoring.

Files:
- `outer_fold_ensemble_tau_test_rows.parquet`: row-level held-out predictions with model-specific raw/calibrated tau and ensemble calibrated tau.
- `outer_fold_ensemble_cate_surface_30_300.csv`: 30-300s primary cell source data, 15-second bins, shown cells require n >= 20.
- `outer_fold_ensemble_cate_surface_0_30.csv`: 0-30s late-clock validity / separate-regime diagnostic cell source data, 3-second bins, shown cells require n >= 10.
- `outer_fold_ensemble_cate_surface_summary.csv`: fold-level summary of row-level calibrated ensemble tau.

Outer folds: train2000_2009_test2010_2012, train2000_2012_test2013_2015, train2000_2015_test2016_2018, train2000_2018_test2019_2021, train2000_2021_test2022_2024
""",
        encoding="utf-8",
    )
    print(f"[saved] {outdir}")


if __name__ == "__main__":
    main()
