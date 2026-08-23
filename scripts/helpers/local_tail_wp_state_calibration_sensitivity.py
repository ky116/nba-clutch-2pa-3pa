#!/usr/bin/env python3
"""Season-LOSO state-aware WP calibration for final-seconds sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def make_state_rows(shots: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for prefix, p_col in (("before", "wp_before"), ("next", "wp_next")):
        parts.append(
            pd.DataFrame(
                {
                    "season": shots["season"],
                    "GAME_ID": shots["GAME_ID"],
                    "time_left_game": shots[f"{prefix}_time_left_game"],
                    "score_diff": shots[f"{prefix}_score_diff"],
                    "home_possession": shots[f"{prefix}_home_possession"],
                    "start_type": shots[f"{prefix}_start_type"],
                    "p_raw": shots[p_col],
                    "final_home_win": shots["final_home_win"],
                    "terminal": shots["next_is_terminal"] if prefix == "next" else 0,
                }
            )
        )
    states = pd.concat(parts, ignore_index=True)
    numeric = [
        "season",
        "time_left_game",
        "score_diff",
        "home_possession",
        "p_raw",
        "final_home_win",
    ]
    for col in numeric:
        states[col] = pd.to_numeric(states[col], errors="coerce")
    states = states.dropna(subset=numeric)
    states = states[states["terminal"].fillna(0).eq(0)].copy()
    states["GAME_ID"] = states["GAME_ID"].astype(str)
    states["start_type"] = states["start_type"].astype(str)
    return states.drop_duplicates(
        ["GAME_ID", "time_left_game", "score_diff", "home_possession", "start_type", "p_raw"]
    )


def frame(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "logit_p": logit(df["p_raw"].to_numpy(), eps),
            "time_left_game": df["time_left_game"].round().astype(int).astype(str),
            "score_diff": df["score_diff"].round().astype(int).astype(str),
            "home_possession": df["home_possession"].round().astype(int).astype(str),
            "start_type": df["start_type"].astype(str),
        },
        index=df.index,
    )


def fit_model(train: pd.DataFrame, eps: float, c_value: float) -> Pipeline:
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["logit_p"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["time_left_game", "score_diff", "home_possession", "start_type"],
            ),
        ]
    )
    model = Pipeline(
        [
            ("transform", transform),
            ("logistic", LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000)),
        ]
    )
    model.fit(frame(train, eps), train["final_home_win"])
    return model


def predict(model: Pipeline, df: pd.DataFrame, eps: float) -> np.ndarray:
    return model.predict_proba(frame(df, eps))[:, 1]


def metrics(df: pd.DataFrame, p_col: str, label: str) -> dict:
    y = df["final_home_win"].to_numpy(dtype=int)
    p = np.clip(df[p_col].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
    return {
        "season": int(df["season"].iloc[0]),
        "prediction": label,
        "n": len(df),
        "prevalence": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def apply_frame(shots: pd.DataFrame, apply: pd.Series, prefix: str) -> pd.DataFrame:
    source = "before" if prefix == "before" else "next"
    return pd.DataFrame(
        {
            "p_raw": pd.to_numeric(shots.loc[apply, f"wp_{source}"], errors="coerce"),
            "time_left_game": pd.to_numeric(
                shots.loc[apply, f"{prefix}_time_left_game"], errors="coerce"
            ),
            "score_diff": pd.to_numeric(
                shots.loc[apply, f"{prefix}_score_diff"], errors="coerce"
            ),
            "home_possession": pd.to_numeric(
                shots.loc[apply, f"{prefix}_home_possession"], errors="coerce"
            ),
            "start_type": shots.loc[apply, f"{prefix}_start_type"].astype(str),
        },
        index=shots.index[apply],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--time-max", type=float, default=5.0)
    p.add_argument("--score-abs-max", type=float, default=10.0)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--c-value", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shots = pd.read_csv(args.input, low_memory=False)
    states = make_state_rows(shots)
    local = states[
        states["time_left_game"].le(args.time_max)
        & states["score_diff"].abs().le(args.score_abs_max)
    ].copy()
    shots["wp_before_tail_state_cal"] = pd.to_numeric(shots["wp_before"], errors="coerce")
    shots["wp_next_tail_state_cal"] = pd.to_numeric(shots["wp_next"], errors="coerce")
    metric_rows = []
    fit_rows = []

    seasons = sorted(pd.to_numeric(shots["season"], errors="coerce").dropna().astype(int).unique())
    for season in seasons:
        train = local[local["season"].ne(season)]
        test = local[local["season"].eq(season)].copy()
        model = fit_model(train, args.eps, args.c_value)
        test["p_cal"] = predict(model, test, args.eps)
        metric_rows += [metrics(test, "p_raw", "raw"), metrics(test, "p_cal", "tail_state_logit")]
        fit_rows.append(
            {
                "season": season,
                "intercept": float(model.named_steps["logistic"].intercept_[0]),
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        season_mask = pd.to_numeric(shots["season"], errors="coerce").eq(season)
        for prefix, out_col in (
            ("before", "wp_before_tail_state_cal"),
            ("next", "wp_next_tail_state_cal"),
        ):
            apply = (
                season_mask
                & pd.to_numeric(
                    shots[f"{prefix}_time_left_game"], errors="coerce"
                ).le(args.time_max)
                & pd.to_numeric(shots[f"{prefix}_score_diff"], errors="coerce")
                .abs()
                .le(args.score_abs_max)
            )
            if prefix == "next":
                apply &= pd.to_numeric(
                    shots["next_is_terminal"], errors="coerce"
                ).fillna(0).eq(0)
            shots.loc[apply, out_col] = predict(
                model, apply_frame(shots, apply, prefix), args.eps
            )

    shots["delta_wp_tail_state_cal"] = (
        shots["wp_next_tail_state_cal"] - shots["wp_before_tail_state_cal"]
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        outdir / "tail_state_logit_oof_metrics_by_season.csv", index=False
    )
    pd.DataFrame(fit_rows).to_csv(
        outdir / "tail_state_logit_oof_fits.csv", index=False
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
