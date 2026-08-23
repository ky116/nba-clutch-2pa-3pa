#!/usr/bin/env python3
"""Season-LOSO local WP calibration for final-seconds sensitivity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def make_state_rows(shots: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for prefix, p_col in (("before", "wp_before"), ("next", "wp_next")):
        x = pd.DataFrame(
            {
                "season": shots["season"],
                "GAME_ID": shots["GAME_ID"],
                "state_source": prefix,
                "time_left_game": shots[f"{prefix}_time_left_game"],
                "score_diff": shots[f"{prefix}_score_diff"],
                "home_possession": shots[f"{prefix}_home_possession"],
                "start_type": shots[f"{prefix}_start_type"],
                "p_raw": shots[p_col],
                "final_home_win": shots["final_home_win"],
                "terminal": shots["next_is_terminal"] if prefix == "next" else 0,
            }
        )
        parts.append(x)
    states = pd.concat(parts, ignore_index=True)
    for c in ("season", "time_left_game", "score_diff", "home_possession", "p_raw", "final_home_win"):
        states[c] = pd.to_numeric(states[c], errors="coerce")
    states = states.dropna(
        subset=["season", "time_left_game", "score_diff", "home_possession", "p_raw", "final_home_win"]
    )
    states = states[states["terminal"].fillna(0).eq(0)].copy()
    states["GAME_ID"] = states["GAME_ID"].astype(str)
    states["start_type"] = states["start_type"].astype(str)
    states = states.drop_duplicates(
        ["GAME_ID", "time_left_game", "score_diff", "home_possession", "start_type", "p_raw"]
    )
    return states


def local_mask(df: pd.DataFrame, time_max: float, score_abs_max: float) -> pd.Series:
    return df["time_left_game"].le(time_max) & df["score_diff"].abs().le(score_abs_max)


def fit_platt(train: pd.DataFrame, eps: float, c_value: float) -> LogisticRegression:
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000)
    model.fit(logit(train["p_raw"].to_numpy(), eps).reshape(-1, 1), train["final_home_win"])
    return model


def apply_model(model: LogisticRegression, p: pd.Series, eps: float) -> np.ndarray:
    return model.predict_proba(logit(p.to_numpy(), eps).reshape(-1, 1))[:, 1]


def state_metrics(df: pd.DataFrame, p_col: str, label: str) -> dict:
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
    mask = local_mask(states, args.time_max, args.score_abs_max)
    local_states = states[mask].copy()

    shots["wp_before_tail_cal"] = pd.to_numeric(shots["wp_before"], errors="coerce")
    shots["wp_next_tail_cal"] = pd.to_numeric(shots["wp_next"], errors="coerce")
    metric_rows = []
    coef_rows = []

    seasons = sorted(pd.to_numeric(shots["season"], errors="coerce").dropna().astype(int).unique())
    for season in seasons:
        train = local_states[local_states["season"].ne(season)]
        test = local_states[local_states["season"].eq(season)].copy()
        if len(train) < 100 or test.empty or train["final_home_win"].nunique() < 2:
            continue
        model = fit_platt(train, args.eps, args.c_value)
        test["p_cal"] = apply_model(model, test["p_raw"], args.eps)
        metric_rows.extend(
            [state_metrics(test, "p_raw", "raw"), state_metrics(test, "p_cal", "tail_platt")]
        )
        coef_rows.append(
            {
                "season": season,
                "intercept": float(model.intercept_[0]),
                "slope": float(model.coef_[0, 0]),
                "n_train": len(train),
                "n_test": len(test),
            }
        )

        shot_season = pd.to_numeric(shots["season"], errors="coerce").eq(season)
        for prefix, source, out_col in (
            ("before", "before", "wp_before_tail_cal"),
            ("next", "next", "wp_next_tail_cal"),
        ):
            apply = (
                shot_season
                & pd.to_numeric(shots[f"{prefix}_time_left_game"], errors="coerce").le(args.time_max)
                & pd.to_numeric(shots[f"{prefix}_score_diff"], errors="coerce").abs().le(args.score_abs_max)
            )
            if prefix == "next":
                apply &= pd.to_numeric(shots["next_is_terminal"], errors="coerce").fillna(0).eq(0)
            shots.loc[apply, out_col] = apply_model(
                model, pd.to_numeric(shots.loc[apply, f"wp_{source}"], errors="coerce"), args.eps
            )

    shots["delta_wp_tail_cal"] = shots["wp_next_tail_cal"] - shots["wp_before_tail_cal"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(outdir / "tail_platt_oof_metrics_by_season.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(outdir / "tail_platt_oof_coefficients.csv", index=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {outdir / 'tail_platt_oof_metrics_by_season.csv'}")
    print(f"Wrote {outdir / 'tail_platt_oof_coefficients.csv'}")


if __name__ == "__main__":
    main()
