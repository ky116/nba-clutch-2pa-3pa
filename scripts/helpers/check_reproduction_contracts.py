#!/usr/bin/env python3
"""Check reproduction-stage contracts required by downstream workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "scripts" / "core"
sys.path.insert(0, str(CORE_DIR))

from treatment_utils import apply_treatment_scheme  # noqa: E402


SEASONS = list(range(2000, 2025))
RAW_KINDS = ("nbastats", "pbpstats", "shotdetail")
WF_FOLDS = (
    "train2000_2009_test2010_2012",
    "train2000_2012_test2013_2015",
    "train2000_2015_test2016_2018",
    "train2000_2018_test2019_2021",
    "train2000_2021_test2022_2024",
)
MODEL_DIRS = {
    "catboost": ROOT / "results" / "nested_wf_catboost_gpu",
    "xgb": ROOT / "results" / "nested_wf_xgb",
    "lgbm": ROOT / "results" / "nested_wf_lgbm",
}
FULL_DIRS = {
    "catboost": ROOT / "results" / "full_data_catboost_state_fixed_loso",
    "xgb": ROOT / "results" / "full_data_xgb_state_fixed_loso",
    "lgbm": ROOT / "results" / "full_data_lgbm_state_fixed_loso",
}


class ContractError(RuntimeError):
    pass


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ContractError(f"missing file: {rel(path)}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise ContractError(f"missing directory: {rel(path)}")


def read_columns(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    require_file(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, usecols=columns)
    raise ContractError(f"unsupported table format: {rel(path)}")


def check_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(f"{label} missing columns: {missing}")


def nonnull_count(df: pd.DataFrame, col: str) -> int:
    return int(pd.to_numeric(df[col], errors="coerce").notna().sum())


def check_raw() -> None:
    missing: list[str] = []
    for season in SEASONS:
        for kind in RAW_KINDS:
            path = ROOT / "data" / "nba_raw" / f"{kind}_{season}.csv"
            if not path.is_file():
                missing.append(rel(path))
    if missing:
        raise ContractError("raw NBA CSV inputs are incomplete:\n  " + "\n  ".join(missing[:30]))
    print("[ok] raw: nbastats/pbpstats/shotdetail CSVs exist for 2000-2024")


def check_core() -> None:
    required = {"game_id", "season", "elo_diff_pregame_k20"}
    missing_files: list[str] = []
    for season in SEASONS:
        path = ROOT / "data" / "processed" / f"games_{season}_rs.parquet"
        if not path.is_file():
            missing_files.append(rel(path))
            continue
        cols = pd.read_parquet(path, columns=None).columns
        missing = sorted(required - set(cols))
        if missing:
            raise ContractError(f"{rel(path)} missing downstream Elo/game columns: {missing}")
    if missing_files:
        raise ContractError("processed game files are incomplete:\n  " + "\n  ".join(missing_files[:30]))
    print("[ok] core: processed game files include pregame Elo for all seasons")


def check_wp() -> None:
    paths = [
        ROOT / "data" / "wp" / "wp_states_2000_2024_rs.csv.gz",
        ROOT / "data" / "wp" / "shot_decision_states_2000_2024_rs.csv.gz",
    ]
    for path in paths:
        require_file(path)
    shot = pd.read_csv(paths[1], nrows=5)
    check_columns(
        shot,
        [
            "GAME_ID",
            "GAME_EVENT_ID",
            "season",
            "before_time_left_game",
            "before_score_diff",
            "before_home_possession",
        ],
        rel(paths[1]),
    )
    print("[ok] wp: combined WP and shot-decision state files exist with panel keys")


def check_context() -> None:
    team_path = ROOT / "data" / "analysis" / "team_shot_stats_2000_2024.parquet"
    foul_path = ROOT / "data" / "analysis" / "cumulative_team_fouls_2000_2024_rs.parquet"
    team = read_columns(team_path)
    foul = read_columns(foul_path)
    check_columns(team, ["GAME_ID", "GAME_EVENT_ID"], rel(team_path))
    eb_cols = [c for c in team.columns if c.endswith("_eb")]
    if not eb_cols:
        raise ContractError(f"{rel(team_path)} has no empirical-Bayes team feature columns")
    check_columns(
        foul,
        ["GAME_ID", "GAME_EVENT_ID", "home_fouls_period", "visitor_fouls_period"],
        rel(foul_path),
    )
    print(f"[ok] context: team EB features={len(eb_cols)} and cumulative foul inputs exist")


def check_panel() -> None:
    panel_path = ROOT / "data" / "analysis" / "shotchoice_panel_clutch_rs.parquet"
    required = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "season",
        "period",
        "shot_zone_choice",
        "delta_wp",
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
    df = read_columns(panel_path)
    check_columns(df, required, rel(panel_path))
    if len(df) == 0:
        raise ContractError(f"{rel(panel_path)} has zero rows")
    for col in ("delta_wp", "time_left_game", "score_diff", "elo_diff"):
        n = nonnull_count(df, col)
        if n == 0:
            raise ContractError(f"{rel(panel_path)} column {col} is entirely missing")
    for col in ("own_fouls_period", "opp_fouls_period", "timeouts_left_us", "timeouts_left_them"):
        n = nonnull_count(df, col)
        if n == 0:
            raise ContractError(f"{rel(panel_path)} column {col} is entirely missing")
    eb_cols = [
        c for c in df.columns
        if (c.startswith("own_") or c.startswith("opp_")) and c.endswith("_eb")
    ]
    if not eb_cols:
        raise ContractError(f"{rel(panel_path)} has no own_/opp_ empirical-Bayes team features")
    mapped = apply_treatment_scheme(
        df[["shot_zone_choice"]].copy(),
        treat_col="shot_zone_choice",
        scheme="binary",
        out_col="shot_zone_choice",
        drop_unknown=True,
    )
    levels = set(mapped["shot_zone_choice"].astype(str).unique())
    if not {"two-point", "three-point"}.issubset(levels):
        raise ContractError(f"binary treatment mapping missing two-point/three-point levels: {sorted(levels)}")
    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).unique().tolist())
    if not seasons:
        raise ContractError("panel has no valid season values")
    if seasons[0] > 2000 or seasons[-1] < 2024:
        raise ContractError(f"panel season coverage is unexpected: {seasons[0]}-{seasons[-1]}")
    print(
        "[ok] panel: clutch panel satisfies WF/full-data feature contract "
        f"(rows={len(df):,}, EB features={len(eb_cols)})"
    )


def check_wf() -> None:
    for model, root in MODEL_DIRS.items():
        require_dir(root)
        for fold in WF_FOLDS:
            fdir = root / fold
            require_dir(fdir)
            tau = fdir / "tau_test.parquet"
            meta = fdir / "meta.json"
            blp = fdir / "blp_metrics_oos_train.json"
            for path in (tau, meta, blp):
                require_file(path)
            tau_cols = pd.read_parquet(tau, columns=None).columns
            check_columns(pd.DataFrame(columns=tau_cols), ["GAME_ID", "GAME_EVENT_ID", "season", "tau_hat"], rel(tau))
            obj = json.loads(meta.read_text(encoding="utf-8"))
            missing = [
                k for k in ("nuisance_best_prop", "nuisance_best_outcome", "tau_best_params")
                if not isinstance(obj.get(k), dict)
            ]
            if missing:
                raise ContractError(f"{rel(meta)} missing fixed-hparam keys: {missing}")
    print("[ok] wf: all model fold artifacts exist for surface rebuild and fixed hparam export")


def check_full() -> None:
    for model, root in FULL_DIRS.items():
        require_dir(root)
        for name in (
            "nuisance_oos_train.parquet",
            "tau_oos_train.parquet",
            "full_data_tau_oos_train_calibrated.parquet",
            "tau_full_train.parquet",
            "tau_model.joblib",
            "full_data_t30_300_tau_surface_three-point_vs_two-point.parquet",
            "full_data_t0_30_tau_surface_three-point_vs_two-point.parquet",
        ):
            require_file(root / name)
    ensemble = ROOT / "results" / "full_data_ensemble_state_fixed_loso"
    require_dir(ensemble)
    for name in (
        "nuisance_oos_train.parquet",
        "tau_oos_train.parquet",
        "ensemble_tau_metadata.json",
        "full_data_t30_300_cate_surface_equal_weight.csv",
        "full_data_t0_30_cate_surface_equal_weight.csv",
    ):
        require_file(ensemble / name)
    print("[ok] full: learner artifacts and equal-weight ensemble source data exist")


def check_figures() -> None:
    for path in (
        ROOT / "results" / "wf_cate_surfaces" / "outer_fold_ensemble_tau_test_rows.parquet",
        ROOT / "results" / "wf_cate_surfaces" / "outer_fold_ensemble_cate_surface_30_300.csv",
        ROOT / "results" / "figure_source_data" / "figure2_full_data_t30_300_cate_surface_source_data.csv",
        ROOT / "results" / "figure_source_data" / "figures1_cate_surface_0_30s_masked_n50_source_data.csv",
        ROOT / "figures" / "figure2_full_data_t30_300_cate_surface.png",
        ROOT / "figures" / "figure3_outer_fold_t30_300_cate_surface.png",
        ROOT / "figures" / "figures1_cate_surface_0_30s_masked_n50.png",
        ROOT / "results" / "wp_calibration" / "model_dependence_sensitivity" / "wp_model_dependence_sensitivity_surface.csv",
        ROOT / "results" / "wp_calibration" / "model_dependence_sensitivity" / "wp_model_dependence_sensitivity_summary.csv",
        ROOT / "results" / "wp_calibration" / "model_dependence_sensitivity" / "wp_model_dependence_sensitivity_by_score.csv",
        ROOT / "results" / "wp_calibration" / "model_dependence_sensitivity" / "wp_model_dependence_sensitivity_by_time_band.csv",
        ROOT / "results" / "wp_calibration" / "model_dependence_sensitivity" / "wp_model_dependence_sensitivity_extreme_cells.csv",
        ROOT / "results" / "wp_calibration" / "model_dependence_sensitivity" / "wp_model_dependence_sensitivity_late_240_300_extreme_cells.csv",
    ):
        require_file(path)
    print("[ok] figures: manuscript PNGs, source-data tables, and WP model-dependence sensitivity outputs exist")


CHECKS = {
    "raw": check_raw,
    "core": check_core,
    "wp": check_wp,
    "context": check_context,
    "panel": check_panel,
    "wf": check_wf,
    "full": check_full,
    "figures": check_figures,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        action="append",
        choices=list(CHECKS),
        help="Stage contract to check. Repeatable. Defaults to all stages.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stages = args.stage or list(CHECKS)
    for stage in stages:
        CHECKS[stage]()


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
