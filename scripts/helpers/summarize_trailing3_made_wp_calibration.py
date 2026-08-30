#!/usr/bin/env python
"""Summarize made-shot next-state WP calibration when trailing by 3."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["GAME_ID_norm"] = (
        out["GAME_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    )
    out["GAME_EVENT_ID_norm"] = pd.to_numeric(
        out["GAME_EVENT_ID"], errors="coerce"
    ).astype("Int64")
    return out


def load_relevant_rows(
    path: Path,
    max_time: float,
    start_season: int | None = None,
    end_season: int | None = None,
) -> pd.DataFrame:
    usecols = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "shot_made",
        "next_is_terminal",
        "before_score_diff",
        "before_time_left_game",
        "before_home_possession",
        "wp_next",
        "final_home_win",
        "season",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        season = pd.to_numeric(chunk["season"], errors="coerce")
        if start_season is not None:
            chunk = chunk[season.ge(start_season)].copy()
            season = pd.to_numeric(chunk["season"], errors="coerce")
        if end_season is not None:
            chunk = chunk[season.le(end_season)].copy()
        if chunk.empty:
            continue
        time = pd.to_numeric(chunk["before_time_left_game"], errors="coerce")
        home_poss = pd.to_numeric(chunk["before_home_possession"], errors="coerce")
        home_margin = pd.to_numeric(chunk["before_score_diff"], errors="coerce")
        off_margin = pd.Series(
            np.where(home_poss.eq(1), home_margin, np.where(home_poss.eq(0), -home_margin, np.nan)),
            index=chunk.index,
        )
        made = pd.to_numeric(chunk["shot_made"], errors="coerce").eq(1)
        terminal = pd.to_numeric(chunk["next_is_terminal"], errors="coerce").eq(1)
        mask = time.ge(0) & time.le(max_time) & off_margin.eq(-3) & made & ~terminal
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        hp = home_poss.loc[mask]
        home_wp = pd.to_numeric(sub["wp_next"], errors="coerce")
        home_win = pd.to_numeric(sub["final_home_win"], errors="coerce")
        sub["off_wp_next"] = np.where(hp.eq(1), home_wp, np.where(hp.eq(0), 1 - home_wp, np.nan))
        sub["offense_win"] = np.where(hp.eq(1), home_win, np.where(hp.eq(0), 1 - home_win, np.nan))
        parts.append(normalize_ids(sub))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach_shot_type(rows: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    maps = []
    for season in sorted(rows["season"].dropna().astype(int).unique()):
        path = raw_dir / f"shotdetail_{season}.csv"
        if not path.exists():
            continue
        ids = rows.loc[
            rows["season"].eq(season), ["GAME_ID_norm", "GAME_EVENT_ID_norm"]
        ].drop_duplicates()
        shots = pd.read_csv(
            path,
            usecols=["GAME_ID", "GAME_EVENT_ID", "SHOT_TYPE", "SHOT_ATTEMPTED_FLAG"],
            low_memory=False,
        )
        shots = shots[pd.to_numeric(shots["SHOT_ATTEMPTED_FLAG"], errors="coerce").eq(1)].copy()
        shots = normalize_ids(shots)
        shots = shots.merge(ids, on=["GAME_ID_norm", "GAME_EVENT_ID_norm"], how="inner")
        maps.append(shots[["GAME_ID_norm", "GAME_EVENT_ID_norm", "SHOT_TYPE"]].drop_duplicates())
    if not maps:
        raise SystemExit("No shotdetail rows matched the calibration sample.")
    shotmap = pd.concat(maps, ignore_index=True).drop_duplicates(
        ["GAME_ID_norm", "GAME_EVENT_ID_norm"]
    )
    out = rows.merge(shotmap, on=["GAME_ID_norm", "GAME_EVENT_ID_norm"], how="left")
    shot_text = out["SHOT_TYPE"].astype(str)
    out["shot_type"] = np.where(
        shot_text.str.contains("3PT", na=False),
        "3PA",
        np.where(shot_text.str.contains("2PT", na=False), "2PA", np.nan),
    )
    return out


def summarize(rows: pd.DataFrame, bins: list[tuple[float, float]]) -> pd.DataFrame:
    rows = rows[rows["shot_type"].isin(["2PA", "3PA"])].copy()
    results = []
    for lo, hi in bins:
        cell = rows[
            pd.to_numeric(rows["before_time_left_game"], errors="coerce").ge(lo)
            & pd.to_numeric(rows["before_time_left_game"], errors="coerce").le(hi)
        ]
        for shot_type in ["2PA", "3PA"]:
            g = cell[cell["shot_type"].eq(shot_type)]
            fitted = pd.to_numeric(g["off_wp_next"], errors="coerce")
            empirical = pd.to_numeric(g["offense_win"], errors="coerce")
            results.append(
                {
                    "time_bin": f"{int(lo)}-{int(hi)}s",
                    "shot_type": shot_type,
                    "n": int(len(g)),
                    "n_games": int(g["GAME_ID_norm"].nunique()),
                    "fitted_next_state_wp": float(fitted.mean()) if len(g) else np.nan,
                    "empirical_win_rate": float(empirical.mean()) if len(g) else np.nan,
                    "wp_minus_empirical": float(fitted.mean() - empirical.mean()) if len(g) else np.nan,
                }
            )
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-wp",
        type=Path,
        default=Path("data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz"),
    )
    parser.add_argument("--shotdetail-dir", type=Path, default=Path("data/nba_raw"))
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/wp_calibration_trailing3_made_shots_by_time.csv"),
    )
    args = parser.parse_args()

    bins = [(0, 15), (15, 30), (30, 45), (45, 60)]
    rows = load_relevant_rows(
        args.with_wp,
        max_time=max(hi for _, hi in bins),
        start_season=args.start_season,
        end_season=args.end_season,
    )
    rows = attach_shot_type(rows, args.shotdetail_dir)
    result = summarize(rows, bins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
