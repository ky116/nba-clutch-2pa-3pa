#!/usr/bin/env python3
"""
Attach leak-free pregame Elo features to wp_states CSV.

Minimal protocol:
- Pregame expected home win prob:
    p = 1 / (1 + 10^(-(R_home + H - R_away)/400))
- Update after each game:
    R_home' = R_home + K * (win - p)
    R_away' = R_away - K * (win - p)
- Season carryover (recommended mean reversion):
    R_new = mean + carry * (R_old - mean)

Output columns added:
- elo_home_pregame
- elo_away_pregame
- elo_diff_pregame
- elo_exp_home_pregame
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from build_wp_features import (
    build_game_team_map,
    load_games,
    load_pbp,
    load_possessions,
    normalize_game_id,
)


@dataclass
class EloConfig:
    k: float = 20.0
    h: float = 0.0
    mean_rating: float = 1500.0
    init_rating: float = 1500.0
    carry: float = 0.75
    invert_home_win: bool = False


def _build_game_meta(start_season: int, end_season: int, seasontype: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for season in range(start_season, end_season + 1):
        pbp = load_pbp(season, seasontype)
        games = load_games(season, seasontype)
        gmap = build_game_team_map(pbp, games)
        if gmap.empty:
            continue

        gmap = gmap.copy()
        gmap["game_id"] = normalize_game_id(gmap["game_id"])
        gmap["home_team_id"] = pd.to_numeric(gmap.get("home_team_id"), errors="coerce")
        gmap["away_team_id"] = pd.to_numeric(gmap.get("away_team_id"), errors="coerce")

        poss = load_possessions(season, seasontype)
        if poss.empty or "GAMEID" not in poss.columns:
            date_map = pd.DataFrame(columns=["game_id", "game_date"])  # fallback
        else:
            tmp = poss[["GAMEID", "GAMEDATE"]].copy()
            tmp = tmp.rename(columns={"GAMEID": "game_id", "GAMEDATE": "game_date"})
            tmp["game_id"] = normalize_game_id(tmp["game_id"])
            tmp["game_date"] = pd.to_datetime(tmp["game_date"], errors="coerce")
            date_map = tmp.groupby("game_id", as_index=False)["game_date"].min()

        g = gmap.merge(date_map, on="game_id", how="left")
        g["season"] = int(season)
        rows.append(g[["season", "game_id", "home_team_id", "away_team_id", "game_date"]])

    if not rows:
        return pd.DataFrame(columns=["season", "game_id", "home_team_id", "away_team_id", "game_date"])

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["season", "game_id"], keep="first")
    return out


def _compute_elo_features(game_df: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["season", "game_date", "game_id"], kind="mergesort").reset_index(drop=True)

    ratings: Dict[int, float] = {}
    cur_season = None

    elo_home = np.full(len(df), np.nan, dtype=float)
    elo_away = np.full(len(df), np.nan, dtype=float)
    elo_p = np.full(len(df), np.nan, dtype=float)

    for i, row in enumerate(df.itertuples(index=False)):
        season = int(row.season)
        if cur_season is None:
            cur_season = season
        elif season != cur_season:
            for tid, r in list(ratings.items()):
                ratings[tid] = cfg.mean_rating + cfg.carry * (r - cfg.mean_rating)
            cur_season = season

        home = int(row.home_team_id)
        away = int(row.away_team_id)
        win_obs = float(row.final_home_win)
        win = 1.0 - win_obs if cfg.invert_home_win else win_obs

        r_home = ratings.get(home, cfg.init_rating)
        r_away = ratings.get(away, cfg.init_rating)

        p = 1.0 / (1.0 + 10.0 ** (-((r_home + cfg.h) - r_away) / 400.0))

        elo_home[i] = r_home
        elo_away[i] = r_away
        elo_p[i] = p

        delta = cfg.k * (win - p)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    df["elo_home_pregame"] = elo_home
    df["elo_away_pregame"] = elo_away
    df["elo_diff_pregame"] = df["elo_home_pregame"] - df["elo_away_pregame"]
    df["elo_exp_home_pregame"] = elo_p
    return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attach pregame Elo features to wp_states.")
    p.add_argument("--in-path", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--seasontype", choices=["rs"], required=True)
    p.add_argument("--start-season", type=int, default=2000)
    p.add_argument("--end-season", type=int, default=2024)
    p.add_argument("--k", type=float, default=20.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--mean-rating", type=float, default=1500.0)
    p.add_argument("--init-rating", type=float, default=1500.0)
    p.add_argument("--carry", type=float, default=0.75)
    p.add_argument("--invert-home-win", type=int, default=0, choices=[0, 1])
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = EloConfig(
        k=float(args.k),
        h=float(args.h),
        mean_rating=float(args.mean_rating),
        init_rating=float(args.init_rating),
        carry=float(args.carry),
        invert_home_win=bool(int(args.invert_home_win)),
    )

    print(f"[info] load {args.in_path}")
    df = pd.read_csv(args.in_path)

    need = {"season", "game_id", "final_home_win"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in input: {sorted(miss)}")

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["game_id"] = normalize_game_id(df["game_id"])
    df["final_home_win"] = pd.to_numeric(df["final_home_win"], errors="coerce")

    game_outcome = df[["season", "game_id", "final_home_win"]].drop_duplicates(subset=["season", "game_id"], keep="first")

    print(f"[info] build game metadata for {args.seasontype} {args.start_season}-{args.end_season}")
    game_meta = _build_game_meta(args.start_season, args.end_season, args.seasontype)
    if game_meta.empty:
        raise RuntimeError("Failed to build game metadata (home/away team ids).")

    game_level = game_outcome.merge(game_meta, on=["season", "game_id"], how="left")
    before = len(game_level)
    game_level = game_level.dropna(subset=["home_team_id", "away_team_id", "final_home_win"]).copy()
    dropped = before - len(game_level)
    if dropped > 0:
        print(f"[warn] dropped {dropped} games with missing team ids/outcome")

    elo_games = _compute_elo_features(game_level, cfg)

    use_cols = [
        "season",
        "game_id",
        "elo_home_pregame",
        "elo_away_pregame",
        "elo_diff_pregame",
        "elo_exp_home_pregame",
    ]
    df_out = df.merge(elo_games[use_cols], on=["season", "game_id"], how="left")

    for c in ["elo_home_pregame", "elo_away_pregame", "elo_diff_pregame", "elo_exp_home_pregame"]:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")

    miss_rows = int(df_out["elo_home_pregame"].isna().sum())
    if miss_rows > 0:
        print(f"[warn] rows missing elo_home_pregame after merge: {miss_rows}")

    df_out["elo_k"] = cfg.k
    df_out["elo_h"] = cfg.h
    df_out["elo_carry"] = cfg.carry

    print(f"[info] write {args.out_path}")
    df_out.to_csv(args.out_path, index=False, compression="gzip")
    print("[done]")


if __name__ == "__main__":
    main()
