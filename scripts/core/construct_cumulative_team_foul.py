#!/usr/bin/env python3
"""
construct_cumulative_team_foul.py

Build cumulative team foul counts by event from nbastats play-by-play.

Inputs:
  - data/nba_raw/nbastats_{season}.csv (or nbastats_{seasontype}_{season}.csv)
Output:
  - data/analysis/cumulative_team_fouls_{start}_{end}_{seasontype}.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "nba_raw"
OUT_DIR = BASE_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_game_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def find_nbastats_file(season: int, seasontype: str) -> Path | None:
    candidates = [
        RAW_DIR / f"nbastats_{seasontype}_{season}.csv",
        RAW_DIR / f"nbastats_{season}_{seasontype}.csv",
        RAW_DIR / f"nbastats_{season}.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def infer_home_visitor(df: pd.DataFrame) -> pd.DataFrame:
    required = {"GAME_ID", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "PLAYER1_TEAM_ABBREVIATION"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["GAME_ID", "home_team", "visitor_team"])

    home_events = df[df["HOMEDESCRIPTION"].notna() & df["PLAYER1_TEAM_ABBREVIATION"].notna()].copy()
    visitor_events = df[df["VISITORDESCRIPTION"].notna() & df["PLAYER1_TEAM_ABBREVIATION"].notna()].copy()

    if home_events.empty or visitor_events.empty:
        return pd.DataFrame(columns=["GAME_ID", "home_team", "visitor_team"])

    home_map = home_events.groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"].agg(
        lambda x: x.value_counts().index[0]
    )
    visitor_map = visitor_events.groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"].agg(
        lambda x: x.value_counts().index[0]
    )

    common = sorted(set(home_map.index) & set(visitor_map.index))
    rows = []
    for gid in common:
        h = home_map[gid]
        v = visitor_map[gid]
        if pd.notna(h) and pd.notna(v) and h != v:
            rows.append({"GAME_ID": str(gid).strip().zfill(10), "home_team": h, "visitor_team": v})

    if not rows:
        return pd.DataFrame(columns=["GAME_ID", "home_team", "visitor_team"])
    return pd.DataFrame(rows)


def build_cumulative_for_season(season: int, seasontype: str) -> pd.DataFrame:
    path = find_nbastats_file(season, seasontype)
    if path is None:
        print(f"[WARN] nbastats CSV not found for season={season}, seasontype={seasontype}")
        return pd.DataFrame()

    usecols = [
        "GAME_ID",
        "EVENTNUM",
        "EVENTMSGTYPE",
        "PERIOD",
        "PCTIMESTRING",
        "HOMEDESCRIPTION",
        "VISITORDESCRIPTION",
        "PLAYER1_TEAM_ABBREVIATION",
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    df["GAME_ID"] = normalize_game_id(df["GAME_ID"])
    df["EVENTNUM"] = pd.to_numeric(df["EVENTNUM"], errors="coerce")
    df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce")
    df = df.dropna(subset=["GAME_ID", "EVENTNUM", "PERIOD"]).copy()

    team_map = infer_home_visitor(df)
    if team_map.empty:
        print(f"[WARN] Home/visitor inference failed for season={season}")
        return pd.DataFrame()

    df = df.merge(team_map, on="GAME_ID", how="left")
    df = df.dropna(subset=["home_team", "visitor_team"]).copy()

    df["GAME_EVENT_ID"] = df["EVENTNUM"].astype("Int64")
    df = df.sort_values(["GAME_ID", "EVENTNUM"]).reset_index(drop=True)

    is_foul = df["EVENTMSGTYPE"] == 6
    team_abbrev = df["PLAYER1_TEAM_ABBREVIATION"]
    df["home_foul"] = (is_foul & (team_abbrev == df["home_team"])).astype(int)
    df["visitor_foul"] = (is_foul & (team_abbrev == df["visitor_team"])).astype(int)

    df["home_fouls_game"] = df.groupby("GAME_ID", sort=False)["home_foul"].cumsum()
    df["visitor_fouls_game"] = df.groupby("GAME_ID", sort=False)["visitor_foul"].cumsum()
    df["home_fouls_period"] = df.groupby(["GAME_ID", "PERIOD"], sort=False)["home_foul"].cumsum()
    df["visitor_fouls_period"] = df.groupby(["GAME_ID", "PERIOD"], sort=False)["visitor_foul"].cumsum()

    keep_cols = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "PERIOD",
        "PCTIMESTRING",
        "home_team",
        "visitor_team",
        "home_fouls_game",
        "visitor_fouls_game",
        "home_fouls_period",
        "visitor_fouls_period",
    ]
    return df[keep_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct cumulative team foul counts from nbastats.")
    parser.add_argument("--start-season", type=int, default=2000)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--seasontype", type=str, default="rs", choices=["rs"])
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path (default: data/analysis/cumulative_team_fouls_{start}_{end}_{seasontype}.parquet)",
    )
    args = parser.parse_args()

    seasons = list(range(args.start_season, args.end_season + 1))
    parts = []
    for season in seasons:
        print(f"Processing season {season}...")
        part = build_cumulative_for_season(season, args.seasontype)
        if not part.empty:
            parts.append(part)

    if not parts:
        print("[WARN] No data produced. Exiting.")
        return

    out_path = (
        Path(args.output)
        if args.output is not None
        else OUT_DIR / f"cumulative_team_fouls_{args.start_season}_{args.end_season}_{args.seasontype}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.concat(parts, ignore_index=True)
    df_out.to_parquet(out_path, index=False)
    print(f"✓ Saved: {out_path} ({len(df_out):,} rows)")


if __name__ == "__main__":
    main()
