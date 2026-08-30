#!/usr/bin/env python3
"""
build_shotchoice_panel_from_wp.py

fit_wp_and_score_shots.Rの出力（shot_decision_states_*_with_wp.csv.gz）から
DML分析用のショットチョイスパネルデータを作成する。

入力:
  - data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz
  - data/nba_raw/shotdetail_{season}.csv

出力:
  - data/analysis/shotchoice_panel_strict_clutch_rs.parquet
  - data/analysis/shotchoice_panel_clutch_rs.parquet
  - data/analysis/shotchoice_panel_strict_clutch_rs.parquet

処理内容:
  1. clutchフラグの計算
  2. shotdetail を結合してショットゾーン情報を追加
  3. チーム情報の追加 (offense_team, defense_team, home_offense)
  4. eraの追加
  5. 不要カラムの削除
  6. clutch/strict clutchサブセットの保存
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("data")
WP_DIR = BASE_DIR / "wp"
RAW_DIR = BASE_DIR / "nba_raw"
OUT_DIR = BASE_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEAM_STATS_DEFAULT = OUT_DIR / "team_shot_stats_2000_2024.parquet"
TEAM_FOULS_DEFAULT = OUT_DIR / "cumulative_team_fouls_2000_2024_rs.parquet"
PROCESSED_DIR = BASE_DIR / "processed"


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


def normalize_game_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def _resolve_elo_col(columns: pd.Index, elo_col: str, elo_k: int) -> str | None:
    candidates = [
        elo_col,
        f"{elo_col}_k{elo_k}",
        "elo_diff_pregame",
        f"elo_diff_pregame_k{elo_k}",
    ]
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def load_elo_diff_map(elo_path: Path, elo_col: str = "elo_diff_pregame", elo_k: int = 20) -> pd.DataFrame:
    if not elo_path.exists():
        print(f"[WARN] Elo source not found: {elo_path}")
        return pd.DataFrame(columns=["GAME_ID", "elo_diff"])

    try:
        if elo_path.suffix == ".parquet":
            elo = pd.read_parquet(elo_path)
        else:
            elo = pd.read_csv(
                elo_path,
                compression="gzip" if str(elo_path).endswith(".gz") else None,
                low_memory=False,
            )
    except Exception as e:
        print(f"[WARN] Failed to read Elo source {elo_path}: {e}")
        return pd.DataFrame(columns=["GAME_ID", "elo_diff"])

    game_id_col = None
    for cand in ["game_id", "GAME_ID"]:
        if cand in elo.columns:
            game_id_col = cand
            break
    resolved_elo_col = _resolve_elo_col(elo.columns, elo_col, elo_k)
    if game_id_col is None or resolved_elo_col is None:
        print(
            f"[WARN] Elo source missing required columns: one of [game_id, GAME_ID], "
            f"one of [{elo_col}, {elo_col}_k{elo_k}, elo_diff_pregame, elo_diff_pregame_k{elo_k}]"
        )
        return pd.DataFrame(columns=["GAME_ID", "elo_diff"])

    elo = elo[[game_id_col, resolved_elo_col]].copy()
    elo = elo.rename(columns={game_id_col: "GAME_ID", resolved_elo_col: "elo_diff"})
    elo["GAME_ID"] = normalize_game_id_series(elo["GAME_ID"])
    elo["elo_diff"] = pd.to_numeric(elo["elo_diff"], errors="coerce")
    elo = elo.dropna(subset=["GAME_ID", "elo_diff"])
    elo = elo.drop_duplicates(subset=["GAME_ID"], keep="first")
    return elo[["GAME_ID", "elo_diff"]]


def load_elo_diff_map_from_processed_games(
    seasons: list[int],
    seasontype: str,
    elo_col: str = "elo_diff_pregame",
    elo_k: int = 20,
    processed_dir: Path = PROCESSED_DIR,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for season in sorted(seasons):
        path = processed_dir / f"games_{season}_{seasontype}.parquet"
        if not path.exists():
            print(f"[WARN] Processed games Elo source not found for season={season}: {path}")
            continue
        sub = load_elo_diff_map(path, elo_col=elo_col, elo_k=elo_k)
        if not sub.empty:
            parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=["GAME_ID", "elo_diff"])

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["GAME_ID"], keep="first")
    return out[["GAME_ID", "elo_diff"]]


def season_to_era(season: int) -> str:
    """ルール変更や戦術トレンドに基づく era マッピング"""
    if season <= 2003:
        return "transition_pre2004"
    elif season <= 2014:
        return "post_handcheck_pre3p"
    elif season <= 2017:
        return "pace_space_early"
    else:
        return "modern_3p"


def map_shot_zone_basic_to_choice(basic: str) -> str:
    s = "" if basic is None else str(basic).strip()
    if s == "Restricted Area":
        return "Restricted Area"
    if s == "In The Paint (Non-RA)":
        return "In The Paint (Non-RA)"
    if s == "Mid-Range":
        return "Mid-Range"
    if s == "Above the Break 3":
        return "Above the Break 3"
    if s in ("Left Corner 3", "Right Corner 3"):
        return "Corner 3"
    return "Other"


def map_start_type_to_group(start_type: object) -> str:
    s = "" if start_type is None else str(start_type).strip().lower()
    dead = {
        "off dead ball",
        "off timeout",
        "off ft make",
        "off at rim make",
        "off long mid-range make",
        "off short mid-range make",
        "off arc 3 make",
        "off corner 3 make",
    }
    live = {
        "off steal",
        "off ft miss",
        "off long mid-range miss",
        "off short mid-range miss",
        "off arc 3 miss",
        "off corner 3 miss",
        "off at rim miss",
        "off at rim block",
        "off block",
    }
    if s in live:
        return "live_ball"
    if s in dead:
        return "dead_ball"
    if ("inbound" in s) or ("jump ball" in s):
        return "dead_ball"
    if s in ("", "unknown", "nan"):
        return "dead_ball"
    return "dead_ball"


def find_shotdetail_file(season: int, seasontype: str) -> Path | None:
    candidates = [
        RAW_DIR / f"shotdetail_{seasontype}_{season}.csv",
        RAW_DIR / f"shotdetail_{season}_{seasontype}.csv",
        RAW_DIR / f"shotdetail_{season}.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def build_shot_zone_map(seasons: list[int], seasontype: str) -> pd.DataFrame:
    parts = []
    for season in sorted(seasons):
        shot_path = find_shotdetail_file(season, seasontype)
        if shot_path is None:
            print(f"[WARN] shotdetail CSV not found for season={season}, seasontype={seasontype}")
            continue

        shots = pd.read_csv(shot_path, low_memory=False)
        required_shot = [
            "GAME_ID",
            "GAME_EVENT_ID",
            "SHOT_ZONE_BASIC",
            "SHOT_ATTEMPTED_FLAG",
        ]
        missing_s = [c for c in required_shot if c not in shots.columns]
        if missing_s:
            print(f"[WARN] {shot_path} missing columns: {missing_s}")
            continue

        shots = shots[shots["SHOT_ATTEMPTED_FLAG"] == 1].copy()
        shots["GAME_ID"] = (
            shots["GAME_ID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )
        shots["GAME_EVENT_ID"] = pd.to_numeric(shots["GAME_EVENT_ID"], errors="coerce").astype("Int64")
        shots = shots.dropna(subset=["GAME_EVENT_ID"])
        if "TEAM_ID" in shots.columns:
            shots["shot_team_id"] = pd.to_numeric(shots["TEAM_ID"], errors="coerce").astype("Int64")
        if "OPPONENT_TEAM_ID" in shots.columns:
            shots["shot_opponent_team_id"] = pd.to_numeric(shots["OPPONENT_TEAM_ID"], errors="coerce").astype("Int64")
        elif "shot_team_id" in shots.columns:
            game_teams = shots[["GAME_ID", "shot_team_id"]].dropna().drop_duplicates()
            opp_pairs = game_teams.merge(game_teams, on="GAME_ID", suffixes=("", "_opp"))
            opp_pairs = opp_pairs[opp_pairs["shot_team_id"] != opp_pairs["shot_team_id_opp"]].copy()
            opp_pairs = opp_pairs.drop_duplicates(subset=["GAME_ID", "shot_team_id"])
            shots = shots.merge(
                opp_pairs.rename(columns={"shot_team_id_opp": "shot_opponent_team_id"}),
                on=["GAME_ID", "shot_team_id"],
                how="left",
            )
        shots["shot_zone_choice"] = shots["SHOT_ZONE_BASIC"].apply(map_shot_zone_basic_to_choice)
        keep_cols = ["GAME_ID", "GAME_EVENT_ID", "shot_zone_choice"]
        for c in ["shot_team_id", "shot_opponent_team_id"]:
            if c in shots.columns:
                keep_cols.append(c)
        shots = shots[keep_cols].drop_duplicates(
            subset=["GAME_ID", "GAME_EVENT_ID"]
        )
        parts.append(shots)

    if not parts:
        return pd.DataFrame(columns=["GAME_ID", "GAME_EVENT_ID", "shot_zone_choice", "shot_team_id", "shot_opponent_team_id"])

    return pd.concat(parts, ignore_index=True)


def add_shot_zone_from_map(df: pd.DataFrame, shot_map: pd.DataFrame) -> pd.DataFrame:
    if "GAME_EVENT_ID" not in df.columns:
        raise ValueError("GAME_EVENT_ID column not found in input data")

    df["GAME_ID"] = (
        df["GAME_ID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )
    df["GAME_EVENT_ID"] = pd.to_numeric(df["GAME_EVENT_ID"], errors="coerce").astype("Int64")

    if shot_map.empty:
        if "shot_zone_choice" not in df.columns:
            df["shot_zone_choice"] = None
        return df

    if "shot_zone_choice" in df.columns:
        df = df.rename(columns={"shot_zone_choice": "shot_zone_choice_existing"})

    before = len(df)
    df = df.merge(shot_map, on=["GAME_ID", "GAME_EVENT_ID"], how="left")
    if "shot_zone_choice_existing" in df.columns:
        df["shot_zone_choice"] = df["shot_zone_choice"].combine_first(
            df["shot_zone_choice_existing"]
        )
        df = df.drop(columns=["shot_zone_choice_existing"])
    matched = df["shot_zone_choice"].notna().sum()
    print(f"  -> Shotdetail matched: {matched:,} / {before:,} ({matched / max(before, 1) * 100:.2f}%)")
    return df


def add_shot_ids_from_map(df: pd.DataFrame, shot_map: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in ["shot_team_id", "shot_opponent_team_id"] if c in shot_map.columns]
    if not id_cols:
        return df
    if "GAME_EVENT_ID" not in df.columns:
        return df

    df["GAME_ID"] = normalize_game_id_series(df["GAME_ID"])
    df["GAME_EVENT_ID"] = pd.to_numeric(df["GAME_EVENT_ID"], errors="coerce").astype("Int64")
    merge_map = shot_map[["GAME_ID", "GAME_EVENT_ID"] + id_cols].drop_duplicates(
        subset=["GAME_ID", "GAME_EVENT_ID"]
    )
    before = len(df)
    df = df.merge(merge_map, on=["GAME_ID", "GAME_EVENT_ID"], how="left")
    if len(df) != before:
        print(f"[WARN] Row count changed after shot-id merge: {before} -> {len(df)}")
    for c in id_cols:
        matched = int(df[c].notna().sum())
        print(f"  -> {c} matched: {matched:,} / {len(df):,} ({matched / max(len(df), 1) * 100:.2f}%)")
    return df


def build_team_info_map(seasons: list[int], seasontype: str) -> pd.DataFrame:
    all_team_info = []
    for season in sorted(seasons):
        team_info = load_team_abbreviations_from_nbastats(season, seasontype)
        if not team_info.empty:
            all_team_info.append(team_info)

    if not all_team_info:
        return pd.DataFrame(columns=["GAME_ID", "home_team", "visitor_team"])

    return pd.concat(all_team_info, ignore_index=True).drop_duplicates(subset=["GAME_ID"])


def add_team_info_from_map(df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    if team_df.empty:
        df["home_team"] = None
        df["visitor_team"] = None
        df["offense_team"] = None
        df["defense_team"] = None
        if "before_home_possession" in df.columns:
            df["home_offense"] = df["before_home_possession"].astype(float)
        else:
            df["home_offense"] = None
        return df

    df["GAME_ID"] = (
        df["GAME_ID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )

    before_len = len(df)
    df = df.merge(team_df, on="GAME_ID", how="left")
    after_len = len(df)

    if before_len != after_len:
        print(f"[WARN] Row count changed after merge: {before_len} -> {after_len}")

    if "before_home_possession" in df.columns:
        df["offense_team"] = np.where(
            df["before_home_possession"] == 1,
            df["home_team"],
            df["visitor_team"]
        )
        df["defense_team"] = np.where(
            df["before_home_possession"] == 1,
            df["visitor_team"],
            df["home_team"]
        )
        df["home_offense"] = df["before_home_possession"].astype(float)
    else:
        df["offense_team"] = None
        df["defense_team"] = None
        df["home_offense"] = None
        print("[WARN] before_home_possession column not found")

    matched = df["offense_team"].notna().sum()
    total = len(df)
    print(f"  -> Team info matched: {matched:,} / {total:,} ({matched/total*100:.2f}%)")

    return df


def load_team_abbreviations_from_nbastats(season: int, seasontype: str) -> pd.DataFrame:
    """
    nbastats CSVからゲームIDごとのホーム/ビジターチーム略称を取得
    Returns: DataFrame with columns [GAME_ID, home_team, visitor_team]
    """
    nba_path = find_nbastats_file(season, seasontype)

    if nba_path is None:
        print(f"[WARN] nbastats CSV not found for season={season}, seasontype={seasontype}")
        return pd.DataFrame(columns=["GAME_ID", "home_team", "visitor_team"])

    df_nba = pd.read_csv(nba_path, low_memory=False)
    if "GAME_ID" in df_nba.columns:
        df_nba["GAME_ID"] = (
            df_nba["GAME_ID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )

    # Filter for play-by-play events that have team info
    home_events = df_nba[df_nba["HOMEDESCRIPTION"].notna()].copy()
    visitor_events = df_nba[df_nba["VISITORDESCRIPTION"].notna()].copy()

    if "PLAYER1_TEAM_ABBREVIATION" not in df_nba.columns:
        print(f"[WARN] PLAYER1_TEAM_ABBREVIATION not found in nbastats CSV")
        return pd.DataFrame(columns=["GAME_ID", "home_team", "visitor_team"])

    # Use majority vote to determine home/visitor teams (robust against spurious events)
    game_home = home_events.groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else None
    )
    game_visitor = visitor_events.groupby("GAME_ID")["PLAYER1_TEAM_ABBREVIATION"].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else None
    )

    # Combine and validate (exclude games where home == visitor)
    common_games = sorted(set(game_home.index) & set(game_visitor.index))

    valid_games = []
    for gid in common_games:
        h = game_home[gid]
        v = game_visitor[gid]
        if h != v and pd.notna(h) and pd.notna(v):
            # Normalize GAME_ID to 10 digits
            gid_str = str(gid).strip()
            if '.' in gid_str:
                gid_str = gid_str.split('.')[0]
            gid_str = gid_str.zfill(10)
            valid_games.append({"GAME_ID": gid_str, "home_team": h, "visitor_team": v})

    if not valid_games:
        print(f"[WARN] No valid game-team mappings found for season {season}")
        return pd.DataFrame(columns=["GAME_ID", "home_team", "visitor_team"])

    return pd.DataFrame(valid_games)


def _extract_timeout_used(desc: pd.Series) -> pd.Series:
    txt = desc.fillna("").astype(str)
    full = pd.to_numeric(
        txt.str.extract(r"(?i)full\D*([0-9]+)", expand=False),
        errors="coerce",
    )
    reg = pd.to_numeric(
        txt.str.extract(r"(?i)reg\.?\D*([0-9]+)", expand=False),
        errors="coerce",
    )
    short = pd.to_numeric(
        txt.str.extract(r"(?i)short\D*([0-9]+)", expand=False),
        errors="coerce",
    ).fillna(0.0)
    base = full.fillna(reg)
    out = base + short
    return out.where(base.notna(), np.nan)


def build_timeout_remaining_map(
    seasons: list[int],
    seasontype: str,
    total_timeouts_per_team: int = 7,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for season in sorted(seasons):
        nba_path = find_nbastats_file(season, seasontype)
        if nba_path is None:
            print(f"[WARN] nbastats CSV not found for season={season}, seasontype={seasontype}")
            continue

        usecols = [
            "GAME_ID",
            "EVENTNUM",
            "EVENTMSGTYPE",
            "HOMEDESCRIPTION",
            "VISITORDESCRIPTION",
            "NEUTRALDESCRIPTION",
        ]
        df_nba = pd.read_csv(nba_path, low_memory=False, usecols=lambda c: c in usecols)
        required = {"GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "HOMEDESCRIPTION", "VISITORDESCRIPTION"}
        if not required.issubset(df_nba.columns):
            missing = sorted(required - set(df_nba.columns))
            print(f"[WARN] {nba_path} missing columns for timeout map: {missing}")
            continue

        ev = df_nba.copy()
        ev["GAME_ID"] = (
            ev["GAME_ID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )
        ev["EVENTNUM"] = pd.to_numeric(ev["EVENTNUM"], errors="coerce").astype("Int64")
        ev["EVENTMSGTYPE"] = pd.to_numeric(ev["EVENTMSGTYPE"], errors="coerce").astype("Int64")
        ev = ev.dropna(subset=["EVENTNUM"]).copy()
        ev["EVENTNUM"] = ev["EVENTNUM"].astype(int)
        ev = ev.sort_values(["GAME_ID", "EVENTNUM"], kind="mergesort").copy()

        is_to = ev["EVENTMSGTYPE"] == 9
        home_desc = ev["HOMEDESCRIPTION"].fillna("").astype(str)
        vis_desc = ev["VISITORDESCRIPTION"].fillna("").astype(str)
        neu_desc = ev["NEUTRALDESCRIPTION"].fillna("").astype(str) if "NEUTRALDESCRIPTION" in ev.columns else ""

        home_team_to = is_to & home_desc.str.contains("timeout", case=False, na=False)
        vis_team_to = is_to & vis_desc.str.contains("timeout", case=False, na=False)
        official_to = is_to & (
            home_desc.str.contains("official", case=False, na=False)
            | vis_desc.str.contains("official", case=False, na=False)
            | (neu_desc.str.contains("official", case=False, na=False) if isinstance(neu_desc, pd.Series) else False)
        )
        home_team_to = home_team_to & ~official_to
        vis_team_to = vis_team_to & ~official_to

        home_used = pd.Series(np.nan, index=ev.index, dtype=float)
        vis_used = pd.Series(np.nan, index=ev.index, dtype=float)
        home_used.loc[home_team_to] = _extract_timeout_used(home_desc.loc[home_team_to])
        vis_used.loc[vis_team_to] = _extract_timeout_used(vis_desc.loc[vis_team_to])

        home_used_fallback = home_team_to.groupby(ev["GAME_ID"]).cumsum().astype(float)
        vis_used_fallback = vis_team_to.groupby(ev["GAME_ID"]).cumsum().astype(float)
        home_used.loc[home_team_to & home_used.isna()] = home_used_fallback.loc[home_team_to & home_used.isna()]
        vis_used.loc[vis_team_to & vis_used.isna()] = vis_used_fallback.loc[vis_team_to & vis_used.isna()]

        ev["home_timeouts_used_after"] = home_used.groupby(ev["GAME_ID"]).ffill().fillna(0.0)
        ev["visitor_timeouts_used_after"] = vis_used.groupby(ev["GAME_ID"]).ffill().fillna(0.0)
        # PBP text occasionally contains inconsistent timeout counters; enforce monotone used counts.
        ev["home_timeouts_used_after"] = ev.groupby("GAME_ID")["home_timeouts_used_after"].cummax()
        ev["visitor_timeouts_used_after"] = ev.groupby("GAME_ID")["visitor_timeouts_used_after"].cummax()

        ev["home_timeouts_used_before"] = (
            ev.groupby("GAME_ID")["home_timeouts_used_after"].shift(1).fillna(0.0)
        )
        ev["visitor_timeouts_used_before"] = (
            ev.groupby("GAME_ID")["visitor_timeouts_used_after"].shift(1).fillna(0.0)
        )

        ev["home_timeouts_left"] = (
            float(total_timeouts_per_team) - ev["home_timeouts_used_before"]
        ).clip(lower=0.0)
        ev["visitor_timeouts_left"] = (
            float(total_timeouts_per_team) - ev["visitor_timeouts_used_before"]
        ).clip(lower=0.0)

        out = ev[["GAME_ID", "EVENTNUM", "home_timeouts_left", "visitor_timeouts_left"]].copy()
        out = out.rename(columns={"EVENTNUM": "GAME_EVENT_ID"})
        out["GAME_EVENT_ID"] = pd.to_numeric(out["GAME_EVENT_ID"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["GAME_EVENT_ID"]).drop_duplicates(subset=["GAME_ID", "GAME_EVENT_ID"])
        parts.append(out)

    if not parts:
        return pd.DataFrame(columns=["GAME_ID", "GAME_EVENT_ID", "home_timeouts_left", "visitor_timeouts_left"])
    return pd.concat(parts, ignore_index=True)


def load_team_shot_stats(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        print(f"[WARN] Team shot stats not found: {path}")
        return pd.DataFrame(), []

    team_stats = pd.read_parquet(path)
    required_cols = {"GAME_ID", "GAME_EVENT_ID"}
    if not required_cols.issubset(team_stats.columns):
        print(f"[WARN] Team shot stats missing keys: {sorted(required_cols - set(team_stats.columns))}")
        return pd.DataFrame(), []

    feature_cols = [
        c for c in team_stats.columns
        if c.startswith("own_") or c.startswith("opp_")
    ]
    keep_cols = ["GAME_ID", "GAME_EVENT_ID"] + feature_cols
    team_stats = team_stats[keep_cols].copy()
    team_stats["GAME_ID"] = (
        team_stats["GAME_ID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )
    team_stats["GAME_EVENT_ID"] = pd.to_numeric(team_stats["GAME_EVENT_ID"], errors="coerce").astype("Int64")
    team_stats = team_stats.dropna(subset=["GAME_EVENT_ID"]).drop_duplicates(
        subset=["GAME_ID", "GAME_EVENT_ID"]
    )
    return team_stats, feature_cols


def load_team_shot_stats_season_fixed(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        return pd.DataFrame(), []

    team_stats = pd.read_parquet(path)
    required_cols = {"season", "team_id"}
    if not required_cols.issubset(team_stats.columns):
        return pd.DataFrame(), []

    base_feature_cols = [
        c for c in team_stats.columns
        if c not in {"season", "team_id"} and not c.startswith("_")
    ]
    if not base_feature_cols:
        print("[WARN] Season-fixed team stats has no feature columns.")
        return pd.DataFrame(), []

    keep_cols = ["season", "team_id"] + base_feature_cols
    out = team_stats[keep_cols].copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["season", "team_id"]).drop_duplicates(subset=["season", "team_id"], keep="last")
    return out, base_feature_cols


def merge_team_stats_from_season_fixed(
    df: pd.DataFrame,
    team_stats_fixed: pd.DataFrame,
    base_feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if df.empty or team_stats_fixed.empty or not base_feature_cols:
        return df, []
    needed_df_cols = {"season", "shot_team_id", "shot_opponent_team_id"}
    if not needed_df_cols.issubset(df.columns):
        print(f"[WARN] Season-fixed team-stats merge skipped; missing df cols: {sorted(needed_df_cols - set(df.columns))}")
        return df, []

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["shot_team_id"] = pd.to_numeric(df["shot_team_id"], errors="coerce").astype("Int64")
    df["shot_opponent_team_id"] = pd.to_numeric(df["shot_opponent_team_id"], errors="coerce").astype("Int64")

    prefixed_cols = [f"own_{c}" for c in base_feature_cols] + [f"opp_{c}" for c in base_feature_cols]
    for c in prefixed_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    own = team_stats_fixed.rename(columns={"team_id": "shot_team_id", **{c: f"own_{c}" for c in base_feature_cols}})
    opp = team_stats_fixed.rename(columns={"team_id": "shot_opponent_team_id", **{c: f"opp_{c}" for c in base_feature_cols}})

    before = len(df)
    df = df.merge(own, on=["season", "shot_team_id"], how="left")
    df = df.merge(opp, on=["season", "shot_opponent_team_id"], how="left")
    if len(df) != before:
        print(f"[WARN] Row count changed after season-fixed team stats merge: {before} -> {len(df)}")

    sample_col = next((c for c in prefixed_cols if c in df.columns), None)
    if sample_col is not None:
        matched = int(df[sample_col].notna().sum())
        print(
            f"  -> Season-fixed team stats matched: {matched:,} / {len(df):,} "
            f"({matched / max(len(df), 1) * 100:.2f}%)"
        )
    return df, prefixed_cols


def load_team_fouls(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Team fouls not found: {path}")
        return pd.DataFrame()

    team_fouls = pd.read_parquet(path)

    # Accept common schema variants (e.g., away_* instead of visitor_*).
    alias_map = {
        "visitor_team": ["away_team"],
        "visitor_fouls_game": ["away_fouls_game"],
        "visitor_fouls_period": ["away_fouls_period"],
    }
    rename_map: dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        if canonical in team_fouls.columns:
            continue
        for alias in aliases:
            if alias in team_fouls.columns:
                rename_map[alias] = canonical
                break
    if rename_map:
        team_fouls = team_fouls.rename(columns=rename_map)
        print(f"[INFO] Renamed team fouls columns: {rename_map}")

    required_cols = {
        "GAME_ID",
        "GAME_EVENT_ID",
        "home_fouls_game",
        "visitor_fouls_game",
        "home_fouls_period",
        "visitor_fouls_period",
    }
    missing = sorted(required_cols - set(team_fouls.columns))
    if missing:
        print(f"[WARN] Team fouls missing columns: {missing}")
        return pd.DataFrame()

    keep_cols = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "home_fouls_game",
        "visitor_fouls_game",
        "home_fouls_period",
        "visitor_fouls_period",
    ]
    team_fouls = team_fouls[keep_cols].copy()
    team_fouls["GAME_ID"] = (
        team_fouls["GAME_ID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )
    team_fouls["GAME_EVENT_ID"] = pd.to_numeric(team_fouls["GAME_EVENT_ID"], errors="coerce").astype("Int64")
    team_fouls = team_fouls.dropna(subset=["GAME_EVENT_ID"]).drop_duplicates(
        subset=["GAME_ID", "GAME_EVENT_ID"]
    )
    return team_fouls


def extract_season_from_game_id(game_id: str) -> int:
    """
    GAME_ID (e.g., '0020000001') から season を抽出
    Format: 00YYXXXXX where YY is season start year (e.g., 00 -> 2000)
    """
    try:
        gid_str = str(game_id).strip().zfill(10)
        yy = int(gid_str[3:5])
        return 2000 + yy
    except:
        return None


def apply_offense_perspective(df: pd.DataFrame) -> pd.DataFrame:
    if "home_offense" not in df.columns:
        return df
    home_offense = df["home_offense"]
    sign = np.where(home_offense == 1, 1.0, np.where(home_offense == 0, -1.0, np.nan))

    if "score_diff" in df.columns and "score_diff_home" not in df.columns:
        df["score_diff_home"] = df["score_diff"]
    if "score_diff_home" in df.columns:
        df["score_diff_offense"] = df["score_diff_home"] * sign
        df["score_diff"] = df["score_diff_offense"]

    if "wp_before" in df.columns and "wp_before_home" not in df.columns:
        df["wp_before_home"] = df["wp_before"]
    if "wp_before_home" in df.columns:
        df["wp_before_offense"] = np.where(
            home_offense == 1,
            df["wp_before_home"],
            np.where(home_offense == 0, 1.0 - df["wp_before_home"], np.nan),
        )
        df["wp_before"] = df["wp_before_offense"]

    if "wp_next" in df.columns and "wp_next_home" not in df.columns:
        df["wp_next_home"] = df["wp_next"]
    if "wp_next_home" in df.columns:
        df["wp_next_offense"] = np.where(
            home_offense == 1,
            df["wp_next_home"],
            np.where(home_offense == 0, 1.0 - df["wp_next_home"], np.nan),
        )
        df["wp_next"] = df["wp_next_offense"]

    if "delta_wp" in df.columns and "delta_wp_home" not in df.columns:
        df["delta_wp_home"] = df["delta_wp"]
    if "delta_wp_home" in df.columns:
        df["delta_wp_offense"] = df["delta_wp_home"] * sign
        df["delta_wp"] = df["delta_wp_offense"]

    if "elo_diff" in df.columns and "elo_diff_home" not in df.columns:
        df["elo_diff_home"] = pd.to_numeric(df["elo_diff"], errors="coerce")
    if "elo_diff_home" in df.columns:
        df["elo_diff"] = df["elo_diff_home"] * sign

    return df



def main():
    parser = argparse.ArgumentParser(description="Build DML shot choice panel from WP-scored shot states")
    parser.add_argument("--input", type=str, default="data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz",
                        help="Input CSV from fit_wp_and_score_shots.R")
    parser.add_argument("--seasontype", type=str, default="rs", choices=["rs"],
                        help="Season type")
    parser.add_argument("--team-stats", type=str, default=str(TEAM_STATS_DEFAULT),
                        help="Team shot stats parquet (construct_team_shot_stats.py output)")
    parser.add_argument("--team-fouls", type=str, default=str(TEAM_FOULS_DEFAULT),
                        help="Team foul counts parquet (construct_cumulative_team_foul.py output)")
    parser.add_argument(
        "--elo-ref-path",
        type=str,
        default="",
        help="Optional Elo source CSV(.gz) with game_id and elo_diff_pregame. "
             "Default: read data/processed/games_*_{seasontype}.parquet.",
    )
    parser.add_argument(
        "--elo-k",
        type=int,
        default=20,
        help="Elo K used when auto-resolving --elo-ref-path.",
    )
    parser.add_argument(
        "--elo-col",
        type=str,
        default="elo_diff_pregame",
        help="Column name in --elo-ref-path to use as Elo diff.",
    )
    parser.add_argument("--output-dir", type=str, default="data/analysis",
                        help="Output directory for parquet files")
    parser.add_argument(
        "--dml-output",
        type=str,
        default="",
        help=(
            "Optional output path for the all-shot DML CSV. "
            "Default: data/wp/shot_decision_panel_{start}_{end}_{seasontype}_dml.csv.gz"
        ),
    )
    parser.add_argument(
        "--total-timeouts-per-team",
        type=int,
        default=7,
        help="Assumed total timeout budget per team per game for remaining-timeout calculation.",
    )
    parser.add_argument(
        "--with-strict",
        action="store_true",
        help="Also write strict clutch panel (<=5). Default: off",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path, compression="gzip")
    print(f"  -> Total shots: {len(df):,}")
    print(f"  -> Columns: {list(df.columns)}")

    # Rename columns from R output to match DML expectations
    # R output has: before_score_diff, before_time_left_game, before_OT_flag
    # DML expects: score_diff, time_left_game, OT_flag, period
    rename_map = {
        "before_score_diff": "score_diff",
        "before_time_left_game": "time_left_game",
        "before_OT_flag": "OT_flag",
        "before_start_type": "start_type",
    }
    df = df.rename(columns=rename_map)

    # Add period if not present (derive from time_left_game)
    if "period" not in df.columns:
        # Approximate: if time_left_game > 2400, period < 4
        # This is rough - ideally period should come from shot_decision_states
        # For now, assume all shots are in period 4+ since we filter to clutch
        df["period"] = 4  # Placeholder - should be in input data

    # Add season if not present
    if "season" not in df.columns:
        df["season"] = df["GAME_ID"].apply(extract_season_from_game_id)
    seasons = [int(s) for s in sorted(df["season"].dropna().unique())]

    if "start_type" not in df.columns:
        df["start_type"] = None
    df["start_type_group"] = df["start_type"].apply(map_start_type_to_group).astype("category")

    if args.elo_ref_path:
        elo_path = Path(args.elo_ref_path)
        elo_map = load_elo_diff_map(elo_path, elo_col=args.elo_col, elo_k=args.elo_k)
        elo_source_label = str(elo_path)
    else:
        elo_map = load_elo_diff_map_from_processed_games(
            seasons=seasons,
            seasontype=args.seasontype,
            elo_col=args.elo_col,
            elo_k=args.elo_k,
        )
        elo_source_label = f"{PROCESSED_DIR}/games_*_{args.seasontype}.parquet"
    if not elo_map.empty:
        df["GAME_ID"] = normalize_game_id_series(df["GAME_ID"])
        before_len = len(df)
        df = df.merge(elo_map, on="GAME_ID", how="left")
        after_len = len(df)
        if before_len != after_len:
            print(f"[WARN] Row count changed after Elo merge: {before_len} -> {after_len}")
        matched = int(df["elo_diff"].notna().sum())
        print(f"  -> Elo matched: {matched:,} / {len(df):,} ({matched / max(len(df), 1) * 100:.2f}%) from {elo_source_label}")
    else:
        df["elo_diff"] = np.nan
        print("  -> Elo merge skipped (no valid elo source); elo_diff filled with NaN.")

    shot_map = build_shot_zone_map(seasons, args.seasontype)
    df = add_shot_ids_from_map(df, shot_map)

    team_stats_path = Path(args.team_stats)
    team_stats, team_feature_cols = load_team_shot_stats(team_stats_path)
    team_stats_fixed, team_stats_fixed_base_cols = load_team_shot_stats_season_fixed(team_stats_path)
    team_stats_event_matched = 0
    if not team_stats.empty:
        df["GAME_ID"] = (
            df["GAME_ID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )
        df["GAME_EVENT_ID"] = pd.to_numeric(df["GAME_EVENT_ID"], errors="coerce").astype("Int64")
        before_len = len(df)
        df = df.merge(team_stats, on=["GAME_ID", "GAME_EVENT_ID"], how="left")
        after_len = len(df)
        if before_len != after_len:
            print(f"[WARN] Row count changed after team stats merge: {before_len} -> {after_len}")
        sample_col = team_feature_cols[0] if team_feature_cols else None
        if sample_col and sample_col in df.columns:
            team_stats_event_matched = int(df[sample_col].notna().sum())
            print(
                f"  -> Team stats event-level matched: {team_stats_event_matched:,} / {len(df):,} "
                f"({team_stats_event_matched / max(len(df), 1) * 100:.2f}%)"
            )
    team_fouls_path = Path(args.team_fouls)
    team_fouls = load_team_fouls(team_fouls_path)
    if not team_fouls.empty:
        df["GAME_ID"] = (
            df["GAME_ID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )
        df["GAME_EVENT_ID"] = pd.to_numeric(df["GAME_EVENT_ID"], errors="coerce").astype("Int64")
        before_len = len(df)
        df = df.merge(team_fouls, on=["GAME_ID", "GAME_EVENT_ID"], how="left")
        after_len = len(df)
        if before_len != after_len:
            print(f"[WARN] Row count changed after team fouls merge: {before_len} -> {after_len}")

    # Add clutch flags
    print("\nCalculating clutch flags...")
    is_clutch_window = (df["period"] >= 4) & (df["time_left_game"] <= 300)
    abs_score_diff = df["score_diff"].abs()

    df["clutch_flag"] = is_clutch_window & (abs_score_diff <= 10)
    df["strict_clutch_flag"] = is_clutch_window & (abs_score_diff <= 5)
    df["is_nc_nonclutch"] = (df["period"] >= 1) & (df["time_left_game"] > 300) & (abs_score_diff <= 10)

    print(f"  -> Clutch (<=10): {df['clutch_flag'].sum():,} shots")
    print(f"  -> Strict clutch (<=5): {df['strict_clutch_flag'].sum():,} shots")
    print(f"  -> Non-clutch control: {df['is_nc_nonclutch'].sum():,} shots")

    has_before_timeout_cols = {
        "before_home_timeouts_left",
        "before_visitor_timeouts_left",
    }.issubset(df.columns)
    if has_before_timeout_cols:
        print("Using timeout features from before_* columns in WP input.")
    else:
        timeout_map = build_timeout_remaining_map(
            seasons=seasons,
            seasontype=args.seasontype,
            total_timeouts_per_team=args.total_timeouts_per_team,
        )
        if not timeout_map.empty:
            # Ensure merge keys are consistently typed even when upstream optional
            # merges (team stats/fouls) are skipped.
            df["GAME_ID"] = normalize_game_id_series(df["GAME_ID"])
            df["GAME_EVENT_ID"] = pd.to_numeric(df["GAME_EVENT_ID"], errors="coerce").astype("Int64")
            timeout_map["GAME_ID"] = normalize_game_id_series(timeout_map["GAME_ID"])
            timeout_map["GAME_EVENT_ID"] = pd.to_numeric(
                timeout_map["GAME_EVENT_ID"], errors="coerce"
            ).astype("Int64")
            before_len = len(df)
            df = df.merge(timeout_map, on=["GAME_ID", "GAME_EVENT_ID"], how="left")
            after_len = len(df)
            if before_len != after_len:
                print(f"[WARN] Row count changed after timeout merge: {before_len} -> {after_len}")

    team_map = build_team_info_map(seasons, args.seasontype)

    def build_dml_all(df_in: pd.DataFrame) -> pd.DataFrame:
        df_local = df_in.copy()
        df_local = add_shot_zone_from_map(df_local, shot_map)
        df_local = add_team_info_from_map(df_local, team_map)
        df_local = apply_offense_perspective(df_local)
        df_local["era"] = df_local["season"].apply(season_to_era)
        if "home_offense" in df_local.columns:
            foul_cols = {"home_fouls_period", "visitor_fouls_period"}
            if foul_cols.issubset(df_local.columns):
                df_local["own_fouls_period"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["home_fouls_period"],
                    df_local["visitor_fouls_period"],
                )
                df_local["opp_fouls_period"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["visitor_fouls_period"],
                    df_local["home_fouls_period"],
                )
            else:
                df_local["own_fouls_period"] = np.nan
                df_local["opp_fouls_period"] = np.nan
            if {"before_home_timeouts_left", "before_visitor_timeouts_left"}.issubset(df_local.columns):
                df_local["timeouts_left_us"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["before_home_timeouts_left"],
                    df_local["before_visitor_timeouts_left"],
                )
                df_local["timeouts_left_them"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["before_visitor_timeouts_left"],
                    df_local["before_home_timeouts_left"],
                )
            elif {"home_timeouts_left", "visitor_timeouts_left"}.issubset(df_local.columns):
                df_local["timeouts_left_us"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["home_timeouts_left"],
                    df_local["visitor_timeouts_left"],
                )
                df_local["timeouts_left_them"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["visitor_timeouts_left"],
                    df_local["home_timeouts_left"],
                )
            else:
                df_local["timeouts_left_us"] = np.nan
                df_local["timeouts_left_them"] = np.nan

        dml_cols = [
            "delta_wp",
            "shot_zone_choice",
            "GAME_ID",
            "GAME_EVENT_ID",
            "period",
            "time_left_game",
            "season",
            "seasontype",
            "era",
            "score_diff",
            "OT_flag",
            "start_type",
            "start_type_group",
            "after_off_reb",
            "elo_diff",
            "before_home_possession",
            "own_fouls_period",
            "opp_fouls_period",
            "timeouts_left_us",
            "timeouts_left_them",
        ]
        if team_feature_cols:
            dml_cols.extend(team_feature_cols)

        dml_cols = [c for c in dml_cols if c in df_local.columns]
        missing = [c for c in ["delta_wp", "shot_zone_choice"] if c not in df_local.columns]
        if missing:
            raise ValueError(f"Required columns missing for DML: {missing}")

        df_out = df_local[dml_cols].copy()
        df_out = df_out.dropna(subset=["delta_wp", "shot_zone_choice"])
        if "start_type" in df_out.columns:
            before = len(df_out)
            df_out = df_out[df_out["start_type"].notna()]
            dropped = before - len(df_out)
            if dropped:
                print(f"\n  -> Removed {dropped:,} rows with missing start_type (DML all)")

        if "shot_zone_choice" in df_out.columns:
            before = len(df_out)
            df_out = df_out[df_out["shot_zone_choice"] != "Other"]
            after = len(df_out)
            print(f"\n  -> Removed {before - after:,} 'Other' zone shots (DML all)")

        df_out["shot_zone_choice"] = df_out["shot_zone_choice"].astype("category")
        return df_out

    def build_panel(df_in: pd.DataFrame) -> pd.DataFrame:
        df_local = df_in.copy()
        df_local = add_shot_zone_from_map(df_local, shot_map)
        df_local = add_team_info_from_map(df_local, team_map)
        df_local = apply_offense_perspective(df_local)
        df_local["era"] = df_local["season"].apply(season_to_era)
        if "home_offense" in df_local.columns:
            foul_cols = {"home_fouls_period", "visitor_fouls_period"}
            if foul_cols.issubset(df_local.columns):
                df_local["own_fouls_period"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["home_fouls_period"],
                    df_local["visitor_fouls_period"],
                )
                df_local["opp_fouls_period"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["visitor_fouls_period"],
                    df_local["home_fouls_period"],
                )
            else:
                df_local["own_fouls_period"] = np.nan
                df_local["opp_fouls_period"] = np.nan
            if {"before_home_timeouts_left", "before_visitor_timeouts_left"}.issubset(df_local.columns):
                df_local["timeouts_left_us"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["before_home_timeouts_left"],
                    df_local["before_visitor_timeouts_left"],
                )
                df_local["timeouts_left_them"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["before_visitor_timeouts_left"],
                    df_local["before_home_timeouts_left"],
                )
            elif {"home_timeouts_left", "visitor_timeouts_left"}.issubset(df_local.columns):
                df_local["timeouts_left_us"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["home_timeouts_left"],
                    df_local["visitor_timeouts_left"],
                )
                df_local["timeouts_left_them"] = np.where(
                    df_local["home_offense"] == 1,
                    df_local["visitor_timeouts_left"],
                    df_local["home_timeouts_left"],
                )
            else:
                df_local["timeouts_left_us"] = np.nan
                df_local["timeouts_left_them"] = np.nan

        dml_cols = [
            "delta_wp",
            "shot_zone_choice",
            "GAME_ID",
            "GAME_EVENT_ID",
            "period",
            "time_left_game",
            "season",
            "era",
            "score_diff",
            "OT_flag",
            "start_type",
            "start_type_group",
            "after_off_reb",
            "elo_diff",
            "before_home_possession",
            "clutch_flag",
            "strict_clutch_flag",
            "is_nc_nonclutch",
            "own_fouls_period",
            "opp_fouls_period",
            "timeouts_left_us",
            "timeouts_left_them",
        ]
        if team_feature_cols:
            dml_cols.extend(team_feature_cols)

        dml_cols = [c for c in dml_cols if c in df_local.columns]
        missing = [c for c in ["delta_wp", "shot_zone_choice", "GAME_ID"] if c not in df_local.columns]
        if missing:
            raise ValueError(f"Required columns missing: {missing}")

        df_out = df_local[dml_cols].copy()
        df_out = df_out.dropna(subset=["delta_wp", "shot_zone_choice"])
        if "start_type" in df_out.columns:
            before = len(df_out)
            df_out = df_out[df_out["start_type"].notna()]
            dropped = before - len(df_out)
            if dropped:
                print(f"\n  -> Removed {dropped:,} rows with missing start_type")

        if "shot_zone_choice" in df_out.columns:
            before = len(df_out)
            df_out = df_out[df_out["shot_zone_choice"] != "Other"]
            after = len(df_out)
            print(f"\n  -> Removed {before - after:,} 'Other' zone shots")

        df_out["shot_zone_choice"] = df_out["shot_zone_choice"].astype("category")
        return df_out

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nBuilding DML panel: all shots")
    df_all = build_dml_all(df.copy())
    if seasons:
        start_season = min(seasons)
        end_season = max(seasons)
    else:
        start_season = None
        end_season = None
    if args.dml_output:
        dml_path = Path(args.dml_output)
        dml_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        dml_name = f"shot_decision_panel_{start_season}_{end_season}_{args.seasontype}_dml.csv.gz"
        dml_path = WP_DIR / dml_name
    df_all.to_csv(dml_path, index=False, compression="gzip")
    print(f"✓ Saved DML all: {dml_path} ({len(df_all):,} rows)")

    print("\nBuilding panel: clutch (<=10)")
    df_clutch = build_panel(df[df["clutch_flag"] == True].copy())
    clutch_name = f"shotchoice_panel_clutch_{args.seasontype}.parquet"
    clutch_path = out_dir / clutch_name
    df_clutch.to_parquet(clutch_path, index=False)
    print(f"✓ Saved clutch panel: {clutch_path} ({len(df_clutch):,} rows)")

    if args.with_strict:
        print("\nBuilding panel: strict clutch")
        df_strict = build_panel(df[df["strict_clutch_flag"] == True].copy())
        strict_name = f"shotchoice_panel_strict_clutch_{args.seasontype}.parquet"
        strict_path = out_dir / strict_name
        df_strict.to_parquet(strict_path, index=False)
        print(f"✓ Saved strict clutch panel: {strict_path} ({len(df_strict):,} rows)")
    else:
        print("\nSkipping strict clutch panel (use --with-strict to enable).")

    print("\n" + "=" * 80)
    print("Shot choice panel construction complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
