#!/usr/bin/env python3
"""
build_wp_features.py

Build state tables for:
  (1) Possession-start states (for fitting WP model in R)
  (2) Shot-level decision state pairs:
        delta_WP = WP(next decision state) - WP(shot-before)

Next decision state definition:
  - If missed + offensive rebound: OR event state (rebound moment)
  - Else: next possession start state
  - If terminal (no next possession): wp_next will be set to final_home_win in R

Inputs (examples):
  - data/nba_raw/nbastats_2000.csv
  - data/processed/poss_2000_rs.parquet
  - data/processed/games_2000_rs.parquet (optional)

Outputs:
  - data/wp/wp_states_{season}_{seasontype}.csv.gz
  - data/wp/shot_decision_states_{season}_{seasontype}.csv.gz

Notes:
  - This script DOES NOT compute wp_hat. Do that in R (mgcv::bam) and then
    predict twice: before_* and next_*.
  - home_possession is added (before/next and poss-start) if home/away team ids can be inferred.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "nba_raw"
PROCESSED_DIR = BASE_DIR / "processed"
WP_DIR = BASE_DIR / "wp"
WP_DIR.mkdir(parents=True, exist_ok=True)


def ensure_regular_output_path(path: Path) -> None:
    """
    Prevent writing through stale symlinks (e.g., rs -> rg aliases).
    If output path is a symlink, remove it so writes create a real file.
    """
    if path.is_symlink():
        print(f"  -> Removing symlink output path before write: {path}")
        path.unlink()


# ----------------------------
# Utils
# ----------------------------
def season_suffix(seasontype: str) -> str:
    return ""


def pick_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = set(df.columns)
    for c in cands:
        if c in cols:
            return c
    return None


def normalize_game_id(series: pd.Series) -> pd.Series:
    """
    Normalize GAME_ID as 10-digit string with leading zeros preserved.
    Handles float-like strings: '200000001.0' -> '0200000001'
    """
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    mask = s.str.fullmatch(r"\d+")
    s.loc[mask] = s.loc[mask].str.zfill(10)
    return s


def parse_time_to_seconds_any(x) -> float:
    """
    Convert "MM:SS", "M:SS.0", numeric, or "PT11M32S" to seconds left in the period.
    Unexpected -> NaN.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if not s:
        return np.nan

    if ":" in s and not s.startswith("PT"):
        try:
            mm, ss = s.split(":", 1)
            return int(mm) * 60 + int(float(ss))
        except Exception:
            return np.nan

    if s.startswith("PT") and s.endswith("S"):
        body = s[2:-1]
        if "M" in body:
            m_str, sec_str = body.split("M", 1)
            try:
                m = int(m_str) if m_str else 0
                sec = int(float(sec_str)) if sec_str else 0
                return m * 60 + sec
            except Exception:
                return np.nan
        try:
            return float(body)
        except Exception:
            return np.nan

    try:
        return float(s)
    except Exception:
        return np.nan


def period_length_seconds(period: int) -> int:
    return 12 * 60 if int(period) <= 4 else 5 * 60


def calculate_time_left_game(period: int, time_left_period: float) -> float:
    """
    Remaining time on a regulation-plus-OT scale.
    OT1 vs OT2 distinction intentionally NOT encoded.
    """
    if pd.isna(period) or pd.isna(time_left_period):
        return np.nan
    period = int(period)
    if period <= 4:
        return (4 - period) * 12 * 60 + float(time_left_period)
    return float(time_left_period)


# ----------------------------
# Loaders
# ----------------------------
def load_pbp(season: int, seasontype: str) -> pd.DataFrame:
    suf = season_suffix(seasontype)
    path = RAW_DIR / f"nbastats{suf}_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(f"PBP file not found: {path}")

    print(f"  -> Loading PBP: {path}")
    pbp = pd.read_csv(path, dtype={"GAME_ID": "string"}, low_memory=False)

    req = ["GAME_ID", "EVENTNUM", "PERIOD", "PCTIMESTRING"]
    miss = [c for c in req if c not in pbp.columns]
    if miss:
        raise ValueError(f"PBP missing required columns: {miss}")

    if "SCORE" not in pbp.columns and "SCOREMARGIN" not in pbp.columns:
        raise ValueError("PBP must have SCORE or SCOREMARGIN")

    pbp = pbp.copy()
    pbp["GAME_ID"] = normalize_game_id(pbp["GAME_ID"])
    pbp["EVENTNUM"] = pd.to_numeric(pbp["EVENTNUM"], errors="coerce")
    pbp["PERIOD"] = pd.to_numeric(pbp["PERIOD"], errors="coerce")
    pbp = pbp[pbp["GAME_ID"].notna() & pbp["EVENTNUM"].notna() & pbp["PERIOD"].notna()].copy()
    pbp["EVENTNUM"] = pbp["EVENTNUM"].astype(int)
    pbp["PERIOD"] = pbp["PERIOD"].astype(int)

    pbp["TIME_LEFT_SEC"] = pbp["PCTIMESTRING"].apply(parse_time_to_seconds_any)
    return pbp


def derive_score_diffs(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.sort_values(["GAME_ID", "EVENTNUM"], kind="mergesort").copy()

    if "SCOREMARGIN" in pbp.columns:
        print("  -> Using SCOREMARGIN")
        s = pbp["SCOREMARGIN"].astype("string").str.strip()
        s = s.replace({"TIE": "0", "": pd.NA, "None": pd.NA, "nan": pd.NA})
        pbp["score_diff_after_event"] = pd.to_numeric(s, errors="coerce")
        pbp["score_diff_after_event"] = pbp.groupby("GAME_ID")["score_diff_after_event"].ffill().fillna(0)
        pbp["score_diff_before_event"] = pbp.groupby("GAME_ID")["score_diff_after_event"].shift(1).fillna(0)
    else:
        print("  -> WARNING: SCOREMARGIN not found; parsing SCORE (away-home)")

        def parse_score(x: str) -> Tuple[float, float]:
            if pd.isna(x):
                return np.nan, np.nan
            try:
                a, b = str(x).split("-", 1)
                # nbastats SCORE is visitor-home
                away = float(a.strip())
                home = float(b.strip())
                return home, away
            except Exception:
                return np.nan, np.nan

        ha = pbp["SCORE"].apply(parse_score)
        pbp["home_score"] = [t[0] for t in ha]
        pbp["away_score"] = [t[1] for t in ha]
        pbp["home_score"] = pbp.groupby("GAME_ID")["home_score"].ffill().fillna(0)
        pbp["away_score"] = pbp.groupby("GAME_ID")["away_score"].ffill().fillna(0)
        pbp["score_diff_after_event"] = pbp["home_score"] - pbp["away_score"]
        pbp["score_diff_before_event"] = pbp.groupby("GAME_ID")["score_diff_after_event"].shift(1).fillna(0)

    # Explicit project convention: *_home means (home - away)
    pbp["score_diff_after_event_home"] = pbp["score_diff_after_event"]
    pbp["score_diff_before_event_home"] = pbp["score_diff_before_event"]
    return pbp


def _extract_timeout_used_from_desc(desc: pd.Series) -> pd.Series:
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


def build_timeout_remaining_map_from_pbp(
    pbp: pd.DataFrame,
    total_timeouts_per_team: int = 7,
) -> pd.DataFrame:
    evmsg_col = pick_col(pbp, "EVENTMSGTYPE", "EVENT_MSG_TYPE")
    required = {"GAME_ID", "EVENTNUM", "HOMEDESCRIPTION", "VISITORDESCRIPTION"}
    if evmsg_col is None or not required.issubset(set(pbp.columns)):
        return pd.DataFrame(columns=["GAME_ID", "EVENTNUM", "home_timeouts_left", "visitor_timeouts_left"])

    ev = pbp[["GAME_ID", "EVENTNUM", evmsg_col, "HOMEDESCRIPTION", "VISITORDESCRIPTION"]].copy()
    if "NEUTRALDESCRIPTION" in pbp.columns:
        ev["NEUTRALDESCRIPTION"] = pbp["NEUTRALDESCRIPTION"]
    else:
        ev["NEUTRALDESCRIPTION"] = ""

    ev["GAME_ID"] = normalize_game_id(ev["GAME_ID"])
    ev["EVENTNUM"] = pd.to_numeric(ev["EVENTNUM"], errors="coerce")
    ev[evmsg_col] = pd.to_numeric(ev[evmsg_col], errors="coerce")
    ev = ev.dropna(subset=["GAME_ID", "EVENTNUM"]).copy()
    ev["EVENTNUM"] = ev["EVENTNUM"].astype(int)
    ev = ev.sort_values(["GAME_ID", "EVENTNUM"], kind="mergesort")

    is_to = ev[evmsg_col] == 9
    home_desc = ev["HOMEDESCRIPTION"].fillna("").astype(str)
    vis_desc = ev["VISITORDESCRIPTION"].fillna("").astype(str)
    neu_desc = ev["NEUTRALDESCRIPTION"].fillna("").astype(str)

    home_team_to = is_to & home_desc.str.contains("timeout", case=False, na=False)
    vis_team_to = is_to & vis_desc.str.contains("timeout", case=False, na=False)
    official_to = is_to & (
        home_desc.str.contains("official", case=False, na=False)
        | vis_desc.str.contains("official", case=False, na=False)
        | neu_desc.str.contains("official", case=False, na=False)
    )
    home_team_to = home_team_to & ~official_to
    vis_team_to = vis_team_to & ~official_to

    home_used = pd.Series(np.nan, index=ev.index, dtype=float)
    vis_used = pd.Series(np.nan, index=ev.index, dtype=float)
    home_used.loc[home_team_to] = _extract_timeout_used_from_desc(home_desc.loc[home_team_to])
    vis_used.loc[vis_team_to] = _extract_timeout_used_from_desc(vis_desc.loc[vis_team_to])

    home_used_fallback = home_team_to.groupby(ev["GAME_ID"]).cumsum().astype(float)
    vis_used_fallback = vis_team_to.groupby(ev["GAME_ID"]).cumsum().astype(float)
    home_used.loc[home_team_to & home_used.isna()] = home_used_fallback.loc[home_team_to & home_used.isna()]
    vis_used.loc[vis_team_to & vis_used.isna()] = vis_used_fallback.loc[vis_team_to & vis_used.isna()]

    ev["home_timeouts_used_after"] = home_used.groupby(ev["GAME_ID"]).ffill().fillna(0.0)
    ev["visitor_timeouts_used_after"] = vis_used.groupby(ev["GAME_ID"]).ffill().fillna(0.0)
    ev["home_timeouts_used_after"] = ev.groupby("GAME_ID")["home_timeouts_used_after"].cummax()
    ev["visitor_timeouts_used_after"] = ev.groupby("GAME_ID")["visitor_timeouts_used_after"].cummax()

    ev["home_timeouts_used_before"] = ev.groupby("GAME_ID")["home_timeouts_used_after"].shift(1).fillna(0.0)
    ev["visitor_timeouts_used_before"] = ev.groupby("GAME_ID")["visitor_timeouts_used_after"].shift(1).fillna(0.0)

    ev["home_timeouts_left"] = (float(total_timeouts_per_team) - ev["home_timeouts_used_before"]).clip(lower=0.0)
    ev["visitor_timeouts_left"] = (float(total_timeouts_per_team) - ev["visitor_timeouts_used_before"]).clip(lower=0.0)
    return ev[["GAME_ID", "EVENTNUM", "home_timeouts_left", "visitor_timeouts_left"]].copy()


def load_possessions(season: int, seasontype: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"poss_{season}_{seasontype}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Poss parquet not found: {path}")

    print(f"  -> Loading possessions: {path}")
    poss = pd.read_parquet(path)

    game_col = pick_col(poss, "GAMEID", "GAME_ID", "game_id")
    per_col = pick_col(poss, "period", "PERIOD")
    st_col = pick_col(poss, "STARTTIME", "starttime", "START_TIME", "start_time")
    en_col = pick_col(poss, "ENDTIME", "endtime", "END_TIME", "end_time")

    miss = [k for k, v in [("GAMEID", game_col), ("period", per_col), ("STARTTIME", st_col), ("ENDTIME", en_col)] if v is None]
    if miss:
        raise ValueError(f"poss parquet missing columns {miss}. Available: {list(poss.columns)}")

    poss = poss.rename(columns={
        game_col: "GAMEID",
        per_col: "period",
        st_col: "STARTTIME",
        en_col: "ENDTIME",
    }).copy()

    poss["GAMEID"] = normalize_game_id(poss["GAMEID"])
    poss["period"] = pd.to_numeric(poss["period"], errors="coerce")
    poss = poss[poss["GAMEID"].notna() & poss["period"].notna()].copy()
    poss["period"] = poss["period"].astype(int)

    poss["STARTTIME_SEC"] = poss["STARTTIME"].apply(parse_time_to_seconds_any)
    poss["ENDTIME_SEC"] = poss["ENDTIME"].apply(parse_time_to_seconds_any)

    poss = poss.reset_index(drop=True)
    poss["poss_id"] = np.arange(len(poss), dtype="int64")

    # optional: offense team id if present
    off_col = pick_col(poss, "OFF_TEAM_ID", "OFFTEAM", "offense_team_id", "POSSESSION_TEAM_ID", "possession_team_id")
    if off_col:
        poss = poss.rename(columns={off_col: "off_team_id"})
        poss["off_team_id"] = pd.to_numeric(poss["off_team_id"], errors="coerce")
    else:
        poss["off_team_id"] = np.nan

    # optional: possession start type context (from pbpstats STARTTYPE)
    start_type_col = pick_col(poss, "STARTTYPE", "start_type", "START_TYPE")
    if start_type_col:
        poss["start_type"] = poss[start_type_col].astype("string").str.strip()
        poss.loc[poss["start_type"] == "", "start_type"] = pd.NA
    else:
        poss["start_type"] = pd.NA

    # chronological order per game -> poss_seq (do NOT change poss_id)
    tmp = poss[["poss_id", "GAMEID", "period", "STARTTIME_SEC"]].copy()
    tmp = tmp.dropna(subset=["STARTTIME_SEC"]).copy()
    tmp["start_elapsed"] = tmp.apply(lambda r: period_length_seconds(int(r["period"])) - float(r["STARTTIME_SEC"]), axis=1)
    tmp = tmp.sort_values(["GAMEID", "period", "start_elapsed", "poss_id"], kind="mergesort").reset_index(drop=True)
    tmp["poss_seq"] = tmp.groupby("GAMEID").cumcount().astype("int64")
    poss = poss.merge(tmp[["poss_id", "poss_seq"]], on="poss_id", how="left")
    # fallback if STARTTIME missing -> still assign a poss_seq via poss_id order within game
    if poss["poss_seq"].isna().any():
        fb = poss.sort_values(["GAMEID", "period", "poss_id"], kind="mergesort").copy()
        fb["poss_seq_fb"] = fb.groupby("GAMEID").cumcount().astype("int64")
        poss = poss.merge(fb[["poss_id", "poss_seq_fb"]], on="poss_id", how="left")
        poss["poss_seq"] = poss["poss_seq"].fillna(poss["poss_seq_fb"]).astype("int64")
        poss = poss.drop(columns=["poss_seq_fb"], errors="ignore")
    else:
        poss["poss_seq"] = poss["poss_seq"].astype("int64")

    return poss


def load_games(season: int, seasontype: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"games_{season}_{seasontype}.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    gid_col = pick_col(df, "game_id", "GAME_ID")
    if gid_col is None:
        return pd.DataFrame()

    df = df.copy()
    df["game_id"] = normalize_game_id(df[gid_col])

    # final_home_win
    home_pts = pick_col(df, "home_pts_final", "HOME_PTS")
    away_pts = pick_col(df, "away_pts_final", "AWAY_PTS")
    if home_pts and away_pts:
        df["final_home_win"] = (pd.to_numeric(df[home_pts], errors="coerce") > pd.to_numeric(df[away_pts], errors="coerce")).astype(int)
    elif "final_home_win" in df.columns:
        df["final_home_win"] = pd.to_numeric(df["final_home_win"], errors="coerce")
    else:
        df["final_home_win"] = np.nan

    # team ids (optional)
    h_col = pick_col(df, "home_team_id", "HOME_TEAM_ID", "HOME_TEAMID")
    a_col = pick_col(df, "away_team_id", "AWAY_TEAM_ID", "VISITOR_TEAM_ID", "VISITOR_TEAMID")
    if h_col:
        df["home_team_id"] = pd.to_numeric(df[h_col], errors="coerce")
    else:
        df["home_team_id"] = np.nan
    if a_col:
        df["away_team_id"] = pd.to_numeric(df[a_col], errors="coerce")
    else:
        df["away_team_id"] = np.nan

    out_cols = ["game_id", "final_home_win", "home_team_id", "away_team_id"]
    if "game_date" in df.columns:
        out_cols.append("game_date")
    # Keep all Elo-related columns (e.g., base + k-specific variants like *_k20/*_k40).
    elo_cols = sorted([c for c in df.columns if c.startswith("elo_")])
    for c in elo_cols:
        if c not in out_cols:
            out_cols.append(c)
    return df[out_cols].copy()


def build_game_team_map(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame: [game_id, home_team_id, away_team_id]
    Priority:
      1) games parquet columns
      2) pbp columns (HOME_TEAM_ID / VISITOR_TEAM_ID)
      3) pbp HOME/VISITOR descriptions + PLAYER1_TEAM_ID (mode per game)
    """
    def _infer_team_abbrev() -> pd.DataFrame:
        abbr_col = pick_col(pbp, "PLAYER1_TEAM_ABBREVIATION", "PLAYER1_TEAM_ABBREV")
        if not abbr_col or not {"HOMEDESCRIPTION", "VISITORDESCRIPTION"}.issubset(set(pbp.columns)):
            return pd.DataFrame(columns=["game_id", "home_team_abbrev", "away_team_abbrev"])

        tmp = pbp[["GAME_ID", "HOMEDESCRIPTION", "VISITORDESCRIPTION", abbr_col]].copy()
        tmp = tmp.rename(columns={abbr_col: "team_abbrev"})
        tmp["team_abbrev"] = tmp["team_abbrev"].astype("string")

        home_tmp = tmp[tmp["HOMEDESCRIPTION"].notna() & tmp["team_abbrev"].notna()][["GAME_ID", "team_abbrev"]]
        away_tmp = tmp[tmp["VISITORDESCRIPTION"].notna() & tmp["team_abbrev"].notna()][["GAME_ID", "team_abbrev"]]

        if home_tmp.empty or away_tmp.empty:
            return pd.DataFrame(columns=["game_id", "home_team_abbrev", "away_team_abbrev"])

        home_map = home_tmp.groupby("GAME_ID")["team_abbrev"].agg(lambda x: x.value_counts().idxmax())
        away_map = away_tmp.groupby("GAME_ID")["team_abbrev"].agg(lambda x: x.value_counts().idxmax())
        out = (
            pd.DataFrame({"home_team_abbrev": home_map, "away_team_abbrev": away_map})
            .reset_index()
            .rename(columns={"GAME_ID": "game_id"})
        )
        out["game_id"] = normalize_game_id(out["game_id"])
        return out

    # 1) from games
    if games is not None and not games.empty:
        g = games.copy()
        if {"game_id", "home_team_id", "away_team_id"}.issubset(set(g.columns)):
            out = g[["game_id", "home_team_id", "away_team_id"]].copy()
            out["game_id"] = normalize_game_id(out["game_id"])
            out["home_team_id"] = pd.to_numeric(out["home_team_id"], errors="coerce")
            out["away_team_id"] = pd.to_numeric(out["away_team_id"], errors="coerce")
            out = out.dropna(subset=["home_team_id", "away_team_id"])
            if not out.empty:
                abbr = _infer_team_abbrev()
                if not abbr.empty:
                    out = out.merge(abbr, on="game_id", how="left")
                return out.drop_duplicates(subset=["game_id"], keep="first")

    # 2) from pbp
    h_col = pick_col(pbp, "HOME_TEAM_ID", "HOME_TEAMID")
    a_col = pick_col(pbp, "VISITOR_TEAM_ID", "VISITOR_TEAMID", "AWAY_TEAM_ID", "AWAY_TEAMID")
    if h_col and a_col:
        tmp = pbp[["GAME_ID", h_col, a_col]].copy()
        tmp = tmp.dropna(subset=[h_col, a_col]).copy()
        if not tmp.empty:
            tmp["home_team_id"] = pd.to_numeric(tmp[h_col], errors="coerce")
            tmp["away_team_id"] = pd.to_numeric(tmp[a_col], errors="coerce")
            tmp = tmp.dropna(subset=["home_team_id", "away_team_id"])
            out = tmp.groupby("GAME_ID", as_index=False)[["home_team_id", "away_team_id"]].first()
            out = out.rename(columns={"GAME_ID": "game_id"})
            out["game_id"] = normalize_game_id(out["game_id"])
            abbr = _infer_team_abbrev()
            if not abbr.empty:
                out = out.merge(abbr, on="game_id", how="left")
            return out

    # 3) infer from descriptions + PLAYER1_TEAM_ID
    team_col = pick_col(pbp, "PLAYER1_TEAM_ID", "PLAYER1_TEAMID", "TEAM_ID", "TEAMID")
    if team_col and {"HOMEDESCRIPTION", "VISITORDESCRIPTION"}.issubset(set(pbp.columns)):
        tmp = pbp[["GAME_ID", "HOMEDESCRIPTION", "VISITORDESCRIPTION", team_col]].copy()
        tmp = tmp.rename(columns={team_col: "team_id"})
        tmp["team_id"] = pd.to_numeric(tmp["team_id"], errors="coerce")

        home_tmp = tmp[tmp["HOMEDESCRIPTION"].notna() & tmp["team_id"].notna()][["GAME_ID", "team_id"]]
        away_tmp = tmp[tmp["VISITORDESCRIPTION"].notna() & tmp["team_id"].notna()][["GAME_ID", "team_id"]]

        if not home_tmp.empty and not away_tmp.empty:
            home_map = home_tmp.groupby("GAME_ID")["team_id"].agg(lambda x: x.value_counts().idxmax())
            away_map = away_tmp.groupby("GAME_ID")["team_id"].agg(lambda x: x.value_counts().idxmax())

            out = (
                pd.DataFrame({"home_team_id": home_map, "away_team_id": away_map})
                .reset_index()
                .rename(columns={"GAME_ID": "game_id"})
            )
            out["game_id"] = normalize_game_id(out["game_id"])
            out = out.dropna(subset=["home_team_id", "away_team_id"])
            if not out.empty:
                abbr = _infer_team_abbrev()
                if not abbr.empty:
                    out = out.merge(abbr, on="game_id", how="left")
                return out

    abbr = _infer_team_abbrev()
    if not abbr.empty:
        return abbr
    return pd.DataFrame(columns=["game_id", "home_team_id", "away_team_id", "home_team_abbrev", "away_team_abbrev"])


# ----------------------------
# Event -> possession mapping (time-based, within game+period)
# ----------------------------
def build_event_poss_map(pbp: pd.DataFrame, poss: pd.DataFrame, tol: float = 0.5) -> pd.DataFrame:
    """
    Map each PBP event to poss_id / poss_seq using period+time interval matching.
    Returns columns: GAME_ID, EVENTNUM, poss_id, poss_seq
    """
    ev = pbp[pbp["TIME_LEFT_SEC"].notna()].copy()
    common_games = sorted(set(ev["GAME_ID"].unique()) & set(poss["GAMEID"].unique()))
    maps = []

    for gid in common_games:
        poss_g = poss[poss["GAMEID"] == gid]
        ev_g = ev[ev["GAME_ID"] == gid]

        for per in sorted(poss_g["period"].unique()):
            poss_gp = poss_g[poss_g["period"] == per]
            ev_gp = ev_g[ev_g["PERIOD"] == per]
            if poss_gp.empty or ev_gp.empty:
                continue

            L = period_length_seconds(int(per))

            poss_loc = poss_gp[["poss_id", "poss_seq", "STARTTIME_SEC", "ENDTIME_SEC"]].copy()
            poss_loc = poss_loc.dropna(subset=["STARTTIME_SEC", "ENDTIME_SEC"]).copy()
            if poss_loc.empty:
                continue

            poss_loc["start_elapsed"] = L - poss_loc["STARTTIME_SEC"]
            poss_loc["end_elapsed"] = L - poss_loc["ENDTIME_SEC"]
            poss_loc = poss_loc.sort_values("start_elapsed", kind="mergesort").reset_index(drop=True)

            ev_loc = ev_gp[["EVENTNUM", "TIME_LEFT_SEC"]].copy()
            ev_loc["elapsed"] = L - ev_loc["TIME_LEFT_SEC"]
            ev_loc = ev_loc.sort_values(["elapsed", "EVENTNUM"], kind="mergesort").reset_index(drop=True)

            merged = pd.merge_asof(
                ev_loc,
                poss_loc,
                left_on="elapsed",
                right_on="start_elapsed",
                direction="backward",
                # Boundary events at exact next-possession STARTTIME should belong
                # to the previous possession in this pipeline.
                allow_exact_matches=False,
            )

            # in-interval match with ±tol buffer
            ok = (
                merged["poss_id"].notna()
                & (merged["elapsed"] >= merged["start_elapsed"] - tol)
                & (merged["elapsed"] <= merged["end_elapsed"] + tol)
            )
            matched = merged[ok].copy()

            # fallback: if an event doesn't match any interval, pick latest prior
            # possession by start time (tie/edge goes to previous possession).
            unmatched = merged[~ok].copy()
            if not unmatched.empty:
                starts = poss_loc["start_elapsed"].to_numpy()
                poss_ids = poss_loc["poss_id"].to_numpy()
                poss_seqs = poss_loc["poss_seq"].to_numpy()

                elapsed = unmatched["elapsed"].to_numpy()
                choose = np.searchsorted(starts, elapsed, side="left") - 1
                choose = np.clip(choose, 0, len(starts) - 1)

                unmatched["poss_id"] = poss_ids[choose]
                unmatched["poss_seq"] = poss_seqs[choose]

            merged = pd.concat([matched, unmatched], ignore_index=True)
            if merged.empty:
                continue

            out = merged[["EVENTNUM", "poss_id", "poss_seq"]].copy()
            out["GAME_ID"] = gid
            maps.append(out)

    if not maps:
        return pd.DataFrame(columns=["GAME_ID", "EVENTNUM", "poss_id", "poss_seq"])

    mp = pd.concat(maps, ignore_index=True)
    mp = mp.drop_duplicates(subset=["GAME_ID", "EVENTNUM"], keep="last")
    mp["EVENTNUM"] = mp["EVENTNUM"].astype(int)
    mp["poss_id"] = mp["poss_id"].astype("int64")
    mp["poss_seq"] = mp["poss_seq"].astype("int64")
    return mp


# ----------------------------
# Possession-start states (for WP model fit)
# ----------------------------
def build_poss_start_states(
    season: int,
    seasontype: str,
    pbp: pd.DataFrame,
    poss: pd.DataFrame,
    event_poss_map: pd.DataFrame,
    games: pd.DataFrame,
    timeout_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each possession, use earliest mapped event's score_diff_before_event as score_diff_start.
    home_possession: offense team is home (best-effort).
    """
    act_team_col = pick_col(pbp, "PLAYER1_TEAM_ID", "PLAYER1_TEAMID", "TEAM_ID", "TEAMID")
    evmsg_col = pick_col(pbp, "EVENTMSGTYPE", "EVENT_MSG_TYPE")

    keep = ["GAME_ID", "EVENTNUM", "score_diff_before_event"]
    if evmsg_col:
        keep.append(evmsg_col)
    if act_team_col:
        keep.append(act_team_col)
    ev = pbp[keep].copy()
    if evmsg_col:
        ev = ev.rename(columns={evmsg_col: "eventmsgtype"})
        ev["eventmsgtype"] = pd.to_numeric(ev["eventmsgtype"], errors="coerce")
    else:
        ev["eventmsgtype"] = np.nan
    if act_team_col:
        ev = ev.rename(columns={act_team_col: "act_team_id"})
        ev["act_team_id"] = pd.to_numeric(ev["act_team_id"], errors="coerce")
    else:
        ev["act_team_id"] = np.nan

    ev = ev.merge(event_poss_map, how="left", on=["GAME_ID", "EVENTNUM"])
    ev = ev[ev["poss_id"].notna()].copy()
    ev["poss_id"] = ev["poss_id"].astype("int64")

    # earliest event per poss_id
    idx = ev.groupby("poss_id")["EVENTNUM"].idxmin()
    first = ev.loc[idx, ["poss_id", "EVENTNUM", "score_diff_before_event", "act_team_id"]].copy()
    first = first.rename(columns={
        "EVENTNUM": "start_eventnum",
        "score_diff_before_event": "score_diff",
        "act_team_id": "first_act_team_id",
    })

    # shot-sequence diagnostics
    # shot events: 1=made FG, 2=missed FG
    shot_diag = pd.DataFrame(columns=["poss_id", "shot_events_in_poss", "max_shot_sequence_in_poss", "has_shot_sequence_gt1"])
    shot_seq_at_event = pd.DataFrame(columns=["poss_id", "EVENTNUM", "shot_sequence"])
    if "eventmsgtype" in ev.columns:
        shot_ev = ev[ev["eventmsgtype"].isin([1, 2])].copy()
        if not shot_ev.empty:
            shot_ev = shot_ev.sort_values(["GAME_ID", "poss_id", "EVENTNUM"], kind="mergesort")
            shot_ev["shot_sequence"] = shot_ev.groupby(["GAME_ID", "poss_id"], dropna=False).cumcount() + 1
            shot_seq_at_event = shot_ev[["poss_id", "EVENTNUM", "shot_sequence"]].copy()
            shot_diag = (
                shot_ev.groupby("poss_id", as_index=False)
                .agg(
                    shot_events_in_poss=("shot_sequence", "size"),
                    max_shot_sequence_in_poss=("shot_sequence", "max"),
                )
            )
            shot_diag["has_shot_sequence_gt1"] = (shot_diag["max_shot_sequence_in_poss"] > 1).astype(int)
            first = first.merge(
                shot_seq_at_event.rename(columns={"EVENTNUM": "start_eventnum", "shot_sequence": "shot_sequence_at_state"}),
                on=["poss_id", "start_eventnum"],
                how="left",
            )
        else:
            first["shot_sequence_at_state"] = 0
    else:
        first["shot_sequence_at_state"] = 0

    poss_cols = ["GAMEID", "poss_id", "poss_seq", "period", "STARTTIME_SEC", "off_team_id"]
    if "OPPONENT" in poss.columns:
        poss_cols.append("OPPONENT")
    if "STARTSCOREDIFFERENTIAL" in poss.columns:
        poss_cols.append("STARTSCOREDIFFERENTIAL")
    if "start_type" in poss.columns:
        poss_cols.append("start_type")
    poss0 = poss[poss_cols].copy()
    poss0 = poss0.rename(columns={"GAMEID": "game_id", "STARTTIME_SEC": "time_left_sec"})
    poss0 = poss0.merge(first, on="poss_id", how="left")
    poss0 = poss0.merge(shot_diag, on="poss_id", how="left")
    poss0["shot_sequence_at_state"] = pd.to_numeric(poss0.get("shot_sequence_at_state"), errors="coerce").fillna(0).astype(int)
    poss0["after_off_reb_state"] = (poss0["shot_sequence_at_state"] > 1).astype(int)
    poss0["shot_events_in_poss"] = pd.to_numeric(poss0.get("shot_events_in_poss"), errors="coerce").fillna(0).astype(int)
    poss0["max_shot_sequence_in_poss"] = pd.to_numeric(poss0.get("max_shot_sequence_in_poss"), errors="coerce").fillna(0).astype(int)
    poss0["has_shot_sequence_gt1"] = pd.to_numeric(poss0.get("has_shot_sequence_gt1"), errors="coerce").fillna(0).astype(int)
    # fallback: use pbpstats start score diff when pbp event mapping is missing
    if "STARTSCOREDIFFERENTIAL" in poss0.columns:
        poss0["score_diff"] = poss0["score_diff"].fillna(
            pd.to_numeric(poss0["STARTSCOREDIFFERENTIAL"], errors="coerce")
        )

    poss0["time_left_game"] = poss0.apply(lambda r: calculate_time_left_game(r["period"], r["time_left_sec"]), axis=1)
    poss0["OT_flag"] = (poss0["period"] >= 5).astype(int)

    # game-level columns (final_home_win + optional Elo)
    if games is not None and not games.empty:
        game_cols = ["game_id", "final_home_win"]
        elo_cols = sorted([c for c in games.columns if c.startswith("elo_")])
        for c in elo_cols:
            if c not in game_cols:
                game_cols.append(c)
        poss0 = poss0.merge(games[game_cols], how="left", on="game_id")
    else:
        poss0["final_home_win"] = np.nan

    # home_possession (best-effort)
    game_map = build_game_team_map(pbp, games)  # game_id, home_team_id, away_team_id
    poss0 = poss0.merge(game_map, on="game_id", how="left")

    poss0["off_team_id_final"] = poss0["off_team_id"]
    poss0.loc[poss0["off_team_id_final"].isna(), "off_team_id_final"] = poss0["first_act_team_id"]

    poss0["home_possession"] = np.where(
        poss0["off_team_id_final"].notna() & poss0["home_team_id"].notna(),
        (pd.to_numeric(poss0["off_team_id_final"], errors="coerce") == poss0["home_team_id"]).astype(int),
        np.nan,
    )

    # fallback: infer home_possession from OPPONENT + home/away abbreviations
    if "OPPONENT" in poss0.columns and {"home_team_abbrev", "away_team_abbrev"}.issubset(poss0.columns):
        opp = poss0["OPPONENT"].astype("string")
        home_abbr = poss0["home_team_abbrev"].astype("string")
        away_abbr = poss0["away_team_abbrev"].astype("string")
        off_abbr = np.where(opp == home_abbr, away_abbr, np.where(opp == away_abbr, home_abbr, pd.NA))
        need_fill = poss0["home_possession"].isna() & pd.notna(off_abbr)
        poss0.loc[need_fill, "home_possession"] = (off_abbr[need_fill] == home_abbr[need_fill]).astype(int)

    poss0["season"] = season
    poss0["seasontype"] = seasontype
    poss0["state_type"] = "poss_start"
    # By definition, possession-start states are not after offensive rebound.
    poss0["after_off_reb_state"] = 0

    if timeout_map is not None and not timeout_map.empty:
        tm = timeout_map.rename(columns={"EVENTNUM": "start_eventnum"}).copy()
        poss0 = poss0.merge(
            tm[["GAME_ID", "start_eventnum", "home_timeouts_left", "visitor_timeouts_left"]],
            how="left",
            left_on=["game_id", "start_eventnum"],
            right_on=["GAME_ID", "start_eventnum"],
        ).drop(columns=["GAME_ID"], errors="ignore")
    else:
        poss0["home_timeouts_left"] = np.nan
        poss0["visitor_timeouts_left"] = np.nan
    if "start_type" in poss0.columns:
        poss0["start_type"] = poss0["start_type"].astype("string").str.strip()
        poss0.loc[poss0["start_type"] == "", "start_type"] = pd.NA
    else:
        poss0["start_type"] = pd.NA

    out_cols = [
        "season", "seasontype", "state_type", "game_id", "poss_id", "poss_seq",
        "period", "time_left_sec", "time_left_game", "score_diff", "OT_flag",
        "start_type", "shot_sequence_at_state", "after_off_reb_state",
        "shot_events_in_poss", "max_shot_sequence_in_poss", "has_shot_sequence_gt1",
        "home_possession",
        "home_timeouts_left", "visitor_timeouts_left",
        "final_home_win",
        "start_eventnum",
    ]
    elo_cols = sorted([c for c in poss0.columns if c.startswith("elo_")])
    for c in elo_cols:
        if c not in out_cols:
            out_cols.append(c)
    poss0 = poss0[out_cols].copy()

    poss0["score_diff"] = pd.to_numeric(poss0["score_diff"], errors="coerce")
    poss0 = poss0.dropna(subset=["time_left_sec", "time_left_game", "score_diff"])
    return poss0


# ----------------------------
# Shot decision states for outcome C
# ----------------------------
def build_shot_decision_states(
    season: int,
    seasontype: str,
    pbp: pd.DataFrame,
    poss_start: pd.DataFrame,
    event_poss_map: pd.DataFrame,
    games: pd.DataFrame,
    timeout_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per shot with:
      - before state: (score_diff_before_event, time_left_game, OT_flag, home_possession)
      - next decision state:
          * off_reb -> rebound event state
          * else -> next poss start state
    """
    evmsg_col = pick_col(pbp, "EVENTMSGTYPE", "EVENT_MSG_TYPE")
    if evmsg_col is None:
        raise ValueError("PBP must contain EVENTMSGTYPE to detect shots and rebounds.")

    # --- Filter PBP for shot events (EVENTMSGTYPE = 1 or 2) ---
    # 1 = made field goal, 2 = missed field goal
    shot_events = pbp[pbp[evmsg_col].isin([1, 2])].copy()

    if shot_events.empty:
        print(f"[WARN] No shot events found in PBP for season {season}")
        return pd.DataFrame()

    # --- team id candidates in pbp ---
    pbp_team_col = pick_col(pbp, "PLAYER1_TEAM_ID", "PLAYER1_TEAMID", "TEAM_ID", "TEAMID")

    # ---- Build shot DataFrame from PBP ----
    sh = shot_events[["GAME_ID", "EVENTNUM", "PERIOD", "TIME_LEFT_SEC", "score_diff_before_event", evmsg_col]].copy()
    sh = sh.rename(columns={"EVENTNUM": "GAME_EVENT_ID"})

    # Add team info if available
    if pbp_team_col:
        sh["shot_team_id"] = pd.to_numeric(shot_events[pbp_team_col], errors="coerce")
    else:
        sh["shot_team_id"] = np.nan

    # Determine shot_made from EVENTMSGTYPE (1=made, 2=missed)
    sh["shot_made"] = (shot_events[evmsg_col] == 1).astype(int)

    # Ensure numeric types
    sh["PERIOD"] = pd.to_numeric(sh["PERIOD"], errors="coerce")
    sh["TIME_LEFT_SEC"] = pd.to_numeric(sh["TIME_LEFT_SEC"], errors="coerce")
    sh["score_diff_before_event"] = pd.to_numeric(sh["score_diff_before_event"], errors="coerce")

    # attach poss_id/poss_seq
    sh = sh.merge(
        event_poss_map.rename(columns={"EVENTNUM": "GAME_EVENT_ID"}),
        how="left",
        on=["GAME_ID", "GAME_EVENT_ID"],
    )

    # basic filters
    sh = sh.dropna(subset=["PERIOD", "TIME_LEFT_SEC", "score_diff_before_event"]).copy()
    sh["PERIOD"] = sh["PERIOD"].astype(int)

    sh["before_OT_flag"] = (sh["PERIOD"] >= 5).astype(int)
    sh["before_time_left_game"] = sh.apply(lambda r: calculate_time_left_game(r["PERIOD"], r["TIME_LEFT_SEC"]), axis=1)

    # shot_team_id and shot_made are already set from PBP above
    # No need for additional processing

    # shot_sequence within possession
    sh = sh.sort_values(["GAME_ID", "poss_id", "GAME_EVENT_ID"], kind="mergesort").copy()
    sh["shot_sequence"] = sh.groupby(["GAME_ID", "poss_id"], dropna=False).cumcount() + 1

    # before_start_type: possession-start context on the current possession
    if "start_type" in poss_start.columns:
        curr_poss = poss_start[["game_id", "poss_seq", "start_type"]].copy().rename(
            columns={"poss_seq": "poss_seq_curr", "start_type": "before_start_type"}
        )
        sh = sh.merge(
            curr_poss,
            how="left",
            left_on=["GAME_ID", "poss_seq"],
            right_on=["game_id", "poss_seq_curr"],
        ).drop(columns=["game_id", "poss_seq_curr"], errors="ignore")
        sh["before_start_type"] = sh["before_start_type"].astype("string").str.strip()
        sh.loc[sh["before_start_type"] == "", "before_start_type"] = pd.NA
    else:
        sh["before_start_type"] = pd.NA

    # ---- game home/away team ids + before_home_possession ----
    game_map = build_game_team_map(pbp, games)  # game_id, home_team_id, away_team_id
    sh = sh.merge(game_map, how="left", left_on="GAME_ID", right_on="game_id").drop(columns=["game_id"], errors="ignore")

    sh["before_home_possession"] = np.where(
        sh["shot_team_id"].notna() & sh["home_team_id"].notna(),
        (pd.to_numeric(sh["shot_team_id"], errors="coerce") == sh["home_team_id"]).astype(int),
        np.nan,
    )

    if timeout_map is not None and not timeout_map.empty:
        sh = sh.merge(
            timeout_map[["GAME_ID", "EVENTNUM", "home_timeouts_left", "visitor_timeouts_left"]].rename(
                columns={
                    "EVENTNUM": "GAME_EVENT_ID",
                    "home_timeouts_left": "before_home_timeouts_left",
                    "visitor_timeouts_left": "before_visitor_timeouts_left",
                }
            ),
            how="left",
            on=["GAME_ID", "GAME_EVENT_ID"],
        )
    else:
        sh["before_home_timeouts_left"] = np.nan
        sh["before_visitor_timeouts_left"] = np.nan


    def match_next_rebound_per_poss(miss: pd.DataFrame, reb2: pd.DataFrame) -> pd.DataFrame:
        """
        For each row in miss, find the first rebound row in reb2 with
        same (GAME_ID, poss_id) and REB_EVENTNUM > GAME_EVENT_ID.
        This avoids pandas.merge_asof sorted-key pitfalls.
        """
        # ensure minimal cols exist
        need_m = ["GAME_ID", "poss_id", "GAME_EVENT_ID"]
        need_r = ["GAME_ID", "poss_id", "REB_EVENTNUM"]
        for c in need_m:
            if c not in miss.columns:
                raise KeyError(f"miss missing {c}")
        for c in need_r:
            if c not in reb2.columns:
                raise KeyError(f"reb2 missing {c}")

        miss = miss.copy()
        reb2 = reb2.copy()

        # enforce dtypes + sorting (important)
        miss["GAME_ID"] = miss["GAME_ID"].astype("string")
        reb2["GAME_ID"] = reb2["GAME_ID"].astype("string")
        miss["poss_id"] = pd.to_numeric(miss["poss_id"], errors="coerce").astype("int64")
        reb2["poss_id"] = pd.to_numeric(reb2["poss_id"], errors="coerce").astype("int64")
        miss["GAME_EVENT_ID"] = pd.to_numeric(miss["GAME_EVENT_ID"], errors="coerce").astype("int64")
        reb2["REB_EVENTNUM"] = pd.to_numeric(reb2["REB_EVENTNUM"], errors="coerce").astype("int64")

        miss = miss.sort_values(["GAME_ID", "poss_id", "GAME_EVENT_ID"], kind="mergesort").reset_index(drop=True)
        reb2 = reb2.sort_values(["GAME_ID", "poss_id", "REB_EVENTNUM"], kind="mergesort").reset_index(drop=True)

        # columns to carry from rebound
        reb_cols = [
            "REB_EVENTNUM",
            "PERIOD_reb_src",
            "TIME_LEFT_SEC_reb_src",
            "time_left_game_reb",
            "score_diff_before_event_reb_src",
            "OT_flag_reb",
            "reb_team_id",
            "home_timeouts_left_reb",
            "visitor_timeouts_left_reb",
        ]
        for c in reb_cols:
            if c not in reb2.columns:
                # allow missing optional cols, but create them
                reb2[c] = np.nan
            miss[c] = np.nan

        miss_groups = miss.groupby(["GAME_ID", "poss_id"], sort=False).indices
        reb_groups = reb2.groupby(["GAME_ID", "poss_id"], sort=False).indices

        for key, idx_m in miss_groups.items():
            idx_r = reb_groups.get(key)
            if idx_r is None:
                continue

            idx_m = np.asarray(idx_m, dtype=np.int64)
            idx_r = np.asarray(idx_r, dtype=np.int64)

            r_events = reb2.loc[idx_r, "REB_EVENTNUM"].to_numpy(np.int64)
            m_events = miss.loc[idx_m, "GAME_EVENT_ID"].to_numpy(np.int64)

            pos = np.searchsorted(r_events, m_events, side="right")  # strictly greater
            hit = pos < len(r_events)
            if not hit.any():
                continue

            src = idx_r[pos[hit]]
            dst = idx_m[hit]

            for c in reb_cols:
                miss.loc[dst, c] = reb2.loc[src, c].to_numpy()

        return miss

    # ---- Rebounds (EVENTMSGTYPE==4) ----
    reb = pbp[pbp[evmsg_col] == 4].copy()

    reb_keep = ["GAME_ID", "EVENTNUM", "PERIOD", "TIME_LEFT_SEC", "score_diff_before_event"]
    reb_team_col = pick_col(reb, "PLAYER1_TEAM_ID", "PLAYER1_TEAMID", "TEAM_ID", "TEAMID")
    if reb_team_col:
        reb_keep.append(reb_team_col)

    reb = reb[reb_keep].copy().rename(columns={
        "EVENTNUM": "REB_EVENTNUM",
        "PERIOD": "PERIOD_reb_src",
        "TIME_LEFT_SEC": "TIME_LEFT_SEC_reb_src",
        "score_diff_before_event": "score_diff_before_event_reb_src",
    })
    if reb_team_col:
        reb = reb.rename(columns={reb_team_col: "reb_team_id"})
        reb["reb_team_id"] = pd.to_numeric(reb["reb_team_id"], errors="coerce")
    else:
        reb["reb_team_id"] = np.nan

    # attach poss_id for rebound
    reb = reb.merge(
        event_poss_map,
        how="left",
        left_on=["GAME_ID", "REB_EVENTNUM"],
        right_on=["GAME_ID", "EVENTNUM"],
    ).drop(columns=["EVENTNUM"], errors="ignore")
    if timeout_map is not None and not timeout_map.empty:
        reb = reb.merge(
            timeout_map[["GAME_ID", "EVENTNUM", "home_timeouts_left", "visitor_timeouts_left"]].rename(
                columns={
                    "EVENTNUM": "REB_EVENTNUM",
                    "home_timeouts_left": "home_timeouts_left_reb",
                    "visitor_timeouts_left": "visitor_timeouts_left_reb",
                }
            ),
            how="left",
            on=["GAME_ID", "REB_EVENTNUM"],
        )
    else:
        reb["home_timeouts_left_reb"] = np.nan
        reb["visitor_timeouts_left_reb"] = np.nan

    reb = reb.dropna(subset=["poss_id", "REB_EVENTNUM"]).copy()
    reb["poss_id"] = pd.to_numeric(reb["poss_id"], errors="coerce").astype("int64")
    reb["REB_EVENTNUM"] = pd.to_numeric(reb["REB_EVENTNUM"], errors="coerce").astype("int64")
    reb["PERIOD_reb_src"] = pd.to_numeric(reb["PERIOD_reb_src"], errors="coerce")
    reb["TIME_LEFT_SEC_reb_src"] = pd.to_numeric(reb["TIME_LEFT_SEC_reb_src"], errors="coerce")
    reb["score_diff_before_event_reb_src"] = pd.to_numeric(reb["score_diff_before_event_reb_src"], errors="coerce")

    reb["OT_flag_reb"] = (reb["PERIOD_reb_src"] >= 5).astype(int)
    reb["time_left_game_reb"] = reb.apply(lambda r: calculate_time_left_game(r["PERIOD_reb_src"], r["TIME_LEFT_SEC_reb_src"]), axis=1)

    # ---- Match each MISSED shot to the NEXT rebound in same (GAME_ID, poss_id) ----
    miss = sh[sh["shot_made"] == 0].copy()
    miss = miss.dropna(subset=["GAME_ID", "poss_id", "GAME_EVENT_ID"]).copy()

    # harden dtypes + enforce full sorting for merge_asof
    miss["GAME_ID"] = miss["GAME_ID"].astype("string")
    miss["poss_id"] = pd.to_numeric(miss["poss_id"], errors="coerce").astype("int64")
    miss["GAME_EVENT_ID"] = pd.to_numeric(miss["GAME_EVENT_ID"], errors="coerce").astype("int64")

    reb2 = reb.dropna(subset=["GAME_ID", "poss_id", "REB_EVENTNUM"]).copy()
    reb2["GAME_ID"] = reb2["GAME_ID"].astype("string")
    reb2["poss_id"] = pd.to_numeric(reb2["poss_id"], errors="coerce").astype("int64")
    reb2["REB_EVENTNUM"] = pd.to_numeric(reb2["REB_EVENTNUM"], errors="coerce").astype("int64")

    miss = miss.sort_values(["GAME_ID", "poss_id", "GAME_EVENT_ID"], kind="mergesort").reset_index(drop=True)
    reb2 = reb2.sort_values(["GAME_ID", "poss_id", "REB_EVENTNUM"], kind="mergesort").reset_index(drop=True)

    miss2 = match_next_rebound_per_poss(miss, reb2)

    # offensive rebound?
    miss2["shot_team_id"] = pd.to_numeric(miss2["shot_team_id"], errors="coerce")
    miss2["reb_team_id"] = pd.to_numeric(miss2["reb_team_id"], errors="coerce")
    miss2["is_off_reb"] = (
        miss2["reb_team_id"].notna()
        & miss2["shot_team_id"].notna()
        & (miss2["reb_team_id"] == miss2["shot_team_id"])
    ).astype(int)

    # bring rebound match info back to sh
    sh = sh.merge(
        miss2[[
            "GAME_ID", "GAME_EVENT_ID",
            "REB_EVENTNUM", "is_off_reb",
            "PERIOD_reb_src", "TIME_LEFT_SEC_reb_src", "time_left_game_reb",
            "score_diff_before_event_reb_src", "OT_flag_reb", "reb_team_id",
            "home_timeouts_left_reb", "visitor_timeouts_left_reb",
        ]],
        how="left",
        on=["GAME_ID", "GAME_EVENT_ID"],
    )
    sh["is_off_reb"] = pd.to_numeric(sh["is_off_reb"], errors="coerce").fillna(0).astype(int)
    sh["prev_shot_is_off_reb"] = (
        sh.groupby(["GAME_ID", "poss_id"], dropna=False)["is_off_reb"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )
    sh["after_off_reb"] = (
        (pd.to_numeric(sh["shot_sequence"], errors="coerce") > 1)
        & (sh["prev_shot_is_off_reb"] == 1)
    ).astype(int)

    # ---- Next possession start state ----
    # poss_start includes: game_id, poss_seq, ... , home_possession
    poss_key_cols = [
        "game_id", "poss_seq",
        "period", "time_left_sec", "time_left_game", "score_diff", "OT_flag",
        "home_possession",
        "home_timeouts_left", "visitor_timeouts_left",
        "final_home_win",
    ]
    if "start_type" in poss_start.columns:
        poss_key_cols.append("start_type")
    poss_key = poss_start[poss_key_cols].copy()

    # attach poss_seq to sh (from event_poss_map already exists, but ensure)
    sh["poss_seq"] = pd.to_numeric(sh["poss_seq"], errors="coerce")
    sh["next_poss_seq"] = sh["poss_seq"] + 1

    next_poss = poss_key.rename(columns={
        "poss_seq": "next_poss_seq",
        "period": "next_period_poss",
        "time_left_sec": "next_time_left_sec_poss",
        "time_left_game": "next_time_left_game_poss",
        "score_diff": "next_score_diff_poss",
        "OT_flag": "next_OT_flag_poss",
        "home_possession": "next_home_possession_poss",
        "home_timeouts_left": "next_home_timeouts_left_poss",
        "visitor_timeouts_left": "next_visitor_timeouts_left_poss",
        "start_type": "next_start_type_poss",
        "final_home_win": "final_home_win",
    })

    sh = sh.merge(
        next_poss,
        how="left",
        left_on=["GAME_ID", "next_poss_seq"],
        right_on=["game_id", "next_poss_seq"],
    ).drop(columns=["game_id"], errors="ignore")

    # ---- Choose next decision state ----
    use_reb = (sh["shot_made"] == 0) & (sh["is_off_reb"] == 1)
    # off_reb rows have a valid next decision state (rebound state), so they are non-terminal.
    sh["next_is_terminal"] = ((~use_reb) & sh["next_score_diff_poss"].isna()).astype(int)

    sh["next_type"] = np.where(use_reb, "off_reb", "next_poss_start")
    sh.loc[sh["next_is_terminal"] == 1, "next_type"] = "terminal"

    sh["next_period"] = np.where(use_reb, sh["PERIOD_reb_src"], sh["next_period_poss"])
    sh["next_time_left_sec"] = np.where(use_reb, sh["TIME_LEFT_SEC_reb_src"], sh["next_time_left_sec_poss"])
    sh["next_time_left_game"] = np.where(use_reb, sh["time_left_game_reb"], sh["next_time_left_game_poss"])
    sh["next_score_diff"] = np.where(use_reb, sh["score_diff_before_event_reb_src"], sh["next_score_diff_poss"])
    sh["next_OT_flag"] = np.where(use_reb, sh["OT_flag_reb"], sh["next_OT_flag_poss"])
    if "next_start_type_poss" not in sh.columns:
        sh["next_start_type_poss"] = pd.NA
    sh["next_start_type"] = np.where(use_reb, sh["before_start_type"], sh["next_start_type_poss"])

    # ---- next_home_possession ----
    sh["next_home_possession"] = np.nan

    # off_reb: rebound team is offense
    sh.loc[use_reb, "next_home_possession"] = np.where(
        sh.loc[use_reb, "reb_team_id"].notna() & sh.loc[use_reb, "home_team_id"].notna(),
        (sh.loc[use_reb, "reb_team_id"] == sh.loc[use_reb, "home_team_id"]).astype(int),
        np.nan,
    )

    # next poss start: prefer poss_start's home_possession; fallback to flip
    mask_nextpos = (~use_reb) & (sh["next_is_terminal"] == 0)
    sh.loc[mask_nextpos, "next_home_possession"] = sh.loc[mask_nextpos, "next_home_possession_poss"]

    mask_need_flip = mask_nextpos & sh["next_home_possession"].isna() & sh["before_home_possession"].notna()
    sh.loc[mask_need_flip, "next_home_possession"] = 1 - sh.loc[mask_need_flip, "before_home_possession"]

    sh["next_home_timeouts_left"] = np.where(
        use_reb,
        sh["home_timeouts_left_reb"],
        sh["next_home_timeouts_left_poss"],
    )
    sh["next_visitor_timeouts_left"] = np.where(
        use_reb,
        sh["visitor_timeouts_left_reb"],
        sh["next_visitor_timeouts_left_poss"],
    )

    # before fields
    sh["before_state_type"] = "shot_state"
    sh["before_period"] = sh["PERIOD"]
    sh["before_time_left_sec"] = sh["TIME_LEFT_SEC"]
    sh["before_time_left_game"] = sh["before_time_left_game"]
    sh["before_score_diff"] = sh["score_diff_before_event"]

    # attach/fill final_home_win from games (for terminal rows, etc.)
    if games is not None and not games.empty:
        sh = sh.merge(
            games[["game_id", "final_home_win"]],
            how="left",
            left_on="GAME_ID",
            right_on="game_id",
            suffixes=("", "_game"),
        ).drop(columns=["game_id"], errors="ignore")
        if "final_home_win_game" in sh.columns:
            if "final_home_win" in sh.columns:
                sh["final_home_win"] = sh["final_home_win"].fillna(sh["final_home_win_game"])
            else:
                sh["final_home_win"] = sh["final_home_win_game"]
            sh = sh.drop(columns=["final_home_win_game"], errors="ignore")
    elif "final_home_win" not in sh.columns:
        sh["final_home_win"] = np.nan

    out = sh[[
        "GAME_ID", "GAME_EVENT_ID",
        "REB_EVENTNUM",
        "poss_id", "poss_seq", "shot_sequence",
        "shot_made",
        "after_off_reb",

        "before_state_type",
        "before_period", "before_time_left_sec", "before_time_left_game", "before_score_diff", "before_OT_flag",
        "before_home_possession",
        "before_home_timeouts_left", "before_visitor_timeouts_left",
        "before_start_type",

        "next_type", "next_is_terminal",
        "next_period", "next_time_left_sec", "next_time_left_game", "next_score_diff", "next_OT_flag",
        "next_home_possession",
        "next_home_timeouts_left", "next_visitor_timeouts_left",
        "next_start_type",

        "final_home_win",
    ]].copy()

    out["season"] = season
    out["seasontype"] = seasontype

    print(f"  -> shots total: {len(out):,}")
    print(f"  -> poss_id coverage: {out['poss_id'].notna().mean()*100:.2f}%")
    print("  -> next_type counts:")
    print(out["next_type"].value_counts(dropna=False))

    return out


# ----------------------------
# Build WP training states (poss_start + raw off-rebound event states)
# ----------------------------
def build_wp_states_for_training(poss_start: pd.DataFrame, shot_states: pd.DataFrame) -> pd.DataFrame:
    """
    Expand WP training states from possession-start rows to a mixed state set:
      - base: possession-start rows
      - added: shot-state rows (same state as shot_decision_states before_*)
      - added: offensive-rebound event states derived from raw PBP alignment
    """
    base = poss_start.copy()
    if shot_states is None or shot_states.empty:
        print("  -> wp_states add rows: 0 (shot_states empty)")
        return base

    # carry per-possession diagnostics from base when available
    diag_cols = [
        "shot_events_in_poss",
        "max_shot_sequence_in_poss",
        "has_shot_sequence_gt1",
    ]
    diag_cols = [c for c in diag_cols if c in base.columns]
    diag_key = pd.DataFrame()
    if diag_cols:
        diag_key = base[["game_id", "poss_id"] + diag_cols].drop_duplicates(subset=["game_id", "poss_id"])

    # carry game-level elo features from base when available
    elo_cols = sorted([c for c in base.columns if c.startswith("elo_")])
    elo_key = pd.DataFrame()
    if elo_cols:
        elo_key = base[["game_id"] + elo_cols].drop_duplicates(subset=["game_id"])

    def _attach_context(add_df: pd.DataFrame) -> pd.DataFrame:
        out_df = add_df.copy()
        if not diag_key.empty:
            out_df = out_df.merge(
                diag_key,
                how="left",
                left_on=["GAME_ID", "poss_id"],
                right_on=["game_id", "poss_id"],
            ).drop(columns=["game_id"], errors="ignore")
        if not elo_key.empty:
            out_df = out_df.merge(
                elo_key,
                how="left",
                left_on="GAME_ID",
                right_on="game_id",
            ).drop(columns=["game_id"], errors="ignore")
        return out_df

    # (A) shot-state rows: one row per shot event
    add_before = shot_states.drop_duplicates(subset=["GAME_ID", "GAME_EVENT_ID"], keep="first").copy()
    add_before = _attach_context(add_before)
    before_seq = pd.to_numeric(add_before.get("shot_sequence"), errors="coerce")
    before_after_or = pd.to_numeric(add_before.get("after_off_reb"), errors="coerce")
    add_before_states = pd.DataFrame({
        "season": add_before["season"],
        "seasontype": add_before["seasontype"],
        "state_type": "shot_state",
        "game_id": add_before["GAME_ID"],
        "poss_id": add_before["poss_id"],
        "poss_seq": add_before["poss_seq"],
        "period": add_before["before_period"],
        "time_left_sec": add_before["before_time_left_sec"],
        "time_left_game": add_before["before_time_left_game"],
        "score_diff": add_before["before_score_diff"],
        "OT_flag": add_before["before_OT_flag"],
        "start_type": add_before["before_start_type"],
        "shot_sequence_at_state": before_seq.fillna(1),
        "after_off_reb_state": before_after_or.fillna(0),
        "home_possession": add_before["before_home_possession"],
        "home_timeouts_left": add_before["before_home_timeouts_left"],
        "visitor_timeouts_left": add_before["before_visitor_timeouts_left"],
        "final_home_win": add_before["final_home_win"],
        "start_eventnum": add_before["GAME_EVENT_ID"],
    })
    for c in diag_cols + elo_cols:
        add_before_states[c] = add_before[c]

    # (B) raw off-rebound event states: one row per rebound event
    add_or = shot_states[shot_states["next_type"] == "off_reb"].copy()
    if not add_or.empty:
        if "REB_EVENTNUM" in add_or.columns:
            add_or = add_or.drop_duplicates(subset=["GAME_ID", "REB_EVENTNUM"], keep="first")
        else:
            add_or = add_or.drop_duplicates(subset=["GAME_ID", "GAME_EVENT_ID"], keep="first")
        add_or = _attach_context(add_or)

    add_or_states = pd.DataFrame()
    if not add_or.empty:
        add_or_states = pd.DataFrame({
            "season": add_or["season"],
            "seasontype": add_or["seasontype"],
            "state_type": "off_reb",
            "game_id": add_or["GAME_ID"],
            "poss_id": add_or["poss_id"],
            "poss_seq": add_or["poss_seq"],
            "period": add_or["next_period"],
            "time_left_sec": add_or["next_time_left_sec"],
            "time_left_game": add_or["next_time_left_game"],
            "score_diff": add_or["next_score_diff"],
            "OT_flag": add_or["next_OT_flag"],
            "start_type": add_or["next_start_type"],
            "shot_sequence_at_state": 2,
            "after_off_reb_state": 1,
            "home_possession": add_or["next_home_possession"],
            "home_timeouts_left": add_or["next_home_timeouts_left"],
            "visitor_timeouts_left": add_or["next_visitor_timeouts_left"],
            "final_home_win": add_or["final_home_win"],
            "start_eventnum": (
                add_or["REB_EVENTNUM"] if "REB_EVENTNUM" in add_or.columns else add_or["GAME_EVENT_ID"]
            ),
        })
        for c in diag_cols + elo_cols:
            add_or_states[c] = add_or[c]

    add_states = pd.concat([add_before_states, add_or_states], ignore_index=True, sort=False)
    if add_states.empty:
        print("  -> wp_states add rows: 0 (no shot/off_reb rows)")
        return base

    if "shot_events_in_poss" not in add_states.columns:
        add_states["shot_events_in_poss"] = np.nan
    if "max_shot_sequence_in_poss" not in add_states.columns:
        add_states["max_shot_sequence_in_poss"] = add_states["shot_sequence_at_state"]
    if "has_shot_sequence_gt1" not in add_states.columns:
        add_states["has_shot_sequence_gt1"] = (pd.to_numeric(add_states["max_shot_sequence_in_poss"], errors="coerce") > 1).astype(int)

    # align column order to base schema
    for c in base.columns:
        if c not in add_states.columns:
            add_states[c] = np.nan
    add_states = add_states[base.columns]

    # enforce numeric consistency
    for c in ["period", "time_left_sec", "time_left_game", "score_diff", "OT_flag", "home_possession", "shot_sequence_at_state", "after_off_reb_state", "start_eventnum"]:
        if c in add_states.columns:
            add_states[c] = pd.to_numeric(add_states[c], errors="coerce")

    out = pd.concat([base, add_states], ignore_index=True)
    print(f"  -> wp_states add rows (shot_state rows): {len(add_before_states):,}")
    print(f"  -> wp_states add rows (raw off_reb event states): {len(add_or_states):,}")
    print(f"  -> wp_states total rows: {len(out):,}")
    return out


# ----------------------------
# Sanity checks
# ----------------------------
def report_wp_states_sanity(wp_states: pd.DataFrame, season: int, seasontype: str) -> None:
    print(f"  -> Sanity check (wp_states, season={season}, type={seasontype})")
    if wp_states is None or wp_states.empty:
        print("     [WARN] wp_states is empty")
        return

    n = len(wp_states)
    s = pd.to_numeric(wp_states.get("shot_sequence_at_state"), errors="coerce").fillna(0)
    a = pd.to_numeric(wp_states.get("after_off_reb_state"), errors="coerce").fillna(0)

    gt1 = (s > 1).astype(int)
    rate_gt1 = gt1.mean()
    mismatch = (a.astype(int) != gt1).sum()

    print(f"     rows={n:,}")
    print(f"     shot_sequence_at_state > 1: {gt1.sum():,} ({rate_gt1:.4f})")
    print(f"     after_off_reb_state==1: {(a == 1).sum():,} ({(a == 1).mean():.4f})")
    print(f"     mismatch(after_off_reb_state vs shot_sequence_at_state>1): {int(mismatch):,}")

    if "start_type" in wp_states.columns:
        st = wp_states["start_type"].astype("string")
        mask_make = st.str.contains("Make", case=False, na=False)
        mask_miss = st.str.contains("Miss", case=False, na=False)
        if mask_make.any():
            print(f"     rate(shot_sequence_at_state>1 | start_type has Make): {gt1[mask_make].mean():.4f}")
        if mask_miss.any():
            print(f"     rate(shot_sequence_at_state>1 | start_type has Miss): {gt1[mask_miss].mean():.4f}")

    if "max_shot_sequence_in_poss" in wp_states.columns:
        max_seq = pd.to_numeric(wp_states["max_shot_sequence_in_poss"], errors="coerce").fillna(0)
        print(f"     poss with max_shot_sequence_in_poss > 1: {(max_seq > 1).sum():,} ({(max_seq > 1).mean():.4f})")


# ----------------------------
# Season runner
# ----------------------------
def process_one_season(season: int, seasontype: str, tol: float, total_timeouts_per_team: int) -> None:
    print(f"\n=== Season {season} ({seasontype}) ===")

    pbp = load_pbp(season, seasontype)
    pbp = derive_score_diffs(pbp)
    timeout_map = build_timeout_remaining_map_from_pbp(
        pbp,
        total_timeouts_per_team=total_timeouts_per_team,
    )

    poss = load_possessions(season, seasontype)
    games = load_games(season, seasontype)

    event_poss_map = build_event_poss_map(pbp, poss, tol=tol)
    if event_poss_map.empty:
        print("[WARN] event_poss_map is empty; skipping.")
        return

    print(f"  -> event->poss map rows: {len(event_poss_map):,}")

    poss_start = build_poss_start_states(
        season, seasontype, pbp, poss, event_poss_map, games, timeout_map
    )
    shot_states = build_shot_decision_states(
        season, seasontype, pbp, poss_start, event_poss_map, games, timeout_map
    )
    wp_states = build_wp_states_for_training(poss_start, shot_states)
    report_wp_states_sanity(wp_states, season, seasontype)

    poss_out = WP_DIR / f"wp_states_{season}_{seasontype}.csv.gz"
    ensure_regular_output_path(poss_out)
    wp_states.to_csv(poss_out, index=False, compression="gzip")
    print(f"  -> Saved wp states: {poss_out} (rows={len(wp_states):,})")

    shot_out = WP_DIR / f"shot_decision_states_{season}_{seasontype}.csv.gz"
    ensure_regular_output_path(shot_out)
    shot_states.to_csv(shot_out, index=False, compression="gzip")
    print(f"  -> Saved shot decision states: {shot_out} (rows={len(shot_states):,})")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2000)
    p.add_argument("--end-season", type=int, default=2024)
    p.add_argument("--seasontype", type=str, default="rs", choices=["rs"])
    p.add_argument("--tol", type=float, default=0.5, help="seconds tolerance for event-in-possession matching")
    p.add_argument("--total-timeouts-per-team", type=int, default=7, help="Assumed total timeout budget per team per game.")
    return p.parse_args()


def main():
    args = parse_args()
    for season in range(args.start_season, args.end_season + 1):
        try:
            process_one_season(
                season,
                args.seasontype,
                tol=args.tol,
                total_timeouts_per_team=args.total_timeouts_per_team,
            )
        except FileNotFoundError as e:
            print(f"[WARN] {e} -> skipping season {season}")
        except Exception as e:
            print(f"[ERROR] Failed season {season}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
