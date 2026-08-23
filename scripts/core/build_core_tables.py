#!/usr/bin/env python
"""
nba_data (shufinskiy) から落とした raw CSV から、解析用のコアテーブルを構築するスクリプト。

- PBP レベル（1行1イベント）
- Possession レベル
- Shot レベル
- Game レベル
- Team×Season 集約
- Lineup レベル

※ ファイル名パターンや pbpstats の列名は環境で微妙に異なりうるので、
   実際の `data/nba_raw` を `head` / `df.columns` で確認しながらチューニングしてください。
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Literal, List

import pandas as pd
import numpy as np


# ==============================
# 設定
# ==============================

RAW_DIR = Path("data/nba_raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SeasonType = Literal["rs"]


# ==============================
# ユーティリティ
# ==============================

def _first_existing(patterns: list[str]) -> Path | None:
    for pat in patterns:
        candidates = sorted(RAW_DIR.glob(pat))
        if candidates:
            return candidates[0]
    return None


def find_nbastats_file(season: int, seasontype: SeasonType) -> Path:
    """
    nba_on_court.load_nba_data が吐いた nbastats CSV を見つける。

    いくつかの regular-season 命名パターンを順番に試す。
    """
    p = _first_existing([
        f"nbastats_{seasontype}_{season}.csv",
        f"nbastats_{season}_{seasontype}.csv",
        f"nbastats_{season}.csv",
        f"nbastats*{season}*{seasontype}*.csv",
        f"nbastats*{seasontype}*{season}*.csv",
        f"nbastats*{season}*.csv",
    ])
    if p is None:
        raise FileNotFoundError(f"nbastats CSV not found for season={season}, type={seasontype}")
    return p


def find_pbpstats_file(season: int, seasontype: SeasonType) -> Path:
    p = _first_existing([
        f"pbpstats_{seasontype}_{season}.csv",
        f"pbpstats_{season}_{seasontype}.csv",
        f"pbpstats_{season}.csv",
        f"pbpstats*{season}*{seasontype}*.csv",
        f"pbpstats*{seasontype}*{season}*.csv",
        f"pbpstats*{season}*.csv",
    ])
    if p is None:
        raise FileNotFoundError(f"pbpstats CSV not found for season={season}, type={seasontype}")
    return p


def find_shotdetail_file(season: int, seasontype: SeasonType) -> Path:
    p = _first_existing([
        f"shotdetail_{seasontype}_{season}.csv",
        f"shotdetail_{season}_{seasontype}.csv",
        f"shotdetail_{season}.csv",
        f"shotdetail*{season}*{seasontype}*.csv",
        f"shotdetail*{seasontype}*{season}*.csv",
        f"shotdetail*{season}*.csv",
    ])
    if p is None:
        raise FileNotFoundError(f"shotdetail CSV not found for season={season}, type={seasontype}")
    return p


def parse_pctimestring_to_seconds(pct: str | float | int) -> float:
    """
    PCTIMESTRING "MM:SS" → 残り秒数 (int) に変換。
    欠損 / 変な値は NaN を返す。
    """
    if pd.isna(pct):
        return float("nan")
    if isinstance(pct, (int, float)):
        # すでに秒数ならそのまま返す
        return float(pct)
    s = str(pct)
    if ":" not in s:
        return float("nan")
    try:
        mm, ss = s.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return float("nan")


def normalize_scoremargin(sm: str | float | int) -> float:
    """
    SCOREMARGIN 列（"TIE", "5", "-3", "" など）を数値に変換。
    """
    if pd.isna(sm):
        return float("nan")
    s = str(sm).strip()
    if s == "" or s.upper() == "NONE":
        return float("nan")
    if s.upper() == "TIE":
        return 0.0
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_nbastats_score_away_home(score_str: str | float | int) -> tuple[float, float]:
    """
    nbastats の SCORE は "away-home"（visitor-home）。
    戻り値は (home_pts, away_pts)。
    """
    if pd.isna(score_str):
        return float("nan"), float("nan")
    parts = str(score_str).split("-")
    if len(parts) != 2:
        return float("nan"), float("nan")
    try:
        away_pts = float(parts[0].strip())
        home_pts = float(parts[1].strip())
        return home_pts, away_pts
    except Exception:
        return float("nan"), float("nan")


def parse_nbastats_score_home_away(score_str: str | float | int) -> tuple[float, float]:
    """
    SCORE を "home-away" とみなして (home_pts, away_pts) を返す。
    """
    if pd.isna(score_str):
        return float("nan"), float("nan")
    parts = str(score_str).split("-")
    if len(parts) != 2:
        return float("nan"), float("nan")
    try:
        home_pts = float(parts[0].strip())
        away_pts = float(parts[1].strip())
        return home_pts, away_pts
    except Exception:
        return float("nan"), float("nan")


def infer_score_string_orientation(score_str: pd.Series, score_diff_home: pd.Series) -> str:
    """
    SCORE 文字列の並びを推定する。
    戻り値: "away-home" または "home-away"
    """
    away_home = score_str.apply(parse_nbastats_score_away_home)
    home_a = away_home.map(lambda t: t[0])
    away_a = away_home.map(lambda t: t[1])
    margin_a = home_a - away_a
    ok_a = (
        margin_a.notna()
        & score_diff_home.notna()
        & (np.sign(margin_a) == np.sign(score_diff_home))
    )

    home_away = score_str.apply(parse_nbastats_score_home_away)
    home_b = home_away.map(lambda t: t[0])
    away_b = home_away.map(lambda t: t[1])
    margin_b = home_b - away_b
    ok_b = (
        margin_b.notna()
        & score_diff_home.notna()
        & (np.sign(margin_b) == np.sign(score_diff_home))
    )

    agree_a = int(ok_a.sum())
    agree_b = int(ok_b.sum())
    return "away-home" if agree_a >= agree_b else "home-away"


# ==============================
# 1. PBP レベル標準化
# ==============================

def standardize_pbp(df_nbastats: pd.DataFrame, season: int, seasontype: SeasonType) -> pd.DataFrame:
    """
    stats.nba.com 由来の nbastats PBP を「1行1イベント」の標準形にする。

    主な処理:
      - 列名を snake_case にリネーム（必要なものだけ）
      - time_left_sec, score_diff を数値に変換
      - season, seasontype を列として付与
    """
    # 典型的な列名から snake_case へのマッピング
    rename_map = {
        "GAME_ID": "game_id",
        "EVENTNUM": "event_num",
        "PERIOD": "period",
        "PCTIMESTRING": "pctimestring",
        "SCOREMARGIN": "scoremargin",
        "SCORE": "score_str",
        "EVENTMSGTYPE": "eventmsgtype",
        "EVENTMSGACTIONTYPE": "eventmsgactiontype",
        "HOMEDESCRIPTION": "home_description",
        "VISITORDESCRIPTION": "visitor_description",
        "NEUTRALDESCRIPTION": "neutral_description",
        "PLAYER1_ID": "player1_id",
        "PLAYER2_ID": "player2_id",
        "PLAYER3_ID": "player3_id",
        "PLAYER1_TEAM_ID": "player1_team_id",
        "PLAYER2_TEAM_ID": "player2_team_id",
        "PLAYER3_TEAM_ID": "player3_team_id",
        "PLAYER1_TEAM_ABBREVIATION": "player1_team_abbrev",
        "PLAYER2_TEAM_ABBREVIATION": "player2_team_abbrev",
        "PLAYER3_TEAM_ABBREVIATION": "player3_team_abbrev",
    }
    df = df_nbastats.rename(columns={k: v for k, v in rename_map.items() if k in df_nbastats.columns}).copy()

    # 必須列が無ければ警告的に例外
    required_cols = ["GAME_ID", "EVENTNUM", "PERIOD", "PCTIMESTRING", "SCOREMARGIN", "EVENTMSGTYPE"]
    missing = [c for c in required_cols if c not in df_nbastats.columns]
    if missing:
        raise ValueError(f"nbastats is missing required columns: {missing}")

    # 便利列の追加
    df["time_left_sec"] = df["pctimestring"].map(parse_pctimestring_to_seconds)
    # project-wide convention: score_diff_home = home - away
    df["score_diff_home"] = df["scoremargin"].map(normalize_scoremargin)
    # backward compatibility for existing downstream scripts
    df["score_diff"] = df["score_diff_home"]
    df["season"] = season
    df["seasontype"] = seasontype

    # SCORE がある場合、並び（away-home / home-away）を推定して記録。
    if "score_str" in df.columns:
        orient = infer_score_string_orientation(df["score_str"], df["score_diff_home"])
        df["score_str_orientation"] = orient
        print(f"[check] inferred SCORE orientation: {orient}")
    else:
        df["score_str_orientation"] = pd.NA

    # ソート（game, period, イベント順）
    df = df.sort_values(["game_id", "period", "event_num"]).reset_index(drop=True)

    return df


# ==============================
# 2. shot レベルテーブル
# ==============================

def build_shot_table(df_pbp: pd.DataFrame, df_shotdetail: pd.DataFrame) -> pd.DataFrame:
    """
    PBP と shotdetail をマージして 1行1ショットのテーブルを作る。
    """

    # 1) 列名リネーム（存在するものだけ）
    rename_map = {
        "GAME_ID": "game_id",
        "GAME_EVENT_ID": "game_event_id",   # 一旦別名に
        "EVENTNUM": "eventnum_raw",
        "EVENT_NUM": "eventnum_raw",

        "PERIOD": "period",
        "PCTIMESTRING": "pctimestring",
        "LOC_X": "loc_x",
        "LOC_Y": "loc_y",
        "SHOT_TYPE": "shot_type",
        "SHOT_ZONE_BASIC": "shot_zone_basic",
        "SHOT_ZONE_AREA": "shot_zone_area",
        "SHOT_ZONE_RANGE": "shot_zone_range",
        "SHOT_MADE_FLAG": "shot_made_flag",
        "SHOT_ATTEMPTED_FLAG": "shot_attempted_flag",
        "SHOT_DISTANCE": "shot_distance",
        "TEAM_ID": "team_id",
        "PLAYER_ID": "player_id",
        "SHOT_VALUE": "shot_value",
    }

    sd = df_shotdetail.rename(
        columns={k: v for k, v in rename_map.items() if k in df_shotdetail.columns}
    ).copy()

    # 2) イベント番号候補から event_num を決める
    event_id_candidates = [
        "eventnum_raw",      # EVENTNUM / EVENT_NUM をリネームしたもの
        "GAME_EVENT_ID",
        "GAME_EVENT_NUM",
        "EVENT_ID",
        "event_id",
    ]
    event_col = None
    for col in event_id_candidates:
        if col in df_shotdetail.columns:
            event_col = col
            break
        if col in sd.columns:
            event_col = col
            break

    if event_col is None:
        raise ValueError(
            f"Could not find event id column in shotdetail. "
            f"Tried candidates: {event_id_candidates}. "
            f"Available columns: {list(df_shotdetail.columns)}"
        )

    # unify name
    if event_col in sd.columns:
        sd = sd.rename(columns={event_col: "event_num"})
    else:
        # event_col は元dfにだけあるケース
        sd["event_num"] = df_shotdetail[event_col]

    # 3) マージキー
    key_cols = ["game_id", "event_num"]

    # PBP 側のショットイベント抽出 (EVENTMSGTYPE=1,2)
    shot_events = df_pbp[df_pbp["eventmsgtype"].isin([1, 2])].copy()

    # 4) left join
    shots = shot_events.merge(sd, on=key_cols, how="left", suffixes=("", "_sd"))

    # 必要なら shot_value / is_3pt を補完
    # shots["is_3pt"] = shots["shot_value"].eq(3) | shots["shot_type"].astype(str).str.contains("3PT", na=False)

    return shots


# ==============================
# 3. game レベルテーブル
# ==============================

def _safe_mode(x: pd.Series):
    vc = x.dropna().astype(str).value_counts()
    if vc.empty:
        return pd.NA
    return vc.index[0]


def _infer_game_team_map_from_pbp(df_pbp: pd.DataFrame) -> pd.DataFrame:
    cols_need = {"game_id", "home_description", "visitor_description", "player1_team_id", "player1_team_abbrev"}
    if not cols_need.issubset(set(df_pbp.columns)):
        return pd.DataFrame(columns=["game_id", "home_team_id", "away_team_id", "home_team_abbrev", "away_team_abbrev"])

    tmp = df_pbp[["game_id", "home_description", "visitor_description", "player1_team_id", "player1_team_abbrev"]].copy()
    tmp["player1_team_id"] = pd.to_numeric(tmp["player1_team_id"], errors="coerce")

    home_abbr = (
        tmp[tmp["home_description"].notna() & tmp["player1_team_abbrev"].notna()]
        .groupby("game_id")["player1_team_abbrev"]
        .agg(_safe_mode)
        .rename("home_team_abbrev")
    )
    away_abbr = (
        tmp[tmp["visitor_description"].notna() & tmp["player1_team_abbrev"].notna()]
        .groupby("game_id")["player1_team_abbrev"]
        .agg(_safe_mode)
        .rename("away_team_abbrev")
    )
    abbr_map = pd.concat([home_abbr, away_abbr], axis=1).reset_index()

    id_by_abbr = (
        tmp[tmp["player1_team_abbrev"].notna() & tmp["player1_team_id"].notna()]
        .groupby("player1_team_abbrev")["player1_team_id"]
        .agg(_safe_mode)
        .to_dict()
    )

    abbr_map["home_team_id"] = pd.to_numeric(abbr_map["home_team_abbrev"].map(id_by_abbr), errors="coerce")
    abbr_map["away_team_id"] = pd.to_numeric(abbr_map["away_team_abbrev"].map(id_by_abbr), errors="coerce")
    return abbr_map[["game_id", "home_team_id", "away_team_id", "home_team_abbrev", "away_team_abbrev"]].copy()


def _extract_game_date_from_pbpstats(df_pbpstats: pd.DataFrame) -> pd.DataFrame:
    gid_col = "GAMEID" if "GAMEID" in df_pbpstats.columns else ("game_id" if "game_id" in df_pbpstats.columns else None)
    date_col = "GAMEDATE" if "GAMEDATE" in df_pbpstats.columns else ("game_date" if "game_date" in df_pbpstats.columns else None)
    if gid_col is None or date_col is None:
        return pd.DataFrame(columns=["game_id", "game_date"])
    out = df_pbpstats[[gid_col, date_col]].copy().rename(columns={gid_col: "game_id", date_col: "game_date"})
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_id"]).groupby("game_id", as_index=False)["game_date"].min()
    return out


def build_game_table(df_pbp: pd.DataFrame, df_pbpstats: pd.DataFrame) -> pd.DataFrame:
    """
    1行1ゲームのテーブルを構築する。

    - home_team_id, away_team_id, season, seasontype
    - 最終スコア, margin
    - （必要なら）日付やプレーオフフラグなど

    ※ nbastats PBPに home/away team id が無い場合は、別の game log テーブルと join が必要。
      nba_data 内に game-level の nbastats/pbpstats があればそれを使う想定。
    """
    # 最後のスコア文字列から取り出す（"away-home" の並び）
    # SCORE 列（score_str）を持っている前提
    last_score = (
        df_pbp.dropna(subset=["score_str"])
        .sort_values(["game_id", "period", "event_num"])
        .groupby("game_id")
        .tail(1)[["game_id", "score_str", "season", "seasontype"]]
        .copy()
    )

    orient = "away-home"
    if "score_str_orientation" in df_pbp.columns:
        vc = df_pbp["score_str_orientation"].dropna().astype(str).value_counts()
        if not vc.empty:
            orient = vc.index[0]

    if orient == "home-away":
        parsed = last_score["score_str"].apply(parse_nbastats_score_home_away)
    else:
        parsed = last_score["score_str"].apply(parse_nbastats_score_away_home)
    last_score[["home_pts_final", "away_pts_final"]] = parsed.apply(lambda s: pd.Series(s))
    last_score["margin_final_home_away"] = last_score["home_pts_final"] - last_score["away_pts_final"]
    # backward compatibility
    last_score["margin_final"] = last_score["margin_final_home_away"]

    last_score["final_home_win"] = (last_score["margin_final_home_away"] > 0).astype(int)
    team_map = _infer_game_team_map_from_pbp(df_pbp)
    game_date_map = _extract_game_date_from_pbpstats(df_pbpstats)

    games = last_score.merge(team_map, on="game_id", how="left").merge(game_date_map, on="game_id", how="left")
    games = games[[
        "game_id", "season", "seasontype", "game_date",
        "home_team_id", "away_team_id", "home_team_abbrev", "away_team_abbrev",
        "home_pts_final", "away_pts_final", "margin_final_home_away", "margin_final", "final_home_win"
    ]].copy()
    return games


def refresh_elo_from_games(
    start_season: int,
    end_season: int,
    k: float = 20.0,
    extra_k: List[float] | None = None,
    h: float = 0.0,
    carry: float = 0.75,
    mean_rating: float = 1500.0,
    init_rating: float = 1500.0,
) -> None:
    def _k_label(kv: float) -> str:
        kf = float(kv)
        if kf.is_integer():
            return str(int(kf))
        return f"{kf:.6f}".rstrip("0").rstrip(".").replace(".", "p")

    def _compute_elo_frame(base_df: pd.DataFrame, k_val: float) -> pd.DataFrame:
        allg = base_df.copy()
        allg = allg.sort_values(["season", "game_date", "_st_rank", "game_id"], kind="mergesort").reset_index(drop=True)

        ratings = {}
        cur_season = None
        elo_home = np.full(len(allg), np.nan, dtype=float)
        elo_away = np.full(len(allg), np.nan, dtype=float)
        elo_p = np.full(len(allg), np.nan, dtype=float)

        for i, r in enumerate(allg.itertuples(index=False)):
            season = int(r.season)
            if cur_season is None:
                cur_season = season
            elif season != cur_season:
                for tid, val in list(ratings.items()):
                    ratings[tid] = mean_rating + carry * (val - mean_rating)
                cur_season = season

            home = int(r.home_team_id)
            away = int(r.away_team_id)
            win = float(r.final_home_win)
            r_home = ratings.get(home, init_rating)
            r_away = ratings.get(away, init_rating)
            p_home = 1.0 / (1.0 + 10.0 ** (-((r_home + h) - r_away) / 400.0))

            elo_home[i] = r_home
            elo_away[i] = r_away
            elo_p[i] = p_home

            delta = float(k_val) * (win - p_home)
            ratings[home] = r_home + delta
            ratings[away] = r_away - delta

        allg["elo_home_pregame"] = elo_home
        allg["elo_away_pregame"] = elo_away
        allg["elo_diff_pregame"] = allg["elo_home_pregame"] - allg["elo_away_pregame"]
        allg["elo_exp_home_pregame"] = elo_p
        allg["elo_k"] = float(k_val)
        allg["elo_h"] = float(h)
        allg["elo_carry"] = float(carry)
        return allg

    rows = []
    for season in range(start_season, end_season + 1):
        for st in ("rs",):
            path = OUT_DIR / f"games_{season}_{st}.parquet"
            if not path.exists():
                continue
            g = pd.read_parquet(path).copy()
            need = {"game_id", "season", "seasontype", "game_date", "home_team_id", "away_team_id", "final_home_win"}
            if not need.issubset(set(g.columns)):
                print(f"[WARN] skip Elo: missing columns in {path.name}")
                continue
            g["season"] = pd.to_numeric(g["season"], errors="coerce").astype("Int64")
            g["seasontype"] = g["seasontype"].astype(str)
            g["game_id"] = pd.to_numeric(g["game_id"], errors="coerce").astype("Int64")
            g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
            g["home_team_id"] = pd.to_numeric(g["home_team_id"], errors="coerce")
            g["away_team_id"] = pd.to_numeric(g["away_team_id"], errors="coerce")
            g["final_home_win"] = pd.to_numeric(g["final_home_win"], errors="coerce")
            g["file_path"] = str(path)
            rows.append(g)

    if not rows:
        print("[WARN] no games parquet found for Elo refresh")
        return

    base_allg = pd.concat(rows, ignore_index=True)
    base_allg = base_allg.dropna(subset=["season", "game_id", "home_team_id", "away_team_id", "final_home_win"]).copy()
    base_allg["_st_rank"] = base_allg["seasontype"].map({"rs": 0}).fillna(9).astype(int)

    # Canonical unsuffixed Elo is fixed to k=20 for compatibility/clarity.
    canonical_k = 20.0
    ks = [canonical_k, float(k)]
    if extra_k:
        for kv in extra_k:
            kf = float(kv)
            if kf not in ks:
                ks.append(kf)
    elo_frames = {kf: _compute_elo_frame(base_allg, kf) for kf in ks}
    allg = elo_frames[canonical_k]

    # Sanity check: Elo should be positively correlated with home win outcome.
    chk = allg.dropna(subset=["final_home_win", "elo_diff_pregame", "elo_exp_home_pregame"]).copy()
    if len(chk) < 1000:
        print(f"[WARN] Elo correlation check skipped: too few rows ({len(chk):,})")
    else:
        corr_diff = chk["elo_diff_pregame"].corr(chk["final_home_win"])
        corr_exp = chk["elo_exp_home_pregame"].corr(chk["final_home_win"])
        print(
            "[ELO] correlation check: "
            f"corr(elo_diff_pregame, final_home_win)={corr_diff:.4f}, "
            f"corr(elo_exp_home_pregame, final_home_win)={corr_exp:.4f}"
        )
        if not np.isfinite(corr_diff) or not np.isfinite(corr_exp) or corr_diff <= 0 or corr_exp <= 0:
            raise ValueError(
                "Elo sanity check failed: expected positive correlation with final_home_win, "
                f"got corr_diff={corr_diff}, corr_exp={corr_exp}"
            )

    for path_str, grp in allg.groupby("file_path"):
        path = Path(path_str)
        base = pd.read_parquet(path).copy()
        base["game_id"] = pd.to_numeric(base["game_id"], errors="coerce").astype("Int64")
        out = base.drop(columns=["elo_home_pregame", "elo_away_pregame", "elo_diff_pregame", "elo_exp_home_pregame", "elo_k", "elo_h", "elo_carry"], errors="ignore")

        # Add suffixed Elo columns for all requested k values (including main k).
        for kv in ks:
            grp_k = elo_frames[kv]
            grp_k = grp_k[grp_k["file_path"] == str(path)]
            if grp_k.empty:
                continue
            ktag = _k_label(kv)
            key_k = grp_k[[
                "game_id", "elo_home_pregame", "elo_away_pregame", "elo_diff_pregame",
                "elo_exp_home_pregame", "elo_k", "elo_h", "elo_carry"
            ]].rename(columns={
                "elo_home_pregame": f"elo_home_pregame_k{ktag}",
                "elo_away_pregame": f"elo_away_pregame_k{ktag}",
                "elo_diff_pregame": f"elo_diff_pregame_k{ktag}",
                "elo_exp_home_pregame": f"elo_exp_home_pregame_k{ktag}",
                "elo_k": f"elo_k_k{ktag}",
                "elo_h": f"elo_h_k{ktag}",
                "elo_carry": f"elo_carry_k{ktag}",
            })
            out = out.drop(
                columns=[
                    f"elo_home_pregame_k{ktag}",
                    f"elo_away_pregame_k{ktag}",
                    f"elo_diff_pregame_k{ktag}",
                    f"elo_exp_home_pregame_k{ktag}",
                    f"elo_k_k{ktag}",
                    f"elo_h_k{ktag}",
                    f"elo_carry_k{ktag}",
                ],
                errors="ignore",
            )
            out = out.merge(key_k, on="game_id", how="left")
        out.to_parquet(path, index=False)
        print(f"[ELO] updated {path.name} rows={len(out):,}")


# ==============================
# 4. team×season 集約テーブル
# ==============================

def build_team_season_table(
    df_shots: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    # オフェンス側：team_id が shotdetail にある前提
    if "player1_team_id" in df_shots.columns:
        team_col = "player1_team_id"
    elif "team_id" in df_shots.columns:
        team_col = "team_id"
    else:
        raise ValueError("Shot table must contain team id column (e.g., player1_team_id or team_id)")

    df = df_shots.copy()

    # ---- ここを robust にする ----
    # shot_value があればそれも使う / 無ければ False として扱う
    if "shot_value" in df.columns:
        is_3pt_from_value = df["shot_value"].eq(3)
    else:
        # 全部 False の Series を用意（長さは df と同じ）
        is_3pt_from_value = pd.Series(False, index=df.index)

    # SHOT_TYPE に "3PT" が含まれていれば3Pと判定
    if "shot_type" in df.columns:
        is_3pt_from_type = df["shot_type"].astype(str).str.contains("3PT", na=False)
    else:
        is_3pt_from_type = pd.Series(False, index=df.index)

    df["is_3pt"] = is_3pt_from_value | is_3pt_from_type
    # ----------------------------

    # 1行1FG想定
    df["is_fg"] = True

    # made flag を0/1に揃えておく
    if "shot_made_flag" in df.columns:
        df["shot_made_flag"] = df["shot_made_flag"].fillna(0).astype(int)
    else:
        df["shot_made_flag"] = 0

    # 3P成功フラグ
    df["three_pm_flag"] = df["is_3pt"] & (df["shot_made_flag"] == 1)

    # 加重計算に使う値を用意
    default_shot_values = pd.Series(2, index=df.index)
    default_shot_values[df["is_3pt"]] = 3
    if "shot_value" in df.columns:
        shot_values = pd.to_numeric(df["shot_value"], errors="coerce").fillna(0)
        shot_values = shot_values.where(shot_values > 0, default_shot_values)
    else:
        shot_values = default_shot_values

    df["_points_made"] = df["shot_made_flag"] * shot_values
    df["_points_3p"] = df["shot_made_flag"] * df["is_3pt"] * shot_values

    if "shot_zone_basic" in df.columns:
        sz_basic = df["shot_zone_basic"].astype(str)
        paint_mask = sz_basic.str.contains("paint", case=False, na=False) | sz_basic.str.contains("restricted", case=False, na=False)
    else:
        paint_mask = pd.Series(False, index=df.index)
    df["_points_pitp"] = df["_points_made"].where(paint_mask, 0)

    # 任意列の検知
    def _first_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    fta_col = _first_col(["fta", "ft_attempt", "ft_attempts", "freethrow_attempts", "free_throw_attempts"])
    ftm_col = _first_col(["ftm", "ft_made", "freethrow_made", "free_throw_made"])
    tov_col = _first_col(["tov", "turnover", "turnovers"])
    oreb_col = _first_col(["oreb", "offensive_rebound", "offensive_rebounds"])
    opp_dreb_col = _first_col(["opp_dreb", "opponent_dreb", "dreb_opp"])

    agg_dict = {
        "fga": ("is_fg", "sum"),
        "fgm": ("shot_made_flag", "sum"),
        "three_pa": ("is_3pt", "sum"),
        "three_pm": ("three_pm_flag", "sum"),
        "points": ("_points_made", "sum"),
        "points_3p": ("_points_3p", "sum"),
        "points_pitp": ("_points_pitp", "sum"),
        "games_played": ("game_id", "nunique"),
    }

    if fta_col:
        df["_fta"] = pd.to_numeric(df[fta_col], errors="coerce").fillna(0)
        agg_dict["fta"] = ("_fta", "sum")
    if ftm_col:
        df["_ftm"] = pd.to_numeric(df[ftm_col], errors="coerce").fillna(0)
        agg_dict["ftm"] = ("_ftm", "sum")
    if tov_col:
        df["_tov"] = pd.to_numeric(df[tov_col], errors="coerce").fillna(0)
        agg_dict["tov"] = ("_tov", "sum")
    if oreb_col:
        df["_oreb"] = pd.to_numeric(df[oreb_col], errors="coerce").fillna(0)
        agg_dict["oreb"] = ("_oreb", "sum")
    if opp_dreb_col:
        df["_opp_dreb"] = pd.to_numeric(df[opp_dreb_col], errors="coerce").fillna(0)
        agg_dict["opp_dreb"] = ("_opp_dreb", "sum")

    # 集約
    agg = (
        df.groupby(["season", team_col])
        .agg(**agg_dict)
        .reset_index()
    )

    agg["fg_pct"] = agg["fgm"] / agg["fga"]
    agg["three_p_pct"] = agg["three_pm"].where(agg["three_pa"] > 0) / agg["three_pa"].where(agg["three_pa"] > 0)
    agg["ftr"] = agg["fta"] / agg["fga"] if "fta" in agg.columns else pd.NA

    if "tov" in agg.columns:
        fta_for_tov = agg["fta"] if "fta" in agg.columns else 0
        denom_tov = agg["fga"] + 0.44 * fta_for_tov + agg["tov"]
        agg["tov_pct"] = agg["tov"] / denom_tov
    else:
        agg["tov_pct"] = pd.NA

    if "oreb" in agg.columns and "opp_dreb" in agg.columns:
        reb_denom = agg["oreb"] + agg["opp_dreb"]
        agg["orb_pct"] = agg["oreb"] / reb_denom
    else:
        agg["orb_pct"] = pd.NA

    fta_for_poss = agg["fta"] if "fta" in agg.columns else 0
    tov_for_poss = agg["tov"] if "tov" in agg.columns else 0
    oreb_for_poss = agg["oreb"] if "oreb" in agg.columns else 0
    games_played = agg["games_played"].replace(0, pd.NA)
    poss = agg["fga"] + 0.44 * fta_for_poss + tov_for_poss - oreb_for_poss
    agg["pace"] = poss / games_played

    ft_points = agg["ftm"] if "ftm" in agg.columns else 0
    total_points = agg["points"] + ft_points
    total_points = total_points.replace(0, pd.NA)
    agg["pct_pts_3p"] = agg["points_3p"] / total_points
    agg["pct_pts_pitp"] = agg["points_pitp"] / total_points

    # カラム名調整
    agg = agg.rename(columns={team_col: "team_id"})

    return agg



# ==============================
# 5. Possession レベルテーブル（雛形）
# ==============================

def build_possession_table(
    df_pbp: pd.DataFrame,
    df_pbpstats: pd.DataFrame,
    seasontype: SeasonType,
) -> pd.DataFrame:
    """
    pbpstats 由来のポゼッション情報と PBP を使って、1行1ポゼッションのテーブルを構築する雛形。

    pbpstats CSV の schema はバージョンによって異なるため、
    ここでは pbpstats の STARTTIME/ENDTIME を基準に、
    同一ポゼッションとみなせる行を集約して「1行1ポゼッション」を作る。

    具体的には:
      - GAMEID, PERIOD, STARTTIME, ENDTIME, STARTSCOREDIFFERENTIAL, STARTTYPE, OPPONENT
        をキーに groupby
      - 数値列は max、文字列列は first を基本に集約
    """
    poss = df_pbpstats.copy()

    # groupby key（存在する列だけ使用）
    key_candidates = [
        "GAMEID",
        "PERIOD",
        "STARTTIME",
        "ENDTIME",
        "STARTSCOREDIFFERENTIAL",
        "STARTTYPE",
        "OPPONENT",
    ]
    group_cols = [c for c in key_candidates if c in poss.columns]
    if not group_cols:
        raise ValueError("pbpstats にポゼッション識別用の列が見つかりません。")

    # 集約設定: 数値は max、文字列は first
    numeric_cols = poss.select_dtypes(include="number").columns.tolist()
    agg = {}
    for c in poss.columns:
        if c in group_cols:
            continue
        if c in numeric_cols:
            agg[c] = "max"
        else:
            agg[c] = "first"

    poss = poss.groupby(group_cols, dropna=False, as_index=False).agg(agg)

    # 下流と合わせて period 列は小文字に寄せる
    if "PERIOD" in poss.columns:
        poss = poss.rename(columns={"PERIOD": "period"})

    return poss


# ==============================
# 6. Lineup レベルテーブル（雛形）
# ==============================

def build_lineup_table(df_pbp_with_lineups: pd.DataFrame) -> pd.DataFrame:
    """
    コート上10人が付与された PBP（players_on_court 済み）から、
    lineup_id ごとの集約テーブルを作る。

    ここでは:
      - lineup_id を (home5人, away5人) から構成
      - lineup_id × season × team で minutes, off_rating, def_rating 等を計算する骨組みだけ置く。

    実際には nba_on_court.players_on_court(...) を使って df_pbp_with_lineups を先に作る想定。
    """
    df = df_pbp_with_lineups.copy()

    # TODO:
    # - home側の5人, away側の5人をソートして文字列連結し lineup_id を作成
    #   例: "H:playerA-playerB-playerC-playerD-playerE|A:playerF-..."
    # - 各行の経過時間 dt_sec を計算し、lineup_id × team_id × season ごとに合計時間を集約
    # - ポゼッションベースで ORtg, DRtg を計算して lineup テーブルに join

    lineup_agg = pd.DataFrame()  # TODO: 実装
    return lineup_agg


# ==============================
# メイン（1シーズン分を処理）
# ==============================

def process_one_season(season: int, seasontype: SeasonType) -> None:
    print(f"=== Processing season={season}, type={seasontype} ===")

    nbastats_path = find_nbastats_file(season, seasontype)
    pbpstats_path = find_pbpstats_file(season, seasontype)
    shotdetail_path = find_shotdetail_file(season, seasontype)

    print(f" nbastats  : {nbastats_path.name}")
    print(f" pbpstats  : {pbpstats_path.name}")
    print(f" shotdetail: {shotdetail_path.name}")

    df_nbastats = pd.read_csv(nbastats_path)
    df_pbp = standardize_pbp(df_nbastats, season, seasontype)

    df_shotdetail = pd.read_csv(shotdetail_path)
    df_shots = build_shot_table(df_pbp, df_shotdetail)

    df_pbpstats = pd.read_csv(pbpstats_path)
    df_games = build_game_table(df_pbp, df_pbpstats)
    df_poss = build_possession_table(df_pbp, df_pbpstats, seasontype=seasontype)

    # TODO: lineup の players_on_court を事前に作成してから build_lineup_table に渡す
    # df_pbp_with_lineups = ...
    # df_lineups = build_lineup_table(df_pbp_with_lineups)
    df_lineups = pd.DataFrame()

    # チーム×シーズン集約（ここではオフェンス側だけ軽く）
    df_team_season = build_team_season_table(df_shots, df_games)

    # 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st = seasontype

    df_pbp.to_parquet(OUT_DIR / f"pbp_events_{season}_{st}.parquet", index=False)
    df_shots.to_parquet(OUT_DIR / f"shots_{season}_{st}.parquet", index=False)
    df_games.to_parquet(OUT_DIR / f"games_{season}_{st}.parquet", index=False)
    df_poss.to_parquet(OUT_DIR / f"poss_{season}_{st}.parquet", index=False)
    if not df_lineups.empty:
        df_lineups.to_parquet(OUT_DIR / f"lineups_{season}_{st}.parquet", index=False)

    # team_season は複数シーズン分まとめて後で1ファイルにしてもよいが、
    # ここではとりあえず append モードで CSV/Parquet に追記する運用を想定しても良い。
    df_team_season.to_parquet(OUT_DIR / f"team_season_{season}_{st}.parquet", index=False)

    print(f"=== Done season={season}, type={seasontype} ===")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument("--seasontype", type=str, default="rs", choices=["rs"])
    parser.add_argument("--refresh-elo", type=int, default=1, choices=[0, 1])
    parser.add_argument("--elo-k", type=float, default=20.0)
    parser.add_argument("--elo-extra-k", type=str, default="")
    parser.add_argument("--elo-h", type=float, default=0.0)
    parser.add_argument("--elo-carry", type=float, default=0.75)
    parser.add_argument("--elo-mean", type=float, default=1500.0)
    parser.add_argument("--elo-init", type=float, default=1500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    for season in range(args.start_season, args.end_season + 1):
        process_one_season(season, args.seasontype)
    if int(args.refresh_elo) == 1:
        extra_k = []
        if str(args.elo_extra_k).strip():
            extra_k = [float(x.strip()) for x in str(args.elo_extra_k).split(",") if x.strip()]
        refresh_elo_from_games(
            start_season=args.start_season,
            end_season=args.end_season,
            k=float(args.elo_k),
            extra_k=extra_k,
            h=float(args.elo_h),
            carry=float(args.elo_carry),
            mean_rating=float(args.elo_mean),
            init_rating=float(args.elo_init),
        )


if __name__ == "__main__":
    main()
