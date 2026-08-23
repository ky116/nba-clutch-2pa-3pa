from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, List
import re

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _ensure_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    # Handle common YYYYMMDD integers/strings explicitly.
    s_str = s.astype(str)
    if s_str.str.fullmatch(r"\d{8}").all():
        parsed = pd.to_datetime(s_str, format="%Y%m%d", errors="coerce")
        if parsed.notna().any():
            return parsed
    return pd.to_datetime(s, errors="coerce")


def _infer_shot_cat(
    df: pd.DataFrame,
    *,
    shot_type_col: Optional[str] = "SHOT_TYPE",
    shot_value_col: Optional[str] = None,
) -> pd.Series:
    """
    Return categorical shot type: '2P' or '3P'.
    Tries shot_value_col first (2/3), else parses shot_type_col string.
    """
    if shot_value_col is not None and shot_value_col in df.columns:
        v = pd.to_numeric(df[shot_value_col], errors="coerce")
        out = np.where(v == 3, "3P", np.where(v == 2, "2P", np.nan))
        return pd.Series(out, index=df.index, name="shot_cat")

    if shot_type_col is None or shot_type_col not in df.columns:
        raise ValueError("Need either shot_value_col or shot_type_col to infer 2P/3P.")

    st = df[shot_type_col].astype(str).str.upper()
    out = np.where(st.str.contains("3PT"), "3P", np.where(st.str.contains("2PT"), "2P", None))
    return pd.Series(out, index=df.index, name="shot_cat", dtype="object")


def _beta_prior_from_league_mean(mu: float, prior_strength: float) -> Tuple[float, float]:
    """
    Simple empirical-Bayes prior:
      alpha = mu * s, beta = (1-mu) * s
    where s is prior_strength (pseudo-attempts).
    """
    mu = float(np.clip(mu, 1e-6, 1 - 1e-6))
    s = float(prior_strength)
    return mu * s, (1.0 - mu) * s


@dataclass(frozen=True)
class EBPriors:
    # priors per shot_cat: { '2P': (alpha,beta), '3P': (alpha,beta) }
    priors: Dict[str, Tuple[float, float]]

    def alpha_beta(self, shot_cat: str) -> Tuple[float, float]:
        if shot_cat not in self.priors:
            raise KeyError(f"Missing prior for shot_cat={shot_cat}")
        return self.priors[shot_cat]


def build_league_priors(
    shots: pd.DataFrame,
    *,
    made_col: str = "SHOT_MADE_FLAG",
    shot_type_col: str = "SHOT_TYPE",
    shot_value_col: Optional[str] = None,
    prior_strength_2p: float = 400.0,
    prior_strength_3p: float = 200.0,
) -> EBPriors:
    """
    Build league-level beta priors separately for 2P and 3P.

    Notes:
      - This uses ALL rows in `shots` to compute league mean. For strict no-lookahead,
        pass only a training subset (e.g., seasons < target season) when calling this.
    """
    tmp = shots.copy()
    tmp["shot_cat"] = _infer_shot_cat(tmp, shot_type_col=shot_type_col, shot_value_col=shot_value_col)
    tmp = tmp[tmp["shot_cat"].isin(["2P", "3P"])].copy()

    made = pd.to_numeric(tmp[made_col], errors="coerce").fillna(0.0).astype(float)
    tmp["_made"] = made
    tmp["_att"] = 1.0

    priors: Dict[str, Tuple[float, float]] = {}
    for cat, strength in [("2P", prior_strength_2p), ("3P", prior_strength_3p)]:
        sub = tmp[tmp["shot_cat"] == cat]
        mu = (sub["_made"].sum() / max(sub["_att"].sum(), 1.0)) if len(sub) else 0.5
        priors[cat] = _beta_prior_from_league_mean(mu, strength)

    return EBPriors(priors=priors)


def _eb_rate(k: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return (k + alpha) / (n + alpha + beta)


def _safe_rate(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.divide(k, n, out=np.full_like(k, np.nan, dtype=float), where=n > 0)


# -----------------------------
# Season helpers
# -----------------------------
def _parse_season_value(x) -> float:
    """
    Accepts:
      - 2019, 2020 ...
      - "2019-20"
      - "SEASON_2019"
      - 22019 (NBA season_id style) -> 2019
    Returns: season_start_year (e.g., 2019) as float (NaN possible)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x)
    m = re.search(r"(\d{4})", s)
    if not m:
        return np.nan
    year = int(m.group(1))
    # handle NBA season_id like 22019, 22020...
    if year < 1900:
        # fallback: take last 4 digits if possible
        m2 = re.search(r"(\d{4})$", s)
        if m2:
            year = int(m2.group(1))
    return float(year)


def _season_from_game_date(game_date: pd.Series) -> pd.Series:
    """
    Infer NBA season start year from date:
      Oct(10)-Dec -> same year
      Jan-Sep    -> previous year
    """
    dt = pd.to_datetime(game_date, errors="coerce")
    y = dt.dt.year.astype("float")
    m = dt.dt.month.astype("float")
    # NBA season starts around Oct
    season = np.where(m >= 10, y, y - 1)
    return pd.Series(season, index=game_date.index, name="_season")


def _build_cat_priors_by_season(
    shots: pd.DataFrame,
    *,
    season_col: str,
    made_col: str,
    shot_cat_col: str = "shot_cat",
    prior_strength_2p: float = 400.0,
    prior_strength_3p: float = 200.0,
) -> tuple[dict[int, dict[str, tuple[float, float]]], dict[str, tuple[float, float]]]:
    """
    Returns:
      priors_by_season[season_int]["2P"/"3P"] = (alpha,beta)
      global_priors["2P"/"3P"] = (alpha,beta)   # fallback
    """
    tmp = shots[[season_col, shot_cat_col, made_col]].copy()
    tmp = tmp[tmp[shot_cat_col].isin(["2P", "3P"])].copy()
    tmp[season_col] = pd.to_numeric(tmp[season_col], errors="coerce")
    tmp = tmp.dropna(subset=[season_col])

    tmp["_made"] = pd.to_numeric(tmp[made_col], errors="coerce").fillna(0.0).astype(float)
    tmp["_att"] = 1.0

    # global mean (fallback)
    global_priors: dict[str, tuple[float, float]] = {}
    for cat, strength in [("2P", prior_strength_2p), ("3P", prior_strength_3p)]:
        sub = tmp[tmp[shot_cat_col] == cat]
        mu = (sub["_made"].sum() / max(sub["_att"].sum(), 1.0)) if len(sub) else 0.5
        global_priors[cat] = _beta_prior_from_league_mean(mu, strength)

    priors_by_season: dict[int, dict[str, tuple[float, float]]] = {}
    for season, g in tmp.groupby(season_col):
        season_int = int(season)
        priors_by_season[season_int] = {}
        for cat, strength in [("2P", prior_strength_2p), ("3P", prior_strength_3p)]:
            sub = g[g[shot_cat_col] == cat]
            mu = (sub["_made"].sum() / max(sub["_att"].sum(), 1.0)) if len(sub) else 0.5
            priors_by_season[season_int][cat] = _beta_prior_from_league_mean(mu, strength)

    # Shift priors by one season: use prior from previous season when available.
    shifted: dict[int, dict[str, tuple[float, float]]] = {}
    seasons_sorted = sorted(priors_by_season.keys())
    for season in seasons_sorted:
        prev = season - 1
        if prev in priors_by_season:
            shifted[season] = priors_by_season[prev]
        else:
            shifted[season] = global_priors

    return shifted, global_priors


def _build_3pa_share_priors_by_season(
    shots: pd.DataFrame,
    *,
    season_col: str,
    shot_cat_col: str = "shot_cat",
    prior_strength_3pa_share: float = 200.0,
) -> tuple[dict[int, tuple[float, float]], tuple[float, float]]:
    """
    Beta prior for 3PA share = P(3P attempt | 2P/3P attempt).
    Returns:
      share_prior_by_season[season_int] = (alpha,beta)
      global_share_prior = (alpha,beta)  # fallback
    """
    tmp = shots[[season_col, shot_cat_col]].copy()
    tmp = tmp[tmp[shot_cat_col].isin(["2P", "3P"])].copy()
    tmp[season_col] = pd.to_numeric(tmp[season_col], errors="coerce")
    tmp = tmp.dropna(subset=[season_col])

    # global
    att = tmp[shot_cat_col].value_counts()
    k = float(att.get("3P", 0.0))
    n = float(att.get("2P", 0.0) + att.get("3P", 0.0))
    mu = (k / n) if n > 0 else 0.35
    global_prior = _beta_prior_from_league_mean(mu, prior_strength_3pa_share)

    # per season
    share_prior_by_season: dict[int, tuple[float, float]] = {}
    for season, g in tmp.groupby(season_col):
        att_s = g[shot_cat_col].value_counts()
        k_s = float(att_s.get("3P", 0.0))
        n_s = float(att_s.get("2P", 0.0) + att_s.get("3P", 0.0))
        mu_s = (k_s / n_s) if n_s > 0 else mu
        share_prior_by_season[int(season)] = _beta_prior_from_league_mean(mu_s, prior_strength_3pa_share)

    # Shift priors by one season: use previous season prior when available.
    shifted: dict[int, tuple[float, float]] = {}
    seasons_sorted = sorted(share_prior_by_season.keys())
    for season in seasons_sorted:
        prev = season - 1
        if prev in share_prior_by_season:
            shifted[season] = share_prior_by_season[prev]
        else:
            shifted[season] = global_prior

    return shifted, global_prior


def _hybrid_prior_mu(
    team_mu: np.ndarray,
    team_n: np.ndarray,
    league_mu: np.ndarray,
    continuity: float,
    k: np.ndarray | float,
) -> np.ndarray:
    continuity = float(np.clip(continuity, 0.0, 1.0))
    team_n = np.asarray(team_n, dtype=float)
    league_mu = np.asarray(league_mu, dtype=float)
    team_mu = np.asarray(team_mu, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    w = continuity * (team_n / (team_n + k_arr))
    w = np.where(np.isfinite(team_mu), w, 0.0)
    w = np.clip(w, 0.0, 1.0)
    return (w * np.where(np.isfinite(team_mu), team_mu, league_mu)) + ((1.0 - w) * league_mu)


def _build_prev_team_cat_stats(
    df: pd.DataFrame,
    *,
    season_col: str,
    team_col: str,
    shot_cat_col: str,
    made_col: str,
    prefix: str,
) -> pd.DataFrame:
    agg = (
        df.groupby([season_col, team_col, shot_cat_col], as_index=False)
        .agg(_att=(shot_cat_col, "size"), _made=(made_col, "sum"))
        .rename(columns={team_col: "team_id", shot_cat_col: "shot_cat"})
    )
    agg["_season"] = pd.to_numeric(agg[season_col], errors="coerce").astype("Int64")
    agg = agg.dropna(subset=["_season"]).copy()
    agg["_season"] = agg["_season"].astype(int) + 1
    agg[f"{prefix}_team_att_prev"] = agg["_att"].astype(float)
    agg[f"{prefix}_team_mu_prev"] = np.divide(
        agg["_made"].astype(float),
        agg["_att"].astype(float),
        out=np.full(len(agg), np.nan, dtype=float),
        where=agg["_att"].to_numpy() > 0,
    )
    return agg[["_season", "team_id", "shot_cat", f"{prefix}_team_mu_prev", f"{prefix}_team_att_prev"]]


def _build_prev_league_cat_means(
    df: pd.DataFrame,
    *,
    season_col: str,
    shot_cat_col: str,
    made_col: str,
    prefix: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    agg = (
        df.groupby([season_col, shot_cat_col], as_index=False)
        .agg(_att=(shot_cat_col, "size"), _made=(made_col, "sum"))
        .rename(columns={shot_cat_col: "shot_cat"})
    )
    agg["_season"] = pd.to_numeric(agg[season_col], errors="coerce").astype("Int64")
    agg = agg.dropna(subset=["_season"]).copy()
    agg["_season"] = agg["_season"].astype(int) + 1
    agg[f"{prefix}_league_mu_prev"] = np.divide(
        agg["_made"].astype(float),
        agg["_att"].astype(float),
        out=np.full(len(agg), np.nan, dtype=float),
        where=agg["_att"].to_numpy() > 0,
    )
    prev = agg[["_season", "shot_cat", f"{prefix}_league_mu_prev"]]

    g = df.groupby(shot_cat_col, as_index=False).agg(_att=(shot_cat_col, "size"), _made=(made_col, "sum"))
    global_mu: dict[str, float] = {}
    for _, r in g.iterrows():
        cat = str(r[shot_cat_col])
        att = float(r["_att"])
        made = float(r["_made"])
        global_mu[cat] = (made / att) if att > 0 else 0.5
    return prev, global_mu


def _build_prev_team_share_stats(
    df: pd.DataFrame,
    *,
    season_col: str,
    team_col: str,
    shot_cat_col: str,
    prefix: str,
) -> pd.DataFrame:
    tmp = df[[season_col, team_col, shot_cat_col]].copy()
    tmp = tmp[tmp[shot_cat_col].isin(["2P", "3P"])].copy()
    agg = (
        tmp.groupby([season_col, team_col, shot_cat_col], as_index=False)
        .size()
        .pivot_table(index=[season_col, team_col], columns=shot_cat_col, values="size", fill_value=0)
        .reset_index()
    )
    agg["_season"] = pd.to_numeric(agg[season_col], errors="coerce").astype("Int64")
    agg = agg.dropna(subset=["_season"]).copy()
    agg["_season"] = agg["_season"].astype(int) + 1
    att_2p = pd.to_numeric(agg.get("2P", 0), errors="coerce").fillna(0.0).astype(float)
    att_3p = pd.to_numeric(agg.get("3P", 0), errors="coerce").fillna(0.0).astype(float)
    n = att_2p + att_3p
    k = att_3p
    agg[f"{prefix}_team_3pa_att_prev"] = n
    agg[f"{prefix}_team_3pa_mu_prev"] = np.divide(
        k,
        n,
        out=np.full(len(agg), np.nan, dtype=float),
        where=n.to_numpy() > 0,
    )
    return agg[["_season", team_col, f"{prefix}_team_3pa_mu_prev", f"{prefix}_team_3pa_att_prev"]].rename(
        columns={team_col: "team_id"}
    )


def _build_prev_league_share_means(
    df: pd.DataFrame,
    *,
    season_col: str,
    shot_cat_col: str,
    prefix: str,
) -> tuple[pd.DataFrame, float]:
    tmp = df[[season_col, shot_cat_col]].copy()
    tmp = tmp[tmp[shot_cat_col].isin(["2P", "3P"])].copy()
    agg = (
        tmp.groupby([season_col, shot_cat_col], as_index=False)
        .size()
        .pivot_table(index=[season_col], columns=shot_cat_col, values="size", fill_value=0)
        .reset_index()
    )
    agg["_season"] = pd.to_numeric(agg[season_col], errors="coerce").astype("Int64")
    agg = agg.dropna(subset=["_season"]).copy()
    agg["_season"] = agg["_season"].astype(int) + 1
    att_2p = pd.to_numeric(agg.get("2P", 0), errors="coerce").fillna(0.0).astype(float)
    att_3p = pd.to_numeric(agg.get("3P", 0), errors="coerce").fillna(0.0).astype(float)
    n = att_2p + att_3p
    k = att_3p
    agg[f"{prefix}_league_3pa_mu_prev"] = np.divide(
        k,
        n,
        out=np.full(len(agg), np.nan, dtype=float),
        where=n.to_numpy() > 0,
    )
    prev = agg[["_season", f"{prefix}_league_3pa_mu_prev"]]

    att = tmp[shot_cat_col].value_counts()
    k_g = float(att.get("3P", 0.0))
    n_g = float(att.get("2P", 0.0) + att.get("3P", 0.0))
    global_mu = (k_g / n_g) if n_g > 0 else 0.35
    return prev, global_mu


# -----------------------------
# Main feature builder
# -----------------------------
def add_team_eb_context_features(
    shots: pd.DataFrame,
    *,
    # required columns
    game_id_col: str = "GAME_ID",
    game_date_col: str = "GAME_DATE",
    season_col: str = "SEASON",  # if missing, inferred from GAME_DATE
    offense_team_col: str = "TEAM_ID",
    defense_team_col: str = "OPPONENT_TEAM_ID",
    made_col: str = "SHOT_MADE_FLAG",
    shot_type_col: str = "SHOT_TYPE",
    shot_value_col: Optional[str] = None,
    # prior strengths
    prior_strength_2p: float = 400.0,
    prior_strength_3p: float = 200.0,
    add_3pa_share: bool = True,
    prior_strength_3pa_share: float = 200.0,
    continuity_off_pct: float = 0.70,
    continuity_allowed_pct: float = 0.40,
    continuity_off_3pa_share: float = 0.35,
    continuity_allowed_3pa_share: float = 0.25,
    team_weight_k_2p: float = 400.0,
    team_weight_k_3p: float = 800.0,
    team_weight_k_3pa_share: float = 600.0,
    use_eb: bool = True,
) -> pd.DataFrame:
    """
    Season-specific shrinkage (lagged by one season) when use_eb=True:
      - 2P%/3P% EB priors use the previous season's league means when available
      - 3PA share EB priors use the previous season's league means when available
    Adds (pre-game):
      own_off_2p_pct_eb, own_off_3p_pct_eb, own_allowed_2p_pct_eb, own_allowed_3p_pct_eb,
      opp_off_2p_pct_eb, opp_off_3p_pct_eb, opp_allowed_2p_pct_eb, opp_allowed_3p_pct_eb,
      and *_att_prev/*_made_prev plus (optional) *_3pa_share_eb for off/allowed.
    """
    df = shots.copy()

    # --- basic checks ---
    for c in [game_id_col, game_date_col, offense_team_col, made_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df[game_date_col] = _ensure_datetime(df[game_date_col])
    if df[game_date_col].isna().any():
        raise ValueError(f"{game_date_col} has NaT after parsing.")

    # --- season column (no need to be perfect; used only for season-level priors) ---
    if season_col in df.columns:
        df["_season"] = df[season_col].apply(_parse_season_value)
        # if failed to parse many, fallback to date-derived for those
        miss = df["_season"].isna()
        if miss.any():
            df.loc[miss, "_season"] = _season_from_game_date(df.loc[miss, game_date_col])
    else:
        df["_season"] = _season_from_game_date(df[game_date_col])

    df["_season"] = pd.to_numeric(df["_season"], errors="coerce")
    if df["_season"].isna().any():
        raise ValueError("Could not infer season for some rows. Provide a usable season_col or valid game dates.")

    # --- shot cat (2P/3P) ---
    df["shot_cat"] = _infer_shot_cat(df, shot_type_col=shot_type_col, shot_value_col=shot_value_col)
    df = df[df["shot_cat"].isin(["2P", "3P"])].copy()
    df[made_col] = pd.to_numeric(df[made_col], errors="coerce").fillna(0).astype(int)

    # --- opponent inference if needed (same as before) ---
    if defense_team_col not in df.columns:
        teams_in_game = (
            df[[game_id_col, offense_team_col]].drop_duplicates().rename(columns={offense_team_col: "team_id"})
        )
        pairs = teams_in_game.merge(teams_in_game, on=game_id_col, suffixes=("", "_opp"))
        pairs = pairs[pairs["team_id"] != pairs["team_id_opp"]].copy()
        opp_map = pairs[[game_id_col, "team_id", "team_id_opp"]].drop_duplicates()
        df = df.merge(
            opp_map.rename(columns={"team_id": offense_team_col, "team_id_opp": defense_team_col}),
            on=[game_id_col, offense_team_col],
            how="left",
        )
        if df[defense_team_col].isna().any():
            raise ValueError(f"Could not infer {defense_team_col} for some rows. Provide it explicitly.")

    priors_by_season = global_priors = None
    off_team_prev = allowed_team_prev = None
    off_league_prev = allowed_league_prev = None
    off_global_mu = allowed_global_mu = None
    off_share_prev = allowed_share_prev = None
    off_share_league_prev = allowed_share_league_prev = None
    off_share_global_mu = allowed_share_global_mu = None
    if use_eb:
        # League-level EB strengths remain the prior "sample size" s in Beta(alpha,beta).
        priors_by_season, global_priors = _build_cat_priors_by_season(
            df,
            season_col="_season",
            made_col=made_col,
            shot_cat_col="shot_cat",
            prior_strength_2p=prior_strength_2p,
            prior_strength_3p=prior_strength_3p,
        )

        off_team_prev = _build_prev_team_cat_stats(
            df,
            season_col="_season",
            team_col=offense_team_col,
            shot_cat_col="shot_cat",
            made_col=made_col,
            prefix="off",
        )
        allowed_team_prev = _build_prev_team_cat_stats(
            df,
            season_col="_season",
            team_col=defense_team_col,
            shot_cat_col="shot_cat",
            made_col=made_col,
            prefix="allowed",
        )
        off_league_prev, off_global_mu = _build_prev_league_cat_means(
            df,
            season_col="_season",
            shot_cat_col="shot_cat",
            made_col=made_col,
            prefix="off",
        )
        allowed_league_prev, allowed_global_mu = _build_prev_league_cat_means(
            df,
            season_col="_season",
            shot_cat_col="shot_cat",
            made_col=made_col,
            prefix="allowed",
        )

        off_share_prev = _build_prev_team_share_stats(
            df,
            season_col="_season",
            team_col=offense_team_col,
            shot_cat_col="shot_cat",
            prefix="off",
        )
        allowed_share_prev = _build_prev_team_share_stats(
            df,
            season_col="_season",
            team_col=defense_team_col,
            shot_cat_col="shot_cat",
            prefix="allowed",
        )
        off_share_league_prev, off_share_global_mu = _build_prev_league_share_means(
            df,
            season_col="_season",
            shot_cat_col="shot_cat",
            prefix="off",
        )
        allowed_share_league_prev, allowed_share_global_mu = _build_prev_league_share_means(
            df,
            season_col="_season",
            shot_cat_col="shot_cat",
            prefix="allowed",
        )

    # --- aggregate by game-team-shot_cat (offense / defense-allowed) ---
    off_agg = (
        df.groupby([game_id_col, game_date_col, "_season", offense_team_col, "shot_cat"], as_index=False)
        .agg(off_att=("shot_cat", "size"), off_made=(made_col, "sum"))
        .rename(columns={offense_team_col: "team_id"})
    )
    def_agg = (
        df.groupby([game_id_col, game_date_col, "_season", defense_team_col, "shot_cat"], as_index=False)
        .agg(allowed_att=("shot_cat", "size"), allowed_made=(made_col, "sum"))
        .rename(columns={defense_team_col: "team_id"})
    )

    game_team = (
        pd.concat(
            [
                off_agg[[game_id_col, game_date_col, "_season", "team_id"]],
                def_agg[[game_id_col, game_date_col, "_season", "team_id"]],
            ],
            axis=0,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    cats = pd.DataFrame({"shot_cat": ["2P", "3P"]})
    gt = game_team.merge(cats, how="cross")

    gt = gt.merge(off_agg, on=[game_id_col, game_date_col, "_season", "team_id", "shot_cat"], how="left")
    gt = gt.merge(def_agg, on=[game_id_col, game_date_col, "_season", "team_id", "shot_cat"], how="left")
    for c in ["off_att", "off_made", "allowed_att", "allowed_made"]:
        gt[c] = gt[c].fillna(0).astype(int)

    # --- cumulative prior-to-game counts (no leakage) ---
    gt = gt.sort_values(["team_id", "shot_cat", game_date_col, game_id_col]).reset_index(drop=True)

    def _cumsum_shift(s: pd.Series) -> pd.Series:
        return s.cumsum().shift(1).fillna(0)

    gt["off_att_prev"] = gt.groupby(["team_id", "shot_cat"])["off_att"].transform(_cumsum_shift).astype(int)
    gt["off_made_prev"] = gt.groupby(["team_id", "shot_cat"])["off_made"].transform(_cumsum_shift).astype(int)
    gt["allowed_att_prev"] = gt.groupby(["team_id", "shot_cat"])["allowed_att"].transform(_cumsum_shift).astype(int)
    gt["allowed_made_prev"] = gt.groupby(["team_id", "shot_cat"])["allowed_made"].transform(_cumsum_shift).astype(int)

    if use_eb:
        gt["_season"] = gt["_season"].astype(int)
        gt = gt.merge(
            off_team_prev,
            on=["_season", "team_id", "shot_cat"],
            how="left",
        ).merge(
            allowed_team_prev,
            on=["_season", "team_id", "shot_cat"],
            how="left",
        ).merge(
            off_league_prev,
            on=["_season", "shot_cat"],
            how="left",
        ).merge(
            allowed_league_prev,
            on=["_season", "shot_cat"],
            how="left",
        )

        is2 = (gt["shot_cat"].values == "2P")
        k_cat = np.where(is2, float(team_weight_k_2p), float(team_weight_k_3p))
        s_cat = np.where(is2, float(prior_strength_2p), float(prior_strength_3p))

        off_l_mu = gt["off_league_mu_prev"].astype(float).values
        allowed_l_mu = gt["allowed_league_mu_prev"].astype(float).values
        off_l_mu = np.where(
            np.isfinite(off_l_mu),
            off_l_mu,
            np.where(is2, float(off_global_mu.get("2P", 0.5)), float(off_global_mu.get("3P", 0.5))),
        )
        allowed_l_mu = np.where(
            np.isfinite(allowed_l_mu),
            allowed_l_mu,
            np.where(is2, float(allowed_global_mu.get("2P", 0.5)), float(allowed_global_mu.get("3P", 0.5))),
        )

        off_mu_prior = _hybrid_prior_mu(
            team_mu=gt["off_team_mu_prev"].astype(float).values,
            team_n=gt["off_team_att_prev"].fillna(0.0).astype(float).values,
            league_mu=off_l_mu,
            continuity=continuity_off_pct,
            k=k_cat,
        )
        allowed_mu_prior = _hybrid_prior_mu(
            team_mu=gt["allowed_team_mu_prev"].astype(float).values,
            team_n=gt["allowed_team_att_prev"].fillna(0.0).astype(float).values,
            league_mu=allowed_l_mu,
            continuity=continuity_allowed_pct,
            k=k_cat,
        )
        alpha_off = off_mu_prior * s_cat
        beta_off = (1.0 - off_mu_prior) * s_cat
        alpha_allowed = allowed_mu_prior * s_cat
        beta_allowed = (1.0 - allowed_mu_prior) * s_cat

        gt["off_pct_eb"] = _eb_rate(gt["off_made_prev"].values, gt["off_att_prev"].values, alpha_off, beta_off)
        gt["allowed_pct_eb"] = _eb_rate(
            gt["allowed_made_prev"].values, gt["allowed_att_prev"].values, alpha_allowed, beta_allowed
        )
    else:
        gt["off_pct_eb"] = _safe_rate(gt["off_made_prev"].values, gt["off_att_prev"].values)
        gt["allowed_pct_eb"] = _safe_rate(gt["allowed_made_prev"].values, gt["allowed_att_prev"].values)

    # --- 3PA share (offense + allowed) with season-specific priors ---
    if add_3pa_share:
        gt_w = (
            gt.pivot_table(
                index=[game_id_col, game_date_col, "_season", "team_id"],
                columns="shot_cat",
                values=["off_att_prev", "allowed_att_prev"],
                aggfunc="first",
            )
            .reset_index()
        )
        gt_w.columns = [
            "_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else col
            for col in gt_w.columns
        ]
        if "_season" not in gt_w.columns:
            season_map = gt[[game_id_col, game_date_col, "team_id", "_season"]].drop_duplicates()
            gt_w = gt_w.merge(
                season_map,
                on=[game_id_col, game_date_col, "team_id"],
                how="left",
            )

        off2 = gt_w.get("off_att_prev_2P", pd.Series(0, index=gt_w.index)).astype(int)
        off3 = gt_w.get("off_att_prev_3P", pd.Series(0, index=gt_w.index)).astype(int)
        n = (off2 + off3).astype(int)
        k = off3.astype(int)

        if use_eb:
            gt_w["_season"] = gt_w["_season"].astype(int)
            gt_w = gt_w.merge(
                off_share_prev,
                on=["_season", "team_id"],
                how="left",
            ).merge(
                allowed_share_prev,
                on=["_season", "team_id"],
                how="left",
            ).merge(
                off_share_league_prev,
                on=["_season"],
                how="left",
            ).merge(
                allowed_share_league_prev,
                on=["_season"],
                how="left",
            )

            off_share_l_mu = gt_w["off_league_3pa_mu_prev"].astype(float).values
            allowed_share_l_mu = gt_w["allowed_league_3pa_mu_prev"].astype(float).values
            off_share_l_mu = np.where(np.isfinite(off_share_l_mu), off_share_l_mu, float(off_share_global_mu))
            allowed_share_l_mu = np.where(
                np.isfinite(allowed_share_l_mu), allowed_share_l_mu, float(allowed_share_global_mu)
            )
            s_share = float(prior_strength_3pa_share)

            off_share_mu_prior = _hybrid_prior_mu(
                team_mu=gt_w["off_team_3pa_mu_prev"].astype(float).values,
                team_n=gt_w["off_team_3pa_att_prev"].fillna(0.0).astype(float).values,
                league_mu=off_share_l_mu,
                continuity=continuity_off_3pa_share,
                k=float(team_weight_k_3pa_share),
            )
            allowed_share_mu_prior = _hybrid_prior_mu(
                team_mu=gt_w["allowed_team_3pa_mu_prev"].astype(float).values,
                team_n=gt_w["allowed_team_3pa_att_prev"].fillna(0.0).astype(float).values,
                league_mu=allowed_share_l_mu,
                continuity=continuity_allowed_3pa_share,
                k=float(team_weight_k_3pa_share),
            )
            off_a = off_share_mu_prior * s_share
            off_b = (1.0 - off_share_mu_prior) * s_share
            allowed_a = allowed_share_mu_prior * s_share
            allowed_b = (1.0 - allowed_share_mu_prior) * s_share

            gt_w["off_3pa_share_eb"] = _eb_rate(k.values, n.values, off_a, off_b)
            gt_w["off_2pa_share_eb"] = 1.0 - gt_w["off_3pa_share_eb"]
        else:
            gt_w["off_3pa_share_eb"] = _safe_rate(k.values, n.values)
            gt_w["off_2pa_share_eb"] = 1.0 - gt_w["off_3pa_share_eb"]

        # allowed share (defense)
        allow2 = gt_w.get("allowed_att_prev_2P", pd.Series(0, index=gt_w.index)).astype(int)
        allow3 = gt_w.get("allowed_att_prev_3P", pd.Series(0, index=gt_w.index)).astype(int)
        n_allow = (allow2 + allow3).astype(int)
        k_allow = allow3.astype(int)

        if use_eb:
            gt_w["allowed_3pa_share_eb"] = _eb_rate(k_allow.values, n_allow.values, allowed_a, allowed_b)
            gt_w["allowed_2pa_share_eb"] = 1.0 - gt_w["allowed_3pa_share_eb"]
        else:
            gt_w["allowed_3pa_share_eb"] = _safe_rate(k_allow.values, n_allow.values)
            gt_w["allowed_2pa_share_eb"] = 1.0 - gt_w["allowed_3pa_share_eb"]

        # merge back to gt
        gt = gt.merge(
            gt_w[
                [
                    game_id_col, game_date_col, "_season", "team_id",
                    "off_3pa_share_eb", "off_2pa_share_eb",
                    "allowed_3pa_share_eb", "allowed_2pa_share_eb",
                ]
            ],
            on=[game_id_col, game_date_col, "_season", "team_id"],
            how="left",
        )

    # --- pivot to wide for mersing into shot-level ---
    keep_cols = [
        game_id_col, game_date_col, "_season", "team_id", "shot_cat",
        "off_att_prev", "off_made_prev", "off_pct_eb",
        "allowed_att_prev", "allowed_made_prev", "allowed_pct_eb",
    ]
    if add_3pa_share:
        keep_cols += ["off_3pa_share_eb", "off_2pa_share_eb", "allowed_3pa_share_eb", "allowed_2pa_share_eb"]

    gt_small = gt[keep_cols].copy()

    wide = gt_small.pivot_table(
        index=[game_id_col, game_date_col, "_season", "team_id"],
        columns="shot_cat",
        values=[
            "off_att_prev", "off_made_prev", "off_pct_eb",
            "allowed_att_prev", "allowed_made_prev", "allowed_pct_eb",
        ],
        aggfunc="first",
    ).reset_index()

    wide.columns = [
        "_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else col
        for col in wide.columns
    ]
    if "_season" not in wide.columns:
        season_map = gt[[game_id_col, game_date_col, "team_id", "_season"]].drop_duplicates()
        wide = wide.merge(
            season_map,
            on=[game_id_col, game_date_col, "team_id"],
            how="left",
        )

    ren = {
        "off_pct_eb_2P": "off_2p_pct_eb",
        "off_pct_eb_3P": "off_3p_pct_eb",
        "allowed_pct_eb_2P": "allowed_2p_pct_eb",
        "allowed_pct_eb_3P": "allowed_3p_pct_eb",
        "off_att_prev_2P": "off_2p_att_prev",
        "off_att_prev_3P": "off_3p_att_prev",
        "off_made_prev_2P": "off_2p_made_prev",
        "off_made_prev_3P": "off_3p_made_prev",
        "allowed_att_prev_2P": "allowed_2p_att_prev",
        "allowed_att_prev_3P": "allowed_3p_att_prev",
        "allowed_made_prev_2P": "allowed_2p_made_prev",
        "allowed_made_prev_3P": "allowed_3p_made_prev",
    }
    wide = wide.rename(columns=ren)

    if add_3pa_share:
        share = (
            gt_small[
                [game_id_col, game_date_col, "_season", "team_id",
                 "off_3pa_share_eb", "off_2pa_share_eb",
                 "allowed_3pa_share_eb", "allowed_2pa_share_eb"]
            ]
            .drop_duplicates()
        )
        merge_keys = [game_id_col, game_date_col, "team_id"]
        if "_season" in wide.columns and "_season" in share.columns:
            merge_keys.append("_season")
        wide = wide.merge(share, on=merge_keys, how="left")

    # own = offense team
    own = wide.rename(columns={"team_id": offense_team_col})
    own_pref = {
        c: f"own_{c}"
        for c in own.columns
        if c not in [game_id_col, game_date_col, "_season", offense_team_col]
    }
    own = own.rename(columns=own_pref)

    # opp = defense team
    opp = wide.rename(columns={"team_id": defense_team_col})
    opp_pref = {
        c: f"opp_{c}"
        for c in opp.columns
        if c not in [game_id_col, game_date_col, "_season", defense_team_col]
    }
    opp = opp.rename(columns=opp_pref)

    out = (
        df.merge(own, on=[game_id_col, game_date_col, "_season", offense_team_col], how="left", validate="m:1")
          .merge(opp, on=[game_id_col, game_date_col, "_season", defense_team_col], how="left", validate="m:1")
    )
    return out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Build team-level shot context features (optionally EB).")
    parser.add_argument("--start-season", type=int, default=2000, help="First season to output.")
    parser.add_argument("--end-season", type=int, default=2024, help="Last season to output.")
    parser.add_argument(
        "--input-start-season",
        type=int,
        default=1999,
        help="First season to load for priors.",
    )
    parser.add_argument(
        "--input-end-season",
        type=int,
        default=2024,
        help="Last season to load for priors.",
    )
    parser.add_argument(
        "--seasontype",
        type=str,
        default="rs",
        choices=["rs"],
        help="Season type (used for defaults only).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/analysis/team_shot_stats_2000_2024.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--no-eb",
        action="store_true",
        help="Disable EB smoothing (use raw cumulative rates and shares).",
    )
    args = parser.parse_args()

    base_dir = Path("data/nba_raw")
    seasons: List[int] = list(range(args.input_start_season, args.input_end_season + 1))
    shot_frames: List[pd.DataFrame] = []

    for season in seasons:
        candidates = [base_dir / f"shotdetail_{season}.csv"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        df = pd.read_csv(path, low_memory=False)
        df["__source_season"] = season
        shot_frames.append(df)

    if not shot_frames:
        raise FileNotFoundError("No shotdetail files found for the requested seasons.")

    shots_df = pd.concat(shot_frames, ignore_index=True)
    enriched = add_team_eb_context_features(shots_df, use_eb=not args.no_eb)

    if "_season" in enriched.columns:
        enriched = enriched[
            (enriched["_season"] >= args.start_season)
            & (enriched["_season"] <= args.end_season)
        ].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(out_path, index=False)
    print(f"[INFO] Saved enriched shot stats to {out_path} ({len(enriched):,} rows)")
