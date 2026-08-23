from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUT = Path("results/slide_update_assets_20260623/forced_late_down3_wp_cells.csv")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type: {path}")


def pick_column(df: pd.DataFrame, explicit: str | None, candidates: list[str], label: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"{label} column not found: {explicit}")
        return explicit
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not infer {label} column. Pass it explicitly. "
        f"Tried: {', '.join(candidates)}"
    )


def normalize_shot_type(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.upper().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype=object)
    out[text.isin(["2", "2P", "2PT", "TWO", "TWO_POINT", "TWO POINT"])] = "2P"
    out[text.isin(["3", "3P", "3PT", "THREE", "THREE_POINT", "THREE POINT"])] = "3P"
    out[text.isin(["RESTRICTED AREA", "IN THE PAINT (NON-RA)", "MID-RANGE"])] = "2P"
    out[text.isin(["ABOVE THE BREAK 3", "LEFT CORNER 3", "RIGHT CORNER 3", "CORNER 3"])] = "3P"
    numeric = pd.to_numeric(series, errors="coerce")
    out[numeric.eq(2)] = "2P"
    out[numeric.eq(3)] = "3P"
    return out


def shotdetail_candidates(raw_dir: Path, season: int, seasontype: str) -> list[Path]:
    return [
        raw_dir / f"shotdetail_{season}.csv",
        raw_dir / f"shotdetail_rs_{season}.csv",
        raw_dir / f"shotdetail_{season}_rs.csv",
    ]


def add_shot_type_from_shotdetail(df: pd.DataFrame, raw_dir: Path, seasontype: str | None) -> pd.DataFrame:
    required = {"GAME_ID", "GAME_EVENT_ID", "season"}
    if not required.issubset(df.columns):
        return df

    seasons = sorted(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int).unique())
    maps = []
    for season in seasons:
        st = seasontype
        if st is None and "seasontype" in df.columns:
            vals = df.loc[pd.to_numeric(df["season"], errors="coerce").eq(season), "seasontype"].dropna().astype(str)
            st = vals.iloc[0] if len(vals) else "rs"
        st = st or "rs"
        for cand in shotdetail_candidates(raw_dir, season, st):
            if cand.exists():
                shots = pd.read_csv(
                    cand,
                    usecols=lambda c: c in {"GAME_ID", "GAME_EVENT_ID", "SHOT_ZONE_BASIC", "SHOT_ATTEMPTED_FLAG"},
                    low_memory=False,
                )
                if {"GAME_ID", "GAME_EVENT_ID", "SHOT_ZONE_BASIC"}.issubset(shots.columns):
                    if "SHOT_ATTEMPTED_FLAG" in shots.columns:
                        shots = shots[pd.to_numeric(shots["SHOT_ATTEMPTED_FLAG"], errors="coerce").eq(1)]
                    shots = shots[["GAME_ID", "GAME_EVENT_ID", "SHOT_ZONE_BASIC"]].copy()
                    shots["GAME_ID"] = shots["GAME_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
                    shots["GAME_EVENT_ID"] = pd.to_numeric(shots["GAME_EVENT_ID"], errors="coerce").astype("Int64")
                    shots["_shot_type_from_detail"] = normalize_shot_type(shots["SHOT_ZONE_BASIC"])
                    maps.append(shots[["GAME_ID", "GAME_EVENT_ID", "_shot_type_from_detail"]])
                break
    if not maps:
        return df

    shot_map = pd.concat(maps, ignore_index=True).dropna(subset=["_shot_type_from_detail"])
    shot_map = shot_map.drop_duplicates(subset=["GAME_ID", "GAME_EVENT_ID"])
    out = df.copy()
    out["GAME_ID"] = out["GAME_ID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    out["GAME_EVENT_ID"] = pd.to_numeric(out["GAME_EVENT_ID"], errors="coerce").astype("Int64")
    return out.merge(shot_map, on=["GAME_ID", "GAME_EVENT_ID"], how="left")


def as_bool_made(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.eq(1)
    text = series.astype(str).str.lower().str.strip()
    return text.isin(["made", "make", "hit", "true", "yes", "1"])


def terminal_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.eq(1)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "terminal"])


def top_values(group: pd.DataFrame, col: str | None, max_items: int = 4) -> str:
    if not col or col not in group.columns or len(group) == 0:
        return ""
    counts = group[col].astype(str).value_counts(dropna=False).head(max_items)
    return "; ".join(f"{idx}:{int(val)}" for idx, val in counts.items())


def summarize(
    group: pd.DataFrame,
    wp_col: str,
    win_col: str,
    terminal_col: str | None,
    next_type_col: str | None,
    off_reb_col: str | None,
) -> dict[str, float | str]:
    model_wp = pd.to_numeric(group[wp_col], errors="coerce")
    empirical = pd.to_numeric(group[win_col], errors="coerce")
    out = {
        "n": int(len(group)),
        "model_next_wp_mean": float(model_wp.mean()),
        "model_next_wp_sd": float(model_wp.std(ddof=1)) if len(group) > 1 else np.nan,
        "empirical_win_rate": float(empirical.mean()),
        "wp_minus_empirical": float(model_wp.mean() - empirical.mean()),
    }
    if terminal_col:
        out["terminal_share"] = float(terminal_mask(group[terminal_col]).mean())
    else:
        out["terminal_share"] = np.nan
    if off_reb_col and off_reb_col in group.columns:
        out["off_reb_share"] = float(as_bool_made(group[off_reb_col]).mean())
    else:
        out["off_reb_share"] = np.nan
    out["next_type_top"] = top_values(group, next_type_col)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit down-3, <=5s forced-rule cells by made/missed and 2P/3P. "
            "The output is intended to diagnose whether the late forced-3P penalty "
            "is driven by made-2P continuation WP calibration."
        )
    )
    parser.add_argument("input", type=Path, help="Shot/play-level CSV or parquet file.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--score-diff-col", default=None)
    parser.add_argument("--shot-type-col", default=None)
    parser.add_argument("--made-col", default=None)
    parser.add_argument("--next-wp-col", default=None)
    parser.add_argument("--win-col", default=None)
    parser.add_argument("--terminal-col", default=None)
    parser.add_argument("--shotdetail-dir", type=Path, default=Path("data/nba_raw"))
    parser.add_argument("--seasontype", default="rs", help="Season type; this release uses regular season only.")
    parser.add_argument("--time-threshold", type=float, default=5.0)
    parser.add_argument("--score-diff", type=float, default=-3.0)
    args = parser.parse_args()

    df = read_table(args.input)

    time_col = pick_column(
        df,
        args.time_col,
        ["time_left", "seconds_left", "sec_left", "game_seconds_remaining", "time_remaining", "time_left_game", "before_time_left_game"],
        "time-left",
    )
    score_col = pick_column(
        df,
        args.score_diff_col,
        ["score_diff", "score_margin", "margin", "score_difference", "before_score_diff"],
        "score-diff",
    )
    shot_col = pick_column(
        df,
        args.shot_type_col,
        ["shot_type", "shot_value", "shot_points", "attempt_type", "action", "shot_zone_choice"],
        "shot-type",
    )
    made_col = pick_column(
        df,
        args.made_col,
        ["made", "shot_made", "is_made", "make", "fgm", "shot_result"],
        "made",
    )
    wp_col = pick_column(
        df,
        args.next_wp_col,
        ["next_wp", "wp_next", "next_state_wp", "pred_next_wp", "next_win_prob"],
        "next-state WP",
    )
    win_col = pick_column(
        df,
        args.win_col,
        ["won", "win", "team_win", "eventual_win", "outcome_win", "final_home_win"],
        "eventual-win",
    )

    terminal_col = args.terminal_col
    if terminal_col is None:
        for candidate in ["terminal", "is_terminal", "next_terminal", "terminal_next_state", "next_is_terminal"]:
            if candidate in df.columns:
                terminal_col = candidate
                break
    elif terminal_col not in df.columns:
        raise KeyError(f"terminal column not found: {terminal_col}")

    next_type_col = "next_type" if "next_type" in df.columns else None
    off_reb_col = "after_off_reb" if "after_off_reb" in df.columns else None

    time_left = pd.to_numeric(df[time_col], errors="coerce")
    score_diff = pd.to_numeric(df[score_col], errors="coerce")
    if score_col == "before_score_diff" and "before_home_possession" in df.columns:
        home_offense = pd.to_numeric(df["before_home_possession"], errors="coerce")
        score_diff = pd.Series(
            np.where(home_offense.eq(1), score_diff, np.where(home_offense.eq(0), -score_diff, np.nan)),
            index=df.index,
        )
    if wp_col == "wp_next" and "before_home_possession" in df.columns:
        home_wp_next = pd.to_numeric(df[wp_col], errors="coerce")
        home_offense = pd.to_numeric(df["before_home_possession"], errors="coerce")
        df["_wp_next_offense"] = np.where(
            home_offense.eq(1),
            home_wp_next,
            np.where(home_offense.eq(0), 1.0 - home_wp_next, np.nan),
        )
        wp_col = "_wp_next_offense"
    shot_type = normalize_shot_type(df[shot_col])
    if not shot_type.isin(["2P", "3P"]).any():
        df = add_shot_type_from_shotdetail(df, args.shotdetail_dir, args.seasontype)
        if "_shot_type_from_detail" in df.columns:
            shot_type = df["_shot_type_from_detail"]
    made = as_bool_made(df[made_col])
    if win_col == "final_home_win" and "before_home_possession" in df.columns:
        home_win = pd.to_numeric(df[win_col], errors="coerce")
        home_offense = pd.to_numeric(df["before_home_possession"], errors="coerce")
        df["_offense_win"] = np.where(
            home_offense.eq(1),
            home_win,
            np.where(home_offense.eq(0), 1.0 - home_win, np.nan),
        )
        win_col = "_offense_win"

    region = df[
        time_left.le(args.time_threshold)
        & score_diff.eq(args.score_diff)
        & shot_type.isin(["2P", "3P"])
    ].copy()
    region["_shot_type_norm"] = shot_type.loc[region.index]
    region["_made_norm"] = made.loc[region.index]

    rows = []
    for shot in ["2P", "3P"]:
        for made_value, made_label in [(True, "made"), (False, "missed")]:
            raw_cell = region[
                region["_shot_type_norm"].eq(shot)
                & region["_made_norm"].eq(made_value)
            ]
            cell = raw_cell
            analysis_sample = "all attempts"
            if made_value and terminal_col:
                cell = raw_cell[~terminal_mask(raw_cell[terminal_col])]
                analysis_sample = "nonterminal next state"
            stats = (
                summarize(cell, wp_col, win_col, terminal_col, next_type_col, off_reb_col)
                if len(cell)
                else {
                    "n": 0,
                    "model_next_wp_mean": np.nan,
                    "model_next_wp_sd": np.nan,
                    "empirical_win_rate": np.nan,
                    "wp_minus_empirical": np.nan,
                    "terminal_share": np.nan,
                    "off_reb_share": np.nan,
                    "next_type_top": "",
                }
            )
            rows.append(
                {
                    "condition": f"down 3, <=5s, {made_label} {shot}"
                    + (", nonterminal next state" if made_value and terminal_col else ""),
                    "shot_type": shot,
                    "made": made_label,
                    "analysis_sample": analysis_sample,
                    "raw_n": int(len(raw_cell)),
                    **stats,
                }
            )

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
