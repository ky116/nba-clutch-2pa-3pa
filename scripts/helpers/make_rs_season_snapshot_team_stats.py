#!/usr/bin/env python3
"""Build one end-of-regular-season team-stat snapshot per team and season."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    df = pd.read_parquet(source)
    season_col = "own_season" if "own_season" in df.columns else "_season"
    feature_cols = [
        col for col in df.columns
        if col.startswith("own_") and col != season_col
    ]
    required = {season_col, "TEAM_ID", "GAME_DATE", "GAME_ID", "GAME_EVENT_ID"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    if not feature_cols:
        raise SystemExit("no own_* team-stat features found")

    snapshot = df[
        [season_col, "TEAM_ID", "GAME_DATE", "GAME_ID", "GAME_EVENT_ID"]
        + feature_cols
    ].copy()
    snapshot[season_col] = pd.to_numeric(
        snapshot[season_col], errors="coerce"
    ).astype("Int64")
    snapshot["TEAM_ID"] = pd.to_numeric(
        snapshot["TEAM_ID"], errors="coerce"
    ).astype("Int64")
    snapshot["GAME_DATE"] = pd.to_datetime(snapshot["GAME_DATE"], errors="coerce")
    snapshot["GAME_EVENT_ID"] = pd.to_numeric(
        snapshot["GAME_EVENT_ID"], errors="coerce"
    )
    snapshot = snapshot.dropna(subset=[season_col, "TEAM_ID"])
    snapshot = snapshot.sort_values(
        [season_col, "TEAM_ID", "GAME_DATE", "GAME_ID", "GAME_EVENT_ID"]
    )
    snapshot = snapshot.drop_duplicates(
        subset=[season_col, "TEAM_ID"], keep="last"
    )

    snapshot = snapshot.rename(columns={
        season_col: "season",
        "TEAM_ID": "team_id",
        **{col: col.removeprefix("own_") for col in feature_cols},
    })
    snapshot = snapshot[
        ["season", "team_id"]
        + [col.removeprefix("own_") for col in feature_cols]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(output, index=False)
    print(f"[done] rows={len(snapshot):,} output={output}")


if __name__ == "__main__":
    main()
