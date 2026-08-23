#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preflight checks for run_nested_walk_forward.py settings.

Goal:
- Fail fast before expensive nested walk-forward runs.
- Detect common configuration/data issues:
  - no outer/inner splits
  - missing columns / features
  - invalid treatment mapping and labels
  - sparse treatment counts per split (risk for fallback-heavy training)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from treatment_utils import add_treatment_scheme_arg, apply_treatment_scheme


@dataclass(frozen=True)
class SplitRange:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def iter_outer_splits(
    train_start: int,
    train_end_init: int,
    test_span: int,
    step: int,
    max_season: int,
) -> Iterable[SplitRange]:
    train_end = train_end_init
    while True:
        test_start = train_end + 1
        test_end = test_start + test_span - 1
        if test_end > max_season:
            break
        yield SplitRange(train_start, train_end, test_start, test_end)
        train_end += step


def iter_inner_splits(
    train_start: int,
    train_end_max: int,
    train_init_span: int,
    block_span: int,
    step: int,
) -> Iterable[SplitRange]:
    train_end = train_start + train_init_span - 1
    while True:
        test_start = train_end + 1
        test_end = test_start + block_span - 1
        if test_end > train_end_max:
            break
        yield SplitRange(train_start, train_end, test_start, test_end)
        train_end += step


def default_feature_cols(df: pd.DataFrame, include_team_ids: bool = False) -> List[str]:
    candidate = [
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
    if include_team_ids:
        candidate.extend(["offense_team", "defense_team"])
    redundant_share_cols = {
        "own_allowed_2pa_share_eb",
        "opp_allowed_2pa_share_eb",
    }
    team_feature_cols = [
        c for c in df.columns
        if (c.startswith("own_") or c.startswith("opp_")) and c.endswith("_eb")
        and c not in redundant_share_cols
    ]
    for c in team_feature_cols:
        if c not in candidate:
            candidate.append(c)
    return [c for c in candidate if c in df.columns]


def _parse_csv_cols(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _summarize_counts(df: pd.DataFrame, treat_col: str) -> str:
    vc = df[treat_col].astype(str).value_counts(dropna=False).sort_index()
    return ", ".join([f"{k}:{int(v)}" for k, v in vc.items()])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preflight for nested walk-forward run.")
    p.add_argument("--input", required=True, help="panel parquet")
    p.add_argument("--season-col", default="season")
    p.add_argument("--treat-col", default="shot_zone_choice")
    p.add_argument("--outcome-col", default="delta_wp")
    p.add_argument("--features", default=None, help="comma-separated feature columns")
    p.add_argument("--include-team-ids", action="store_true")
    p.add_argument("--treat-a", default="three-point")
    p.add_argument("--treat-b", default="two-point")
    add_treatment_scheme_arg(p, default="binary")

    p.add_argument("--train-start", type=int, default=2000)
    p.add_argument("--train-end-init", type=int, default=2009)
    p.add_argument("--test-span", type=int, default=3)
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--max-season", type=int, default=None)

    p.add_argument("--inner-train-init-span", type=int, default=4)
    p.add_argument("--inner-block-span", type=int, default=3)
    p.add_argument("--inner-step", type=int, default=3)
    p.add_argument("--min-samples-per-treat", type=int, default=200)
    p.add_argument("--strict", action="store_true", help="treat warnings as errors")
    return p


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[info] ignored unknown args for preflight: {' '.join(unknown)}")

    errors: List[str] = []
    warnings: List[str] = []

    df = pd.read_parquet(args.input)
    print(f"[info] input rows={len(df):,}, cols={len(df.columns)}")

    # Basic column checks
    for c in [args.season_col, args.treat_col, args.outcome_col]:
        if c not in df.columns:
            errors.append(f"missing required column: {c}")
    if errors:
        for e in errors:
            print(f"[error] {e}")
        raise SystemExit(2)

    # Normalize season
    season_num = pd.to_numeric(df[args.season_col], errors="coerce")
    bad_season = int(season_num.isna().sum())
    if bad_season > 0:
        warnings.append(f"season has {bad_season} non-numeric rows; they will be dropped in run.")
    df = df.loc[season_num.notna()].copy()
    df[args.season_col] = season_num.loc[season_num.notna()].astype(int)
    df = df.rename(columns={args.season_col: "season"})

    # Treatment mapping
    df = apply_treatment_scheme(
        df,
        treat_col=args.treat_col,
        scheme=args.treatment_scheme,
        out_col=args.treat_col,
        drop_unknown=True,
    )
    if df.empty:
        errors.append("no rows left after treatment mapping")
    else:
        print(f"[info] after mapping rows={len(df):,}")
        print(f"[info] treatment counts: {_summarize_counts(df, args.treat_col)}")

    # Feature checks
    feature_cols = _parse_csv_cols(args.features) if args.features else default_feature_cols(
        df, include_team_ids=args.include_team_ids
    )
    if not feature_cols:
        errors.append("feature list is empty")
    miss_feat = [c for c in feature_cols if c not in df.columns]
    if miss_feat:
        errors.append(f"missing feature columns: {miss_feat}")
    else:
        print(f"[info] n_features={len(feature_cols)}")

    # Treatment labels
    levels = sorted(df[args.treat_col].astype(str).dropna().unique().tolist())
    if args.treat_a not in levels or args.treat_b not in levels:
        errors.append(
            f"treat labels not found after mapping: treat_a={args.treat_a}, treat_b={args.treat_b}, levels={levels}"
        )

    # Season range and outer splits
    if df.empty:
        errors.append("empty dataframe after preprocessing")
    else:
        smin = int(df["season"].min())
        smax = int(df["season"].max())
        max_season = args.max_season if args.max_season is not None else smax
        print(f"[info] season range in data: {smin}-{smax}, max_season used: {max_season}")
        outer = list(
            iter_outer_splits(
                train_start=args.train_start,
                train_end_init=args.train_end_init,
                test_span=args.test_span,
                step=args.step,
                max_season=max_season,
            )
        )
        if not outer:
            errors.append("no outer splits created (check train-end-init / test-span / max-season)")
        else:
            print(f"[info] outer_splits={len(outer)}")
            print(
                "[info] first outer split: "
                f"train {outer[0].train_start}-{outer[0].train_end}, "
                f"test {outer[0].test_start}-{outer[0].test_end}"
            )
            print(
                "[info] last outer split: "
                f"train {outer[-1].train_start}-{outer[-1].train_end}, "
                f"test {outer[-1].test_start}-{outer[-1].test_end}"
            )

            # Per-outer checks
            for o in outer:
                tag = f"train{o.train_start}_{o.train_end}_test{o.test_start}_{o.test_end}"
                df_tr = df[(df["season"] >= o.train_start) & (df["season"] <= o.train_end)]
                df_te = df[(df["season"] >= o.test_start) & (df["season"] <= o.test_end)]
                if df_tr.empty or df_te.empty:
                    errors.append(f"{tag}: empty train/test rows")
                    continue

                inner = list(
                    iter_inner_splits(
                        train_start=o.train_start,
                        train_end_max=o.train_end,
                        train_init_span=args.inner_train_init_span,
                        block_span=args.inner_block_span,
                        step=args.inner_step,
                    )
                )
                if not inner:
                    errors.append(f"{tag}: no inner splits created")
                    continue

                # sparsity warning
                vc = df_tr[args.treat_col].astype(str).value_counts()
                low = vc[vc < args.min_samples_per_treat]
                if not low.empty:
                    warnings.append(
                        f"{tag}: low treatment counts under min-samples-per-treat="
                        f"{args.min_samples_per_treat} -> {dict((k, int(v)) for k, v in low.items())}"
                    )

    # Report
    if warnings:
        print("\n[warn] potential issues:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n[error] blocking issues:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(2)

    if args.strict and warnings:
        print("\n[error] --strict enabled and warnings found.")
        raise SystemExit(3)

    print("\n[ok] preflight passed.")


if __name__ == "__main__":
    main()
