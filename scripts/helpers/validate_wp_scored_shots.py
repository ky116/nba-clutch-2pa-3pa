#!/usr/bin/env python3
"""Validate WP-scored shot states after RS/PO panel regeneration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Check WP-scored shot states for terminal-shot retention, "
            "delta_wp identity, and late trailing problem-region counts."
        )
    )
    ap.add_argument(
        "--with-wp",
        default="data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz",
        help="Scored shot-state CSV produced by scripts/core/fit_wp_and_score_shots.r.",
    )
    ap.add_argument(
        "--panel",
        default="data/analysis/shotchoice_panel_clutch_rs.parquet",
        help="Shotchoice panel parquet produced by scripts/core/build_shotchoice_panel_from_wp.py.",
    )
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--identity-tol", type=float, default=1e-10)
    ap.add_argument("--skip-panel", action="store_true")
    return ap.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[error] missing file: {path}")


def summarize_with_wp(path: Path, chunksize: int, identity_tol: float) -> None:
    usecols = [
        "GAME_ID", "GAME_EVENT_ID", "shot_zone_choice", "shot_made",
        "next_type", "next_is_terminal", "before_time_left_game",
        "before_score_diff", "before_home_possession", "wp_before",
        "wp_next", "delta_wp", "final_home_win",
    ]
    total = terminal = terminal_with_wp = 0
    max_identity_err = 0.0
    identity_bad = wp_before_oob = wp_next_oob = delta_oob = 0
    region_total = region_terminal = 0
    region_by_choice: dict[str, int] = {}
    region_terminal_by_choice: dict[str, int] = {}
    delta_sum = region_delta_sum = 0.0
    delta_n = region_delta_n = 0

    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        total += len(ch)
        nt = pd.to_numeric(ch["next_is_terminal"], errors="coerce").fillna(0).astype(int)
        wb = pd.to_numeric(ch["wp_before"], errors="coerce")
        wn = pd.to_numeric(ch["wp_next"], errors="coerce")
        dw = pd.to_numeric(ch["delta_wp"], errors="coerce")
        fhw = pd.to_numeric(ch["final_home_win"], errors="coerce")

        is_terminal = (nt == 1) | (ch["next_type"].astype(str) == "terminal")
        terminal += int(is_terminal.sum())
        terminal_with_wp += int((is_terminal & wn.notna()).sum())

        err = (dw - (wn - wb)).abs()
        finite_err = err[np.isfinite(err)]
        if len(finite_err) > 0:
            max_identity_err = max(max_identity_err, float(finite_err.max()))
            identity_bad += int((finite_err > identity_tol).sum())

        wp_before_oob += int(((wb < -1e-9) | (wb > 1 + 1e-9)).sum())
        wp_next_oob += int(((wn < -1e-9) | (wn > 1 + 1e-9)).sum())
        delta_oob += int(((dw < -1 - 1e-9) | (dw > 1 + 1e-9)).sum())

        valid_dw = dw[np.isfinite(dw)]
        delta_sum += float(valid_dw.sum())
        delta_n += int(valid_dw.size)

        home_poss = pd.to_numeric(ch["before_home_possession"], errors="coerce")
        sign = np.where(home_poss == 1, 1.0, np.where(home_poss == 0, -1.0, np.nan))
        off_score_diff = pd.to_numeric(ch["before_score_diff"], errors="coerce") * sign
        time_left = pd.to_numeric(ch["before_time_left_game"], errors="coerce")
        region = (time_left <= 15) & (off_score_diff > -5) & (off_score_diff <= -2)

        region_total += int(region.sum())
        region_terminal += int((region & is_terminal).sum())
        r_dw = dw[region & np.isfinite(dw)]
        region_delta_sum += float(r_dw.sum())
        region_delta_n += int(r_dw.size)

        choice = ch["shot_zone_choice"].astype(str)
        for key, val in choice[region].value_counts(dropna=False).items():
            region_by_choice[key] = region_by_choice.get(key, 0) + int(val)
        for key, val in choice[region & is_terminal].value_counts(dropna=False).items():
            region_terminal_by_choice[key] = region_terminal_by_choice.get(key, 0) + int(val)

        mismatch = is_terminal & wn.notna() & fhw.notna() & ((wn - fhw).abs() > identity_tol)
        if mismatch.any():
            raise SystemExit(f"[error] terminal wp_next differs from final_home_win: {int(mismatch.sum()):,} rows")

    if total == 0:
        raise SystemExit("[error] scored shot file has no rows")

    print("[with_wp]")
    print(f"  path: {path}")
    print(f"  rows: {total:,}")
    print(f"  terminal rows kept/scored: {terminal_with_wp:,} / {terminal:,}")
    print(f"  max |delta_wp - (wp_next - wp_before)|: {max_identity_err:.12g}")
    print(f"  identity violations > {identity_tol:g}: {identity_bad:,}")
    print(f"  out of bounds: wp_before={wp_before_oob:,} wp_next={wp_next_oob:,} delta_wp={delta_oob:,}")
    print(f"  mean(delta_wp): {delta_sum / max(delta_n, 1):.8f}")
    print("  problem region: before_time_left_game <= 15 and offense -5 < score_diff <= -2")
    print(f"    rows: {region_total:,}")
    print(f"    terminal rows: {region_terminal:,}")
    print(f"    mean(delta_wp): {region_delta_sum / max(region_delta_n, 1):.8f}")
    print(f"    rows by choice: {region_by_choice}")
    print(f"    terminal rows by choice: {region_terminal_by_choice}")

    if terminal == 0 or terminal_with_wp == 0:
        raise SystemExit("[error] no terminal rows were kept with wp_next; terminal-shot fix is not reflected")
    if identity_bad > 0 or wp_before_oob > 0 or wp_next_oob > 0 or delta_oob > 0:
        raise SystemExit("[error] scored shot validation failed")


def summarize_panel(path: Path) -> None:
    df = pd.read_parquet(path)
    print("[panel]")
    print(f"  path: {path}")
    print(f"  rows: {len(df):,}")
    for col in ("season", "shot_zone_choice", "delta_wp", "time_left_game", "score_diff"):
        if col not in df.columns:
            raise SystemExit(f"[error] panel missing required column: {col}")

    time_left = pd.to_numeric(df["time_left_game"], errors="coerce")
    score_diff = pd.to_numeric(df["score_diff"], errors="coerce")
    region = time_left.le(15) & score_diff.gt(-5) & score_diff.le(-2)
    print(f"  seasons: {int(df['season'].min())}-{int(df['season'].max())}")
    print(f"  choices: {df['shot_zone_choice'].value_counts(dropna=False).to_dict()}")
    print("  problem region: time_left_game <= 15 and -5 < score_diff <= -2")
    print(f"    rows: {int(region.sum()):,}")
    print(f"    rows by choice: {df.loc[region, 'shot_zone_choice'].value_counts(dropna=False).to_dict()}")
    print(f"    mean(delta_wp): {pd.to_numeric(df.loc[region, 'delta_wp'], errors='coerce').mean():.8f}")


def main() -> None:
    args = parse_args()
    with_wp = Path(args.with_wp)
    panel = Path(args.panel)
    require_file(with_wp)
    summarize_with_wp(with_wp, args.chunksize, args.identity_tol)
    if not args.skip_panel:
        require_file(panel)
        summarize_panel(panel)
    print("[ok] WP-scored shot validation completed")


if __name__ == "__main__":
    main()
