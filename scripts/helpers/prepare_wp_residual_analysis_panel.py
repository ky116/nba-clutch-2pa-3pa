#!/usr/bin/env python3
"""Attach the offense-oriented next-state WP residual to the main CATE panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fit_wp_residual_differential_surface import attach_residual, load_panel, load_residuals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=Path("data/analysis/shotchoice_panel_clutch_rs.parquet"))
    p.add_argument(
        "--with-wp",
        type=Path,
        default=Path("data/wp/shot_decision_states_2000_2024_rs_with_wp.csv.gz"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/wp_calibration/model_dependence_refit/wp_residual_analysis_panel.parquet"),
    )
    p.add_argument("--metadata", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_panel(args.panel)
    residuals = load_residuals(args.with_wp)
    merged, metadata = attach_residual(panel, residuals)
    merged = merged.dropna(subset=["wp_residual_offense"]).reset_index(drop=True)

    metadata.update(
        {
            "rows_written": int(len(merged)),
            "outcome": "wp_residual_offense",
            "residual_definition": "offense_wp_next - offense_final_win",
            "panel": str(args.panel),
            "with_wp": str(args.with_wp),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[saved] {args.output} rows={len(merged):,}")
    print(f"[saved] {metadata_path}")
    print(pd.Series(metadata).to_string())


if __name__ == "__main__":
    main()
