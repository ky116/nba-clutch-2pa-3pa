#!/usr/bin/env python3
"""Validate the direct-grid WP model-dependence sensitivity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["time_left_game", "score_diff"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--main-surface",
        type=Path,
        default=Path("results/full_data_ensemble_state_fixed_loso/full_data_t30_300_cate_surface_equal_weight.csv"),
    )
    p.add_argument(
        "--bias-surface",
        type=Path,
        default=Path(
            "results/wp_calibration/model_dependence_refit/ensemble/"
            "wp_residual_t30_300_cate_surface_equal_weight.csv"
        ),
    )
    p.add_argument(
        "--sensitivity-surface",
        type=Path,
        default=Path(
            "results/wp_calibration/model_dependence_sensitivity/"
            "wp_model_dependence_sensitivity_surface.csv"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/wp_calibration/model_dependence_refit/validation.json"),
    )
    return p.parse_args()


def normalized_grid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing grid columns: {missing}")
    out = df[KEYS].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any() or out.duplicated(KEYS).any():
        raise ValueError(f"{label} has missing or duplicate grid keys")
    return out.sort_values(KEYS).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    main_surface = pd.read_csv(args.main_surface)
    bias_surface = pd.read_csv(args.bias_surface)
    sensitivity = pd.read_csv(args.sensitivity_surface)

    grids = {
        "main": normalized_grid(main_surface, "main surface"),
        "bias": normalized_grid(bias_surface, "bias surface"),
        "sensitivity": normalized_grid(sensitivity, "sensitivity surface"),
    }
    expected_times = np.linspace(30.0, 300.0, 25)
    expected_scores = np.arange(-10.0, 11.0)
    expected = pd.MultiIndex.from_product(
        [expected_times, expected_scores], names=KEYS
    ).to_frame(index=False).sort_values(KEYS).reset_index(drop=True)

    for label, grid in grids.items():
        if len(grid) != 525 or not np.allclose(grid.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-6):
            raise ValueError(f"{label} grid is not the required 25 x 21 direct grid (rows={len(grid)})")

    required = ["tau_wp", "b_hat", "tau_sensitivity"]
    missing = [c for c in required if c not in sensitivity.columns]
    if missing:
        raise ValueError(f"sensitivity surface missing columns: {missing}")
    numeric = sensitivity[required].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("sensitivity surface has missing effect values")
    if not np.allclose(
        numeric["tau_sensitivity"], numeric["tau_wp"] - numeric["b_hat"], rtol=0, atol=1e-12
    ):
        raise ValueError("tau_sensitivity != tau_wp - b_hat")

    result = {
        "valid": True,
        "n_time_points": 25,
        "time_min": 30.0,
        "time_max": 300.0,
        "n_score_points": 21,
        "score_min": -10.0,
        "score_max": 10.0,
        "n_cells": 525,
        "interpolation": False,
        "subtraction": "tau_sensitivity = tau_wp - b_hat",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
