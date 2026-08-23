from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("results/late_clock_diagnostics/forced_late_three_raw_rows.parquet")
OUT = Path("results/late_clock_diagnostics/down3_le5_four_cell_uncertainty.csv")
BOOT_OUT = Path("results/late_clock_diagnostics/down3_le5_four_cell_cluster_bootstrap_draws.csv")


def wilson_interval(successes: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return float(center - half), float(center + half)


def cluster_bootstrap(
    g: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int = 5000,
) -> pd.DataFrame:
    cluster = (
        g.assign(GAME_ID=g["GAME_ID"].astype(str))
        .groupby("GAME_ID", as_index=False)
        .agg(
            n=("off_wp_next", "size"),
            sum_wp=("off_wp_next", "sum"),
            sum_win=("offense_win", "sum"),
        )
    )
    n = cluster["n"].to_numpy(dtype=float)
    sum_wp = cluster["sum_wp"].to_numpy(dtype=float)
    sum_win = cluster["sum_win"].to_numpy(dtype=float)
    idx = rng.integers(0, len(cluster), size=(n_boot, len(cluster)))
    denom = n[idx].sum(axis=1)
    model = sum_wp[idx].sum(axis=1) / denom
    empirical = sum_win[idx].sum(axis=1) / denom
    return pd.DataFrame(
        {
            "boot_id": np.arange(n_boot),
            "model_next_wp_mean": model,
            "empirical_win_rate": empirical,
            "wp_minus_empirical": model - empirical,
        }
    )


def ci(draws: pd.Series) -> tuple[float, float]:
    lo, hi = np.quantile(pd.to_numeric(draws, errors="coerce").dropna(), [0.025, 0.975])
    return float(lo), float(hi)


def summarize_cell(
    raw: pd.DataFrame,
    treatment: str,
    made: int,
    label: str,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    raw_cell = raw[(raw["treatment"].eq(treatment)) & (raw["shot_made"].eq(made))].copy()
    cell = raw_cell
    analysis_sample = "all attempts"
    if made == 1:
        cell = raw_cell[raw_cell["next_is_terminal"].eq(0)].copy()
        analysis_sample = "nonterminal next state"

    n = int(len(cell))
    successes = float(cell["offense_win"].sum())
    model = float(cell["off_wp_next"].mean()) if n else np.nan
    empirical = float(cell["offense_win"].mean()) if n else np.nan
    gap = model - empirical if n else np.nan
    wilson_lo, wilson_hi = wilson_interval(successes, n)

    if n and cell["GAME_ID"].nunique() > 1:
        draws = cluster_bootstrap(cell, rng)
        model_lo, model_hi = ci(draws["model_next_wp_mean"])
        emp_lo, emp_hi = ci(draws["empirical_win_rate"])
        gap_lo, gap_hi = ci(draws["wp_minus_empirical"])
    else:
        draws = pd.DataFrame(columns=["boot_id", "model_next_wp_mean", "empirical_win_rate", "wp_minus_empirical"])
        model_lo = model_hi = emp_lo = emp_hi = gap_lo = gap_hi = np.nan

    row = {
        "condition": label,
        "shot_type": "3P" if treatment == "three-point" else "2P",
        "made": "made" if made == 1 else "missed",
        "analysis_sample": analysis_sample,
        "raw_n": int(len(raw_cell)),
        "n": n,
        "n_games": int(cell["GAME_ID"].astype(str).nunique()) if n else 0,
        "model_next_wp_mean": model,
        "model_next_wp_boot_ci_low": model_lo,
        "model_next_wp_boot_ci_high": model_hi,
        "empirical_win_rate": empirical,
        "empirical_win_wilson_low": wilson_lo,
        "empirical_win_wilson_high": wilson_hi,
        "empirical_win_boot_ci_low": emp_lo,
        "empirical_win_boot_ci_high": emp_hi,
        "wp_minus_empirical": gap,
        "gap_boot_ci_low": gap_lo,
        "gap_boot_ci_high": gap_hi,
        "terminal_share": float(cell["next_is_terminal"].mean()) if n else np.nan,
        "off_reb_share": float(cell["off_reb_next"].astype(bool).mean()) if n else np.nan,
    }
    draws.insert(0, "condition", label)
    return row, draws


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    BOOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(RAW)
    raw = raw[(raw["time_left_game"].le(5)) & (raw["score_diff"].eq(-3))].copy()
    rng = np.random.default_rng(20260624)
    specs = [
        ("two-point", 1, "down 3, <=5s, made 2P, nonterminal next state"),
        ("three-point", 1, "down 3, <=5s, made 3P, nonterminal next state"),
        ("two-point", 0, "down 3, <=5s, missed 2P"),
        ("three-point", 0, "down 3, <=5s, missed 3P"),
    ]
    rows = []
    boot = []
    for treatment, made, label in specs:
        row, draws = summarize_cell(raw, treatment, made, label, rng)
        rows.append(row)
        boot.append(draws)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.concat(boot, ignore_index=True).to_csv(BOOT_OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote: {OUT}")
    print(f"Wrote: {BOOT_OUT}")


if __name__ == "__main__":
    main()
