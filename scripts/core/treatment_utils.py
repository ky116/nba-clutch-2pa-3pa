from __future__ import annotations

import argparse
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

MULTI_TREATMENT_LABELS = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Above the Break 3",
    "Corner 3",
]

BINARY_TREATMENT_LABELS = ["two-point", "three-point"]

_MULTI_TO_BINARY = {
    "Restricted Area": "two-point",
    "In The Paint (Non-RA)": "two-point",
    "Mid-Range": "two-point",
    "Above the Break 3": "three-point",
    "Corner 3": "three-point",
    "Left Corner 3": "three-point",
    "Right Corner 3": "three-point",
}

_BINARY_ALIASES = {
    "two-point": "two-point",
    "two point": "two-point",
    "2pt": "two-point",
    "2pt fg": "two-point",
    "three-point": "three-point",
    "three point": "three-point",
    "3pt": "three-point",
    "3pt fg": "three-point",
}


def add_treatment_scheme_arg(parser: argparse.ArgumentParser, default: str = "binary") -> None:
    parser.add_argument(
        "--treatment-scheme",
        choices=["binary", "multi"],
        default=default,
        help="treatment の粒度（binary=two-point/three-point, multi=RA/PITP/Mid/AB3/Corner3）",
    )


def normalize_treatment_label(value: object, scheme: str) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() == "other":
        return None

    if scheme == "multi":
        if raw in MULTI_TREATMENT_LABELS:
            return raw
        if raw in ("Left Corner 3", "Right Corner 3"):
            return "Corner 3"
        # If binary labels are passed under multi mode, drop them.
        if raw.lower() in _BINARY_ALIASES:
            return None
        return raw

    # binary
    if raw in _MULTI_TO_BINARY:
        return _MULTI_TO_BINARY[raw]
    key = raw.lower()
    if key in _BINARY_ALIASES:
        return _BINARY_ALIASES[key]
    return None


def apply_treatment_scheme(
    df: pd.DataFrame,
    treat_col: str,
    scheme: str,
    out_col: Optional[str] = None,
    drop_unknown: bool = True,
) -> pd.DataFrame:
    if treat_col not in df.columns:
        raise ValueError(f"Column '{treat_col}' not found for treatment mapping.")

    out_col = out_col or treat_col
    mapped = df[treat_col].map(lambda v: normalize_treatment_label(v, scheme))
    df = df.copy()
    df[out_col] = mapped
    if drop_unknown:
        df = df[df[out_col].notna()].copy()
    df[out_col] = df[out_col].astype("category")
    return df


def default_treatment_pairs(labels: Iterable[str], scheme: str) -> List[Tuple[str, str]]:
    labels = list(labels)
    if scheme == "binary":
        if all(lab in labels for lab in BINARY_TREATMENT_LABELS):
            return [("three-point", "two-point")]
        return [(labels[0], labels[1])] if len(labels) >= 2 else []

    # multi
    default_pairs = [
        ("Above the Break 3", "In The Paint (Non-RA)"),
        ("Corner 3", "In The Paint (Non-RA)"),
        ("Above the Break 3", "Mid-Range"),
        ("Corner 3", "Mid-Range"),
    ]
    return [pair for pair in default_pairs if pair[0] in labels and pair[1] in labels]


def ensure_nuisance_columns(
    df: pd.DataFrame,
    treat_col: str,
    prefixes: Iterable[str] = ("m_hat_", "e_hat_"),
) -> None:
    if treat_col not in df.columns:
        raise ValueError(f"Column '{treat_col}' not found for nuisance validation.")
    labels = df[treat_col].astype("category").cat.categories.tolist()
    missing = []
    for prefix in prefixes:
        for label in labels:
            col = f"{prefix}{label}"
            if col not in df.columns:
                missing.append(col)
    if missing:
        raise ValueError(
            "Missing nuisance columns for treatment labels. "
            f"Missing: {missing}. Check --treatment-scheme and input files."
        )


def scheme_suffix(scheme: str) -> str:
    if scheme == "multi":
        return "_multi"
    return "_binary"


def filter_clutch_subset(df: pd.DataFrame, subset: str = "clutch") -> pd.DataFrame:
    """
    Standard clutch subset filter.
    - clutch(default): period>=4, time_left_game<=300, abs(score_diff)<=10
    - strict         : period>=4, time_left_game<=300, abs(score_diff)<=5
    - extended       : period>=4, time_left_game<=300, 5<abs(score_diff)<=10
    - all            : no filtering

    If *_clutch_flag columns exist, they are used first for compatibility.
    """
    name = (subset or "clutch").strip().lower()
    if name in {"all"}:
        return df.copy()

    # Prefer existing flags when available.
    if name in {"clutch", "clutch_union"}:
        if "clutch_flag" in df.columns:
            return df[df["clutch_flag"] == True].copy()
        if "strict_clutch_flag" in df.columns and "extended_clutch_flag" in df.columns:
            return df[(df["strict_clutch_flag"] == True) | (df["extended_clutch_flag"] == True)].copy()
    if name in {"strict", "strict_clutch"} and "strict_clutch_flag" in df.columns:
        return df[df["strict_clutch_flag"] == True].copy()
    if name in {"extended", "extended_clutch"} and "extended_clutch_flag" in df.columns:
        return df[df["extended_clutch_flag"] == True].copy()

    # Fallback: compute from core columns.
    req = {"period", "time_left_game", "score_diff"}
    if not req.issubset(df.columns):
        return df.copy()

    period = pd.to_numeric(df["period"], errors="coerce")
    time_left = pd.to_numeric(df["time_left_game"], errors="coerce")
    score_diff = pd.to_numeric(df["score_diff"], errors="coerce")
    abs_diff = np.abs(score_diff)
    in_window = (period >= 4) & (time_left <= 300)

    if name in {"strict", "strict_clutch"}:
        mask = in_window & (abs_diff <= 5)
    elif name in {"extended", "extended_clutch"}:
        mask = in_window & (abs_diff > 5) & (abs_diff <= 10)
    else:
        mask = in_window & (abs_diff <= 10)
    return df[mask].copy()
