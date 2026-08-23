#!/usr/bin/env python3
"""
Combine per-season outputs from build_wp_features.py into single files.

Inputs:
  data/wp/wp_states_{season}_{seasontype}.csv.gz
  data/wp/shot_decision_states_{season}_{seasontype}.csv.gz

Outputs:
  data/wp/wp_states_{start}_{end}_{seasontype}.csv.gz
  data/wp/shot_decision_states_{start}_{end}_{seasontype}.csv.gz
"""

import argparse
from pathlib import Path
import pandas as pd

WP_DIR = Path("data/wp")


def ensure_regular_output_path(path: Path) -> None:
    """
    If output path is a symlink, remove it to force creation of a real file.
    """
    if path.is_symlink():
        print(f"[info] removing symlink output path: {path}")
        path.unlink()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", type=int, default=2000)
    p.add_argument("--end-season", type=int, default=2024)
    p.add_argument("--seasontype", type=str, default="rs", choices=["rs"])
    return p.parse_args()

def main():
    args = parse_args()

    poss_parts = []
    shot_parts = []

    for season in range(args.start_season, args.end_season + 1):
        poss_path = WP_DIR / f"wp_states_{season}_{args.seasontype}.csv.gz"
        shot_path = WP_DIR / f"shot_decision_states_{season}_{args.seasontype}.csv.gz"

        if poss_path.exists():
            poss_parts.append(pd.read_csv(poss_path, compression="infer", low_memory=False))
        else:
            print(f"[WARN] missing {poss_path}")

        if shot_path.exists():
            shot_parts.append(pd.read_csv(shot_path, compression="infer", low_memory=False))
        else:
            print(f"[WARN] missing {shot_path}")

    if poss_parts:
        poss_all = pd.concat(poss_parts, ignore_index=True)
        out_poss = WP_DIR / f"wp_states_{args.start_season}_{args.end_season}_{args.seasontype}.csv.gz"
        ensure_regular_output_path(out_poss)
        poss_all.to_csv(out_poss, index=False, compression="gzip")
        print(f"[OK] wrote {out_poss} rows={len(poss_all):,}")
    else:
        print("[WARN] no poss parts")

    if shot_parts:
        shot_all = pd.concat(shot_parts, ignore_index=True)
        out_shot = WP_DIR / f"shot_decision_states_{args.start_season}_{args.end_season}_{args.seasontype}.csv.gz"
        ensure_regular_output_path(out_shot)
        shot_all.to_csv(out_shot, index=False, compression="gzip")
        print(f"[OK] wrote {out_shot} rows={len(shot_all):,}")
    else:
        print("[WARN] no shot parts")

if __name__ == "__main__":
    main()
