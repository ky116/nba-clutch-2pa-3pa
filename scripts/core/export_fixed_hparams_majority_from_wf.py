#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PARAM_KEYS = ("nuisance_best_prop", "nuisance_best_outcome", "tau_best_params")


def _canon(d: Dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _quote(v: str) -> str:
    return shlex.quote(v)


def _emit_exports(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    for k, v in payload.items():
        if isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False)
        else:
            s = str(v)
        lines.append(f"export {k}={_quote(s)}")
    return "\n".join(lines)


def _infer_model_name(wf_root: Path, fallback: str = "lgbm") -> str:
    name = wf_root.name.lower()
    for m in ("xgb", "lgbm", "catboost"):
        if m in name:
            return m
    return fallback


def _iter_meta_paths(wf_root: Path) -> Iterable[Path]:
    for p in sorted(wf_root.glob("*/meta.json")):
        if p.is_file():
            yield p


def _select_majority(payloads: List[Tuple[str, Dict[str, Any]]], tie_break: str) -> Tuple[Dict[str, Any], Dict[str, int], List[str]]:
    if not payloads:
        raise ValueError("No payloads for majority selection.")
    keys = [k for k, _ in payloads]
    counts = Counter(keys)
    top = max(counts.values())
    winners = sorted([k for k, c in counts.items() if c == top])
    if len(winners) == 1:
        key = winners[0]
        tied_keys: List[str] = []
    else:
        if tie_break == "latest":
            key = payloads[-1][0]
            if key not in winners:
                # Pick latest among ties.
                for k, _ in reversed(payloads):
                    if k in winners:
                        key = k
                        break
        elif tie_break == "smallest":
            key = winners[0]
        else:
            raise ValueError(f"Unknown tie-break rule: {tie_break}")
        tied_keys = winners
    inv = {k: v for k, v in payloads}
    return inv[key], dict(counts), tied_keys


def _load_best_params(meta_paths: List[Path]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {k: [] for k in PARAM_KEYS}
    for p in meta_paths:
        obj = json.loads(p.read_text(encoding="utf-8"))
        for k in PARAM_KEYS:
            v = obj.get(k)
            if isinstance(v, dict):
                out[k].append((_canon(v), v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate WF outer-fold best params and export majority-vote fixed hyperparameters."
    )
    ap.add_argument("--wf-root", required=True, help="WF root directory (e.g., results/nested_wf_lgbm)")
    ap.add_argument("--model", default=None, choices=["xgb", "lgbm", "catboost"], help="Model name for output env vars.")
    ap.add_argument("--emit", default="sh", choices=["sh", "json"], help="Output format.")
    ap.add_argument(
        "--tie-break",
        default="latest",
        choices=["latest", "smallest"],
        help="Tie break when multiple params share max vote count.",
    )
    ap.add_argument(
        "--write-json-dir",
        default=None,
        help="Optional directory to save fixed_prop_params.json / fixed_outcome_params.json / fixed_tau_params.json.",
    )
    args = ap.parse_args()

    wf_root = Path(args.wf_root)
    if not wf_root.exists():
        raise FileNotFoundError(str(wf_root))

    meta_paths = list(_iter_meta_paths(wf_root))
    if not meta_paths:
        raise FileNotFoundError(f"No meta.json found under: {wf_root}")

    best = _load_best_params(meta_paths)
    if any(len(best[k]) == 0 for k in PARAM_KEYS):
        missing = [k for k in PARAM_KEYS if len(best[k]) == 0]
        raise ValueError(f"Some parameter groups are missing in meta.json files: {missing}")

    prop_params, prop_counts, prop_ties = _select_majority(best["nuisance_best_prop"], tie_break=args.tie_break)
    out_params, out_counts, out_ties = _select_majority(best["nuisance_best_outcome"], tie_break=args.tie_break)
    tau_params, tau_counts, tau_ties = _select_majority(best["tau_best_params"], tie_break=args.tie_break)

    model = args.model or _infer_model_name(wf_root)
    summary = {
        "wf_root": str(wf_root),
        "n_outer": len(meta_paths),
        "tie_break": args.tie_break,
        "vote_counts": {
            "nuisance_best_prop": prop_counts,
            "nuisance_best_outcome": out_counts,
            "tau_best_params": tau_counts,
        },
        "tied_keys": {
            "nuisance_best_prop": prop_ties,
            "nuisance_best_outcome": out_ties,
            "tau_best_params": tau_ties,
        },
    }

    payload = {
        "USE_FIXED_HPARAMS": 1,
        "PROP_MODEL": model,
        "OUTCOME_MODEL": model,
        "TAU_MODEL": model,
        "FIXED_PROP_PARAMS_JSON": json.dumps(prop_params, ensure_ascii=False),
        "FIXED_OUTCOME_PARAMS_JSON": json.dumps(out_params, ensure_ascii=False),
        "FIXED_TAU_PARAMS_JSON": json.dumps(tau_params, ensure_ascii=False),
        "WF_MAJORITY_ROOT": str(wf_root),
        "WF_MAJORITY_N_OUTER": len(meta_paths),
        "WF_MAJORITY_TIE_BREAK": args.tie_break,
    }

    if args.write_json_dir:
        outdir = Path(args.write_json_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "fixed_prop_params.json").write_text(json.dumps(prop_params, ensure_ascii=False, indent=2), encoding="utf-8")
        (outdir / "fixed_outcome_params.json").write_text(json.dumps(out_params, ensure_ascii=False, indent=2), encoding="utf-8")
        (outdir / "fixed_tau_params.json").write_text(json.dumps(tau_params, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["write_json_dir"] = str(outdir)

    if args.emit == "json":
        print(json.dumps({"env": payload, "summary": summary}, ensure_ascii=False, indent=2))
    else:
        print(_emit_exports(payload))
        print("")
        print("# summary")
        print("# " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
