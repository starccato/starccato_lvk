"""Audit completeness and paired one- versus two-detector results.

The real-noise array writes one row per ``(catalogue index, class)``.  A direct
comparison of two detector configurations is only paired when the same index and
class succeeded in both runs.  This script makes that cohort explicit and saves
the quantities used by the manuscript's coherence claim.

Example
-------
uv run python studies/paired_network_audit.py \
    --single slurm/out/rn_L1 \
    --network slurm/out/rn_H1_L1 \
    --output out_rn_figs/paired_network_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")
BACKGROUND = ("noise", "inj_glitch", "real_glitch")


def _load_rows(outdir: Path) -> list[dict]:
    path = outdir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing collected results: {path}")
    rows = json.loads(path.read_text())
    keys = [(int(row["index"]), row["cls"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate (index, class) rows in {path}")
    unknown = sorted({row["cls"] for row in rows} - set(CLASSES))
    if unknown:
        raise ValueError(f"Unexpected classes in {path}: {unknown}")
    return rows


def _by_key(rows: Iterable[dict]) -> dict[tuple[int, str], dict]:
    return {(int(row["index"]), row["cls"]): row for row in rows}


def _indices(by_key: dict[tuple[int, str], dict], cls: str) -> set[int]:
    return {index for index, row_class in by_key if row_class == cls}


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann--Whitney AUC with half credit for ties."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not positive.size or not negative.size:
        return float("nan")
    wins = sum(
        np.sum(value > negative) + 0.5 * np.sum(value == negative)
        for value in positive
    )
    return float(wins / (positive.size * negative.size))


def _exact_mcnemar_p(n_10: int, n_01: int) -> float:
    """Two-sided exact McNemar p-value for the discordant pairs."""
    n = n_10 + n_01
    if n == 0:
        return 1.0
    k = min(n_10, n_01)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * lower_tail))


def _run_profile(rows: list[dict]) -> dict:
    by_key = _by_key(rows)
    counts = {cls: len(_indices(by_key, cls)) for cls in CLASSES}
    complete = set.intersection(*(_indices(by_key, cls) for cls in CLASSES))
    return {
        "n_rows": len(rows),
        "class_counts": counts,
        "n_complete_four_class_groups": len(complete),
        "n_incomplete_groups": len({index for index, _ in by_key} - complete),
    }


def audit(single_rows: list[dict], network_rows: list[dict]) -> dict:
    single = _by_key(single_rows)
    network = _by_key(network_rows)
    paired_indices = {
        cls: sorted(_indices(single, cls) & _indices(network, cls))
        for cls in CLASSES
    }

    paired_scores: dict[str, dict[str, np.ndarray]] = {"single": {}, "network": {}}
    for label, rows in (("single", single), ("network", network)):
        for cls, indices in paired_indices.items():
            paired_scores[label][cls] = np.asarray(
                [rows[(index, cls)]["log_odds"] for index in indices], dtype=float
            )

    aucs = {}
    for label in ("single", "network"):
        signal = paired_scores[label]["inj_ccsn"]
        background = np.concatenate([paired_scores[label][cls] for cls in BACKGROUND])
        aucs[label] = {
            "signal_vs_combined_background": _auc(signal, background),
            **{
                f"signal_vs_{cls}": _auc(signal, paired_scores[label][cls])
                for cls in BACKGROUND
            },
        }

    transitions = {}
    for cls in ("real_glitch", "inj_ccsn"):
        first = paired_scores["single"][cls]
        second = paired_scores["network"][cls]
        if cls == "real_glitch":
            first_error = first > 0
            second_error = second > 0
        else:
            first_error = first < 0
            second_error = second < 0
        corrected = int(np.sum(first_error & ~second_error))
        worsened = int(np.sum(~first_error & second_error))
        delta = second - first
        transitions[cls] = {
            "n_pairs": int(first.size),
            "single_error_count": int(first_error.sum()),
            "single_error_rate": float(first_error.mean()),
            "network_error_count": int(second_error.sum()),
            "network_error_rate": float(second_error.mean()),
            "error_in_both": int(np.sum(first_error & second_error)),
            "corrected_by_network": corrected,
            "worsened_by_network": worsened,
            "exact_mcnemar_p": _exact_mcnemar_p(corrected, worsened),
            "median_log_odds_delta": float(np.median(delta)),
            "log_odds_delta_q16_q84": np.quantile(delta, [0.16, 0.84]).tolist(),
        }

    return {
        "single": _run_profile(single_rows),
        "network": _run_profile(network_rows),
        "paired_class_counts": {cls: len(indices) for cls, indices in paired_indices.items()},
        "paired_auc": aucs,
        "threshold_transitions": transitions,
        "notes": [
            "AUC cohorts use the class-specific index intersection between configurations.",
            "Threshold transitions are paired at identical catalogue indices.",
            "Backfill incomplete four-class groups before final population figures.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(_load_rows(args.single), _load_rows(args.network))
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
