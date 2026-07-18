"""Aggregate per-event JSON rows from real_noise_event.py into ROC/AUC summaries.

uv run python studies/real_noise_aggregate.py --outdir out_rn
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from snr_vs_odds_roc import _roc_auc

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")
PRODUCTION_CLASSES = ("noise", "inj_ccsn", "real_glitch")


def _boot_auc_err(
    fg: np.ndarray,
    bg: np.ndarray,
    fg_idx: np.ndarray | None = None,
    bg_idx: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> float:
    """Bootstrap standard error of the ROC AUC.

    Per-score resampling by default. When per-event indices are given, whole
    events are resampled instead (block bootstrap): the four classes at one
    trigger share a noise segment, so their scores are not independent draws.
    """
    rng = np.random.default_rng(seed)
    if fg_idx is None or bg_idx is None:
        aucs = [
            _roc_auc(
                rng.choice(fg, fg.size, replace=True),
                rng.choice(bg, bg.size, replace=True),
            )
            for _ in range(n_boot)
        ]
        return float(np.std(aucs))
    fg_by = {e: fg[fg_idx == e] for e in np.unique(fg_idx)}
    bg_by = {e: bg[bg_idx == e] for e in np.unique(bg_idx)}
    events = np.unique(np.concatenate([fg_idx, bg_idx]))
    empty = np.empty(0)
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(events, events.size, replace=True)
        f = np.concatenate([fg_by.get(e, empty) for e in pick])
        b = np.concatenate([bg_by.get(e, empty) for e in pick])
        if f.size and b.size:
            aucs.append(_roc_auc(f, b))
    return float(np.std(aucs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("out_rn"))
    args = p.parse_args()

    combined = args.outdir / "results.json"  # written by collect_results.py
    if combined.exists():
        rows = json.loads(combined.read_text())
    else:
        rows = [
            json.loads(Path(f).read_text())
            for f in glob.glob(str(args.outdir / "results" / "e*_*.json"))
        ]
    if not rows:
        print(f"No result JSONs under {args.outdir/'results'}.")
        return
    odds = {
        c: np.array(
            [r["log_odds"] for r in rows if r["cls"] == c], dtype=float
        )
        for c in CLASSES
    }
    snr = {
        c: np.array([r["snr"] for r in rows if r["cls"] == c], dtype=float)
        for c in CLASSES
    }
    idx = {
        c: np.array([r["index"] for r in rows if r["cls"] == c], dtype=int)
        for c in CLASSES
    }
    missing_required = [c for c in PRODUCTION_CLASSES if not odds[c].size]
    if missing_required:
        raise RuntimeError(
            f"Missing required production classes: {missing_required}"
        )
    present_classes = [c for c in CLASSES if odds[c].size]
    background_classes = [
        c for c in ("noise", "inj_glitch", "real_glitch") if odds[c].size
    ]
    index_sets = {c: set(idx[c].tolist()) for c in present_classes}
    all_indices = set().union(*index_sets.values())
    complete_indices = set.intersection(*index_sets.values())
    incomplete_indices = sorted(all_indices - complete_indices)
    bg = np.concatenate([odds[c] for c in background_classes])
    bg_snr = np.concatenate([snr[c] for c in background_classes])
    bg_idx = np.concatenate([idx[c] for c in background_classes])
    sig_idx = idx["inj_ccsn"]

    summary = {
        "n_events": len(rows),
        "n": {c: int(odds[c].size) for c in CLASSES},
        "classes": present_classes,
        "background_classes": background_classes,
        "n_catalogue_indices": len(all_indices),
        "n_complete_class_groups": len(complete_indices),
        "n_complete_four_class_groups": (
            len(complete_indices) if len(present_classes) == 4 else None
        ),
        "n_incomplete_groups": len(incomplete_indices),
        "incomplete_indices": incomplete_indices,
        "auc_odds_signal_vs_background": _roc_auc(odds["inj_ccsn"], bg),
        "auc_odds_signal_vs_background_err": _boot_auc_err(
            odds["inj_ccsn"], bg, sig_idx, bg_idx
        ),
        "auc_snr_signal_vs_background": _roc_auc(snr["inj_ccsn"], bg_snr),
        "auc_snr_signal_vs_background_err": _boot_auc_err(
            snr["inj_ccsn"], bg_snr, sig_idx, bg_idx
        ),
        "auc_odds_signal_vs_real_glitch": _roc_auc(
            odds["inj_ccsn"], odds["real_glitch"]
        ),
        "auc_odds_signal_vs_real_glitch_err": _boot_auc_err(
            odds["inj_ccsn"], odds["real_glitch"], sig_idx, idx["real_glitch"]
        ),
        "auc_odds_signal_vs_inj_glitch": (
            _roc_auc(odds["inj_ccsn"], odds["inj_glitch"])
            if odds["inj_glitch"].size
            else None
        ),
        "auc_odds_signal_vs_inj_glitch_err": (
            _boot_auc_err(
                odds["inj_ccsn"],
                odds["inj_glitch"],
                sig_idx,
                idx["inj_glitch"],
            )
            if odds["inj_glitch"].size
            else None
        ),
        "median_logBCR": {
            c: float(np.median(odds[c])) if odds[c].size else None
            for c in CLASSES
        },
        "real_glitch_misclassified": int(np.sum(odds["real_glitch"] > 0)),
        "inj_glitch_misclassified": (
            int(np.sum(odds["inj_glitch"] > 0))
            if odds["inj_glitch"].size
            else None
        ),
        "signal_missed": int(np.sum(odds["inj_ccsn"] < 0)),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if incomplete_indices:
        print(
            f"WARNING: {len(incomplete_indices)} catalogue indices lack at least one class; "
            "backfill them before interpreting marginal population metrics."
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
