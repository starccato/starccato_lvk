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


def _boot_auc_err(fg: np.ndarray, bg: np.ndarray, fg_idx: np.ndarray | None = None,
                  bg_idx: np.ndarray | None = None, n_boot: int = 1000, seed: int = 0) -> float:
    """Bootstrap standard error of the ROC AUC.

    Per-score resampling by default. When per-event indices are given, whole
    events are resampled instead (block bootstrap): the four classes at one
    trigger share a noise segment, so their scores are not independent draws.
    """
    rng = np.random.default_rng(seed)
    if fg_idx is None or bg_idx is None:
        aucs = [_roc_auc(rng.choice(fg, fg.size, replace=True), rng.choice(bg, bg.size, replace=True))
                for _ in range(n_boot)]
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
        rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(args.outdir / "results" / "e*_*.json"))]
    if not rows:
        print(f"No result JSONs under {args.outdir/'results'}.")
        return
    odds = {c: np.array([r["log_odds"] for r in rows if r["cls"] == c], dtype=float) for c in CLASSES}
    snr = {c: np.array([r["snr"] for r in rows if r["cls"] == c], dtype=float) for c in CLASSES}
    idx = {c: np.array([r["index"] for r in rows if r["cls"] == c], dtype=int) for c in CLASSES}
    bg = np.concatenate([odds["noise"], odds["inj_glitch"], odds["real_glitch"]])
    bg_snr = np.concatenate([snr["noise"], snr["inj_glitch"], snr["real_glitch"]])
    bg_idx = np.concatenate([idx["noise"], idx["inj_glitch"], idx["real_glitch"]])
    sig_idx = idx["inj_ccsn"]

    summary = {
        "n_events": len(rows),
        "n": {c: int(odds[c].size) for c in CLASSES},
        "auc_odds_signal_vs_background": _roc_auc(odds["inj_ccsn"], bg),
        "auc_odds_signal_vs_background_err": _boot_auc_err(odds["inj_ccsn"], bg, sig_idx, bg_idx),
        "auc_snr_signal_vs_background": _roc_auc(snr["inj_ccsn"], bg_snr),
        "auc_snr_signal_vs_background_err": _boot_auc_err(snr["inj_ccsn"], bg_snr, sig_idx, bg_idx),
        "auc_odds_signal_vs_real_glitch": _roc_auc(odds["inj_ccsn"], odds["real_glitch"]),
        "auc_odds_signal_vs_real_glitch_err": _boot_auc_err(odds["inj_ccsn"], odds["real_glitch"], sig_idx, idx["real_glitch"]),
        "auc_odds_signal_vs_inj_glitch": _roc_auc(odds["inj_ccsn"], odds["inj_glitch"]),
        "auc_odds_signal_vs_inj_glitch_err": _boot_auc_err(odds["inj_ccsn"], odds["inj_glitch"], sig_idx, idx["inj_glitch"]),
        "median_logBCR": {c: float(np.median(odds[c])) if odds[c].size else None for c in CLASSES},
        "real_glitch_misclassified": int(np.sum(odds["real_glitch"] > 0)),
        "inj_glitch_misclassified": int(np.sum(odds["inj_glitch"] > 0)),
        "signal_missed": int(np.sum(odds["inj_ccsn"] < 0)),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
