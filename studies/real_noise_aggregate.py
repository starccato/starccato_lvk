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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("out_rn"))
    args = p.parse_args()

    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(args.outdir / "results" / "e*_*.json"))]
    if not rows:
        print(f"No result JSONs under {args.outdir/'results'}.")
        return
    odds = {c: np.array([r["log_odds"] for r in rows if r["cls"] == c], dtype=float) for c in CLASSES}
    snr = {c: np.array([r["snr"] for r in rows if r["cls"] == c], dtype=float) for c in CLASSES}
    bg = np.concatenate([odds["noise"], odds["inj_glitch"], odds["real_glitch"]])
    bg_snr = np.concatenate([snr["noise"], snr["inj_glitch"], snr["real_glitch"]])

    summary = {
        "n_events": len(rows),
        "n": {c: int(odds[c].size) for c in CLASSES},
        "auc_odds_signal_vs_background": _roc_auc(odds["inj_ccsn"], bg),
        "auc_snr_signal_vs_background": _roc_auc(snr["inj_ccsn"], bg_snr),
        "auc_odds_signal_vs_real_glitch": _roc_auc(odds["inj_ccsn"], odds["real_glitch"]),
        "auc_odds_signal_vs_inj_glitch": _roc_auc(odds["inj_ccsn"], odds["inj_glitch"]),
        "median_logBCR": {c: float(np.median(odds[c])) if odds[c].size else None for c in CLASSES},
        "real_glitch_misclassified": int(np.sum(odds["real_glitch"] > 0)),
        "inj_glitch_misclassified": int(np.sum(odds["inj_glitch"] > 0)),
        "signal_missed": int(np.sum(odds["inj_ccsn"] < 0)),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
