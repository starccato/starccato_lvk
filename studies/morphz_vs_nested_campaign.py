"""How often morphZ agreed with nested sampling in production, and how often
it was overruled.

Reads every ``raw_consolidated/*/results.json`` in a downloaded campaign
bundle and tabulates each signal/glitch evidence term's ``evidence_status``:
whether NUTS converged (else nested is used directly, bypassing morphZ), and
when morphZ was cross-checked (railed amplitude or |lnZ|>50), whether it was
kept (``verified``, agreed with nested within the 3-nat production tolerance)
or overruled (``fallback``, disagreed and nested's value was used instead).

The per-event message recording the *magnitude* of a morphZ/nested
disagreement is only ever printed to the SLURM job's stdout -- it is not
persisted in ``evidence_status`` -- so this script reports rates, not a
disagreement-size distribution. To get the latter, grep
``slurm/logs/*.out`` for "verified vs nested" / "disagrees with nested"
before those logs are cleaned up.

Usage:
    uv run python studies/morphz_vs_nested_campaign.py \\
        --bundle ../nuts_morphlnz_v041_analysis_bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_evidence_status(bundle: Path) -> pd.DataFrame:
    root = bundle / "raw_consolidated"
    rows = []
    for cfg_dir in sorted(root.iterdir()):
        results_json = cfg_dir / "results.json"
        if not results_json.is_file():
            continue
        for r in json.loads(results_json.read_text()):
            for term, es in r.get("evidence_status", {}).items():
                rows.append(
                    {
                        "config": cfg_dir.name,
                        "index": r["index"],
                        "cls": r["cls"],
                        "term": term,
                        "method": es.get("method"),
                        "status": es.get("status"),
                        "conv_failure": es.get("nuts_convergence_failure"),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args()

    df = load_evidence_status(args.bundle)
    total = len(df)
    routed = df[df.conv_failure.notna()]
    remaining = df[df.conv_failure.isna()]
    triggered = remaining[remaining.status.isin(["verified", "fallback"])]
    verified = triggered[triggered.status == "verified"]
    fallback = triggered[triggered.status == "fallback"]

    print(f"total evidence terms: {total}")
    print(
        f"routed directly to nested (NUTS non-convergence): {len(routed)} "
        f"({len(routed) / total:.1%})"
    )
    print(f"remaining (morphZ attempted): {len(remaining)}")
    print(
        f"  cross-check triggered (railed amp. or |lnZ|>50): {len(triggered)} "
        f"({len(triggered) / len(remaining):.1%} of remaining)"
    )
    print(
        f"    verified (morphZ kept): {len(verified)} "
        f"({len(verified) / len(triggered):.1%} of triggered)"
    )
    print(
        f"    fallback (nested used instead): {len(fallback)} "
        f"({len(fallback) / len(triggered):.1%} of triggered)"
    )


if __name__ == "__main__":
    main()
