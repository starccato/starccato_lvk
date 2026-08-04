"""Write a ``campaign_event_metrics.csv`` bundle from a combined-results HDF5.

``paper_figures_v044.py`` reads the flat per-event CSV that the OzSTAR bundle
script used to produce. Later campaigns ship the single consolidated HDF5
instead, so this flattens ``/lno`` and ``/lno_baseline`` back into that CSV and
the figure/table pipeline runs unchanged. It also writes
``campaign_provenance.json`` with the archive identity and evidence-estimator
routing counts used in the manuscript.

Usage:
    uv run python studies/bundle_from_combined.py \
        --h5 ../combined_results_nuts_morphlnz_v044.h5 \
        --outdir ../nuts_morphlnz_v044_analysis_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# cohort -> (detector count, blip host detector)
COHORTS = {
    "rn_L1": (1, "L1"),
    "rn_H1_blipH1": (1, "H1"),
    "rn_H1_L1": (2, "L1"),
    "rn_H1_L1_blipH1": (2, "H1"),
}


def _flat(grp: h5py.Group, name: str, n: int) -> np.ndarray:
    if name not in grp:
        return np.full(n, np.nan)
    return grp[name][:]


def _decode_json(value) -> dict:
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    rows: list[pd.DataFrame] = []
    estimator_counts: Counter = Counter()
    status_counts: Counter = Counter()
    n_analysis_rows = 0
    n_evidence_failures = 0
    with h5py.File(args.h5, "r") as f:
        source_attrs = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in f.attrs.items()
        }
        for cohort, (ndet, host) in COHORTS.items():
            if f"lno/{cohort}" not in f:
                continue
            for cls in f[f"lno/{cohort}"]:
                g = f[f"lno/{cohort}/{cls}"]
                n = g["index"].shape[0]
                # SNR lives only in the per-event JSON blob
                raw_rows = [_decode_json(s) for s in g["raw_json"][:]]
                snr = np.array([row.get("snr", np.nan) for row in raw_rows], dtype=float)
                n_analysis_rows += n
                for row in raw_rows:
                    n_evidence_failures += int(row.get("evidence_failures", 0))
                    for term in row.get("evidence_status", {}).values():
                        estimator_counts[str(term.get("method", "unknown"))] += 1
                        status_counts[str(term.get("status", "unknown"))] += 1
                df = pd.DataFrame(
                    {
                        "configuration": cohort,
                        "detector_count": ndet,
                        "glitch_host": host,
                        "index": g["index"][:].astype(int),
                        "class": cls,
                        "detectors": "H1,L1" if ndet == 2 else cohort.split("_")[1],
                        "injected_or_catalog_snr": snr,
                        "log_odds": g["log_odds"][:],
                        "logZ_signal": g["logZ_signal"][:],
                        "logZ_glitch": g["logZ_glitch"][:],
                        "logZ_glitch_H1": _flat(g, "logZ_glitch_H1", n),
                        "logZ_glitch_L1": _flat(g, "logZ_glitch_L1", n),
                        "evidence_failures": _flat(g, "evidence_failures", n),
                        "evidence_fallbacks": _flat(g, "evidence_fallbacks", n),
                        "data_quality_H1_mean": _flat(g, "data_quality_H1_mean", n),
                        "data_quality_L1_mean": _flat(g, "data_quality_L1_mean", n),
                    }
                )
                b = f.get(f"lno_baseline/{cohort}/{cls}_baseline")
                if b is not None:
                    base = pd.DataFrame(
                        {
                            "index": b["index"][:].astype(int),
                            "new_snr": b["new_snr"][:],
                            "mf_snr": b["mf_snr"][:],
                            "new_snr_H1": _flat(b, "new_snr_H1", b["index"].shape[0]),
                            "new_snr_L1": _flat(b, "new_snr_L1", b["index"].shape[0]),
                        }
                    )
                    df = df.merge(base, on="index", how="left")
                rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    dest = args.outdir / "campaign_event_metrics.csv"
    out.to_csv(dest, index=False)
    direct_nested = int(estimator_counts["nested"] - status_counts["fallback"])
    attempted_morph = int(estimator_counts["morph"] + status_counts["fallback"])
    crosschecked = int(status_counts["verified"] + status_counts["fallback"])
    provenance = {
        "source_h5": str(args.h5.resolve()),
        "source_h5_sha256": _sha256(args.h5),
        "source_attrs": source_attrs,
        "n_analysis_rows": n_analysis_rows,
        "n_evidence_terms": int(sum(estimator_counts.values())),
        "n_evidence_failures": n_evidence_failures,
        "estimator_counts": dict(sorted(estimator_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "direct_nested_terms": direct_nested,
        "morph_attempted_terms": attempted_morph,
        "crosschecked_terms": crosschecked,
    }
    (args.outdir / "campaign_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {dest} ({len(out)} rows)")
    print(f"wrote {args.outdir / 'campaign_provenance.json'}")
    print(out.groupby(["configuration", "class"])["index"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
