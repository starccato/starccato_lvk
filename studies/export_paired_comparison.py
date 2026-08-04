"""Export a compact lnO-vs-BayesWave paired table from the combined HDF5.

The full archive carries the posterior waveform arrays and is multi-GB; the
paper's evidence comparison only needs a few numbers per event. This writes a
small CSV (and a matching slim HDF5) with one row per (class, index) event that
has BOTH a morphZ log odds and a BayesWave signal-vs-glitch Bayes factor, so it
can be copied to a laptop and iterated on quickly.

Events are paired by index: per-event draws are seeded by index, so e<N> in the
lnO campaign is the same physical event as e<N> in the BayesWave campaign
(verified by comparing the injected per-detector SNRs).

Provenance flags are carried through -- ``recovered`` rows finished sampling but
had their evidence reconstructed from bayeswave.log or from an evidence.dat
whose uncertainty underflowed, and have no posterior draws. Filter on
``recovered``/``degenerate_uncertainty`` before treating a row as a full run.

Usage:
    python studies/export_paired_comparison.py \
        --h5 .../combined_results_nuts_morphlnz_v044.h5 \
        --lno-cohort rn_H1_L1 \
        --pairs bwcomp_nml_v044_ccsn:inj_ccsn bwcomp_nml_v044_glitch:real_glitch \
        --out .../paired_lno_vs_bayeswave.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np

COLUMNS = [
    "cls", "index",
    "lno_log_odds", "lno_log_odds_err",
    "lno_logZ_signal", "lno_logZ_glitch",
    "bw_lnbf_signal_glitch", "bw_lnbf_signal_glitch_err",
    "bw_logZ_signal", "bw_logZ_glitch", "bw_logZ_noise",
    "bw_target_snr", "bw_reconstructed_snr_median",
    "dq_H1_mean", "dq_L1_mean",
    "bw_recovered", "bw_degenerate_uncertainty",
    "bw_posteriors_reported", "bw_posteriors_available",
    "bw_posteriors_comparison_ready",
    "bw_evidence_source",
]


def _col(grp: h5py.Group, name: str):
    if name not in grp:
        return None
    data = grp[name][:]
    if data.dtype.kind in ("S", "O"):
        return [v.decode() if isinstance(v, bytes) else str(v) for v in data]
    return data


def _lno_err(signal_err: float, g_h1: float, g_l1: float) -> float:
    """Independent-error propagation for lnZ_signal - sum_i lnZ_glitch_i."""
    vals = [signal_err, g_h1, g_l1]
    if any(v is None or not math.isfinite(v) for v in vals):
        return float("nan")
    return math.sqrt(signal_err**2 + g_h1**2 + g_l1**2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--lno-cohort", default="rn_H1_L1")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="bw_campaign:class entries")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows: list[dict] = []
    unmatched = 0
    with h5py.File(args.h5, "r") as f:
        for spec in args.pairs:
            campaign, _, cls = spec.partition(":")
            bw_path = f"bayeswave/{campaign}/{cls}"
            lno_path = f"lno/{args.lno_cohort}/{cls}"
            if bw_path not in f or lno_path not in f:
                print(f"  skip {spec}: missing {bw_path} or {lno_path}")
                continue
            bw, lno = f[bw_path], f[lno_path]

            # index -> row position, so pairing is an explicit lookup
            lno_idx = {int(v): i for i, v in enumerate(lno["index"][:])}
            lno_odds = _col(lno, "log_odds")
            lno_zs, lno_zg = _col(lno, "logZ_signal"), _col(lno, "logZ_glitch")
            lno_se = _col(lno, "logZ_signal_err")
            lno_geh1, lno_gel1 = _col(lno, "logZ_glitch_err_H1"), _col(lno, "logZ_glitch_err_L1")
            dq_h1, dq_l1 = _col(lno, "data_quality_H1_mean"), _col(lno, "data_quality_L1_mean")

            src = _col(bw, "evidence_source") or ["bayeswave_post"] * bw["index"].shape[0]
            reported_posteriors = _col(bw, "posteriors_available")
            posterior_group = bw.get("posteriors")
            for j, raw_index in enumerate(bw["index"][:]):
                index = int(raw_index)
                i = lno_idx.get(index)
                if i is None:
                    unmatched += 1
                    continue
                embedded = (
                    posterior_group is not None
                    and f"e{index}" in posterior_group
                )
                event_posteriors = (
                    posterior_group[f"e{index}"] if embedded else None
                )
                waveform_keys = (
                    "whitened_waveform_draws_H1",
                    "whitened_waveform_draws_L1",
                )
                comparison_keys = waveform_keys + (
                    "whitened_data_H1",
                    "whitened_data_L1",
                    "signal_median_psd_H1",
                    "signal_median_psd_L1",
                )

                def has_nonempty(keys: tuple[str, ...]) -> bool:
                    return bool(
                        event_posteriors is not None
                        and all(
                            key in event_posteriors
                            and event_posteriors[key].size > 0
                            for key in keys
                        )
                    )

                rows.append({
                    "cls": cls,
                    "index": index,
                    "lno_log_odds": float(lno_odds[i]),
                    "lno_log_odds_err": _lno_err(
                        float(lno_se[i]), float(lno_geh1[i]), float(lno_gel1[i])),
                    "lno_logZ_signal": float(lno_zs[i]),
                    "lno_logZ_glitch": float(lno_zg[i]),
                    "bw_lnbf_signal_glitch": float(bw["log_bayeswave_signal_glitch"][j]),
                    "bw_lnbf_signal_glitch_err": float(
                        bw["log_bayeswave_signal_glitch_unc"][j]),
                    "bw_logZ_signal": float(bw["logZ_signal"][j]),
                    "bw_logZ_glitch": float(bw["logZ_glitch"][j]),
                    "bw_logZ_noise": float(bw["logZ_noise"][j]),
                    "bw_target_snr": float(bw["target_snr"][j]),
                    "bw_reconstructed_snr_median": float(
                        bw["signal_reconstructed_snr_median"][j]),
                    "dq_H1_mean": float(dq_h1[i]),
                    "dq_L1_mean": float(dq_l1[i]),
                    "bw_recovered": int(bw["recovered"][j]),
                    "bw_degenerate_uncertainty": int(bw["degenerate_uncertainty"][j]),
                    "bw_posteriors_reported": int(reported_posteriors[j])
                    if reported_posteriors is not None
                    else int(not bw["recovered"][j]),
                    "bw_posteriors_available": int(has_nonempty(waveform_keys)),
                    "bw_posteriors_comparison_ready": int(
                        has_nonempty(comparison_keys)
                    ),
                    "bw_evidence_source": src[j],
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # slim HDF5 alongside, for anyone who prefers arrays to CSV
    slim = args.out.with_suffix(".h5")
    with h5py.File(slim, "w") as out:
        out.attrs["source_archive"] = str(args.h5)
        out.attrs["n_paired"] = len(rows)
        for cls in sorted({r["cls"] for r in rows}):
            sub = [r for r in rows if r["cls"] == cls]
            g = out.create_group(cls)
            for key in COLUMNS:
                if key == "cls":
                    continue
                vals = [r[key] for r in sub]
                if isinstance(vals[0], str):
                    g.create_dataset(key, data=np.array(vals, dtype=object),
                                     dtype=h5py.string_dtype(encoding="utf-8"))
                else:
                    g.create_dataset(key, data=np.asarray(vals))

    by_cls: dict[str, int] = {}
    for r in rows:
        by_cls[r["cls"]] = by_cls.get(r["cls"], 0) + 1
    full = sum(1 for r in rows if not r["bw_recovered"])
    print(f"wrote {args.out} ({len(rows)} paired events: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_cls.items()))
          + f"); {full} full-quality, {len(rows)-full} recovered")
    if unmatched:
        print(f"  {unmatched} BayesWave events had no lnO counterpart in {args.lno_cohort}")
    print(f"wrote {slim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
