"""Analysis study in REAL detector noise (1- or 2-detector), four event classes.

This is the paper's analysis section. Background is REAL GWOSC strain; we form
four kinds of event and rank them with the VAE odds:

    noise        : real noise only                                  (background)
    inj_ccsn     : real noise + injected held-out CCSN              (foreground)
                   (coherent across detectors for the multi-det run)
    inj_glitch   : real noise + injected held-out blip              (background)
                   (single detector, incoherent)
    real_glitch  : a real catalogue blip (L1) + real noise elsewhere(background)

Injections use HELD-OUT waveforms the VAEs never trained on (off-manifold). The
1-detector run uses ``--detectors L1``; the 2-detector run ``--detectors H1 L1``
and demonstrates the coherence rejection of single-detector features.

    uv run python studies/real_noise_study.py --detectors L1 --n-events 8
    uv run python studies/real_noise_study.py --detectors H1 L1 --n-events 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from starccato_lvk.analysis import run_bcr_posteriors
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
from starccato_lvk.acquisition.io.glitch_catalog import get_blip_trigger_time, load_blip_glitch_catalog

from real_data_smoke_test import _build_bundle
from real_data_roc import _inject_into_bundle
from snr_vs_odds_roc import SAMPLE_RATE, EventRow, _roc_auc
from snr_vs_odds_roc_coherent import _inject_coherent, _inject_single_glitch

try:
    from starccato_jax.data.training_data import TrainValData
except Exception:  # pragma: no cover
    TrainValData = None

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")


def _bundles_for(detectors, gps, outdir, *, require_cat3=True):
    """Build/reuse a real bundle per detector at ``gps`` (skips on download failure)."""
    out = {}
    for det in detectors:
        out[det] = _build_bundle(gps, outdir / det, detector=det)
    return out


def _run(bundles: Dict[str, Path], detectors, outdir, flow, fmax, nsm, nw, ns):
    return run_bcr_posteriors(
        detectors=detectors, outdir=str(outdir),
        bundle_paths={d: str(b) for d, b in bundles.items()},
        signal_model="ccsne", glitch_model="blip", flow=flow, fmax=fmax,
        num_warmup=nw, num_samples=ns, save_artifacts=False, lnz_method="morph",
        noise_scale_marginal=nsm, nested_num_live_points=300, nested_max_samples=6000,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detectors", nargs="+", default=["L1"])
    p.add_argument("--glitch-det", default="L1", help="Detector hosting injected/real glitches.")
    p.add_argument("--outdir", type=Path, default=Path("out_real_noise"))
    p.add_argument("--n-events", type=int, default=8)
    p.add_argument("--noise-offset", type=float, default=100.0)
    p.add_argument("--snr-grid", type=float, nargs="+", default=[10, 20, 40])
    p.add_argument("--flow", type=float, default=300.0)
    p.add_argument("--fmax", type=float, default=800.0)
    p.add_argument("--num-warmup", type=int, default=200)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--no-marginal", action="store_true")
    args = p.parse_args()

    detectors = [d.upper() for d in args.detectors]
    gdet = args.glitch_det.upper()
    args.outdir.mkdir(parents=True, exist_ok=True)
    nsm = not args.no_marginal
    dt = 1.0 / SAMPLE_RATE
    n_seg = int(round(4.0 * SAMPLE_RATE))
    cat = load_blip_glitch_catalog()
    ccsn_pool = np.asarray(TrainValData.load(source="ccsne", seed=0).val) if TrainValData else None
    blip_pool = np.asarray(TrainValData.load(source="blip", seed=0).val) if TrainValData else None
    rng = np.random.default_rng(11)
    print(f"[real-noise] detectors={detectors} glitch_det={gdet} marginal={nsm} band=[{args.flow},{args.fmax}]")

    rows: List[EventRow] = []
    eid = 0
    for i in range(args.n_events):
        blip_gps = get_blip_trigger_time(i)
        noise_gps = blip_gps - args.noise_offset
        cat_snr = float(cat.iloc[i]["snr"])
        target = float(args.snr_grid[i % len(args.snr_grid)])
        edir = args.outdir / f"e{i}"
        try:
            noise_b = _bundles_for(detectors, noise_gps, edir / "noise")
            # real glitch lives in gdet at blip_gps; other detectors get real noise at blip_gps
            real_g = {}
            for d in detectors:
                real_g[d] = _build_bundle(blip_gps, edir / "realg" / d, detector=d)
        except Exception as exc:  # noqa: BLE001
            print(f"[real-noise] e{i} bundle FAILED: {type(exc).__name__}: {exc}")
            continue

        prep = prepare_multi_detector_data(detectors, bundle_paths=noise_b, flow=args.flow, fmax=args.fmax)
        # injected CCSN (coherent) and injected blip (single-det) into the SAME real noise
        ccsn_wf = ccsn_pool[rng.integers(ccsn_pool.shape[0])] if ccsn_pool is not None else None
        blip_wf = blip_pool[rng.integers(blip_pool.shape[0])] if blip_pool is not None else None
        inj_c, net_snr = _inject_coherent(prep, target, n_seg, dt, args.flow, args.fmax, ccsn_wf)
        inj_g, g_snr = _inject_single_glitch(prep, gdet, target, n_seg, dt, args.flow, args.fmax, blip_wf)
        ccsn_b = {d: _inject_into_bundle(noise_b[d], inj_c[d], edir / "ccsn" / f"{d}.hdf5") for d in detectors}
        glitch_b = dict(noise_b)
        glitch_b[gdet] = _inject_into_bundle(noise_b[gdet], inj_g, edir / "iglitch" / f"{gdet}.hdf5")

        events = {"noise": (noise_b, 0.0), "inj_ccsn": (ccsn_b, net_snr),
                  "inj_glitch": (glitch_b, g_snr), "real_glitch": (real_g, cat_snr)}
        for cls, (bundles, snr) in events.items():
            try:
                r = _run(bundles, detectors, edir / cls, args.flow, args.fmax, nsm, args.num_warmup, args.num_samples)
                gl = r.get("glitch", {})
                row = EventRow(event_id=eid, cls=cls, target_snr=snr, injected_snr=snr,
                               logZ_signal=float(r["signal"]["logZ"]),
                               logZ_glitch=float(np.nanmax(list(gl.values()))) if gl else np.nan,
                               logZ_noise=0.0, log_odds=float(r["bcr_log"]),
                               evidence_failures=int(r.get("evidence_failures", 0)))
                rows.append(row)
                (args.outdir / "rows.json").write_text(json.dumps([asdict(x) for x in rows], indent=2))
                print(f"[{cls:>11s} e{i}] snr={snr:6.1f} logZ_s={row.logZ_signal:8.1f} "
                      f"logZ_g={row.logZ_glitch:8.1f} logBCR={row.log_odds:9.1f}")
            except Exception as exc:  # noqa: BLE001
                print(f"[real-noise] {cls} e{i} FAILED: {type(exc).__name__}: {exc}")
            eid += 1

    if not rows:
        print("[real-noise] no events completed.")
        return
    odds = {c: np.array([r.log_odds for r in rows if r.cls == c]) for c in CLASSES}
    snr = {c: np.array([r.injected_snr for r in rows if r.cls == c]) for c in CLASSES}
    bg = np.concatenate([odds["noise"], odds["inj_glitch"], odds["real_glitch"]])
    bg_snr = np.concatenate([snr["noise"], snr["inj_glitch"], snr["real_glitch"]])
    summary = {
        "detectors": detectors,
        "auc_odds_signal_vs_background": _roc_auc(odds["inj_ccsn"], bg),
        "auc_snr_signal_vs_background": _roc_auc(snr["inj_ccsn"], bg_snr),
        "auc_odds_signal_vs_real_glitch": _roc_auc(odds["inj_ccsn"], odds["real_glitch"]),
        "auc_odds_signal_vs_inj_glitch": _roc_auc(odds["inj_ccsn"], odds["inj_glitch"]),
        "n": {c: int(odds[c].size) for c in CLASSES},
        "median_logBCR": {c: float(np.median(odds[c])) if odds[c].size else None for c in CLASSES},
        "real_glitch_misclassified": int(np.sum(odds["real_glitch"] > 0)),
        "inj_glitch_misclassified": int(np.sum(odds["inj_glitch"] > 0)),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
