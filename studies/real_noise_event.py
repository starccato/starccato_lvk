"""Single-event runner for the real-noise analysis study -- built for SLURM arrays.

One array task = one blip-catalogue index = the four event classes of
``real_noise_study`` (noise, inj_ccsn, inj_glitch, real_glitch) in REAL GWOSC
noise, for the chosen detector network.

Two stages (matching the existing slurm/ design):
  --stage prep     : download the real bundles (needs internet -> data-mover node),
                     build the injected bundles, and write a per-index manifest.json.
  --stage analysis : read the manifest and run the BCR for each class (compute node),
                     writing results/e{index}_{cls}.json.
  --stage both     : do both in one process (local / internet-capable node).

Aggregate the per-event JSONs with ``real_noise_aggregate.py``.

    uv run python studies/real_noise_event.py --index 0 --detectors L1 --stage both --outdir out_rn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from starccato_lvk.analysis import run_bcr_posteriors
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
from starccato_lvk.acquisition.io.glitch_catalog import get_blip_trigger_time, load_blip_glitch_catalog

from real_data_smoke_test import _build_bundle
from real_data_roc import _inject_into_bundle
from snr_vs_odds_roc import SAMPLE_RATE
from snr_vs_odds_roc_coherent import _inject_coherent, _inject_single_glitch

try:
    from starccato_jax.data.training_data import TrainValData
except Exception:  # pragma: no cover
    TrainValData = None

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")


def prep_index(index, detectors, gdet, band, snr_grid, noise_offset, outdir, blip_ifo="L1") -> dict:
    """Download real bundles, build injected bundles, and return/write a manifest."""
    flow, fmax = band
    dt = 1.0 / SAMPLE_RATE
    n_seg = int(round(4.0 * SAMPLE_RATE))
    edir = outdir / f"e{index}"
    cat = load_blip_glitch_catalog(ifo=blip_ifo)
    blip_gps = get_blip_trigger_time(index, ifo=blip_ifo)
    noise_gps = blip_gps - noise_offset
    cat_snr = float(cat.iloc[index]["snr"])
    rng = np.random.default_rng(1000 + index)  # reproducible per-event draws
    # continuous injected SNR (log-uniform over the grid's span -> no discrete stripes)
    lo, hi = float(min(snr_grid)), float(max(snr_grid))
    target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    # isotropic targeted sky (inject AND recover here -- known-direction analysis);
    # sampling the sky is slow, non-converging under HMC, and lets glitches find
    # blind-spot skies under nested -- so we fix a matched per-event direction.
    sky = {"ra": float(rng.uniform(0.0, 2.0 * np.pi)),
           "dec": float(np.arcsin(rng.uniform(-1.0, 1.0))),
           "psi": float(rng.uniform(0.0, np.pi)), "t_c": 0.0}

    noise_b = {d: _build_bundle(noise_gps, edir / "noise" / d, detector=d) for d in detectors}
    real_g = {d: _build_bundle(blip_gps, edir / "realg" / d, detector=d) for d in detectors}

    prep = prepare_multi_detector_data(detectors, bundle_paths=noise_b, flow=flow, fmax=fmax)
    ccsn_pool = np.asarray(TrainValData.load(source="ccsne", seed=0).val) if TrainValData else None
    blip_pool = np.asarray(TrainValData.load(source="blip", seed=0).val) if TrainValData else None
    ccsn_wf = ccsn_pool[rng.integers(ccsn_pool.shape[0])] if ccsn_pool is not None else None
    blip_wf = blip_pool[rng.integers(blip_pool.shape[0])] if blip_pool is not None else None
    inj_c, net_snr = _inject_coherent(prep, target, n_seg, dt, flow, fmax, ccsn_wf, sky=sky)
    inj_g, g_snr = _inject_single_glitch(prep, gdet, target, n_seg, dt, flow, fmax, blip_wf)
    ccsn_b = {d: _inject_into_bundle(noise_b[d], inj_c[d], edir / "ccsn" / f"{d}.hdf5") for d in detectors}
    glitch_b = dict(noise_b)
    glitch_b[gdet] = _inject_into_bundle(noise_b[gdet], inj_g, edir / "iglitch" / f"{gdet}.hdf5")

    manifest = {
        "index": index, "detectors": detectors, "glitch_det": gdet, "blip_ifo": blip_ifo,
        "band": list(band),
        "sky": sky,  # targeted direction: recover the signal model here for every class
        "snr": {"noise": 0.0, "inj_ccsn": float(net_snr), "inj_glitch": float(g_snr), "real_glitch": cat_snr},
        "bundles": {
            "noise": {d: str(noise_b[d]) for d in detectors},
            "inj_ccsn": {d: str(ccsn_b[d]) for d in detectors},
            "inj_glitch": {d: str(glitch_b[d]) for d in detectors},
            "real_glitch": {d: str(real_g[d]) for d in detectors},
        },
    }
    edir.mkdir(parents=True, exist_ok=True)
    (edir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def analyse_manifest(manifest, outdir, num_warmup, num_samples, nsm) -> None:
    detectors = manifest["detectors"]
    flow, fmax = manifest["band"]
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    index = manifest["index"]
    for cls in CLASSES:
        out_json = results_dir / f"e{index}_{cls}.json"
        if out_json.exists():
            continue  # idempotent: skip already-analysed class (safe for SLURM re-runs)
        try:
            r = run_bcr_posteriors(
                detectors=detectors, outdir=str(outdir / f"e{index}" / cls),
                bundle_paths=manifest["bundles"][cls],
                extrinsic_params=manifest.get("sky"),  # recover signal model at targeted sky
                signal_model="ccsne", glitch_model="blip", flow=flow, fmax=fmax,
                num_warmup=num_warmup, num_samples=num_samples, save_artifacts=False,
                lnz_method="morph", noise_scale_marginal=nsm,
                nested_num_live_points=300, nested_max_samples=6000,
            )
            gl = r.get("glitch", {})
            row = {
                "index": index, "cls": cls, "detectors": detectors, "snr": manifest["snr"][cls],
                "logZ_signal": float(r["signal"]["logZ"]),
                "logZ_glitch": float(np.nanmax(list(gl.values()))) if gl else float("nan"),
                "log_odds": float(r["bcr_log"]),
                "evidence_failures": int(r.get("evidence_failures", 0)),
            }
            out_json.write_text(json.dumps(row, indent=2))
            print(f"[e{index} {cls:>11s}] snr={row['snr']:6.1f} logBCR={row['log_odds']:9.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[e{index} {cls}] FAILED: {type(exc).__name__}: {exc}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--detectors", nargs="+", default=["L1"])
    p.add_argument("--glitch-det", default="L1")
    p.add_argument("--blip-ifo", default="L1", choices=["H1", "L1"],
                   help="Gravity Spy catalogue for the real_glitch class (blip host detector).")
    p.add_argument("--stage", choices=["prep", "analysis", "both"], default="both")
    p.add_argument("--outdir", type=Path, default=Path("out_rn"))
    p.add_argument("--snr-grid", type=float, nargs="+", default=[10, 20, 40])
    p.add_argument("--noise-offset", type=float, default=100.0)
    p.add_argument("--flow", type=float, default=300.0)
    p.add_argument("--fmax", type=float, default=800.0)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--no-marginal", action="store_true")
    args = p.parse_args()

    detectors = [d.upper() for d in args.detectors]
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / f"e{args.index}" / "manifest.json"

    if args.stage in ("prep", "both"):
        prep_index(args.index, detectors, args.glitch_det.upper(), (args.flow, args.fmax),
                   args.snr_grid, args.noise_offset, args.outdir, blip_ifo=args.blip_ifo)
        print(f"[e{args.index}] prep done -> {manifest_path}")
    if args.stage in ("analysis", "both"):
        manifest = json.loads(manifest_path.read_text())
        analyse_manifest(manifest, args.outdir, args.num_warmup, args.num_samples, not args.no_marginal)


if __name__ == "__main__":
    main()
