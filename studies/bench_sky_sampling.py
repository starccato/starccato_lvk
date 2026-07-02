"""Benchmark: fixed-sky vs sampled-sky signal model (speed + convergence).

Answers two questions before we commit to sky sampling in the 2-detector study:
  1. How much slower is NUTS when ra/dec/psi/t_c are sampled?
  2. Does it still converge (r_hat, ESS, divergences)?

Runs on SIMULATED design-PSD noise (no real strain needed), reusing the coherent
ROC harness. For a coherently-injected CCSN and a single-detector glitch, it runs
the signal model twice -- sky fixed at (0,0,0) and sky sampled -- with several
chains, and reports wall-time and convergence diagnostics side by side.

    uv run python studies/bench_sky_sampling.py --detectors H1 L1 --num-chains 4
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import numpy as np
import jax

from starccato_jax.waveforms import get_model
from starccato_jax.data.training_data import TrainValData
from starccato_lvk.analysis.jim_waveform import StarccatoJimWaveform
from starccato_lvk.analysis.jim_likelihood import build_transient_likelihood, run_numpyro_sampling, run_nested_sampling
from starccato_lvk.analysis.main import DEFAULT_EXTRINSICS
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data

from simulated_design_psd import build_design_psd, pycbc_psd_to_gwpy
from snr_vs_odds_roc import SAMPLE_RATE, SEGMENT_DURATION, BASE_TRIGGER, _event_noise_and_psd, _write_event_bundle
from snr_vs_odds_roc_coherent import _inject_coherent, _inject_single_glitch


def _rhat_ess(grouped: Dict[str, np.ndarray]) -> Dict[str, tuple]:
    """Return {param: (max_rhat, min_ess)} across scalar components using arviz."""
    import arviz as az

    out = {}
    for name, arr in grouped.items():
        a = np.asarray(arr)  # (chain, draw) or (chain, draw, dim)
        if a.ndim == 2:
            a = a[:, :, None]
        rhats, esss = [], []
        for k in range(a.shape[2]):
            d = az.convert_to_dataset(a[:, :, k])  # (chain, draw) -> scalar variable
            rhats.append(float(np.asarray(az.rhat(d).x.values)))
            esss.append(float(np.asarray(az.ess(d).x.values)))
        out[name] = (max(rhats), min(esss))
    return out


def _build_event(detectors, cls, snr, psd_by_det, n_seg, dt, flow, fmax, outdir, raw_ccsn, raw_blip):
    trigger = BASE_TRIGGER
    noise, psd_fs = {}, {}
    for di, det in enumerate(detectors):
        n, pfs = _event_noise_and_psd(8 + di, cls, n_seg, dt,
                                      design_psd_4s=psd_by_det[det]["vals"],
                                      psd_fs_design=psd_by_det[det]["fs"],
                                      design_psd_long=None, use_welch=False)
        noise[det], psd_fs[det] = n, pfs
    nb = {d: _write_event_bundle(outdir / cls / "noise" / d, noise[d], psd_fs[d], trigger) for d in detectors}
    prep = prepare_multi_detector_data(detectors, bundle_paths=nb, flow=flow, fmax=fmax)
    if cls == "signal":
        inj, _ = _inject_coherent(prep, snr, n_seg, dt, flow, fmax, raw_ccsn)
        for d in detectors:
            noise[d] = noise[d] + inj[d]
    else:
        inj, _ = _inject_single_glitch(prep, detectors[-1], snr, n_seg, dt, flow, fmax, raw_blip)
        noise[detectors[-1]] = noise[detectors[-1]] + inj
    bundles = {d: str(_write_event_bundle(outdir / cls / "data" / d, noise[d], psd_fs[d], trigger)) for d in detectors}
    return prepare_multi_detector_data(detectors, bundle_paths=bundles, flow=flow, fmax=fmax)


def _run(prep, sample_sky, num_warmup, num_samples, num_chains):
    waveform = StarccatoJimWaveform(model=get_model("ccsne"), sample_rate=SAMPLE_RATE, window=prep.window)
    like = build_transient_likelihood(prep.detectors, waveform, trigger_time=prep.trigger_time,
                                      duration=prep.duration, post_trigger_duration=prep.post_trigger_duration)
    extr = dict(DEFAULT_EXTRINSICS, gmst=prep.gmst, trigger_time=prep.trigger_time)
    t0 = time.perf_counter()
    res = run_numpyro_sampling(like, latent_names=waveform.latent_names, fixed_params=extr,
                               rng_key=jax.random.PRNGKey(0), num_warmup=num_warmup,
                               num_samples=num_samples, num_chains=num_chains,
                               progress_bar=False, sample_sky=sample_sky)
    wall = time.perf_counter() - t0
    div = int(np.sum(res.extra.get("diverging", np.zeros(1)))) if "diverging" in res.extra else -1
    # skip wrapped deterministics (ra/dec) in the aggregate; diagnose the sampled sites
    grouped = {k: v for k, v in res.extra["samples_grouped"].items() if k not in ("ra", "dec")}
    diag = _rhat_ess(grouped)
    return wall, div, diag


def _run_nested(prep, sample_sky, live, max_samples):
    waveform = StarccatoJimWaveform(model=get_model("ccsne"), sample_rate=SAMPLE_RATE, window=prep.window)
    like = build_transient_likelihood(prep.detectors, waveform, trigger_time=prep.trigger_time,
                                      duration=prep.duration, post_trigger_duration=prep.post_trigger_duration)
    extr = dict(DEFAULT_EXTRINSICS, gmst=prep.gmst, trigger_time=prep.trigger_time)
    t0 = time.perf_counter()
    res = run_nested_sampling(like, latent_names=waveform.latent_names, fixed_params=extr,
                              rng_key=jax.random.PRNGKey(0), num_live_points=live,
                              max_samples=max_samples, num_posterior_samples=200,
                              verbose=False, sample_sky=sample_sky)
    return time.perf_counter() - t0, res.logZ, res.logZ_err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    p.add_argument("--cache-dir", default="design_psd_cache")
    p.add_argument("--snr", type=float, default=20.0)
    p.add_argument("--flow", type=float, default=100.0)
    p.add_argument("--fmax", type=float, default=1024.0)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-chains", type=int, default=4)
    p.add_argument("--outdir", default="out_bench_sky")
    p.add_argument("--nested", action="store_true", help="Benchmark nested sampling (evidence) instead of NUTS diagnostics.")
    p.add_argument("--live", type=int, default=500)
    p.add_argument("--max-samples", type=int, default=40000)
    args = p.parse_args()

    from pathlib import Path
    detectors = [d.upper() for d in args.detectors]
    outdir = Path(args.outdir)
    n_seg = int(round(SEGMENT_DURATION * SAMPLE_RATE))
    dt = 1.0 / SAMPLE_RATE

    psd_by_det = {}
    for det in detectors:
        pc = build_design_psd(det, 1.0 / SEGMENT_DURATION, n_seg // 2 + 1, Path(args.cache_dir))
        psd_by_det[det] = {"vals": np.asarray(pc.numpy(), dtype=np.float64), "fs": pycbc_psd_to_gwpy(pc)}

    raw_ccsn = np.asarray(TrainValData.load(source="ccsne", seed=0).val)[0]
    raw_blip = np.asarray(TrainValData.load(source="blip", seed=0).val)[0]

    print(f"detectors={detectors} snr={args.snr} chains={args.num_chains} "
          f"warmup={args.num_warmup} samples={args.num_samples}\n")
    for cls in ("signal", "glitch"):
        prep = _build_event(detectors, cls, args.snr, psd_by_det, n_seg, dt,
                            args.flow, args.fmax, outdir, raw_ccsn, raw_blip)
        print(f"### event class: {cls}")
        if args.nested:
            for sky in (False, True):
                wall, logZ, err = _run_nested(prep, sky, args.live, args.max_samples)
                tag = "sampled-sky" if sky else "fixed-sky  "
                print(f"  [nested {tag}] wall={wall:6.1f}s  logZ_signal={logZ:9.2f} +- {err:.2f}")
            print()
            continue
        for sky in (False, True):
            wall, div, diag = _run(prep, sky, args.num_warmup, args.num_samples, args.num_chains)
            tag = "sampled-sky" if sky else "fixed-sky  "
            worst_rhat = max(v[0] for v in diag.values())
            min_ess = min(v[1] for v in diag.values())
            print(f"  [{tag}] wall={wall:6.1f}s  divergences={div:4d}  "
                  f"max_rhat={worst_rhat:.3f}  min_ess={min_ess:6.0f}")
            if sky:
                for nm in ("u_sky", "psi", "t_c"):
                    if nm in diag:
                        print(f"        {nm:8s} rhat={diag[nm][0]:.3f} ess={diag[nm][1]:6.0f}")
        print()


if __name__ == "__main__":
    main()
