"""Multi-detector COHERENT SNR-vs-Odds ROC (default H1+L1) on design-PSD noise.

This is the headline demonstration: the differentiator from SNR is *coherence*. A
real CCSN appears in every detector as one waveform from a shared sky location
(shared latent z, projected through each detector's antenna response + geocentre
delay); a blip glitch appears incoherently in a single detector. SNR (network
quadrature sum) cannot tell an SNR-matched coherent signal from an incoherent
glitch -- the coherent Bayesian odds can.

Event classes:
    noise   : design-PSD noise in every detector                 (background)
    signal  : ONE held-out CCSN injected COHERENTLY into all      (foreground)
              detectors at fixed sky, scaled to a network SNR
    glitch  : ONE held-out blip injected into a SINGLE detector   (background)
              (incoherent), noise in the others, same network SNR

Recovery (``run_bcr_posteriors``) already implements the coherent signal model
(shared latents across detectors) and per-detector glitch models, and assembles
the BCR. Injections are off-manifold (held-out real waveforms) and the PSD is an
off-source Welch estimate by default -- the honest setup.

    uv run python studies/snr_vs_odds_roc_coherent.py --detectors H1 L1 \
        --n-per-class 5 --snr-grid 8 14 22 --outdir out_roc_2det
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import jax.numpy as jnp

from starccato_jax.waveforms import get_model
from starccato_lvk.analysis.jim_waveform import StarccatoJimWaveform, StarccatoGlitchWaveform
from starccato_lvk.analysis.main import run_bcr_posteriors, _clone_no_response_detector
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data

from simulated_design_psd import build_design_psd, pycbc_psd_to_gwpy
from snr_vs_odds_roc import (  # reuse the validated single-detector machinery
    SAMPLE_RATE, SEGMENT_DURATION, PSD_OFFSOURCE_SECONDS, BASE_TRIGGER,
    EventRow, simulate_noise_fd, _center_pad, _event_noise_and_psd,
    _write_event_bundle, _roc_auc,
)


def _build_h_sky_unit(prep, model_name: str, n_seg: int, dt: float,
                      raw_wf: Optional[np.ndarray]):
    """Unit-amplitude sky-frame FD waveform (full grid): held-out raw or VAE z=0."""
    full_freqs = np.fft.rfftfreq(n_seg, d=dt)
    fj = jnp.asarray(full_freqs)
    if raw_wf is not None:
        td = _center_pad(np.asarray(raw_wf, dtype=np.float64), n_seg) * 1e-21 * np.asarray(prep.window)
        return jnp.fft.rfft(jnp.asarray(td)) * dt
    sr = 1.0 / dt
    wf = (StarccatoJimWaveform(model=get_model(model_name), sample_rate=sr, window=prep.window)
          if model_name == "ccsne" else
          StarccatoGlitchWaveform(model=get_model(model_name), sample_rate=sr,
                                  window=prep.window, strain_scale=1e-21))
    params = {name: 0.0 for name in wf.latent_names}
    params["log_amp"] = 0.0
    return wf(fj, params)["p"]


def _snr(hdec: np.ndarray, psd_full: np.ndarray, band: np.ndarray, df: float) -> float:
    return float(np.sqrt(4.0 * np.sum(np.abs(hdec[band]) ** 2 / psd_full[band]) * df))


def _inject_coherent(prep, target_net_snr: float, n_seg: int, dt: float,
                     flow: float, fmax: float, raw_wf,
                     sky: Optional[dict] = None) -> tuple[Dict[str, np.ndarray], float]:
    """Inject ONE CCSN coherently across all detectors, scaled to a network SNR.

    ``sky`` gives the source direction (ra/dec/psi/t_c); default is the network
    zenith (0,0,0). For a realistic targeted study pass an isotropic draw and
    recover at the SAME sky (extrinsic_params) -- see studies/real_noise_event.py.
    """
    full_freqs = np.fft.rfftfreq(n_seg, d=dt)
    fj = jnp.asarray(full_freqs)
    df = float(full_freqs[1] - full_freqs[0])
    band = (full_freqs >= flow) & (full_freqs <= fmax)
    h_sky = _build_h_sky_unit(prep, "ccsne", n_seg, dt, raw_wf)
    sky = {"t_c": 0.0, "ra": 0.0, "dec": 0.0, "psi": 0.0, **(sky or {})}
    sky.update(gmst=float(prep.gmst), trigger_time=float(prep.trigger_time))

    unit_hdec, net_sq = {}, 0.0
    for det in prep.detectors:
        hd = np.asarray(det.fd_response(fj, {"p": h_sky, "c": h_sky}, sky))
        unit_hdec[det.name] = hd
        net_sq += _snr(hd, np.asarray(det.psd.values), band, df) ** 2
    amp = target_net_snr / np.sqrt(net_sq)

    inj, net_sq2 = {}, 0.0
    for det in prep.detectors:
        hd = unit_hdec[det.name] * amp
        inj[det.name] = np.fft.irfft(hd, n=n_seg) / dt
        net_sq2 += _snr(hd, np.asarray(det.psd.values), band, df) ** 2
    return inj, float(np.sqrt(net_sq2))


def _inject_single_glitch(prep, gdet_name: str, target_snr: float, n_seg: int, dt: float,
                          flow: float, fmax: float, raw_wf) -> tuple[np.ndarray, float]:
    """Inject ONE blip into a single detector (no response, incoherent)."""
    full_freqs = np.fft.rfftfreq(n_seg, d=dt)
    fj = jnp.asarray(full_freqs)
    df = float(full_freqs[1] - full_freqs[0])
    band = (full_freqs >= flow) & (full_freqs <= fmax)
    src = next(d for d in prep.detectors if d.name == gdet_name)
    det = _clone_no_response_detector(src)
    psd_full = np.asarray(det.psd.values)
    h_sky = _build_h_sky_unit(prep, "blip", n_seg, dt, raw_wf)
    hd_unit = np.asarray(det.fd_response(fj, {"p": h_sky, "c": h_sky}, {}))
    amp = target_snr / _snr(hd_unit, psd_full, band, df)
    hd = hd_unit * amp
    return np.fft.irfft(hd, n=n_seg) / dt, _snr(hd, psd_full, band, df)


def run_event(event_id: int, cls: str, target_snr: float, *, detectors: List[str],
              psd_by_det: dict, n_seg: int, outdir: Path, flow: float, fmax: float,
              bcr_kwargs: dict, raw_wf=None, noise_psd: Optional[dict] = None,
              glitch_det: Optional[str] = None) -> EventRow:
    dt = 1.0 / SAMPLE_RATE
    trigger = BASE_TRIGGER + event_id
    event_dir = outdir / cls / f"event_{event_id}"
    gdet = (glitch_det or detectors[-1]).upper()

    noise, psd_fs = {}, {}
    for di, det in enumerate(detectors):
        long = (noise_psd or {}).get("long", {}).get(det) if noise_psd else None
        n, pfs = _event_noise_and_psd(
            8 * event_id + di, cls, n_seg, dt,
            design_psd_4s=psd_by_det[det]["vals"], psd_fs_design=psd_by_det[det]["fs"],
            design_psd_long=long, use_welch=(noise_psd or {}).get("use_welch", False),
        )
        noise[det], psd_fs[det] = n, pfs

    injected_snr = 0.0
    if cls in ("signal", "glitch"):
        noise_bundles = {det: _write_event_bundle(event_dir / "noise" / det, noise[det], psd_fs[det], trigger)
                         for det in detectors}
        prep = prepare_multi_detector_data(detectors, bundle_paths=noise_bundles, flow=flow, fmax=fmax)
        if cls == "signal":
            inj, injected_snr = _inject_coherent(prep, target_snr, n_seg, dt, flow, fmax, raw_wf)
            for det in detectors:
                noise[det] = noise[det] + inj[det]
        else:
            inj, injected_snr = _inject_single_glitch(prep, gdet, target_snr, n_seg, dt, flow, fmax, raw_wf)
            noise[gdet] = noise[gdet] + inj

    bundles = {det: str(_write_event_bundle(event_dir / "data" / det, noise[det], psd_fs[det], trigger))
               for det in detectors}
    result = run_bcr_posteriors(
        detectors=detectors, outdir=str(event_dir / "analysis"),
        bundle_paths=bundles, flow=flow, fmax=fmax, save_artifacts=False, **bcr_kwargs,
    )
    glitch_d = result.get("glitch", {})
    return EventRow(
        event_id=event_id, cls=cls, target_snr=target_snr, injected_snr=injected_snr,
        logZ_signal=float(result["signal"].get("logZ", np.nan)),
        logZ_glitch=float(np.nanmax(list(glitch_d.values()))) if glitch_d else np.nan,
        logZ_noise=0.0, log_odds=float(result.get("bcr_log", np.nan)),
        evidence_failures=int(result.get("evidence_failures", 0)),
    )


def _summary(rows: List[EventRow]) -> dict:
    by = {c: [r for r in rows if r.cls == c] for c in ("noise", "signal", "glitch")}
    snr = {c: np.array([r.injected_snr for r in by[c]]) for c in by}
    odds = {c: np.array([r.log_odds for r in by[c]]) for c in by}
    bg_snr = np.concatenate([snr["noise"], snr["glitch"]])
    bg_odds = np.concatenate([odds["noise"], odds["glitch"]])
    return {
        "auc_snr_signal_vs_background": _roc_auc(snr["signal"], bg_snr),
        "auc_odds_signal_vs_background": _roc_auc(odds["signal"], bg_odds),
        "auc_snr_signal_vs_glitch": _roc_auc(snr["signal"], snr["glitch"]),
        "auc_odds_signal_vs_glitch": _roc_auc(odds["signal"], odds["glitch"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    p.add_argument("--glitch-det", default=None, help="Detector hosting the blip (default: last).")
    p.add_argument("--outdir", type=Path, default=Path("out_roc_2det"))
    p.add_argument("--cache-dir", type=Path, default=Path("design_psd_cache"))
    p.add_argument("--n-per-class", type=int, default=5)
    p.add_argument("--snr-grid", type=float, nargs="+", default=[8, 14, 22, 32])
    p.add_argument("--flow", type=float, default=100.0)
    p.add_argument("--fmax", type=float, default=1024.0)
    p.add_argument("--num-warmup", type=int, default=150)
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--lnz-method", choices=["morph", "nested"], default="morph")
    p.add_argument("--on-manifold", action="store_true")
    p.add_argument("--ideal-psd", action="store_true")
    args = p.parse_args()

    detectors = [d.upper() for d in args.detectors]
    args.outdir.mkdir(parents=True, exist_ok=True)
    n_seg = int(round(SEGMENT_DURATION * SAMPLE_RATE))
    n_freq = n_seg // 2 + 1

    psd_by_det = {}
    for det in detectors:
        pc = build_design_psd(det, 1.0 / SEGMENT_DURATION, n_freq, args.cache_dir)
        psd_by_det[det] = {"vals": np.asarray(pc.numpy(), dtype=np.float64), "fs": pycbc_psd_to_gwpy(pc)}

    use_welch = not args.ideal_psd
    noise_psd = {"use_welch": use_welch, "long": {}}
    if use_welch:
        n_long = int(round(PSD_OFFSOURCE_SECONDS * SAMPLE_RATE)) + n_seg
        for det in detectors:
            lp = build_design_psd(det, 1.0 / (n_long / SAMPLE_RATE), n_long // 2 + 1, args.cache_dir)
            noise_psd["long"][det] = np.asarray(lp.numpy(), dtype=np.float64)

    held_out = {}
    if not args.on_manifold:
        from starccato_jax.data.training_data import TrainValData
        held_out["signal"] = np.asarray(TrainValData.load(source="ccsne", seed=0).val)
        held_out["glitch"] = np.asarray(TrainValData.load(source="blip", seed=0).val)
    wf_rng = np.random.default_rng(20240501)

    print(f"[roc-2det] detectors={detectors} glitch_det={(args.glitch_det or detectors[-1]).upper()} "
          f"off_manifold={not args.on_manifold} welch_psd={use_welch}")

    bcr_kwargs = dict(signal_model="ccsne", glitch_model="blip",
                      num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=1,
                      lnz_method=args.lnz_method, nested_num_live_points=200, nested_max_samples=4000,
                      verify_logz_threshold=None)  # morphZ reliable on sim; skip cross-check for speed

    rows: List[EventRow] = []
    event_id = 0
    for snr in args.snr_grid:
        for _ in range(args.n_per_class):
            for cls in ("noise", "signal", "glitch"):
                raw_wf = None
                if cls in held_out:
                    pool = held_out[cls]
                    raw_wf = pool[wf_rng.integers(pool.shape[0])]
                row = run_event(event_id, cls, snr, detectors=detectors, psd_by_det=psd_by_det,
                                n_seg=n_seg, outdir=args.outdir, flow=args.flow, fmax=args.fmax,
                                bcr_kwargs=bcr_kwargs, raw_wf=raw_wf, noise_psd=noise_psd,
                                glitch_det=args.glitch_det)
                print(f"[{cls:>6s} id={event_id:3d}] net_snr={row.injected_snr:6.2f} "
                      f"logZ_s={row.logZ_signal:8.1f} logZ_g={row.logZ_glitch:8.1f} logBCR={row.log_odds:9.1f}")
                rows.append(row)
                event_id += 1

    (args.outdir / "roc_rows.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
    aucs = _summary(rows)
    (args.outdir / "roc_summary.json").write_text(json.dumps(aucs, indent=2))
    print("\nAUC summary (coherent multi-detector):")
    for k, v in aucs.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
