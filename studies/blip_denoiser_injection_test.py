"""Phase 0c: does PCA-truncate-and-unwhiten recover a known waveform's shape?

Injects known waveforms (drawn from the CCSN decoder, standing in for an
arbitrary compact transient -- what matters is the shape is KNOWN, not that
it is a real blip) into real off-source noise at controlled in-band SNR, runs
the full extract -> PCA-denoise -> un-whiten chain used in the cheap route,
and scores the recovered raw-strain waveform against the injected truth.

The PCA basis for each test point is fit on the OTHER injections only
(leave-one-out), so this measures genuine out-of-sample denoising, not
self-consistency.

Gate (see docs/blip_prior_rebuild_plan.md): ship the cheap route if recovered
match >= 0.98 for in-band SNR >= 20; otherwise fall back to BayesWave.

    ./.venv/bin/python studies/blip_denoiser_injection_test.py \
        --noise-bundles ../local_ppc_paperfig/rn_H1_L1/e0/noise/L1/analysis_bundle_1261022268.hdf5 \
        --n-injections 40 --outdir ../blip_denoiser_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blip_core_extract import (
    CORE_HALF_WIDTH,
    EXTRACT_FLOW,
    EXTRACT_FMAX,
    extract_core,
    locate_peak,
    pca_denoise,
    unwhiten,
    whiten_bundle,
)
from real_noise_io import inject_into_bundle
from starccato_jax.waveforms import get_model

SEGMENT_N = 16384
SAMPLE_RATE = 4096.0


def _truth_waveforms(n: int, seed: int) -> np.ndarray:
    """n distinct 512-sample raw waveforms, standardized to unit RMS.

    Draws from the CCSN decoder: an independently-trained model with no
    relation to the blip population, so any recovery success here is not an
    artifact of testing a model against its own family.
    """
    dec = get_model("ccsne")
    z = np.random.default_rng(seed).normal(size=(n, 5)).astype(np.float32)
    wf = np.asarray(dec.generate(z=z))
    return wf / np.std(wf, axis=1, keepdims=True)


def _pad_centered(core: np.ndarray, n: int = SEGMENT_N) -> np.ndarray:
    out = np.zeros(n)
    mid = n // 2
    half = core.size // 2
    out[mid - half: mid + half] = core
    return out


def _whiten_raw(raw: np.ndarray, seg) -> np.ndarray:
    """Whiten an arbitrary raw time series with a segment's own PSD (no injection)."""
    n = seg.n
    df = 1.0 / (n * seg.dt)
    w = np.where(np.isfinite(seg.psd) & (seg.psd > 0), seg.psd, np.inf)
    spec = np.fft.rfft(raw) * seg.dt / np.sqrt(w / (4.0 * df))
    return np.fft.irfft(spec, n=n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noise-bundles", nargs="+", type=Path, required=True,
                   help="Real off-source noise bundles to inject into (cycled).")
    p.add_argument("--n-injections", type=int, default=40)
    p.add_argument("--snr-values", type=float, nargs="+", default=[10, 20, 40, 80])
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 15, 30, 60])
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    truths = _truth_waveforms(args.n_injections, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    target_snrs = rng.choice(args.snr_values, size=args.n_injections)

    # 1. build one injected bundle per truth waveform, scaled to target in-band SNR
    injected_segments = []
    truth_padded = []
    for i in range(args.n_injections):
        noise_bundle = args.noise_bundles[i % len(args.noise_bundles)]
        seg0 = whiten_bundle(noise_bundle, "L1", flow=EXTRACT_FLOW, fmax=EXTRACT_FMAX)
        raw = _pad_centered(truths[i])
        # optimal SNR = L2 norm of the whitened, noise-free template
        template_whitened = _whiten_raw(raw, seg0)
        current_snr = float(np.linalg.norm(template_whitened))
        scale = float(target_snrs[i]) / current_snr
        raw_scaled = raw * scale

        dest = args.outdir / f"injected_{i:03d}.hdf5"
        inject_into_bundle(noise_bundle, raw_scaled, dest)
        seg = whiten_bundle(dest, "L1", flow=EXTRACT_FLOW, fmax=EXTRACT_FMAX)
        injected_segments.append(seg)
        truth_padded.append(raw_scaled)

    # 2. extract a core from each (peak should sit at the segment centre, since
    #    the injection was centred -- but locate it for real, as production does)
    cores = []
    peak_indices = []
    for seg in injected_segments:
        pk = locate_peak(seg, search_half_width_ms=60.0)
        cores.append(extract_core(seg, pk, sign_align=False))
        peak_indices.append(pk)
    cores = np.array(cores)

    # 3. leave-one-out PCA denoise + un-whiten + score against truth, per k
    results = {k: [] for k in args.k_values}
    for i in range(args.n_injections):
        others = np.delete(cores, i, axis=0)
        for k in args.k_values:
            _, basis, mean = pca_denoise(others, k=k)  # basis fit on OTHERS only
            centred = cores[i] - mean
            projected = mean + centred @ basis[:k].T @ basis[:k]
            recovered_raw = unwhiten(projected, injected_segments[i], peak_indices[i])

            # score in RAW strain (the physically meaningful quantity a
            # template built this way would be used as), restricted to the
            # core window where the injection actually lives
            n = SEGMENT_N
            mid = n // 2
            window = slice(mid - CORE_HALF_WIDTH, mid + CORE_HALF_WIDTH)
            t = truth_padded[i][window]
            r = recovered_raw[window]
            match = float(np.dot(t, r) / (np.linalg.norm(t) * np.linalg.norm(r) + 1e-300))
            results[k].append({"snr": float(target_snrs[i]), "match": match})

    summary = {}
    for k in args.k_values:
        rows = results[k]
        snrs = np.array([r["snr"] for r in rows])
        matches = np.array([r["match"] for r in rows])
        by_snr = {}
        for s in sorted(set(args.snr_values)):
            m = snrs == s
            if m.sum():
                by_snr[str(s)] = {
                    "median_match": float(np.median(matches[m])),
                    "min_match": float(np.min(matches[m])),
                    "n": int(m.sum()),
                }
        summary[str(k)] = by_snr

    (args.outdir / "injection_recovery_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"{'k':>4} " + " ".join(f"snr={s:<5g}" for s in sorted(set(args.snr_values))))
    for k in args.k_values:
        row = summary[str(k)]
        cells = " ".join(
            f"{row.get(str(s), {}).get('median_match', float('nan')):9.3f}"
            for s in sorted(set(args.snr_values))
        )
        print(f"{k:4d} {cells}")
    print(f"\nwrote {args.outdir / 'injection_recovery_summary.json'}")

    gate_pass = all(
        summary[str(k)].get("20.0", summary[str(k)].get(str(20), {})).get("median_match", 0) >= 0.98
        for k in args.k_values
    )
    print("\nGate (median match >= 0.98 at SNR=20, all k):",
          "PASS -- cheap route is viable" if gate_pass else
          "FAIL -- consider BayesWave (Phase 3)")


if __name__ == "__main__":
    main()
