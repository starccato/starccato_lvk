"""Phase 0c, real-population variant: same band as the self-similarity check.

The generic version of this test (blip_denoiser_injection_test.py) uses CCSN
decoder draws as "truth" -- a structurally diverse family -- and measures a
~0.85 ceiling independent of SNR. That conflates two different limits: PCA
basis generalization on a diverse family, and degradation from noise.

Real blip cores are far more self-similar (median pairwise match 0.918,
noise-free leave-one-out match 0.983, both in the SAME 300-800 Hz band used
throughout this investigation). This script asks the sharper question: take
TRUSTED loud real cores (already shown self-similar), rescale them down to a
target SNR, re-inject into FRESH off-source noise, and see how much a basis
built from the OTHER trusted cores recovers -- isolating the noise-robustness
question for the population that actually matters, apples-to-apples with the
0.983 no-noise number.

Runs in the 300-800 Hz analysis band (not the wide 30-1024 Hz extraction band
used elsewhere), matching the band the 0.983 baseline was measured in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blip_core_extract import (
    CORE_HALF_WIDTH,
    extract_core,
    locate_peak,
    pca_denoise,
    peak_to_floor,
    unwhiten,
    whiten_bundle,
)
from real_noise_io import inject_into_bundle

FLOW, FMAX = 300.0, 800.0
SEGMENT_N = 16384
PTF_THRESHOLD = 10.0  # matches the earlier self-similarity study's cut


def _pad_centered(core: np.ndarray, n: int = SEGMENT_N) -> np.ndarray:
    out = np.zeros(n)
    mid = n // 2
    half = core.size // 2
    out[mid - half: mid + half] = core
    return out


def main() -> None:
    # 1. trusted cores: real blips whose in-band peak clearly dominates the
    #    floor, exactly the population the earlier self-similarity check used.
    trusted = []
    for bdir in sorted(Path("../blip_ff_study/bundles").glob("e*/L1"),
                       key=lambda p: int(p.parent.name[1:])):
        b = sorted(bdir.glob("analysis_bundle_*.hdf5"))
        if not b:
            continue
        seg = whiten_bundle(b[0], "L1", flow=FLOW, fmax=FMAX)
        pk = locate_peak(seg, search_half_width_ms=250.0)
        ptf = peak_to_floor(seg, pk)
        if ptf < PTF_THRESHOLD:
            continue
        core = extract_core(seg, pk, sign_align=True)
        truth_raw = unwhiten(core, seg, pk)
        mid = seg.n // 2
        truth_core_raw = truth_raw[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]
        trusted.append({"idx": int(bdir.parent.name[1:]), "core_wh": core,
                        "truth_core_raw": truth_core_raw, "ptf": ptf})
    print(f"trusted real cores (peak/floor>={PTF_THRESHOLD}, {FLOW}-{FMAX} Hz): {len(trusted)}")
    if len(trusted) < 6:
        print("Too few trusted cores for a leave-one-out test; widen the sample "
              "(Phase 0a) or lower PTF_THRESHOLD.")
        return

    fresh_noise = sorted(Path("../blip_denoiser_gate/noise_pool").glob("n*/analysis_bundle_*.hdf5"))
    if not fresh_noise:
        print("No fresh noise pool found; run the noise_pool fetch first.")
        return

    snr_grid = [10.0, 20.0, 40.0, 80.0]
    k_grid = [3, 5, 8]
    out = Path("../blip_denoiser_gate/realpop")
    out.mkdir(parents=True, exist_ok=True)

    results = {k: {s: [] for s in snr_grid} for k in k_grid}
    for i, t in enumerate(trusted):
        noise_bundle = fresh_noise[i % len(fresh_noise)]
        seg_noise = whiten_bundle(noise_bundle, "L1", flow=FLOW, fmax=FMAX)
        template_padded = _pad_centered(t["truth_core_raw"])
        template_wh = np.fft.irfft(
            np.fft.rfft(template_padded) * seg_noise.dt
            / np.sqrt(np.where(np.isfinite(seg_noise.psd) & (seg_noise.psd > 0),
                               seg_noise.psd, np.inf) / (4.0 / (seg_noise.n * seg_noise.dt))),
            n=seg_noise.n,
        )
        current_norm = float(np.linalg.norm(template_wh))

        for target_snr in snr_grid:
            scale = target_snr / current_norm
            raw_scaled = template_padded * scale
            dest = out / f"inj_{i:03d}_snr{int(target_snr)}.hdf5"
            inject_into_bundle(noise_bundle, raw_scaled, dest)
            seg = whiten_bundle(dest, "L1", flow=FLOW, fmax=FMAX)
            pk = locate_peak(seg, search_half_width_ms=60.0)
            core = extract_core(seg, pk, sign_align=False)

            others = np.array([o["core_wh"] for j, o in enumerate(trusted) if j != i])
            mid = seg.n // 2
            truth_window = raw_scaled[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]
            for k in k_grid:
                _, basis, mean = pca_denoise(others, k=k)
                centred = core - mean
                projected = mean + centred @ basis[:k].T @ basis[:k]
                recovered = unwhiten(projected, seg, pk)
                r = recovered[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]
                match = float(np.dot(truth_window, r)
                             / (np.linalg.norm(truth_window) * np.linalg.norm(r) + 1e-300))
                results[k][target_snr].append(match)

    print(f"\n{'k':>4} " + " ".join(f"snr={s:<5g}" for s in snr_grid))
    summary = {}
    for k in k_grid:
        summary[str(k)] = {}
        cells = []
        for s in snr_grid:
            vals = np.array(results[k][s])
            summary[str(k)][str(s)] = {"median": float(np.median(vals)),
                                       "min": float(np.min(vals)), "n": len(vals)}
            cells.append(f"{np.median(vals):9.3f}")
        print(f"{k:4d} " + " ".join(cells))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nreference (no added noise, same band): LOO match median 0.983 (n=19, earlier session)")
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
