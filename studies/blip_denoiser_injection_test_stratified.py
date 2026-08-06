"""Phase 0c follow-up: does frequency-stratifying the PCA basis fix the wide-band plateau?

The pooled wide-band gate (blip_denoiser_injection_test_realpop_wide.py)
plateaus well below the 0.98 target because peak_frequency spans ~90-1200 Hz
-- structurally different time-domain shapes sharing one PCA basis. Phase 1's
actual design already splits the population this way (gengli below 250 Hz,
real cores above, per the Phase 0b gate), so the decision-relevant test is
whether a basis fit WITHIN a frequency group recovers something closer to the
narrow-band ceiling (0.975) than the pooled 0.73.

    ./.venv/bin/python studies/blip_denoiser_injection_test_stratified.py
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
from blip_stratified_sample import load_catalog
from real_noise_io import inject_into_bundle

SEGMENT_N = 16384
PTF_THRESHOLD = 10.0
SPLIT_HZ = 250.0


def _pad_centered(core: np.ndarray, n: int = SEGMENT_N) -> np.ndarray:
    out = np.zeros(n)
    mid = n // 2
    half = core.size // 2
    out[mid - half: mid + half] = core
    return out


def main() -> None:
    catalogs = {"L1": load_catalog("L1", "O3b"), "H1": load_catalog("H1", "O3b")}

    trusted = []
    for ifo_dir, ifo in ((Path("../blip_stratified/L1_O3b"), "L1"),
                         (Path("../blip_stratified/H1_O3b"), "H1")):
        for b in sorted(ifo_dir.glob(f"e*/{ifo}/analysis_bundle_*.hdf5")):
            row = int(b.parent.parent.name[1:])
            peak_f = float(catalogs[ifo].iloc[row].peak_frequency)
            seg = whiten_bundle(b, ifo)
            pk = locate_peak(seg, search_half_width_ms=250.0)
            ptf = peak_to_floor(seg, pk)
            if ptf < PTF_THRESHOLD:
                continue
            core = extract_core(seg, pk, sign_align=True)
            truth_raw = unwhiten(core, seg, pk)
            mid = seg.n // 2
            truth_core_raw = truth_raw[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]
            trusted.append({"ifo": ifo, "peak_f": peak_f, "core_wh": core,
                            "truth_core_raw": truth_core_raw})

    low = [t for t in trusted if t["peak_f"] < SPLIT_HZ]
    high = [t for t in trusted if t["peak_f"] >= SPLIT_HZ]
    print(f"trusted cores: {len(trusted)} total -> {len(low)} below {SPLIT_HZ} Hz, "
          f"{len(high)} at/above")

    fresh_noise = sorted(Path("../blip_denoiser_gate/noise_pool").glob("n*/analysis_bundle_*.hdf5"))
    if not fresh_noise:
        print("No fresh L1 noise pool found.")
        return

    snr_grid = [10.0, 20.0, 40.0]
    k_grid = [5, 10, 15]
    out = Path("../blip_denoiser_gate/stratified")
    out.mkdir(parents=True, exist_ok=True)

    def score_group(group: list[dict], label: str) -> dict:
        l1_group = [t for t in group if t["ifo"] == "L1"]
        if len(l1_group) < 6:
            print(f"{label}: only {len(l1_group)} L1 cores, too few for leave-one-out; skipping")
            return {}
        results = {k: {s: [] for s in snr_grid} for k in k_grid}
        for i, t in enumerate(l1_group):
            noise_bundle = fresh_noise[i % len(fresh_noise)]
            seg_noise = whiten_bundle(noise_bundle, "L1")
            template_padded = _pad_centered(t["truth_core_raw"])
            n = seg_noise.n
            df = 1.0 / (n * seg_noise.dt)
            w = np.where(np.isfinite(seg_noise.psd) & (seg_noise.psd > 0), seg_noise.psd, np.inf)
            template_wh = np.fft.irfft(
                np.fft.rfft(template_padded) * seg_noise.dt / np.sqrt(w / (4.0 * df)), n=n
            )
            current_norm = float(np.linalg.norm(template_wh))
            if current_norm == 0 or not np.isfinite(current_norm):
                continue
            others = np.array([o["core_wh"] for j, o in enumerate(l1_group) if j != i])

            for target_snr in snr_grid:
                scale = target_snr / current_norm
                raw_scaled = template_padded * scale
                dest = out / f"{label}_{i:03d}_snr{int(target_snr)}.hdf5"
                inject_into_bundle(noise_bundle, raw_scaled, dest)
                seg = whiten_bundle(dest, "L1")
                pk = locate_peak(seg, search_half_width_ms=60.0)
                core = extract_core(seg, pk, sign_align=False)
                mid = n // 2
                truth_window = raw_scaled[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]
                for k in k_grid:
                    if k >= others.shape[0]:
                        continue
                    _, basis, mean = pca_denoise(others, k=k)
                    centred = core - mean
                    projected = mean + centred @ basis[:k].T @ basis[:k]
                    recovered = unwhiten(projected, seg, pk)
                    r = recovered[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]
                    match = float(np.dot(truth_window, r)
                                 / (np.linalg.norm(truth_window) * np.linalg.norm(r) + 1e-300))
                    results[k][target_snr].append(match)
        return results

    for group, label in ((low, "low"), (high, "high")):
        results = score_group(group, label)
        if not results:
            continue
        print(f"\n=== {label} ({'<' if label=='low' else '>='} {SPLIT_HZ} Hz), "
              f"within-group basis ===")
        print(f"{'k':>4} " + " ".join(f"snr={s:<5g}" for s in snr_grid))
        summary = {}
        for k in k_grid:
            cells = []
            summary[str(k)] = {}
            for s in snr_grid:
                vals = np.array(results[k][s])
                if len(vals) == 0:
                    cells.append("     n/a")
                    continue
                summary[str(k)][str(s)] = {"median": float(np.median(vals)), "n": len(vals)}
                cells.append(f"{np.median(vals):9.3f}")
            print(f"{k:4d} " + " ".join(cells))
        (out / f"summary_{label}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
