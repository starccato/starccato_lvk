"""Phase 0c, real-population, WIDE band, full stratified sample.

Extends the original real-population gate (300-800 Hz, n=19 loud/narrowband
cores) to the wide 30-1024 Hz extraction band that Phase 1 curation will
actually use, on the full stratified sample (peak/floor >= 10, both
detectors) rather than just the loud tail. Both blockers found in the first
pass (unwhiten peak-misalignment, wide-band boundary artifact) are fixed in
blip_core_extract.py -- this is the number that should decide Phase 1.

    ./.venv/bin/python studies/blip_denoiser_injection_test_realpop_wide.py
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

SEGMENT_N = 16384
PTF_THRESHOLD = 10.0


def _pad_centered(core: np.ndarray, n: int = SEGMENT_N) -> np.ndarray:
    out = np.zeros(n)
    mid = n // 2
    half = core.size // 2
    out[mid - half: mid + half] = core
    return out


def main() -> None:
    trusted = []
    for ifo_dir, ifo in ((Path("../blip_stratified/L1_O3b"), "L1"),
                         (Path("../blip_stratified/H1_O3b"), "H1")):
        for b in sorted(ifo_dir.glob(f"e*/{ifo}/analysis_bundle_*.hdf5")):
            seg = whiten_bundle(b, ifo)
            pk = locate_peak(seg, search_half_width_ms=250.0)
            ptf = peak_to_floor(seg, pk)
            if ptf < PTF_THRESHOLD:
                continue
            core = extract_core(seg, pk, sign_align=True)
            truth_raw = unwhiten(core, seg, pk)
            mid = seg.n // 2
            truth_core_raw = truth_raw[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]
            trusted.append({"ifo": ifo, "bundle": b, "core_wh": core,
                            "truth_core_raw": truth_core_raw, "ptf": ptf})
    print(f"trusted cores (peak/floor>={PTF_THRESHOLD}, wide band, both detectors): {len(trusted)}")

    fresh_noise = {
        "L1": sorted(Path("../blip_denoiser_gate/noise_pool").glob("n*/analysis_bundle_*.hdf5")),
    }
    if not fresh_noise["L1"]:
        print("No fresh L1 noise pool found.")
        return

    snr_grid = [10.0, 20.0, 40.0, 80.0]
    k_grid = [5, 10, 20, 30, 50, 70]
    out = Path("../blip_denoiser_gate/realpop_wide")
    out.mkdir(parents=True, exist_ok=True)

    # only L1 truths have a matching noise pool right now. Keep each item's
    # index into the GLOBAL `trusted` list (not list.index(), which would
    # compare dicts containing numpy arrays and raise on the ambiguous
    # truth-value of an array equality).
    l1_trusted = [(gi, t) for gi, t in enumerate(trusted) if t["ifo"] == "L1"]
    print(f"scoring {len(l1_trusted)} L1 trusted cores against the L1 noise pool")

    results = {k: {s: [] for s in snr_grid} for k in k_grid}
    for i, (global_idx, t) in enumerate(l1_trusted):
        noise_bundle = fresh_noise["L1"][i % len(fresh_noise["L1"])]
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

        for target_snr in snr_grid:
            scale = target_snr / current_norm
            raw_scaled = template_padded * scale
            dest = out / f"inj_{i:03d}_snr{int(target_snr)}.hdf5"
            inject_into_bundle(noise_bundle, raw_scaled, dest)
            seg = whiten_bundle(dest, "L1")
            pk = locate_peak(seg, search_half_width_ms=60.0)
            core = extract_core(seg, pk, sign_align=False)

            others = np.array([o["core_wh"] for j, o in enumerate(trusted) if j != global_idx])
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

    print(f"\n{'k':>4} " + " ".join(f"snr={s:<5g}" for s in snr_grid))
    summary = {}
    for k in k_grid:
        summary[str(k)] = {}
        cells = []
        for s in snr_grid:
            vals = np.array(results[k][s])
            if len(vals) == 0:
                cells.append("     n/a")
                continue
            summary[str(k)][str(s)] = {"median": float(np.median(vals)), "n": len(vals)}
            cells.append(f"{np.median(vals):9.3f}")
        print(f"{k:4d} " + " ".join(cells))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
