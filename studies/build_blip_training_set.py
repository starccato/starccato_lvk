"""Phase 1 step 1: build the blip-v2 training set from real O3a cores.

Extracts in the 300-800 Hz analysis band (NOT the wide band -- see
docs/blip_prior_rebuild_plan.md, "RESOLVED 2026-08-06": a 512-sample core
cannot hold a 30 Hz high-pass's impulse response, so wide-band extraction was
abandoned). No gengli mixing: gengli's in-band power fraction is 0.0001,
contributing nothing where the likelihood looks.

Pipeline per event: whiten with its own off-source PSD -> locate peak ->
quality cut (in-band peak/floor >= 10) -> extract a 512-sample core,
sign-aligned -> PCA-denoise (basis fit on the retained population, k=8,
matches the flat-in-k plateau measured in Phase 0c) -> un-whiten back to raw
strain (the physically meaningful template) -> augment with the negated copy
(the VAE amplitude is exp(log_amp) > 0 and cannot flip polarity, so the
decoder must see both signs) -> write an .npz TrainValData.load() can read
directly (source=<path to this file>.npz).

Also saves three diagnostic figures per a handful of example events, so the
pipeline's effect is visible rather than asserted:
  - fig_stage_raw_vs_denoised_<i>.pdf: whitened core before/after PCA denoise
  - fig_stage_unwhitened_<i>.pdf: final raw-strain training waveform
  - fig_pca_spectrum.pdf: variance explained vs component, population-wide

    ./.venv/bin/python studies/build_blip_training_set.py \
        --l1-dir ../blip_train_o3a/L1/L1_O3a --h1-dir ../blip_train_o3a/H1/H1_O3a \
        --out ../blip_train_o3a/blip_o3a_real.npz --plots ../blip_train_o3a/plots
"""

from __future__ import annotations

import argparse
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

NARROW_FLOW, NARROW_FMAX = 300.0, 800.0
PTF_THRESHOLD = 10.0
PCA_K = 8  # flat-in-k plateau measured in Phase 0c (0.894 at k=3, 0.907 at k=15)


def collect_cores(stratified_dir: Path, ifo: str, run: str) -> list[dict]:
    """Whitened + raw cores for every bundle passing the quality cut.

    GravitySpy carries duplicate rows for the same physical glitch (repeated
    pipeline triggers / re-classification passes on one event) -- confirmed
    directly on H1 O3a: 8 distinct events appear as 12 extra rows, one
    triplicated. Deduplicating by GPS (encoded in the bundle filename) before
    the quality cut keeps one copy per physical event; without it those
    events would be weighted 2-3x in the training set.
    """
    catalog = load_catalog(ifo, run)
    out = []
    seen_gps: set[str] = set()
    n_dropped_dup = 0
    for b in sorted(stratified_dir.glob(f"e*/{ifo}/analysis_bundle_*.hdf5")):
        gps_tag = b.stem.rsplit("_", 1)[-1]
        if gps_tag in seen_gps:
            n_dropped_dup += 1
            continue
        seen_gps.add(gps_tag)
        row = int(b.parent.parent.name[1:])
        seg = whiten_bundle(b, ifo, flow=NARROW_FLOW, fmax=NARROW_FMAX)
        pk = locate_peak(seg, search_half_width_ms=250.0)
        ptf = peak_to_floor(seg, pk)
        if ptf < PTF_THRESHOLD:
            continue
        core_wh = extract_core(seg, pk, sign_align=True)
        out.append({
            "ifo": ifo, "row": row, "ptf": ptf,
            "peak_frequency": float(catalog.iloc[row].peak_frequency),
            "core_wh": core_wh, "seg": seg, "peak_idx": pk,
        })
    if n_dropped_dup:
        print(f"  {ifo}: dropped {n_dropped_dup} duplicate-GPS catalog rows")
    return out


def make_plots(cores: list[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    all_wh = np.array([c["core_wh"] for c in cores])
    mean = all_wh.mean(axis=0)
    _, _, vt = np.linalg.svd(all_wh - mean, full_matrices=False)

    # --- PCA variance-explained spectrum -----------------------------------
    _, s, _ = np.linalg.svd(all_wh - mean, full_matrices=False)
    var = s**2 / np.sum(s**2)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(var) + 1)[:30], np.cumsum(var)[:30], "o-", ms=3)
    ax.axvline(PCA_K, color="r", ls="--", lw=1, label=f"k={PCA_K} (used)")
    ax.set_xlabel("number of PCA components")
    ax.set_ylabel("cumulative variance explained")
    ax.set_title(f"Blip core PCA spectrum (n={len(cores)} real O3a cores)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig_pca_spectrum.pdf")
    plt.close(fig)

    # --- per-event stages: raw whitened core vs PCA-denoised, and the
    # final un-whitened (raw strain) training waveform ----------------------
    rng = np.random.default_rng(0)
    example_idx = rng.choice(len(cores), size=min(6, len(cores)), replace=False)
    t_ms = (np.arange(2 * CORE_HALF_WIDTH) - CORE_HALF_WIDTH) / 4096.0 * 1e3

    fig, axes = plt.subplots(len(example_idx), 2, figsize=(8, 2.2 * len(example_idx)))
    if len(example_idx) == 1:
        axes = axes[None, :]
    for row_i, idx in enumerate(example_idx):
        c = cores[idx]
        raw = c["core_wh"]
        denoised = mean + (raw - mean) @ vt[:PCA_K].T @ vt[:PCA_K]
        unwh = unwhiten(denoised, c["seg"], c["peak_idx"])
        mid = c["seg"].n // 2
        unwh_core = unwh[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]

        ax = axes[row_i, 0]
        ax.plot(t_ms, raw, color="0.5", lw=0.8, label="whitened (raw)")
        ax.plot(t_ms, denoised, color="#0072B2", lw=1.2, label=f"PCA-denoised (k={PCA_K})")
        ax.set_title(f"{c['ifo']} e{c['row']} peak_f={c['peak_frequency']:.0f}Hz ptf={c['ptf']:.1f}",
                    fontsize=8)
        if row_i == 0:
            ax.legend(fontsize=6, loc="upper right")
        ax.set_ylabel("whitened [$\\sigma$]", fontsize=7)

        ax = axes[row_i, 1]
        norm = unwh_core / np.max(np.abs(unwh_core))
        ax.plot(t_ms, norm, color="#D55E00", lw=1.0)
        ax.set_title("final training waveform (peak-normalized)", fontsize=8)
        ax.set_ylabel("strain (norm.)", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [ms]")
    fig.tight_layout()
    fig.savefig(outdir / "fig_stages_examples.pdf")
    plt.close(fig)

    # --- population overlay: all denoised cores, peak-normalized ----------
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for c in cores:
        denoised = mean + (c["core_wh"] - mean) @ vt[:PCA_K].T @ vt[:PCA_K]
        norm = denoised / np.max(np.abs(denoised))
        ax.plot(t_ms, norm, color="0.3", lw=0.3, alpha=0.15)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("whitened, peak-normalized")
    ax.set_title(f"All {len(cores)} denoised cores overlaid (300-800 Hz)")
    fig.tight_layout()
    fig.savefig(outdir / "fig_population_overlay.pdf")
    plt.close(fig)

    print(f"wrote plots to {outdir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l1-dir", type=Path, required=True)
    p.add_argument("--h1-dir", type=Path, required=True)
    p.add_argument("--run", default="O3a")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plots", type=Path, default=None)
    args = p.parse_args()

    cores = (
        collect_cores(args.l1_dir, "L1", args.run)
        + collect_cores(args.h1_dir, "H1", args.run)
    )
    print(f"collected {len(cores)} cores passing peak/floor>={PTF_THRESHOLD} "
          f"({sum(1 for c in cores if c['ifo']=='L1')} L1, "
          f"{sum(1 for c in cores if c['ifo']=='H1')} H1)")
    if len(cores) < PCA_K + 2:
        raise SystemExit(f"only {len(cores)} usable cores; need at least {PCA_K + 2}")

    if args.plots:
        make_plots(cores, args.plots)

    all_wh = np.array([c["core_wh"] for c in cores])
    mean = all_wh.mean(axis=0)
    _, _, vt = np.linalg.svd(all_wh - mean, full_matrices=False)

    waveforms = []
    for c in cores:
        denoised = mean + (c["core_wh"] - mean) @ vt[:PCA_K].T @ vt[:PCA_K]
        unwh = unwhiten(denoised, c["seg"], c["peak_idx"])
        mid = c["seg"].n // 2
        wf = unwh[mid - CORE_HALF_WIDTH: mid + CORE_HALF_WIDTH]
        waveforms.append(wf)
        waveforms.append(-wf)  # polarity augmentation (log_amp > 0 cannot flip sign)

    data = np.array(waveforms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, data=data)
    print(f"wrote {data.shape[0]} waveforms ({data.shape[0]//2} cores x2 polarity) -> {args.out}")
    print(f"  load with: TrainValData.load(source='{args.out}')")


if __name__ == "__main__":
    main()
