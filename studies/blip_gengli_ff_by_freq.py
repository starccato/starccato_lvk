"""Phase 0b: fitting factor of gengli (and CCSN, as reference) vs peak frequency.

The earlier fitting-factor study sampled the loudest-N cataloged blips, whose
median peak frequency (630 Hz) sits almost entirely outside gengli's own
training range ([30,250] Hz, from L1 O2 blips). This script uses the
stratified sample from blip_stratified_sample.py to test gengli where it was
actually built to work, and reports FF(gengli), FF(ccsne) as a function of
GravitySpy peak_frequency.

Decides: is gengli valid to mix into a rebuilt blip training set (and if so,
at what population weight)?

    ./.venv/bin/python studies/blip_gengli_ff_by_freq.py --ifo L1 --run O3b \
        --stratified-dir ../blip_stratified/L1_O3b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Reuse the ORIGINAL fitting-factor function, WITH its +-25ms time-shift
# search. An earlier version of this script dropped the shift search on the
# reasoning that real catalog blips are already at the "right" time -- that
# reasoning was wrong: GravitySpy event_time is not guaranteed sub-sample
# exact, and normalized correlation collapses under a few-ms misalignment
# (measured directly elsewhere in this investigation: 0.93 -> 0.008 for a
# 2-sample offset). blip_fitting_factors._fitting_factor is validated and
# unchanged; only the target's whitening band differs (see below).
from blip_fitting_factors import _fitting_factor
from blip_stratified_sample import load_catalog, FREQ_BINS
from blip_core_extract import EXTRACT_FLOW, EXTRACT_FMAX, WIDE_BAND_ROLL_OFF
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
import jax.numpy as jnp


def _whitened_target(bundle, detector: str):
    """Like blip_fitting_factors._whitened_target, but in the WIDE band.

    The production 300-800 Hz analysis band structurally excludes most of
    gengli's population by construction (measured earlier: median in-band
    power fraction ~0 for peak_frequency < 300 Hz). Scoring gengli's own
    validity in that band would show near-zero FF for BOTH decoders on those
    events -- not because gengli fits badly, but because there is nothing
    there for anything to fit (verified directly: a peak_frequency=176 Hz
    blip has a 300-800 Hz whitened peak of 0.03 sigma vs 0.40 sigma in
    30-1024 Hz). This asks the population-validity question in the band
    where the signal actually lives.
    """
    d = prepare_multi_detector_data(
        (detector,), bundle_paths={detector: bundle}, flow=EXTRACT_FLOW, fmax=EXTRACT_FMAX,
        roll_off=WIDE_BAND_ROLL_OFF,
    ).detector_data[detector]
    psd = np.asarray(d.psd_likelihood)
    x = np.asarray(d.windowed_strain)
    n = x.size
    dt = float(d.dt)
    df = 1.0 / (n * dt)
    w = jnp.asarray(np.where(np.isfinite(psd) & (psd > 0), psd, np.inf))

    def whiten(v):
        return jnp.fft.irfft(jnp.fft.rfft(v) * dt / jnp.sqrt(w / (4.0 * df)), n=n)

    target = whiten(jnp.asarray(x))
    target = target / jnp.linalg.norm(target)
    return whiten, target, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ifo", required=True, choices=["H1", "L1"])
    p.add_argument("--run", required=True, choices=["O2", "O3a", "O3b"])
    p.add_argument("--stratified-dir", type=Path, required=True)
    args = p.parse_args()

    catalog = load_catalog(args.ifo, args.run)
    bundles = sorted(args.stratified_dir.glob(f"e*/{args.ifo}/analysis_bundle_*.hdf5"))
    print(f"scoring {len(bundles)} fetched blips from {args.stratified_dir}")

    rows = []
    for b in bundles:
        row_idx = int(b.parent.parent.name[1:])
        cat_row = catalog.iloc[row_idx]
        try:
            whiten, target, n = _whitened_target(b, args.ifo)
        except Exception as exc:
            print(f"  row{row_idx}: prep failed ({type(exc).__name__})")
            continue
        ff_ccsne = _fitting_factor("ccsne", whiten, target, n, seed=row_idx)
        ff_blip = _fitting_factor("blip", whiten, target, n, seed=row_idx)
        rows.append({
            "row": row_idx, "peak_frequency": float(cat_row.peak_frequency),
            "snr": float(cat_row.snr), "ff_ccsne": ff_ccsne, "ff_blip": ff_blip,
        })
        print(f"  row{row_idx:5d} peak_f={cat_row.peak_frequency:6.1f} Hz "
              f"snr={cat_row.snr:6.1f}  ff_ccsne={ff_ccsne:.3f}  ff_blip={ff_blip:.3f}")

    print(f"\n{'peak_f bin [Hz]':>18} {'n':>4} {'ff_ccsne':>9} {'ff_blip':>8} "
          f"{'blip wins':>10}")
    for lo, hi in FREQ_BINS:
        m = [r for r in rows if lo <= r["peak_frequency"] < hi]
        if not m:
            continue
        cc = np.array([r["ff_ccsne"] for r in m])
        bb = np.array([r["ff_blip"] for r in m])
        print(f"      [{lo:4d},{hi:4d})  {len(m):4d} {np.median(cc):9.3f} "
              f"{np.median(bb):8.3f} {int((bb > cc).sum()):6d}/{len(m):<3d}")

    # gengli's own training range. The relevant question is RELATIVE: does
    # gengli beat ccsne where it should, not whether it clears some absolute
    # FF value. Absolute FF is population-dependent -- a properly stratified,
    # SNR-diverse sample scores much lower (~0.15-0.3) than a loud-only
    # sample (~0.5-0.9) for BOTH decoders, so an absolute threshold like 0.85
    # is miscalibrated and was retracted after the first run on real data.
    m = [r for r in rows if r["peak_frequency"] < 250]
    if m:
        cc = np.array([r["ff_ccsne"] for r in m]); bb = np.array([r["ff_blip"] for r in m])
        win_rate = float((bb > cc).mean())
        print(f"\ngengli's own range (peak_f<250 Hz): n={len(m)}, "
              f"median ff_ccsne={np.median(cc):.3f}, median ff_blip={np.median(bb):.3f}, "
              f"blip wins {int((bb>cc).sum())}/{len(m)} ({100*win_rate:.0f}%)")
        verdict = ("gengli beats ccsne in its own range -- valid to mix, weighted by "
                   "population share" if win_rate > 0.5 and np.median(bb) > np.median(cc) else
                   "gengli does not beat ccsne even in its own range -- drop it, do not mix")
        print(verdict)


if __name__ == "__main__":
    main()
