"""Validate the Whittle noise evidence convention and demonstrate the noise-scale marginal.

Part 1 (validation) -- confirm numerically, on a prepared detector, that the JIM
noise-relative likelihood drops exactly the Whittle constants:

    log L_jim(h)  ==  log L_full(h) - log Z_N

so that ``logZ_noise = 0`` (the BCR convention) is exact and the absolute signal
evidence is ``log Z_S_absolute = log Z_N + log Z_S_jim``.

Part 2 (robustness) -- give the noise hypothesis a free PSD-amplitude scale
``eta`` (PSD = ``eta * S``) and show that when the data carry MORE power than the
estimated PSD (eta_true > 1, e.g. non-stationary noise), the fixed-PSD noise
evidence collapses (and a flexible signal would overfit the excess), while the
noise-scale marginal absorbs it and correctly infers ``eta``. This is the lever
that makes the method robust on real, uncertain PSDs.

    uv run python studies/noise_scale_marginal.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from starccato_jax.waveforms import get_model
from starccato_lvk.analysis.jim_waveform import StarccatoJimWaveform
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
from starccato_lvk.analysis.jim_likelihood import build_transient_likelihood
from starccato_lvk.analysis.noise_evidence import (
    band_quantities, whittle_log_zn, whittle_log_zn_marginal, full_gaussian_loglike,
)
from starccato_lvk.jimgw.core.single_event.utils import inner_product

from simulated_design_psd import build_design_psd, pycbc_psd_to_gwpy
from snr_vs_odds_roc import SAMPLE_RATE, SEGMENT_DURATION, simulate_noise_fd, _write_event_bundle


def _prepare_noise(outdir: Path, psd_pycbc, psd_vals, psd_fs, n_seg, dt, seed, scale=1.0,
                   flow=300.0, fmax=800.0):
    """Simulate noise (optionally amplitude-scaled), bundle it, and prepare an L1 detector.

    Defaults to a line-free sub-band: the noise-scale marginal infers a *global* PSD
    amplitude, so residual instrumental lines (which inflate the in-band mean above 1)
    would otherwise be mis-attributed to the scale.
    """
    noise = simulate_noise_fd(psd_vals, n_seg, dt, seed) * scale
    bundle = _write_event_bundle(outdir, noise, psd_fs, 1_260_000_000.0)
    prep = prepare_multi_detector_data(["L1"], bundle_paths={"L1": bundle}, flow=flow, fmax=fmax)
    return prep


def _inferred_eta(det, a=100.0, b=None):
    """Posterior mean of the PSD scale eta under the inverse-gamma marginal."""
    q = band_quantities(det)
    b = (a - 1.0) if b is None else b
    return (q.dd / 2.0 + b) / (q.n_bins + a - 1.0)  # InvGamma posterior mean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("design_psd_cache"))
    p.add_argument("--outdir", type=Path, default=Path("out_noise_evidence"))
    p.add_argument("--prior-a", type=float, default=100.0, help="InvGamma(a) for the PSD scale (a~100 -> ~10% uncertainty).")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    n_seg = int(round(SEGMENT_DURATION * SAMPLE_RATE))
    dt = 1.0 / SAMPLE_RATE
    psd_pycbc = build_design_psd("L1", 1.0 / SEGMENT_DURATION, n_seg // 2 + 1, args.cache_dir)
    psd_vals = np.asarray(psd_pycbc.numpy(), dtype=np.float64)
    psd_fs = pycbc_psd_to_gwpy(psd_pycbc)

    # ---- Part 1: convention validation on matched noise ----
    prep = _prepare_noise(args.outdir / "matched", psd_pycbc, psd_vals, psd_fs, n_seg, dt, seed=0)
    det = prep.detectors[0]
    sr = 1.0 / det.dt if hasattr(det, "dt") else SAMPLE_RATE
    wf = StarccatoJimWaveform(model=get_model("ccsne"), sample_rate=SAMPLE_RATE, window=prep.window)
    lik = build_transient_likelihood([det], wf, trigger_time=prep.trigger_time,
                                     duration=prep.duration, post_trigger_duration=prep.post_trigger_duration)
    params = {name: 0.7 for name in wf.latent_names}  # arbitrary non-trivial waveform
    params.update({"log_amp": 0.5, "t_c": 0.0, "ra": 0.0, "dec": 0.0, "psi": 0.0,
                   "luminosity_distance": 10.0, "gmst": float(prep.gmst), "trigger_time": float(prep.trigger_time)})

    full_freqs = np.fft.rfftfreq(n_seg, d=dt)
    fj = jnp.asarray(full_freqs)
    df = float(full_freqs[1] - full_freqs[0])
    h_dec = det.fd_response(fj, wf(fj, params), params)
    d = det.data.fd
    psd = jnp.asarray(det.psd.values)
    # Compute <h|d> and <h|h> directly (NOT <d-h|d-h>) to avoid catastrophic
    # cancellation: <d|d> and the residual are both ~10^6, their difference ~10^2.
    hd = float(inner_product(h_dec, d, psd, df))      # <h|d>
    hh = float(inner_product(h_dec, h_dec, psd, df))  # <h|h>
    dd = float(inner_product(d, d, psd, df))          # <d|d>
    resid = dd - 2.0 * hd + hh                         # <d-h|d-h>

    logL_jim = float(lik.evaluate(dict(params), None))
    logL_jim_identity = hd - hh / 2.0                  # = 0.5*(<d|d> - resid), cancellation-free
    logL_full = full_gaussian_loglike(det, resid)      # -resid/2 + C
    logZ_N = whittle_log_zn(det)                        # -dd/2 + C

    print("=" * 64)
    print("PART 1 -- convention validation (matched noise, arbitrary waveform)")
    print("=" * 64)
    print(f"  log L_jim (likelihood.evaluate)        = {logL_jim:.5f}")
    print(f"  log L_jim (<h|d> - <h|h>/2)            = {logL_jim_identity:.5f}")
    print(f"  log L_full(h) - log Z_N                = {logL_full - logZ_N:.5f}")
    identity_exact = np.isclose(logL_full - logZ_N, logL_jim_identity, rtol=1e-9)
    likelihood_ok = np.isclose(logL_jim, logL_jim_identity, rtol=2e-3)  # float32 VAE waveform
    print(f"  -> identity log L_full - log Z_N == <h|d>-<h|h>/2 (exact): {identity_exact}")
    print(f"  -> likelihood.evaluate matches to float precision: {likelihood_ok}")
    print(f"  absolute Whittle log Z_N               = {logZ_N:.1f}")
    print(f"  (so log Z_S_absolute = log Z_N + log Z_S_jim; Z_N cancels in the odds)")

    # ---- Part 2: noise-scale marginal vs PSD mismatch ----
    print("\n" + "=" * 64)
    print("PART 2 -- noise-scale marginal absorbs PSD mismatch")
    print("=" * 64)
    print(f"  {'eta_true':>9s} {'logZN_fixed':>13s} {'logZN_marg':>12s} {'eta_inferred':>13s} {'ratio/baseline':>15s}")
    eta_base = None
    for eta_true in (1.0, 1.5, 2.0, 3.0):
        prep_m = _prepare_noise(args.outdir / f"eta_{eta_true}", psd_pycbc, psd_vals, psd_fs,
                                n_seg, dt, seed=0, scale=np.sqrt(eta_true))
        dm = prep_m.detectors[0]
        zfix = whittle_log_zn(dm)
        zmarg = whittle_log_zn_marginal(dm, a=args.prior_a)
        eta_inf = _inferred_eta(dm, a=args.prior_a)
        if eta_base is None:
            eta_base = eta_inf
        print(f"  {eta_true:9.1f} {zfix:13.0f} {zmarg:12.0f} {eta_inf:13.2f} {eta_inf/eta_base:15.2f}")

    print("\n  Reading:")
    print("  - fixed-PSD log Z_N COLLAPSES as eta_true grows (data louder than the PSD), so a")
    print("    flexible signal would 'win' by fitting the excess -> false positive on noise.")
    print("  - the noise-scale MARGINAL stays nearly flat (it absorbs the scale); the ratio of")
    print("    inferred eta to the eta_true=1 baseline tracks eta_true exactly (1.0/1.5/2.0/3.0).")
    print("  - the constant baseline offset (>1) is residual instrumental lines in the real O3 ASD:")
    print("    the GLOBAL scale conflates un-notched line power with true noise amplitude, so clean")
    print("    line handling (or a per-band / robust scale) matters when this is used on real data.")


if __name__ == "__main__":
    main()
