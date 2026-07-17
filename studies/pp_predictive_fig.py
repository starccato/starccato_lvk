"""Manuscript posterior-predictive figure from one P-P injection.

Re-runs a single pp_test.py injection (same seeds, same production NUTS
configuration) and plots the data, the injected waveform, and the signal-VAE
posterior predictive band in the time and frequency domains, with the noise
ASD and the 300-800 Hz analysis band marked.

    uv run python studies/pp_predictive_fig.py --index 0 --outdir out_pp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starccato_lvk.analysis.jim_likelihood import build_transient_likelihood, run_numpyro_sampling
from starccato_lvk.analysis.jim_waveform import StarccatoJimWaveform
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
from starccato_jax.waveforms import get_model

from snr_vs_odds_roc import SAMPLE_RATE, BASE_TRIGGER

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
})

DATA_COL = "#9a9a9a"
PSD_COL = "black"
INJ_COL = "#e08214"
POST_COL = "#1b7837"


def _whiten(x: np.ndarray, psd: np.ndarray, dt: float) -> np.ndarray:
    """Whitened time series (dimensionless); infinite out-of-band PSD acts as a bandpass."""
    n = x.size
    df = 1.0 / (n * dt)
    w = np.where(np.isfinite(psd) & (psd > 0), psd, np.inf)
    return np.fft.irfft(np.fft.rfft(x) * dt / np.sqrt(w / (4.0 * df)), n=n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=Path("out_pp"))
    p.add_argument("--flow", type=float, default=300.0)
    p.add_argument("--fmax", type=float, default=800.0)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-chains", type=int, default=4)
    p.add_argument("--n-draws", type=int, default=500)
    args = p.parse_args()

    truth = json.loads((args.outdir / "results" / f"inj_{args.index}.json").read_text())["truth"]
    trigger = int(BASE_TRIGGER + args.index)
    bundle = args.outdir / f"inj_{args.index}" / "data" / f"analysis_bundle_{trigger}.hdf5"

    prep = prepare_multi_detector_data(["L1"], bundle_paths={"L1": str(bundle)},
                                       flow=args.flow, fmax=args.fmax)
    det = prep.detectors[0]
    data_info = prep.detector_data[det.name]
    dt = float(data_info.dt)
    n_seg = data_info.windowed_strain.size
    freqs = np.fft.rfftfreq(n_seg, d=dt)
    fj = jnp.asarray(freqs)
    df = freqs[1] - freqs[0]

    wf = StarccatoJimWaveform(model=get_model("ccsne"), sample_rate=SAMPLE_RATE, window=prep.window)
    latent_names = wf.latent_names
    resp = {"t_c": 0.0, "ra": 0.0, "dec": 0.0, "psi": 0.0, "luminosity_distance": 10.0,
            "gmst": float(prep.gmst), "trigger_time": float(prep.trigger_time)}

    def _htd(params: dict) -> np.ndarray:
        full = {**resp, **params}
        return np.fft.irfft(np.asarray(det.fd_response(fj, wf(fj, full), full)), n=n_seg) / dt

    inj_td = _htd(truth)

    likelihood = build_transient_likelihood(
        prep.detectors, wf, trigger_time=prep.trigger_time,
        duration=prep.duration, post_trigger_duration=prep.post_trigger_duration,
    )
    result = run_numpyro_sampling(
        likelihood, latent_names=latent_names,
        fixed_params={k: resp[k] for k in ("t_c", "ra", "dec", "psi", "luminosity_distance",
                                           "gmst", "trigger_time")},
        rng_key=jax.random.PRNGKey(args.index), latent_sigma=1.0, log_amp_sigma=1.0,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=args.num_chains, progress_bar=False,
    )
    samples = {k: np.asarray(v) for k, v in result.samples.items()}
    n_post = samples[latent_names[0]].size
    idx = np.random.default_rng(0).choice(n_post, size=min(args.n_draws, n_post), replace=False)
    draws_td = np.array([_htd({k: samples[k][i] for k in (*latent_names, "log_amp")}) for i in idx])

    data_td = np.asarray(data_info.windowed_strain)
    t = np.asarray(data_info.time)  # already relative to the trigger
    psd = np.asarray(det.psd.values)

    # frequency-domain amplitude spectra, |x(f)| dt sqrt(2/df dt) ~ sqrt(PSD) units
    def _asd(x: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(x) * dt) * np.sqrt(2 * df)

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # time domain: whitened (in-band PSD only, so whitening also band-limits)
    psd_band = np.where((freqs >= args.flow) & (freqs <= args.fmax), psd, np.inf)
    sigma = np.std(_whiten(data_td, psd_band, dt))  # express whitened series in noise-sigma units
    wh = lambda x: _whiten(x, psd_band, dt) / sigma
    lo, med, hi = np.percentile(np.array([wh(d) for d in draws_td]), [5, 50, 95], axis=0)
    ax_t.plot(t, wh(data_td), color=DATA_COL, lw=0.8, label="whitened data")
    ax_t.fill_between(t, lo, hi, color=POST_COL, alpha=0.3, lw=0, label="posterior 90\\% CI")
    ax_t.plot(t, med, color=POST_COL, lw=1.2, label="posterior median")
    ax_t.plot(t, wh(inj_td), color=INJ_COL, lw=1.2, ls="--", label="injection")
    snr = np.sqrt(4 * df * np.nansum(np.abs(np.fft.rfft(inj_td) * dt)[np.isfinite(psd_band)] ** 2
                                     / psd_band[np.isfinite(psd_band)]))
    ax_t.text(0.02, 0.02, f"SNR = {snr:.0f}", transform=ax_t.transAxes, fontsize=8)
    ax_t.set_xlim(-0.04, 0.04)
    ax_t.set_xlabel("time [s] relative to trigger")
    ax_t.set_ylabel(r"whitened strain [$\sigma$]")
    ax_t.legend(frameon=False, loc="upper left")

    # frequency domain
    band = (freqs >= 100) & (freqs <= 1024)
    lo_f, med_f, hi_f = np.percentile(np.array([_asd(d) for d in draws_td]), [5, 50, 95], axis=0)
    ax_f.loglog(freqs[band], _asd(data_td)[band], color=DATA_COL, lw=0.7, alpha=0.8, label="data")
    ax_f.loglog(freqs[band], np.sqrt(psd[band]), color=PSD_COL, lw=1.0, label="noise ASD")
    ax_f.fill_between(freqs[band], lo_f[band], hi_f[band], color=POST_COL, alpha=0.3, lw=0)
    ax_f.loglog(freqs[band], med_f[band], color=POST_COL, lw=1.2, label="posterior median")
    ax_f.loglog(freqs[band], _asd(inj_td)[band], color=INJ_COL, lw=1.2, ls="--", label="injection")
    ax_f.axvspan(args.flow, args.fmax, color="k", alpha=0.06, lw=0, label="analysis band")
    ax_f.set_xlabel("frequency [Hz]")
    ax_f.set_ylabel(r"ASD [$1/\sqrt{\rm Hz}$]")
    ax_f.legend(frameon=False, loc="lower left", fontsize=7)

    fig.tight_layout()
    out = args.outdir / "posterior_predictive.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
