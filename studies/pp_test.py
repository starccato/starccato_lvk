"""P-P (probability-probability) calibration test for the production inference.

Replaces the old bilby-based P-P run (32-dim model): this draws injections from
the EXACT prior the production sampler uses -- z ~ N(0, I_5), log_amp ~ N(0, 1) --
injects the z=5 CCSN VAE waveform through the same detector response, likelihood
band, and NUTS configuration as ``run_bcr_posteriors``, and checks that the true
parameters fall in each posterior credible interval at the expected rate.

Noise is simulated from the L1 design PSD (the "simple noise" setting: the P-P
test validates the inference machinery, not the data conditioning). The noise
amplitude is globally rescaled so that the z=0, log_amp=0 waveform has a
reference SNR (--snr-ref), which controls how informative the typical posterior
is without touching the parameter priors (so the test stays valid).

Writes credible levels incrementally (survives interruption) and produces
pp_plot.pdf + a combined KS p-value.

    uv run python studies/pp_test.py --n-inj 300 --outdir out_pp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import kstest, combine_pvalues, norm

from starccato_lvk.analysis.jim_likelihood import build_transient_likelihood, run_numpyro_sampling
from starccato_lvk.analysis.jim_waveform import StarccatoJimWaveform
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
from starccato_jax.waveforms import get_model

from snr_vs_odds_roc import (
    SAMPLE_RATE,
    BASE_TRIGGER,
    simulate_noise_fd,
    _write_event_bundle,
)
from simulated_design_psd import build_design_psd, pycbc_psd_to_gwpy

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PSD_CACHE = REPO_ROOT / "studies" / "design_psd_cache"
PSD_CACHE = Path(os.environ.get("STARCCATO_PSD_CACHE", DEFAULT_PSD_CACHE))


def _design_psd(n_seg: int, dt: float):
    """LINE-FREE design-PSD values on the analysis rfft grid + FrequencySeries.

    Narrow spectral lines (violin modes) are capped at the local running-median
    floor. The noise is drawn per-bin from this PSD, but the analysis windows the
    segment, and window leakage from a design-PSD line contaminates hundreds of
    neighbouring bins -- far beyond the +-3-bin line notch -- leaving kept bins
    with up to ~100x the power the PSD claims (whitened band power ~23 instead
    of 1). That noise/PSD mismatch is what a P-P test detects. A line-free PSD
    sidesteps it: this test validates the inference machinery, not line handling.
    """
    from scipy.ndimage import median_filter
    from pycbc.types import FrequencySeries as PycbcFrequencySeries

    delta_f = 1.0 / (n_seg * dt)
    psd_pycbc = build_design_psd("L1", delta_f, n_seg // 2 + 1, PSD_CACHE)
    vals = np.asarray(psd_pycbc.numpy(), dtype=np.float64)
    k = 2 * int(round(4.0 / delta_f)) + 1  # ~8 Hz median window
    vals = np.minimum(vals, median_filter(vals, size=k, mode="nearest"))
    smooth = PycbcFrequencySeries(vals, delta_f=delta_f)
    return vals, pycbc_psd_to_gwpy(smooth)


def run_injection(index: int, outdir: Path, *, snr_ref: float, flow: float, fmax: float,
                  num_warmup: int, num_samples: int, num_chains: int = 4) -> dict:
    dt = 1.0 / SAMPLE_RATE
    n_seg = int(round(4.0 * SAMPLE_RATE))
    # truth seed must differ from the noise seed (5000+index): default_rng streams
    # with the same seed are identical, so truth draws would replicate noise bins.
    rng = np.random.default_rng(900_000 + index)

    psd_vals, psd_fs = _design_psd(n_seg, dt)
    noise = simulate_noise_fd(psd_vals, n_seg, dt, seed=5000 + index)
    trigger = BASE_TRIGGER + index
    edir = outdir / f"inj_{index}"

    # noise-only prep to get window/response, and the recovery waveform
    bundle0 = _write_event_bundle(edir / "noise", noise, psd_fs, trigger)
    prep = prepare_multi_detector_data(["L1"], bundle_paths={"L1": bundle0}, flow=flow, fmax=fmax)
    det = prep.detectors[0]
    wf = StarccatoJimWaveform(model=get_model("ccsne"), sample_rate=SAMPLE_RATE, window=prep.window)
    latent_names = wf.latent_names

    full_freqs = np.fft.rfftfreq(n_seg, d=dt)
    fj = jnp.asarray(full_freqs)
    band = (full_freqs >= flow) & (full_freqs <= fmax)
    df = float(full_freqs[1] - full_freqs[0])
    psd_full = np.asarray(det.psd.values)
    resp = {"t_c": 0.0, "ra": 0.0, "dec": 0.0, "psi": 0.0, "luminosity_distance": 10.0,
            "gmst": float(prep.gmst), "trigger_time": float(prep.trigger_time)}

    def _hdec(params):
        return np.asarray(det.fd_response(fj, wf(fj, params), params))

    # scale the NOISE (data + PSD together) so the z=0, log_amp=0 waveform has
    # SNR = snr_ref; the parameter priors are untouched, so the P-P test stays valid.
    p0 = {name: 0.0 for name in latent_names}
    p0.update({"log_amp": 0.0, **resp})
    h0 = _hdec(p0)
    snr_unit = float(np.sqrt(4.0 * np.sum(np.abs(h0[band]) ** 2 / psd_full[band]) * df))
    scale = snr_unit / snr_ref
    psd_fs_scaled = psd_fs * scale**2

    # truth drawn from the recovery prior
    truth = {name: float(rng.standard_normal()) for name in latent_names}
    truth["log_amp"] = float(rng.standard_normal())
    p_true = dict(p0)
    p_true.update(truth)
    inj_td = np.fft.irfft(_hdec(p_true), n=n_seg) / dt

    data = noise * scale + inj_td
    bundle = _write_event_bundle(edir / "data", data, psd_fs_scaled, trigger)
    prep = prepare_multi_detector_data(["L1"], bundle_paths={"L1": bundle}, flow=flow, fmax=fmax)
    wf = StarccatoJimWaveform(model=get_model("ccsne"), sample_rate=SAMPLE_RATE, window=prep.window)
    likelihood = build_transient_likelihood(
        prep.detectors, wf, trigger_time=prep.trigger_time,
        duration=prep.duration, post_trigger_duration=prep.post_trigger_duration,
    )
    result = run_numpyro_sampling(
        likelihood, latent_names=latent_names,
        fixed_params={k: resp[k] for k in ("t_c", "ra", "dec", "psi", "luminosity_distance",
                                           "gmst", "trigger_time")},
        rng_key=jax.random.PRNGKey(index), latent_sigma=1.0, log_amp_sigma=1.0,
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        progress_bar=False,
    )

    cred = {name: float(np.mean(result.samples[name] < truth[name]))
            for name in [*latent_names, "log_amp"]}
    row = {"index": index, "truth": truth, "credible_levels": cred,
           "divergences": int(np.sum(result.extra.get("diverging", 0)))}
    (outdir / "results").mkdir(parents=True, exist_ok=True)
    (outdir / "results" / f"inj_{index}.json").write_text(json.dumps(row, indent=2))
    return row


def make_pp_plot(rows: list[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(rows[0]["credible_levels"].keys())
    n = len(rows)
    x = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for ci, alpha in zip((0.68, 0.95, 0.997), (0.28, 0.18, 0.10)):
        z = norm.ppf(0.5 + ci / 2)
        band = z * np.sqrt(x * (1 - x) / n)
        ax.fill_between(x, x - band, x + band, color="gray", alpha=alpha, lw=0,
                        label=f"{100 * ci:g}%")
    pvals = []
    for name in names:
        levels = np.sort([r["credible_levels"][name] for r in rows])
        ax.plot(levels, np.arange(1, n + 1) / n, lw=1.0, alpha=0.8)
        pvals.append(kstest(levels, "uniform").pvalue)
    _, p_combined = combine_pvalues(pvals, method="fisher")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("credible interval")
    ax.set_ylabel("fraction of injections in C.I.")
    ax.set_title(f"N={n}, combined KS p={p_combined:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "pp_plot.pdf")

    summary = {"n_injections": n,
               "ks_pvalues": dict(zip(names, map(float, pvals))),
               "combined_p": float(p_combined),
               "total_divergences": int(sum(r["divergences"] for r in rows))}
    (outdir / "pp_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-inj", type=int, default=300)
    p.add_argument("--index", type=int, default=None,
                   help="Run a single injection (for SLURM arrays); omit to loop over all.")
    p.add_argument("--outdir", type=Path, default=Path("out_pp"))
    p.add_argument("--snr-ref", type=float, default=20.0)
    p.add_argument("--flow", type=float, default=300.0)
    p.add_argument("--fmax", type=float, default=800.0)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1000)
    # the z-posterior is multimodal (VAE decoder); a single NUTS chain mode-captures
    # (credible levels rail to 0/1 seed-dependently), failing the P-P test. Pooling
    # independently-initialised chains restores (approximate) mode coverage.
    p.add_argument("--num-chains", type=int, default=4)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        indices = [args.index] if args.index is not None else range(args.n_inj)
        for i in indices:
            out = args.outdir / "results" / f"inj_{i}.json"
            if out.exists():
                continue  # idempotent, like the real-noise runner
            row = run_injection(i, args.outdir, snr_ref=args.snr_ref, flow=args.flow,
                                fmax=args.fmax, num_warmup=args.num_warmup,
                                num_samples=args.num_samples, num_chains=args.num_chains)
            print(f"[pp {i}] cred={ {k: round(v, 2) for k, v in row['credible_levels'].items()} }")

    rows = [json.loads(f.read_text()) for f in sorted((args.outdir / "results").glob("inj_*.json"))]
    if rows and args.index is None:
        make_pp_plot(rows, args.outdir)


if __name__ == "__main__":
    main()
