import os

import jax
from gwpy.timeseries import TimeSeries
# set float64 precision for JAX
jax.config.update("jax_enable_x64", True)

import arviz as az

from starccato_lvk.sampler import run_sampler,StarccatoLVKLikelihood
from starccato_lvk.likelihood import whiten
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax
from jax import random, vmap
import numpy as np
import matplotlib.pyplot as plt


def plot_posterior_predictions(
        posterior_waveforms: jnp.ndarray,
        likelihood:StarccatoLVKLikelihood,
        plot_whitened: bool = True,
        n_plot_samples: int = 100,
        alpha: float = 0.1,
        figsize: tuple = (12, 10)
):
    """
    Plot data with posterior prediction overlay.

    Args:
        data: Original strain data (512 samples)
        posterior_waveforms: Transformed posterior waveforms
        likelihood: StarccatoLVKLikelihood object (for whitening)
        plot_whitened: If True, plot whitened data and predictions
        n_plot_samples: Number of posterior samples to plot
        alpha: Transparency for posterior samples
        figsize: Figure size
    """
    n_rows = 3 if plot_whitened and likelihood is not None else 2
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Time array
    n_samps = 512
    fs = 4096.0
    t_offset = n_samps / 2 / fs
    t = np.arange(n_samps) / fs - t_offset  # time array relative to trigger
    axes[0].plot(t, likelihood.whitened_data, label='Data', color='black', lw=1.5)
    whitened_signals = np.array([whiten(wf, likelihood.psd_interp, fs) for wf in posterior_waveforms])
    # CI
    lower_bound = np.percentile(whitened_signals, 5, axis=0)
    upper_bound = np.percentile(whitened_signals, 95, axis=0)
    median_waveform = np.median(whitened_signals, axis=0)
    axes[0].fill_between(t, lower_bound, upper_bound, color='gray', alpha=0.3)
    axes[0].plot(t, median_waveform, color='gray',  lw=1)
    axes[0].set_xscale('linear')

    # freq domain
    freq_array = likelihood.freq_array
    df = likelihood.df
    freq_mask = likelihood.freq_mask
    f = freq_array[freq_mask]
    axes[1].loglog(f, likelihood.psd_interp[freq_mask], label='PSD', color='blue', lw=1.5)
    axes[1].plot(f, np.abs(likelihood.data_fft[freq_mask])**2, label='Data FFT', color='black', lw=1.5)
    signal_fft = jnp.fft.rfft(posterior_waveforms, axis=1)[:, likelihood.freq_mask]
    signal_psd = jnp.abs(signal_fft)**2
    lower_bound_psd = jnp.percentile(signal_psd, 5, axis=0)
    upper_bound_psd = jnp.percentile(signal_psd, 95, axis=0)
    median_psd = jnp.median(signal_psd, axis=0)
    axes[1].fill_between(f, lower_bound_psd, upper_bound_psd, color='gray', alpha=0.3)
    axes[1].plot(f, median_psd, color='gray', lw=1)





    return fig







def compute_snr(waveform: jnp.ndarray, likelihood) -> float:
    """
    Compute the matched-filter SNR of a waveform.

    Args:
        waveform: Time domain waveform
        likelihood: StarccatoLVKLikelihood object

    Returns:
        SNR value
    """
    # FFT waveform
    waveform_fft = jnp.fft.rfft(waveform)[likelihood.freq_mask]

    # # Compute (h|h) = 4 * Re[∫ h*(f) h(f) / S(f) df]
    # waveform_fft = waveform_fft[likelihood.freq_mask]

    snr_squared = 4 * jnp.real(
        jnp.sum(jnp.conj(waveform_fft) * waveform_fft / likelihood.psd_interp[likelihood.freq_mask]) * likelihood.df
    )

    return jnp.sqrt(snr_squared)


def summarize_posterior(posterior_samples: dict, likelihood=None, posterior_waveforms=None):
    """Print summary statistics of posterior samples."""
    print("Posterior Summary:")
    print("=" * 50)

    for param in ['distance', 'time_shift_ms']:
        if param in posterior_samples:
            samples = posterior_samples[param]
            print(f"{param}:")
            print(f"  Mean: {jnp.mean(samples):.3f}")
            print(f"  Std:  {jnp.std(samples):.3f}")
            print(f"  90% CI: [{jnp.percentile(samples, 5):.3f}, "
                  f"{jnp.percentile(samples, 95):.3f}]")
            print()

    # # Check if samples are stuck (all identical)
    # for param in ['distance', 'time_shift']:
    #     if param in posterior_samples:
    #         samples = posterior_samples[param]
    #         if jnp.std(samples) < 1e-10:
    #             print(f"⚠️  Warning: {param} samples appear stuck (std = {jnp.std(samples):.2e})")

    print(f"Number of samples: {len(posterior_samples['z'])}")
    print(f"Latent dimension: {posterior_samples['z'].shape[1]}")

    # Compute SNRs if waveforms provided
    if likelihood is not None and posterior_waveforms is not None:
        print("\nSNR Analysis:")
        print("-" * 20)

        # Compute SNR for median prediction
        median_waveform = jnp.median(posterior_waveforms, axis=0)
        median_snr = compute_snr(median_waveform, likelihood)
        print(f"Median prediction SNR: {median_snr:.2f}")

        # Compute SNR distribution
        snrs = [compute_snr(wf, likelihood) for wf in posterior_waveforms[:10]]  # Sample for speed
        print(f"SNR range (10 samples): {jnp.min(jnp.array(snrs)):.2f} - {jnp.max(jnp.array(snrs)):.2f}")


# Usage example for your specific code:
def test_mcmc(outdir, analysis_data):
    paths = analysis_data

    likelihood, mcmc = run_sampler(
        strain_file=paths["strain_file"],
        psd_file=paths["psd_file"],
        num_warmup=100  ,
        num_samples=100,
    )

    posterior_samples = mcmc.get_samples()

    # Get posterior predictions (your existing code)
    posterior_predictions = likelihood.call_model(
        posterior_samples['z'],
        random.PRNGKey(0),
        np.zeros_like(posterior_samples['z'][:, 0]),  # time_shift
        posterior_samples['distance'][:, None],
    )


    # Print comprehensive summary
    summarize_posterior(posterior_samples, likelihood, posterior_predictions)

    # Create plots
    print("Creating posterior prediction plots...")
    fig1 = plot_posterior_predictions(
        posterior_waveforms=posterior_predictions,
        likelihood=likelihood,
        plot_whitened=True,  # Include whitened comparison
        n_plot_samples=50
    )
    fig1.savefig(os.path.join(outdir, "posterior_predictions_comprehensive.png"), dpi=150)


    inference_obj = az.from_numpyro(mcmc)
    az.plot_trace(inference_obj)
    plt.savefig(os.path.join(outdir, "trace_plot.png"), dpi=150)

    return likelihood, posterior_samples, posterior_predictions