import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import nfft
import jax


def _sample_prior_waveforms(
    model,
    window,
    n_samples: int,
    rng_key: Any,
    log_amp_mu: float = -6.0,
    log_amp_sigma: float = 2.0,
) -> np.ndarray:
    """Draw prior waveform samples in physical strain units."""
    latent_dim = model.latent_dim
    keys = jax.random.split(rng_key, n_samples)
    theta = np.asarray(
        jax.random.normal(keys[0], shape=(n_samples, latent_dim))
    )
    log_amps = np.asarray(
        jax.random.normal(keys[1], shape=(n_samples,)) * log_amp_sigma + log_amp_mu
    )

    samples = []
    base_key = keys[2] if len(keys) > 2 else rng_key
    for i in range(n_samples):
        gen = model.generate(
            z=np.asarray(theta[i][None, :]),
            rng=base_key,
        )[0]
        waveform = np.asarray(gen) * np.exp(log_amps[i])
        waveform = waveform * np.asarray(window)
        samples.append(waveform)
    return np.stack(samples)


def plot_prior_samples(
    lvk_data,
    model,
    outdir: str,
    rng_key: Any,
    n_samples: int = 16,
    log_amp_mu: float = -6.0,
    log_amp_sigma: float = 2.0,
):
    """Plot prior samples in time and frequency domain alongside data."""
    os.makedirs(outdir, exist_ok=True)
    rescaled = lvk_data.rescaled_data
    strain_scale = float(rescaled["strain_scale"])
    psd_scale = float(rescaled["psd_scale"])
    sampling_freq = float(rescaled["sampling_frequency"])
    time_axis = rescaled.get("time_array") if "time_array" in rescaled else np.arange(len(rescaled["strain_td"])) / sampling_freq
    df = float(rescaled["df"])

    data_td = np.asarray(rescaled["strain_td"]) / strain_scale
    data_psd = np.asarray(rescaled["psd_array"]) / psd_scale
    freq_axis = np.asarray(rescaled["frequency_array"])
    positive_freq_mask = freq_axis > 0.0
    freq_axis = freq_axis[positive_freq_mask]
    data_psd = data_psd[positive_freq_mask]

    prior_samples = _sample_prior_waveforms(
        model=model,
        window=lvk_data.window,
        n_samples=n_samples,
        rng_key=rng_key,
        log_amp_mu=log_amp_mu,
        log_amp_sigma=log_amp_sigma,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    axes[0].plot(time_axis, data_td, color="black", label="Data", linewidth=1.2)
    for i, waveform in enumerate(prior_samples):
        label = "Prior sample" if i == 0 else None
        axes[0].plot(time_axis, waveform, alpha=0.25, color="tab:blue", label=label)
    axes[0].set_title("Time-domain prior samples vs data")
    axes[0].set_xlabel("Time [s] relative to trigger")
    axes[0].set_ylabel("Strain")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(frameon=False, loc="upper right")
    # center x-axis around t0 (middle of the time series)
    t0 = time_axis[len(time_axis) // 2]
    axes[0].set_xlim(t0 - 0.5, t0 + 0.5)

    # Frequency domain data
    data_fd, data_freq_axis = nfft(data_td, sampling_freq)
    data_freq_axis = np.asarray(data_freq_axis)
    positive_data_freq = data_freq_axis > 0.0

    axes[1].loglog(freq_axis, data_psd, color="black", label="PSD", linewidth=1.2)
    axes[1].loglog(
        data_freq_axis[positive_data_freq],
        (np.abs(data_fd[positive_data_freq]) ** 2) * 2.0 * df,
        color="gray",
        alpha=0.5,
        label="Data PSD from TD",
    )
    for i, waveform in enumerate(prior_samples):
        wf_fd, wf_freq = nfft(waveform, sampling_freq)
        wf_freq = np.asarray(wf_freq)
        positive_wf_freq = wf_freq > 0.0
        wf_psd = (np.abs(wf_fd) ** 2) * 2.0 * df
        label = "Prior sample" if i == 0 else None
        axes[1].loglog(
            wf_freq[positive_wf_freq],
            wf_psd[positive_wf_freq],
            alpha=0.25,
            color="tab:orange",
            label=label,
        )

    if freq_axis.size > 1:
        axes[1].set_xlim(freq_axis[0], freq_axis[-1])
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("PSD [$1/\mathrm{Hz}$]")
    axes[1].set_title("PSD of prior samples vs data PSD")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle(
        f"Prior diagnostics ({model.model_name}) | samples={n_samples}, "
        f"log_amp~N({log_amp_mu}, {log_amp_sigma}^2)"
    )

    fname = os.path.join(outdir, f"prior_diagnostics_{model.model_name}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
