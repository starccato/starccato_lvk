from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from bilby.gw import utils as gwutils

from ..multidet_data_prep import MultiDetPreparedData
from ..jim_waveform import StarccatoJimWaveform
import jax.numpy as jnp


def plot_data_overview(prepared: MultiDetPreparedData, outpath: Path) -> None:
    """Plot time- and frequency-domain views of the prepared data."""
    detector_names = list(prepared.detector_data.keys())
    n_det = len(detector_names)
    fig, axes = plt.subplots(n_det, 2, figsize=(10, 3 * n_det), squeeze=False)

    for idx, det_name in enumerate(detector_names):
        data = prepared.detector_data[det_name]
        time_ax = axes[idx, 0]
        freq_ax = axes[idx, 1]

        time_ax.plot(data.time, data.strain, color="tab:gray", lw=0.8, label="Raw")
        time_ax.plot(data.time, data.windowed_strain, color="tab:blue", lw=0.8, alpha=0.6, label="Windowed")
        time_ax.set_title(f"{det_name} strain (time domain)")
        time_ax.set_xlabel("Time [s] relative to trigger")
        time_ax.set_ylabel("Strain")
        time_ax.grid(True, alpha=0.3)
        time_ax.legend(frameon=False)

        psd_values = np.asarray(data.psd.values)
        freqs = np.asarray(data.psd.frequencies)
        finite_mask = np.isfinite(psd_values) & (psd_values > 0)
        freqs_masked = freqs[finite_mask]
        psd_masked = psd_values[finite_mask]
        if freqs_masked.size > 0:
            freq_ax.loglog(freqs_masked, np.sqrt(psd_masked), color="tab:blue", lw=0.9, label="Noise ASD")
        data_asd_full = gwutils.asd_from_freq_series(data.data_fd_likelihood, prepared.df)
        data_asd = data_asd_full[finite_mask]
        if data_asd.size > 0:
            freq_ax.loglog(freqs, data_asd, color="tab:orange", lw=0.8, alpha=0.7, label="Data ASD")
        freq_ax.set_title(f"{det_name} frequency domain")
        freq_ax.set_xlabel("Frequency [Hz]")
        freq_ax.set_ylabel("Strain / sqrt(Hz)")
        freq_ax.grid(True, which="both", alpha=0.3)
        freq_ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_posterior_mean_waveform(
    prepared: MultiDetPreparedData,
    waveform: StarccatoJimWaveform,
    mean_params: Dict[str, float],
    outpath: Path,
) -> None:
    """Plot posterior-mean waveform against data in the time domain."""
    detector_names = list(prepared.detector_data.keys())
    n_det = len(detector_names)
    fig, axes = plt.subplots(n_det, 1, figsize=(10, 3 * n_det), squeeze=False)

    waveform_td = waveform.time_domain_waveform_numpy(mean_params)

    for idx, det_name in enumerate(detector_names):
        ax = axes[idx, 0]
        data = prepared.detector_data[det_name]
        n = min(len(data.windowed_strain), len(waveform_td))
        ax.plot(data.time[:n], data.windowed_strain[:n], color="tab:gray", lw=0.8, label="Windowed data")
        ax.plot(data.time[:n], waveform_td[:n], color="tab:green", lw=1.0, alpha=0.8, label="Posterior mean")
        ax.set_title(f"{det_name} posterior-mean waveform")
        ax.set_xlabel("Time [s] relative to trigger")
        ax.set_ylabel("Strain")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_posterior_predictive_from_samples(
    detectors,
    detector_data,
    waveform,
    latent_names: Iterable[str],
    samples: Dict[str, np.ndarray],
    fixed_params: Dict[str, float],
    outpath: Path,
    *,
    n_draws: int = 200,
    ci: tuple[int, int] = (5, 95),
    log_f: bool = True,
    title_prefix: str = "",
) -> None:
    """Plot posterior predictive bands for each detector using sampled parameters."""

    if not samples:
        return

    outpath.parent.mkdir(parents=True, exist_ok=True)

    latent_names = list(latent_names)
    sample_count = len(samples[latent_names[0]]) if latent_names else len(samples.get("log_amp", []))
    if sample_count == 0:
        return

    rng = np.random.default_rng(0)
    draw_idx = rng.choice(sample_count, size=min(n_draws, sample_count), replace=False)

    det_list = list(detectors)
    n_det = len(det_list)

    fig, axes = plt.subplots(n_det, 2, figsize=(12, 4 * n_det), squeeze=False)

    for row, det in enumerate(det_list):
        data_info = detector_data[det.name]
        time_array = np.asarray(data_info.time)
        data_td = data_info.windowed_strain
        freqs_full = np.asarray(data_info.frequency)
        mask = data_info.band_mask
        dt = data_info.dt
        n_time = time_array.size

        sample_td = []
        sample_psd = []

        for idx in draw_idx:
            params = {name: float(samples[name][idx]) for name in latent_names}
            if "log_amp" in samples:
                params["log_amp"] = float(samples["log_amp"][idx])
            params.update(fixed_params or {})

            waveform_fd = waveform(jnp.asarray(freqs_full), params)
            response_fd = det.fd_response(jnp.asarray(freqs_full), waveform_fd, params)
            response_fd = np.asarray(response_fd)

            full_fd = np.zeros_like(response_fd, dtype=np.complex128)
            full_fd[mask] = response_fd[mask]

            td = np.fft.irfft(full_fd, n=n_time) / dt
            sample_td.append(td)
            psd = (np.abs(full_fd) ** 2) / n_time
            sample_psd.append(psd)

        sample_td = np.asarray(sample_td)
        sample_psd = np.asarray(sample_psd)

        lower = np.percentile(sample_td, ci[0], axis=0)
        upper = np.percentile(sample_td, ci[1], axis=0)
        median = np.percentile(sample_td, 50, axis=0)

        ax_time = axes[row, 0]
        ax_time.plot(time_array, data_td, color="tab:gray", lw=0.8, label=f"{det.name} data")
        ax_time.plot(time_array, median, color="tab:orange", lw=1.2, label="Posterior median")
        ax_time.fill_between(time_array, lower, upper, color="tab:orange", alpha=0.3,
                             label=f"{ci[0]}–{ci[1]}% CI")
        ax_time.set_title(f"{title_prefix} {det.name} time domain")
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Strain")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend(frameon=False)

        data_psd = (np.abs(data_info.data_fd_likelihood) ** 2) / n_time
        median_psd = np.percentile(sample_psd, 50, axis=0)
        lower_psd = np.percentile(sample_psd, ci[0], axis=0)
        upper_psd = np.percentile(sample_psd, ci[1], axis=0)

        ax_psd = axes[row, 1]
        finite_mask = (median_psd > 0) & np.isfinite(median_psd)
        if log_f:
            ax_psd.loglog(freqs_full[mask], data_psd[mask], color="tab:blue", alpha=0.6, label="Data PSD")
            if np.any(finite_mask & mask):
                ax_psd.loglog(freqs_full[mask], median_psd[mask], color="tab:orange", label="Posterior median")
                ax_psd.fill_between(freqs_full[mask], lower_psd[mask], upper_psd[mask],
                                    color="tab:orange", alpha=0.2, label=f"{ci[0]}–{ci[1]}% CI")
        else:
            ax_psd.plot(freqs_full[mask], data_psd[mask], color="tab:blue", alpha=0.6, label="Data PSD")
            ax_psd.plot(freqs_full[mask], median_psd[mask], color="tab:orange", label="Posterior median")
            ax_psd.fill_between(freqs_full[mask], lower_psd[mask], upper_psd[mask], color="tab:orange", alpha=0.2,
                                label=f"{ci[0]}–{ci[1]}% CI")

        ax_psd.set_title(f"{title_prefix} {det.name} PSD")
        ax_psd.set_xlabel("Frequency [Hz]")
        ax_psd.set_ylabel("Power spectral density")
        ax_psd.grid(True, which="both", alpha=0.3)
        ax_psd.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
