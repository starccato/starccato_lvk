import arviz
import numpy as np
import matplotlib.pyplot as plt
import jax

from bilby.core.utils import nfft
from bilby.gw import utils as gwutils


# Constants
FMAX = 1024.0
FLOW = 100.0

# Plot styling
DATA_COL = "tab:gray"
SIGNAL_COL = "tab:orange"
PSD_COL = "black"
POSTERIOR_COL = "tab:blue"


def _whiten(x, asd_freq, asd_vals, fs, lowcut=FLOW, eps=1e-20):
    """Whiten time series data using provided ASD."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x * np.hanning(n), n=n)
    PSD_interp = np.interp(freqs, asd_freq, asd_vals ** 2)
    ASD_interp = np.sqrt(np.maximum(PSD_interp, eps))
    if lowcut is not None:
        ASD_interp[freqs < lowcut] = np.inf
    norm = np.sqrt(0.5 * fs * n)
    X_white = X / (ASD_interp * norm)
    return np.fft.irfft(X_white, n=n)


def plot_posterior_predictive(inf_obj: arviz.InferenceData):
    """Plot data with posterior predictive distribution using 90% CI."""
    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract all data from arviz object
    strain_scale = float(inf_obj.constant_data.strain_scale.values[0])
    psd_scale = float(inf_obj.constant_data.psd_scale.values[0])
    sampling_frequency = float(inf_obj.constant_data.sampling_frequency.values[0])
    df = float(inf_obj.constant_data.df.values[0])
    ifo_name = str(inf_obj.constant_data.ifo_name.values)
    t0 = float(inf_obj.constant_data.start_time.values[0])

    # Observational data
    strain_td_rescaled = inf_obj.observed_data.strain_td.values
    strain_fd_rescaled = inf_obj.observed_data.strain_fd.values
    psd_array_rescaled = inf_obj.observed_data.psd_array.values
    freqs = inf_obj.constant_data.frequency_array.values
    t = inf_obj.constant_data.time_array.values

    # Convert back to original units for plotting
    strain = strain_td_rescaled / strain_scale
    asd = np.sqrt(np.clip(psd_array_rescaled / psd_scale, 1e-50, None))


    # Extract pre-computed posterior predictive quantiles
    pp_quantiles  = inf_obj.constant_data.strain_td_quantiles.values

    # Plot 90% CI as shaded region
    ax_time.fill_between(t, pp_quantiles[0], pp_quantiles[2], color=POSTERIOR_COL, alpha=0.3,
                         label='90% CI Posterior Predictive', linewidth=0)
    ax_time.plot(t, pp_quantiles[1], color=POSTERIOR_COL, alpha=0.8, linewidth=1.5,
                 label='Median Posterior Predictive')

    # Add SNR info from arviz attributes
    if hasattr(inf_obj, 'attrs'):
        attrs = inf_obj.attrs
        if 'optimal_snr' in attrs and 'matched_snr' in attrs:
            snr_text = f'Optimal SNR: {attrs["optimal_snr"]:.2f}\nMatched Filter SNR: {attrs["matched_snr"]:.2f}'
            ax_time.text(0.02, 0.98, snr_text, transform=ax_time.transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_time.set_xlabel(f"Time [s] from GPS {t0}")
    ax_time.set_ylabel("Whitened Strain")
    ax_time.legend(frameon=False)
    ax_time.grid(False)
    ax_time.set_title("Time Domain with Posterior Predictive")

    # FREQUENCY DOMAIN PLOT
    mask_s = (freqs >= FLOW) & (freqs <= FMAX)

    freq_strain_original = strain_fd_rescaled / strain_scale
    asd_data = gwutils.asd_from_freq_series(freq_data=freq_strain_original, df=df)
    ax_freq.loglog(freqs[mask_s], asd_data[mask_s], color=DATA_COL, alpha=0.2, linewidth=2,
                   label=f"{ifo_name} Data")

    # Convert posterior predictive quantiles to frequency domain
    pp_quantiles_fd = []
    for pp_waveform in pp_quantiles:
        waveform_fd, _ = nfft(pp_waveform, sampling_frequency)
        asd_model = gwutils.asd_from_freq_series(freq_data=waveform_fd, df=df)

        # Crop to valid frequency range
        if len(asd_model) > len(freqs):
            asd_model = asd_model[:len(freqs)]
        elif len(asd_model) < len(freqs):
            asd_model = np.pad(asd_model, (0, len(freqs) - len(asd_model)), constant_values=asd_model[-1])

        pp_quantiles_fd.append(asd_model)

    pp_lower_fd, pp_median_fd, pp_upper_fd = pp_quantiles_fd

    # Plot 90% CI as shaded region in log space
    ax_freq.fill_between(freqs[mask_s], pp_lower_fd[mask_s], pp_upper_fd[mask_s],
                         color=POSTERIOR_COL, alpha=0.3,
                         label='90% CI Posterior Predictive', linewidth=0)
    ax_freq.loglog(freqs[mask_s], pp_median_fd[mask_s], color=POSTERIOR_COL, alpha=0.8,
                   linewidth=1.5, label='Median Posterior Predictive')

    ax_freq.loglog(freqs[mask_s], asd[mask_s], color=PSD_COL, linewidth=1, alpha=0.7,
                   label=f"{ifo_name} ASD")
    ax_freq.set_xlabel("Frequency [Hz]")
    ax_freq.set_ylabel(r"Strain [strain/$\sqrt{\rm Hz}$]")
    ax_freq.legend(frameon=False)
    ax_freq.grid(False)
    ax_freq.set_title("Frequency Domain with Posterior Predictive")

    # Injection signal (if provided in constant_data)
    if 'injection_signal' in inf_obj.constant_data:
        inj = inf_obj.constant_data.injection_signal.values
        freq_sig, _ = nfft(inj, sampling_frequency)
        asd_sig = gwutils.asd_from_freq_series(freq_data=freq_sig, df=df)
        ax_time.plot(t, inj, color=SIGNAL_COL, label='Injected Signal', linewidth=2, alpha=0.9)
        ax_freq.loglog(freqs[mask_s], asd_sig[mask_s], color=SIGNAL_COL, linewidth=2, alpha=0.9,
                       label="Injected Signal")

    plt.tight_layout()
    return fig


def plot_posterior_comparision(ccsne_obj, glitch_obj, fname="comparison.pdf"):
    """Compare posterior predictives from CCSN and glitch models."""
    pass