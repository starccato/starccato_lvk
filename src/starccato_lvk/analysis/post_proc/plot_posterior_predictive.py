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
CCSN_COL = "tab:blue"
GLITCH_COL = "tab:green"



def _extract_data_from_arviz(inf_obj):
    """Extract and scale data from arviz InferenceData object."""
    # Extract all data from arviz object
    strain_scale = float(inf_obj.constant_data.strain_scale.values[0])
    psd_scale = float(inf_obj.constant_data.psd_scale.values[0])
    sampling_frequency = float(inf_obj.constant_data.sampling_frequency.values[0])
    df = float(inf_obj.constant_data.df.values[0])
    ifo_name = str(inf_obj.constant_data.ifo_name.values).strip("[]'")  # Clean up string
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
    pp_quantiles = inf_obj.constant_data.strain_td_quantiles.values
    pp_quantiles_fd = inf_obj.constant_data.strain_fd_quantiles.values

    model_name = inf_obj.attrs.get('model_name', 'Model')

    data = {
        'strain_scale': strain_scale,
        'psd_scale': psd_scale,
        'sampling_frequency': sampling_frequency,
        'df': df,
        'ifo_name': ifo_name,
        't0': t0,
        'strain': strain,
        'strain_fd_rescaled': strain_fd_rescaled,
        'asd': asd,
        'freqs': freqs,
        't': t,
        'pp_quantiles': pp_quantiles,
        'pp_quantiles_fd': pp_quantiles_fd,  # Add frequency domain quantiles
        'model_name': model_name
    }

    # Add injection signal if available
    if 'injection_signal' in inf_obj.constant_data:
        data['injection_signal'] = inf_obj.constant_data.injection_signal.values

    return data


def _plot_time_domain_posterior(ax, data, color:str, label_prefix="", alpha=0.3, plot_injection=True):
    """Plot time domain posterior predictive on given axes."""
    t = data['t']
    pp_quantiles = data['pp_quantiles']
    t0 = data['t0']

    # Plot 90% CI as shaded region
    ax.fill_between(t, pp_quantiles[0], pp_quantiles[2], color=color, alpha=alpha,
                    label=label_prefix, linewidth=0)
    ax.plot(t, pp_quantiles[1], color=color, alpha=0.8, linewidth=1.5)

    # Plot injection signal if available and requested
    if plot_injection and 'injection_signal' in data:
        inj = data['injection_signal']
        ax.plot(t, inj, color=SIGNAL_COL, label='Injected Signal', linewidth=2, alpha=0.9, zorder=-10)

    ax.set_xlabel(f"Time [s] from GPS {t0}")
    ax.set_ylabel("Strain")
    ax.grid(False)


def _plot_frequency_domain_posterior(ax, data, color:str, label_prefix="", alpha=0.3,
                                     plot_data=True, plot_injection=True):
    """Plot frequency domain posterior predictive on given axes."""
    freqs = data['freqs']
    asd = data['asd']
    pp_quantiles_fd = data['pp_quantiles_fd']  # Now pre-computed!
    sampling_frequency = data['sampling_frequency']
    df = data['df']
    strain_fd_rescaled = data['strain_fd_rescaled']
    strain_scale = data['strain_scale']
    ifo_name = data['ifo_name']

    mask_s = (freqs >= FLOW) & (freqs <= FMAX)

    # Plot data ASD if requested
    if plot_data:
        freq_strain_original = strain_fd_rescaled / strain_scale
        asd_data = gwutils.asd_from_freq_series(freq_data=freq_strain_original, df=df)
        ax.loglog(freqs[mask_s], asd_data[mask_s], color=DATA_COL, alpha=0.2, linewidth=2,
                  label=f"{ifo_name} Data")
        ax.loglog(freqs[mask_s], asd[mask_s], color=PSD_COL, linewidth=1, alpha=0.7,
                  label=f"{ifo_name} ASD")

    # Use pre-computed frequency domain quantiles
    pp_lower_fd, pp_median_fd, pp_upper_fd = pp_quantiles_fd

    # Plot 90% CI as shaded region
    ax.fill_between(freqs[mask_s], pp_lower_fd[mask_s], pp_upper_fd[mask_s],
                    color=color, alpha=alpha,
                    label=label_prefix, linewidth=0)
    ax.loglog(freqs[mask_s], pp_median_fd[mask_s], color=color, alpha=0.8,
              linewidth=1.5)

    # Plot injection signal if available and requested
    if plot_injection and 'injection_signal' in data:
        inj = data['injection_signal']
        freq_sig, _ = nfft(inj, sampling_frequency)
        asd_sig = gwutils.asd_from_freq_series(freq_data=freq_sig, df=df)
        ax.loglog(freqs[mask_s], asd_sig[mask_s], color=SIGNAL_COL, linewidth=2, alpha=0.9,
                  label="Injected Signal")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Strain [strain/$\sqrt{\rm Hz}$]")
    ax.grid(False)


def plot_posterior_predictive(inf_obj: arviz.InferenceData, fname='posterior_predictive.pdf'):
    """Plot data with posterior predictive distribution using 90% CI."""
    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data (now includes frequency domain quantiles)
    data = _extract_data_from_arviz(inf_obj)
    model_name = data.get('model_name', 'Model')
    color = CCSN_COL if model_name.lower() == 'ccsn' else GLITCH_COL

    # Plot time domain
    _plot_time_domain_posterior(ax_time, data, color=color)

    # Add SNR info from arviz attributes
    if hasattr(inf_obj, 'attrs'):
        attrs = inf_obj.attrs
        if 'optimal_snr' in attrs and 'matched_snr' in attrs:
            snr_text = f'[SNRs: Optimal:{attrs["optimal_snr"]:.2f}, Matched:{attrs["matched_snr"]:.2f}'
            ax_time.text(0.02, 0.98, snr_text, transform=ax_time.transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_time.legend(frameon=False)
    ax_time.set_title("Time Domain with Posterior Predictive")

    # Plot frequency domain
    _plot_frequency_domain_posterior(ax_freq, data, color=color)
    ax_freq.legend(frameon=False)
    ax_freq.set_title("Frequency Domain with Posterior Predictive")

    plt.tight_layout()

    # Save the figure
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Posterior-predictive saved to {fname}")

    return fig


def plot_posterior_comparison(ccsne_obj, glitch_obj, fname="comparison.pdf"):
    """Compare posterior predictives from CCSN and glitch models."""
    fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data from both objects (frequency domain conversion done automatically)
    ccsn_data = _extract_data_from_arviz(ccsne_obj)
    glitch_data = _extract_data_from_arviz(glitch_obj)

    # Plot time domain - data and injection only once (from first object)
    _plot_time_domain_posterior(ax_time, ccsn_data, color=CCSN_COL,
                                label_prefix="CCSN ", plot_injection=True)
    _plot_time_domain_posterior(ax_time, glitch_data, color=GLITCH_COL,
                                label_prefix="Glitch ",  plot_injection=False)

    # Add SNR comparisons if available
    snr_texts = []
    if hasattr(ccsne_obj, 'attrs'):
        attrs = ccsne_obj.attrs
        if 'optimal_snr' in attrs and 'matched_snr' in attrs:
            snr_texts.append(f'CCSN [SNRs: Optimal:{attrs["optimal_snr"]:.2f}, Matched:{attrs["matched_snr"]:.2f}')

    if hasattr(glitch_obj, 'attrs'):
        attrs = glitch_obj.attrs
        if 'optimal_snr' in attrs and 'matched_snr' in attrs:
            snr_texts.append(
                f'Glitch [SNRs: Optimal:{attrs["optimal_snr"]:.2f}, Matched:{attrs["matched_snr"]:.2f}')

    if snr_texts:
        snr_text = '\n'.join(snr_texts)
        ax_time.text(0.02, 0.98, snr_text, transform=ax_time.transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_time.legend(frameon=False)
    ax_time.set_title("Time Domain Model Comparison")

    # Plot frequency domain - data and ASD only once (from first object)
    _plot_frequency_domain_posterior(ax_freq, ccsn_data, color=CCSN_COL,
                                     label_prefix="CCSN ", plot_injection=True)
    _plot_frequency_domain_posterior(ax_freq, glitch_data, color=GLITCH_COL,
                                     label_prefix="Glitch ", plot_data=False, plot_injection=False)

    ax_freq.legend(frameon=False)
    ax_freq.set_title("Frequency Domain Model Comparison")

    plt.tight_layout()

    # Save the figure
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {fname}")

    return fig