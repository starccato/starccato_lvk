import arviz as az
import jax
import numpy as np
import jax.numpy as jnp
from starccato_jax.starccato_model import StarccatoModel
from numpyro.contrib.nested_sampling import NestedSampler
from .lvk_data_prep import LvkDataPrep
from bilby.core.utils import nfft
from .constants import SAMPLING_FREQUENCY, FMAX, FLOW
from bilby.gw import utils as gwutils

def to_posterior_dict(inf_obj: az.InferenceData):
    """Convert arviz InferenceData to a posterior dictionary."""
    posterior = {}
    for var in inf_obj.posterior.data_vars:
        posterior[var] = inf_obj.posterior[var].values
    return posterior


def _generate_posterior_predictive_quantiles(
        data: LvkDataPrep,
        theta_samples: np.ndarray, strain_amplitude_samples: np.ndarray,
        starccato_model: StarccatoModel,
        window: jnp.array,
        num_samples=200) -> np.ndarray:
    """Generate posterior predictive waveforms using the StarCCaTo model."""

    strain_scale = data.rescaled_data['strain_scale']
    strain_td_length = len(data.rescaled_data['strain_td'])
    df = data.rescaled_data['df']

    waveforms = []
    waveforms_asd = []
    snrs = []
    nsamp = len(theta_samples)  # Fixed: theta_samples is already the array

    for i in range(min(num_samples, nsamp)):
        theta_sample = theta_samples[i]
        strain_amplitude = strain_amplitude_samples[i]  # Fixed: removed [0] indexing

        # Generate waveform with StarCCaTo
        rng_key = jax.random.PRNGKey(i)
        y_model = starccato_model.generate(z=theta_sample, rng=rng_key)
        y_model = np.array(y_model, dtype=np.float64)
        y_model = y_model * np.array(window)  # Apply windowing to reduce edge effects

        # Apply strain amplitude AND rescaling (same as in the NumPyro model)
        y_model_rescaled = y_model * strain_amplitude / strain_scale

        # Crop/pad to match data length if needed
        if len(y_model_rescaled) > strain_td_length:
            y_model_rescaled = y_model_rescaled[:strain_td_length]
        elif len(y_model_rescaled) < strain_td_length:
            y_model_rescaled = np.pad(y_model_rescaled, (0, strain_td_length - len(y_model_rescaled)))

        y_model_rescaled = np.array(y_model_rescaled)
        waveform_fd, _ = nfft(y_model_rescaled, SAMPLING_FREQUENCY)
        asd = gwutils.asd_from_freq_series(freq_data=waveform_fd, df=df)
        snr = data.compute_snr(y_model_rescaled)

        waveforms_asd.append(asd)
        waveforms.append(np.array(y_model_rescaled))
        snrs.append(snr)  # Store both optimal and matched SNR


    waveforms = np.array(waveforms)
    waveforms_fd = np.array(waveforms_asd)
    snrs = np.array(snrs)


    td_qtles = np.quantile(waveforms, [0.1, 0.5, 0.9], axis=0)
    fd_qtles = np.quantile(waveforms_fd, [0.1, 0.5, 0.9], axis=0)
    snr_qtles = np.quantile(snrs, [0.1, 0.5, 0.9], axis=0)

    return td_qtles, fd_qtles, snr_qtles


def _to_inference_obj(ns_obj: NestedSampler, rng: jax.random.PRNGKey, num_samples: int, data: LvkDataPrep,
                      runtime: float, starccato_model: StarccatoModel) -> az.InferenceData:
    """
    Convert NestedSampler results to arviz InferenceData object.
    """
    # Get posterior samples from nested sampler
    post = ns_obj.get_samples(rng, num_samples=num_samples)

    # Extract parameters
    theta_samples = post["theta"]  # Shape: (num_samples, latent_dims)
    log_amp_samples = post["log_strain_amplitude"]  # Shape: (num_samples,)
    amp_samples = np.exp(log_amp_samples)

    # Compute posterior predictive quantiles
    td_quantiles, fd_quantiles, snr_quantiles = _generate_posterior_predictive_quantiles(
        data=data,
        theta_samples=theta_samples,
        strain_amplitude_samples=amp_samples,
        starccato_model=starccato_model,
        window=data.window,
        num_samples=200
    )

    # Get log likelihood values
    lnl = post.get("log_likelihood_value", np.zeros(num_samples))

    # Compute nested sampling statistics
    stats = {
        'log_evidence': float(ns_obj._results.log_Z_mean),
        'log_evidence_uncertainty': float(ns_obj._results.log_Z_uncert),
        'log_evidence_noise': float(data.lnz_noise),
        'effective_sample_size': int(getattr(ns_obj._results, 'ESS', num_samples)),
        'num_likelihood_evaluations': int(ns_obj._results.total_num_likelihood_evaluations),
        'total_samples': int(ns_obj._results.total_num_samples),
        'runtime': runtime,
        'model_latent_dim': starccato_model.latent_dim,
        'model_name': starccato_model.model_name,
        **data.get_snrs(),
    }

    # Create arviz InferenceData object
    # Add chain dimension for compatibility (nested sampling doesn't have chains)
    posterior_dict = {
        'theta': theta_samples[np.newaxis, :, :],  # Add chain dim: (1, num_samples, latent_dims)
        'log_strain_amplitude': log_amp_samples[np.newaxis, :],  # (1, num_samples)
        'strain_amplitude': amp_samples[np.newaxis, :],  # (1, num_samples)
    }

    # Sample statistics
    sample_stats_dict = {
        'log_likelihood': lnl[np.newaxis, :],  # (1, num_samples)
    }

    # Extract observational data and constants from data.rescaled_data
    rescaled = data.rescaled_data

    # Observed strain and PSD data
    observed_data_dict = {
        'strain_td': rescaled['strain_td'],
        'strain_fd': rescaled['strain_fd'],
        'psd_array': rescaled['psd_array'],
    }

    # Constants and metadata (including posterior predictive quantiles)
    constant_data_dict = {
        'frequency_array': rescaled['frequency_array'],
        'sampling_frequency': np.array([rescaled['sampling_frequency']]),
        'df': np.array([rescaled['df']]),
        'strain_scale': np.array([rescaled['strain_scale']]),
        'psd_scale': np.array([rescaled['psd_scale']]),
        # Additional data for plotting
        'ifo_name': data.ifo.name,
        'start_time': np.array([data.ifo.strain_data.start_time]),
        'time_array': data.ifo.strain_data.time_array,
        # Posterior predictive quantiles as constant data
        'strain_td_quantiles': td_quantiles,  # Shape: (3, time_length) for [5%, 50%, 95%]
        'strain_fd_quantiles': fd_quantiles,  # Shape: (3, (freqs, waveform_fd)
        'snr_quantiles': snr_quantiles,  # Shape: (3, 2) for [5%, 50%, 95%] SNR (optimal, matched)
    }

    # Handle injection data if present
    if hasattr(data, 'injection_params') and data.injection_params is not None:
        for key, value in data.injection_params.items():
            # Ensure arrays have proper shape for arviz
            if isinstance(value, (int, float)):
                constant_data_dict[f'injection_{key}'] = np.array([value])
            else:
                constant_data_dict[f'injection_{key}'] = np.asarray(value)

    if hasattr(data, 'injection_signal') and data.injection_signal is not None:
        constant_data_dict['injection_signal'] = np.asarray(data.injection_signal)

    # Create coordinate system
    coords = {
        'chain': [0],  # Single chain for nested sampling
        'draw': np.arange(num_samples),
        'theta_dim': np.arange(starccato_model.latent_dim),
        'time': np.arange(len(observed_data_dict['strain_td'])),
        'frequency': constant_data_dict['frequency_array'],
        'quantile': ['q05', 'q50', 'q95'],  # Labels for the quantiles,
        'snr_type': ['optimal', 'matched']  # Labels for SNR types
    }

    # Specify dimensions for variables
    dims = {
        'theta': ['chain', 'draw', 'theta_dim'],
        'log_strain_amplitude': ['chain', 'draw'],
        'strain_amplitude': ['chain', 'draw'],
        'log_likelihood': ['chain', 'draw'],
        'strain_td': ['time'],
        'strain_fd': ['frequency'],
        'psd_array': ['frequency'],
        'frequency_array': ['frequency'],
        'time_array': ['time'],
        'strain_td_quantiles': ['quantile', 'time'],  # Quantiles x Time
        'strain_fd_quantiles': ['quantile', 'frequency'],  # Quantiles x Frequency
        'snr_quantiles': ['quantile', 'snr_type'],  # Quantiles for SNR
    }

    # Add dimensions for injection data if present
    if 'injection_signal' in constant_data_dict:
        dims['injection_signal'] = ['time']
    if 'injection_theta' in constant_data_dict:
        dims['injection_theta'] = ['theta_dim']

    # Build the groups dictionary for az.from_dict
    groups = {
        'posterior': posterior_dict,
        'sample_stats': sample_stats_dict,
        'observed_data': observed_data_dict,
        'constant_data': constant_data_dict,  # Quantiles moved here
        'coords': coords,
        'dims': dims,
        'attrs': stats,
    }

    # Create InferenceData
    inference_data = az.from_dict(**groups)

    return inference_data
