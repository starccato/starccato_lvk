import os

import jax

# set float64 precision for JAX
jax.config.update("jax_enable_x64", True)

from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe

from gwpy.timeseries import TimeSeries
from starccato_lvk.likelihood import StarccatoLVKLikelihood
from starccato_lvk.sampler import run_sampler
import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt
from jax import vmap



def apply_posterior_transformations(
        posterior_samples: dict,
        model: StarccatoCCSNe,
        reference_distance: float = 100.0,
        sample_rate: float = 4096.0,
) -> jnp.ndarray:
    """
    Apply distance scaling and time shifts to posterior predictions.

    Args:
        posterior_samples: MCMC samples from numpyro
        model: Trained starccato model
        reference_distance: Reference distance used in training (Mpc)
        sample_rate: Sample rate for time shifting (Hz)

    Returns:
        Transformed waveforms with shape (n_samples, n_timepoints)
    """
    n_samples = len(posterior_samples["z"])

    # Generate posterior predictions (vectorized)
    rng_keys = random.split(random.PRNGKey(42), n_samples)

    # Vectorized generation
    generate_fn = vmap(model.generate, in_axes=(0, 0))
    posterior_waveforms = generate_fn(posterior_samples["z"], rng_keys)

    # Apply distance scaling (vectorized)
    distances = posterior_samples["distance"]
    distance_factors = reference_distance / distances
    scaled_waveforms = posterior_waveforms * distance_factors[:, None]

    # Apply time shifts (vectorized)
    time_shifts = posterior_samples["time_shift"]

    def apply_time_shift_single(waveform, time_shift):
        """Apply time shift to a single waveform."""
        # FFT to frequency domain
        waveform_fft = jnp.fft.fft(waveform)

        # Create frequency array
        n_samples = len(waveform)
        dt = 1.0 / sample_rate
        freq_array = jnp.fft.fftfreq(n_samples, dt)

        # Apply phase shift
        phase_shift = jnp.exp(-2j * jnp.pi * freq_array * time_shift)
        waveform_fft_shifted = waveform_fft * phase_shift

        # Return to time domain
        return jnp.real(jnp.fft.ifft(waveform_fft_shifted))

    # Vectorized time shifting
    time_shift_fn = vmap(apply_time_shift_single, in_axes=(0, 0))
    transformed_waveforms = time_shift_fn(scaled_waveforms, time_shifts)

    return transformed_waveforms


def plot_posterior_predictions(
        data: TimeSeries,
        posterior_waveforms: jnp.ndarray,
        posterior_samples: dict,
        n_plot_samples: int = 100,
        alpha: float = 0.1,
        figsize: tuple = (12, 8)
):
    """
    Plot data with posterior prediction overlay.

    Args:
        data: Original strain data (TimeSeries)
        posterior_waveforms: Transformed posterior waveforms
        posterior_samples: MCMC samples
        n_plot_samples: Number of posterior samples to plot
        alpha: Transparency for posterior samples
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Crop data to 512 samples around trigger (matching your likelihood setup)
    fs = data.sample_rate.value
    t0 = data.times.value[-1] - 1  # Same trigger time as in likelihood
    n_samps = 512
    t_offset = n_samps / 2 / fs
    data_cropped = data.crop(t0 - t_offset, t0 + t_offset)

    # Time array for plotting
    time_rel = data_cropped.times.value - t0  # Time relative to trigger

    # Plot 1: Data + Posterior Predictions
    axes[0].plot(time_rel, data_cropped.value, 'k-', linewidth=1, label='Data', alpha=0.8)

    # Plot subset of posterior samples
    n_samples = min(n_plot_samples, len(posterior_waveforms))
    indices = np.random.choice(len(posterior_waveforms), n_samples, replace=False)

    for i, idx in enumerate(indices):
        label = 'Posterior samples' if i == 0 else None
        axes[0].plot(time_rel, posterior_waveforms[idx], 'r-',
                     alpha=alpha, linewidth=0.5, label=label)

    # Plot median prediction
    median_waveform = jnp.median(posterior_waveforms, axis=0)
    axes[0].plot(time_rel, median_waveform, 'b-', linewidth=2,
                 label='Median prediction')

    axes[0].set_ylabel('Strain')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Data vs Posterior Predictions')

    # Plot 2: Residuals
    residuals = posterior_waveforms - data_cropped.value[None, :]
    median_residual = jnp.median(residuals, axis=0)

    # Plot residual distribution
    for i, idx in enumerate(indices):
        label = 'Residual samples' if i == 0 else None
        axes[1].plot(time_rel, residuals[idx], 'gray',
                     alpha=alpha, linewidth=0.5, label=label)

    axes[1].plot(time_rel, median_residual, 'r-', linewidth=2,
                 label='Median residual')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)

    axes[1].set_xlabel('Time from trigger (s)')
    axes[1].set_ylabel('Residual strain')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Residuals (Prediction - Data)')

    plt.tight_layout()
    return fig


def summarize_posterior(posterior_samples: dict):
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

    # Check if samples are stuck (all identical)
    for param in ['distance', 'time_shift']:
        if param in posterior_samples:
            samples = posterior_samples[param]
            if jnp.std(samples) < 1e-10:
                print(f"⚠️  Warning: {param} samples appear stuck (std = {jnp.std(samples):.2e})")

    print(f"Number of samples: {len(posterior_samples['z'])}")
    print(f"Latent dimension: {posterior_samples['z'].shape[1]}")


def test_mcmc(outdir, mock_data_dir, analysis_data):
    paths = analysis_data
    assert os.path.exists(paths["strain_file"])

    mcmc = run_sampler(
        strain_file=paths["strain_file"],
        psd_file=paths["psd_file"],
    )
    assert mcmc is not None

    model = StarccatoCCSNe()
    posterior_samples = mcmc.get_samples()
    data = TimeSeries.read(paths["strain_file"], format='hdf5')

    # Print posterior summary
    summarize_posterior(posterior_samples)

    # Apply transformations to posterior predictions
    print("Applying distance scaling and time shifts...")
    transformed_waveforms = apply_posterior_transformations(
        posterior_samples=posterior_samples,
        model=model,
        reference_distance=100.0,  # Adjust based on your starccato training
        sample_rate=4096.0
    )

    # Plot results
    print("Creating plots...")
    fig = plot_posterior_predictions(
        data=data,
        posterior_waveforms=transformed_waveforms,
        posterior_samples=posterior_samples,
        n_plot_samples=50  # Plot 50 posterior samples
    )

    plt.savefig(os.path.join(outdir, "posterior_predictions.png"))
