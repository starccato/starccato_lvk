import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.nested_sampling import NestedSampler
import arviz as az
from jax.random import PRNGKey
from starccato_jax.starccato_model import StarccatoModel
import time
from .lvk_data_prep import LvkDataPrep
from .arviz_utils import _to_inference_obj
from .jax_utils import nfft_jax, compute_log_likelihood_jax


def _bayesian_model(rescaled_data, starccato_model, window, rng):
    """
    NumPyro Bayesian model for nested sampling.
    """
    # Extract data from dictionary
    data_fd = jnp.array(rescaled_data['strain_fd'])
    psd_array = jnp.array(rescaled_data['psd_array'])
    df = rescaled_data['df']
    sampling_freq = rescaled_data['sampling_frequency']

    latent_dims = starccato_model.latent_dim

    # Sample latent parameters for waveform generation
    theta = numpyro.sample(
        "theta",
        dist.Normal(0, 1).expand([latent_dims])
    )

    # Sample log strain amplitude
    log_strain_amplitude = numpyro.sample(
        "log_strain_amplitude",
        dist.Normal(-2.3, 3.0)
    )

    strain_amplitude = jnp.exp(log_strain_amplitude)

    # Generate waveform using StarCCaTo model
    y_model_td = starccato_model.generate(z=theta.reshape(1, -1), rng=rng)[0] * strain_amplitude
    # window the waveform to reduce edge effects
    y_model_td = y_model_td * window


    # Convert to frequency domain using JAX-compatible nfft
    y_model_fd, _ = nfft_jax(y_model_td, sampling_freq)

    # Match the data length exactly
    if len(y_model_fd) != len(data_fd):
        if len(y_model_fd) < len(data_fd):
            y_model_fd = jnp.pad(y_model_fd, (0, len(data_fd) - len(y_model_fd)))
        else:
            y_model_fd = y_model_fd[:len(data_fd)]

    # Compute log likelihood
    log_l = compute_log_likelihood_jax(y_model_fd, data_fd, psd_array, df)

    # Store deterministic quantities for monitoring
    numpyro.deterministic("log_likelihood_value", log_l)
    numpyro.deterministic("strain_amplitude", strain_amplitude)

    # Add likelihood to the model
    numpyro.factor("likelihood", log_l)


def run_inference(
        data: LvkDataPrep,
        starccato_model: StarccatoModel,
        num_samples: int,
        rng: PRNGKey = jax.random.PRNGKey(0),
        nlive=500,
        max_samples=10000,
        verbose=False,
) -> az.InferenceData:
    """
    Run nested sampling inference for supernova waveform parameters.

    Args:
        starccato_model: Trained StarCCaTo model for waveform generation
        num_samples: Number of posterior samples to draw
        rng: JAX random key
        progress_bar: Whether to show progress
        **ns_kwargs: Additional arguments for NestedSampler

    Returns:
        arviz.InferenceData: Inference results with posterior samples and log evidence
    """
    likelihood_computation_check(data.rescaled_data)
    # Create nested sampler
    ns = NestedSampler(
        _bayesian_model,
        constructor_kwargs=dict(
            num_live_points=nlive,
            gradient_guided=True,
            verbose=verbose
        ),
        termination_kwargs=dict(dlogZ=0.001, ess=500, max_samples=max_samples)
    )

    t0 = time.process_time()
    ns.run(
        rng,
        rescaled_data=data.rescaled_data,
        starccato_model=starccato_model,
        window=data.window,
        rng=rng,
    )

    runtime = time.process_time() - t0
    print(f"Nested sampling completed in {runtime:.2f} seconds.")
    ns.print_summary()

    result = _to_inference_obj(ns, rng, num_samples, data, runtime, starccato_model)
    print_evidence_summary(result)
    return result


def likelihood_computation_check(rescaled_data):
    """Test the likelihood computation with known values."""
    data_fd = jnp.array(rescaled_data['strain_fd'])
    psd_array = jnp.array(rescaled_data['psd_array'])
    df = rescaled_data['df']

    # Test with data as model (should give high likelihood)
    log_l_perfect = compute_log_likelihood_jax(data_fd, data_fd, psd_array, df)

    # Test with zero model (should give lower likelihood)
    zero_model = jnp.zeros_like(data_fd)
    log_l_zero = compute_log_likelihood_jax(zero_model, data_fd, psd_array, df)

    print(f"Likelihood test - Perfect match: {log_l_perfect:.2e}, Zero model: {log_l_zero:.2e}")

    return log_l_perfect, log_l_zero


def print_evidence_summary(results: az.InferenceData):
    log_Z = results.attrs['log_evidence']
    log_Z_err = results.attrs['log_evidence_uncertainty']
    n_evals = results.attrs['num_likelihood_evaluations']
    ess = results.attrs['effective_sample_size']

    print("=" * 50)
    print("NESTED SAMPLING EVIDENCE SUMMARY")
    print("=" * 50)
    print(f"Log Evidence (ln Z):     {log_Z:.3f} ± {log_Z_err:.3f}")
    print(f"Evidence (Z):            {np.exp(log_Z):.2e}")
    print(f"Likelihood evaluations:  {n_evals:,}")
    print(f"Effective sample size:   {ess}")
    print("=" * 50)


