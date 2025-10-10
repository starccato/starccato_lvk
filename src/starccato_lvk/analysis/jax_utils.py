import jax.numpy as jnp
from .constants import FLOW, FMAX



def nfft_jax(time_domain_strain, sampling_frequency):
    """JAX-compatible FFT implementation."""
    frequency_domain_strain = jnp.fft.rfft(time_domain_strain)
    frequency_domain_strain = frequency_domain_strain / sampling_frequency
    frequency_array = jnp.linspace(
        0, sampling_frequency / 2, len(frequency_domain_strain)
    )
    return frequency_domain_strain, frequency_array


def compute_log_likelihood_jax(model_fd: jnp.ndarray, data_fd: jnp.ndarray, psd_array: jnp.ndarray, df: float):
    """Compute log likelihood in frequency domain with JAX."""
    residual = data_fd - model_fd

    # Apply frequency mask to avoid low/high frequency issues
    freq_array = jnp.arange(len(psd_array)) * df
    freq_mask = (freq_array >= FLOW) & (freq_array <= FMAX) & (psd_array > 0)

    # Only compute likelihood in valid frequency range
    residual_masked = jnp.where(freq_mask, residual, 0.0)
    psd_masked = jnp.where(freq_mask, psd_array, jnp.inf)

    # Inner product with safety checks
    inner_product_integrand = jnp.conj(residual_masked) * residual_masked / psd_masked
    inner_product_integrand = jnp.where(
        jnp.isfinite(inner_product_integrand),
        inner_product_integrand,
        0.0
    )

    inner_product = 4.0 * df * jnp.real(jnp.sum(inner_product_integrand))

    # Safety check for the final result
    log_l = -0.5 * inner_product
    log_l = jnp.where(jnp.isfinite(log_l), log_l, -jnp.inf)

    return log_l