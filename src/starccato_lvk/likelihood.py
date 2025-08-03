from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from jax import random
from numpyro.infer import MCMC, NUTS
from scipy.signal.windows import tukey

from starccato_jax.waveforms import StarccatoCCSNe
from starccato_jax.waveforms import StarccatoVAE


class StarccatoLVKLikelihood:
    """
    LVK likelihood using starccato_jax latent variable model.

    Assumptions:
    - Single detector
    - Known trigger time (no time marginalization)
    - Single polarization
    - Fixed sky location
    - Starccato generates time-domain signals from latent variables
    - FFT to frequency domain for proper GW likelihood computation
    """

    def __init__(
            self,
            strain_data: jnp.ndarray,
            time_array: jnp.ndarray,
            psd_freq: jnp.ndarray,
            psd_values: jnp.ndarray,
            starccato_model: StarccatoVAE,
            freq_range: Tuple[float, float] = (100.0, 2048.0),
            injection_params: Optional[dict] = None,
    ):

        self.starccato_model = starccato_model

        # Store original noise data
        self.noise_data = strain_data.copy()



        self.time_array = time_array
        self.psd_freq = psd_freq
        self.psd_values = psd_values

        # Validate inputs
        if len(strain_data) != 512:
            raise ValueError(
                "Strain data must have exactly 512 samples. "
                f"Current length: {len(strain_data)}")

        self.dt = time_array[1] - time_array[0]
        self.duration = len(time_array) * self.dt
        self.sample_rate = 1.0 / self.dt

        self.df = 1.0 / self.duration
        self.freq_array = jnp.fft.rfftfreq(len(time_array), self.dt)
        self.freq_mask = (self.freq_array >= freq_range[0]) & (self.freq_array <= freq_range[1])

        # Ensure PSD interpolation is stable
        self.psd_interp = jnp.interp(
            self.freq_array,
            psd_freq,
            psd_values
        )

        # Critical fix: Only use frequencies where we have valid PSD data
        valid_psd_mask = (self.freq_array >= psd_freq.min()) & (self.freq_array <= psd_freq.max())
        self.freq_mask = self.freq_mask & valid_psd_mask

        # Add small regularization only where needed
        self.psd_interp = jnp.where(
            self.psd_interp > 0,
            self.psd_interp,
            jnp.inf  # Set invalid PSD values to inf (they'll be masked out)
        )



        # Generate and add injection if parameters provided
        self.injection_params = injection_params
        if injection_params is not None:
            print("Generating injection signal using Starccato model...")
            injection_signal = self._generate_injection(injection_params)
            self.injection = injection_signal
            self.strain_data = strain_data + injection_signal
            print(f"Added injection with amplitude {jnp.sqrt(jnp.mean(injection_signal ** 2)):.2e}")
        else:
            self.injection = jnp.zeros_like(strain_data)
            self.strain_data = strain_data

        self.data_fft = jnp.fft.rfft(self.strain_data)
        # Print some diagnostics
        injection_str = ""
        if injection_params is not None:
            injection_str = f" (with injection: SNR={self.injection_snr:.2f})"
        else:
            injection_str = " (noise only)"

        print(f"Frequency range: {freq_range[0]:.1f} - {freq_range[1]:.1f} Hz")
        print(f"Analysis frequencies: {jnp.sum(self.freq_mask)} bins")
        print(f"Sample rate: {self.sample_rate:.1f} Hz")
        print(f"Duration: {self.duration:.3f} s")
        print(f"Frequency resolution: {self.df:.2f} Hz")
        print(f"Data loaded{injection_str}")

    def plot(self, outdir):
        """ """
        # plot whitened time domain strain (and signal if present)
        # plot frequency domain PSD and data FFT, (and signal FFT if present)
        # plot data spectogram
        # save 1 fig to outdir
        pass

    @classmethod
    def from_hdf5_files(
            cls,
            strain_file: str,
            psd_file: str,
            starccato_model: StarccatoVAE,
            injection_params: Optional[dict] = None,
            **kwargs
    ):
        # Load strain data
        strain_data = TimeSeries.read(strain_file, format='hdf5')
        psd_data = FrequencySeries.read(psd_file, format='hdf5')

        # Crop to 512 samples, centered around t0
        fs = strain_data.sample_rate.value
        t0 = strain_data.times.value[-1] - 1  # hardcoded trigger time to be 1 second before the last sample
        n_samps = 512
        t_offset = n_samps / 2 / fs  # 256 samples before and after t0

        strain_data = strain_data.crop(t0 - t_offset, t0 + t_offset)  # 512 samples

        return cls(
            strain_data=strain_data.value,
            time_array=strain_data.times.value,
            psd_freq=psd_data.frequencies.value,
            psd_values=psd_data.value,
            starccato_model=starccato_model,
            injection_params=injection_params,
            **kwargs
        )

    def _generate_injection(self, injection_params: dict) -> jnp.ndarray:
        """
        Generate injection signal using the same Starccato model.

        Expected injection_params format:
        {
            'z': jnp.ndarray,  # Latent variables (required)
            'time_shift': float,  # Time shift in seconds (optional, default=0.0)
            'amplitude': float,  # Signal amplitude (optional, default=1e-21)
            'rng_seed': int,  # Random seed for generation (optional, default=42)
        }
        """
        # Extract parameters with defaults
        z = injection_params['z']
        time_shift = injection_params.get('time_shift', 0.0)
        amplitude = injection_params.get('amplitude', 1e-21)
        rng_seed = injection_params.get('rng_seed', 42)

        # Validate latent variables
        if len(z) != self.starccato_model.latent_dim:
            raise ValueError(
                f"Injection z must have {self.starccato_model.latent_dim} dimensions, "
                f"got {len(z)}"
            )

        # Generate injection using the same method as likelihood
        rng = jax.random.PRNGKey(rng_seed)

        # Generate base signal
        injection_signal = self.call_model(z, rng, time_shift=time_shift, strain_amplitude=amplitude)
        return injection_signal

    def get_injection_parameters(self) -> Optional[dict]:
        """Return the parameters used to generate the injection."""
        return self.injection_params

    def frequency_domain_log_likelihood(
            self,
            y_model: jnp.ndarray,
    ) -> float:
        """
        Compute the log-likelihood in frequency domain.
        Uses the standard GW likelihood: ln L = -0.5 * (d-h|d-h)
        where (a|b) = 4 * Re[∫ a*(f) b(f) / S_n(f) df]
        """
        model_fft = jnp.fft.rfft(y_model)
        residual = self.data_fft - model_fft

        # Only use frequencies in our analysis band with valid PSD
        residual_band = residual[self.freq_mask]
        psd_band = self.psd_interp[self.freq_mask]

        # Double-check for finite values
        valid_indices = jnp.isfinite(residual_band) & jnp.isfinite(psd_band) & (psd_band > 0)

        if jnp.sum(valid_indices) == 0:
            return -jnp.inf

        residual_valid = residual_band[valid_indices]
        psd_valid = psd_band[valid_indices]

        # Compute inner product: (r|r) = 4 * Re[∫ r*(f) r(f) / S_n(f) df]
        # Note: for real signals, we can use the full spectrum including negative frequencies
        # The factor of 4 accounts for this (2 for positive freqs, 2 for the real part)
        inner_product_integrand = jnp.conj(residual_valid) * residual_valid / psd_valid
        inner_product = 4.0 * jnp.real(jnp.sum(inner_product_integrand)) * self.df

        # The log likelihood is -0.5 * (d-h|d-h)
        log_likelihood = -0.5 * inner_product

        # Safeguard against numerical issues
        if not jnp.isfinite(log_likelihood):
            return -jnp.inf

        return log_likelihood


    def call_model(self,
                   theta: jnp.ndarray,
                   rng: random.PRNGKey,
                   time_shift: float = 0.0,
                   strain_amplitude: float = 1e-21,
                   ) -> jnp.ndarray:
        y_model = self.starccato_model.generate(z=theta, rng=rng)
        # time jitter
        y_fft = jnp.fft.rfft(y_model)
        phase_shift = -2j * jnp.pi * self.freq_array * time_shift
        shifted_fft = y_fft * jnp.exp(phase_shift)
        y_model = jnp.fft.irfft(shifted_fft, n=len(y_model))
        return y_model * strain_amplitude

    def log_likelihood(
            self,
            theta: jnp.ndarray,
            rng: random.PRNGKey,
            time_shift: float = 0.0,
            strain_amplitude: float = 1e-21,
    ) -> float:
        """Compute log likelihood with explicit strain amplitude scaling."""
        try:
            # Generate base model (unitless, O(1))
            y_model = self.call_model(theta, rng, time_shift, strain_amplitude)
            return self.frequency_domain_log_likelihood(y_model)
        except Exception as e:
            print(f"Error in likelihood computation: {e}")
            return -jnp.inf


    @property
    def whitened_data(self) -> jnp.ndarray:
        """Return whitened data in time domain."""
        return whiten(self.strain_data, self.psd_interp, self.sample_rate)

    @property
    def whitened_noise(self) -> jnp.ndarray:
        """Return whitened noise (data without injection) in time domain."""
        return whiten(self.noise_data, self.psd_interp, self.sample_rate)

    @property
    def whitened_injection(self) -> jnp.ndarray:
        """Return whitened injection in time domain."""
        return whiten(self.injection, self.psd_interp, self.sample_rate)

    def get_snr(self, signal: jnp.ndarray) -> float:
        """Calculate matched-filter SNR of a signal given the PSD."""
        signal_fft = jnp.fft.rfft(signal)
        signal_band = signal_fft[self.freq_mask]
        psd_band = self.psd_interp[self.freq_mask]

        # SNR^2 = 4 * Re[∫ |h(f)|^2 / S_n(f) df]
        snr_squared = 4.0 * jnp.real(
            jnp.sum(jnp.conj(signal_band) * signal_band / psd_band)
        ) * self.df

        return jnp.sqrt(jnp.maximum(snr_squared, 0.0))

    @property
    def injection_snr(self) -> float:
        """Get the SNR of the injected signal."""
        if jnp.any(self.injection != 0):
            return self.get_snr(self.injection)
        return 0.0


def whiten(data, psd, fs):
    """Whiten time series data using PSD."""
    data_fft = jnp.fft.rfft(data)
    # Standard whitening formula
    whitening_filter = jnp.sqrt(2.0 / (psd * fs))
    whitened_fft = data_fft * whitening_filter
    return jnp.fft.irfft(whitened_fft, n=len(data))


def starccato_numpyro_model_with_jitter(
        likelihood: StarccatoLVKLikelihood,
        time_jitter_std: float = 1.0 / 4096.0,  # One sample at 4096 Hz
        strain_amplitude_prior: Tuple[float, float] = (1e-23, 1e-19),  # Realistic strain amplitudes
):
    """NumPyro model with physically motivated priors and strain amplitude."""

    dims = likelihood.starccato_model.latent_dim

    # Standard normal prior for latent variables (as intended by the VAE training)
    theta = numpyro.sample("z", dist.Normal(0, 1).expand([dims]))

    # Small time shift - should be much smaller than signal duration
    time_shift = numpyro.sample("time_shift", dist.Normal(0, time_jitter_std))

    # Add strain amplitude as a free parameter to account for unknown scaling
    log_strain_amp = numpyro.sample("log_strain_amplitude", dist.Uniform(
        jnp.log(strain_amplitude_prior[0]),
        jnp.log(strain_amplitude_prior[1])
    ))
    strain_amplitude = jnp.exp(log_strain_amp)
    numpyro.deterministic("strain_amplitude", strain_amplitude)

    # Generate fresh RNG key
    model_rng = numpyro.prng_key()

    # Compute log likelihood with strain amplitude scaling
    lnl = likelihood.log_likelihood(
        theta, model_rng,
        time_shift=time_shift,
        strain_amplitude=strain_amplitude,
    )

    # Debug: track the likelihood value
    numpyro.deterministic("log_likelihood", lnl)

    # Factor in the likelihood
    numpyro.factor("likelihood", lnl)


def run_sampler(
        strain_file: str,
        psd_file: str,
        injection_params: Optional[dict] = None,
        rng_key: int = 1,
        num_warmup: int = 500,
        num_samples: int = 500,
        num_chains: int = 4,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        step_size: float = 1e-3,  # Start with smaller step size
) -> Tuple[StarccatoLVKLikelihood, MCMC]:
    """Run MCMC sampler with conservative settings."""

    # Create likelihood object
    likelihood = StarccatoLVKLikelihood.from_hdf5_files(
        strain_file=strain_file,
        psd_file=psd_file,
        starccato_model=StarccatoCCSNe(),
        injection_params=injection_params
    )

    # Print injection info if provided
    if injection_params is not None:
        print(f"Injection generated with SNR: {likelihood.injection_snr:.2f}")
        print(f"Injection parameters: {injection_params}")
    else:
        print("Running on noise-only data")

    # Test likelihood function first
    print("Testing likelihood function...")
    rng_test = jax.random.PRNGKey(42)
    test_theta = jax.random.normal(rng_test, (likelihood.starccato_model.latent_dim,))
    test_lnl = likelihood.log_likelihood(test_theta, rng_test, 0.0, 1e-21)
    print(f"Test log likelihood: {test_lnl}")

    if not jnp.isfinite(test_lnl):
        raise ValueError("Likelihood function returns non-finite values!")

    if test_lnl < -1e10:
        print("WARNING: Very negative likelihood - check data scaling and PSD")

    rng_key = jax.random.PRNGKey(rng_key)

    # Configure NUTS with conservative settings
    nuts_kernel = NUTS(
        starccato_numpyro_model_with_jitter,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        step_size=step_size,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,  # Use diagonal mass matrix for high-dimensional problems
        init_strategy=numpyro.infer.init_to_sample,  # Random initialization
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True
    )

    mcmc.run(rng_key, likelihood=likelihood)

    # Print basic diagnostics
    mcmc.print_summary()

    return likelihood, mcmc


def check_likelihood_sanity_with_scaling(likelihood: StarccatoLVKLikelihood, num_tests: int = 10):
    """Test likelihood function with proper strain amplitude scaling."""
    rng = jax.random.PRNGKey(42)

    print("=== Likelihood Sanity Check with Strain Scaling ===")
    print(f"Data statistics:")
    print(f"  Max strain: {jnp.max(jnp.abs(likelihood.strain_data)):.2e}")
    print(f"  RMS strain: {jnp.sqrt(jnp.mean(likelihood.strain_data ** 2)):.2e}")

    # Show injection info if present
    if likelihood.injection_params is not None:
        print(f"Injection present with SNR: {likelihood.injection_snr:.2f}")
        print(f"Injection parameters: {likelihood.injection_params}")

    # Test a range of strain amplitudes
    strain_amplitudes_to_test = [1e-23, 1e-22, 1e-21, 1e-20, 1e-19]

    for strain_amp in strain_amplitudes_to_test:
        print(f"\nTesting with strain amplitude: {strain_amp:.0e}")

        # Test noise-only likelihood
        zero_signal = jnp.zeros_like(likelihood.strain_data)
        noise_lnl = likelihood.frequency_domain_log_likelihood(zero_signal)

        # Test with scaled model signal
        rng, subkey = jax.random.split(rng)
        theta = jax.random.normal(subkey, (likelihood.starccato_model.latent_dim,))

        lnl = likelihood.log_likelihood(
            theta, subkey, time_shift=0.0, amplitude=strain_amp
        )

        print(f"  Noise-only log likelihood: {noise_lnl:.2f}")
        print(f"  Signal log likelihood: {lnl:.2f}")
        print(f"  Difference (signal vs noise): {lnl - noise_lnl:.2f}")

        # Check if this amplitude gives reasonable likelihoods
        if -1e6 < lnl < 1e6 and -1e6 < noise_lnl < 1e6:
            print(f"  ✓ Reasonable likelihood values for strain amplitude {strain_amp:.0e}")
        else:
            print(f"  ✗ Extreme likelihood values")

    return strain_amplitudes_to_test


def debug_data_and_model(likelihood: StarccatoLVKLikelihood):
    """Debug data and model properties."""
    print("=== Data and Model Debug ===")

    # Check data properties
    print(f"Strain data shape: {likelihood.strain_data.shape}")
    print(f"Time array shape: {likelihood.time_array.shape}")
    print(
        f"Strain statistics: min={jnp.min(likelihood.strain_data):.2e}, max={jnp.max(likelihood.strain_data):.2e}, std={jnp.std(likelihood.strain_data):.2e}")

    # Check injection
    if jnp.any(likelihood.injection != 0):
        print(
            f"Injection statistics: min={jnp.min(likelihood.injection):.2e}, max={jnp.max(likelihood.injection):.2e}, std={jnp.std(likelihood.injection):.2e}")
        print(f"Injection SNR: {likelihood.injection_snr:.2f}")
        print(
            f"Noise statistics: min={jnp.min(likelihood.noise_data):.2e}, max={jnp.max(likelihood.noise_data):.2e}, std={jnp.std(likelihood.noise_data):.2e}")
    else:
        print("No injection present")

    # Check PSD
    print(f"PSD shape: {likelihood.psd_interp.shape}")
    print(f"PSD statistics: min={jnp.min(likelihood.psd_interp):.2e}, max={jnp.max(likelihood.psd_interp):.2e}")
    print(f"Frequency mask: {jnp.sum(likelihood.freq_mask)} / {len(likelihood.freq_mask)} frequencies used")

    # Test model generation
    rng = jax.random.PRNGKey(42)
    theta = jax.random.normal(rng, (likelihood.starccato_model.latent_dim,))

    try:
        signal = likelihood.starccato_model.generate(z=theta, rng=rng)
        print(f"Model signal shape: {signal.shape}")
        print(
            f"Model signal statistics: min={jnp.min(signal):.2e}, max={jnp.max(signal):.2e}, std={jnp.std(signal):.2e}")

        # Check signal-to-noise ratio estimate with proper scaling
        test_strain_amp = 1e-21
        scaled_signal = signal * test_strain_amp
        signal_power = jnp.mean(scaled_signal ** 2)
        data_power = jnp.mean(likelihood.strain_data ** 2)
        print(f"Scaled signal/data power ratio (with 1e-21 scaling): {signal_power / data_power:.2e}")

        # Calculate what the model SNR would be at this amplitude
        model_snr = likelihood.get_snr(scaled_signal)
        print(f"Model SNR with 1e-21 amplitude scaling: {model_snr:.2f}")

    except Exception as e:
        print(f"Error generating model signal: {e}")


def create_injection_params(
        latent_dim: int,
        rng_seed: int = 42,
        strain_amplitude: float = 1e-21,
        time_shift: float = 0.0,
        z: Optional[jnp.ndarray] = None,
) -> dict:
    """
    Helper function to create injection parameters.

    Args:
        latent_dim: Dimension of latent space
        rng_seed: Random seed for generation
        strain_amplitude: Strain amplitude
        time_shift: Time shift in seconds
        z: Specific latent variables (if None, random ones are generated)

    Returns:
        Dictionary of injection parameters
    """
    if z is None:
        rng = jax.random.PRNGKey(rng_seed)
        z = jax.random.normal(rng, (latent_dim,))

    return {
        'z': z,
        'strain_amplitude': strain_amplitude,
        'time_shift': time_shift,
        'rng_seed': rng_seed,
    }