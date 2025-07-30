import jax
import jax.numpy as jnp
from jax import random, vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from typing import Dict, Any, Optional, Tuple
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

from starccato_jax.waveforms import StarccatoVAE, StarccatoCCSNe


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
    ):
        """
        Initialize the likelihood.

        Args:
            strain_data: Time-domain strain data
            time_array: Time array corresponding to strain data
            psd_freq: Frequency array for PSD
            psd_values: PSD values
            starccato_model: Trained starccato model for waveform generation
            noise_sigma: Noise level for likelihood computation
        """
        self.strain_data = strain_data
        self.time_array = time_array
        self.psd_freq = psd_freq
        self.psd_values = psd_values
        self.starccato_model = starccato_model

        # Time domain properties
        self.dt = time_array[1] - time_array[0]
        self.duration = len(time_array) * self.dt
        self.sample_rate = 1.0 / self.dt

        # Frequency domain properties for proper GW likelihood
        self.df = 1.0 / self.duration
        self.freq_array = jnp.fft.fftfreq(len(time_array), self.dt)
        self.positive_freq_mask = self.freq_array > 0

        # Interpolate PSD to match our frequency grid
        self.psd_interp = jnp.interp(
            self.freq_array[self.positive_freq_mask],
            psd_freq,
            psd_values,
            left=jnp.inf,
            right=jnp.inf
        )

        # Pre-compute data FFT for frequency-domain likelihood
        self.data_fft = jnp.fft.fft(strain_data)

    @classmethod
    def from_hdf5_files(
            cls,
            strain_file: str,
            psd_file: str,
            starccato_model: StarccatoVAE,
            strain_channel: str = "H1:STRAIN",
            psd_channel: str = "H1:PSD",
            **kwargs
    ):
        """
        Load data from HDF5 files in GWPy format.

        Args:
            strain_file: Path to HDF5 file containing strain data
            psd_file: Path to HDF5 file containing PSD data
            starccato_model: Trained starccato model
            strain_channel: Channel name for strain data
            psd_channel: Channel name for PSD data
        """
        # Load strain data
        with h5py.File(strain_file, 'r') as f:
            strain_data = jnp.array(f[strain_channel]['data'][:])
            t0 = f[strain_channel].attrs['t0']
            dt = f[strain_channel].attrs['dt']

        time_array = jnp.arange(len(strain_data)) * dt + t0

        # Load PSD data
        with h5py.File(psd_file, 'r') as f:
            psd_values = jnp.array(f[psd_channel]['data'][:])
            f0 = f[psd_channel].attrs['f0']
            df = f[psd_channel].attrs['df']

        psd_freq = jnp.arange(len(psd_values)) * df + f0

        return cls(strain_data, time_array, psd_freq, psd_values, starccato_model, **kwargs)

    def log_prior(self, theta: jnp.ndarray) -> float:
        """
        Compute the log-prior of the latent variables.
        Standard normal prior for starccato latent space.

        Args:
            theta: latent variables [latent_dim]
        """
        return dist.Normal(0, 1).log_prob(theta).sum()


    def frequency_domain_log_likelihood(
            self,
            y_model: jnp.ndarray,
    ) -> float:
        """
        Proper GW frequency-domain likelihood using PSD.

        Args:
            y_model: model strain data in time domain
        """
        # FFT model to frequency domain
        model_fft = jnp.fft.fft(y_model)

        # Compute residual
        residual = model_fft - self.data_fft

        # Inner product using only positive frequencies
        # (h|h) = 4 * Re[∫ h*(f) h(f) / S(f) df]
        residual_pos = residual[self.positive_freq_mask]

        inner_product = 4 * jnp.real(
            jnp.sum(jnp.conj(residual_pos) * residual_pos / self.psd_interp) * self.df
        )

        # Handle invalid PSD values
        inner_product = jnp.where(
            jnp.isfinite(inner_product),
            inner_product,
            jnp.inf
        )

        return -0.5 * inner_product

    def apply_time_shift(self, signal: jnp.ndarray, time_shift: float) -> jnp.ndarray:
        """
        Apply time shift to signal using FFT phase rotation.

        Args:
            signal: time-domain signal
            time_shift: time shift in seconds (positive = delay signal)
        """
        # FFT to frequency domain
        signal_fft = jnp.fft.fft(signal)

        # Apply phase shift: exp(-2πift) for time shift t
        phase_shift = jnp.exp(-2j * jnp.pi * self.freq_array * time_shift)
        signal_fft_shifted = signal_fft * phase_shift

        # Return to time domain
        return jnp.real(jnp.fft.ifft(signal_fft_shifted))

    def apply_distance_scaling(self, signal: jnp.ndarray, distance: float,
                               reference_distance: float = 100.0) -> jnp.ndarray:
        """
        Scale signal amplitude by distance.

        Args:
            signal: time-domain signal
            distance: luminosity distance in Mpc
            reference_distance: reference distance for scaling (Mpc)
        """
        distance_factor = reference_distance / distance
        return signal * distance_factor

    def log_likelihood(
            self,
            theta: jnp.ndarray,
            rng: random.PRNGKey,
            time_shift: float = 0.0,
            distance: float = 100.0,
    ) -> float:
        """
        Compute log likelihood for given latent variables with time and distance parameters.

        Args:
            theta: latent variables
            rng: random key for starccato generation
            time_shift: time jitter in seconds
            distance: luminosity distance in Mpc
        """
        # Generate base waveform from latent variables
        y_model = self.starccato_model.generate(z=theta, rng=rng)

        # Apply distance scaling
        y_model = self.apply_distance_scaling(y_model, distance)

        # Apply time shift
        if time_shift != 0.0:
            y_model = self.apply_time_shift(y_model, time_shift)

        # Ensure same length as data
        if len(y_model) != len(self.strain_data):
            # Pad or truncate as needed
            if len(y_model) < len(self.strain_data):
                pad_length = len(self.strain_data) - len(y_model)
                y_model = jnp.concatenate([jnp.zeros(pad_length), y_model])
            else:
                y_model = y_model[:len(self.strain_data)]

        return self.frequency_domain_log_likelihood(y_model)


def starccato_numpyro_model_with_jitter(
        likelihood: StarccatoLVKLikelihood,
        beta: float = 1.0,
        time_jitter_std: float = 0.01,  # 10 ms jitter std
        distance_prior_range: Tuple[float, float] = (50.0, 1000.0),  # Mpc
):
    """
    NumPyro model with time jitter and distance parameters.

    Args:
        likelihood: StarccatoLVKLikelihood instance
        beta: tempering parameter (beta=1 for full posterior)
        reference_prior: optional reference prior for importance sampling
        use_frequency_domain: whether to use frequency-domain likelihood
        time_jitter_std: standard deviation for time jitter prior (seconds)
        distance_prior_range: (min, max) for distance uniform prior (Mpc)
    """
    # Get latent dimension from starccato model
    dims = likelihood.starccato_model.latent_dim

    # Sample latent variables with standard normal prior
    theta = numpyro.sample("z", dist.Normal(0, 1).expand([dims]))

    # Sample time jitter (centered at 0, indicating no shift from trigger time)
    time_shift = numpyro.sample("time_shift", dist.Normal(0, time_jitter_std))

    # Sample distance with uniform prior
    distance = numpyro.sample("distance", dist.Uniform(
        distance_prior_range[0], distance_prior_range[1]
    ))

    # Generate random key for starccato
    rng = numpyro.prng_key()

    # Compute likelihood with time and distance parameters
    lnl = likelihood.log_likelihood(
        theta, rng,
        time_shift=time_shift,
        distance=distance,
    )

    # Save untempered log-likelihood
    numpyro.deterministic("untempered_loglike", lnl)

    # Save derived quantities for monitoring
    numpyro.deterministic("time_shift_ms", time_shift * 1000)  # in milliseconds
    numpyro.deterministic("distance_mpc", distance)

    # Apply tempering
    numpyro.factor("likelihood", beta * lnl)





# Example usage
if __name__ == "__main__":
    print("Starccato LVK Likelihood with Latent Variables")
    print("=" * 60)

    # Mock data for testing
    sample_rate = 4096.0
    duration = 1.0
    dt = 1.0 / sample_rate
    time_array = jnp.arange(0, duration, dt)

    # Mock strain data
    key = random.PRNGKey(42)
    strain_data = random.normal(key, shape=(len(time_array),)) * 1e-23

    # Mock PSD
    freq_psd = jnp.logspace(1, 3, 1000)
    psd_values = 1e-48 * (freq_psd / 100.0) ** (-4.8)

    # Create mock starccato model
    print("Creating mock starccato model...")
    starccato_model = StarccatoCCSNe()

    # Create likelihood
    likelihood = StarccatoLVKLikelihood(
        strain_data=strain_data,
        time_array=time_array,
        psd_freq=freq_psd,
        psd_values=psd_values,
        starccato_model=starccato_model,
    )

    # Test likelihood evaluation with jitter and distance
    test_z = random.normal(random.PRNGKey(123), shape=(starccato_model.latent_dim,))
    test_rng = random.PRNGKey(456)
    test_time_shift = 0.005  # 5 ms
    test_distance = 200.0  # 200 Mpc

    try:
        # Test basic likelihood
        log_l_basic = likelihood.log_likelihood(test_z, test_rng, use_frequency_domain=False)
        print(f"Basic log likelihood: {log_l_basic}")

        # Test with time shift and distance
        log_l_full = likelihood.log_likelihood(
            test_z, test_rng,
            time_shift=test_time_shift,
            distance=test_distance,
            use_frequency_domain=True
        )
        print(f"Log likelihood with jitter (5ms) and distance (200 Mpc): {log_l_full}")

        print("Likelihood evaluation successful!")

    except Exception as e:
        print(f"Error in likelihood evaluation: {e}")
