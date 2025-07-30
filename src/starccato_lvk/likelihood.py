import jax.numpy as jnp
from jax import random
from typing import Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from jax import random

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
            freq_range: Tuple[float, float] = (20.0, 2048.0),
    ):
        """
        Initialize the likelihood.

        Args:
            strain_data: Time-domain strain data
            time_array: Time array corresponding to strain data
            psd_freq: Frequency array for PSD
            psd_values: PSD values
            starccato_model: Trained starccato model for waveform generation
            freq_range: Frequency range for analysis (Hz)
        """
        self.strain_data = strain_data
        self.time_array = time_array
        self.psd_freq = psd_freq
        self.psd_values = psd_values
        self.starccato_model = starccato_model

        # Validate inputs
        if len(strain_data) != 512:
            raise ValueError(
                "Strain data must have exactly 512 samples. "
                f"Current length: {len(strain_data)}")

        # Time domain properties
        self.dt = time_array[1] - time_array[0]
        self.duration = len(time_array) * self.dt
        self.sample_rate = 1.0 / self.dt

        # Frequency domain properties for proper GW likelihood
        self.df = 1.0 / self.duration
        self.freq_array = jnp.fft.fftfreq(len(time_array), self.dt)
        self.freq_mask = (self.freq_array >= freq_range[0]) & (self.freq_array <= freq_range[1])

        # Interpolate PSD to match our frequency grid
        self.psd_interp = jnp.interp(
            self.freq_array[self.freq_mask],
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
            **kwargs
    ):
        # Load strain data
        strain_data = TimeSeries.read(strain_file, format='hdf5')
        psd_data = FrequencySeries.read(psd_file, format='hdf5')

        # so crop it to 512 samples, centered around t0
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
            **kwargs
        )

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
        residual_pos = residual[self.freq_mask]

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
        y_model = self.starccato_model.generate(z=theta, rng=rng) * 10e-20

        # Apply distance scaling
        y_model = self.apply_distance_scaling(y_model, distance)

        # Apply time shift
        y_model = self.apply_time_shift(y_model, time_shift)

        return self.frequency_domain_log_likelihood(y_model)


