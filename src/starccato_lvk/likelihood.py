import os
from typing import Tuple, Optional

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from jax import random
from numpyro.infer import MCMC, NUTS
from pycbc import noise, psd
import numpy as np

from starccato_jax.waveforms import StarccatoCCSNe
from starccato_jax.waveforms import StarccatoVAE

jax.config.update("jax_enable_x64", True)

FS = 4096.0  # Sample rate in Hz
N = 512
T = N / FS  # Duration in seconds
DEFAULT_STRAIN_AMPLITUDE = 1e-22  # Default strain amplitude for injections

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
        self.injection = jnp.zeros_like(strain_data)
        self.strain_data = strain_data
        if injection_params is not None:
            print("Generating injection signal using Starccato model...")
            injection_signal = self._generate_injection(injection_params)
            self.injection = injection_signal
            self.injection_fft = jnp.fft.rfft(injection_signal)
            self.strain_data = strain_data + injection_signal

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
        self.plot()
        self._test_lnl()  # Run a test likelihood computation

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
        t0 = strain_data.times.value[-1] - 1  # hardcoded trigger time to be 1 second before the last sample
        t_offset = N / 2 / FS  # 256 samples before and after t0

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

    @classmethod
    def from_simulated_noise(
            cls,
            starccato_model: StarccatoVAE,
            injection_params: Optional[dict] = None,
            **kwargs
    ):
        #  Build an analytic LVK-like PSD (example: Zero-Det High Power aLIGO)
        delta_t = 1.0 / FS  # time resolution
        delta_f = 1.0 / T  # frequency resolution for PSD
        flow = 20.0  # low-frequency cutoff (Hz)
        flen = 4096  # length of PSD array
        psd_series = psd.aLIGOAPlusDesignSensitivityT1800042(flen, delta_f, flow)
        ts = noise.noise_from_psd(4 * N, delta_t, psd_series, seed=0)
        strain_data = ts.data[:N]  # take first N samples
        time_array = ts.sample_times.numpy()[:N]
        return cls(
            strain_data=strain_data,
            time_array=time_array,
            psd_freq=psd_series.sample_frequencies.numpy(),
            psd_values=psd_series.data,
            starccato_model=starccato_model,
            injection_params=injection_params,
            **kwargs
        )

    def plot(self):
        """Plot strain (timeseries) and PSD."""
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(self.time_array, self.strain_data, label='Strain Data', color='black')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Strain')
        ax[1].loglog(self.psd_freq, self.psd_values, label='PSD', color='blue')
        ax[1].loglog(self.freq_array[self.freq_mask], jnp.abs(self.data_fft[self.freq_mask]) ** 2, label='Data FFT',
                     color='black')
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Strain PSD')
        if self.injection_params is not None:
            ax[0].plot(self.time_array, self.injection, label='Injection', color='red', alpha=0.5)
            ax[1].loglog(self.freq_array[self.freq_mask], jnp.abs(self.injection_fft[self.freq_mask]) ** 2,
                         label='Injection FFT', color='red', alpha=0.5)
        plt.tight_layout()
        plt.savefig('strain_and_psd.png', dpi=150)

    def _generate_injection(self, injection_params: dict) -> jnp.ndarray:
        # Extract parameters with defaults
        z = injection_params['z']
        time_shift = injection_params.get('time_shift', 0.0)
        amplitude = injection_params.get('strain_amplitude', DEFAULT_STRAIN_AMPLITUDE)
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
                   strain_amplitude: float = DEFAULT_STRAIN_AMPLITUDE,
                   ) -> jnp.ndarray:
        y_model = self.starccato_model.generate(z=theta, rng=rng)
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
            strain_amplitude: float = DEFAULT_STRAIN_AMPLITUDE,
    ) -> float:
        """Compute log likelihood with explicit strain amplitude scaling."""
        try:
            # Generate base model (unitless, O(1))
            y_model = self.call_model(theta, rng, time_shift, strain_amplitude)
            return self.frequency_domain_log_likelihood(y_model)
        except Exception as e:
            print(f"Error in likelihood computation: {e}")
            return -jnp.inf

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

    def _test_lnl(self):
        rng_test = jax.random.PRNGKey(42)
        test_theta = jax.random.normal(rng_test, (self.starccato_model.latent_dim,))
        test_lnl = self.log_likelihood(test_theta, rng_test, 0.0, 1e-21)
        print(f"Test log likelihood: {test_lnl}")
        if not jnp.isfinite(test_lnl):
            raise ValueError("Likelihood function returns non-finite values!")

        if test_lnl < -1e10:
            print("WARNING: Very negative likelihood - check data scaling and PSD")


def starccato_numpyro_model_with_jitter(
        likelihood: StarccatoLVKLikelihood,
        time_jitter_std: float = 1.0 / 4096.0,  # One sample at 4096 Hz
        strain_amplitude_prior: Tuple[float, float] = (1e-23, 1e-19),  # Realistic strain amplitudes
):
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
        likelihood: StarccatoLVKLikelihood,
        rng_key: int = 1,
        num_warmup: int = 500,
        num_samples: int = 500,
        num_chains: int = 4,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        step_size: float = 1e-3,  # Start with smaller step size
) -> Tuple[StarccatoLVKLikelihood, MCMC]:
    """Run MCMC sampler with conservative settings."""

    print(f"Injection generated with SNR: {likelihood.injection_snr:.2f}")

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

    return mcmc


def plot_mcmc_results(likelihood, mcmc, outdir='mcmc_results'):
    os.makedirs(outdir, exist_ok=True)
    inf_data = az.from_numpyro(mcmc)
    axes = az.plot_trace(inf_data, var_names=["z", "time_shift", "strain_amplitude"])
    true_params = likelihood.injection_params

    # get flat list of variable names in the order ArviZ plotted them
    var_names_plotted = ["z", "time_shift", "strain_amplitude"]

    for i, var in enumerate(var_names_plotted):
        if var not in true_params:
            continue

        if np.ndim(true_params[var]) == 0:
            # scalar case
            axes[i, 0].axvline(true_params[var], color="red", ls="--", label="True value")
            axes[i, 1].axhline(true_params[var], color="red", ls="--")
        else:
            for j, val in enumerate(np.atleast_1d(true_params[var])):
                axes[i, 0].axvline(val, color="red", ls="--", label="True value")
                axes[i, 1].axhline(val, color="red", ls="--")
    plt.tight_layout()
    plt.savefig(f'{outdir}/trace_plot.png', dpi=150)

    # plot time domain injection and predictions
    plt.figure(figsize=(10, 6))
    plt.plot(likelihood.time_array, likelihood.injection, label='Injection', alpha=0.5, color='k', ls='--')
    # posterior predictions
    posterior_samples = mcmc.get_samples()
    for i in range(min(50, len(posterior_samples['z']))):
        z_sample = posterior_samples['z'][i]
        time_shift = posterior_samples['time_shift'][i]
        strain_amplitude = posterior_samples['strain_amplitude'][i]
        y_model = likelihood.call_model(z_sample, random.PRNGKey(i), time_shift, strain_amplitude)
        plt.plot(likelihood.time_array, y_model, color='C0', alpha=0.1)
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.savefig(f'{outdir}/posterior_predictions.png', dpi=150)


if __name__ == '__main__':
    model = StarccatoCCSNe()
    likelihood = StarccatoLVKLikelihood.from_simulated_noise(
        starccato_model=model,
        injection_params=dict(
            z=jax.random.normal(random.PRNGKey(0), (model.latent_dim,)),
            time_shift=0.0,
            strain_amplitude=DEFAULT_STRAIN_AMPLITUDE,
        )
    )

    mcmc = run_sampler(
        likelihood=likelihood,
        rng_key=42,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4
    )

    plot_mcmc_results(likelihood, mcmc)
