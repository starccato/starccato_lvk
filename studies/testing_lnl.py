import os
from typing import Tuple, Optional

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from jax import random
from numpyro.infer import MCMC, NUTS
from pycbc import noise, psd

from starccato_jax.waveforms import StarccatoCCSNe
from starccato_jax.waveforms import StarccatoVAE

plt.rcParams['grid.linestyle'] = 'None'
jax.config.update("jax_enable_x64", True)

FS = 4096.0  # Sample rate in Hz
N = 512
T = N / FS  # Duration in seconds
DEFAULT_AMPL = 1

class StarccatoLVKLikelihood:
    """
    LVK likelihood using starccato_jax latent variable model with proper normalization.
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
        self.noise_data = np.array(strain_data)

        self.time_array = time_array
        self.psd_freq = psd_freq
        self.psd_values = psd_values

        # Validate inputs
        if len(strain_data) != 512:
            raise ValueError(f"Strain data must have exactly 512 samples. Current: {len(strain_data)}")

        self.dt = time_array[1] - time_array[0]
        self.duration = len(time_array) * self.dt
        self.sample_rate = 1.0 / self.dt

        self.df = 1.0 / self.duration
        self.freq_array = jnp.fft.rfftfreq(len(time_array), self.dt)
        self.freq_mask = (self.freq_array >= freq_range[0]) & (self.freq_array <= freq_range[1])

        # Ensure PSD interpolation is stable
        self.psd_interp = jnp.interp(self.freq_array, psd_freq, psd_values)

        # Only use frequencies where we have valid PSD data
        valid_psd_mask = (self.freq_array >= psd_freq.min()) & (self.freq_array <= psd_freq.max())
        self.freq_mask = self.freq_mask & valid_psd_mask

        # CRITICAL: Calculate characteristic normalization for the model
        self._calculate_model_normalization()

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
        self.noise_fft = jnp.fft.rfft(self.noise_data)

        # Print diagnostics
        injection_str = ""
        if injection_params is not None:
            injection_str = f" (with injection: SNR={self.injection_snr:.2f})"
        else:
            injection_str = " (noise only)"

        print(f"Frequency range: {freq_range[0]:.1f} - {freq_range[1]:.1f} Hz")
        print(f"Analysis frequencies: {jnp.sum(self.freq_mask)} bins")
        print(f"Data loaded{injection_str}")
        self.plot()
        self._test_lnl()

    def _calculate_model_normalization(self):
        """
        Calculate a FIXED normalization constant for the model.
        This ensures consistent scaling across all model evaluations.
        """
        # Sample multiple realizations to get stable statistics
        rng = jax.random.PRNGKey(42)
        n_samples = 100
        total_power = 0.0

        for i in range(n_samples):
            rng, subkey = jax.random.split(rng)
            # Sample from standard normal (the prior)
            z = jax.random.normal(subkey, (self.starccato_model.latent_dim,))
            y = self.starccato_model.generate(z=z, rng=subkey)

            # Calculate power spectral density
            y_fft = jnp.fft.rfft(y)
            y_power = jnp.abs(y_fft[self.freq_mask]) ** 2

            # Weight by 1/PSD to get characteristic SNR
            psd_band = self.psd_interp[self.freq_mask]
            weighted_power = jnp.sum(y_power / jnp.maximum(psd_band, 1e-50))
            total_power += weighted_power

        # Average weighted power
        avg_weighted_power = total_power / n_samples

        # Calculate amplitude that gives SNR = 1
        # SNR^2 = 4 * df * sum(|h|^2 / S_n)
        # For SNR = 1: amplitude = 1 / sqrt(4 * df * avg_weighted_power)
        self.model_characteristic_amplitude = 1.0 / jnp.sqrt(4.0 * self.df * avg_weighted_power)

        print(f"Average model weighted power: {avg_weighted_power:.2e}")
        print(f"Characteristic amplitude for SNR=1: {self.model_characteristic_amplitude:.2e}")

    @classmethod
    def from_simulated_noise(
            cls,
            starccato_model: StarccatoVAE,
            injection_params: Optional[dict] = None,
            noise_level: float = 1e-5,
            use_flat_psd: bool = False,
            **kwargs
    ):
        delta_t = 1.0 / FS
        delta_f = 1.0 / T
        flow = 20.0
        flen = 4096

        if use_flat_psd:
            psd_series = psd.flat_unity(flen, delta_f, flow)
        else:
            psd_series = psd.aLIGOAPlusDesignSensitivityT1800042(flen, delta_f, flow)

        ts = noise.noise_from_psd(10 * N, delta_t, psd_series, seed=0) * noise_level
        # welch = psd.welch(ts, window='hann')

        strain_data = ts.data[:N]
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

    def _generate_injection(self, injection_params: dict) -> jnp.ndarray:
        z = injection_params['z']
        time_shift = injection_params.get('time_shift', 0.0)

        amplitude = injection_params.get('strain_amplitude', DEFAULT_AMPL)

        rng_seed = injection_params.get('rng_seed', 42)

        if len(z) != self.starccato_model.latent_dim:
            raise ValueError(f"Injection z must have {self.starccato_model.latent_dim} dimensions")

        rng = jax.random.PRNGKey(rng_seed)
        injection_signal = self.call_model(z, rng, time_shift=time_shift, strain_amplitude=amplitude)

        # Store the actual amplitude used
        self.injection_params['strain_amplitude'] = amplitude

        return injection_signal

    def call_model(self,
                   theta: jnp.ndarray,
                   rng: random.PRNGKey,
                   time_shift: float = 0.0,
                   strain_amplitude: float = DEFAULT_AMPL,
                   ) -> jnp.ndarray:
        """
        Generate model with FIXED normalization.
        The normalization is deterministic and doesn't depend on the specific realization.
        """

        # Generate raw model output (unitless)
        y_model = self.starccato_model.generate(z=theta, rng=rng)

        # Apply time shift if needed
        y_fft = jnp.fft.rfft(y_model)
        phase_shift = -2j * jnp.pi * self.freq_array * time_shift
        shifted_fft = y_fft * jnp.exp(phase_shift)
        y_model = jnp.fft.irfft(shifted_fft, n=len(y_model))

        return y_model * strain_amplitude

    def frequency_domain_log_likelihood(self, y_model: jnp.ndarray) -> float:
        """
        Compute the Whittle log-likelihood in frequency domain.
        """
        model_fft = jnp.fft.rfft(y_model)
        residual = self.data_fft - model_fft

        # Only use frequencies in our analysis band
        residual_band = residual[self.freq_mask]
        psd_band = self.psd_interp[self.freq_mask]

        # Compute inner product with numerical stability
        eps = 1e-50  # Small epsilon to avoid division by zero
        safe_psd = jnp.maximum(psd_band, eps)

        # Standard GW likelihood: -0.5 * (d-h|d-h)
        # where (a|b) = 4 * Re[∫ a* b / S_n df]
        inner_product = jnp.sum(jnp.abs(residual_band) ** 2 / safe_psd)
        log_likelihood = -2.0 * self.df * inner_product

        return log_likelihood

    def log_likelihood(
            self,
            theta: jnp.ndarray,
            rng: random.PRNGKey,
            time_shift: float = 0.0,
            strain_amplitude: float = DEFAULT_AMPL,
    ) -> float:
        """Compute log likelihood."""
        y_model = self.call_model(theta, rng, time_shift, strain_amplitude)
        return self.frequency_domain_log_likelihood(y_model)

    def get_snr(self, signal: jnp.ndarray) -> float:
        """Calculate matched-filter SNR of a signal."""
        signal_fft = jnp.fft.rfft(signal)
        band = self.freq_mask
        psd = jnp.maximum(self.psd_interp[band], 1e-50)
        df = self.df

        # Matched filter numerator: 4 * Re[sum(data* conj(signal) / psd)] * df
        num = 4.0 * jnp.real(jnp.sum(self.data_fft[band] * jnp.conj(signal_fft[band]) / psd)) * df

        # Signal norm: 4 * sum(|signal|^2 / psd) * df
        denom = 4.0 * jnp.sum(jnp.abs(signal_fft[band]) ** 2 / psd) * df

        return  num / jnp.sqrt(denom)

    @property
    def injection_snr(self) -> float:
        """Get the SNR of the injected signal."""
        if jnp.any(self.injection != 0):
            return float(self.get_snr(self.injection))
        return 0.0

    def _test_lnl(self):
        """Test likelihood computation."""
        rng_test = jax.random.PRNGKey(42)

        # Test with random z
        test_theta = jax.random.normal(rng_test, (self.starccato_model.latent_dim,))
        test_lnl = self.log_likelihood(test_theta, rng_test)
        print(f"Test log likelihood (random z): {test_lnl:.2f}")

        if not jnp.isfinite(test_lnl):
            raise ValueError("Likelihood function returns non-finite values!")

        # Test at injection if available
        if self.injection_params is not None:
            lnl_at_injection = self.log_likelihood(
                self.injection_params['z'],
                jax.random.PRNGKey(0),
                time_shift=self.injection_params.get('time_shift', 0.0),
                strain_amplitude=self.injection_params.get('strain_amplitude', DEFAULT_AMPL)
            )
            print(f"Log likelihood at injection: {lnl_at_injection:.2f}")

            # Also test with slightly perturbed z
            perturbed_z = self.injection_params['z'] + 0.1 * jax.random.normal(rng_test,
                                                                               (self.starccato_model.latent_dim,))
            lnl_perturbed = self.log_likelihood(
                perturbed_z,
                rng_test,
                time_shift=self.injection_params.get('time_shift', 0.0),
                strain_amplitude=self.injection_params.get('strain_amplitude', DEFAULT_AMPL)
            )
            print(f"Log likelihood at perturbed injection: {lnl_perturbed:.2f}")

            if lnl_at_injection < lnl_perturbed:
                print("WARNING: Injection is not at likelihood maximum!")

    def plot(self):
        """Plot strain and PSD."""
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        # Plot strain data

        PDGRM_FACTOR = 2.0 / (FS * N)  # Power spectral density normalization factor


        # Masked frequency array for plotting
        ax[0].plot(self.time_array, self.strain_data, label='Strain Data', color='black', alpha=0.7)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Strain')


        ax[1].loglog(self.freq_array[self.freq_mask], self.psd_interp[self.freq_mask],
                     label='PSD', color='blue', alpha=0.7)


        ax[1].loglog(self.freq_array[self.freq_mask], PDGRM_FACTOR*jnp.abs(self.data_fft[self.freq_mask]) ** 2,
                     label='Data Power', color='black', alpha=0.5)

        if self.injection_params is not None:
            ax[0].plot(self.time_array, self.injection, label='Injection', color='red', alpha=0.7)


            ax[0].plot(self.time_array, self.noise_data, label='Noise', color='green', alpha=0.5)
            ax[1].loglog(self.freq_array[self.freq_mask], PDGRM_FACTOR*jnp.abs(self.injection_fft[self.freq_mask]) ** 2,
                         label='Injection Power', color='red', alpha=0.7)
            ax[1].loglog(self.freq_array[self.freq_mask], PDGRM_FACTOR*jnp.abs(self.noise_fft[self.freq_mask]) ** 2,
                         label='Noise Power', color='green', alpha=0.5)

        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power Spectral Density')

        plt.suptitle(f"SNR: {self.injection_snr:.2f} ")

        plt.tight_layout()
        plt.savefig('strain_and_psd.png', dpi=150)


def starccato_numpyro_model(
        likelihood: StarccatoLVKLikelihood,
        time_jitter_std: float = 1.0 / 4096.0,
        strain_amplitude_range: Tuple[float, float] = (0.5, 2.0),  # As multiple of characteristic
):
    """NumPyro model with proper priors."""
    dims = likelihood.starccato_model.latent_dim

    # Standard normal prior for latent variables
    theta = numpyro.sample("z", dist.Normal(0, 10).expand([dims]))

    # # Time shift prior
    # time_shift = numpyro.sample("time_shift", dist.Normal(0, time_jitter_std))
    #
    # # Strain amplitude as multiple of characteristic amplitude
    # log_strain_scale = numpyro.sample("log_strain_scale",
    #                                   dist.Uniform(jnp.log(strain_amplitude_range[0]),
    #                                                jnp.log(strain_amplitude_range[1])))
    # strain_scale = jnp.exp(log_strain_scale)
    # strain_amplitude = strain_scale * likelihood.model_characteristic_amplitude
    # numpyro.deterministic("strain_amplitude", strain_amplitude)
    # numpyro.deterministic("strain_scale", strain_scale)


    strain_amplitude =DEFAULT_AMPL
    time_shift = 0
    numpyro.deterministic("strain_amplitude", strain_amplitude)
    numpyro.deterministic("time_shift", time_shift)


    # Generate fresh RNG key
    model_rng = numpyro.prng_key()

    # Compute log likelihood
    lnl = likelihood.log_likelihood(
        theta, model_rng,
        time_shift=time_shift,
        strain_amplitude=strain_amplitude,
    )

    # Track likelihood value
    numpyro.deterministic("log_likelihood", lnl)

    # Factor in the likelihood
    numpyro.factor("likelihood", lnl)


def run_sampler(
        likelihood: StarccatoLVKLikelihood,
        rng_key: int = 1,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
) -> MCMC:
    """Run MCMC sampler."""

    print(f"Injection SNR: {likelihood.injection_snr:.2f}")

    rng_key = jax.random.PRNGKey(rng_key)

    # Initialize near injection if available
    init_strategy = numpyro.infer.init_to_sample
    if likelihood.injection_params is not None:
        # Add small noise to injection parameters for initialization
        init_values = {
            "z": likelihood.injection_params['z'] + 0.01 * jax.random.normal(rng_key,
                                                                             (likelihood.starccato_model.latent_dim,)),
            "time_shift": likelihood.injection_params.get('time_shift', 0.0),
            "log_strain_scale": 0.0,  # Scale of 1.0
        }
        init_strategy = numpyro.infer.init_to_value(values=init_values)

    # Configure NUTS
    nuts_kernel = NUTS(
        starccato_numpyro_model,
        # target_accept_prob=target_accept_prob,
        # max_tree_depth=max_tree_depth,
        # adapt_step_size=True,
        # adapt_mass_matrix=True,
        # dense_mass=T,  # Use diagonal for high dimensions
        init_strategy=init_strategy,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True
    )

    mcmc.run(rng_key, likelihood=likelihood)
    mcmc.print_summary()

    return mcmc


def plot_mcmc_results(likelihood, mcmc, outdir='mcmc_results'):
    os.makedirs(outdir, exist_ok=True)
    inf_data = az.from_numpyro(mcmc)
    axes = az.plot_trace(inf_data, var_names=["z"])
    true_params = likelihood.injection_params

    # get flat list of variable names in the order ArviZ plotted them
    var_names_plotted = ["z"] # , "time_shift", "strain_amplitude"]

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



# Example usage
if __name__ == '__main__':
    model = StarccatoCCSNe()


    likelihood = StarccatoLVKLikelihood.from_simulated_noise(
        starccato_model=model,
        injection_params=dict(
            z=jnp.zeros(model.latent_dim),
            time_shift=0.0,
            strain_amplitude=DEFAULT_AMPL
        ),
        noise_level=1e-2,  # Low noise as requested
        use_flat_psd=True,  # Use flat PSD for simplicity
    )

    mcmc = run_sampler(
        likelihood=likelihood,
        rng_key=42,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1
    )

    plot_mcmc_results(likelihood, mcmc, outdir='mcmc_results')