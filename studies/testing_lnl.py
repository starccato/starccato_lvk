import os
from typing import Tuple, Optional

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS
from pycbc import noise, psd
from starccato_jax.waveforms import StarccatoCCSNe

plt.rcParams['grid.linestyle'] = 'None'

FS = 4096.0  # Sample rate in Hz
N = 512
T = N / FS  # Duration in seconds
DEFAULT_AMPL = 200
N_BURNIN = 100
N_STEPS = 100
OUTDIR = "mcmc_results"
os.makedirs(OUTDIR, exist_ok=True)


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
            starccato_model: StarccatoCCSNe,
            injection_params: dict,
            freq_range: Tuple[float, float] = (100.0, 2048.0),
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
        # Generate and add injection if parameters provided
        self.injection_params = injection_params
        self.injection = jnp.zeros_like(strain_data)
        self.strain_data = strain_data

        print("Generating injection signal using Starccato model...")
        injection_signal = self.call_model(injection_params['z'], jax.random.PRNGKey(0))
        self.injection = injection_signal
        self.injection_fft = jnp.fft.rfft(injection_signal)
        self.strain_data = strain_data + injection_signal

        self.data_fft = jnp.fft.rfft(self.strain_data)
        self.noise_fft = jnp.fft.rfft(self.noise_data)

        # Print diagnostics
        injection_str = f" (with injection: SNR={self.get_snr(self.injection):.2f})"

        print(f"Frequency range: {freq_range[0]:.1f} - {freq_range[1]:.1f} Hz")
        print(f"Analysis frequencies: {jnp.sum(self.freq_mask)} bins")
        print(f"Data loaded{injection_str}")
        self.plot()
        self._test_lnl()

    @classmethod
    def from_simulated_noise(
            cls,
            starccato_model: StarccatoCCSNe,
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

    def call_model(self, theta: jnp.ndarray, rng: random.PRNGKey, ) -> jnp.ndarray:
        return self.starccato_model.generate(z=theta, rng=rng) * DEFAULT_AMPL

    def log_likelihood(
            self,
            theta: jnp.ndarray,
            rng: random.PRNGKey,
    ) -> float:
        """Compute log likelihood in frequency domain."""
        y_model = self.call_model(theta, rng)

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
        return float(num / jnp.sqrt(denom))

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
            )
            print(f"Log likelihood at injection: {lnl_at_injection:.2f}")

            # Also test with slightly perturbed z
            perturbed_z = self.injection_params['z'] + 0.1 * jax.random.normal(rng_test,
                                                                               (self.starccato_model.latent_dim,))
            lnl_perturbed = self.log_likelihood(
                perturbed_z,
                rng_test,
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

        ax[1].loglog(self.freq_array[self.freq_mask], PDGRM_FACTOR * jnp.abs(self.data_fft[self.freq_mask]) ** 2,
                     label='Data Power', color='black', alpha=0.5)

        if self.injection_params is not None:
            ax[0].plot(self.time_array, self.injection, label='Injection', color='red', alpha=0.7)

            ax[0].plot(self.time_array, self.noise_data, label='Noise', color='green', alpha=0.5)
            ax[1].loglog(self.freq_array[self.freq_mask],
                         PDGRM_FACTOR * jnp.abs(self.injection_fft[self.freq_mask]) ** 2,
                         label='Injection Power', color='red', alpha=0.7)
            ax[1].loglog(self.freq_array[self.freq_mask], PDGRM_FACTOR * jnp.abs(self.noise_fft[self.freq_mask]) ** 2,
                         label='Noise Power', color='green', alpha=0.5)

        ax[0].legend()
        ax[1].legend()
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Power Spectral Density')

        plt.suptitle(f"SNR: {self.get_snr(self.injection):.2f} ")

        plt.tight_layout()
        plt.savefig('strain_and_psd.png', dpi=150)


def starccato_numpyro_model(
        likelihood: StarccatoLVKLikelihood,
):
    """NumPyro model with proper priors."""
    dims = likelihood.starccato_model.latent_dim

    # Standard normal prior for latent variables
    theta = numpyro.sample("z", dist.Normal(0, 10).expand([dims]))

    # Generate fresh RNG key
    model_rng = numpyro.prng_key()

    # Compute log likelihood
    lnl = likelihood.log_likelihood(
        theta, model_rng,
    )

    # Track likelihood value
    numpyro.deterministic("log_likelihood", lnl)

    # Factor in the likelihood
    numpyro.factor("likelihood", lnl)


def run_sampler(
        likelihood: StarccatoLVKLikelihood,
        rng_key: int = 1,
        **kwargs
) -> MCMC:
    """Run MCMC sampler."""
    rng_key = jax.random.PRNGKey(rng_key)
    nuts_kernel = NUTS(starccato_numpyro_model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=N_BURNIN,
        num_samples=N_STEPS,
        num_chains=1,
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
    for zi in true_params['z']:
        axes[0, 0].axvline(zi, color='red', linestyle='--', label='True z')
        axes[0, 1].axhline(zi, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{outdir}/trace_plot.png', dpi=150)

    # plot time domain + freq domain posterior predictions
    posterior_samples = mcmc.get_samples()
    y_models = []
    for i in range(min(50, len(posterior_samples['z']))):
        z_sample = posterior_samples['z'][i]
        y_model = likelihood.call_model(z_sample, random.PRNGKey(i))
        y_models.append(y_model)
    y_models = jnp.array(y_models)


    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # TIME DOMAIN
    ax[0].plot(likelihood.time_array, likelihood.injection, label='Injection', alpha=0.5, color='k', ls='--')
    ax[0].fill_between(
        likelihood.time_array,
        jnp.percentile(y_models, 5, axis=0),
        jnp.percentile(y_models, 95, axis=0),
        color='C0', alpha=0.2, label='90% Credible Interval'
    )
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Strain')

    # FREQ DOMAIN
    freq_mask = likelihood.freq_mask
    psd_interp = likelihood.psd_interp[freq_mask]
    freq_array = likelihood.freq_array[freq_mask]
    injection_fft = likelihood.injection_fft[freq_mask]
    posterior_fft = jnp.fft.rfft(y_models, axis=1)[:, freq_mask]
    periodogram_factor = 2.0 / (FS * N)  # Power spectral density normalization factor
    injection_pdgrm = periodogram_factor* jnp.abs(injection_fft) ** 2
    posterior_pdgrm = periodogram_factor * jnp.abs(posterior_fft) ** 2

    ax[1].loglog(freq_array, psd_interp, label='PSD', color='blue', alpha=0.7)
    ax[1].loglog(freq_array, periodogram_factor * jnp.abs(injection_fft) ** 2,  color='k', ls='--')
    ax[1].fill_between(
        freq_array,
        jnp.percentile(posterior_pdgrm, 5, axis=0),
        jnp.percentile(posterior_pdgrm, 95, axis=0),
        color='C0', alpha=0.2, label='90% Credible Interval'
    )
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power Spectral Density')
    plt.savefig(f'{outdir}/posterior_predictions.png', dpi=150)


if __name__ == '__main__':
    model = StarccatoCCSNe()

    likelihood = StarccatoLVKLikelihood.from_simulated_noise(
        starccato_model=model,
        injection_params=dict(
            z=jnp.zeros(model.latent_dim),
        ),
        noise_level=1,  # Low noise as requested
        use_flat_psd=True,  # Use flat PSD for simplicity
    )

    mcmc = run_sampler(
        likelihood=likelihood,
        rng_key=42,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1
    )

    plot_mcmc_results(likelihood, mcmc, outdir=OUTDIR)
