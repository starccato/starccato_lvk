import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

from starccato_jax.waveforms import StarccatoCCSNe
import arviz as az

jax.config.update("jax_enable_x64", True)  # Enable float64 precision for JAX

# Constants
FS, N, T = 4096.0, 512, 512 / 4096.0
DT = 1.0 / FS
DF = 1.0 / T  # Frequency resolution
PSD_N = 4096
TIMES = jnp.arange(0, T, DT)[:N]
FREQS = jnp.fft.rfftfreq(N, d=DT)
FLOW = 100.0
FMASK = (FREQS >= FLOW) & (FREQS <= 2048)
DEFAULT_AMPL = 1e-22  # Default amplitude for the waveform
N_BURNIN, N_STEPS = 100, 100


class AnalysisObj:
    def __init__(self, noise: jnp.ndarray, psd: jnp.ndarray, z_injection: jnp.ndarray, outdir: str = 'results'):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.noise = noise
        self.psd = psd
        self.model = StarccatoCCSNe()

        self.z_injection = z_injection
        self.injection = self.model.generate(z_injection, jax.random.PRNGKey(0)) * DEFAULT_AMPL
        self.data = self.noise + self.injection

        # freq series
        self.noise_fft = jnp.fft.rfft(self.noise)
        self.data_fft = jnp.fft.rfft(self.data)
        self.injection_fft = jnp.fft.rfft(self.injection)
        self.injection_snr = self._compute_snr(self.injection_fft)
        print(f"Injection SNR: {self.injection_snr:.2f}")

        # Compute likelihood at injection point for debugging
        inj_loglike = self.log_likelihood(z_injection, jax.random.PRNGKey(0))
        print(f"Log-likelihood at injection: {inj_loglike:.2f}")

        self.plot()[0].savefig(f"{outdir}/injection_signal.png", dpi=150)

    @classmethod
    def from_simulated_noise(cls, z_injection: jnp.ndarray, outdir: str = 'results'):
        from pycbc import noise as pycbc_noise
        from pycbc import psd as pycbc_psd
        psd_series = pycbc_psd.aLIGOAPlusDesignSensitivityT1800042(PSD_N, DF, FLOW)
        ts = pycbc_noise.noise_from_psd(10 * N, DT, psd_series, seed=0)
        psd = jnp.interp(
            FREQS,
            xp=psd_series.sample_frequencies.numpy(),
            fp=psd_series.data
        ) * (FS * N / 2.0)  # Scale PSD to match FFT normalization
        noise = np.array(ts.data[:N])  # Use first N samples of noise
        return cls(noise, psd, z_injection, outdir)

    def _compute_snr(self, signal_fft):
        num = inner_product(signal_fft, self.data_fft, self.psd)  # <H|D>
        den = inner_product(signal_fft, signal_fft, self.psd)  # <H|H>
        return float(num / jnp.sqrt(den))

    def log_likelihood(self, z, rng_key):
        return log_likelihood(self.data_fft, self.psd, self.model, z, rng_key)

    def log_likelihood_td(self, noise_sigma, z, rng_key):
        return log_likelihood_td(noise_sigma, self.data_fft, self.psd, self.model, z, rng_key)

    def plot(self):
        return plot_signal(
            data=self.data,
            inj=self.injection,
            noise=self.noise,
            data_fft=self.data_fft,
            inj_fft=self.injection_fft,
            noise_fft=self.noise_fft,
            psd=self.psd,
            snr=self.injection_snr
        )


@jax.jit
def inner_product(a: jnp.ndarray, b: jnp.ndarray, psd: jnp.ndarray) -> float:
    a, b, psd = a[FMASK], b[FMASK], psd[FMASK]
    return jnp.sum((jnp.conj(a) * b) / psd).real


# @jax.jit(static_argnames=("model",))
def log_likelihood(data_fft: jnp.ndarray, psd: jnp.ndarray, model: StarccatoCCSNe, z: jnp.ndarray,
                   rng_key: jax.random.PRNGKey):
    y_model = model.generate(z, rng_key) * DEFAULT_AMPL
    model_fft = jnp.fft.rfft(y_model)
    num = inner_product(model_fft, data_fft, psd)  # ⟨h,d⟩
    den = inner_product(model_fft, model_fft, psd)  # ⟨h,h⟩
    logL = num - 0.5 * den  # + constant (can be ignored in MCMC)
    return logL


# @jax.jit(static_argnames=("model",))
def log_likelihood_td(noise_sigma, data: jnp.ndarray, model: StarccatoCCSNe, z: jnp.ndarray,
                      rng_key: jax.random.PRNGKey):
    y_model = model.generate(z, rng_key) * DEFAULT_AMPL
    resid = data - y_model
    logL = -0.5 * jnp.sum((resid / noise_sigma) ** 2) - 0.5 * jnp.log(2 * jnp.pi * noise_sigma ** 2) * len(data)
    return logL


def numpyro_model(likelihood):
    z = numpyro.sample("z", dist.Normal(0, 1).expand([likelihood.model.latent_dim]))
    rng_key = numpyro.prng_key()
    lnl = likelihood.log_likelihood(z, rng_key)
    numpyro.deterministic("log_likelihood", lnl)
    numpyro.factor("likelihood", lnl)


def run_mcmc_with_diagnostics(likelihood, n_burnin=N_BURNIN, n_steps=N_STEPS, n_chains=2, outdir='results'):
    """Run MCMC with better diagnostics and multiple chains"""

    # Initialize NUTS with more conservative settings
    nuts_kernel = NUTS(
        numpyro_model,
        target_accept_prob=0.8,  # Lower target accept prob for better exploration
        max_tree_depth=10,  # Limit tree depth to prevent extremely long trajectories
        init_strategy=numpyro.infer.init_to_median  # Better initialization
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=n_burnin,
        num_samples=n_steps,
        num_chains=n_chains,
        progress_bar=True
    )

    # Run with different random keys for each chain
    mcmc.run(jax.random.PRNGKey(42), likelihood=likelihood)

    # Print detailed diagnostics
    mcmc.print_summary(exclude_deterministic=False)

    # Get samples and compute additional diagnostics
    samples = mcmc.get_samples()

    # Compute R-hat and ESS manually if needed
    print("\n=== Additional Diagnostics ===")
    for param in samples.keys():
        if param != 'log_likelihood':  # Skip deterministic variables
            param_samples = samples[param]
            # Reshape for arviz: (chain, draw, *param_shape)
            param_reshaped = param_samples.reshape(n_chains, n_steps, *param_samples.shape[1:])

            # Convert to arviz InferenceData for better diagnostics
            az_data = az.convert_to_inference_data({param: param_reshaped})

            # Compute diagnostics
            rhat = az.rhat(az_data)[param].values
            ess_bulk = az.ess(az_data, method="bulk")[param].values
            ess_tail = az.ess(az_data, method="tail")[param].values

            print(f"{param}:")
            print(f"  R-hat: {np.mean(rhat):.3f} (should be < 1.01)")
            print(f"  ESS (bulk): {np.mean(ess_bulk):.1f}")
            print(f"  ESS (tail): {np.mean(ess_tail):.1f}")

    plot_results(likelihood, mcmc, outdir=outdir)
    return mcmc


def plot_signal(
        data, inj, noise,
        data_fft, inj_fft, noise_fft, psd,
        snr
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    ax1.plot(TIMES, data, label='Data', color='gray', alpha=0.7)
    ax1.plot(TIMES, inj, label='Injection', color='orange', alpha=0.5, linewidth=2.5, ls='--', zorder=-1)
    ax1.plot(TIMES, noise, label='Noise', color='green', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Strain')
    ax1.legend()

    # Frequency domain
    f = FREQS[FMASK]
    data_pdgrm = jnp.abs(data_fft[FMASK]) ** 2
    inj_pdgrm = jnp.abs(inj_fft[FMASK]) ** 2
    noise_pdgrm = jnp.abs(noise_fft[FMASK]) ** 2
    ax2.loglog(f, psd[FMASK], label='PSD', color='blue', linewidth=2)
    ax2.loglog(f, data_pdgrm, color='gray', alpha=0.7, label='Data FFT')
    ax2.loglog(f, inj_pdgrm, label='Injection FFT', color='orange', alpha=0.5, linewidth=2.5, ls='--', zorder=-1)
    ax2.loglog(f, noise_pdgrm, label='Noise FFT', color='green', alpha=0.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.legend()

    # add SNR to the plot in textbox top left corner
    ax1.text(0.05, 0.95, f'SNR: {snr:.2f}', transform=ax1.transAxes,
             fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax2.set_xlim(FLOW, 2048)
    plt.tight_layout()
    return fig, [ax1, ax2]


def plot_results(likelihood, mcmc, outdir='results'):
    os.makedirs(outdir, exist_ok=True)
    samples = mcmc.get_samples()['z'][:50]

    # Generate predictions
    predictions = []
    for i, z in enumerate(samples):
        pred = likelihood.model.generate(z, random.PRNGKey(i)) * DEFAULT_AMPL
        predictions.append(pred)
    predictions = jnp.array(predictions)

    pred_fft = jnp.abs(jnp.fft.rfft(predictions, axis=1)) ** 2

    fig, (ax1, ax2) = likelihood.plot()

    # Add posterior predictions
    ax1.fill_between(TIMES,
                     jnp.percentile(predictions, 5, axis=0),
                     jnp.percentile(predictions, 95, axis=0),
                     alpha=0.3, color='red', label='90% CI')
    ax1.plot(TIMES,
             jnp.median(predictions, axis=0),
             color='red', linewidth=2, label='Posterior Median')

    ci_95 = jnp.percentile(pred_fft[:, FMASK], 95, axis=0)
    ci_5 = jnp.percentile(pred_fft[:, FMASK], 5, axis=0)
    median_fft = jnp.median(pred_fft[:, FMASK], axis=0)

    ax2.fill_between(FREQS[FMASK], ci_5, ci_95,
                     alpha=0.3, color='red', label='90% CI')
    ax2.loglog(FREQS[FMASK], median_fft,
               color='red', linewidth=2, label='Posterior Median')

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/posterior.png', dpi=150, bbox_inches='tight')
    plt.close()

    inf_data = az.from_numpyro(mcmc)
    az.plot_trace(inf_data)
    plt.savefig(f'{outdir}/trace_plot.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    outdir = 'out_mcmc'
    likelihood = AnalysisObj.from_simulated_noise(z_injection=jnp.zeros(32), outdir=outdir)
    mcmc = run_mcmc_with_diagnostics(likelihood, n_burnin=200, n_steps=200, n_chains=2, outdir=outdir)
