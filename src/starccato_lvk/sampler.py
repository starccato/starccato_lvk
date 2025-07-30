from typing import Tuple

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import jax
from .likelihood import StarccatoLVKLikelihood
from starccato_jax.waveforms import StarccatoBlip

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


def run_sampler(
        strain_file: str,
        psd_file: str,
        rng_key:int = 1,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 1,
)-> MCMC:
    """
    Run MCMC sampling with NUTS kernel.

    Args:
        likelihood: StarccatoLVKLikelihood instance
        rng_key: JAX random key
        num_warmup: number of warmup iterations
        num_samples: number of samples to draw
        num_chains: number of MCMC chains
    """

    likelihood = StarccatoLVKLikelihood.from_hdf5_files(
        strain_file=strain_file,
        psd_file=psd_file,
        starccato_model=StarccatoBlip()
    )

    rng_key = jax.random.PRNGKey(rng_key)
    # Run MCMC
    nuts_kernel = NUTS(starccato_numpyro_model_with_jitter)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key, likelihood=likelihood, beta=1.0)

    return mcmc
