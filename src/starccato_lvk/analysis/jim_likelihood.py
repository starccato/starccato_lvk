from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional
import importlib
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import time
from jax.scipy.special import logsumexp
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.infer import MCMC, NUTS, init_to_uniform
from numpyro.util import enable_x64
from numpyro.infer.util import log_density

from ..jimgw.core.single_event.likelihood import BaseTransientLikelihoodFD as TransientLikelihoodFD


jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()
enable_x64()


@dataclass
class LikelihoodRunResult:
    samples: Mapping[str, np.ndarray]
    logZ: float
    logZ_err: float
    runtime: float
    extra: Dict[str, np.ndarray | float]


def build_log_density_fn(model, model_kwargs):
    model_kwargs = jax.tree_util.tree_map(jnp.asarray, model_kwargs)

    def _logpost(params):
        log_prob, _ = log_density(model, (), model_kwargs, params)
        return log_prob

    return jax.jit(_logpost)


def build_transient_likelihood(
    detectors,
    waveform,
    *,
    trigger_time: float,
    duration: float,
    post_trigger_duration: float,
) -> TransientLikelihoodFD:
    """Construct a JIM transient likelihood from prepared detectors and waveform."""
    return TransientLikelihoodFD(
        detectors=detectors,
        waveform=waveform,
        trigger_time=float(trigger_time),
    )


def _normalise_latent_sigma(latent_sigma: float | Iterable[float], latent_names: Iterable[str]) -> np.ndarray:
    latent_sigma = np.asarray(latent_sigma, dtype=np.float64)
    latent_names = list(latent_names)
    if latent_sigma.ndim == 0:
        latent_sigma = np.full((len(latent_names),), float(latent_sigma))
    elif latent_sigma.shape != (len(latent_names),):
        raise ValueError("latent_sigma must be scalar or match number of latent dimensions.")
    return latent_sigma


def _build_numpyro_model(
    likelihood: TransientLikelihoodFD,
    latent_names,
    latent_sigma,
    log_amp_sigma,
    fixed_params,
):
    latent_names = list(latent_names)
    latent_sigma_arr = _normalise_latent_sigma(latent_sigma, latent_names)
    log_amp_sigma = float(log_amp_sigma)

    def model():
        params = {}
        for idx, name in enumerate(latent_names):
            params[name] = numpyro.sample(name, dist.Normal(0.0, latent_sigma_arr[idx]))
        params["log_amp"] = numpyro.sample("log_amp", dist.Normal(0.0, log_amp_sigma))
        params.update(fixed_params)
        log_like = likelihood.evaluate(params, None)
        numpyro.factor("log_likelihood", log_like)

    return model, latent_sigma_arr, log_amp_sigma


def run_numpyro_sampling(
    likelihood: TransientLikelihoodFD,
    *,
    latent_names,
    fixed_params: Mapping[str, float],
    rng_key: jax.Array,
    latent_sigma: float | Iterable[float] = 1.0,
    log_amp_sigma: float = 1.0,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    dense_mass: bool = True,
    progress_bar: bool = True,
    init_strategy=None,
) -> LikelihoodRunResult:
    """Run NumPyro NUTS sampling for the supplied likelihood."""
    model, latent_sigma_arr, log_amp_sigma_val = _build_numpyro_model(
        likelihood, latent_names, latent_sigma, log_amp_sigma, fixed_params
    )

    strategy = init_strategy if init_strategy is not None else init_to_uniform()
    kernel = NUTS(model, dense_mass=dense_mass, init_strategy=strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    t0 = time.perf_counter()
    mcmc.run(rng_key)
    runtime = time.perf_counter() - t0
    samples = {name: np.asarray(value) for name, value in mcmc.get_samples().items()}
    samples_grouped = {
        name: np.asarray(value)
        for name, value in mcmc.get_samples(group_by_chain=True).items()
    }
    sample_stats = {name: np.asarray(value) for name, value in mcmc.get_extra_fields().items()}

    logpost_single = build_log_density_fn(model, {})
    logpost_vectorized = jax.vmap(logpost_single)
    log_posterior = np.asarray(
        logpost_vectorized({k: jnp.asarray(v) for k, v in samples.items()})
    )

    return LikelihoodRunResult(
        samples=samples,
        logZ=np.nan,
        logZ_err=np.nan,
        runtime=float(runtime),
        extra={
            "num_chains": num_chains,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "samples_grouped": samples_grouped,
            "log_posterior": log_posterior,
            "latent_sigma": np.asarray(latent_sigma_arr),
            "log_amp_sigma": float(log_amp_sigma_val),
            **sample_stats,
        },
    )


def run_nested_sampling(
    likelihood: TransientLikelihoodFD,
    *,
    latent_names,
    fixed_params: Mapping[str, float],
    rng_key: jax.Array,
    latent_sigma: float | Iterable[float] = 1.0,
    log_amp_sigma: float = 1.0,
    num_live_points: int = 500,
    max_samples: int = 20000,
    num_posterior_samples: int = 2000,
    verbose: bool = False,
) -> LikelihoodRunResult:
    """Run JIM nested sampling using the supplied likelihood."""
    model, _, _ = _build_numpyro_model(
        likelihood, latent_names, latent_sigma, log_amp_sigma, fixed_params
    )

    ns = NestedSampler(
        model,
        constructor_kwargs=dict(num_live_points=num_live_points, gradient_guided=True, verbose=verbose),
        termination_kwargs=dict(dlogZ=0.001, ess=500, max_samples=max_samples),
    )
    run_key, sample_key = jax.random.split(rng_key)
    t0 = time.perf_counter()
    ns.run(run_key)
    runtime = time.perf_counter() - t0

    weighted_samples, log_weights = ns.get_weighted_samples()
    log_weights = np.asarray(log_weights)
    weights = np.exp(log_weights - logsumexp(log_weights))
    posterior_samples = ns.get_samples(
        sample_key,
        num_posterior_samples,
        group_by_chain=False,
    )
    samples = {name: np.asarray(values) for name, values in posterior_samples.items()}

    try:
        logZ = float(ns.evidence)
        logZ_err = float(ns.evidence_error)
    except AttributeError:
        logZ = float("nan")
        logZ_err = float("nan")

    extra = {
        "weighted_samples": {name: np.asarray(values) for name, values in weighted_samples.items()},
        "weights": weights,
        "log_weights": log_weights,
        "num_live_points": num_live_points,
        "max_samples": max_samples,
        "num_posterior_samples": num_posterior_samples,
    }

    return LikelihoodRunResult(
        samples=samples,
        logZ=logZ,
        logZ_err=logZ_err,
        runtime=float(runtime),
        extra=extra,
    )


def posterior_means(samples: Mapping[str, np.ndarray], parameter_names: Iterable[str]) -> Dict[str, float | list[float]]:
    """Return posterior mean for listed parameters (scalars or vectors)."""
    means: Dict[str, float | list[float]] = {}
    for name in parameter_names:
        if name not in samples:
            continue
        value = np.asarray(samples[name])
        mean_val = np.mean(value, axis=0)
        if np.isscalar(mean_val) or mean_val.ndim == 0:
            means[name] = float(mean_val)
        else:
            means[name] = mean_val.tolist()
    return means
