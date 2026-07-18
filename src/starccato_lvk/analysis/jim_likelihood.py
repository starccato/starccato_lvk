from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence
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

from ..jimgw.core.single_event.likelihood import (
    BaseTransientLikelihoodFD as TransientLikelihoodFD,
)

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


@dataclass
class MAPInitializationResult:
    """Best local MAP point and diagnostics from a cheap multistart search."""

    values: Dict[str, float]
    log_density: float
    attempts: list[Dict[str, object]]
    runtime_seconds: float = 0.0
    basins: list[Dict[str, object]] = field(default_factory=list)
    best_basin_broad_hits: int = 0
    best_basin_refinement_hits: int = 0
    best_basin_reproduced: bool = False
    next_basin_delta_log_density: float = np.inf
    selected_chain_basin_ranks: list[int] = field(default_factory=list)


def _summarize_map_basins(
    attempts: Sequence[Mapping[str, object]],
    parameter_names: Sequence[str],
    parameter_scales: np.ndarray,
    basin_radius: float,
) -> list[Dict[str, object]]:
    """Greedily cluster optimized points in prior-standardized coordinates."""
    finite = [
        attempt
        for attempt in attempts
        if np.isfinite(float(attempt["log_density"]))
    ]
    ranked = sorted(
        finite, key=lambda item: float(item["log_density"]), reverse=True
    )
    basins: list[Dict[str, object]] = []
    for attempt in ranked:
        values = attempt["values"]
        point = (
            np.asarray([float(values[name]) for name in parameter_names])
            / parameter_scales
        )
        match = None
        for basin in basins:
            if np.linalg.norm(point - basin["center_scaled"]) <= basin_radius:
                match = basin
                break
        if match is None:
            match = {
                "values": dict(values),
                "log_density": float(attempt["log_density"]),
                "center_scaled": point,
                "attempt_indices": [],
                "broad_hits": 0,
                "refinement_hits": 0,
                "successful_hits": 0,
            }
            basins.append(match)
        match["attempt_indices"].append(int(attempt["attempt_index"]))
        stage = str(attempt.get("stage", "broad"))
        hit_key = "refinement_hits" if stage == "refinement" else "broad_hits"
        match[hit_key] += 1
        match["successful_hits"] += int(bool(attempt["success"]))

    if not basins:
        return []
    best_log_density = float(basins[0]["log_density"])
    for rank, basin in enumerate(basins):
        basin["rank"] = rank
        basin["delta_log_density"] = best_log_density - float(
            basin["log_density"]
        )
        basin["total_hits"] = int(
            basin["broad_hits"] + basin["refinement_hits"]
        )
        del basin["center_scaled"]
    return basins


def build_log_density_fn(model, model_kwargs):
    model_kwargs = jax.tree_util.tree_map(jnp.asarray, model_kwargs)

    def _logpost(params):
        log_prob, _ = log_density(model, (), model_kwargs, params)
        return log_prob

    return jax.jit(_logpost)


def find_multistart_map(
    likelihood: TransientLikelihoodFD,
    *,
    latent_names: Iterable[str],
    fixed_params: Mapping[str, float],
    rng_key: jax.Array,
    latent_sigma: float | Iterable[float] = 1.0,
    log_amp_sigma: float = 5.0,
    num_starts: int = 16,
    start_design: str = "sobol",
    log_amp_starts: Sequence[float] = (-2.0, 0.0, 2.0, 4.0),
    initial_values: Optional[Sequence[Mapping[str, float]]] = None,
    maxiter: int = 300,
    num_refine_candidates: int = 0,
    refine_starts_per_candidate: int = 0,
    refine_scale: float = 0.1,
    basin_radius: float = 0.25,
    minimum_broad_hits: int = 2,
    noise_scale_marginal: bool = False,
    nsm_a: float = 100.0,
    nsm_b: float | None = None,
    marginalize_amplitude: bool = False,
) -> MAPInitializationResult:
    """Find a local posterior mode for NUTS without nested sampling.

    The posterior is only ``len(latent_names) + 1`` dimensional for the
    single-detector glitch model, so several bounded L-BFGS searches are much
    cheaper than an evidence run. The search includes the all-zero latent
    point, a low-discrepancy Sobol design (or random prior draws), and a small
    grid in ``log_amp``. Caller-supplied starts (for example an encoder mean)
    are tried first.

    Optionally, the highest-density distinct basins are refined from nearby
    jittered points without rebuilding or recompiling the objective. Basin
    diagnostics report whether the winning basin was reached by multiple broad
    starts, rather than merely confirmed by local refinement.

    This locates high-density basins; it does not estimate their posterior
    masses. Separate NUTS runs should be used when multiple competitive MAP
    solutions are found.
    """
    from scipy.optimize import minimize
    from scipy.special import ndtri
    from scipy.stats import qmc

    started = time.perf_counter()
    latent_names = list(latent_names)
    if num_starts < 1:
        raise ValueError("num_starts must be at least 1.")
    if start_design not in {"random", "sobol"}:
        raise ValueError("start_design must be 'random' or 'sobol'.")
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1.")
    if not log_amp_starts:
        raise ValueError("log_amp_starts must contain at least one value.")
    if num_refine_candidates < 0 or refine_starts_per_candidate < 0:
        raise ValueError("MAP refinement counts must be non-negative.")
    if refine_scale <= 0.0:
        raise ValueError("refine_scale must be positive.")
    if basin_radius <= 0.0:
        raise ValueError("basin_radius must be positive.")
    if minimum_broad_hits < 1:
        raise ValueError("minimum_broad_hits must be at least 1.")

    latent_sigma_arr = _normalise_latent_sigma(latent_sigma, latent_names)
    model, _, log_amp_sigma_val = _build_numpyro_model(
        likelihood,
        latent_names,
        latent_sigma_arr,
        log_amp_sigma,
        fixed_params,
        noise_scale_marginal=noise_scale_marginal,
        nsm_a=nsm_a,
        nsm_b=nsm_b,
        marginalize_amplitude=marginalize_amplitude,
    )
    logpost = build_log_density_fn(model, {})
    parameter_names = list(latent_names)
    if not marginalize_amplitude:
        parameter_names.append("log_amp")

    def vector_logpost(vector):
        return logpost(
            {name: vector[idx] for idx, name in enumerate(parameter_names)}
        )

    value_and_grad = jax.jit(jax.value_and_grad(lambda x: -vector_logpost(x)))

    def objective(vector):
        value, grad = value_and_grad(jnp.asarray(vector, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    starts: list[np.ndarray] = []
    for supplied in initial_values or ():
        starts.append(
            np.asarray(
                [supplied[name] for name in parameter_names], dtype=np.float64
            )
        )

    if start_design == "sobol":
        key_words = np.asarray(rng_key, dtype=np.uint32).reshape(-1)
        design_seed = int(
            np.bitwise_xor.reduce(key_words, initial=np.uint32(0))
        )
        design = qmc.Sobol(
            d=len(latent_names), scramble=True, seed=design_seed
        )
        exponent = int(np.ceil(np.log2(max(num_starts, 2))))
        unit_points = design.random_base2(exponent)[:num_starts]
        eps = np.finfo(np.float64).eps
        prior_latents = ndtri(np.clip(unit_points, eps, 1.0 - eps))
        prior_latents *= latent_sigma_arr
    else:
        prior_latents = (
            np.asarray(
                jax.random.normal(rng_key, (num_starts, len(latent_names))),
                dtype=np.float64,
            )
            * latent_sigma_arr
        )
    prior_latents[0] = 0.0
    for idx in range(num_starts):
        if marginalize_amplitude:
            starts.append(np.asarray(prior_latents[idx], dtype=np.float64))
        else:
            starts.append(
                np.concatenate(
                    [
                        prior_latents[idx],
                        [float(log_amp_starts[idx % len(log_amp_starts)])],
                    ]
                )
            )

    bounds = [
        (-8.0 * float(sigma), 8.0 * float(sigma)) for sigma in latent_sigma_arr
    ]
    if not marginalize_amplitude:
        bounds.append(
            (-4.0 * float(log_amp_sigma_val), 4.0 * float(log_amp_sigma_val))
        )

    attempts: list[Dict[str, object]] = []

    def optimize_start(
        start: np.ndarray, stage: str, candidate_rank: int | None = None
    ) -> None:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-10, "gtol": 1e-6},
        )
        log_density_value = -float(result.fun)
        attempts.append(
            {
                "attempt_index": len(attempts),
                "stage": stage,
                "candidate_rank": candidate_rank,
                "initial_values": {
                    name: float(start[idx])
                    for idx, name in enumerate(parameter_names)
                },
                "values": {
                    name: float(result.x[idx])
                    for idx, name in enumerate(parameter_names)
                },
                "log_density": log_density_value,
                "success": bool(result.success),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )

    for start in starts:
        optimize_start(start, "broad")

    parameter_scales = (
        np.asarray(latent_sigma_arr, dtype=np.float64)
        if marginalize_amplitude
        else np.concatenate([latent_sigma_arr, [float(log_amp_sigma_val)]])
    )
    broad_basins = _summarize_map_basins(
        attempts, parameter_names, parameter_scales, basin_radius
    )
    n_refine_candidates = min(num_refine_candidates, len(broad_basins))
    if n_refine_candidates and refine_starts_per_candidate:
        n_refinements = n_refine_candidates * refine_starts_per_candidate
        jitter = np.asarray(
            jax.random.normal(
                jax.random.fold_in(rng_key, 1),
                (n_refinements, len(parameter_names)),
            ),
            dtype=np.float64,
        ) * (refine_scale * parameter_scales)
        refinement_index = 0
        lower = np.asarray([bound[0] for bound in bounds])
        upper = np.asarray([bound[1] for bound in bounds])
        for candidate_rank in range(n_refine_candidates):
            center_values = broad_basins[candidate_rank]["values"]
            center = np.asarray(
                [center_values[name] for name in parameter_names],
                dtype=np.float64,
            )
            for _ in range(refine_starts_per_candidate):
                start = np.clip(
                    center + jitter[refinement_index], lower, upper
                )
                refinement_index += 1
                optimize_start(start, "refinement", candidate_rank)

    finite = [
        attempt for attempt in attempts if np.isfinite(attempt["log_density"])
    ]
    if not finite:
        raise RuntimeError(
            "All multistart MAP optimizations returned non-finite values."
        )
    basins = _summarize_map_basins(
        attempts, parameter_names, parameter_scales, basin_radius
    )
    best = max(finite, key=lambda attempt: float(attempt["log_density"]))
    best_basin = basins[0]
    next_delta = (
        float(basins[1]["delta_log_density"]) if len(basins) > 1 else np.inf
    )
    return MAPInitializationResult(
        values=dict(best["values"]),
        log_density=float(best["log_density"]),
        attempts=attempts,
        runtime_seconds=time.perf_counter() - started,
        basins=basins,
        best_basin_broad_hits=int(best_basin["broad_hits"]),
        best_basin_refinement_hits=int(best_basin["refinement_hits"]),
        best_basin_reproduced=(
            int(best_basin["broad_hits"]) >= minimum_broad_hits
        ),
        next_basin_delta_log_density=next_delta,
    )


def build_transient_likelihood(
    detectors,
    waveform,
    *,
    trigger_time: float,
    duration: float,
    post_trigger_duration: float,
) -> TransientLikelihoodFD:
    """Construct a JIM transient likelihood from prepared detectors and waveform.

    NOTE: the likelihood evaluates over the *full* rfft grid (its ``__init__``
    resets the detector frequency bounds to ``[0, inf]``, and the Starccato
    waveform requires the full grid anyway). The analysis band ``[flow, fmax]``
    and line notches are therefore enforced through the detector PSD, whose values
    are set to ``+inf`` outside the band and at lines (see
    ``multidet_data_prep._frequency_domain_representation``); those bins then
    contribute zero to the matched-filter inner product.
    """
    return TransientLikelihoodFD(
        detectors=detectors,
        waveform=waveform,
        trigger_time=float(trigger_time),
    )


def _normalise_latent_sigma(
    latent_sigma: float | Iterable[float], latent_names: Iterable[str]
) -> np.ndarray:
    latent_sigma = np.asarray(latent_sigma, dtype=np.float64)
    latent_names = list(latent_names)
    if latent_sigma.ndim == 0:
        latent_sigma = np.full((len(latent_names),), float(latent_sigma))
    elif latent_sigma.shape != (len(latent_names),):
        raise ValueError(
            "latent_sigma must be scalar or match number of latent dimensions."
        )
    return latent_sigma


SKY_PARAM_NAMES = ("ra", "sin_dec", "psi", "t_c")


_LN2 = float(np.log(2.0))
# log_amp marginalization grid: uniform in u = log_amp over +/-5 prior sigmas.
# 4096 nodes resolve likelihood peaks of width ~1/rho for rho up to ~300 at
# the production log_amp_sigma = 5.
_AMP_GRID_POINTS = 4096
_AMP_GRID_HALF_WIDTH_SIGMAS = 5.0


def _amplitude_grid(log_amp_sigma: float):
    """Return (u nodes, log-prior at nodes, log du) for the log_amp marginal."""
    u = jnp.linspace(
        -_AMP_GRID_HALF_WIDTH_SIGMAS * log_amp_sigma,
        _AMP_GRID_HALF_WIDTH_SIGMAS * log_amp_sigma,
        _AMP_GRID_POINTS,
    )
    log_prior = dist.Normal(0.0, log_amp_sigma).log_prob(u)
    log_du = jnp.log(u[1] - u[0])
    return u, log_prior, log_du


def _amplitude_quadratic(likelihood, params):
    """Coefficients of ``log L(A) = A b - A^2 c / 2`` from two evaluations.

    The template enters the Gaussian JIM likelihood linearly in the amplitude
    ``A = exp(log_amp)``, so two likelihood calls determine the quadratic
    exactly without touching the vendored likelihood internals.
    """
    l1 = likelihood.evaluate({**params, "log_amp": 0.0}, None)
    l2 = likelihood.evaluate({**params, "log_amp": _LN2}, None)
    b = (4.0 * l1 - l2) / 2.0
    c = 2.0 * (b - l1)
    return b, c


def _build_numpyro_model(
    likelihood: TransientLikelihoodFD,
    latent_names,
    latent_sigma,
    log_amp_sigma,
    fixed_params,
    noise_scale_marginal: bool = False,
    nsm_a: float = 100.0,
    nsm_b: float | None = None,
    sample_sky: bool = False,
    t_c_sigma: float = 0.01,
    marginalize_amplitude: bool = False,
):
    """Build the NumPyro model for the signal/glitch posterior.

    With ``noise_scale_marginal`` the Gaussian likelihood is replaced by its
    PSD-amplitude-marginalised form: a single global noise scale ``eta`` (PSD =
    ``eta * S``) with an inverse-gamma prior is integrated out analytically,
    yielding the noise-relative log-likelihood

        log L = -(N + a) [ log(R/2 + b) - log(<d|d>/2 + b) ],   R = <d|d> - 2 log L_jim

    where ``N`` is the total in-band bin count and ``<d|d>`` is summed over
    detectors. This is 0 at ``h = 0`` (so ``logZ_noise = 0`` still holds), reduces
    to the Gaussian when the PSD is correct, and discounts spurious residual
    reduction by ~``1/eta`` when the data are louder than the PSD -- making the
    evidences robust to mis-estimated / non-stationary PSDs.

    With ``marginalize_amplitude`` the ``log_amp`` dimension is integrated out
    numerically under its exact N(0, log_amp_sigma) prior (dense trapezoid grid
    in ``log_amp``; the likelihood is an exact quadratic in ``A = exp(log_amp)``
    so the grid needs only two likelihood evaluations per latent point). This
    removes the low-amplitude funnel that makes NUTS diverge whenever the data
    do not constrain the amplitude, leaving a smooth ``len(latent_names)``-D
    posterior. The evidence is unchanged: Z = int p(z) L_marg(z) dz.
    """
    latent_names = list(latent_names)
    latent_sigma_arr = _normalise_latent_sigma(latent_sigma, latent_names)
    log_amp_sigma = float(log_amp_sigma)

    dd_total = n_total = b_val = None
    if noise_scale_marginal:
        from .noise_evidence import band_quantities

        qs = [band_quantities(det) for det in likelihood.detectors]
        dd_total = float(sum(q.dd for q in qs))
        n_total = int(sum(q.n_bins for q in qs))
        b_val = float(nsm_a - 1.0) if nsm_b is None else float(nsm_b)

    if marginalize_amplitude:
        u_grid, u_log_prior, u_log_du = _amplitude_grid(log_amp_sigma)
        amp_grid = jnp.exp(u_grid)

    def model():
        params = {}
        for idx, name in enumerate(latent_names):
            params[name] = numpyro.sample(
                name, dist.Normal(0.0, latent_sigma_arr[idx])
            )
        if not marginalize_amplitude:
            params["log_amp"] = numpyro.sample(
                "log_amp", dist.Normal(0.0, log_amp_sigma)
            )
        params.update(fixed_params)
        if sample_sky:
            # Isotropic sky + small time offset, sampled so the COHERENT signal must
            # explain every detector jointly (an incoherent single-detector glitch
            # cannot, regardless of the network geometry at the trigger time).
            # Sample sky on the 2-sphere via ProjectedNormal to avoid the ra wrap
            # boundary and the arcsin(dec) pole singularity that wreck NUTS.
            u_sky = numpyro.sample("u_sky", dist.ProjectedNormal(jnp.zeros(3)))
            params["ra"] = jnp.arctan2(u_sky[1], u_sky[0]) % (2.0 * jnp.pi)
            params["dec"] = jnp.arcsin(u_sky[2])
            params["psi"] = numpyro.sample("psi", dist.Uniform(0.0, jnp.pi))
            params["t_c"] = numpyro.sample("t_c", dist.Normal(0.0, t_c_sigma))
        if marginalize_amplitude:
            b, c = _amplitude_quadratic(likelihood, params)
            log_like_u = amp_grid * b - 0.5 * amp_grid**2 * c
            if noise_scale_marginal:
                resid = dd_total - 2.0 * log_like_u
                log_like_u = -(n_total + nsm_a) * (
                    jnp.log(resid / 2.0 + b_val)
                    - jnp.log(dd_total / 2.0 + b_val)
                )
            log_like = (
                jax.scipy.special.logsumexp(log_like_u + u_log_prior)
                + u_log_du
            )
        else:
            log_like = likelihood.evaluate(params, None)
            if noise_scale_marginal:
                resid = dd_total - 2.0 * log_like  # R(theta) = <d-h|d-h>
                log_like = -(n_total + nsm_a) * (
                    jnp.log(resid / 2.0 + b_val)
                    - jnp.log(dd_total / 2.0 + b_val)
                )
        numpyro.factor("log_likelihood", log_like)

    if sample_sky:
        from numpyro.infer.reparam import ProjectedNormalReparam

        model = numpyro.handlers.reparam(
            model, config={"u_sky": ProjectedNormalReparam()}
        )

    return model, latent_sigma_arr, log_amp_sigma


def draw_conditional_log_amp(
    likelihood: TransientLikelihoodFD,
    samples: Mapping[str, np.ndarray],
    latent_names,
    fixed_params: Mapping[str, float],
    rng_key: jax.Array,
    *,
    log_amp_sigma: float = 5.0,
    noise_scale_marginal: bool = False,
    nsm_a: float = 100.0,
    nsm_b: float | None = None,
) -> np.ndarray:
    """Draw ``log_amp`` from its exact 1-D conditional for each latent sample.

    Used after amplitude-marginalized NUTS so the returned posterior keeps a
    ``log_amp`` column for plotting, railing checks, and saved samples. The
    conditional p(log_amp | z, d) is categorical on the same grid the marginal
    used, which is exact to the grid resolution.
    """
    latent_names = list(latent_names)
    u_grid, u_log_prior, _ = _amplitude_grid(float(log_amp_sigma))
    amp_grid = jnp.exp(u_grid)

    dd_total = n_total = b_val = None
    if noise_scale_marginal:
        from .noise_evidence import band_quantities

        qs = [band_quantities(det) for det in likelihood.detectors]
        dd_total = float(sum(q.dd for q in qs))
        n_total = int(sum(q.n_bins for q in qs))
        b_val = float(nsm_a - 1.0) if nsm_b is None else float(nsm_b)

    fixed = {
        key: jnp.asarray(value) for key, value in dict(fixed_params).items()
    }

    def conditional_draw(z_row, key):
        params = {
            name: z_row[idx] for idx, name in enumerate(latent_names)
        }
        b, c = _amplitude_quadratic(likelihood, {**params, **fixed})
        log_like_u = amp_grid * b - 0.5 * amp_grid**2 * c
        if noise_scale_marginal:
            resid = dd_total - 2.0 * log_like_u
            log_like_u = -(n_total + nsm_a) * (
                jnp.log(resid / 2.0 + b_val)
                - jnp.log(dd_total / 2.0 + b_val)
            )
        logits = log_like_u + u_log_prior
        return u_grid[dist.Categorical(logits=logits).sample(key)]

    z_matrix = jnp.stack(
        [jnp.asarray(samples[name]) for name in latent_names], axis=1
    )
    keys = jax.random.split(rng_key, z_matrix.shape[0])
    draws = jax.lax.map(
        lambda args: conditional_draw(args[0], args[1]), (z_matrix, keys)
    )
    return np.asarray(draws, dtype=np.float64)


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
    chain_method: str = "parallel",
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    progress_bar: bool = True,
    init_strategy=None,
    init_params: Optional[Mapping[str, jax.Array]] = None,
    noise_scale_marginal: bool = False,
    nsm_a: float = 100.0,
    nsm_b: float | None = None,
    sample_sky: bool = False,
    marginalize_amplitude: bool = False,
) -> LikelihoodRunResult:
    """Run NumPyro NUTS sampling for the supplied likelihood."""
    if not 0.0 < target_accept_prob < 1.0:
        raise ValueError(
            "target_accept_prob must lie strictly between 0 and 1."
        )
    if max_tree_depth < 1:
        raise ValueError("max_tree_depth must be at least 1.")
    if chain_method not in {"parallel", "sequential", "vectorized"}:
        raise ValueError(
            "chain_method must be 'parallel', 'sequential', or 'vectorized'."
        )
    if init_strategy is not None and init_params is not None:
        raise ValueError("Specify only one of init_strategy and init_params.")

    model, latent_sigma_arr, log_amp_sigma_val = _build_numpyro_model(
        likelihood,
        latent_names,
        latent_sigma,
        log_amp_sigma,
        fixed_params,
        noise_scale_marginal=noise_scale_marginal,
        nsm_a=nsm_a,
        nsm_b=nsm_b,
        sample_sky=sample_sky,
        marginalize_amplitude=marginalize_amplitude,
    )

    strategy = (
        init_strategy if init_strategy is not None else init_to_uniform()
    )
    kernel = NUTS(
        model,
        dense_mass=dense_mass,
        init_strategy=strategy,
        target_accept_prob=float(target_accept_prob),
        max_tree_depth=int(max_tree_depth),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    t0 = time.perf_counter()
    mcmc.run(
        rng_key,
        init_params=init_params,
        extra_fields=(
            "diverging",
            "accept_prob",
            "num_steps",
            "energy",
            "potential_energy",
        ),
    )
    runtime = time.perf_counter() - t0
    samples = {
        name: np.asarray(value) for name, value in mcmc.get_samples().items()
    }
    samples_grouped = {
        name: np.asarray(value)
        for name, value in mcmc.get_samples(group_by_chain=True).items()
    }
    sample_stats = {
        name: np.asarray(value)
        for name, value in mcmc.get_extra_fields().items()
    }

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
            "dense_mass": bool(dense_mass),
            "chain_method": chain_method,
            "target_accept_prob": float(target_accept_prob),
            "max_tree_depth": int(max_tree_depth),
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
    num_live_points: int = 100,
    max_samples: int = 2000,
    num_posterior_samples: int = 200,
    verbose: bool = True,
    noise_scale_marginal: bool = False,
    nsm_a: float = 100.0,
    nsm_b: float | None = None,
    sample_sky: bool = False,
    marginalize_amplitude: bool = False,
) -> LikelihoodRunResult:
    """Run JIM nested sampling using the supplied likelihood."""
    model, _, _ = _build_numpyro_model(
        likelihood,
        latent_names,
        latent_sigma,
        log_amp_sigma,
        fixed_params,
        noise_scale_marginal=noise_scale_marginal,
        nsm_a=nsm_a,
        nsm_b=nsm_b,
        sample_sky=sample_sky,
        marginalize_amplitude=marginalize_amplitude,
    )

    ns = NestedSampler(
        model,
        constructor_kwargs=dict(
            num_live_points=num_live_points,
            gradient_guided=True,
            verbose=verbose,
        ),
        termination_kwargs=dict(dlogZ=0.1, ess=100, max_samples=max_samples),
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
    samples = {
        name: np.asarray(values) for name, values in posterior_samples.items()
    }

    try:
        logZ = float(ns.evidence)
        logZ_err = float(ns.evidence_error)
    except AttributeError:
        results = getattr(ns, "_results", None)
        if results is not None and hasattr(results, "log_Z_mean"):
            logZ = float(results.log_Z_mean)
            logZ_err = float(results.log_Z_uncert)
        else:
            logZ = float("nan")
            logZ_err = float("nan")

    extra = {
        "weighted_samples": {
            name: np.asarray(values)
            for name, values in weighted_samples.items()
        },
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


def posterior_means(
    samples: Mapping[str, np.ndarray], parameter_names: Iterable[str]
) -> Dict[str, float | list[float]]:
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
