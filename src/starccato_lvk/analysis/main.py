from __future__ import annotations

import json
import copy
import types
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import init_to_value
from numpyro.diagnostics import effective_sample_size, gelman_rubin

try:
    from arviz_stats import rhat as rank_normalized_rhat
except ImportError:  # pragma: no cover - dependency fallback
    rank_normalized_rhat = None

from starccato_jax.waveforms import MODELS, get_model

from .evidence import EvidenceResult, evidence_with_fallback
from .jim_waveform import StarccatoJimWaveform, StarccatoGlitchWaveform
from .jim_likelihood import (
    LikelihoodRunResult,
    MAPInitializationResult,
    find_multistart_map,
    build_transient_likelihood,
    posterior_means,
    run_nested_sampling,
    run_numpyro_sampling,
    build_log_density_fn,
    _build_numpyro_model,
)
from .post_proc.jim_plots import (
    plot_data_overview,
    plot_posterior_mean_waveform,
    plot_posterior_predictive_from_samples,
)
from .multidet_data_prep import prepare_multi_detector_data

DEFAULT_EXTRINSICS = {
    "t_c": 0.0,
    "ra": 0.0,
    "dec": 0.0,
    "psi": 0.0,
    "luminosity_distance": 10.0,
}


def _clone_no_response_detector(detector_obj):
    det_copy = copy.deepcopy(detector_obj)

    def _fd_response_noresp(self, frequency, h_sky, params, **kwargs):
        if "p" in h_sky:
            return h_sky["p"]
        if "c" in h_sky:
            return h_sky["c"]
        return jnp.zeros_like(frequency, dtype=jnp.complex64)

    det_copy.fd_response = types.MethodType(_fd_response_noresp, det_copy)
    return det_copy


def _analytic_noise_logz(detector) -> float:
    """Log-evidence of the noise-only hypothesis, in the JIM likelihood convention.

    The JIM transient likelihood is noise-relative: it evaluates
    ``log L = sum_det [<h|d> - <h|h>/2]`` and drops the parameter-independent
    ``-<d|d>/2`` and ``log(2*pi*PSD)`` normalisation constants. The signal and
    glitch evidences (from morphZ / nested sampling) are therefore implicitly
    ratios relative to noise, i.e. ``log(Z_X / Z_noise)``. Because the signal
    likelihood sums over all detectors while each glitch/noise term is
    per-detector, the dropped per-detector constants cancel exactly in the BCR
    ratio (see derivation in docs). The self-consistent noise reference in this
    convention is therefore simply ``log Z_noise = 0`` for every detector.
    """
    return 0.0


def _compute_morphz_evidence(
    samples: Dict[str, np.ndarray],
    log_posterior: np.ndarray,
    latent_names: Sequence[str],
    include_log_amp: bool,
    model,
    fixed_params: Mapping[str, float],
    outdir: Path,
    label: str,
    *,
    fallback: Optional[Callable[[], EvidenceResult]] = None,
    log_amp_sigma: float = 1.0,
    verify_logz_threshold: Optional[float] = 50.0,
    rail_sigma: float = 2.5,
) -> EvidenceResult:
    """Estimate ``log Z`` from NUTS samples with morphZ (+ optional fallback).

    Returns an :class:`EvidenceResult` so callers can record which estimator
    produced the value and surface morphZ failures instead of silently emitting
    ``NaN`` into the BCR. When the ``log_amp`` posterior is railed (mean beyond
    ``rail_sigma`` prior sigmas) or the evidence is large, the nested ``fallback``
    is run as a cross-check, because morphZ gives finite-but-wrong evidences on
    such hard posteriors.
    """
    if log_posterior is None:
        return EvidenceResult.failed(
            "no log-posterior values supplied", n_attempts=0
        )

    railed = False
    if include_log_amp and "log_amp" in samples and log_amp_sigma > 0:
        railed = bool(
            abs(float(np.mean(samples["log_amp"])))
            > rail_sigma * log_amp_sigma
        )

    outdir.mkdir(parents=True, exist_ok=True)
    columns: list[np.ndarray] = []
    param_names: list[str] = []
    latent_names = list(latent_names)
    for name in latent_names:
        arr = np.asarray(samples[name])
        if arr.ndim == 1:
            columns.append(arr[:, None])
            param_names.append(name)
        else:
            for idx_col in range(arr.shape[1]):
                columns.append(arr[:, idx_col][:, None])
                param_names.append(f"{name}_{idx_col}")
    if include_log_amp and "log_amp" in samples:
        columns.append(np.asarray(samples["log_amp"])[:, None])
        param_names.append("log_amp")
    post_samples = np.hstack(columns)
    logpost_values = np.asarray(log_posterior)

    log_density_fn = build_log_density_fn(model, {})
    fixed_params = dict(fixed_params)

    def lp_fn(theta: np.ndarray) -> float:
        idx = 0
        params = {}
        for name in latent_names:
            arr = np.asarray(samples[name])
            if arr.ndim == 1:
                params[name] = jnp.array(theta[idx])
                idx += 1
            else:
                size = arr.shape[1]
                params[name] = jnp.array(theta[idx : idx + size])
                idx += size
        if include_log_amp and "log_amp" in samples:
            params["log_amp"] = jnp.array(theta[idx])
        params.update(fixed_params)
        return float(log_density_fn(params))

    result = evidence_with_fallback(
        post_samples,
        lp_fn,
        log_posterior_values=logpost_values,
        param_names=param_names,
        output_path=outdir,
        label=label,
        fallback=fallback,
        verify=railed,
        verify_logz_threshold=verify_logz_threshold,
    )
    if not result.ok:
        print(f"[evidence] '{label}' failed: {result.message}")
    elif result.status != "ok":
        print(f"[evidence] '{label}' used {result.method} ({result.message})")
    return result


def _nested_evidence(
    likelihood,
    latent_names,
    latent_sigma,
    log_amp_sigma,
    fixed_params,
    rng_key,
    *,
    num_live_points: int,
    max_samples: int,
    num_posterior_samples: int,
    noise_scale_marginal: bool = False,
    nsm_a: float = 100.0,
    nsm_b: Optional[float] = None,
) -> EvidenceResult:
    """Run nested sampling purely to obtain a fallback log-evidence."""
    run = run_nested_sampling(
        likelihood,
        latent_names=latent_names,
        fixed_params=fixed_params,
        rng_key=rng_key,
        latent_sigma=latent_sigma,
        log_amp_sigma=log_amp_sigma,
        num_live_points=num_live_points,
        max_samples=max_samples,
        num_posterior_samples=num_posterior_samples,
        noise_scale_marginal=noise_scale_marginal,
        nsm_a=nsm_a,
        nsm_b=nsm_b,
    )
    if not np.isfinite(run.logZ):
        return EvidenceResult.failed(
            "nested sampling produced a non-finite logZ", n_attempts=1
        )
    return EvidenceResult(
        logZ=float(run.logZ),
        logZ_err=float(run.logZ_err),
        method="nested",
        status="ok",
    )


def _compute_log_bcr(
    logZ_signal: float,
    logZ_glitch: Mapping[str, float],
    logZ_noise: Mapping[str, float],
    alpha: float,
    beta: float,
) -> float:
    if not np.isfinite(logZ_signal):
        return float("nan")
    denom_terms = []
    for det_name, logZ_N in logZ_noise.items():
        logZ_G = logZ_glitch.get(det_name, float("-inf"))
        term_glitch = np.log(beta) + logZ_G
        term_noise = np.log1p(-beta) + logZ_N if beta != 1.0 else float("-inf")
        denom_terms.append(np.logaddexp(term_glitch, term_noise))
    return np.log(alpha) + logZ_signal - np.sum(denom_terms)


def _normalise_bundle_paths(
    bundle_paths: Optional[Mapping[str, str]],
) -> Optional[Dict[str, Path]]:
    if bundle_paths is None:
        return None
    return {key.upper(): Path(value) for key, value in bundle_paths.items()}


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_summary_json(outdir: Path, summary: Dict) -> None:
    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _map_initialization_summary(
    result: MAPInitializationResult,
) -> Dict[str, object]:
    next_delta = result.next_basin_delta_log_density
    return {
        "method": "sobol_multistart_lbfgs",
        "values": result.values,
        "log_density": result.log_density,
        "runtime_seconds": result.runtime_seconds,
        "best_basin_broad_hits": result.best_basin_broad_hits,
        "best_basin_refinement_hits": result.best_basin_refinement_hits,
        "best_basin_reproduced": result.best_basin_reproduced,
        "selected_chain_basin_ranks": result.selected_chain_basin_ranks,
        "selected_chain_delta_log_density": [
            float(result.basins[rank]["delta_log_density"])
            for rank in result.selected_chain_basin_ranks
        ],
        "next_basin_delta_log_density": (
            float(next_delta) if np.isfinite(next_delta) else None
        ),
        "basins": result.basins,
        "attempts": result.attempts,
    }


def _save_map_initialization(
    outdir: Path, result: MAPInitializationResult
) -> None:
    (outdir / "map_initialization.json").write_text(
        json.dumps(
            _map_initialization_summary(result), indent=2, sort_keys=True
        )
        + "\n"
    )


def _brief_map_initialization_summary(
    result: MAPInitializationResult,
) -> Dict[str, object]:
    """Return the MAP fields needed to audit a large campaign result."""
    next_delta = result.next_basin_delta_log_density
    return {
        "method": "sobol_multistart_lbfgs",
        "runtime_seconds": float(result.runtime_seconds),
        "best_basin_broad_hits": int(result.best_basin_broad_hits),
        "best_basin_refinement_hits": int(result.best_basin_refinement_hits),
        "best_basin_reproduced": bool(result.best_basin_reproduced),
        "selected_chain_basin_ranks": list(result.selected_chain_basin_ranks),
        "next_basin_delta_log_density": (
            float(next_delta) if np.isfinite(next_delta) else None
        ),
        "n_basins": len(result.basins),
        "n_attempts": len(result.attempts),
    }


def _report_map_initialization(
    label: str, result: MAPInitializationResult
) -> None:
    print(
        f"MAP init [{label}]: logp={result.log_density:.3f}, "
        f"{result.runtime_seconds:.2f}s, "
        f"broad hits={result.best_basin_broad_hits}"
    )
    if not result.best_basin_reproduced:
        print(
            f"[initialization] WARNING: the winning {label} basin was found "
            "by only one broad start."
        )
    if result.next_basin_delta_log_density < 5.0:
        print(
            f"[initialization] WARNING: {label} has a competitive secondary "
            "basin at "
            f"delta log density={result.next_basin_delta_log_density:.3f}."
        )


def _map_nuts_initialization(
    result: MAPInitializationResult,
    parameter_names: Sequence[str],
    num_chains: int,
    *,
    competitive_delta: float = 5.0,
    chain_method: str = "vectorized",
):
    """Spread vectorized NUTS chains across competitive MAP basins."""
    if num_chains < 1:
        raise ValueError("num_chains must be at least 1.")
    if num_chains == 1:
        result.selected_chain_basin_ranks = [0]
        strategy = init_to_value(
            values={
                name: jnp.asarray(value)
                for name, value in result.values.items()
            }
        )
        return strategy, None, "parallel"

    competitive_ranks = [
        rank
        for rank, basin in enumerate(result.basins)
        if float(basin["delta_log_density"]) < competitive_delta
    ]
    if not competitive_ranks:
        competitive_ranks = [0]
    selected_ranks = [
        competitive_ranks[index % len(competitive_ranks)]
        for index in range(num_chains)
    ]
    result.selected_chain_basin_ranks = selected_ranks
    init_params = {
        name: jnp.asarray(
            [result.basins[rank]["values"][name] for rank in selected_ranks]
        )
        for name in parameter_names
    }
    print(
        "Vectorized NUTS chain basins: "
        + ", ".join(
            f"{rank} "
            f"(delta={result.basins[rank]['delta_log_density']:.3f})"
            for rank in selected_ranks
        )
    )
    if len(competitive_ranks) > num_chains:
        print(
            "[initialization] WARNING: more competitive basins than NUTS "
            "chains; increase num_chains to cover every screened basin."
        )
    return None, init_params, chain_method


_NUTS_SAMPLE_STATS = (
    "diverging",
    "accept_prob",
    "num_steps",
    "energy",
    "potential_energy",
)


def _nuts_diagnostics_summary(
    result: LikelihoodRunResult,
) -> Dict[str, object]:
    """Return lightweight convergence diagnostics for a NUTS result."""
    grouped = result.extra.get("samples_grouped")
    if not grouped:
        return {}
    sample_stats = {
        name: np.asarray(result.extra[name])
        for name in _NUTS_SAMPLE_STATS
        if name in result.extra
    }

    ess = _ess_summary(grouped)
    rhat = _maximum_rhat(grouped)

    num_chains = int(result.extra["num_chains"])
    num_samples = int(result.extra["num_samples"])
    ebfmi = None
    if (
        "energy" in sample_stats
        and sample_stats["energy"].size == num_chains * num_samples
    ):
        energy = sample_stats["energy"].reshape(num_chains, num_samples)
        energy_var = np.var(energy, axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ebfmi = (
                np.mean(np.diff(energy, axis=1) ** 2, axis=1) / energy_var
            ).tolist()

    max_tree_depth = int(result.extra["max_tree_depth"])
    max_tree_fraction = None
    if "num_steps" in sample_stats:
        max_tree_fraction = float(
            np.mean(sample_stats["num_steps"] >= (2**max_tree_depth - 1))
        )

    return {
        "divergences": int(np.sum(sample_stats.get("diverging", 0))),
        "mean_accept_prob": (
            float(np.mean(sample_stats["accept_prob"]))
            if "accept_prob" in sample_stats
            else None
        ),
        "max_num_steps": (
            int(np.max(sample_stats["num_steps"]))
            if "num_steps" in sample_stats
            else None
        ),
        "fraction_at_max_tree": max_tree_fraction,
        "ebfmi_by_chain": ebfmi,
        "ess": ess,
        "max_rhat": rhat,
        "rhat_method": (
            "rank_normalized_split"
            if rank_normalized_rhat is not None
            else "gelman_rubin"
        ),
        "num_chains": num_chains,
        "num_warmup": int(result.extra["num_warmup"]),
        "num_samples_per_chain": int(result.extra["num_samples"]),
        "target_accept_prob": float(result.extra["target_accept_prob"]),
        "max_tree_depth": max_tree_depth,
        "chain_method": result.extra.get("chain_method", "unknown"),
    }


def _save_nuts_diagnostics(outdir: Path, result: LikelihoodRunResult) -> None:
    """Persist chain structure and NUTS diagnostics alongside flat samples."""
    grouped = result.extra.get("samples_grouped")
    if not grouped:
        return

    np.savez(
        outdir / "samples_grouped.npz",
        **{name: np.asarray(values) for name, values in grouped.items()},
    )
    sample_stats = {
        name: np.asarray(result.extra[name])
        for name in _NUTS_SAMPLE_STATS
        if name in result.extra
    }
    if sample_stats:
        np.savez(outdir / "sample_stats.npz", **sample_stats)

    summary = _nuts_diagnostics_summary(result)
    (outdir / "nuts_diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )


def _save_lightweight_nuts_diagnostics(
    outdir: Path, result: LikelihoodRunResult
) -> None:
    """Persist convergence summaries without retaining chains or samples."""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "nuts_diagnostics.json").write_text(
        json.dumps(_nuts_diagnostics_summary(result), indent=2, sort_keys=True)
        + "\n"
    )


def _maximum_rhat(
    samples_grouped: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    rhat = {}
    for name, values in samples_grouped.items():
        try:
            if rank_normalized_rhat is not None:
                value = np.asarray(
                    rank_normalized_rhat(np.asarray(values), method="rank")
                )
            else:
                value = np.asarray(gelman_rubin(jnp.asarray(values)))
        except Exception:
            continue
        finite = value[np.isfinite(value)]
        if finite.size:
            rhat[name] = float(np.max(finite))
    return rhat


def _require_nuts_convergence(
    label: str,
    result: LikelihoodRunResult,
    *,
    warn_rhat: float = 1.01,
    fail_rhat: float = 1.05,
) -> None:
    """Stop evidence estimation when multi-chain NUTS did not converge."""
    grouped = result.extra.get("samples_grouped")
    if not grouped:
        raise RuntimeError(f"NUTS [{label}] did not retain grouped chains.")
    num_chains = int(result.extra.get("num_chains", 0))
    if num_chains < 2:
        raise RuntimeError(
            f"NUTS [{label}] requires at least two chains; got {num_chains}."
        )
    rhat = _maximum_rhat(grouped)
    if not rhat:
        raise RuntimeError(f"NUTS [{label}] produced no finite R-hat values.")
    maximum = max(rhat.values())
    divergences = int(np.sum(np.asarray(result.extra.get("diverging", 0))))
    if maximum > fail_rhat or divergences:
        raise RuntimeError(
            f"NUTS [{label}] failed convergence: max R-hat={maximum:.4f}, "
            f"divergences={divergences}. lnZ was not computed."
        )
    if maximum > warn_rhat:
        print(
            f"[sampling] WARNING: NUTS [{label}] max R-hat={maximum:.4f} "
            f"exceeds the publication target {warn_rhat:.2f}."
        )


def _ess_summary(
    samples_grouped: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    for name, values in samples_grouped.items():
        try:
            ess = effective_sample_size(jnp.asarray(values))
        except Exception:
            continue
        ess_arr = np.asarray(ess).ravel()
        ess_arr = ess_arr[np.isfinite(ess_arr)]
        if ess_arr.size == 0:
            continue
        summaries[name] = {
            "min": float(np.min(ess_arr)),
            "median": float(np.median(ess_arr)),
        }
    return summaries


def _report_effective_sample_sizes(
    label: str, samples_grouped: Optional[Mapping[str, np.ndarray]]
) -> None:
    if not samples_grouped:
        return
    summaries = _ess_summary(samples_grouped)
    if not summaries:
        return
    print(f"ESS [{label}] (min/median):")
    for name, stats in summaries.items():
        print(f"  {name}: {stats['min']:.1f} / {stats['median']:.1f}")


def run_starccato_analysis(
    detectors: Sequence[str],
    outdir: str,
    *,
    bundle_paths: Optional[Mapping[str, str]] = None,
    trigger_time: Optional[float] = None,
    model_types: Sequence[str] = MODELS,
    sampler: str = "nuts",
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 2,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    num_live_points: int = 500,
    max_samples: int = 20000,
    latent_sigma: float | Sequence[float] = 1.0,
    log_amp_sigma: float = 1.0,
    map_initialization: bool = True,
    map_num_starts: int = 128,
    map_maxiter: int = 400,
    rng_seed: int = 0,
    extrinsic_params: Optional[Dict[str, float]] = None,
    save_artifacts: bool = True,
) -> Dict[str, LikelihoodRunResult]:
    """
    Run Starccato analysis using the JIM transient likelihood.

    Parameters
    ----------
    detectors
        Sequence of detector identifiers (e.g., ``["H1", "L1"]``).
    outdir
        Output directory for artifacts and summary files.
    bundle_paths
        Optional mapping from detector name to Starccato analysis bundle path.
    trigger_time
        GPS trigger time for data acquisition (required when bundles are not
        provided).
    model_types
        Model identifiers to analyse (defaults to all Starccato models).
    sampler
        ``"nuts"`` (default) or ``"nested"`` to choose the inference engine.
    num_samples, num_warmup, num_chains, target_accept_prob, max_tree_depth
        NumPyro NUTS configuration options (ignored for nested sampling).
    num_live_points, max_samples
        Nested sampling configuration (used only when ``sampler == "nested"``).
    latent_sigma, log_amp_sigma
        Prior standard deviations for latent coordinates and ``log_amp``.
    map_initialization, map_num_starts, map_maxiter
        Use the fast Sobol multistart MAP screen before NUTS and configure its
        broad-start count and per-start optimizer iteration cap.
    rng_seed
        Base seed used to generate independent PRNG keys per model.
    extrinsic_params
        Optional dictionary overriding default extrinsic parameters. ``gmst``
        is inferred from data preparation unless explicitly supplied.
    save_artifacts
        Whether to serialise samples and summary metadata.
    """
    if sampler.lower() == "nuts" and num_chains < 2:
        raise ValueError("NUTS analyses require at least two chains.")
    detector_names = [det.upper() for det in detectors]
    bundle_map = _normalise_bundle_paths(bundle_paths)

    prepared = prepare_multi_detector_data(
        detector_names,
        trigger_time=trigger_time,
        bundle_paths=bundle_map,
    )

    base_outdir = Path(outdir)
    if save_artifacts:
        _ensure_outdir(base_outdir)
        plot_data_overview(prepared, base_outdir / "data_overview.png")

    detector_data = prepared.detector_data[next(iter(prepared.detector_data))]
    sample_rate = 1.0 / detector_data.dt

    extrinsics = dict(DEFAULT_EXTRINSICS)
    if extrinsic_params:
        extrinsics.update(extrinsic_params)
    extrinsics.setdefault("gmst", prepared.gmst)
    extrinsics.setdefault("trigger_time", prepared.trigger_time)

    results: Dict[str, LikelihoodRunResult] = {}
    rng_master = jax.random.PRNGKey(rng_seed)

    print("=" * 60)
    print("STARCCATO JIM ANALYSIS")
    print(f"Detectors: {detector_names}")
    print(f"Models: {[m.upper() for m in model_types]}")
    print(f"Sampler: {sampler.upper()}")
    print("=" * 60)

    for model_index, model_type in enumerate(model_types):
        model = get_model(model_type)
        waveform = StarccatoJimWaveform(
            model=model, sample_rate=sample_rate, window=prepared.window
        )
        latent_names = waveform.latent_names

        rng_key = jax.random.fold_in(rng_master, model_index)
        likelihood = build_transient_likelihood(
            prepared.detectors,
            waveform,
            trigger_time=prepared.trigger_time,
            duration=prepared.duration,
            post_trigger_duration=prepared.post_trigger_duration,
        )

        model_label = model_type.upper()
        print(
            f"\nRunning inference for {model_label} "
            f"(latent_dim={len(latent_names)})"
        )
        map_result = None
        if sampler.lower() == "nuts":
            init_strategy = None
            init_params = None
            chain_method = "parallel"
            if map_initialization:
                map_result = find_multistart_map(
                    likelihood,
                    latent_names=latent_names,
                    fixed_params=extrinsics,
                    rng_key=jax.random.fold_in(rng_key, 10_000),
                    latent_sigma=latent_sigma,
                    log_amp_sigma=log_amp_sigma,
                    num_starts=map_num_starts,
                    start_design="sobol",
                    log_amp_starts=(-2.0, 0.0, 2.0, 4.0, 6.0),
                    maxiter=map_maxiter,
                    num_refine_candidates=4,
                    refine_starts_per_candidate=3,
                    refine_scale=0.15,
                    basin_radius=0.35,
                )
                _report_map_initialization(model_type, map_result)
                init_strategy, init_params, chain_method = (
                    _map_nuts_initialization(
                        map_result,
                        [*latent_names, "log_amp"],
                        num_chains,
                        chain_method=(
                            "sequential"
                            if model_type.lower() == "blip"
                            else "vectorized"
                        ),
                    )
                )
            run_result = run_numpyro_sampling(
                likelihood,
                latent_names=latent_names,
                fixed_params=extrinsics,
                rng_key=rng_key,
                latent_sigma=latent_sigma,
                log_amp_sigma=log_amp_sigma,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                chain_method=chain_method,
                init_strategy=init_strategy,
                init_params=init_params,
            )
        elif sampler.lower() == "nested":
            run_result = run_nested_sampling(
                likelihood,
                latent_names=latent_names,
                fixed_params=extrinsics,
                rng_key=rng_key,
                latent_sigma=latent_sigma,
                log_amp_sigma=log_amp_sigma,
                num_live_points=num_live_points,
                max_samples=max_samples,
                num_posterior_samples=num_samples,
            )
        else:  # pragma: no cover - guarded by CLI
            raise ValueError(
                f"Unknown sampler '{sampler}'. Use 'nuts' or 'nested'."
            )

        results[model_type] = run_result

        if save_artifacts:
            model_outdir = base_outdir / model_type.lower()
            _ensure_outdir(model_outdir)

            posterior_mean_dict = posterior_means(
                run_result.samples, list(run_result.samples.keys())
            )
            summary = {
                "detectors": detector_names,
                "model": model_type,
                "sampler": sampler,
                "latent_dim": len(latent_names),
                "logZ": run_result.logZ,
                "logZ_err": run_result.logZ_err,
                "runtime_seconds": run_result.runtime,
                "posterior_means": posterior_mean_dict,
                "trigger_time": prepared.trigger_time,
                "df": prepared.df,
                "duration": prepared.duration,
                "gmst": prepared.gmst,
                "roll_off": prepared.roll_off,
                "num_samples": num_samples,
                "num_warmup": num_warmup,
                "num_chains": num_chains,
                "target_accept_prob": target_accept_prob,
                "max_tree_depth": max_tree_depth,
            }
            _write_summary_json(model_outdir, summary)
            if map_result is not None:
                _save_map_initialization(model_outdir, map_result)

            sample_payload = {
                name: np.asarray(arr)
                for name, arr in run_result.samples.items()
            }
            if isinstance(run_result.extra, dict):
                if "log_weights" in run_result.extra:
                    sample_payload["log_weights"] = np.asarray(
                        run_result.extra["log_weights"]
                    )
                if "weights" in run_result.extra:
                    sample_payload["weights"] = np.asarray(
                        run_result.extra["weights"]
                    )
            np.savez(model_outdir / "samples.npz", **sample_payload)
            _save_nuts_diagnostics(model_outdir, run_result)

            mean_params = {
                **extrinsics,
                **{
                    k: v
                    for k, v in posterior_mean_dict.items()
                    if k in latent_names or k == "log_amp"
                },
            }
            plot_posterior_mean_waveform(
                prepared,
                waveform,
                mean_params,
                model_outdir / "posterior_mean_waveform.png",
            )
        if sampler.lower() == "nuts":
            _require_nuts_convergence(model_type, run_result)

    print("\nAnalysis complete.")
    print("=" * 60)
    return results


def run_bcr_posteriors(
    detectors: Sequence[str],
    outdir: str,
    *,
    bundle_paths: Optional[Mapping[str, str]] = None,
    trigger_time: Optional[float] = None,
    signal_model: str = "ccsne",
    glitch_model: str = "blip",
    extrinsic_params: Optional[Dict[str, float]] = None,
    latent_sigma_signal: float | Sequence[float] = 1.0,
    log_amp_sigma_signal: float = 5.0,
    latent_sigma_glitch: float | Sequence[float] = 1.0,
    log_amp_sigma_glitch: float = 5.0,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 2,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    map_initialization: bool = True,
    map_num_starts: int = 128,
    map_maxiter: int = 400,
    rng_seed: int = 0,
    save_artifacts: bool = True,
    save_diagnostics: bool = False,
    ci: tuple[int, int] = (5, 95),
    alpha: float = 1.0,
    beta: float = 0.5,
    lnz_method: str = "morph",
    nested_num_live_points: int = 500,
    nested_max_samples: int = 20000,
    nested_num_posterior_samples: Optional[int] = None,
    flow: Optional[float] = None,
    fmax: Optional[float] = None,
    noise_scale_marginal: bool = False,
    nsm_a: float = 100.0,
    nsm_b: Optional[float] = None,
    verify_logz_threshold: Optional[float] = 50.0,
) -> Dict[str, Dict[str, float]]:
    if lnz_method.lower() != "nested" and num_chains < 2:
        raise ValueError("NUTS BCR analyses require at least two chains.")
    detector_names = [det.upper() for det in detectors]
    bundle_map = _normalise_bundle_paths(bundle_paths)
    nsm_kwargs = dict(
        noise_scale_marginal=noise_scale_marginal, nsm_a=nsm_a, nsm_b=nsm_b
    )

    prepare_kwargs = {}
    if flow is not None:
        prepare_kwargs["flow"] = flow
    if fmax is not None:
        prepare_kwargs["fmax"] = fmax
    prepared = prepare_multi_detector_data(
        detector_names,
        trigger_time=trigger_time,
        bundle_paths=bundle_map,
        **prepare_kwargs,
    )

    base_outdir = Path(outdir)
    if save_artifacts:
        _ensure_outdir(base_outdir)
        plot_data_overview(prepared, base_outdir / "data_overview.png")

    reference_data = prepared.detector_data[next(iter(prepared.detector_data))]
    sample_rate = 1.0 / reference_data.dt

    extrinsics = dict(DEFAULT_EXTRINSICS)
    if extrinsic_params:
        extrinsics.update(extrinsic_params)
    extrinsics.setdefault("gmst", prepared.gmst)
    extrinsics.setdefault("trigger_time", prepared.trigger_time)

    lnz_method = lnz_method.lower()
    if lnz_method not in {"morph", "nested"}:
        raise ValueError("lnz_method must be 'morph' or 'nested'.")

    use_nested = lnz_method == "nested"

    rng_master = jax.random.PRNGKey(rng_seed)

    results: Dict[str, Dict[str, float]] = {
        "signal": {},
        "glitch": {},
        "noise": {},
        "nuts_diagnostics": {},
        "map_initialization": {},
    }

    # Coherent signal run
    signal_model_obj = get_model(signal_model)
    signal_waveform = StarccatoJimWaveform(
        model=signal_model_obj,
        sample_rate=sample_rate,
        window=prepared.window,
    )
    signal_latent_names = signal_waveform.latent_names
    signal_likelihood = build_transient_likelihood(
        prepared.detectors,
        signal_waveform,
        trigger_time=prepared.trigger_time,
        duration=prepared.duration,
        post_trigger_duration=prepared.post_trigger_duration,
    )

    signal_rng = jax.random.fold_in(rng_master, 0)
    signal_map_result = None
    if use_nested:
        signal_result = run_nested_sampling(
            signal_likelihood,
            latent_names=signal_latent_names,
            fixed_params=extrinsics,
            rng_key=signal_rng,
            latent_sigma=latent_sigma_signal,
            log_amp_sigma=log_amp_sigma_signal,
            num_live_points=nested_num_live_points,
            max_samples=nested_max_samples,
            num_posterior_samples=nested_num_posterior_samples or num_samples,
            **nsm_kwargs,
        )
    else:
        signal_init_strategy = None
        signal_init_params = None
        signal_chain_method = "parallel"
        if map_initialization:
            signal_map_result = find_multistart_map(
                signal_likelihood,
                latent_names=signal_latent_names,
                fixed_params=extrinsics,
                rng_key=jax.random.fold_in(signal_rng, 10_000),
                latent_sigma=latent_sigma_signal,
                log_amp_sigma=log_amp_sigma_signal,
                num_starts=map_num_starts,
                start_design="sobol",
                log_amp_starts=(-2.0, 0.0, 2.0, 4.0, 6.0),
                maxiter=map_maxiter,
                num_refine_candidates=4,
                refine_starts_per_candidate=3,
                refine_scale=0.15,
                basin_radius=0.35,
                **nsm_kwargs,
            )
            _report_map_initialization("signal", signal_map_result)
            (
                signal_init_strategy,
                signal_init_params,
                signal_chain_method,
            ) = _map_nuts_initialization(
                signal_map_result,
                [*signal_latent_names, "log_amp"],
                num_chains,
                chain_method=(
                    "sequential"
                    if signal_model.lower() == "blip"
                    else "vectorized"
                ),
            )
        signal_result = run_numpyro_sampling(
            signal_likelihood,
            latent_names=signal_latent_names,
            fixed_params=extrinsics,
            rng_key=signal_rng,
            latent_sigma=latent_sigma_signal,
            log_amp_sigma=log_amp_sigma_signal,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            chain_method=signal_chain_method,
            init_strategy=signal_init_strategy,
            init_params=signal_init_params,
            **nsm_kwargs,
        )

    _report_effective_sample_sizes(
        "signal", signal_result.extra.get("samples_grouped")
    )

    signal_dir = base_outdir / "signal"
    if save_artifacts:
        _ensure_outdir(signal_dir)
        np.savez(
            signal_dir / "samples.npz",
            **{k: np.asarray(v) for k, v in signal_result.samples.items()},
        )
        _save_nuts_diagnostics(signal_dir, signal_result)
        if signal_map_result is not None:
            _save_map_initialization(signal_dir, signal_map_result)
        plot_posterior_predictive_from_samples(
            prepared.detectors,
            prepared.detector_data,
            signal_waveform,
            signal_latent_names,
            signal_result.samples,
            extrinsics,
            signal_dir / "posterior_predictive.png",
            n_draws=min(200, num_samples),
            ci=ci,
            title_prefix="Signal",
        )
    elif save_diagnostics and not use_nested:
        _save_lightweight_nuts_diagnostics(signal_dir, signal_result)
        if signal_map_result is not None:
            (signal_dir / "map_initialization.json").write_text(
                json.dumps(
                    _brief_map_initialization_summary(signal_map_result),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

    if not use_nested:
        results["nuts_diagnostics"]["signal"] = _nuts_diagnostics_summary(
            signal_result
        )
        if signal_map_result is not None:
            results["map_initialization"]["signal"] = (
                _brief_map_initialization_summary(signal_map_result)
            )
        _require_nuts_convergence("signal", signal_result)

    evidence_status: Dict[str, Dict[str, object]] = {}
    if use_nested:
        signal_logZ = signal_result.logZ
        signal_logZ_err = signal_result.logZ_err
        evidence_status["signal"] = {"method": "nested", "status": "ok"}
    else:
        signal_model_callable, _, _ = _build_numpyro_model(
            signal_likelihood,
            signal_latent_names,
            latent_sigma_signal,
            log_amp_sigma_signal,
            extrinsics,
            **nsm_kwargs,
        )
        signal_fallback = lambda: _nested_evidence(
            signal_likelihood,
            signal_latent_names,
            latent_sigma_signal,
            log_amp_sigma_signal,
            extrinsics,
            jax.random.fold_in(signal_rng, 1000),
            num_live_points=nested_num_live_points,
            max_samples=nested_max_samples,
            num_posterior_samples=nested_num_posterior_samples or num_samples,
            **nsm_kwargs,
        )
        signal_evidence = _compute_morphz_evidence(
            signal_result.samples,
            signal_result.extra.get("log_posterior"),
            signal_latent_names,
            True,
            signal_model_callable,
            extrinsics,
            signal_dir if save_artifacts else base_outdir,
            "signal",
            fallback=signal_fallback,
            log_amp_sigma=log_amp_sigma_signal,
            verify_logz_threshold=verify_logz_threshold,
        )
        signal_logZ = signal_evidence.logZ
        signal_logZ_err = signal_evidence.logZ_err
        evidence_status["signal"] = {
            "method": signal_evidence.method,
            "status": signal_evidence.status,
            "n_attempts": signal_evidence.n_attempts,
        }
    results["signal"]["logZ"] = signal_logZ
    results["signal"]["logZ_err"] = signal_logZ_err

    # Glitch per detector
    glitch_model_obj = get_model(glitch_model)
    results.setdefault("glitch_err", {})
    for det_index, det in enumerate(prepared.detectors, start=1):
        glitch_waveform = StarccatoGlitchWaveform(
            model=glitch_model_obj,
            sample_rate=sample_rate,
            window=prepared.window,
            strain_scale=1e-21,
        )
        glitch_latent_names = glitch_waveform.latent_names
        det_glitch = _clone_no_response_detector(det)
        det_glitch.set_frequency_bounds(*det.frequency_bounds)

        glitch_likelihood = build_transient_likelihood(
            [det_glitch],
            glitch_waveform,
            trigger_time=prepared.trigger_time,
            duration=prepared.duration,
            post_trigger_duration=prepared.post_trigger_duration,
        )

        glitch_rng = jax.random.fold_in(rng_master, det_index)
        glitch_map_result = None
        if use_nested:
            glitch_result = run_nested_sampling(
                glitch_likelihood,
                latent_names=glitch_latent_names,
                fixed_params={},
                rng_key=glitch_rng,
                latent_sigma=latent_sigma_glitch,
                log_amp_sigma=log_amp_sigma_glitch,
                num_live_points=nested_num_live_points,
                max_samples=nested_max_samples,
                num_posterior_samples=nested_num_posterior_samples
                or num_samples,
                **nsm_kwargs,
            )
        else:
            glitch_init_strategy = None
            glitch_init_params = None
            glitch_chain_method = "parallel"
            if map_initialization:
                glitch_map_result = find_multistart_map(
                    glitch_likelihood,
                    latent_names=glitch_latent_names,
                    fixed_params={},
                    rng_key=jax.random.fold_in(glitch_rng, 10_000),
                    latent_sigma=latent_sigma_glitch,
                    log_amp_sigma=log_amp_sigma_glitch,
                    num_starts=map_num_starts,
                    start_design="sobol",
                    log_amp_starts=(-2.0, 0.0, 2.0, 4.0, 6.0),
                    maxiter=map_maxiter,
                    num_refine_candidates=4,
                    refine_starts_per_candidate=3,
                    refine_scale=0.15,
                    basin_radius=0.35,
                    **nsm_kwargs,
                )
                _report_map_initialization(
                    f"glitch {det.name}", glitch_map_result
                )
                (
                    glitch_init_strategy,
                    glitch_init_params,
                    glitch_chain_method,
                ) = _map_nuts_initialization(
                    glitch_map_result,
                    [*glitch_latent_names, "log_amp"],
                    num_chains,
                    chain_method="sequential",
                )
            glitch_result = run_numpyro_sampling(
                glitch_likelihood,
                latent_names=glitch_latent_names,
                fixed_params={},
                rng_key=glitch_rng,
                latent_sigma=latent_sigma_glitch,
                log_amp_sigma=log_amp_sigma_glitch,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                chain_method=glitch_chain_method,
                init_strategy=glitch_init_strategy,
                init_params=glitch_init_params,
                **nsm_kwargs,
            )

        _report_effective_sample_sizes(
            f"glitch {det.name}", glitch_result.extra.get("samples_grouped")
        )

        glitch_dir = base_outdir / f"glitch_{det.name.lower()}"
        if save_artifacts:
            _ensure_outdir(glitch_dir)
            np.savez(
                glitch_dir / "samples.npz",
                **{k: np.asarray(v) for k, v in glitch_result.samples.items()},
            )
            _save_nuts_diagnostics(glitch_dir, glitch_result)
            if glitch_map_result is not None:
                _save_map_initialization(glitch_dir, glitch_map_result)
            plot_posterior_predictive_from_samples(
                [det_glitch],
                {det.name: prepared.detector_data[det.name]},
                glitch_waveform,
                glitch_latent_names,
                glitch_result.samples,
                {},
                glitch_dir / "posterior_predictive.png",
                n_draws=min(200, num_samples),
                ci=ci,
                title_prefix=f"Glitch {det.name}",
            )
        elif save_diagnostics and not use_nested:
            _save_lightweight_nuts_diagnostics(glitch_dir, glitch_result)
            if glitch_map_result is not None:
                (glitch_dir / "map_initialization.json").write_text(
                    json.dumps(
                        _brief_map_initialization_summary(glitch_map_result),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )

        if not use_nested:
            results["nuts_diagnostics"][f"glitch_{det.name}"] = (
                _nuts_diagnostics_summary(glitch_result)
            )
            if glitch_map_result is not None:
                results["map_initialization"][f"glitch_{det.name}"] = (
                    _brief_map_initialization_summary(glitch_map_result)
                )
            _require_nuts_convergence(f"glitch {det.name}", glitch_result)

        if use_nested:
            glitch_logZ = glitch_result.logZ
            glitch_logZ_err = glitch_result.logZ_err
            evidence_status[f"glitch_{det.name}"] = {
                "method": "nested",
                "status": "ok",
            }
        else:
            glitch_model_callable, _, _ = _build_numpyro_model(
                glitch_likelihood,
                glitch_latent_names,
                latent_sigma_glitch,
                log_amp_sigma_glitch,
                {},
                **nsm_kwargs,
            )
            glitch_fallback = lambda: _nested_evidence(
                glitch_likelihood,
                glitch_latent_names,
                latent_sigma_glitch,
                log_amp_sigma_glitch,
                {},
                jax.random.fold_in(glitch_rng, 1000),
                num_live_points=nested_num_live_points,
                max_samples=nested_max_samples,
                num_posterior_samples=nested_num_posterior_samples
                or num_samples,
                **nsm_kwargs,
            )
            glitch_evidence = _compute_morphz_evidence(
                glitch_result.samples,
                glitch_result.extra.get("log_posterior"),
                glitch_latent_names,
                True,
                glitch_model_callable,
                {},
                glitch_dir if save_artifacts else base_outdir,
                f"glitch_{det.name.lower()}",
                fallback=glitch_fallback,
                log_amp_sigma=log_amp_sigma_glitch,
                verify_logz_threshold=verify_logz_threshold,
            )
            glitch_logZ = glitch_evidence.logZ
            glitch_logZ_err = glitch_evidence.logZ_err
            evidence_status[f"glitch_{det.name}"] = {
                "method": glitch_evidence.method,
                "status": glitch_evidence.status,
                "n_attempts": glitch_evidence.n_attempts,
            }
        results["glitch"][det.name] = glitch_logZ
        results["glitch_err"][det.name] = glitch_logZ_err

    # Noise evidences
    for det in prepared.detectors:
        results["noise"][det.name] = _analytic_noise_logz(det)

    logZ_signal = results["signal"].get("logZ", float("nan"))
    logZ_glitch = results.get("glitch", {})
    logZ_noise = results["noise"]
    log_bcr = _compute_log_bcr(
        logZ_signal, logZ_glitch, logZ_noise, alpha, beta
    )
    results["bcr_log"] = log_bcr
    results["bcr"] = (
        float(np.exp(log_bcr)) if np.isfinite(log_bcr) else float("nan")
    )

    n_failed = sum(
        1 for s in evidence_status.values() if s.get("status") == "failed"
    )
    n_fallback = sum(
        1 for s in evidence_status.values() if s.get("status") == "fallback"
    )
    results["evidence_status"] = evidence_status
    results["evidence_failures"] = n_failed
    results["evidence_fallbacks"] = n_fallback
    if n_failed:
        print(
            f"[evidence] WARNING: {n_failed} evidence term(s) failed; BCR may be NaN."
        )

    if save_artifacts:
        summary = {
            "logZ_signal": results["signal"].get("logZ", float("nan")),
            "logZ_signal_err": results["signal"].get("logZ_err", float("nan")),
            "logZ_glitch": results.get("glitch", {}),
            "logZ_glitch_err": results.get("glitch_err", {}),
            "logZ_noise": results["noise"],
            "log_bcr": log_bcr,
            "bcr": results["bcr"],
            "alpha": alpha,
            "beta": beta,
            "lnz_method": lnz_method,
            "evidence_status": evidence_status,
            "evidence_failures": n_failed,
            "evidence_fallbacks": n_fallback,
            "nuts_diagnostics": results.get("nuts_diagnostics", {}),
            "map_initialization": results.get("map_initialization", {}),
        }
        _write_summary_json(base_outdir, summary)

    return results
