from __future__ import annotations

import json
import copy
import types
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import init_to_value
try:
    from morphZ import evidence as morphz_evidence
except ImportError:  # pragma: no cover - optional dependency
    morphz_evidence = None

from starccato_jax.waveforms import MODELS, get_model

from .jim_waveform import StarccatoJimWaveform, StarccatoGlitchWaveform
from .jim_likelihood import (
    LikelihoodRunResult,
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
    freqs = np.asarray(detector.sliced_frequencies)
    if freqs.size < 2:
        return float("nan")
    df = float(freqs[1] - freqs[0])
    data_fd = np.asarray(detector.sliced_fd_data)
    psd = np.asarray(detector.sliced_psd)
    inner = 4.0 * df * np.real(np.sum(np.conj(data_fd) * data_fd / psd))
    return float(-0.5 * inner)


def _compute_morphz_evidence(
    samples: Dict[str, np.ndarray],
    log_posterior: np.ndarray,
    latent_names: Sequence[str],
    include_log_amp: bool,
    model,
    fixed_params: Mapping[str, float],
    outdir: Path,
    label: str,
):
    if morphz_evidence is None or log_posterior is None:
        return float("nan"), float("nan")
    try:
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

        res = morphz_evidence(
            post_samples=post_samples,
            log_posterior_values=logpost_values,
            log_posterior_function=lp_fn,
            n_resamples=2000,
            morph_type="tree",
            kde_bw="isj",
            param_names=param_names,
            output_path=str(outdir / f"morphZ_{label}"),
            n_estimations=2,
            verbose=False,
        )
        if len(res) == 0:
            return float("nan"), float("nan")
        return float(res[0][0]), float(res[0][1])
    except Exception:  # pragma: no cover - gracefully handle morphZ issues
        return float("nan"), float("nan")


def _compute_log_bcr(logZ_signal: float, logZ_glitch: Mapping[str, float], logZ_noise: Mapping[str, float], alpha: float, beta: float) -> float:
    if not np.isfinite(logZ_signal):
        return float("nan")
    denom_terms = []
    for det_name, logZ_N in logZ_noise.items():
        logZ_G = logZ_glitch.get(det_name, float("-inf"))
        term_glitch = np.log(beta) + logZ_G
        term_noise = np.log1p(-beta) + logZ_N if beta != 1.0 else float("-inf")
        denom_terms.append(np.logaddexp(term_glitch, term_noise))
    return np.log(alpha) + logZ_signal - np.sum(denom_terms)


def _normalise_bundle_paths(bundle_paths: Optional[Mapping[str, str]]) -> Optional[Dict[str, Path]]:
    if bundle_paths is None:
        return None
    return {key.upper(): Path(value) for key, value in bundle_paths.items()}


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_summary_json(outdir: Path, summary: Dict) -> None:
    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


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
    num_chains: int = 1,
    num_live_points: int = 500,
    max_samples: int = 20000,
    latent_sigma: float | Sequence[float] = 1.0,
    log_amp_sigma: float = 1.0,
    rng_seed: int = 0,
    extrinsic_params: Optional[Dict[str, float]] = None,
    save_artifacts: bool = True,
) -> Dict[str, LikelihoodRunResult]:
    """
    Run Starccato analysis using the JIM transient likelihood for one or more detectors.

    Parameters
    ----------
    detectors
        Sequence of detector identifiers (e.g., ``["H1", "L1"]``).
    outdir
        Output directory for artifacts and summary files.
    bundle_paths
        Optional mapping from detector name to Starccato analysis bundle path.
    trigger_time
        GPS trigger time for data acquisition (required when ``bundle_paths`` are not provided).
    model_types
        Iterable of model identifiers to analyse (defaults to all available Starccato models).
    sampler
        ``"nuts"`` (default) or ``"nested"`` to choose the inference engine.
    num_samples, num_warmup, num_chains
        NumPyro NUTS configuration options (ignored for nested sampling).
    num_live_points, max_samples
        Nested sampling configuration (used only when ``sampler == "nested"``).
    latent_sigma, log_amp_sigma
        Prior standard deviations for latent coordinates and ``log_amp`` respectively.
    rng_seed
        Base seed used to generate independent PRNG keys per model.
    extrinsic_params
        Optional dictionary overriding default extrinsic parameters (``t_c``, ``ra``, ``dec``,
        ``psi``, ``luminosity_distance``). ``gmst`` is inferred from the data preparation step
        unless explicitly supplied.
    save_artifacts
        Whether to serialise samples and summary metadata.
    """
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
        waveform = StarccatoJimWaveform(model=model, sample_rate=sample_rate, window=prepared.window)
        latent_names = waveform.latent_names

        rng_key = jax.random.fold_in(rng_master, model_index)
        likelihood = build_transient_likelihood(
            prepared.detectors,
            waveform,
            trigger_time=prepared.trigger_time,
            duration=prepared.duration,
            post_trigger_duration=prepared.post_trigger_duration,
        )

        print(f"\nRunning inference for {model_type.upper()} (latent_dim={len(latent_names)})")
        if sampler.lower() == "nuts":
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
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'nuts' or 'nested'.")

        results[model_type] = run_result

        if save_artifacts:
            model_outdir = base_outdir / model_type.lower()
            _ensure_outdir(model_outdir)

            posterior_mean_dict = posterior_means(run_result.samples, list(run_result.samples.keys()))
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
            }
            _write_summary_json(model_outdir, summary)

            sample_payload = {name: np.asarray(arr) for name, arr in run_result.samples.items()}
            if isinstance(run_result.extra, dict):
                if "log_weights" in run_result.extra:
                    sample_payload["log_weights"] = np.asarray(run_result.extra["log_weights"])
                if "weights" in run_result.extra:
                    sample_payload["weights"] = np.asarray(run_result.extra["weights"])
            np.savez(model_outdir / "samples.npz", **sample_payload)

            mean_params = {**extrinsics, **{k: v for k, v in posterior_mean_dict.items() if k in latent_names or k == "log_amp"}}
            plot_posterior_mean_waveform(
                prepared,
                waveform,
                mean_params,
                model_outdir / "posterior_mean_waveform.png",
            )

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
    log_amp_sigma_signal: float = 1.0,
    latent_sigma_glitch: float | Sequence[float] = 1.0,
    log_amp_sigma_glitch: float = 1.0,
    num_samples: int = 1000,
    num_warmup: int = 500,
    num_chains: int = 1,
    rng_seed: int = 0,
    save_artifacts: bool = True,
    ci: tuple[int, int] = (5, 95),
    alpha: float = 1.0,
    beta: float = 0.5,
) -> Dict[str, Dict[str, float]]:
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

    reference_data = prepared.detector_data[next(iter(prepared.detector_data))]
    sample_rate = 1.0 / reference_data.dt

    extrinsics = dict(DEFAULT_EXTRINSICS)
    if extrinsic_params:
        extrinsics.update(extrinsic_params)
    extrinsics.setdefault("gmst", prepared.gmst)
    extrinsics.setdefault("trigger_time", prepared.trigger_time)

    rng_master = jax.random.PRNGKey(rng_seed)

    results: Dict[str, Dict[str, float]] = {
        "signal": {},
        "glitch": {},
        "noise": {},
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
    )

    signal_dir = base_outdir / "signal"
    if save_artifacts:
        _ensure_outdir(signal_dir)
        np.savez(signal_dir / "samples.npz", **{k: np.asarray(v) for k, v in signal_result.samples.items()})
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

    signal_model_callable, _, _ = _build_numpyro_model(
        signal_likelihood,
        signal_latent_names,
        latent_sigma_signal,
        log_amp_sigma_signal,
        extrinsics,
    )
    signal_logZ, signal_logZ_err = _compute_morphz_evidence(
        signal_result.samples,
        signal_result.extra.get("log_posterior"),
        signal_latent_names,
        True,
        signal_model_callable,
        extrinsics,
        signal_dir if save_artifacts else base_outdir,
        "signal",
    )
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
        init_values = {name: jnp.array(0.0) for name in glitch_latent_names}
        init_values["log_amp"] = jnp.array(0.0)
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
            init_strategy=init_to_value(values=init_values),
        )

        glitch_dir = base_outdir / f"glitch_{det.name.lower()}"
        if save_artifacts:
            _ensure_outdir(glitch_dir)
            np.savez(glitch_dir / "samples.npz", **{k: np.asarray(v) for k, v in glitch_result.samples.items()})
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

        glitch_model_callable, _, _ = _build_numpyro_model(
            glitch_likelihood,
            glitch_latent_names,
            latent_sigma_glitch,
            log_amp_sigma_glitch,
            {},
        )
        glitch_logZ, glitch_logZ_err = _compute_morphz_evidence(
            glitch_result.samples,
            glitch_result.extra.get("log_posterior"),
            glitch_latent_names,
            True,
            glitch_model_callable,
            {},
            glitch_dir if save_artifacts else base_outdir,
            f"glitch_{det.name.lower()}",
        )
        results["glitch"][det.name] = glitch_logZ
        results["glitch_err"][det.name] = glitch_logZ_err

    # Noise evidences
    for det in prepared.detectors:
        results["noise"][det.name] = _analytic_noise_logz(det)

    logZ_signal = results["signal"].get("logZ", float("nan"))
    logZ_glitch = results.get("glitch", {})
    logZ_noise = results["noise"]
    log_bcr = _compute_log_bcr(logZ_signal, logZ_glitch, logZ_noise, alpha, beta)
    results["bcr_log"] = log_bcr
    results["bcr"] = float(np.exp(log_bcr)) if np.isfinite(log_bcr) else float("nan")

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
        }
        _write_summary_json(base_outdir, summary)

    return results
