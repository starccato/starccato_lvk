from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from starccato_lvk.analysis import jim_likelihood
from starccato_lvk.analysis import main as analysis_main


class _FakeMCMC:
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel
        self.kwargs = kwargs
        self.extra_fields = None

    def run(self, rng_key, *, extra_fields, init_params=None):
        self.extra_fields = extra_fields
        self.init_params = init_params

    def get_samples(self, group_by_chain=False):
        if group_by_chain:
            return {"z0": jnp.zeros((1, 2)), "log_amp": jnp.zeros((1, 2))}
        return {"z0": jnp.zeros(2), "log_amp": jnp.zeros(2)}

    def get_extra_fields(self):
        return {name: np.zeros(2) for name in self.extra_fields}


def test_run_numpyro_sampling_forwards_nuts_controls_and_diagnostics(
    monkeypatch,
):
    nuts_kwargs = {}
    mcmc_instances = []

    def fake_build_model(*args, **kwargs):
        return object(), np.ones(1), 5.0

    def fake_nuts(model, **kwargs):
        nuts_kwargs.update(kwargs)
        return object()

    def fake_mcmc(kernel, **kwargs):
        instance = _FakeMCMC(kernel, **kwargs)
        mcmc_instances.append(instance)
        return instance

    monkeypatch.setattr(
        jim_likelihood, "_build_numpyro_model", fake_build_model
    )
    monkeypatch.setattr(jim_likelihood, "NUTS", fake_nuts)
    monkeypatch.setattr(jim_likelihood, "MCMC", fake_mcmc)
    monkeypatch.setattr(
        jim_likelihood,
        "build_log_density_fn",
        lambda model, kwargs: lambda params: jnp.asarray(0.0),
    )

    result = jim_likelihood.run_numpyro_sampling(
        object(),
        latent_names=["z0"],
        fixed_params={},
        rng_key=jax.random.PRNGKey(0),
        target_accept_prob=0.95,
        max_tree_depth=12,
        chain_method="vectorized",
        init_params={"z0": jnp.zeros(1), "log_amp": jnp.zeros(1)},
        progress_bar=False,
    )

    assert nuts_kwargs["target_accept_prob"] == 0.95
    assert nuts_kwargs["max_tree_depth"] == 12
    assert result.extra["target_accept_prob"] == 0.95
    assert result.extra["max_tree_depth"] == 12
    assert result.extra["chain_method"] == "vectorized"
    assert mcmc_instances[0].kwargs["chain_method"] == "vectorized"
    assert mcmc_instances[0].init_params is not None
    assert set(mcmc_instances[0].extra_fields) == {
        "diverging",
        "accept_prob",
        "num_steps",
        "energy",
        "potential_energy",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_accept_prob": 1.0}, "target_accept_prob"),
        ({"target_accept_prob": 0.0}, "target_accept_prob"),
        ({"max_tree_depth": 0}, "max_tree_depth"),
        ({"chain_method": "vmap"}, "chain_method"),
    ],
)
def test_run_numpyro_sampling_rejects_invalid_nuts_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        jim_likelihood.run_numpyro_sampling(
            object(),
            latent_names=["z0"],
            fixed_params={},
            rng_key=jax.random.PRNGKey(0),
            progress_bar=False,
            **kwargs,
        )


def test_run_numpyro_sampling_rejects_two_initialization_interfaces():
    with pytest.raises(ValueError, match="only one"):
        jim_likelihood.run_numpyro_sampling(
            object(),
            latent_names=["z0"],
            fixed_params={},
            rng_key=jax.random.PRNGKey(0),
            init_strategy=object(),
            init_params={"z0": jnp.zeros(2), "log_amp": jnp.zeros(2)},
            progress_bar=False,
        )


def test_find_multistart_map_recovers_quadratic_mode(monkeypatch):
    monkeypatch.setattr(
        jim_likelihood,
        "_build_numpyro_model",
        lambda *args, **kwargs: (object(), np.ones(1), 5.0),
    )

    def fake_log_density(model, kwargs):
        def logpost(params):
            return -(
                (params["z0"] - 0.75) ** 2 + (params["log_amp"] - 2.5) ** 2
            )

        return logpost

    monkeypatch.setattr(
        jim_likelihood, "build_log_density_fn", fake_log_density
    )
    result = jim_likelihood.find_multistart_map(
        object(),
        latent_names=["z0"],
        fixed_params={},
        rng_key=jax.random.PRNGKey(2),
        num_starts=3,
        maxiter=100,
    )

    assert result.values["z0"] == pytest.approx(0.75, abs=1e-5)
    assert result.values["log_amp"] == pytest.approx(2.5, abs=1e-5)
    assert result.log_density == pytest.approx(0.0, abs=1e-8)
    assert len(result.attempts) == 3
    assert result.runtime_seconds >= 0.0


def test_find_multistart_map_refines_and_reports_reproducible_basin(
    monkeypatch,
):
    monkeypatch.setattr(
        jim_likelihood,
        "_build_numpyro_model",
        lambda *args, **kwargs: (object(), np.ones(1), 5.0),
    )

    def fake_log_density(model, kwargs):
        def logpost(params):
            return -(
                (params["z0"] - 0.75) ** 2 + (params["log_amp"] - 2.5) ** 2
            )

        return logpost

    monkeypatch.setattr(
        jim_likelihood, "build_log_density_fn", fake_log_density
    )
    result = jim_likelihood.find_multistart_map(
        object(),
        latent_names=["z0"],
        fixed_params={},
        rng_key=jax.random.PRNGKey(2),
        num_starts=4,
        num_refine_candidates=1,
        refine_starts_per_candidate=3,
        maxiter=100,
    )

    assert len(result.attempts) == 7
    assert result.best_basin_broad_hits == 4
    assert result.best_basin_refinement_hits == 3
    assert result.best_basin_reproduced
    assert result.next_basin_delta_log_density == np.inf
    assert result.basins[0]["total_hits"] == 7
    assert {attempt["stage"] for attempt in result.attempts} == {
        "broad",
        "refinement",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_refine_candidates": -1}, "non-negative"),
        ({"refine_starts_per_candidate": -1}, "non-negative"),
        ({"refine_scale": 0.0}, "refine_scale"),
        ({"basin_radius": 0.0}, "basin_radius"),
        ({"minimum_broad_hits": 0}, "minimum_broad_hits"),
        ({"start_design": "latin_hypercube"}, "start_design"),
    ],
)
def test_find_multistart_map_rejects_invalid_refinement_settings(
    monkeypatch, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        jim_likelihood.find_multistart_map(
            object(),
            latent_names=["z0"],
            fixed_params={},
            rng_key=jax.random.PRNGKey(0),
            **kwargs,
        )


def test_save_map_initialization_writes_json_safe_diagnostics(tmp_path):
    result = jim_likelihood.MAPInitializationResult(
        values={"z0": 0.5, "log_amp": 1.0},
        log_density=4.0,
        attempts=[],
        runtime_seconds=0.25,
        best_basin_broad_hits=2,
        best_basin_reproduced=True,
        next_basin_delta_log_density=np.inf,
    )

    analysis_main._save_map_initialization(tmp_path, result)

    payload = json.loads((tmp_path / "map_initialization.json").read_text())
    assert payload["method"] == "sobol_multistart_lbfgs"
    assert payload["next_basin_delta_log_density"] is None
    assert payload["best_basin_reproduced"] is True


def test_map_nuts_initialization_spreads_vectorized_chains_across_modes():
    basins = [
        {
            "values": {"z0": 0.1, "log_amp": 1.0},
            "delta_log_density": 0.0,
        },
        {
            "values": {"z0": 0.8, "log_amp": 2.0},
            "delta_log_density": 0.5,
        },
        {
            "values": {"z0": -2.0, "log_amp": -1.0},
            "delta_log_density": 10.0,
        },
    ]
    result = jim_likelihood.MAPInitializationResult(
        values=dict(basins[0]["values"]),
        log_density=4.0,
        attempts=[],
        basins=basins,
    )

    strategy, init_params, chain_method = (
        analysis_main._map_nuts_initialization(
            result, ["z0", "log_amp"], num_chains=4
        )
    )

    assert strategy is None
    assert chain_method == "vectorized"
    assert result.selected_chain_basin_ranks == [0, 1, 0, 1]
    np.testing.assert_allclose(init_params["z0"], [0.1, 0.8, 0.1, 0.8])
    np.testing.assert_allclose(init_params["log_amp"], [1.0, 2.0, 1.0, 2.0])


def _likelihood_result_with_chains(chains, *, divergences=0):
    chains = np.asarray(chains, dtype=float)
    return jim_likelihood.LikelihoodRunResult(
        samples={"z0": chains.reshape(-1)},
        logZ=np.nan,
        logZ_err=np.nan,
        runtime=0.1,
        extra={
            "samples_grouped": {"z0": chains},
            "num_chains": chains.shape[0],
            "diverging": np.asarray(
                [True] * divergences + [False] * (chains.size - divergences)
            ),
        },
    )


def test_require_nuts_convergence_accepts_mixed_chains():
    draws = np.random.default_rng(4).normal(size=(2, 1000))
    result = _likelihood_result_with_chains(draws)

    analysis_main._require_nuts_convergence("test", result)


def test_require_nuts_convergence_rejects_bad_rhat_before_evidence():
    draws = np.linspace(-1.0, 1.0, 200)
    result = _likelihood_result_with_chains([draws, draws + 10.0])

    with pytest.raises(RuntimeError, match="lnZ was not computed"):
        analysis_main._require_nuts_convergence("test", result)


def test_require_nuts_convergence_rejects_divergences():
    draws = np.random.default_rng(5).normal(size=(2, 1000))
    result = _likelihood_result_with_chains(draws, divergences=1)

    with pytest.raises(RuntimeError, match="divergences=1"):
        analysis_main._require_nuts_convergence("test", result)


def test_high_level_nuts_workflows_require_two_chains(tmp_path):
    with pytest.raises(ValueError, match="at least two chains"):
        analysis_main.run_starccato_analysis(
            ["L1"], str(tmp_path), num_chains=1
        )
    with pytest.raises(ValueError, match="at least two chains"):
        analysis_main.run_bcr_posteriors(["L1"], str(tmp_path), num_chains=1)
