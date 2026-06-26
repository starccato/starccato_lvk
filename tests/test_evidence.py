"""Fast checks for the hardened morphZ evidence wrapper.

These run a tiny linear-Gaussian model (closed-form ``log Z``) so they are quick
yet still exercise the real morphZ bridge sampler, the retry/finite guard, and
the nested-sampling fallback path.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_uniform
from numpyro.infer.util import log_density

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

from starccato_lvk.analysis.evidence import (
    EvidenceResult,
    evidence_with_fallback,
    morphz_log_evidence,
)


def build_log_density_fn(model):
    """Local jitted joint log-density (avoids importing the GW waveform stack)."""

    def _logpost(params):
        lp, _ = log_density(model, (), {}, params)
        return lp

    return jax.jit(_logpost)


def _analytic_logz(data, M, sigma_p, sigma_n):
    n = data.shape[0]
    C = sigma_p**2 * (M @ M.T) + sigma_n**2 * np.eye(n)
    _, logdet = np.linalg.slogdet(C)
    quad = float(data @ np.linalg.solve(C, data))
    return float(-0.5 * (n * np.log(2 * np.pi) + logdet + quad))


def _toy_problem(seed=0, dim=4, n_data=32, sigma_p=1.0, sigma_n=1.0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n_data, dim)) / np.sqrt(dim)
    theta_true = rng.standard_normal(dim) * sigma_p
    data = M @ theta_true + rng.standard_normal(n_data) * sigma_n
    logz_true = _analytic_logz(data, M, sigma_p, sigma_n)

    data_j, M_j = jnp.asarray(data), jnp.asarray(M)
    n = n_data
    const = -0.5 * (n * np.log(2 * np.pi) + n * np.log(sigma_n**2))

    def model():
        theta = jnp.stack(
            [numpyro.sample(f"theta_{i}", dist.Normal(0.0, sigma_p)) for i in range(dim)]
        )
        resid = data_j - M_j @ theta
        numpyro.factor("ll", const - 0.5 / sigma_n**2 * jnp.sum(resid**2))

    return model, logz_true, dim


def _run_nuts(model, dim, seed=0, n=3000):
    kernel = NUTS(model, dense_mass=True, init_strategy=init_to_uniform())
    mcmc = MCMC(kernel, num_warmup=800, num_samples=n, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    samples = mcmc.get_samples()
    names = [f"theta_{i}" for i in range(dim)]
    post = np.column_stack([np.asarray(samples[k]) for k in names])
    ld = build_log_density_fn(model)
    logpost = np.asarray(jax.vmap(ld)({k: jnp.asarray(v) for k, v in samples.items()}))

    def lp_fn(theta):
        return float(ld({f"theta_{i}": jnp.asarray(theta[i]) for i in range(dim)}))

    return post, logpost, lp_fn, names


def test_morphz_recovers_analytic_logz(tmp_path):
    model, logz_true, dim = _toy_problem(seed=1)
    post, logpost, lp_fn, names = _run_nuts(model, dim, seed=1)

    res = morphz_log_evidence(
        post, lp_fn,
        log_posterior_values=logpost,
        param_names=names,
        output_path=tmp_path,
        n_estimations=3,
    )
    assert res.ok
    assert res.method == "morph"
    # morphZ on a well-sampled low-dim Gaussian should be within ~0.5 nats.
    assert abs(res.logZ - logz_true) < 0.5, (res.logZ, logz_true)


def test_fallback_triggers_when_morphz_fails(tmp_path):
    # A log-posterior function that always raises forces morphZ to fail, so the
    # fallback estimator must supply the value.
    post = np.random.default_rng(0).standard_normal((200, 3))

    def broken_lp(theta):
        raise RuntimeError("boom")

    sentinel = EvidenceResult(logZ=-12.34, logZ_err=0.1, method="nested", status="ok")
    res = evidence_with_fallback(
        post, broken_lp,
        param_names=["a", "b", "c"],
        output_path=tmp_path,
        n_retries=1,
        fallback=lambda: sentinel,
    )
    assert res.status == "fallback"
    assert res.method == "nested"
    assert res.logZ == -12.34


def test_failed_result_when_no_fallback(tmp_path):
    post = np.random.default_rng(0).standard_normal((200, 3))

    def broken_lp(theta):
        raise RuntimeError("boom")

    res = evidence_with_fallback(
        post, broken_lp,
        param_names=["a", "b", "c"],
        output_path=tmp_path,
        n_retries=1,
        fallback=None,
    )
    assert res.status == "failed"
    assert not res.ok
    assert not np.isfinite(res.logZ)
