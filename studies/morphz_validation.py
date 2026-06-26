"""Validate the morphZ log-evidence estimator against an analytic ground truth.

morphZ is the production evidence estimator for the BCR ranking statistic, so we
must show it is *unbiased*, not merely consistent with another sampler. We do
this on a linear-Gaussian model whose log-evidence is known in closed form -- a
faithful stand-in for the real problem, where (near ``z = 0``) the VAE decoder is
approximately linear, the latent prior is Gaussian, and the frequency-domain GW
likelihood is Gaussian in the noise-weighted inner product.

Model
-----
Latent ``theta`` in R^d with prior ``theta ~ N(0, sigma_p^2 I)``. Data ``d`` in
R^n generated as ``d = M theta_true + noise`` with ``noise ~ N(0, sigma_n^2 I)``
and a fixed design matrix ``M``. The Gaussian marginal likelihood gives

    log Z = log N(d; 0, C),   C = sigma_p^2 M M^T + sigma_n^2 I,

including all normalisation constants. The NumPyro model below uses the *same*
fully-normalised Gaussian log-likelihood so that the sampled/bridge estimates of
``log Z`` are directly comparable to this analytic value.

Run
---
    uv run python studies/morphz_validation.py --trials 10 --dim 8 --outdir out_morphz_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_uniform
from numpyro.infer.util import log_density

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

from starccato_lvk.analysis.evidence import morphz_log_evidence  # noqa: E402


def build_log_density_fn(model):
    """Return a jitted joint log-density ``log p(theta, d)`` for a NumPyro model.

    This is the unnormalised log-posterior morphZ integrates; kept local so the
    statistics-only validation does not import the GW waveform stack.
    """

    def _logpost(params):
        lp, _ = log_density(model, (), {}, params)
        return lp

    return jax.jit(_logpost)


def analytic_log_evidence(
    data: np.ndarray, M: np.ndarray, sigma_p: float, sigma_n: float
) -> float:
    """Closed-form ``log Z`` for the linear-Gaussian model (all constants kept)."""
    n = data.shape[0]
    C = (sigma_p**2) * (M @ M.T) + (sigma_n**2) * np.eye(n)
    sign, logdet = np.linalg.slogdet(C)
    Cinv_d = np.linalg.solve(C, data)
    quad = float(data @ Cinv_d)
    return float(-0.5 * (n * np.log(2.0 * np.pi) + logdet + quad))


def _build_model(data: jnp.ndarray, M: jnp.ndarray, sigma_p: float, sigma_n: float, dim: int):
    n = data.shape[0]
    const = -0.5 * (n * np.log(2.0 * np.pi) + n * np.log(sigma_n**2))
    inv_var = 1.0 / (sigma_n**2)

    def model():
        theta = jnp.stack(
            [numpyro.sample(f"theta_{i}", dist.Normal(0.0, sigma_p)) for i in range(dim)]
        )
        resid = data - M @ theta
        loglike = const - 0.5 * inv_var * jnp.sum(resid**2)
        numpyro.factor("log_likelihood", loglike)

    return model


def run_trial(
    seed: int,
    *,
    dim: int,
    n_data: int,
    sigma_p: float,
    sigma_n: float,
    num_warmup: int,
    num_samples: int,
    outdir: Path,
    do_nested: bool,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n_data, dim)) / np.sqrt(dim)
    theta_true = rng.standard_normal(dim) * sigma_p
    data = M @ theta_true + rng.standard_normal(n_data) * sigma_n

    logZ_true = analytic_log_evidence(data, M, sigma_p, sigma_n)

    data_j = jnp.asarray(data)
    M_j = jnp.asarray(M)
    model = _build_model(data_j, M_j, sigma_p, sigma_n, dim)

    # NUTS posterior + joint log-density values for morphZ.
    kernel = NUTS(model, dense_mass=True, init_strategy=init_to_uniform())
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    samples = mcmc.get_samples()

    theta_names = [f"theta_{i}" for i in range(dim)]
    post = np.column_stack([np.asarray(samples[name]) for name in theta_names])

    log_density_fn = build_log_density_fn(model)
    logpost_values = np.asarray(
        jax.vmap(log_density_fn)({k: jnp.asarray(v) for k, v in samples.items()})
    )

    def lp_fn(theta: np.ndarray) -> float:
        params = {f"theta_{i}": jnp.asarray(theta[i]) for i in range(dim)}
        return float(log_density_fn(params))

    morph = morphz_log_evidence(
        post,
        lp_fn,
        log_posterior_values=logpost_values,
        param_names=theta_names,
        output_path=outdir / f"trial_{seed}",
        label="toy",
        n_estimations=3,
    )

    row = {
        "seed": seed,
        "logZ_true": logZ_true,
        "logZ_morph": morph.logZ,
        "logZ_morph_err": morph.logZ_err,
        "morph_status": morph.status,
        "morph_bias": morph.logZ - logZ_true,
    }

    if do_nested:
        # Reuse the nested sampler plumbing by faking a likelihood object exposing
        # `.evaluate`. Simpler: run a dedicated nested sampler on the same model.
        from numpyro.contrib.nested_sampling import NestedSampler

        ns = NestedSampler(
            model,
            constructor_kwargs=dict(num_live_points=max(50, 25 * dim), verbose=False),
            termination_kwargs=dict(dlogZ=0.1),
        )
        ns.run(jax.random.PRNGKey(seed + 7))
        try:
            logZ_nested = float(ns.evidence)
            logZ_nested_err = float(ns.evidence_error)
        except AttributeError:
            res = getattr(ns, "_results", None)
            logZ_nested = float(getattr(res, "log_Z_mean", np.nan))
            logZ_nested_err = float(getattr(res, "log_Z_uncert", np.nan))
        row["logZ_nested"] = logZ_nested
        row["logZ_nested_err"] = logZ_nested_err
        row["nested_bias"] = logZ_nested - logZ_true

    return row


def _plot(rows, outpath: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true = np.array([r["logZ_true"] for r in rows])
    morph = np.array([r["logZ_morph"] for r in rows])
    morph_err = np.array([r["logZ_morph_err"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    lo, hi = float(np.min(true)), float(np.max(true))
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax[0].errorbar(true, morph, yerr=morph_err, fmt="o", ms=4, capsize=2, label="morphZ")
    if "logZ_nested" in rows[0]:
        nested = np.array([r["logZ_nested"] for r in rows])
        nested_err = np.array([r.get("logZ_nested_err", np.nan) for r in rows])
        ax[0].errorbar(true, nested, yerr=nested_err, fmt="s", ms=4, capsize=2, label="nested")
    ax[0].set_xlabel("analytic log Z")
    ax[0].set_ylabel("estimated log Z")
    ax[0].legend()
    ax[0].set_title("Estimator vs analytic ground truth")

    bias = morph - true
    ax[1].axvline(0.0, color="k", ls="--", lw=1)
    ax[1].hist(bias, bins=max(5, len(rows) // 2), alpha=0.7, label="morphZ")
    if "logZ_nested" in rows[0]:
        ax[1].hist(
            np.array([r["nested_bias"] for r in rows]),
            bins=max(5, len(rows) // 2),
            alpha=0.5,
            label="nested",
        )
    ax[1].set_xlabel("log Z bias (estimate - truth)")
    ax[1].set_ylabel("count")
    ax[1].legend()
    ax[1].set_title(f"morphZ bias: {np.mean(bias):+.3f} +/- {np.std(bias):.3f}")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--n-data", type=int, default=64)
    p.add_argument("--sigma-p", type=float, default=1.0)
    p.add_argument("--sigma-n", type=float, default=1.0)
    p.add_argument("--num-warmup", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=4000)
    p.add_argument("--outdir", type=Path, default=Path("out_morphz_val"))
    p.add_argument("--no-nested", action="store_true", help="Skip the nested-sampling cross-check.")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.trials):
        row = run_trial(
            seed,
            dim=args.dim,
            n_data=args.n_data,
            sigma_p=args.sigma_p,
            sigma_n=args.sigma_n,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            outdir=args.outdir,
            do_nested=not args.no_nested,
        )
        print(
            f"[trial {seed}] true={row['logZ_true']:+.3f} "
            f"morph={row['logZ_morph']:+.3f}+/-{row['logZ_morph_err']:.3f} "
            f"bias={row['morph_bias']:+.3f} [{row['morph_status']}]"
        )
        rows.append(row)

    bias = np.array([r["morph_bias"] for r in rows])
    summary = {
        "config": vars(args) | {"outdir": str(args.outdir)},
        "morph_bias_mean": float(np.mean(bias)),
        "morph_bias_std": float(np.std(bias)),
        "morph_rms": float(np.sqrt(np.mean(bias**2))),
        "n_failed": int(sum(r["morph_status"] == "failed" for r in rows)),
        "rows": rows,
    }
    (args.outdir / "morphz_validation.json").write_text(json.dumps(summary, indent=2))
    _plot(rows, args.outdir / "morphz_validation.png")
    print(
        f"\nmorphZ bias = {summary['morph_bias_mean']:+.3f} +/- {summary['morph_bias_std']:.3f} "
        f"(RMS {summary['morph_rms']:.3f}); failures: {summary['n_failed']}/{args.trials}"
    )
    print(f"Wrote {args.outdir/'morphz_validation.json'} and morphz_validation.png")


if __name__ == "__main__":
    main()
