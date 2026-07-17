"""Fitting factors of the CCSN decoder against the held-out validation set.

For each of the 353 held-out Richers waveforms, maximize the noise-weighted
match (design PSD, 300-800 Hz band, no time shift -- the sampler has no time
freedom either) over the 5-D latent space:

    FF = max_z  <v | f(z)> / sqrt(<v|v> <f(z)|f(z)>)

Then join the FFs onto the L1 real-noise injection outcomes (event index ->
held-out waveform index, by replaying the per-event RNG of real_noise_event.py)
to test whether missed injections are the low-FF waveforms.

    .venv/bin/python studies/fitting_factor.py --results slurm/out/rn_L1/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from starccato_jax.data.training_data import TrainValData
from starccato_jax.waveforms import get_model

from pp_test import _design_psd

SAMPLE_RATE = 4096.0
N_SEG = 16384  # 4 s, matching the analysis segment
FLOW, FMAX = 300.0, 800.0


def _whitener():
    dt = 1.0 / SAMPLE_RATE
    psd_vals, _ = _design_psd(N_SEG, dt)
    freqs = np.fft.rfftfreq(N_SEG, d=dt)
    band = (freqs >= FLOW) & (freqs <= FMAX)
    w = np.zeros_like(psd_vals)
    w[band] = 1.0 / np.sqrt(psd_vals[band])
    return jnp.asarray(w)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("slurm/out/rn_L1/results.json"))
    p.add_argument("--outdir", type=Path, default=Path("out_ff"))
    p.add_argument("--n-random", type=int, default=4096)
    p.add_argument("--n-refine", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pool = np.asarray(TrainValData.load(source="ccsne", seed=0).val)  # (353, 512)
    model = get_model("ccsne")
    zdim = int(model.latent_dim)
    w = _whitener()
    pad_left = (N_SEG - pool.shape[1]) // 2

    def _whitened_fd(x_td):
        x = jnp.pad(x_td, (pad_left, N_SEG - pool.shape[1] - pad_left))
        return jnp.fft.rfft(x) * w

    def _match(z, v_wfd):
        h = model.generate(z=z[None, :])[0].astype(jnp.float64)
        h_wfd = _whitened_fd(h)
        num = jnp.real(jnp.vdot(v_wfd, h_wfd))
        return num / (jnp.linalg.norm(v_wfd) * jnp.linalg.norm(h_wfd))

    batch_match = jax.jit(jax.vmap(_match, in_axes=(0, None)))
    grad_step = jax.jit(jax.vmap(jax.value_and_grad(lambda z, v: -_match(z, v)), in_axes=(0, None)))

    rng = np.random.default_rng(0)
    z_rand = jnp.asarray(np.vstack([np.zeros((1, zdim)), rng.standard_normal((args.n_random, zdim))]))
    opt = optax.adam(5e-2)

    ffs = np.zeros(pool.shape[0])
    for i, v in enumerate(pool):
        v_wfd = _whitened_fd(jnp.asarray(v, dtype=jnp.float64))
        m0 = batch_match(z_rand, v_wfd)
        z = z_rand[jnp.argsort(-m0)[: args.n_refine]]
        state = opt.init(z)
        for _ in range(args.steps):
            loss, g = grad_step(z, v_wfd)
            updates, state = opt.update(g, state)
            z = optax.apply_updates(z, updates)
        ffs[i] = float(-loss.min())
        if i % 50 == 0:
            print(f"[{i}/{pool.shape[0]}] FF={ffs[i]:.4f}")

    np.save(args.outdir / "fitting_factors.npy", ffs)
    q = np.percentile(ffs, [5, 25, 50, 75, 95])
    summary = {
        "n": int(pool.shape[0]),
        "median": float(q[2]),
        "quantiles_5_25_50_75_95": [float(x) for x in q],
        "frac_below_0.90": float(np.mean(ffs < 0.90)),
        "frac_below_0.95": float(np.mean(ffs < 0.95)),
    }

    # join onto the L1 injection outcomes by replaying real_noise_event's RNG
    if args.results.exists():
        rows = [r for r in json.loads(args.results.read_text()) if r["cls"] == "inj_ccsn"]
        joined = []
        for r in rows:
            g = np.random.default_rng(1000 + r["index"])
            g.uniform()          # injected-SNR draw
            g.uniform(); g.uniform(); g.uniform()  # ra, dec, psi
            wf_idx = int(g.integers(pool.shape[0]))
            joined.append({"snr": r["snr"], "log_odds": r["log_odds"], "ff": float(ffs[wf_idx])})
        miss = np.array([j["log_odds"] < 0 for j in joined])
        ff_j = np.array([j["ff"] for j in joined])
        snr_j = np.array([j["snr"] for j in joined])
        loud = snr_j >= 16
        summary["events"] = {
            "n_inj": int(len(joined)),
            "ff_median_missed": float(np.median(ff_j[miss])),
            "ff_median_detected": float(np.median(ff_j[~miss])),
            "ff_median_missed_loud": float(np.median(ff_j[miss & loud])),
            "ff_median_detected_loud": float(np.median(ff_j[~miss & loud])),
            "miss_rate_ff_below_med": float(np.mean(miss[ff_j < np.median(ffs)])),
            "miss_rate_ff_above_med": float(np.mean(miss[ff_j >= np.median(ffs)])),
            "miss_rate_loud_ff_lo": float(np.mean(miss[loud & (ff_j < 0.95)])),
            "miss_rate_loud_ff_hi": float(np.mean(miss[loud & (ff_j >= 0.95)])),
        }
        (args.outdir / "joined.json").write_text(json.dumps(joined))

    (args.outdir / "ff_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
