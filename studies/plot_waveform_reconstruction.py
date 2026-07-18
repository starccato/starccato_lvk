"""Time-domain waveform posterior: our VAE odds analysis vs BayesWave.

Overlays, for a single event, the reconstructed signal waveform from
  - our method: posterior latent draws (signal samples.npz, written when
    run_bcr_posteriors is called with save_artifacts=True) decoded through the
    Starccato VAE, and
  - BayesWave: the signal-model waveform draws from BayesWavePost's post/signal/.

Both are standardized to peak-normalized amplitude so the comparison is of
morphology, which is what the odds ratio discriminates on.

    # 1. re-run ONE event with artifacts (see module docstring in real_noise_event):
    #    run_bcr_posteriors(..., save_artifacts=True) -> <edir>/signal/samples.npz
    # 2. plot:
    uv run python studies/plot_waveform_reconstruction.py \
        --our-samples slurm/out/rn_H1_L1/e0/signal/samples.npz \
        --bayeswave-post /fred/oz980/.../bayeswave_H1_L1/e0/inj_ccsn/post/signal \
        --out fig_waveform_reco.pdf
    # our-posterior-only (no BayesWave) works too -- omit --bayeswave-post.

    # self-check (no data needed):
    uv run python studies/plot_waveform_reconstruction.py --self-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Our posterior: latent samples.npz -> decoded time-domain waveform draws
# ---------------------------------------------------------------------------

def _latent_matrix(npz: dict, latent_dim: int = 5) -> np.ndarray:
    """Stack the latent columns of a samples.npz into (n_samples, latent_dim).

    The sampler saves one 1-D array per parameter under keys we don't want to
    hard-code (they vary with model/config), so we take the latent columns as
    the numeric-suffixed keys (z_0.., latent_0.., or similar) in sorted order,
    falling back to the first `latent_dim` non-amplitude arrays.
    """
    import re
    amp_like = re.compile(r"amp|dist|ra|dec|psi|t_c|phase|iota", re.I)
    cand = {k: np.asarray(v) for k, v in npz.items()
            if v.ndim == 1 and not amp_like.search(k)}
    # prefer keys ending in an integer index, ordered by that index
    numbered = sorted((k for k in cand if re.search(r"\d+$", k)),
                      key=lambda k: int(re.search(r"(\d+)$", k).group(1)))
    keys = numbered[:latent_dim] if len(numbered) >= latent_dim else list(cand)[:latent_dim]
    if len(keys) < latent_dim:
        raise ValueError(
            f"found {len(keys)} latent columns {keys}; expected {latent_dim}. "
            f"Available keys: {sorted(npz)}")
    return np.stack([cand[k] for k in keys], axis=1).astype(np.float32)


def our_waveform_draws(samples_npz: Path, model: str = "ccsne",
                       n_draws: int = 300, seed: int = 0) -> np.ndarray:
    """Decode posterior latent draws to peak-normalized time-domain waveforms."""
    from starccato_jax.waveforms import get_model

    with np.load(samples_npz) as d:
        z = _latent_matrix({k: d[k] for k in d.files})
    if z.shape[0] > n_draws:
        z = z[np.random.default_rng(seed).choice(z.shape[0], n_draws, replace=False)]
    wf = np.asarray(get_model(model).generate(z=z))  # (n, 512), standardized
    return wf / np.max(np.abs(wf), axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# BayesWave reconstruction  -- FINALIZE loader once the file format is known.
# ---------------------------------------------------------------------------

def bayeswave_waveform_draws(post_signal_dir: Path) -> np.ndarray:
    """Load BayesWave signal-model waveform draws as (n_draws, n_time).

    TODO(format): BayesWavePost writes the reconstructed signal waveform under
    post/signal/. Confirm the exact file (candidates: signal_waveform.dat.*,
    signal_recovered_whitened.dat, or the median+CI file) and its columns
    (time vs strain) before trusting this. Paste `ls` + `head` of that dir.
    """
    d = Path(post_signal_dir)
    # Best-guess: per-sample waveform files, one column of strain each.
    files = sorted(d.glob("signal_waveform.dat.*")) or sorted(d.glob("*waveform*.dat*"))
    if not files:
        raise FileNotFoundError(
            f"no waveform files under {d}; run with --our-samples only, or paste "
            "`ls`/`head` of this dir so the loader can be matched to the format.")
    draws = [np.loadtxt(f) for f in files]
    arr = np.stack([w[:, -1] if w.ndim == 2 else w for w in draws])
    return arr / np.max(np.abs(arr), axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _band(ax, wf: np.ndarray, t: np.ndarray, color: str, label: str) -> None:
    lo, med, hi = np.percentile(wf, [5, 50, 95], axis=0)
    ax.fill_between(t, lo, hi, color=color, alpha=0.25, lw=0)
    ax.plot(t, med, color=color, lw=1.6, label=label)


def make_plot(our: np.ndarray, bw: np.ndarray | None, truth: np.ndarray | None,
              out: Path, fs: float = 4096.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = our.shape[1]
    t = (np.arange(n) - n // 2) / fs * 1e3  # ms, peak-centered
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    _band(ax, our, t, "#0072B2", r"our posterior ($\ln\mathcal{O}$)")
    if bw is not None:
        # BayesWave may use a different length; interpolate onto our grid.
        tb = (np.arange(bw.shape[1]) - bw.shape[1] // 2) / fs * 1e3
        bw_i = np.stack([np.interp(t, tb, w) for w in bw])
        _band(ax, bw_i, t, "#D55E00", "BayesWave")
    if truth is not None:
        tr = truth / np.max(np.abs(truth))
        ax.plot(t, tr, color="k", ls="--", lw=1.2, label="injected")
    ax.set_xlabel("time relative to peak [ms]")
    ax.set_ylabel("peak-normalized strain")
    ax.legend(frameon=False, fontsize=8)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"wrote {out}")


def _self_test() -> None:
    """Fabricate a samples.npz-like dict, decode, and plot -- exercises our half."""
    import tempfile
    rng = np.random.default_rng(0)
    z = {f"z_{i}": rng.normal(0, 1, 500) for i in range(5)}
    z["log_amp"] = rng.normal(0, 1, 500)
    tmp = Path(tempfile.mkdtemp())
    np.savez(tmp / "samples.npz", **z)
    wf = our_waveform_draws(tmp / "samples.npz", n_draws=100)
    assert wf.shape == (100, 512), wf.shape
    assert np.allclose(np.max(np.abs(wf), axis=1), 1.0), "not peak-normalized"
    make_plot(wf, None, None, tmp / "selftest.pdf")
    print("self-test OK:", wf.shape, "->", tmp / "selftest.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--our-samples", type=Path)
    ap.add_argument("--bayeswave-post", type=Path, default=None)
    ap.add_argument("--truth", type=Path, default=None, help=".npy injected waveform (512,)")
    ap.add_argument("--model", default="ccsne")
    ap.add_argument("--out", type=Path, default=Path("fig_waveform_reco.pdf"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.our_samples is None:
        ap.error("--our-samples required (or --self-test)")
    our = our_waveform_draws(args.our_samples, model=args.model)
    bw = bayeswave_waveform_draws(args.bayeswave_post) if args.bayeswave_post else None
    truth = np.load(args.truth) if args.truth else None
    make_plot(our, bw, truth, args.out)


if __name__ == "__main__":
    main()
