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
# BayesWave reconstruction
# ---------------------------------------------------------------------------

def bayeswave_waveform_draws(post_signal_dir: Path, ifo: str = "H1") -> np.ndarray:
    """Load BayesWave signal-model waveform draws as (n_draws, n_time).

    BayesWavePost writes signal_recovered_whitened_waveform_<IFO>.dat with one
    row per posterior draw and one column per time sample (50 x 8192 for the
    4 s / 2048 Hz segment), i.e. already the (n_draws, n_time) matrix we want.
    Whitened is the right choice here: it is what the sampler actually
    constrains, and we peak-normalize for a morphology comparison anyway.
    """
    f = Path(post_signal_dir) / f"signal_recovered_whitened_waveform_{ifo}.dat"
    if not f.is_file():
        raise FileNotFoundError(
            f"{f} not found. Expected BayesWavePost output under post/signal/; "
            f"available: {sorted(p.name for p in Path(post_signal_dir).glob('*waveform*'))}")
    arr = np.atleast_2d(np.loadtxt(f))
    return arr / np.max(np.abs(arr), axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _band(ax, wf: np.ndarray, t: np.ndarray, color: str, label: str) -> None:
    lo, med, hi = np.percentile(wf, [5, 50, 95], axis=0)
    ax.fill_between(t, lo, hi, color=color, alpha=0.25, lw=0)
    ax.plot(t, med, color=color, lw=1.6, label=label)


def _time_axis(wf: np.ndarray, fs: float) -> np.ndarray:
    """Millisecond axis with t=0 at the peak of the median waveform.

    The two methods are on different grids (our VAE 512 @ 4096 Hz, BayesWave
    8192 @ 2048 Hz over a 4 s segment), so a shared array-centre origin does
    not line the signals up. Anchoring each on its own median peak does.
    """
    med = np.median(wf, axis=0)
    return (np.arange(wf.shape[1]) - int(np.argmax(np.abs(med)))) / fs * 1e3


def make_plot(our: np.ndarray, bw: np.ndarray | None, truth: np.ndarray | None,
              out: Path, fs: float = 4096.0, bw_fs: float = 2048.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _time_axis(our, fs)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    _band(ax, our, t, "#0072B2", r"our posterior ($\ln\mathcal{O}$)")
    if bw is not None:
        # BayesWave runs at its own rate over a much longer segment; put it on
        # our grid, peak-aligned, and crop to our window.
        bw_i = np.stack([np.interp(t, _time_axis(bw, bw_fs), w) for w in bw])
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
    ap.add_argument("--ifo", default="H1", help="which detector's BayesWave reconstruction")
    ap.add_argument("--out", type=Path, default=Path("fig_waveform_reco.pdf"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.our_samples is None:
        ap.error("--our-samples required (or --self-test)")
    our = our_waveform_draws(args.our_samples, model=args.model)
    bw = (bayeswave_waveform_draws(args.bayeswave_post, ifo=args.ifo)
          if args.bayeswave_post else None)
    truth = np.load(args.truth) if args.truth else None
    make_plot(our, bw, truth, args.out)


if __name__ == "__main__":
    main()
