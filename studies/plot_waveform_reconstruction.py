"""Time-domain waveform posterior: our VAE odds analysis vs BayesWave.

Overlays, for a single event, the reconstructed signal waveform from
  - our method: posterior latent draws (signal samples.npz, written under
    <outdir>/e<N>/<class>/analysis/signal/ when real_noise_event.py is run with
    --save-artifacts) decoded through the Starccato VAE, and
  - BayesWave: the signal-model waveform draws from BayesWavePost's post/signal/.

Both are standardized to peak-normalized amplitude so the comparison is of
morphology, which is what the odds ratio discriminates on.

    # 1. re-run ONE event with artifacts (--save-artifacts also suppresses the
    #    prune that would otherwise delete analysis/ straight afterwards):
    python studies/real_noise_event.py --index 3 --detectors H1 L1 \
        --class inj_ccsn --save-artifacts --outdir <OUTDIR>
    # 2. plot:
    uv run python studies/plot_waveform_reconstruction.py \
        --our-samples <OUTDIR>/e3/inj_ccsn/analysis/signal/samples.npz \
        --bayeswave-post /fred/oz980/.../bayeswave_H1_L1/e3/inj_ccsn/post/signal \
        --ifo H1 --out fig_waveform_reco.pdf
    # our-posterior-only (no BayesWave) works too -- omit --bayeswave-post.

    # self-check (no data needed):
    uv run python studies/plot_waveform_reconstruction.py --self-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the whitener behind the manuscript's posterior-predictive figure so both
# figures define "whitened" identically (studies/ cross-imports are the repo norm).
from pp_predictive_fig import _whiten


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
                       n_draws: int = 300, seed: int = 0,
                       normalize: bool = True) -> np.ndarray:
    """Decode posterior latent draws to time-domain waveforms.

    ``normalize=False`` returns raw strain, which is what whitening needs;
    peak-normalisation is then applied after whitening, not before.
    """
    from starccato_jax.waveforms import get_model

    with np.load(samples_npz) as d:
        z = _latent_matrix({k: d[k] for k in d.files})
    if z.shape[0] > n_draws:
        z = z[np.random.default_rng(seed).choice(z.shape[0], n_draws, replace=False)]
    wf = np.asarray(get_model(model).generate(z=z))  # (n, 512), standardized
    if not normalize:
        return wf
    return wf / np.max(np.abs(wf), axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# BayesWave reconstruction
# ---------------------------------------------------------------------------

def bayeswave_whitened_data(post_signal_dir: Path, ifo: str = "H1") -> np.ndarray | None:
    """BayesWavePost's whitened strain, on the same grid as its reconstruction."""
    f = Path(post_signal_dir).resolve().parent / f"whitened_data_{ifo}.dat"
    return np.loadtxt(f).ravel() if f.is_file() else None


def bayeswave_psd(post_signal_dir: Path, ifo: str = "H1") -> np.ndarray | None:
    """(freq, PSD) from BayesWave's median signal-model PSD, for whitening OUR waveform.

    Out-of-band rows carry a placeholder (~1.14) rather than a real estimate, so
    the caller must mask to the analysis band before using it.
    """
    f = Path(post_signal_dir) / f"signal_median_PSD_{ifo}.dat"
    if not f.is_file():
        return None
    psd = np.loadtxt(f)
    return np.column_stack([psd[:, 0], psd[:, 1]])


def whiten_like_bayeswave(
    waveform: np.ndarray, fs: float, freq_psd: np.ndarray, flow: float, fmax: float
) -> np.ndarray:
    """Whiten OUR time-domain waveform with BayesWave's PSD, band-limited.

    Our decoded waveform is raw strain while BayesWave's reconstruction is
    whitened, so overlaying them directly compares different quantities. Whiten
    ours through the same PSD (interpolated onto our own frequency grid) so the
    morphology comparison is like-for-like. Out-of-band PSD is set to inf, which
    makes the whitening double as the analysis-band bandpass.
    """
    n = waveform.shape[-1]
    dt = 1.0 / fs
    freqs = np.fft.rfftfreq(n, d=dt)
    psd = np.interp(freqs, freq_psd[:, 0], freq_psd[:, 1])
    psd = np.where((freqs >= flow) & (freqs <= fmax), psd, np.inf)
    rows = np.atleast_2d(waveform)
    return np.stack([_whiten(row, psd, dt) for row in rows])


def bayeswave_waveform_draws(post_signal_dir: Path, ifo: str = "H1",
                             normalize: bool = True) -> np.ndarray:
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
    if not normalize:
        return arr
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


def bayeswave_evidence_label(post_signal_dir: Path) -> str | None:
    """Read the run's result.json (two levels up from post/signal) for a caption.

    Read rather than passed in, so the annotated numbers cannot drift away from
    the reconstruction plotted beside them. Returns None when the run has no
    usable evidence, which is the common case for a partially-failed run.
    """
    result = Path(post_signal_dir).resolve().parent.parent / "result.json"
    if not result.is_file():
        return None
    d = json.loads(result.read_text())
    lnbf, unc = (d.get("log_bayeswave_signal_glitch"),
                 d.get("log_bayeswave_signal_glitch_uncertainty"))
    if lnbf is None or unc is None:
        return None
    sig_noise = d.get("logZ_signal", 0.0) - d.get("logZ_noise", 0.0)
    return (rf"BayesWave: $\ln\mathcal{{B}}_{{\rm S/G}}={lnbf:.1f}\pm{unc:.1f}$"
            "\n" rf"$\ln Z_{{\rm S}}-\ln Z_{{\rm N}}={sig_noise:+.1f}$")


def _peak_norm(wf: np.ndarray) -> np.ndarray:
    return wf / np.max(np.abs(wf), axis=1, keepdims=True)


def make_plot(our: np.ndarray, bw: np.ndarray | None, truth: np.ndarray | None,
              out: Path, fs: float = 4096.0, bw_fs: float = 2048.0,
              caption: str | None = None, data: np.ndarray | None = None) -> None:
    """Whitened reconstruction comparison.

    ``our`` and ``bw`` are RAW (un-normalised) whitened draws; ``data`` is the
    whitened strain on BayesWave's grid. With data present the figure gets a top
    panel on an absolute noise-sigma scale (does BayesWave fit the data?) above
    the peak-normalised morphology panel (do the two methods agree on shape?).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = _time_axis(our, fs)
    bw_i = None
    if bw is not None:
        # BayesWave runs at its own rate over a much longer segment; put it on
        # our grid, peak-aligned, and crop to our window.
        t_bw = _time_axis(bw, bw_fs)
        bw_i = np.stack([np.interp(t, t_bw, w) for w in bw])

    if data is not None and bw is not None:
        # The data shares BayesWave's grid, so it shares that peak-anchored axis.
        sigma = float(np.std(data))
        data_i = np.interp(t, t_bw, data) / sigma
        fig, (ax_d, ax) = plt.subplots(
            2, 1, figsize=(5.2, 4.6), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.08})
        ax_d.plot(t, data_i, color="#9a9a9a", lw=0.8, label="whitened data")
        _band(ax_d, bw_i / sigma, t, "#D55E00", "BayesWave")
        ax_d.set_ylabel(r"whitened strain [$\sigma$]")
        ax_d.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    else:
        fig, ax = plt.subplots(figsize=(5.2, 3.0))

    _band(ax, _peak_norm(our), t, "#0072B2", r"our posterior ($\ln\mathcal{O}$)")
    if bw_i is not None:
        _band(ax, _peak_norm(bw_i), t, "#D55E00", "BayesWave")
    if truth is not None:
        tr = truth / np.max(np.abs(truth))
        ax.plot(t, tr, color="k", ls="--", lw=1.2, label="injected")
    ax.set_xlabel("time relative to peak [ms]")
    ax.set_ylabel("peak-norm. whitened strain")
    if caption:
        ax.text(0.02, 0.03, caption, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.5))
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
    ap.add_argument("--flow", type=float, default=300.0, help="analysis band low edge (whitening bandpass)")
    ap.add_argument("--fmax", type=float, default=800.0, help="analysis band high edge")
    ap.add_argument("--no-data", action="store_true", help="omit the whitened-data panel")
    ap.add_argument("--out", type=Path, default=Path("fig_waveform_reco.pdf"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.our_samples is None:
        ap.error("--our-samples required (or --self-test)")
    our = our_waveform_draws(args.our_samples, model=args.model, normalize=False)
    bw = data = caption = None
    if args.bayeswave_post:
        bw = bayeswave_waveform_draws(args.bayeswave_post, ifo=args.ifo, normalize=False)
        caption = bayeswave_evidence_label(args.bayeswave_post)
        if not args.no_data:
            data = bayeswave_whitened_data(args.bayeswave_post, ifo=args.ifo)
        # Whiten OUR waveform through BayesWave's own PSD so both traces are the
        # same quantity; without a PSD we would be overlaying raw on whitened.
        psd = bayeswave_psd(args.bayeswave_post, ifo=args.ifo)
        if psd is not None:
            our = whiten_like_bayeswave(our, 4096.0, psd, args.flow, args.fmax)
        else:
            print("WARNING: no BayesWave PSD found; our trace is NOT whitened and "
                  "is not directly comparable to BayesWave's whitened reconstruction.")
    truth = np.load(args.truth) if args.truth else None
    make_plot(our, bw, truth, args.out, caption=caption, data=data)


if __name__ == "__main__":
    main()
