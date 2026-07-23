"""Plot coherent-signal and incoherent-glitch waveform posteriors side by side.

The coherent-signal column overlays Starccato and BayesWave posterior waveform
bands for H1 and L1.  BayesWavePost does not retain glitch waveform draws in
this campaign, so the incoherent-glitch column intentionally shows Starccato's
independent H1 and L1 glitch posterior bands only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_waveform_reconstruction import (
    _peak_norm,
    _time_axis,
    bayeswave_psd,
    bayeswave_waveform_draws,
    our_waveform_draws,
    whiten_like_bayeswave,
)


def _band(ax, waveforms: np.ndarray, time_ms: np.ndarray, color: str, label: str) -> None:
    lo, med, hi = np.percentile(_peak_norm(waveforms), [5, 50, 95], axis=0)
    ax.fill_between(time_ms, lo, hi, color=color, alpha=0.24, lw=0)
    ax.plot(time_ms, med, color=color, lw=1.45, label=label)


def _signal_draws(samples: Path, post: Path, ifo: str) -> tuple[np.ndarray, np.ndarray]:
    ours = our_waveform_draws(samples, model="ccsne", normalize=False)
    psd = bayeswave_psd(post, ifo)
    if psd is None:
        raise FileNotFoundError(f"missing BayesWave signal PSD for {ifo}: {post}")
    ours = whiten_like_bayeswave(ours, 4096.0, psd, 300.0, 800.0)
    bayeswave = bayeswave_waveform_draws(post, ifo=ifo, normalize=False)
    return ours, bayeswave


def _glitch_draws(samples: Path, post: Path, ifo: str) -> np.ndarray:
    """Whiten a detector-specific Blip posterior with the event's PSD.

    BayesWave's signal PSD is used only as a common whitening reference.  The
    figure is morphology-normalized, so this does not assert a BayesWave glitch
    reconstruction where no such posterior draw was exported.
    """
    ours = our_waveform_draws(samples, model="blip", normalize=False)
    psd = bayeswave_psd(post, ifo)
    if psd is None:
        raise FileNotFoundError(f"missing BayesWave signal PSD for {ifo}: {post}")
    return whiten_like_bayeswave(ours, 4096.0, psd, 300.0, 800.0)


def make_figure(
    signal_samples: Path,
    glitch_h1_samples: Path,
    glitch_l1_samples: Path,
    bayeswave_post: Path,
    result: Path,
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evidence = json.loads(result.read_text())
    ln_o = float(evidence["log_odds"])
    bw_result = bayeswave_post.parent.parent / "result.json"
    bw = json.loads(bw_result.read_text())
    ln_bf = float(bw["log_bayeswave_signal_glitch"])
    ln_bf_err = float(bw["log_bayeswave_signal_glitch_uncertainty"])

    fig = plt.figure(figsize=(12.2, 5.8), constrained_layout=True)
    coherent, glitches = fig.subfigures(1, 2, wspace=0.06)
    coherent_axes = coherent.subplots(2, 1, sharex=True)
    glitch_axes = glitches.subplots(2, 1, sharex=True)
    coherent.suptitle("A. Coherent signal posterior", fontsize=12, fontweight="bold")
    glitches.suptitle("B. Incoherent glitch posteriors", fontsize=12, fontweight="bold")

    for ax, ifo in zip(coherent_axes, ("H1", "L1")):
        ours, bayeswave = _signal_draws(signal_samples, bayeswave_post, ifo)
        time_ms = _time_axis(ours, 4096.0)
        _band(ax, ours, time_ms, "#0072B2", "our posterior")
        time_bw = _time_axis(bayeswave, 2048.0)
        interp = np.stack([np.interp(time_ms, time_bw, row) for row in bayeswave])
        _band(ax, interp, time_ms, "#D55E00", "BayesWave posterior")
        ax.set_ylabel(f"{ifo}\npeak-norm. strain")
        ax.axhline(0, color="0.75", lw=0.6, zorder=0)
        ax.margins(x=0)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    for ax, ifo, samples in zip(
        glitch_axes, ("H1", "L1"), (glitch_h1_samples, glitch_l1_samples)
    ):
        ours = _glitch_draws(samples, bayeswave_post, ifo)
        time_ms = _time_axis(ours, 4096.0)
        color = "#009E73" if ifo == "H1" else "#CC79A7"
        _band(ax, ours, time_ms, color, f"our {ifo} glitch posterior")
        ax.set_ylabel(f"{ifo}\npeak-norm. strain")
        ax.axhline(0, color="0.75", lw=0.6, zorder=0)
        ax.margins(x=0)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    for ax in (coherent_axes[-1], glitch_axes[-1]):
        ax.set_xlabel("time relative to posterior median peak [ms]")
    glitches.text(
        0.5,
        0.01,
        "BayesWave glitch posterior draws were not exported by BayesWavePost.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.3",
    )
    event = evidence.get("index", "?")
    fig.suptitle(
        rf"e{event}: morphZ $\ln\mathcal{{O}}_{{S/G}}={ln_o:+.1f}$; "
        rf"BayesWave $\ln\mathcal{{B}}_{{S/G}}={ln_bf:+.1f}\pm{ln_bf_err:.1f}$",
        fontsize=13,
        y=1.03,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-samples", type=Path, required=True)
    parser.add_argument("--glitch-h1-samples", type=Path, required=True)
    parser.add_argument("--glitch-l1-samples", type=Path, required=True)
    parser.add_argument("--bayeswave-post", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True, help="morphZ result JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    make_figure(**vars(args))


if __name__ == "__main__":
    main()
