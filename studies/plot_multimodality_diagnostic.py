"""Plot chain separation and decoded shapes for known multimodal NUTS runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from starccato_jax import StarccatoVAE

COLORS = ("#35618F", "#C58B2A", "#C45A35", "#6D7B3B")
LINESTYLES = ("-", "--", "-.", ":")


def _load_run(path: Path):
    diagnostics = json.loads(path.read_text())
    samples = np.load(path.with_name("samples.npz"))
    num_chains = int(diagnostics["num_chains"])
    num_samples = int(diagnostics["num_samples_per_chain"])
    grouped = {
        name: np.asarray(samples[name]).reshape(num_chains, num_samples)
        for name in samples.files
        if name in diagnostics["max_rhat"]
    }
    worst = max(diagnostics["max_rhat"], key=diagnostics["max_rhat"].get)
    return diagnostics, grouped, worst


def _density_panel(ax, grouped, parameter, max_rhat):
    values = grouped[parameter]
    lower, upper = np.percentile(values, [0.25, 99.75])
    padding = 0.08 * (upper - lower)
    grid = np.linspace(lower - padding, upper + padding, 400)
    for chain, draws in enumerate(values):
        density = gaussian_kde(draws)(grid)
        ax.plot(
            grid,
            density,
            color=COLORS[chain],
            linestyle=LINESTYLES[chain],
            linewidth=1.6,
            label=f"chain {chain + 1}",
        )
    ax.set_xlabel(parameter.replace("_", " "))
    ax.set_ylabel("posterior density")
    ax.text(
        0.02,
        0.96,
        rf"max $\hat{{R}}={max_rhat:.2f}$",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )


def _waveform_panel(ax, diagnostics, grouped, sample_rate):
    model = StarccatoVAE(diagnostics["model_dir"])
    latent_names = sorted(
        (name for name in grouped if name.startswith("z_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    medians = np.column_stack(
        [np.median(grouped[name], axis=1) for name in latent_names]
    )
    waveforms = np.asarray(model.generate(medians))
    waveforms = waveforms / np.max(np.abs(waveforms), axis=1, keepdims=True)
    time = (
        np.arange(waveforms.shape[1]) - waveforms.shape[1] // 2
    ) / sample_rate
    keep = np.abs(time) <= 0.06
    for chain, waveform in enumerate(waveforms):
        ax.plot(
            1e3 * time[keep],
            waveform[keep],
            color=COLORS[chain],
            linestyle=LINESTYLES[chain],
            linewidth=1.35,
        )
    ax.axhline(0.0, color="0.75", linewidth=0.7)
    ax.set_xlabel("time from segment centre [ms]")
    ax.set_ylabel("normalized strain")


def make_figure(
    ccsne_path: Path,
    blip_path: Path,
    output: Path,
    sample_rate: float,
) -> None:
    runs = (
        ("CCSNE candidate", *_load_run(ccsne_path)),
        ("Blip candidate", *_load_run(blip_path)),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.1),
        constrained_layout=True,
        sharey="row",
    )
    for column, (label, diagnostics, grouped, worst) in enumerate(runs):
        maximum = float(diagnostics["max_rhat"][worst])
        _density_panel(axes[0, column], grouped, worst, maximum)
        _waveform_panel(axes[1, column], diagnostics, grouped, sample_rate)
        axes[0, column].set_title(label, fontsize=10)
    axes[0, 0].legend(
        loc="upper center",
        bbox_to_anchor=(1.04, 1.28),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Persistent chain separation in rejected VAE candidates",
        fontsize=11,
    )
    fig.supxlabel(
        "Top: worst-separated posterior coordinate.  "
        "Bottom: waveform decoded at each chain median.",
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccsne", type=Path, required=True)
    parser.add_argument("--blip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=2048.0)
    args = parser.parse_args()
    make_figure(args.ccsne, args.blip, args.output, args.sample_rate)


if __name__ == "__main__":
    main()
