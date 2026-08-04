"""Explain a Starccato posterior prediction in source and analysis frames.

This diagnostic deliberately compares only quantities that share a complete
processing contract.  For one coherently injected CCSN it reconstructs the VAE
signal posterior from the saved latent/amplitude samples, applies the exact
detector response used by inference, and displays it twice:

1. detector-projected strain before the likelihood bandpass; and
2. the same draws after the run's 300--800 Hz mask and PSD whitening.

The exact injected signal is obtained by subtracting the paired noise bundle
from the injected bundle.  All traces use the injected peak as a common time
origin; posterior draws are never peak-aligned independently.  This makes a
timing mismatch visible and shows why the analysis-frame waveform can look
very different from the raw CCSN morphology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from starccato_jax.waveforms import get_model
from starccato_lvk.analysis.jim_waveform import StarccatoJimWaveform
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
from starccato_lvk.analysis.post_proc.jim_plots import _whiten


VAE_COLOR = "#0072B2"
TRUTH_COLOR = "#666666"
DATA_COLOR = "#111111"


def _noise_bundle(event_dir: Path, detector: str) -> Path:
    matches = sorted((event_dir / "noise" / detector).glob("analysis_bundle_*.hdf5"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one noise bundle for {detector}, found {len(matches)}"
        )
    return matches[0]


def _posterior_draw_indices(n_samples: int, n_draws: int, seed: int) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("posterior sample archive is empty")
    n = min(int(n_draws), n_samples)
    return np.random.default_rng(seed).choice(n_samples, size=n, replace=False)


def _normalized_overlap(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _visible(time_ms: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mask = (time_ms >= -35.0) & (time_ms <= 35.0)
    return (time_ms[mask], *(np.asarray(value)[..., mask] for value in arrays))


def make_figure(
    vae_root: Path,
    event_index: int,
    output: Path,
    n_draws: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event_dir = vae_root / f"e{event_index}"
    manifest = json.loads((event_dir / "manifest.json").read_text())
    if "inj_ccsn" not in manifest["bundles"]:
        raise ValueError(f"e{event_index} does not contain an injected CCSN bundle")

    detectors = tuple(manifest["detectors"])
    if detectors != ("H1", "L1"):
        raise ValueError(f"expected the H1--L1 configuration, got {detectors}")
    injection_paths = {
        detector: Path(path)
        for detector, path in manifest["bundles"]["inj_ccsn"].items()
    }
    noise_paths = {
        detector: _noise_bundle(event_dir, detector) for detector in detectors
    }
    flow, fmax = map(float, manifest["band"])
    injected = prepare_multi_detector_data(
        detectors, bundle_paths=injection_paths, flow=flow, fmax=fmax
    )
    noise = prepare_multi_detector_data(
        detectors, bundle_paths=noise_paths, flow=flow, fmax=fmax
    )

    sample_path = event_dir / "inj_ccsn" / "analysis" / "signal" / "samples.npz"
    with np.load(sample_path) as archive:
        samples = {key: np.asarray(archive[key]) for key in archive.files}
    required = tuple(f"z_{i}" for i in range(5)) + ("log_amp",)
    missing = [name for name in required if name not in samples]
    if missing:
        raise KeyError(f"missing posterior parameters in {sample_path}: {missing}")
    sample_count = int(samples[required[0]].size)
    if any(samples[name].size != sample_count for name in required):
        raise ValueError("posterior parameter arrays have inconsistent lengths")
    draw_indices = _posterior_draw_indices(sample_count, n_draws, seed)

    reference = injected.detector_data[detectors[0]]
    sample_rate = 1.0 / float(reference.dt)
    waveform = StarccatoJimWaveform(
        model=get_model("ccsne"),
        sample_rate=sample_rate,
        window=injected.window,
    )
    extrinsics = dict(manifest["sky"])
    extrinsics.setdefault("gmst", injected.gmst)
    extrinsics.setdefault("trigger_time", injected.trigger_time)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5), sharex="col")
    panel_labels = (("(a)", "(b)"), ("(c)", "(d)"))
    metrics: dict[str, dict[str, float]] = {}

    for column, detector in enumerate(detectors):
        data = injected.detector_data[detector]
        noise_data = noise.detector_data[detector]
        frequency = np.asarray(data.frequency)
        dt = float(data.dt)
        n_time = int(np.asarray(data.time).size)

        raw_draws = []
        for index in draw_indices:
            params = {name: float(samples[name][index]) for name in required}
            params.update(extrinsics)
            source_fd = waveform(jnp.asarray(frequency), params)
            response_fd = np.asarray(
                next(det for det in injected.detectors if det.name == detector).fd_response(
                    jnp.asarray(frequency), source_fd, params
                )
            )
            raw_draws.append(np.fft.irfft(response_fd, n=n_time) / dt)
        raw_draws = np.asarray(raw_draws)

        raw_truth = np.asarray(data.windowed_strain) - np.asarray(
            noise_data.windowed_strain
        )
        truth_peak_index = int(np.argmax(np.abs(raw_truth)))
        truth_peak = float(raw_truth[truth_peak_index])
        truth_scale = abs(truth_peak) or float(np.max(np.abs(raw_truth))) or 1.0
        time_ms = (np.asarray(data.time) - np.asarray(data.time)[truth_peak_index]) * 1e3

        mask = np.asarray(data.band_mask, dtype=bool)
        psd = np.where(mask, np.asarray(data.psd.values), np.inf)
        whitened_noise = _whiten(np.asarray(noise_data.windowed_strain), psd, dt)
        noise_rms = float(np.std(whitened_noise)) or 1.0
        whitened_draws = np.asarray(
            [_whiten(draw, psd, dt) / noise_rms for draw in raw_draws]
        )
        whitened_truth = _whiten(raw_truth, psd, dt) / noise_rms
        whitened_data = _whiten(np.asarray(data.windowed_strain), psd, dt) / noise_rms

        raw_median = np.median(raw_draws, axis=0)
        whitened_median = np.median(whitened_draws, axis=0)
        metrics[detector] = {
            "raw_overlap": _normalized_overlap(raw_truth, raw_median),
            "analysis_overlap": _normalized_overlap(
                whitened_truth, whitened_median
            ),
            "posterior_peak_offset_ms": float(
                time_ms[int(np.argmax(np.abs(whitened_median)))]
            ),
            "noise_rms": noise_rms,
        }

        raw_time, raw_truth_plot, raw_draws_plot = _visible(
            time_ms, raw_truth / truth_scale, raw_draws / truth_scale
        )
        analysis_time, analysis_truth_plot, analysis_data_plot, analysis_draws_plot = _visible(
            time_ms, whitened_truth, whitened_data, whitened_draws
        )

        for row, (time_plot, truth_plot, draws_plot) in enumerate(
            (
                (raw_time, raw_truth_plot, raw_draws_plot),
                (analysis_time, analysis_truth_plot, analysis_draws_plot),
            )
        ):
            ax = axes[row, column]
            lower, median, upper = np.percentile(draws_plot, [5, 50, 95], axis=0)
            ax.fill_between(
                time_plot, lower, upper, color=VAE_COLOR, alpha=0.28, lw=0,
                label="VAE 90% credible interval",
            )
            ax.plot(time_plot, median, color=VAE_COLOR, lw=1.1, label="VAE median")
            ax.plot(
                time_plot, truth_plot, color=TRUTH_COLOR, ls="--", lw=1.0,
                label="Injected signal",
            )
            ax.axhline(0.0, color="0.82", lw=0.5, zorder=0)
            ax.set_xlim(-35.0, 35.0)
            ax.text(
                0.02, 0.94, panel_labels[row][column], transform=ax.transAxes,
                va="top", ha="left",
            )
            if row == 0:
                ax.text(
                    0.98, 0.94, detector, transform=ax.transAxes,
                    va="top", ha="right", fontweight="bold",
                )
            if row == 1:
                ax.plot(
                    analysis_time, analysis_data_plot, color=DATA_COLOR,
                    lw=0.42, alpha=0.62, label="Detector data",
                )

    axes[0, 0].set_ylabel("Projected strain / injected peak")
    axes[1, 0].set_ylabel("Whitened strain (noise rms)")
    for ax in axes[1, :]:
        ax.set_xlabel("Time from injected peak (ms)")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(), unique.keys(), loc="upper center", ncol=4,
        frameon=False, bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91), h_pad=0.55, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-root", type=Path, required=True)
    parser.add_argument("--event-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make_figure(**vars(args))


if __name__ == "__main__":
    main()
