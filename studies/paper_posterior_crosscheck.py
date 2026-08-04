"""Build the manuscript's matched posterior-predictive cross-check.

The figure compares the coherent-signal posterior for one injected event that
has complete Starccato and BayesWave products.  The exact detector-projected
injection is recovered by subtracting the noise bundle from the injected
bundle.  By default, all traces in each detector are referenced to that
detector's injection peak: this makes the noisy data readable without aligning
either posterior to the truth.  Each pipeline retains its own whitening
convention; the panels therefore test posterior-predictive morphology and
timing, not equality of evidence scales or sample-by-sample waveform amplitudes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data

VAE_COLOR = "#0072B2"
BW_COLOR = "#D55E00"
TRUTH_COLOR = "#666666"
EVENT_CLASS_TO_BW_CAMPAIGN = {
    "inj_ccsn": "bwcomp_nml_v044_ccsn",
    "real_glitch": "bwcomp_nml_v044_glitch",
}


def _whiten(x: np.ndarray, psd: np.ndarray, dt: float) -> np.ndarray:
    """Match the whitening convention used by the saved VAE predictive."""
    n = np.asarray(x).size
    df = 1.0 / (n * dt)
    safe_psd = np.where(np.isfinite(psd) & (psd > 0), psd, np.inf)
    return np.fft.irfft(
        np.fft.rfft(np.asarray(x)) * dt / np.sqrt(safe_psd / (4.0 * df)),
        n=n,
    )


def _noise_scale(data: np.ndarray) -> float:
    scale = float(np.std(np.asarray(data)))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _visible(time: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Restrict plotted arrays to the displayed window before autoscaling."""
    mask = (np.asarray(time) >= -35.0) & (np.asarray(time) <= 35.0)
    return (np.asarray(time)[mask], *(np.asarray(value)[..., mask] for value in values))


def _injected_truth(vae_root: Path, event_index: int) -> dict[str, np.ndarray]:
    """Return the exact detector-projected injection in VAE-whitened units."""
    event_dir = vae_root / f"e{event_index}"
    manifest = json.loads((event_dir / "manifest.json").read_text())
    injection_paths = {
        detector: Path(path)
        for detector, path in manifest["bundles"]["inj_ccsn"].items()
    }
    noise_paths = {}
    for detector in ("H1", "L1"):
        matches = sorted((event_dir / "noise" / detector).glob("analysis_bundle_*.hdf5"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one noise bundle for {detector}, found {len(matches)}"
            )
        noise_paths[detector] = matches[0]

    flow, fmax = map(float, manifest["band"])
    injected = prepare_multi_detector_data(
        ("H1", "L1"), bundle_paths=injection_paths, flow=flow, fmax=fmax
    )
    noise = prepare_multi_detector_data(
        ("H1", "L1"), bundle_paths=noise_paths, flow=flow, fmax=fmax
    )

    truth = {}
    for detector in ("H1", "L1"):
        inj_data = injected.detector_data[detector]
        noise_data = noise.detector_data[detector]
        injection_td = np.asarray(inj_data.windowed_strain) - np.asarray(
            noise_data.windowed_strain
        )
        psd_band = np.asarray(inj_data.psd.values)
        sigma = _noise_scale(_whiten(inj_data.windowed_strain, psd_band, inj_data.dt))
        truth[f"{detector}_time"] = np.asarray(inj_data.time)
        truth[f"{detector}_whitened"] = (
            _whiten(injection_td, psd_band, inj_data.dt) / sigma
        )
    return truth


def make_figure(
    combined_h5: Path,
    vae_root: Path,
    event_index: int,
    output: Path,
    event_class: str = "inj_ccsn",
    time_reference: str = "auto",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vae_path = (
        vae_root
        / f"e{event_index}"
        / event_class
        / "analysis"
        / "signal"
        / "posterior_predictive.npz"
    )
    if not vae_path.is_file():
        raise FileNotFoundError(vae_path)

    truth = _injected_truth(vae_root, event_index) if event_class == "inj_ccsn" else None
    if time_reference == "auto":
        time_reference = "truth" if truth is not None else "trigger"
    if time_reference == "truth" and truth is None:
        raise ValueError("truth time reference is available only for injected signals")

    bw_campaign = EVENT_CLASS_TO_BW_CAMPAIGN[event_class]
    bw_path = f"bayeswave/{bw_campaign}/{event_class}/posteriors/e{event_index}"
    with np.load(vae_path) as vae, h5py.File(combined_h5, "r") as archive:
        if bw_path not in archive:
            raise KeyError(f"missing {bw_path} in {combined_h5}")
        bw = archive[bw_path]
        bw_time = np.asarray(bw["timesamp"])

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 7,
                "axes.labelsize": 7,
                "legend.fontsize": 5.8,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
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
        # Native single-column dimensions keep the labels legible in the paper;
        # scaling a two-column source down would produce roughly 4-point text.
        # One column, both models overlaid per detector. Side-by-side panels
        # make the reader eyeball two separate traces; the question this figure
        # answers is whether the two reconstructions agree, so they belong on
        # shared axes with the data behind them.
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.9), sharex=True, sharey=True)

        for det_i, detector in enumerate(("H1", "L1")):
            # Starccato posterior predictive: detector response, analysis PSD,
            # and 300--800 Hz mask were applied when the local fit was saved.
            v_data = np.asarray(vae[f"{detector}_whitened_data"])
            v_med = np.asarray(vae[f"{detector}_whitened_median"])
            v_lo = np.asarray(vae[f"{detector}_whitened_lower"])
            v_hi = np.asarray(vae[f"{detector}_whitened_upper"])
            # Both archives already use seconds relative to the common trigger.
            # Keep one reference for data, truth, and both posteriors: aligning
            # each posterior separately would erase a real timing disagreement.
            v_time_native = np.asarray(vae[f"{detector}_time"]) * 1e3
            v_scale = _noise_scale(v_data)

            # BayesWavePost saves 50 whitened signal-waveform draws on the
            # corresponding 2048 Hz grid.
            b_data = np.asarray(bw[f"whitened_data_{detector}"])
            b_draws = np.asarray(bw[f"whitened_waveform_draws_{detector}"])
            b_lo, b_med, b_hi = np.percentile(b_draws, [5, 50, 95], axis=0)
            b_time_native = bw_time * 1e3
            b_scale = _noise_scale(b_data)

            if truth is not None:
                truth_time_native = np.asarray(truth[f"{detector}_time"]) * 1e3
                truth_waveform = np.asarray(truth[f"{detector}_whitened"])
                truth_peak = truth_time_native[int(np.argmax(np.abs(truth_waveform)))]
            else:
                truth_time_native = None
                truth_waveform = None
                truth_peak = 0.0
            offset = truth_peak if time_reference == "truth" else 0.0
            v_time = v_time_native - offset
            b_time = b_time_native - offset
            truth_time = truth_time_native - offset if truth_time_native is not None else None

            ax = axes[det_i]
            for time, lo, hi, scale, color, label in (
                (v_time, v_lo, v_hi, v_scale, VAE_COLOR,
                 "VAE 90% credible interval"),
                (b_time, b_lo, b_hi, b_scale, BW_COLOR,
                 "BayesWave 90% credible interval"),
            ):
                time_plot, lo_plot, hi_plot = _visible(time, lo / scale, hi / scale)
                ax.fill_between(
                    time_plot,
                    lo_plot,
                    hi_plot,
                    color=color,
                    alpha=0.32,
                    lw=0,
                    zorder=1,
                    label=label,
                )
            if truth_time is not None and truth_waveform is not None:
                truth_time_plot, truth_plot = _visible(
                    truth_time, truth_waveform / v_scale
                )
                ax.plot(
                    truth_time_plot,
                    truth_plot,
                    color=TRUTH_COLOR,
                    ls="--",
                    lw=1.0,
                    alpha=0.95,
                    zorder=3,
                    label="Injected signal",
                )
            # Each pipeline whitens with its own PSD, so the two data traces
            # are not identical; showing the VAE-side one is enough for
            # context. Drawn last, thin and dark, so it stays visible where
            # it nearly coincides with the reconstructions.
            data_time_plot, data_plot = _visible(v_time, v_data / v_scale)
            ax.plot(
                data_time_plot,
                data_plot,
                color="black",
                lw=0.45,
                alpha=0.70,
                zorder=2,
                label="Detector data",
            )
            # A per-axis "H1 whitened strain [sigma]" / "L1 whitened strain
            # [sigma]" label repeated the units twice and collided between
            # the stacked subplots; one shared axis label plus a small
            # in-panel detector tag reads more cleanly.
            ax.text(0.015, 0.90, detector, transform=ax.transAxes,
                    fontsize=7, fontweight="bold", va="top")

            vae_peak = v_time_native[int(np.argmax(np.abs(v_med)))]
            bw_peak = b_time_native[int(np.argmax(np.abs(b_med)))]
            if truth is not None:
                print(
                    f"{detector} peak times [ms]: truth={truth_peak:.2f}, "
                    f"VAE={vae_peak:.2f}, BayesWave={bw_peak:.2f}"
                )

        xlabel = (
            "Time from injected peak (ms)"
            if time_reference == "truth"
            else "Time from trigger (ms)"
        )
        fig.supxlabel(xlabel, fontsize=7, y=0.005)
        fig.supylabel("Whitened strain (noise rms)", fontsize=7, x=0.005)
        for ax in axes.flat:
            ax.set_xlim(-35, 35)
            ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.5,
        )
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.84), h_pad=0.45)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-h5", type=Path, required=True)
    parser.add_argument("--vae-root", type=Path, required=True)
    parser.add_argument("--event-index", type=int, default=618)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-class",
        choices=tuple(EVENT_CLASS_TO_BW_CAMPAIGN),
        default="inj_ccsn",
    )
    parser.add_argument(
        "--time-reference",
        choices=("auto", "truth", "trigger"),
        default="auto",
        help="common time origin for all traces in each detector",
    )
    args = parser.parse_args()
    make_figure(**vars(args))


if __name__ == "__main__":
    main()
