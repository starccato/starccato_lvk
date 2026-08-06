"""Three-class posterior-predictive figure for the manuscript.

One trigger index, all three event classes (noise, cataloged blip, injected
CCSN), analysed in the H1--L1 configuration.  Each panel overlays the whitened
strain with the coherent-signal posterior band and that detector's independent
glitch posterior band, so the coherence argument is visible: the blip is fit by
one detector's glitch model while the coherent signal model cannot explain it in
both.

    ./.venv/bin/python studies/paper_three_class_ppc.py \
        --vae-root ../local_ppc_paperfig/rn_H1_L1 --event-index 0 \
        --output ../manuscript/figures/fig_three_class_ppc.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper_posterior_crosscheck import _injected_truth

DETECTORS = ("H1", "L1")
# (directory, row label) in the order the manuscript discusses them.
CASES = (
    ("noise", "Noise"),
    ("real_glitch", "Cataloged blip"),
    ("inj_ccsn", "Injected CCSN"),
)
SIGNAL_COLOR = "#0072B2"
GLITCH_COLOR = "#D55E00"
DATA_COLOR = "#9a9a9a"
TRUTH_COLOR = "#111111"
WINDOW_MS = 35.0


def _predictive(analysis: Path, hypothesis: str, detector: str) -> dict[str, np.ndarray]:
    path = analysis / hypothesis / "posterior_predictive.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as d:
        if f"{detector}_time" not in d:
            raise KeyError(f"{path} has no {detector} predictive")
        return {k: np.asarray(d[f"{detector}_{k}"])
                for k in ("time", "whitened_data", "whitened_median",
                          "whitened_lower", "whitened_upper")}


def _time_ms(time: np.ndarray) -> np.ndarray:
    """Milliseconds relative to the segment centre (the trigger time)."""
    t = np.asarray(time, dtype=float)
    return (t - 0.5 * (t[0] + t[-1])) * 1e3


def _arrival_ms(manifest: dict) -> dict[str, float]:
    """Geocenter-to-detector delay for the event's fixed sky position.

    The signal hypothesis is evaluated at t_c + dt_i, the glitch hypothesis at
    the trigger itself, so these markers say where each detector's coherent
    CCSN burst is allowed to sit.
    """
    import bilby

    sky = manifest["sky"]
    gps = float(manifest["gps"]["blip"])
    return {
        det: bilby.gw.detector.get_empty_interferometer(det).time_delay_from_geocenter(
            sky["ra"], sky["dec"], gps
        ) * 1e3
        for det in DETECTORS
    }


def _band(ax, time_ms, pred, color, label) -> None:
    ax.fill_between(time_ms, pred["whitened_lower"], pred["whitened_upper"],
                    color=color, alpha=0.30, lw=0)
    ax.plot(time_ms, pred["whitened_median"], color=color, lw=1.1, label=label)


def make_figure(vae_root: Path, event_index: int, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event_dir = vae_root / f"e{event_index}"
    manifest = json.loads((event_dir / "manifest.json").read_text())
    truth = _injected_truth(vae_root, event_index)
    arrival_ms = _arrival_ms(manifest)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(len(CASES), len(DETECTORS), figsize=(7.0, 5.2),
                             sharex=True)

    for row, (case, row_label) in enumerate(CASES):
        analysis = event_dir / case / "analysis"
        for col, detector in enumerate(DETECTORS):
            ax = axes[row, col]
            signal = _predictive(analysis, "signal", detector)
            glitch = _predictive(analysis, f"glitch_{detector.lower()}", detector)
            time_ms = _time_ms(signal["time"])
            keep = np.abs(time_ms) <= WINDOW_MS

            def clip(pred):
                return {k: (v[keep] if v.ndim else v) for k, v in pred.items()}

            ax.plot(time_ms[keep], signal["whitened_data"][keep],
                    color=DATA_COLOR, lw=0.6, label="Whitened data")
            if case == "inj_ccsn":
                truth_ms = _time_ms(truth[f"{detector}_time"])
                tkeep = np.abs(truth_ms) <= WINDOW_MS
                ax.plot(truth_ms[tkeep], truth[f"{detector}_whitened"][tkeep],
                        color=TRUTH_COLOR, lw=0.9, ls="--", label="Injection")
            _band(ax, time_ms[keep], clip(signal), SIGNAL_COLOR,
                  r"Coherent signal $\mathcal{H}_S$")
            _band(ax, time_ms[keep], clip(glitch), GLITCH_COLOR,
                  rf"{detector} glitch $\mathcal{{H}}_G$")

            ax.axvline(0.0, color="0.35", lw=0.6, ls=":")
            ax.axvline(arrival_ms[detector], color=SIGNAL_COLOR, lw=0.6, ls=":")

            ax.set_xlim(-WINDOW_MS, WINDOW_MS)
            if row == 0:
                ax.set_title(detector, fontsize=7.5)
            if row == len(CASES) - 1:
                ax.set_xlabel("Time from trigger [ms]")
            if col == 0:
                ax.set_ylabel(rf"{row_label}" "\n" r"whitened strain [$\sigma$]")
            ax.legend(loc="upper left", frameon=False, handlelength=1.4,
                      ncol=2, columnspacing=0.9)

    blip_ifo = manifest.get("blip_ifo", "L1")
    fig.suptitle(
        f"Event {event_index}: H1--L1 posterior predictives "
        f"(blip host {blip_ifo}, 300--800 Hz)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vae-root", type=Path, required=True)
    p.add_argument("--event-index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    make_figure(args.vae_root, args.event_index, args.output)


if __name__ == "__main__":
    main()
