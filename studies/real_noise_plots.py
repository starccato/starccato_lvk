"""Publication figures + LaTeX table for the real-noise analysis study.

Reads the per-event JSONs written by ``real_noise_event.py`` (fields: cls, snr,
logZ_signal, logZ_glitch, log_odds) for the single-detector (L1) and/or
two-detector (H1 L1) runs, and produces:

  fig_roc.pdf          ROC curves -- odds vs SNR, one panel per network.
  fig_score_dist.pdf   logBCR distributions across the four event classes.
  fig_odds_vs_snr.pdf  logBCR vs injected SNR, coloured by class (separation is
                       morphological, not amplitude-driven).
  fig_efficiency.pdf   signal detection efficiency vs injected SNR at a fixed
                       false-alarm rate, odds vs SNR.
  summary_table.tex    AUCs + misclassification rates for the manuscript.

    uv run python studies/real_noise_plots.py --l1 slurm/out/rn_L1 --h1l1 slurm/out/rn_H1_L1

Either --l1 or --h1l1 may be omitted; whatever is present is plotted.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snr_vs_odds_roc import _roc_auc
from real_noise_aggregate import _boot_auc_err

# Publication style: figures are included at \textwidth (figure*) in the
# two-column AASTeX manuscript, so size fonts for a ~7 in wide figure.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
})
PANEL_W, PANEL_H = 3.5, 2.9

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")
BACKGROUND = ("noise", "inj_glitch", "real_glitch")
LABELS = {
    "noise": "noise",
    "inj_ccsn": "injected CCSN",
    "inj_glitch": "injected glitch",
    "real_glitch": "real glitch",
}
COLORS = {
    "inj_ccsn": "#1b7837",       # green  -- signal
    "noise": "#4d4d4d",          # grey   -- noise
    "inj_glitch": "#2166ac",     # blue   -- synthetic glitch
    "real_glitch": "#b2182b",    # red    -- real glitch
}


def load_run(outdir: Path) -> Optional[Dict[str, dict]]:
    """Load per-event rows keyed by class -> arrays. Returns None if empty."""
    combined = outdir / "results.json"  # written by collect_results.py
    if combined.exists():
        rows = json.loads(combined.read_text())
    else:
        files = glob.glob(str(outdir / "results" / "e*_*.json"))
        rows = [json.loads(Path(f).read_text()) for f in files]
    if not rows:
        return None
    out = {}
    for c in CLASSES:
        sub = [r for r in rows if r["cls"] == c]
        out[c] = {
            "log_odds": np.array([r["log_odds"] for r in sub], dtype=float),
            "snr": np.array([r["snr"] for r in sub], dtype=float),
        }
    out["_n"] = len(rows)
    return out


def _roc_curve(pos: np.ndarray, neg: np.ndarray):
    """Exact ROC step curve: one vertex per distinct score, no threshold grid.

    The previous linspace-threshold version placed most of its 200 thresholds
    in the empty tails of the +-600 log-odds range, producing a kinked polyline.
    """
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    scores = np.concatenate([pos, neg])
    is_pos = np.concatenate([np.ones(pos.size, bool), np.zeros(neg.size, bool)])
    order = np.argsort(-scores, kind="stable")
    scores, is_pos = scores[order], is_pos[order]
    tpr = np.cumsum(is_pos) / pos.size
    fpr = np.cumsum(~is_pos) / neg.size
    distinct = np.r_[np.diff(scores) != 0, True]  # collapse tied scores
    return np.r_[0.0, fpr[distinct]], np.r_[0.0, tpr[distinct]]


def _background(run: dict, key: str) -> np.ndarray:
    return np.concatenate([run[c][key] for c in BACKGROUND])


def fig_roc(runs: Dict[str, dict], out: Path) -> None:
    """ROC on a log false-alarm-probability axis (the low-FAP regime is what a
    search operates in; linear axes hide it)."""
    fig, axes = plt.subplots(1, len(runs), figsize=(PANEL_W * len(runs), PANEL_H), squeeze=False)
    for ax, (net, run) in zip(axes[0], runs.items()):
        sig_o, sig_s = run["inj_ccsn"]["log_odds"], run["inj_ccsn"]["snr"]
        bg_o, bg_s = _background(run, "log_odds"), _background(run, "snr")
        fap_min = 1.0 / bg_o[np.isfinite(bg_o)].size  # resolution of the background set
        for sig, bg, color, name in (
            (sig_o, bg_o, "#1b7837", r"$\ln\,\mathcal{O}$"),
            (sig_s, bg_s, "#762a83", "SNR"),
        ):
            fpr, tpr = _roc_curve(sig, bg)
            auc, err = _roc_auc(sig, bg), _boot_auc_err(sig, bg)
            ax.plot(np.clip(fpr, fap_min, None), tpr, drawstyle="steps-post", color=color,
                    label=rf"{name}  (AUC $= {auc:.3f} \pm {err:.3f}$)")
        diag = np.geomspace(fap_min, 1.0, 50)
        ax.plot(diag, diag, ls=":", color="k", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("false-alarm probability")
        ax.set_ylabel("detection efficiency")
        ax.set_title(net)
        ax.set_xlim(fap_min, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_score_dist(runs: Dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(PANEL_W * len(runs), PANEL_H), squeeze=False)
    for ax, (net, run) in zip(axes[0], runs.items()):
        allvals = np.concatenate([run[c]["log_odds"][np.isfinite(run[c]["log_odds"])] for c in CLASSES])
        lo, hi = np.percentile(allvals, [1, 99])
        bins = np.linspace(lo, hi, 45)
        for c in CLASSES:
            v = run[c]["log_odds"]
            v = v[np.isfinite(v)]
            ax.hist(np.clip(v, lo, hi), bins=bins, histtype="step", lw=2,
                    density=True, color=COLORS[c], label=LABELS[c])
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.7)
        ax.set_xlabel(r"$\ln\,\mathcal{O}$  (log Bayesian odds)")
        ax.set_ylabel("density")
        ax.set_title(f"{net} network")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_odds_vs_snr(runs: Dict[str, dict], out: Path) -> None:
    """Symlog y-axis: the decision-relevant structure lives at |ln O| ~ 10,
    which a linear +-600 axis squashes into the ln O = 0 line."""
    fig, axes = plt.subplots(1, len(runs), figsize=(PANEL_W * len(runs), PANEL_H), squeeze=False)
    for i, (ax, (net, run)) in enumerate(zip(axes[0], runs.items())):
        for c in CLASSES:
            snr, odds = run[c]["snr"], run[c]["log_odds"]
            m = np.isfinite(snr) & np.isfinite(odds)
            ax.scatter(snr[m], odds[m], s=7, alpha=0.55, color=COLORS[c],
                       edgecolors="none", rasterized=True, label=LABELS[c])
        ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.7)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("matched-filter SNR")
        ax.set_ylabel(r"$\ln\,\mathcal{O}$")
        ax.set_title(net)
        if i == 0:
            ax.legend(frameon=False, loc="upper right", markerscale=1.6, handletextpad=0.1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _wilson(k: int, n: int, z: float = 1.0):
    """Wilson score interval for a binomial proportion (sane at eff = 0 or 1)."""
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def fig_efficiency(runs: Dict[str, dict], out: Path, far: float = 0.05) -> None:
    """Detection efficiency vs injected SNR at fixed false-alarm rate `far`."""
    fig, axes = plt.subplots(1, len(runs), figsize=(PANEL_W * len(runs), PANEL_H), squeeze=False)
    edges = np.array([0, 8, 12, 16, 24, 32, 48, 100], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for ax, (net, run) in zip(axes[0], runs.items()):
        for score, color, name in (("log_odds", "#1b7837", r"$\ln\,\mathcal{O}$"),
                                    ("snr", "#762a83", "SNR")):
            bg = _background(run, score)
            bg = bg[np.isfinite(bg)]
            thr = np.quantile(bg, 1.0 - far)  # threshold giving the target FAR
            sig_snr = run["inj_ccsn"]["snr"]
            sig_val = run["inj_ccsn"][score]
            eff, lo, hi, xs = [], [], [], []
            for i in range(len(edges) - 1):
                m = (sig_snr >= edges[i]) & (sig_snr < edges[i + 1]) & np.isfinite(sig_val)
                if m.sum() >= 3:
                    k, n = int((sig_val[m] >= thr).sum()), int(m.sum())
                    eff.append(k / n)
                    l, h = _wilson(k, n)
                    lo.append(l)
                    hi.append(h)
                    xs.append(centers[i])
            eff, lo, hi = np.array(eff), np.array(lo), np.array(hi)
            ax.errorbar(xs, eff, yerr=[eff - lo, hi - eff], fmt="-o", ms=3.5,
                        capsize=2, color=color, label=name)
        ax.set_xlabel("injected network SNR")
        ax.set_ylabel(f"efficiency at {far:.0%} false-alarm probability")
        ax.set_title(net)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_table(runs: Dict[str, dict], out: Path) -> None:
    lines = [
        r"\begin{tabular}{l" + "c" * len(runs) + "}",
        r"\hline",
        "metric & " + " & ".join(runs.keys()) + r" \\",
        r"\hline",
    ]

    def row(name, fn, fmt="{:.3f}"):
        return f"{name} & " + " & ".join(fmt.format(fn(r)) for r in runs.values()) + r" \\"

    def auc_row(name, fg_key, bg_fn):
        r"""AUC row with a bootstrap standard error: 0.975 \pm 0.004."""
        def cell(r):
            fg, bg = r["inj_ccsn"][fg_key], bg_fn(r)
            return f"${_roc_auc(fg, bg):.3f} \\pm {_boot_auc_err(fg, bg):.3f}$"

        return f"{name} & " + " & ".join(cell(r) for r in runs.values()) + r" \\"

    lines += [
        row("$N$ events", lambda r: r["_n"], "{:d}"),
        auc_row(r"AUC$_{\ln\mathcal{O}}$ (sig vs.\ bkg)",
                "log_odds", lambda r: _background(r, "log_odds")),
        auc_row("AUC$_{\\rm SNR}$ (sig vs.\\ bkg)",
                "snr", lambda r: _background(r, "snr")),
        auc_row(r"AUC$_{\ln\mathcal{O}}$ (sig vs.\ real glitch)",
                "log_odds", lambda r: r["real_glitch"]["log_odds"]),
        row("real glitch misclass.\\ rate",
            lambda r: (r["real_glitch"]["log_odds"] > 0).mean()),
        row("signal missed rate",
            lambda r: (r["inj_ccsn"]["log_odds"] < 0).mean()),
        r"\hline",
        r"\end{tabular}",
    ]
    out.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--l1", type=Path, default=None, help="Single-detector run dir (rn_L1).")
    p.add_argument("--h1l1", type=Path, default=None, help="Two-detector run dir (rn_H1_L1).")
    p.add_argument("--outdir", type=Path, default=Path("out_rn_figs"))
    args = p.parse_args()

    runs: Dict[str, dict] = {}
    if args.l1 is not None:
        r = load_run(args.l1)
        if r:
            runs["L1"] = r
    if args.h1l1 is not None:
        r = load_run(args.h1l1)
        if r:
            runs["H1L1"] = r
    if not runs:
        raise SystemExit("No result JSONs found in the given --l1 / --h1l1 dirs.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_roc(runs, args.outdir / "fig_roc.pdf")
    fig_score_dist(runs, args.outdir / "fig_score_dist.pdf")
    fig_odds_vs_snr(runs, args.outdir / "fig_odds_vs_snr.pdf")
    fig_efficiency(runs, args.outdir / "fig_efficiency.pdf")
    write_table(runs, args.outdir / "summary_table.tex")

    print(f"Wrote figures + table to {args.outdir}/ for networks: {', '.join(runs)}")
    for net, r in runs.items():
        auc_o = _roc_auc(r["inj_ccsn"]["log_odds"], _background(r, "log_odds"))
        auc_s = _roc_auc(r["inj_ccsn"]["snr"], _background(r, "snr"))
        print(f"  {net}: N={r['_n']}  AUC(odds)={auc_o:.3f}  AUC(SNR)={auc_s:.3f}")


if __name__ == "__main__":
    main()
