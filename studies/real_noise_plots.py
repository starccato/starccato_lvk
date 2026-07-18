"""Publication figures + LaTeX table for the real-noise analysis study.

Reads the per-event JSONs written by ``real_noise_event.py`` (fields: cls, snr,
logZ_signal, logZ_glitch, log_odds) for the single-detector (L1) and/or
two-detector (H1 L1) runs, and produces:

  fig_roc.pdf          ROC curves -- odds vs a descriptive loudness proxy,
                       one panel per network.
  fig_score_dist.pdf   logBCR distributions across the four event classes.
  fig_odds_vs_snr.pdf  logBCR vs transient loudness, with explicit treatment of
                       the incompatible injection and catalogue SNR definitions.
  fig_efficiency.pdf   signal detection efficiency vs injected SNR at a fixed
                       false-alarm rate when ranking by the odds.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from snr_vs_odds_roc import _roc_auc
from real_noise_aggregate import _boot_auc_err

# Publication style for figures included in the two-column AASTeX manuscript.
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
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
})
PANEL_W, PANEL_H = 3.5, 2.9

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")
BACKGROUND = ("noise", "inj_glitch", "real_glitch")
LABELS = {
    "noise": "noise",
    "inj_ccsn": "injected CCSN",
    "inj_glitch": "injected blip",
    "real_glitch": "real blip",
}
COLORS = {
    "inj_ccsn": "#0072B2",       # Okabe--Ito blue
    "noise": "#666666",          # neutral reference
    "inj_glitch": "#E69F00",     # Okabe--Ito orange
    "real_glitch": "#D55E00",    # Okabe--Ito vermillion
}

MARKERS = {
    "inj_ccsn": "o",
    "inj_glitch": "v",
    "real_glitch": "D",
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
    # Optional reweighted-SNR baseline (chisq_baseline.py): new_snr keyed by
    # (cls, index) so it aligns to the SAME events as the odds -- events analysed
    # for odds but not baseline (or vice versa) get NaN and drop from newSNR AUCs.
    base = {}
    for f in glob.glob(str(outdir / "results" / "e*_baseline.json")):
        r = json.loads(Path(f).read_text())
        base[(r["cls"], int(r["index"]))] = float(r["new_snr"])
    out = {}
    for c in CLASSES:
        sub = [r for r in rows if r["cls"] == c]
        out[c] = {
            "log_odds": np.array([r["log_odds"] for r in sub], dtype=float),
            "snr": np.array([r["snr"] for r in sub], dtype=float),
            "index": np.array([r["index"] for r in sub], dtype=int),
        }
        if base:
            out[c]["new_snr"] = np.array(
                [base.get((c, int(r["index"])), np.nan) for r in sub], dtype=float)
    out["_has_baseline"] = bool(base)
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


def _roc_band(pos, neg, pos_idx, neg_idx, fap_grid, n_boot=400, seed=0):
    """Block-bootstrap ROC as (median, lo, hi) detection efficiency on fap_grid.

    Resamples whole events (the four classes at a trigger share a segment), draws
    the exact ROC each time and evaluates it on a common false-alarm grid. The
    band replaces the raw staircase: a near-perfect classifier on a finite
    background yields big discrete steps that read as artefacts, so we show the
    bootstrap median and central 68% interval instead.
    """
    m = np.isfinite(pos); pos, pos_idx = pos[m], pos_idx[m]
    m = np.isfinite(neg); neg, neg_idx = neg[m], neg_idx[m]
    pos_by = {e: pos[pos_idx == e] for e in np.unique(pos_idx)}
    neg_by = {e: neg[neg_idx == e] for e in np.unique(neg_idx)}
    events = np.unique(np.concatenate([pos_idx, neg_idx]))
    rng = np.random.default_rng(seed)
    empty = np.empty(0)
    tprs = []
    for _ in range(n_boot):
        pick = rng.choice(events, events.size, replace=True)
        p = np.concatenate([pos_by.get(e, empty) for e in pick])
        n = np.concatenate([neg_by.get(e, empty) for e in pick])
        if p.size and n.size:
            fpr, tpr = _roc_curve(p, n)
            tprs.append(np.interp(fap_grid, fpr, tpr))
    return np.percentile(np.asarray(tprs), [50, 16, 84], axis=0)


def fig_roc(runs: Dict[str, dict], out: Path) -> None:
    """ROC on a log false-alarm-probability axis (the low-FAP regime is what a
    search operates in; linear axes hide it). Bootstrap median + 68% band."""
    fig, axes = plt.subplots(1, len(runs), figsize=(PANEL_W * len(runs), PANEL_H), squeeze=False)
    for ax, (net, run) in zip(axes[0], runs.items()):
        sig_o, sig_s = run["inj_ccsn"]["log_odds"], run["inj_ccsn"]["snr"]
        bg_o, bg_s = _background(run, "log_odds"), _background(run, "snr")
        sig_i, bg_i = run["inj_ccsn"]["index"], _background(run, "index")
        fap_min = 1.0 / bg_o[np.isfinite(bg_o)].size  # resolution of the background set
        fap_grid = np.geomspace(fap_min, 1.0, 200)
        curves = [
            (sig_o, bg_o, sig_i, bg_i, "#1b7837", r"$\ln\,\mathcal{O}$"),
            (sig_s, bg_s, sig_i, bg_i, "#762a83", "SNR proxy"),
        ]
        if run.get("_has_baseline"):
            curves.append((run["inj_ccsn"]["new_snr"], _background(run, "new_snr"),
                           sig_i, bg_i, "#d95f02", "reweighted SNR"))
        for sig, bg, si, bi, color, name in curves:
            med, lo, hi = _roc_band(sig, bg, si, bi, fap_grid)
            auc, err = _roc_auc(sig, bg), _boot_auc_err(sig, bg, si, bi)
            ax.fill_between(fap_grid, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(fap_grid, med, color=color, lw=1.7,
                    label=rf"{name}  (AUC $= {auc:.3f} \pm {err:.3f}$)")
        ax.plot(fap_grid, fap_grid, ls=":", color="k", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("false-alarm probability")
        ax.set_ylabel("detection efficiency")
        ax.set_title("H1-L1" if net == "H1L1" else net)
        ax.set_xlim(fap_min, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", frameon=False)
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
        display_net = "H1-L1" if net == "H1L1" else net
        ax.set_title(f"{display_net} network")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_odds_vs_snr(runs: Dict[str, dict], out: Path) -> None:
    """Odds versus loudness without conflating incompatible SNR definitions.

    Rows identify detector configurations. Columns separate injections, whose
    x coordinate is optimal network SNR, from real blips, whose x coordinate is
    the Gravity Spy catalogue SNR. Faint points show the observations; opaque
    markers and bars show binned medians and central 80% intervals.
    """

    def binned_summary(ax, snr, odds, edges, color, marker):
        centers, medians, lower, upper = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            select = (snr >= lo) & (snr < hi) & np.isfinite(snr) & np.isfinite(odds)
            if select.sum() < 5:
                continue
            values = odds[select]
            centers.append(np.median(snr[select]))
            medians.append(np.median(values))
            q10, q90 = np.percentile(values, [10, 90])
            lower.append(medians[-1] - q10)
            upper.append(q90 - medians[-1])
        if not centers:
            return
        ax.errorbar(
            centers,
            medians,
            yerr=np.array([lower, upper]),
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=4.2,
            lw=1.2,
            capsize=2.0,
            zorder=4,
        )

    fig, axes = plt.subplots(
        len(runs), 2, figsize=(7.05, 4.55), sharey=True, squeeze=False,
    )
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    injection_edges = np.array([10, 14, 18, 22, 26, 30, 34, 40.01])
    catalogue_edges = np.array([12, 16, 22, 30, 45, 70, 110, 190])
    y_ticks = [-600, -100, -10, 0, 10, 100, 600]
    y_ticklabels = [r"$-600$", r"$-100$", r"$-10$", "$0$", "$10$", "$100$", "$600$"]

    for row, (net, run) in enumerate(runs.items()):
        display_net = "H1-L1" if net == "H1L1" else net
        noise = run["noise"]["log_odds"]
        noise = noise[np.isfinite(noise)]
        noise_lo, noise_hi = np.percentile(noise, [5, 95])

        for col, ax in enumerate(axes[row]):
            ax.axhspan(noise_lo, noise_hi, color="#BDBDBD", alpha=0.5,
                       lw=0, zorder=0)
            ax.axhline(0, color="#333333", ls="--", lw=0.9, zorder=2)
            ax.set_yscale("symlog", linthresh=10, linscale=0.8)
            ax.set_ylim(-800, 700)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels)
            ax.grid(axis="y", color="#D9D9D9", lw=0.5, alpha=0.75,
                    zorder=-1)
            ax.tick_params(length=3, width=0.7)
            panel = 2 * row + col
            panel_kind = "injections" if col == 0 else "catalogued blips"
            ax.set_title(
                f"({panel_letters[panel]}) {display_net}: {panel_kind}",
                loc="left",
            )

        ax_inj, ax_real = axes[row]
        for c in ("inj_ccsn", "inj_glitch"):
            snr, odds = run[c]["snr"], run[c]["log_odds"]
            finite = np.isfinite(snr) & np.isfinite(odds)
            ax_inj.scatter(
                snr[finite], odds[finite], s=5, marker=MARKERS[c],
                color=COLORS[c], edgecolors="none", alpha=0.12,
                rasterized=True, zorder=1,
            )
            binned_summary(
                ax_inj, snr, odds, injection_edges, COLORS[c], MARKERS[c],
            )

        snr = run["real_glitch"]["snr"]
        odds = run["real_glitch"]["log_odds"]
        finite = np.isfinite(snr) & np.isfinite(odds)
        ax_real.scatter(
            snr[finite], odds[finite], s=7, marker=MARKERS["real_glitch"],
            facecolors="none", edgecolors=COLORS["real_glitch"],
            linewidths=0.45, alpha=0.22, rasterized=True, zorder=1,
        )
        binned_summary(
            ax_real, snr, odds, catalogue_edges,
            COLORS["real_glitch"], MARKERS["real_glitch"],
        )

        ax_inj.set_xlim(9, 41)
        ax_inj.set_xticks([10, 20, 30, 40])
        ax_real.set_xscale("log")
        ax_real.set_xlim(11, 195)
        ax_real.set_xticks([12, 20, 40, 80, 160])
        ax_real.set_xticklabels(["12", "20", "40", "80", "160"])

    axes[-1, 0].set_xlabel("optimal network SNR")
    axes[-1, 1].set_xlabel("Gravity Spy catalogue SNR")

    handles = [
        Line2D([], [], marker=MARKERS["inj_ccsn"], ls="none", ms=5,
               color=COLORS["inj_ccsn"], label=LABELS["inj_ccsn"]),
        Line2D([], [], marker=MARKERS["inj_glitch"], ls="none", ms=5,
               color=COLORS["inj_glitch"], label=LABELS["inj_glitch"]),
        Line2D([], [], marker=MARKERS["real_glitch"], ls="none", ms=5,
               markerfacecolor="none", markeredgecolor=COLORS["real_glitch"],
               label=LABELS["real_glitch"]),
        Patch(facecolor="#BDBDBD", edgecolor="none", alpha=0.6,
              label="noise central 90%"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=4, frameon=False, fontsize=8, numpoints=1, handlelength=1.2,
               handletextpad=0.4, columnspacing=0.8)
    fig.supylabel(r"log Bayesian odds, $\ln\mathcal{O}$", x=0.008)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.11, top=0.89,
                        hspace=0.35, wspace=0.17)
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
    """Odds-ranked detection efficiency vs injected SNR at fixed FAP."""
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    edges = np.array([0, 8, 12, 16, 24, 32, 48, 100], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    network_styles = {
        "L1": dict(color="#0072B2", marker="o", ls="-"),
        "H1L1": dict(color="#D55E00", marker="s", ls="--"),
    }
    for net, run in runs.items():
        score = "log_odds"
        bg = _background(run, score)
        bg = bg[np.isfinite(bg)]
        thr = np.quantile(bg, 1.0 - far)
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
        yerr = np.clip([eff - lo, hi - eff], 0, None)
        style = network_styles.get(net, {})
        label = "H1-L1" if net == "H1L1" else net
        ax.errorbar(xs, eff, yerr=yerr, ms=4, capsize=2, label=label,
                    markerfacecolor="white", markeredgewidth=0.9, **style)
    ax.set_xlabel("injected network SNR")
    ax.set_ylabel(f"detection efficiency ({far:.0%} FAP)")
    ax.set_ylim(0.7, 1.0)
    ax.grid(axis="y", color="#D9D9D9", lw=0.55)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_table(runs: Dict[str, dict], out: Path) -> None:
    lines = [
        r"\begin{tabular}{l" + "c" * len(runs) + "}",
        r"\hline",
        "metric & " + " & ".join(
            "H1--L1" if key == "H1L1" else key for key in runs
        ) + r" \\",
        r"\hline",
    ]

    def row(name, fn, fmt="{:.3f}"):
        return f"{name} & " + " & ".join(fmt.format(fn(r)) for r in runs.values()) + r" \\"

    def auc_row(name, fg_key, bg_fn, bg_idx_fn):
        r"""AUC row with a block-bootstrap standard error: 0.975 \pm 0.004."""
        def cell(r):
            fg, bg = r["inj_ccsn"][fg_key], bg_fn(r)
            err = _boot_auc_err(fg, bg, r["inj_ccsn"]["index"], bg_idx_fn(r))
            return f"${_roc_auc(fg, bg):.3f} \\pm {err:.3f}$"

        return f"{name} & " + " & ".join(cell(r) for r in runs.values()) + r" \\"

    lines += [
        row("$N$ events", lambda r: r["_n"], "{:d}"),
        auc_row(r"AUC$_{\ln\mathcal{O}}$ (all bkg.)",
                "log_odds", lambda r: _background(r, "log_odds"), lambda r: _background(r, "index")),
        auc_row("AUC$_{\\rm loudness}$ (all bkg.)",
                "snr", lambda r: _background(r, "snr"), lambda r: _background(r, "index")),
    ]
    # Reweighted-SNR baseline (chisq_baseline.py): the consistently-evaluated
    # search statistic. Only emitted when every run has the baseline JSONs.
    if all(r.get("_has_baseline") for r in runs.values()):
        lines += [
            auc_row(r"AUC$_{\rm newSNR}$ (all bkg.)",
                    "new_snr", lambda r: _background(r, "new_snr"), lambda r: _background(r, "index")),
            auc_row(r"AUC$_{\rm newSNR}$ (real blip)",
                    "new_snr", lambda r: r["real_glitch"]["new_snr"], lambda r: r["real_glitch"]["index"]),
        ]
    lines += [
        auc_row(r"AUC$_{\ln\mathcal{O}}$ (noise)",
                "log_odds", lambda r: r["noise"]["log_odds"], lambda r: r["noise"]["index"]),
        auc_row(r"AUC$_{\ln\mathcal{O}}$ (inj.\ blip)",
                "log_odds", lambda r: r["inj_glitch"]["log_odds"], lambda r: r["inj_glitch"]["index"]),
        auc_row(r"AUC$_{\ln\mathcal{O}}$ (real blip)",
                "log_odds", lambda r: r["real_glitch"]["log_odds"], lambda r: r["real_glitch"]["index"]),
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
