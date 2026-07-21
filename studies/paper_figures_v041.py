"""Manuscript figures and tables from a downloaded campaign bundle.

Reads ``campaign_event_metrics.csv`` (one row per class/event/configuration) and
writes the publication figures plus ``summary_table.tex`` and ``metrics.json``:

    fig_confusion.pdf     three-class confusion matrices (CCSN / blip / noise)
    fig_roc.pdf           ROC (bootstrap median + 68% band), ln O versus the
                          reweighted matched-filter statistic SNR* (displayed
                          as such; the CSV/JSON field remains "new_snr")
    fig_efficiency.pdf    detection efficiency versus injected network SNR
    summary_table.tex     AUC + misclassification table for \\input{}
    metrics.json          every number quoted in the text

Usage:
    uv run python studies/paper_figures_v041.py \\
        --bundle ../nuts_morphlnz_v041_analysis_bundle \\
        --outdir ../manuscript/figures

All statistics use the *paired* population: events analysed in both the one- and
two-detector configuration, so the comparison isolates the added detector rather
than differing event samples. Uncertainties are block bootstrap over events (the
three classes at one trigger share a noise segment, so events, not rows, are the
resampling unit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update(
    {
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
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)

# Physical classes. "real_glitch" is a catalogued blip; "noise" is a real
# signal-free segment; "inj_ccsn" is a held-out CCSN injected coherently.
CLASSES = ("inj_ccsn", "real_glitch", "noise")
CLASS_LABEL = {
    "inj_ccsn": "CCSN",
    "real_glitch": "Blip glitch",
    "noise": "Noise",
}
GROUPS = ("One detector", "Two detectors")
COLOR = {"log_odds": "#1b6ca8", "new_snr": "#d1495b"}


def load_paired(bundle: Path) -> pd.DataFrame:
    """Return rows for events present in BOTH detector configurations."""
    df = pd.read_csv(bundle / "campaign_event_metrics.csv")
    df["group"] = df.detector_count.map({1: GROUPS[0], 2: GROUPS[1]})
    if df.group.isna().any():
        raise ValueError("Unexpected detector_count in the bundle CSV.")
    # an event is (glitch_host, index); pair on it having all classes in both
    counts = df.groupby(["glitch_host", "index"]).group.nunique()
    paired = set(counts[counts == 2].index)
    df = df[
        [(h, i) in paired for h, i in zip(df.glitch_host, df["index"])]
    ].copy()
    df["event"] = df.glitch_host + "_" + df["index"].astype(str)
    return df


def predicted_class(df: pd.DataFrame) -> pd.Series:
    """Three-way model selection: argmax over the hypothesis evidences.

    ``ln Z_noise = 0`` is the exact convention of the noise-relative likelihood,
    so the comparison is between ln Z_signal, ln Z_glitch (best single detector)
    and zero. Equal prior odds -- the confusion matrix reports the evidence
    comparison itself, without the alpha/beta prior weighting of the BCR.
    """
    return pd.Series(
        np.select(
            [
                (df.logZ_signal >= df.logZ_glitch) & (df.logZ_signal >= 0.0),
                (df.logZ_glitch >= df.logZ_signal) & (df.logZ_glitch >= 0.0),
            ],
            ["inj_ccsn", "real_glitch"],
            default="noise",
        ),
        index=df.index,
    )


def _roc_curve(pos: np.ndarray, neg: np.ndarray):
    """Exact ROC: (false-alarm probability, detection efficiency)."""
    thresh = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = np.array([(pos >= t).mean() for t in thresh])
    fpr = np.array([(neg >= t).mean() for t in thresh])
    return np.concatenate([[0], fpr, [1]]), np.concatenate([[0], tpr, [1]])


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (equivalently the Mann-Whitney statistic), ties at 0.5."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return float(
        (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
        / (pos.size * neg.size)
    )


def _boot_events(events: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    uniq = np.unique(events)
    for _ in range(n_boot):
        yield rng.choice(uniq, uniq.size, replace=True)


def auc_with_err(
    df: pd.DataFrame, score: str, bg_classes, n_boot: int = 400, seed: int = 0
):
    """AUC of signal vs the given background classes, with bootstrap SE."""
    sig = df[df["class"] == "inj_ccsn"]
    bkg = df[df["class"].isin(bg_classes)]
    point = _auc(sig[score].to_numpy(), bkg[score].to_numpy())
    sig_by = {e: g[score].to_numpy() for e, g in sig.groupby("event")}
    bkg_by = {e: g[score].to_numpy() for e, g in bkg.groupby("event")}
    empty = np.empty(0)
    vals = []
    for pick in _boot_events(df.event.to_numpy(), n_boot, seed):
        p = np.concatenate([sig_by.get(e, empty) for e in pick])
        n = np.concatenate([bkg_by.get(e, empty) for e in pick])
        if p.size and n.size:
            vals.append(_auc(p, n))
    return point, float(np.std(vals))


def _roc_band(
    pos: np.ndarray,
    neg: np.ndarray,
    pos_events: np.ndarray,
    neg_events: np.ndarray,
    fap_grid: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
):
    """Block-bootstrap ROC as (median, lo, hi) detection efficiency on fap_grid.

    Resamples whole events (the three classes at a trigger share a noise
    segment), draws the exact ROC each time, and evaluates it on a common
    false-alarm grid. A near-perfect classifier on a finite background gives a
    staircase with big discrete jumps that reads as an artefact of the sample
    size rather than the statistic, so we report the bootstrap median and
    central 68% interval instead of the raw curve.
    """
    m = np.isfinite(pos)
    pos, pos_events = pos[m], pos_events[m]
    m = np.isfinite(neg)
    neg, neg_events = neg[m], neg_events[m]
    pos_by = {e: pos[pos_events == e] for e in np.unique(pos_events)}
    neg_by = {e: neg[neg_events == e] for e in np.unique(neg_events)}
    events = np.unique(np.concatenate([pos_events, neg_events]))
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


def fig_confusion(df: pd.DataFrame, out: Path) -> dict:
    """Row-normalised three-class confusion matrices, one panel per network."""
    cmap = LinearSegmentedColormap.from_list("bl", ["#ffffff", "#1b6ca8"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    summary = {}
    for ax, grp in zip(axes, GROUPS):
        sub = df[df.group == grp]
        cm = pd.crosstab(sub["class"], sub["pred"]).reindex(
            index=CLASSES, columns=CLASSES, fill_value=0
        )
        frac = cm.div(cm.sum(axis=1), axis=0)
        summary[grp] = {
            "counts": cm.to_dict(),
            "row_fractions": frac.round(5).to_dict(),
            "accuracy": float(np.diag(cm).sum() / cm.values.sum()),
            "n_events_per_class": int(cm.sum(axis=1).iloc[0]),
        }
        ax.imshow(frac.to_numpy(), cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                v = frac.to_numpy()[i, j]
                ax.text(
                    j,
                    i,
                    f"{v*100:.1f}%\n({cm.to_numpy()[i, j]:d})",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if v > 0.5 else "#222222",
                )
        ax.set_xticks(range(3), [CLASS_LABEL[c] for c in CLASSES])
        ax.set_yticks(range(3), [CLASS_LABEL[c] for c in CLASSES])
        ax.set_xlabel("Selected model")
        ax.set_title(f"{grp} (acc. {summary[grp]['accuracy']*100:.1f}%)")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(out / "fig_confusion.pdf")
    plt.close(fig)
    return summary


def fig_roc(df: pd.DataFrame, out: Path) -> dict:
    """Signal vs combined background, ranking by ln O and reweighted SNR."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    fap = np.logspace(-4, 0, 200)
    summary = {}
    for ax, grp in zip(axes, GROUPS):
        sub = df[df.group == grp]
        summary[grp] = {}
        for score, label in (
            ("log_odds", r"$\ln\mathcal{O}$"),
            ("new_snr", r"$\mathrm{SNR}^{*}$"),
        ):
            sig_rows = sub[sub["class"] == "inj_ccsn"]
            bkg_rows = sub[sub["class"] != "inj_ccsn"]
            a, err = auc_with_err(sub, score, ["noise", "real_glitch"])
            summary[grp][score] = {"auc": a, "auc_err": err}
            med, lo, hi = _roc_band(
                sig_rows[score].to_numpy(),
                bkg_rows[score].to_numpy(),
                sig_rows["event"].to_numpy(),
                bkg_rows["event"].to_numpy(),
                fap,
            )
            ax.fill_between(fap, lo, hi, color=COLOR[score], alpha=0.2, lw=0)
            ax.plot(
                fap,
                med,
                color=COLOR[score],
                lw=1.4,
                label=f"{label}  AUC $={a:.3f}\\pm{err:.3f}$",
            )
        ax.set_xscale("log")
        ax.set_xlim(fap[0], 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False-alarm probability")
        ax.set_title(grp)
        # the reweighted-SNR curve climbs through the lower right, so the
        # legend needs an opaque backing to stay readable
        ax.legend(
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
        )
    axes[0].set_ylabel("Detection efficiency")
    fig.tight_layout()
    fig.savefig(out / "fig_roc.pdf")
    plt.close(fig)
    return summary


def _wilson(k: int, n: int, z: float = 1.0):
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def fig_efficiency(df: pd.DataFrame, out: Path, far: float = 0.05) -> dict:
    """Efficiency vs injected SNR at a fixed false-alarm probability."""
    edges = np.array([10, 15, 20, 25, 30, 40])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    summary = {}
    for grp, marker, color in zip(GROUPS, ("o", "s"), ("#1b6ca8", "#e08214")):
        sub = df[df.group == grp]
        bkg = sub[sub["class"] != "inj_ccsn"].log_odds.to_numpy()
        thr = float(np.quantile(bkg, 1.0 - far))
        sig = sub[sub["class"] == "inj_ccsn"]
        snr = sig.injected_or_catalog_snr.to_numpy()
        det = sig.log_odds.to_numpy() >= thr
        xs, ys, los, his, ns = [], [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (snr >= lo) & (snr < hi)
            if m.sum() < 5:
                continue
            k, n = int(det[m].sum()), int(m.sum())
            l, h = _wilson(k, n)
            xs.append(0.5 * (lo + hi))
            ys.append(k / n)
            los.append(k / n - l)
            his.append(h - k / n)
            ns.append(n)
        ax.errorbar(
            xs,
            ys,
            yerr=[los, his],
            marker=marker,
            ms=3.5,
            lw=1.2,
            capsize=2,
            color=color,
            label=grp,
        )
        summary[grp] = {
            "threshold": thr,
            "snr_centers": xs,
            "efficiency": ys,
            "n": ns,
        }
    ax.set_xlabel("Injected network SNR")
    ax.set_ylabel(f"Efficiency at {far:.0%} FAP")
    ax.set_ylim(0.5, 1.02)  # efficiency never drops below ~0.85 in this range
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_efficiency.pdf")
    plt.close(fig)
    return summary


def write_table(df: pd.DataFrame, out: Path, table_path: Path) -> dict:
    """AUC table: ln O vs reweighted SNR, per background class."""
    rows = {}
    backgrounds = [
        ("Combined background", ["noise", "real_glitch"]),
        ("Real noise", ["noise"]),
        ("Real blip glitches", ["real_glitch"]),
    ]
    for grp in GROUPS:
        sub = df[df.group == grp]
        rows[grp] = {}
        for name, bg in backgrounds:
            for score in ("log_odds", "new_snr"):
                a, e = auc_with_err(sub, score, bg)
                rows[grp].setdefault(name, {})[score] = {
                    "auc": a,
                    "err": e,
                }
    lines = [
        r"\begin{tabular}{llcc}",
        r"\hline\hline",
        r"Network & Background & AUC($\ln\mathcal{O}$) & AUC($\mathrm{SNR}^{*}$) \\",
        r"\hline",
    ]
    for grp in GROUPS:
        for k, (name, _) in enumerate(backgrounds):
            cell = rows[grp][name]
            lines.append(
                f"{grp if k == 0 else ''} & {name} & "
                f"${cell['log_odds']['auc']:.3f}\\pm{cell['log_odds']['err']:.3f}$ & "
                f"${cell['new_snr']['auc']:.3f}\\pm{cell['new_snr']['err']:.3f}$ \\\\"
            )
        lines.append(r"\hline")
    lines += [r"\end{tabular}"]
    table_path.write_text("\n".join(lines) + "\n")
    return rows


def paired_glitch_shift(df: pd.DataFrame) -> dict:
    """McNemar test on blips misclassified as CCSN, one vs two detectors."""
    g = df[df["class"] == "real_glitch"]
    wide = g.pivot_table(
        index="event", columns="group", values="pred", aggfunc="first"
    ).dropna()
    one = wide[GROUPS[0]] == "inj_ccsn"
    two = wide[GROUPS[1]] == "inj_ccsn"
    b = int((one & ~two).sum())  # fixed by adding a detector
    c = int((~one & two).sum())  # broken by adding a detector
    from scipy.stats import binomtest

    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else float("nan")
    return {
        "n_paired": int(len(wide)),
        "misclassified_one": int(one.sum()),
        "misclassified_two": int(two.sum()),
        "fixed_by_second_detector": b,
        "broken_by_second_detector": c,
        "mcnemar_p": float(p),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument(
        "--table",
        type=Path,
        help="Where to write summary_table.tex (default: inside --outdir). "
        "The manuscript keeps it outside the gitignored figures/ directory.",
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    table_path = args.table or (args.outdir / "summary_table.tex")

    df = load_paired(args.bundle)
    df["pred"] = predicted_class(df)
    n_events = df.event.nunique()
    print(f"paired events: {n_events}  rows: {len(df)}")

    metrics = {
        "campaign_id": "nuts_morphlnz_v041",
        "n_paired_events": int(n_events),
        "n_rows": int(len(df)),
        "evidence_failures": int(df.evidence_failures.sum()),
        "evidence_fallbacks": int(df.evidence_fallbacks.sum()),
        "confusion": fig_confusion(df, args.outdir),
        "roc": fig_roc(df, args.outdir),
        "efficiency": fig_efficiency(df, args.outdir),
        "auc_table": write_table(df, args.outdir, table_path),
        "paired_glitch_shift": paired_glitch_shift(df),
    }
    (args.outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(json.dumps(metrics["paired_glitch_shift"], indent=2))
    for grp in GROUPS:
        c = metrics["confusion"][grp]
        print(f"{grp}: accuracy {c['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
