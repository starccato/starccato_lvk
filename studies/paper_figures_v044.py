"""Manuscript figures and tables from a downloaded campaign bundle.

Reads ``campaign_event_metrics.csv`` (one row per class/event/configuration) and
writes the publication figures plus ``summary_table.tex`` and ``metrics.json``:

    fig_confusion.pdf     three-class confusion matrices (CCSN / blip / noise)
    fig_roc.pdf           empirical ROC step curve + 68% bootstrap band, ln O
                          versus the reweighted matched-filter statistic SNR*
                          (the CSV/JSON field remains "new_snr")
    fig_efficiency.pdf    detection efficiency versus injected network SNR
    summary_table.tex     AUC + misclassification table for \\input{}
    metrics.json          every number quoted in the text

Usage:
    uv run python studies/paper_figures_v044.py \\
        --bundle ../nuts_morphlnz_v044_analysis_bundle \\
        --outdir ../manuscript/figures

All statistics use the *paired* population: events analysed in both the one- and
two-detector configuration, so the comparison isolates the added detector rather
than differing event samples. Injected CCSNe with an achieved H1--L1 network SNR
above the configured ceiling are removed from both detector configurations; no
SNR cut is applied to real blips or noise. Uncertainties are block bootstrap over
events (the classes at one trigger share a noise segment, so events, not rows,
are the resampling unit).
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
        # Boxed axes with mirrored, inward ticks and minor ticks -- standard
        # journal-figure convention, rather than the open/minimalist style.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
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
DEFAULT_MAX_CCSN_NETWORK_SNR = 500.0


def load_paired(bundle: Path) -> pd.DataFrame:
    """Return rows for events present in BOTH detector configurations."""
    df = pd.read_csv(bundle / "campaign_event_metrics.csv")
    df["group"] = df.detector_count.map({1: GROUPS[0], 2: GROUPS[1]})
    if df.group.isna().any():
        raise ValueError("Unexpected detector_count in the bundle CSV.")
    # an event is (glitch_host, index); pair on it having all classes in both,
    # so every confusion row has the same denominator in both configurations
    counts = df.groupby(["glitch_host", "index"]).apply(
        lambda g: g.groupby("group")["class"].nunique().eq(len(CLASSES)).sum(),
        include_groups=False,
    )
    paired = set(counts[counts == 2].index)
    df = df[
        [(h, i) in paired for h, i in zip(df.glitch_host, df["index"])]
    ].copy()
    df["event"] = df.glitch_host + "_" + df["index"].astype(str)
    return df


def apply_ccsn_network_snr_ceiling(
    df: pd.DataFrame, max_network_snr: float
) -> tuple[pd.DataFrame, dict]:
    """Remove only high-network-SNR CCSN injections from both configurations.

    The ceiling is defined by the achieved two-detector network SNR. Applying
    the event-level decision to both configurations preserves a matched
    one-versus-two-detector CCSN comparison. Background rows are untouched.
    """
    if not np.isfinite(max_network_snr) or max_network_snr <= 0:
        raise ValueError("The CCSN network-SNR ceiling must be finite and positive.")

    network = df[
        (df["group"] == GROUPS[1]) & (df["class"] == "inj_ccsn")
    ][["event", "injected_or_catalog_snr"]]
    if network["event"].duplicated().any() or network["event"].nunique() != df["event"].nunique():
        raise ValueError(
            "Expected exactly one two-detector CCSN row for every paired event."
        )

    remove_events = set(
        network.loc[
            network["injected_or_catalog_snr"] > max_network_snr, "event"
        ]
    )
    selected = df[
        ~(
            (df["class"] == "inj_ccsn")
            & df["event"].isin(remove_events)
        )
    ].copy()
    n_ccsn_before = int((df["class"] == "inj_ccsn").sum() // len(GROUPS))
    n_ccsn_after = int((selected["class"] == "inj_ccsn").sum() // len(GROUPS))
    return selected, {
        "max_ccsn_network_snr": float(max_network_snr),
        "n_ccsn_before_per_configuration": n_ccsn_before,
        "n_ccsn_removed_per_configuration": n_ccsn_before - n_ccsn_after,
        "n_ccsn_retained_per_configuration": n_ccsn_after,
        "n_blips_per_configuration": int(
            (selected["class"] == "real_glitch").sum() // len(GROUPS)
        ),
        "n_noise_per_configuration": int(
            (selected["class"] == "noise").sum() // len(GROUPS)
        ),
    }


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
    """Exact ROC: (empirical false-positive fraction, detection efficiency)."""
    thresh = np.unique(np.concatenate([pos, neg]))[::-1]
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    tpr = (pos.size - np.searchsorted(pos_sorted, thresh, side="left")) / pos.size
    fpr = (neg.size - np.searchsorted(neg_sorted, thresh, side="left")) / neg.size
    return np.concatenate([[0], fpr, [1]]), np.concatenate([[0], tpr, [1]])


def _roc_at_fraction(
    false_positive_fraction: np.ndarray,
    efficiency: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Evaluate the attainable empirical ROC envelope as a step function."""
    order = np.argsort(false_positive_fraction, kind="stable")
    fpf = false_positive_fraction[order]
    eff = np.maximum.accumulate(efficiency[order])
    starts = np.flatnonzero(np.r_[True, np.diff(fpf) != 0])
    ends = np.r_[starts[1:] - 1, len(fpf) - 1]
    unique_fpf = fpf[starts]
    envelope = eff[ends]
    idx = np.searchsorted(unique_fpf, grid, side="right") - 1
    return envelope[np.clip(idx, 0, len(envelope) - 1)]


def _threshold_at_fraction(
    background: np.ndarray, max_fraction: float
) -> tuple[float, float, int]:
    """Strict-exceedance threshold with observed FPF no larger than target."""
    background = background[np.isfinite(background)]
    max_false_positives = int(np.floor(max_fraction * background.size))
    if max_false_positives >= background.size:
        threshold = -np.inf
    else:
        threshold = float(np.sort(background)[::-1][max_false_positives])
    n_false = int(np.sum(background > threshold))
    return threshold, n_false / background.size, n_false


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
    fraction_grid: np.ndarray,
    n_boot: int = 400,
    seed: int = 0,
):
    """Block-bootstrap central 68% interval on a false-positive-fraction grid.

    Resamples whole events (the three classes at a trigger share a noise
    segment), draws the exact ROC each time, and evaluates its attainable step
    envelope on a common grid. Linear interpolation is deliberately avoided:
    repeated empirical fractions are finite-background steps, not support for
    intermediate operating points.
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
            tprs.append(_roc_at_fraction(fpr, tpr, fraction_grid))
    return np.percentile(np.asarray(tprs), [16, 84], axis=0)


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
            "counts": {
                true: {pred: int(cm.loc[true, pred]) for pred in CLASSES}
                for true in CLASSES
            },
            "row_fractions": {
                true: {
                    pred: float(frac.loc[true, pred])
                    for pred in CLASSES
                }
                for true in CLASSES
            },
            "accuracy": float(np.diag(cm).sum() / cm.values.sum()),
            "n_by_true_class": {
                true: int(cm.loc[true].sum()) for true in CLASSES
            },
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
        ax.set_xlabel(f"Selected model\n{grp}")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(out / "fig_confusion.pdf")
    plt.close(fig)
    return summary


def fig_roc(df: pd.DataFrame, out: Path) -> dict:
    """Signal vs combined background, ranking by ln O and reweighted SNR.

    The full linear ROC makes the overall ranking comparison familiar, while a
    second linear panel enlarges the stringent region where the one- versus
    two-detector difference occurs. Both panels show the exact empirical steps;
    neither interpolates between attainable false-positive fractions.
    """
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 6.5))
    ax_full, ax_tail = axes
    n_bkg = int((df[df.group == GROUPS[0]]["class"] != "inj_ccsn").sum())
    fraction_floor = 1.0 / n_bkg
    band_grids = {
        ax_full: np.linspace(0.0, 1.0, 600),
        ax_tail: np.linspace(0.0, 0.02, 400),
    }
    style = {GROUPS[0]: "#1b6ca8", GROUPS[1]: "#e08214"}
    summary = {}
    for grp in GROUPS:
        sub = df[df.group == grp]
        summary[grp] = {}
        for score, label, ls in (
            ("log_odds", r"$\ln\mathcal{O}$", "-"),
            ("new_snr", r"$\mathrm{SNR}^{*}$", "--"),
        ):
            sig_rows = sub[sub["class"] == "inj_ccsn"]
            bkg_rows = sub[sub["class"] != "inj_ccsn"]
            sig = sig_rows[score].to_numpy()
            bkg = bkg_rows[score].to_numpy()
            a, err = auc_with_err(sub, score, ["noise", "real_glitch"])
            summary[grp][score] = {"auc": a, "auc_err": err}
            fpr, tpr = _roc_curve(sig, bkg)
            if score == "log_odds":
                for ax, grid in band_grids.items():
                    lo, hi = _roc_band(
                        sig, bkg,
                        sig_rows["event"].to_numpy(),
                        bkg_rows["event"].to_numpy(),
                        grid,
                    )
                    ax.fill_between(grid, lo, hi,
                                    color=style[grp], alpha=0.18, lw=0)
            for ax in axes:
                ax.step(
                    fpr,
                    tpr,
                    where="post",
                    color=style[grp],
                    ls=ls,
                    lw=1.4,
                    label=f"{label}, {grp.split()[0].lower()} det.",
                )

    ax_full.set_xlim(0.0, 1.0)
    ax_tail.set_xlim(0.0, 0.02)
    ax_tail.set_xticks([0.0, 0.005, 0.010, 0.015, 0.020])
    ax_tail.set_xticklabels(["0", "0.005", "0.010", "0.015", "0.020"])
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_box_aspect(1)
        ax.set_xlabel("Empirical false-positive fraction")
    ax_full.set_ylabel("Detection efficiency")
    ax_tail.set_ylabel("Detection efficiency")
    ax_full.plot([0.0, 1.0], [0.0, 1.0], color="0.55", lw=0.9, ls=":", zorder=0)
    ax_full.text(
        0.04,
        0.92,
        "(a)",
        transform=ax_full.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    ax_tail.text(
        0.04,
        0.92,
        "(b)",
        transform=ax_tail.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    # Move the y-axis spine outward so the exact vertical ROC segment at FPF=0
    # remains visible without extending either linear axis into negative FPF.
    ax_tail.spines["left"].set_position(("outward", 4))
    ax_tail.axvline(fraction_floor, color="0.6", lw=0.8, ls=":", zorder=0)
    ax_full.legend(
        loc="lower right",
        frameon=False,
        fontsize=7.0,
        ncol=2,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.5,
    )
    fig.tight_layout(h_pad=1.8)
    fig.savefig(out / "fig_roc.pdf")
    plt.close(fig)
    return summary


def _wilson(k: int, n: int, z: float = 1.0):
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def zero_fpf_summary(df: pd.DataFrame) -> dict:
    """Efficiency at the loudest background event (zero observed false positives).

    Not plotted separately -- this operating point is also the left edge of
    Figure~\\ref{fig:rn_roc} -- but kept as an explicit metric since the
    manuscript quotes it directly (5.1% / 60.9%).
    """
    summary = {}
    for grp in GROUPS:
        sub = df[df.group == grp]
        bkg = sub[sub["class"] != "inj_ccsn"].log_odds.to_numpy()
        loudest = float(bkg.max())
        sig = sub[sub["class"] == "inj_ccsn"].log_odds.to_numpy()
        summary[grp] = {
            "loudest_background": loudest,
            "efficiency_at_zero_false_positives": float(np.mean(sig > loudest)),
            "n_signal": int(sig.size),
            "n_background": int(bkg.size),
        }
    return summary


def fig_efficiency(df: pd.DataFrame, out: Path, fpf: float = 0.05) -> dict:
    """Efficiency vs injected SNR at a fixed empirical false-positive fraction."""
    edges = np.array([10, 15, 20, 25, 30, 40])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    summary = {}
    for grp, marker, color in zip(GROUPS, ("o", "s"), ("#1b6ca8", "#e08214")):
        sub = df[df.group == grp]
        bkg = sub[sub["class"] != "inj_ccsn"].log_odds.to_numpy()
        thr, observed_fpf, n_false = _threshold_at_fraction(bkg, fpf)
        sig = sub[sub["class"] == "inj_ccsn"]
        snr = sig.injected_or_catalog_snr.to_numpy()
        det = sig.log_odds.to_numpy() > thr
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
            "target_false_positive_fraction": fpf,
            "observed_false_positive_fraction": observed_fpf,
            "n_false_positives": n_false,
            "snr_centers": xs,
            "efficiency": ys,
            "n": ns,
        }
    ax.set_xlabel("Injected network SNR")
    ax.set_ylabel(f"Efficiency at {fpf:.0%} FPF")
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


def host_stratification(df: pd.DataFrame) -> dict:
    """Confusion accuracy by blip host, as a pooling-consistency diagnostic."""
    out: dict = {}
    for host in sorted(df.glitch_host.unique()):
        out[host] = {}
        for grp in GROUPS:
            sub = df[(df.glitch_host == host) & (df.group == grp)]
            cm = pd.crosstab(sub["class"], sub["pred"]).reindex(
                index=CLASSES, columns=CLASSES, fill_value=0
            )
            out[host][grp] = {
                "n_by_true_class": {
                    true: int(cm.loc[true].sum()) for true in CLASSES
                },
                "accuracy": float(np.diag(cm).sum() / cm.values.sum()),
                "blip_to_signal": int(cm.loc["real_glitch", "inj_ccsn"]),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--campaign-id", default="nuts_morphlnz_v044")
    ap.add_argument(
        "--max-ccsn-network-snr",
        type=float,
        default=DEFAULT_MAX_CCSN_NETWORK_SNR,
        help="Remove CCSN injections above this achieved H1--L1 network SNR "
        "from both configurations; backgrounds are not filtered.",
    )
    ap.add_argument(
        "--table",
        type=Path,
        help="Where to write summary_table.tex (default: inside --outdir). "
        "The manuscript keeps it outside the gitignored figures/ directory.",
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    table_path = args.table or (args.outdir / "summary_table.tex")

    provenance_path = args.bundle / "campaign_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"{provenance_path} is required; regenerate the bundle with "
            "studies/bundle_from_combined.py"
        )
    provenance = json.loads(provenance_path.read_text())
    archived_campaign = provenance.get("source_attrs", {}).get("lno_campaign")
    if archived_campaign != args.campaign_id:
        raise ValueError(
            f"archive campaign {archived_campaign!r} does not match "
            f"--campaign-id {args.campaign_id!r}"
        )

    df = load_paired(args.bundle)
    df, population_selection = apply_ccsn_network_snr_ceiling(
        df, args.max_ccsn_network_snr
    )
    df["pred"] = predicted_class(df)
    n_events = df.event.nunique()
    analyses_with_fallback = int(df.evidence_fallbacks.gt(0).sum())
    print(f"paired events: {n_events}  rows: {len(df)}")

    metrics = {
        "campaign_id": args.campaign_id,
        "source_h5_sha256": provenance["source_h5_sha256"],
        "source_git_commit": provenance["source_attrs"]["git_commit"],
        "source_created_utc": provenance["source_attrs"]["created_utc"],
        "full_campaign_evidence_audit": {
            key: provenance[key]
            for key in (
                "n_analysis_rows",
                "n_evidence_terms",
                "n_evidence_failures",
                "estimator_counts",
                "status_counts",
                "direct_nested_terms",
                "morph_attempted_terms",
                "crosschecked_terms",
            )
        },
        "n_paired_events": int(n_events),
        "n_rows": int(len(df)),
        "population_selection": population_selection,
        "evidence_failures": int(df.evidence_failures.sum()),
        "analyses_with_fallback": analyses_with_fallback,
        "analysis_fallback_fraction": analyses_with_fallback / len(df),
        "fallback_evidence_terms": int(df.evidence_fallbacks.sum()),
        # Backward-compatible alias retained for downstream notebooks.
        "evidence_fallbacks": int(df.evidence_fallbacks.sum()),
        "confusion": fig_confusion(df, args.outdir),
        "roc": fig_roc(df, args.outdir),
        "efficiency": fig_efficiency(df, args.outdir),
        "score_dist": zero_fpf_summary(df),
        "auc_table": write_table(df, args.outdir, table_path),
        "paired_glitch_shift": paired_glitch_shift(df),
        "host_stratification": host_stratification(df),
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
