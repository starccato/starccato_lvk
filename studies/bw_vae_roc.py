"""Fair VAE-versus-BayesWave detection comparison: ROC, AUC, matched-FPF efficiency.

This replaces the "N of M correct" framing, which was unfair to BayesWave in two
independent ways.

**Different hypotheses.**  The VAE statistic is a Bayesian coherence ratio,
signal against a (glitch OR noise) mixture.  BayesWave's native lnB_S/G is signal
against glitch alone.  On a quiet event both transient models are rejected and
their difference is not a detection statistic at all.  So both are reported: the
native factor and the aligned lnB_S/(GvN) built from the same three evidences
with the same mixture weights the VAE uses.

**Different zero points.**  "Correct sign at zero" hard-codes a threshold that is
only meaningful if both statistics share a normalisation, and these do not.  The
VAE's denominator is a product over detectors of per-detector (glitch, noise)
mixtures, so with beta=0.5 it carries a +2 ln 2 offset for an H1-L1 event, while
BayesWave's glitch and noise models are already network-level and its aligned
statistic carries +ln 2.  The two zeros are therefore ln 2 apart before any
physics enters.  A constant offset moves every "correct sign" count while
changing no ranking at all.  Ranking comparisons (AUC) and efficiency at a
*matched empirical* false-positive fraction are invariant to it, which is why
they are the primary numbers here.

Positive class: injected CCSN events.  Negative class (background): real blips.
The threshold at a given FPF is read off the observed blip distribution, so both
pipelines are always compared at the same measured false-alarm rate.

Events that produced no usable score are an explicit outcome category.  They are
reported, and the effect of the worst case -- every unscorable event ranked at
the bottom for the pipeline that failed on it -- is quantified, rather than being
dropped quietly.

Usage:
    python studies/bw_vae_roc.py \
        --paired /fred/oz303/.../paired_lno_vs_bayeswave.csv \
        --outdir /fred/oz303/.../comparison_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from starccato_lvk.bayeswave import (  # noqa: E402
    aligned_signal_vs_glitch_or_noise,
)

POSITIVE_CLASS = "inj_ccsn"
NEGATIVE_CLASS = "real_glitch"

# Statistics compared, in the order they are reported.
STATISTICS = (
    ("vae_lno", "VAE lnO_S/(G+N)"),
    ("bw_native", "BayesWave lnB_S/G (native)"),
    ("bw_aligned", "BayesWave lnB_S/(GvN) (aligned)"),
)

# False-positive fractions at which detection efficiency is quoted. The smallest
# is bounded by the size of the blip background: with ~640 blips, an empirical
# FPF of 0.01 rests on about six events, so nothing tighter is quoted.
FPF_GRID = (0.10, 0.05, 0.01)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {
                "cls": raw["cls"],
                "evidence_source": raw["bw_evidence_source"],
            }
            for key, value in raw.items():
                if key in ("cls", "bw_evidence_source"):
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = math.nan
            rows.append(row)
    return rows


def _score(row: dict[str, Any]) -> dict[str, Any]:
    row["vae_lno"] = row["lno_log_odds"]
    row["bw_native"] = row["bw_lnbf_signal_glitch"]
    evidences = (row["bw_logZ_signal"], row["bw_logZ_glitch"], row["bw_logZ_noise"])
    if all(math.isfinite(v) for v in evidences):
        row["bw_aligned"] = aligned_signal_vs_glitch_or_noise(*evidences)[0]
    else:
        row["bw_aligned"] = math.nan
    return row


def _outcome(row: dict[str, Any]) -> str:
    """Explicit outcome category; nothing is silently dropped."""
    if not math.isfinite(row["vae_lno"]):
        return "vae_no_score"
    if not all(math.isfinite(row[name]) for name, _ in STATISTICS):
        return "bayeswave_no_score"
    if row["bw_recovered"] or row["bw_degenerate_uncertainty"]:
        # A salvaged evidence is still an evidence: it is kept in the primary
        # cohort and flagged, so the reader can see whether the conclusion rests
        # on it. Excluding it would be exactly the post-hoc selection this
        # analysis exists to avoid.
        return "salvaged_evidence"
    return "complete"


def _roc(positive: np.ndarray, negative: np.ndarray) -> dict[str, np.ndarray]:
    """Empirical ROC by sweeping the threshold over every observed score."""
    thresholds = np.unique(np.concatenate([positive, negative]))
    # Descending, with +inf first so the curve starts at (0, 0).
    thresholds = np.concatenate([[np.inf], thresholds[::-1]])
    tpf = np.array([np.mean(positive >= t) for t in thresholds])
    fpf = np.array([np.mean(negative >= t) for t in thresholds])
    return {"threshold": thresholds, "tpf": tpf, "fpf": fpf}


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_signal > score_blip), ties counted as a half."""
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    combined = np.concatenate([positive, negative])[order]
    ranks = np.empty(len(combined), dtype=float)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1] == combined[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    unranked = np.empty(len(combined), dtype=float)
    unranked[order] = ranks
    n_pos, n_neg = len(positive), len(negative)
    rank_sum = float(np.sum(unranked[:n_pos]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auc_ci(positive: np.ndarray, negative: np.ndarray, draws: int,
            rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap, resampling each class independently."""
    values = np.empty(draws)
    for k in range(draws):
        p = rng.choice(positive, size=len(positive), replace=True)
        n = rng.choice(negative, size=len(negative), replace=True)
        values[k] = _auc(p, n)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _efficiency_at_fpf(positive: np.ndarray, negative: np.ndarray,
                       fpf: float) -> dict[str, float]:
    """Detection efficiency at a threshold set by the observed blip background.

    The threshold is the (1-fpf) quantile of the background, so the quoted false
    alarm rate is the one actually measured on these blips -- not a nominal one
    from an assumed distribution.
    """
    threshold = float(np.quantile(negative, 1.0 - fpf, method="higher"))
    achieved = float(np.mean(negative >= threshold))
    efficiency = float(np.mean(positive >= threshold))
    # Wilson interval: the normal approximation is wrong near efficiency 0 or 1,
    # which is exactly where the low-FPF numbers live.
    n = len(positive)
    z = 1.959963985
    denom = 1.0 + z * z / n
    centre = (efficiency + z * z / (2 * n)) / denom
    half = z * math.sqrt(efficiency * (1 - efficiency) / n + z * z / (4 * n * n)) / denom
    return {
        "requested_fpf": fpf,
        "achieved_fpf": achieved,
        "threshold": threshold,
        "efficiency": efficiency,
        "efficiency_ci95": [max(0.0, centre - half), min(1.0, centre + half)],
        "n_detected": int(np.sum(positive >= threshold)),
    }


def analyse(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    for row in rows:
        _score(row)
        row["outcome"] = _outcome(row)

    scorable = [r for r in rows if r["outcome"] in ("complete", "salvaged_evidence")]
    positive_rows = [r for r in scorable if r["cls"] == POSITIVE_CLASS]
    negative_rows = [r for r in scorable if r["cls"] == NEGATIVE_CLASS]

    accounting = {
        "n_paired_rows": len(rows),
        "by_class_and_outcome": {},
        "n_scorable": len(scorable),
        "n_positive": len(positive_rows),
        "n_negative": len(negative_rows),
    }
    for cls in sorted({r["cls"] for r in rows}):
        accounting["by_class_and_outcome"][cls] = {
            outcome: sum(1 for r in rows if r["cls"] == cls and r["outcome"] == outcome)
            for outcome in ("complete", "salvaged_evidence",
                            "bayeswave_no_score", "vae_no_score")
        }

    results: dict[str, Any] = {}
    curves: dict[str, dict[str, list[float]]] = {}
    for name, label in STATISTICS:
        positive = np.array([r[name] for r in positive_rows], dtype=float)
        negative = np.array([r[name] for r in negative_rows], dtype=float)
        auc = _auc(positive, negative)
        lo, hi = _auc_ci(positive, negative, draws, rng)
        roc = _roc(positive, negative)
        curves[name] = {k: [float(x) for x in v] for k, v in roc.items()}
        results[name] = {
            "label": label,
            "auc": auc,
            "auc_ci95": [lo, hi],
            "efficiency_at_fpf": [
                _efficiency_at_fpf(positive, negative, f) for f in FPF_GRID
            ],
            # Reported for continuity with the old framing, and explicitly NOT
            # the comparison statistic: it is the efficiency at whatever
            # false-alarm rate the fixed zero threshold happens to give, and that
            # rate differs between the two pipelines.
            "sign_convention_at_zero": {
                "n_positive_above_zero": int(np.sum(positive > 0)),
                "n_negative_below_zero": int(np.sum(negative < 0)),
                "fraction_correct": float(
                    (np.sum(positive > 0) + np.sum(negative < 0))
                    / (len(positive) + len(negative))
                ),
                "implied_fpf_at_zero": float(np.mean(negative > 0)),
            },
        }

    # Worst case for the unscorable events: rank each below every scored event
    # for the pipeline that failed on it. This bounds how much the missing runs
    # could possibly have helped whichever pipeline lost.
    worst: dict[str, Any] = {}
    for name, _ in STATISTICS:
        failed_key = "vae_no_score" if name == "vae_lno" else "bayeswave_no_score"
        positive = [r[name] for r in positive_rows]
        negative = [r[name] for r in negative_rows]
        floor = min(positive + negative) - 1.0
        n_pos_fail = sum(
            1 for r in rows if r["cls"] == POSITIVE_CLASS and r["outcome"] == failed_key
        )
        n_neg_fail = sum(
            1 for r in rows if r["cls"] == NEGATIVE_CLASS and r["outcome"] == failed_key
        )
        padded_pos = np.array(positive + [floor] * n_pos_fail)
        # A failed background event ranked at the floor cannot raise a false
        # alarm, so only the missed injections cost this pipeline anything.
        worst[name] = {
            "n_positive_unscored": n_pos_fail,
            "n_negative_unscored": n_neg_fail,
            "auc_with_failures_at_floor": _auc(padded_pos, np.array(negative)),
        }

    return {
        "accounting": accounting,
        "statistics": results,
        "worst_case_with_unscored": worst,
        "roc_curves": curves,
        "notes": {
            "positive_class": POSITIVE_CLASS,
            "negative_class": NEGATIVE_CLASS,
            "selection": "no uncertainty threshold is applied; salvaged evidences "
                         "are kept in the primary cohort and flagged",
            "interpretation": "this compares a CCSN-targeted VAE against a "
                              "morphology-agnostic BayesWave baseline; BayesWave "
                              "is not tuned to the CCSN waveform family and an "
                              "AUC gap is expected on that basis alone",
        },
    }


def plot(report: dict[str, Any], rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"vae_lno": "#0072B2", "bw_native": "#D55E00", "bw_aligned": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    for name, _ in STATISTICS:
        curve = report["roc_curves"][name]
        block = report["statistics"][name]
        lo, hi = block["auc_ci95"]
        axes[0].plot(curve["fpf"], curve["tpf"], color=colours[name], lw=1.6,
                     label=f"{block['label']}\nAUC={block['auc']:.3f} "
                           f"[{lo:.3f}, {hi:.3f}]")
    axes[0].plot([0, 1], [0, 1], color="0.6", lw=.8, ls="--")
    axes[0].set(xlabel="false-positive fraction (real blips)",
                ylabel="detection efficiency (injected CCSN)", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=7.5, loc="lower right", frameon=False)

    axes[1].set_xscale("log")
    for name, _ in STATISTICS:
        curve = report["roc_curves"][name]
        fpf = np.array(curve["fpf"])
        axes[1].plot(np.clip(fpf, 1e-4, 1), curve["tpf"], color=colours[name], lw=1.6)
    for f in FPF_GRID:
        axes[1].axvline(f, color="0.85", lw=.8, zorder=0)
    axes[1].set(xlabel="false-positive fraction (log)",
                ylabel="detection efficiency", xlim=(1e-3, 1), ylim=(0, 1))
    axes[1].set_title("low-false-alarm regime", fontsize=10)

    width = 0.26
    positions = np.arange(len(FPF_GRID))
    for k, (name, _) in enumerate(STATISTICS):
        block = report["statistics"][name]
        eff = [e["efficiency"] for e in block["efficiency_at_fpf"]]
        err = np.array([[e["efficiency"] - e["efficiency_ci95"][0] for e in
                         block["efficiency_at_fpf"]],
                        [e["efficiency_ci95"][1] - e["efficiency"] for e in
                         block["efficiency_at_fpf"]]])
        axes[2].bar(positions + (k - 1) * width, eff, width, yerr=err,
                    color=colours[name], label=block["label"], capsize=2.5)
    axes[2].set_xticks(positions, [f"FPF={f:g}" for f in FPF_GRID])
    axes[2].set(ylabel="detection efficiency", ylim=(0, 1))
    axes[2].legend(fontsize=7.5, frameon=False)
    axes[2].set_title("efficiency at matched empirical FPF", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.93", lw=.5, zorder=0)

    acc = report["accounting"]
    fig.suptitle("CCSN-targeted VAE vs morphology-agnostic BayesWave", fontsize=13)
    fig.text(.5, .93,
             f"{acc['n_positive']} injected CCSN vs {acc['n_negative']} real blips; "
             "thresholds set on the observed blip background; no uncertainty cut.",
             ha="center", va="top", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rows = _read_rows(args.paired)
    report = analyse(rows, args.bootstrap, args.seed)
    report["source_paired_table"] = str(args.paired.resolve())

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "roc_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if not args.no_plot:
        plot(report, rows, args.outdir / "roc_comparison")

    acc = report["accounting"]
    print("cohort accounting")
    for cls, block in acc["by_class_and_outcome"].items():
        print(f"  {cls:12s} " + ", ".join(f"{k}={v}" for k, v in block.items() if v))
    print(f"  scorable: {acc['n_positive']} signal / {acc['n_negative']} background\n")
    for name, _ in STATISTICS:
        block = report["statistics"][name]
        lo, hi = block["auc_ci95"]
        print(f"{block['label']}")
        print(f"  AUC = {block['auc']:.4f}  [{lo:.4f}, {hi:.4f}]")
        for eff in block["efficiency_at_fpf"]:
            print(f"  eff @ FPF={eff['requested_fpf']:.2f} "
                  f"(achieved {eff['achieved_fpf']:.4f}): {eff['efficiency']:.3f} "
                  f"({eff['n_detected']}/{acc['n_positive']})")
        sign = block["sign_convention_at_zero"]
        print(f"  [legacy] sign-at-zero correct = {sign['fraction_correct']:.3f}; "
              f"that threshold's own FPF = {sign['implied_fpf_at_zero']:.3f}")
        print()
    print(f"wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
