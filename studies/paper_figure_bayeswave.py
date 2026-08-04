"""Manuscript figure + numbers for the paired lnO-versus-BayesWave comparison.

Reads the slim paired table written by ``export_paired_comparison.py`` (one row
per event carrying both statistics) and writes:

    fig_bayeswave_comparison.pdf   scatter of the two statistics, and the
                                   signal-versus-blip ROC for each
    bayeswave_metrics.json         every number quoted in sec:bayeswave

Usage:
    uv run python studies/paper_figure_bayeswave.py \
        --paired ../paired_lno_vs_bayeswave.csv \
        --outdir ../manuscript/figures

Only rows with ``bw_recovered == 0`` enter the statistics: a recovered row had
its evidence reconstructed from a log file after the post-processing stage died,
and most carry no uncertainty. The ``--max-lnbf-err`` cut defines the
best-converged BayesWave subset used as a robustness check.
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
from scipy import stats

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

COLOR = {"lno": "#1b6ca8", "bw": "#d1495b"}
CLS_COLOR = {"inj_ccsn": "#1b6ca8", "real_glitch": "#d1495b"}
CLS_LABEL = {"inj_ccsn": "Injected CCSN", "real_glitch": "Real blip glitch"}
LNO = "lno_log_odds"
LNBF = "bw_lnbf_signal_glitch"


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties at 0.5."""
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float(
        (ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
        / (pos.size * neg.size)
    )


def _auc_err(pos: np.ndarray, neg: np.ndarray, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    return float(
        np.std(
            [
                _auc(rng.choice(pos, pos.size, True), rng.choice(neg, neg.size, True))
                for _ in range(n_boot)
            ]
        )
    )


def _roc_curve(pos: np.ndarray, neg: np.ndarray):
    """Empirical ROC coordinates, including the zero-exceedance endpoint."""
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
    """Best attainable efficiency at each empirical background fraction.

    Empirical ROC coordinates repeat whenever signal scores occur between two
    adjacent background scores. Linear interpolation across those repeats
    invents unsupported operating points, so collapse each repeat to its
    highest efficiency and evaluate the resulting step envelope.
    """
    order = np.argsort(false_positive_fraction, kind="stable")
    fpf = false_positive_fraction[order]
    eff = np.maximum.accumulate(efficiency[order])
    starts = np.flatnonzero(np.r_[True, np.diff(fpf) != 0])
    ends = np.r_[starts[1:] - 1, len(fpf) - 1]
    unique_fpf = fpf[starts]
    envelope = eff[ends]
    idx = np.searchsorted(unique_fpf, grid, side="right") - 1
    return envelope[np.clip(idx, 0, len(envelope) - 1)]


def _roc_band(pos, neg, grid, n_boot: int = 400, seed: int = 0):
    """Bootstrap central 68% interval on a common empirical-background grid."""
    rng = np.random.default_rng(seed)
    tprs = []
    for _ in range(n_boot):
        p = rng.choice(pos, pos.size, True)
        n = rng.choice(neg, neg.size, True)
        fpr, tpr = _roc_curve(p, n)
        tprs.append(_roc_at_fraction(fpr, tpr, grid))
    return np.percentile(np.asarray(tprs), [16, 84], axis=0)


def _efficiency_at_fraction(
    signal: np.ndarray,
    background: np.ndarray,
    max_fraction: float,
) -> tuple[float, float, int, float]:
    """Efficiency at the most permissive threshold with FPF <= target."""
    signal = signal[np.isfinite(signal)]
    background = background[np.isfinite(background)]
    max_false_positives = int(np.floor(max_fraction * background.size))
    if max_false_positives >= background.size:
        threshold = -np.inf
    else:
        threshold = np.sort(background)[::-1][max_false_positives]
    false_positives = int(np.sum(background > threshold))
    observed_fraction = false_positives / background.size
    efficiency = float(np.mean(signal > threshold))
    return efficiency, observed_fraction, false_positives, float(threshold)


def _stats(d: pd.DataFrame) -> dict:
    """Every per-population number the manuscript quotes, for one row subset."""
    sig, gli = d[d.cls == "inj_ccsn"], d[d.cls == "real_glitch"]
    out: dict = {"n_signal": len(sig), "n_glitch": len(gli)}
    for label, s in (("signal", sig), ("glitch", gli)):
        rho, p = stats.spearmanr(s[LNO], s[LNBF])
        out[label] = {
            "median_lno": float(np.median(s[LNO])),
            "median_lnbf": float(np.median(s[LNBF])),
            "frac_lno_positive": float(np.mean(s[LNO] > 0)),
            "frac_lnbf_positive": float(np.mean(s[LNBF] > 0)),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "sign_agreement": float(np.mean(np.sign(s[LNO]) == np.sign(s[LNBF]))),
            "both_positive": int(np.sum((s[LNO] > 0) & (s[LNBF] > 0))),
            "lno_positive_only": int(np.sum((s[LNO] > 0) & (s[LNBF] <= 0))),
            "lnbf_positive_only": int(np.sum((s[LNO] <= 0) & (s[LNBF] > 0))),
            "both_negative": int(np.sum((s[LNO] <= 0) & (s[LNBF] <= 0))),
        }
    for name, col in (("lno", LNO), ("lnbf", LNBF)):
        f, b = sig[col].to_numpy(), gli[col].to_numpy()
        out[f"auc_{name}"] = _auc(f, b)
        out[f"auc_{name}_err"] = _auc_err(f, b)
        out[f"operating_points_{name}"] = {}
        for target in (0.10, 0.05, 0.01):
            efficiency, observed, n_false, threshold = _efficiency_at_fraction(
                f, b, target
            )
            out[f"operating_points_{name}"][f"{target:g}"] = {
                "target_false_positive_fraction": target,
                "observed_false_positive_fraction": observed,
                "n_false_positives": n_false,
                "threshold": threshold,
                "efficiency": efficiency,
            }
        out[f"efficiency_{name}"] = {
            key: value["efficiency"]
            for key, value in out[f"operating_points_{name}"].items()
        }
    out["median_lnbf_err"] = float(np.nanmedian(d.bw_lnbf_signal_glitch_err))
    out["median_lno_err"] = float(np.nanmedian(d.lno_log_odds_err))
    out.update(_sign_agreement(d))
    return out


def _sign_agreement(d: pd.DataFrame) -> dict:
    """Pooled sign agreement and Cohen's kappa.

    Agreement must be pooled across both populations. Within a single
    population each statistic's sign is near-constant, so the observed
    agreement is fixed by the two marginals and carries no information about
    whether the analyses track each other: per-population kappa is ~0 even
    when the pooled association is strong.
    """
    lno, lnbf = np.sign(d[LNO]), np.sign(d[LNBF])
    observed = float(np.mean(lno == lnbf))
    expected = float(
        np.mean(lno > 0) * np.mean(lnbf > 0) + np.mean(lno < 0) * np.mean(lnbf < 0)
    )
    kappa = (observed - expected) / (1.0 - expected)
    kappa_err = np.sqrt(observed * (1.0 - observed) / len(d)) / (1.0 - expected)
    return {
        "pooled_sign_agreement": observed,
        "chance_expected_agreement": expected,
        "cohen_kappa": float(kappa),
        "cohen_kappa_err": float(kappa_err),
    }


def _snr_trend(d: pd.DataFrame, edges) -> list[dict]:
    """Both statistics on injections, binned by target network SNR."""
    sig = d[d.cls == "inj_ccsn"]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = sig[(sig.bw_target_snr >= lo) & (sig.bw_target_snr < hi)]
        if s.empty:
            continue
        rows.append(
            {
                "snr_lo": lo,
                "snr_hi": hi,
                "n": len(s),
                "median_lnbf": float(np.median(s[LNBF])),
                "median_lno": float(np.median(s[LNO])),
                "frac_lnbf_positive": float(np.mean(s[LNBF] > 0)),
                "frac_lno_positive": float(np.mean(s[LNO] > 0)),
            }
        )
    return rows


def make_figure(d: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))

    ax = axes[0]
    for cls in ("real_glitch", "inj_ccsn"):
        s = d[d.cls == cls]
        ax.scatter(
            s[LNO], s[LNBF], s=6, alpha=0.45, lw=0,
            color=CLS_COLOR[cls], label=CLS_LABEL[cls],
        )
    ax.axhline(0, color="0.5", lw=0.7, ls=":")
    ax.axvline(0, color="0.5", lw=0.7, ls=":")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel(r"$\ln\mathcal{O}$")
    ax.set_ylabel(r"$\ln\mathcal{B}_{\rm S/G}$ (\textsc{BayesWave})".replace("\\textsc{BayesWave}", "BayesWave"))
    ax.legend(loc="lower right", frameon=False, markerscale=2.2)

    ax = axes[1]
    sig, gli = d[d.cls == "inj_ccsn"], d[d.cls == "real_glitch"]
    resolution = 1.0 / len(gli)
    grid = np.logspace(np.log10(resolution), 0, 400)
    for name, col, label in (
        ("lno", LNO, r"$\ln\mathcal{O}$"),
        ("bw", LNBF, r"$\ln\mathcal{B}_{\rm S/G}$"),
    ):
        fpr, tpr = _roc_curve(sig[col].to_numpy(), gli[col].to_numpy())
        empirical = _roc_at_fraction(fpr, tpr, grid)
        lo, hi = _roc_band(sig[col].to_numpy(), gli[col].to_numpy(), grid)
        auc = _auc(sig[col].to_numpy(), gli[col].to_numpy())
        ax.step(
            grid,
            empirical,
            where="post",
            color=COLOR[name],
            lw=1.4,
            label=f"{label}  (AUC ${auc:.3f}$)",
        )
        ax.fill_between(grid, lo, hi, color=COLOR[name], alpha=0.2, lw=0)
    ax.axvline(resolution, color="0.5", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(grid[0], 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive fraction (real blips)")
    ax.set_ylabel("Detection efficiency")
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out / "fig_bayeswave_comparison.pdf")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--max-lnbf-err", type=float, default=5.0,
                    help="Uncertainty cut defining the best-converged subset. "
                         "BayesWave differences two lnZ of order 6e5, so runs "
                         "whose reported uncertainty is comparable to lnBF "
                         "itself carry no sign information.")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.paired)
    full = raw[raw.bw_recovered == 0]
    clean = full[full.bw_lnbf_signal_glitch_err < args.max_lnbf_err]

    make_figure(full, args.outdir)

    metrics = {
        "n_paired_total": len(raw),
        "n_recovered": int(raw.bw_recovered.sum()),
        "n_recovered_without_uncertainty": int(
            (~np.isfinite(raw.bw_lnbf_signal_glitch_err) & (raw.bw_recovered == 1)).sum()
        ),
        "n_posteriors_available": int(raw.bw_posteriors_available.sum()),
        "n_posteriors_reported": int(raw.bw_posteriors_reported.sum())
        if "bw_posteriors_reported" in raw
        else int(raw.bw_posteriors_available.sum()),
        "n_posteriors_comparison_ready": int(raw.bw_posteriors_comparison_ready.sum())
        if "bw_posteriors_comparison_ready" in raw
        else 0,
        "max_lnbf_err": args.max_lnbf_err,
        "full_quality": _stats(full),
        "best_converged": _stats(clean),
        "snr_trend": _snr_trend(full, [10, 15, 20, 25, 40, 1e9]),
        "spearman_lnbf_vs_snr": float(
            stats.spearmanr(
                full[full.cls == "inj_ccsn"].bw_target_snr,
                full[full.cls == "inj_ccsn"][LNBF],
            ).statistic
        ),
        "spearman_lno_vs_snr": float(
            stats.spearmanr(
                full[full.cls == "inj_ccsn"].bw_target_snr,
                full[full.cls == "inj_ccsn"][LNO],
            ).statistic
        ),
    }
    dest = args.outdir / "bayeswave_metrics.json"
    dest.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {dest} and fig_bayeswave_comparison.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
