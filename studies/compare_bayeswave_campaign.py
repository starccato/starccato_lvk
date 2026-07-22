"""Audit and compare morphZ log odds with BayesWave signal/glitch Bayes factors.

The comparison is deliberately paired through each event's immutable manifest.
It writes an inventory (including missing counterparts), a matched table, summary
statistics, an evidence comparison figure, and an atlas of saved NUTS posterior
predictive checks.

Example
-------
python studies/compare_bayeswave_campaign.py \
  --campaign-root /fred/oz303/avajpeyi/results/starccato_lvk/bwcomp_20260723
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _lno_uncertainty(result: dict[str, Any]) -> float | None:
    """Independent-error propagation for lnZ_signal - sum(lnZ_glitch_i)."""
    signal = _finite(result.get("logZ_signal_err"))
    glitch = [_finite(v) for v in result.get("logZ_glitch_err_by_detector", {}).values()]
    if signal is None or any(v is None for v in glitch):
        return None
    return float(math.sqrt(signal**2 + sum(v**2 for v in glitch if v is not None)))


def _nuts_diagnostics(result: dict[str, Any]) -> tuple[int, float | None, float | None]:
    divergences = 0
    rhats: list[float] = []
    ess: list[float] = []
    for model in result.get("nuts_diagnostics", {}).values():
        divergences += int(model.get("divergences", 0))
        rhats.extend(float(v) for v in model.get("max_rhat", {}).values())
        ess.extend(float(v.get("min")) for v in model.get("ess", {}).values())
    return divergences, max(rhats, default=None), min(ess, default=None)


def build_inventory(root: Path, cls: str, detector_tag: str) -> list[dict[str, Any]]:
    data_root = root / "data" / f"rn_{detector_tag}"
    bw_paths = sorted(
        (root / "bw_fixedsky").glob(f"e*/{cls}/result.json"),
        key=lambda p: int(p.parents[1].name[1:]),
    )
    rows: list[dict[str, Any]] = []
    for bw_path in bw_paths:
        bw = _read_json(bw_path)
        index = int(bw["index"])
        our_path = data_root / "results" / f"e{index}_{cls}.json"
        manifest_path = Path(bw["manifest"])
        row: dict[str, Any] = {
            "index": index,
            "bayeswave_result": str(bw_path),
            "our_result": str(our_path),
            "manifest": str(manifest_path),
            "target_snr": _finite(bw.get("target_snr")),
            "bayeswave_ln_bf": _finite(bw.get("log_bayeswave_signal_glitch")),
            "bayeswave_ln_bf_err": _finite(
                bw.get("log_bayeswave_signal_glitch_uncertainty")
            ),
            "bayeswave_reconstructed_snr": _finite(
                bw.get("signal_reconstructed_snr_median")
            ),
            "matched": False,
            "exclusion_reason": "",
        }
        if not our_path.is_file():
            row["exclusion_reason"] = "missing morphZ result"
            rows.append(row)
            continue
        if not manifest_path.is_file():
            row["exclusion_reason"] = "missing manifest"
            rows.append(row)
            continue

        ours, manifest = _read_json(our_path), _read_json(manifest_path)
        row.update(
            morphz_ln_o=_finite(ours.get("log_odds")),
            morphz_ln_o_err=_lno_uncertainty(ours),
            morphz_failures=int(ours.get("evidence_failures", 0)),
            morphz_fallbacks=int(ours.get("evidence_fallbacks", 0)),
        )
        div, max_rhat, min_ess = _nuts_diagnostics(ours)
        row.update(nuts_divergences=div, nuts_max_rhat=max_rhat, nuts_min_ess=min_ess)

        expected = manifest.get("manifest_fingerprint")
        if ours.get("manifest_fingerprint") != expected:
            row["exclusion_reason"] = "manifest fingerprint mismatch"
        elif row["morphz_ln_o"] is None or row["bayeswave_ln_bf"] is None:
            row["exclusion_reason"] = "non-finite evidence"
        elif row["morphz_failures"]:
            row["exclusion_reason"] = "morphZ evidence failure"
        else:
            row["matched"] = True
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.array([r["morphz_ln_o"] for r in rows], dtype=float)
    y = np.array([r["bayeswave_ln_bf"] for r in rows], dtype=float)
    snr = np.array([r["target_snr"] for r in rows], dtype=float)
    yerr = np.array([r["bayeswave_ln_bf_err"] for r in rows], dtype=float)
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    kendall = stats.kendalltau(x, y)
    lno_snr = stats.spearmanr(x, snr)
    bw_snr = stats.spearmanr(y, snr)
    sign = np.sign(x) == np.sign(y)
    missing = [int(r["index"]) for r in inventory if not r["matched"]]
    return {
        "n_bayeswave_results": len(inventory),
        "n_matched": len(rows),
        "excluded_indices": missing,
        "pearson_lnO_vs_lnBF": {"r": float(pearson.statistic), "p": float(pearson.pvalue)},
        "spearman_lnO_vs_lnBF": {"rho": float(spearman.statistic), "p": float(spearman.pvalue)},
        "kendall_lnO_vs_lnBF": {"tau": float(kendall.statistic), "p": float(kendall.pvalue)},
        "sign_agreement": {"n": int(sign.sum()), "fraction": float(sign.mean())},
        "bayeswave_intervals_crossing_zero": int(np.sum(np.abs(y) <= yerr)),
        "spearman_lnO_vs_target_snr": {
            "rho": float(lno_snr.statistic), "p": float(lno_snr.pvalue)
        },
        "spearman_lnBF_vs_target_snr": {
            "rho": float(bw_snr.statistic), "p": float(bw_snr.pvalue)
        },
        "morphz_evidence_failures": int(sum(r["morphz_failures"] for r in rows)),
        "morphz_evidence_fallbacks": int(sum(r["morphz_fallbacks"] for r in rows)),
        "nuts_divergences": int(sum(r["nuts_divergences"] for r in rows)),
        "nuts_max_rhat": max(r["nuts_max_rhat"] for r in rows),
        "nuts_min_ess": min(r["nuts_min_ess"] for r in rows),
    }


def plot_evidence(rows: list[dict[str, Any]], summary: dict[str, Any], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([r["morphz_ln_o"] for r in rows], dtype=float)
    xerr = np.array([r["morphz_ln_o_err"] for r in rows], dtype=float)
    y = np.array([r["bayeswave_ln_bf"] for r in rows], dtype=float)
    yerr = np.array([r["bayeswave_ln_bf_err"] for r in rows], dtype=float)
    snr = np.array([r["target_snr"] for r in rows], dtype=float)
    idx = np.array([r["index"] for r in rows], dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    color = "#0072B2"
    axes[0].errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor="0.68", lw=.7)
    points = axes[0].scatter(x, y, c=snr, cmap="viridis", s=31, edgecolor="white", lw=.4)
    axes[0].axhline(0, color="0.45", lw=.7)
    axes[0].axvline(0, color="0.45", lw=.7)
    axes[0].set(xlabel=r"morphZ $\ln\mathcal{O}_{S/G}$",
                ylabel=r"BayesWave $\ln\mathcal{B}_{S/G}$")
    fig.colorbar(points, ax=axes[0], label="target network SNR", pad=.02)

    axes[1].errorbar(snr, x, yerr=xerr, fmt="o", color=color, ecolor="0.68", ms=4)
    axes[1].axhline(0, color="0.45", lw=.7)
    axes[1].set(xlabel="target network SNR", ylabel=r"morphZ $\ln\mathcal{O}_{S/G}$")

    axes[2].errorbar(snr, y, yerr=yerr, fmt="o", color="#D55E00", ecolor="0.68", ms=4)
    axes[2].axhline(0, color="0.45", lw=.7)
    axes[2].set(xlabel="target network SNR", ylabel=r"BayesWave $\ln\mathcal{B}_{S/G}$")

    for ax, xs, ys in ((axes[0], x, y), (axes[1], snr, x), (axes[2], snr, y)):
        for event, xx, yy in zip(idx, xs, ys):
            ax.annotate(str(event), (xx, yy), xytext=(3, 3), textcoords="offset points",
                        fontsize=6, color="0.25")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", lw=.5, zorder=0)

    rho = summary["spearman_lnO_vs_lnBF"]
    agree = summary["sign_agreement"]
    fig.suptitle("Paired signal-vs-glitch evidence on identical injected events", fontsize=13)
    fig.text(.5, .935,
             f"N={len(rows)}; Spearman rho={rho['rho']:.2f} (p={rho['p']:.3g}); "
             f"sign agreement={agree['n']}/{len(rows)}. Error bars are reported evidence uncertainties.",
             ha="center", va="top", fontsize=9, color="0.3")
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pp_atlas(root: Path, rows: list[dict[str, Any]], out: Path,
                  indices: list[int], cls: str, detector_tag: str) -> list[int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_index = {int(r["index"]): r for r in rows}
    models = (("signal", "coherent signal"), ("glitch_h1", "H1 glitch"),
              ("glitch_l1", "L1 glitch"))
    included = [i for i in indices if i in by_index and all(
        (root / "data" / f"rn_{detector_tag}" / f"e{i}" / cls / "analysis" / model /
         "posterior_predictive.png").is_file() for model, _ in models)]
    if not included:
        return []
    fig, axes = plt.subplots(len(included), 3, figsize=(14, 3.15 * len(included)), squeeze=False)
    for row_no, index in enumerate(included):
        r = by_index[index]
        for col, (model, label) in enumerate(models):
            path = (root / "data" / f"rn_{detector_tag}" / f"e{index}" / cls /
                    "analysis" / model / "posterior_predictive.png")
            axes[row_no, col].imshow(plt.imread(path))
            axes[row_no, col].axis("off")
            if row_no == 0:
                axes[row_no, col].set_title(label, fontsize=11)
        axes[row_no, 0].text(
            -.02, .5,
            f"e{index}\nSNR {r['target_snr']:.1f}\nlnO {r['morphz_ln_o']:.1f}\nlnBF {r['bayeswave_ln_bf']:.1f}",
            transform=axes[row_no, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle("NUTS posterior-predictive checks for representative evidence cases", fontsize=14)
    fig.tight_layout(rect=(.08, 0, 1, .975), h_pad=.8, w_pad=.2)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return included


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", type=Path, required=True)
    ap.add_argument("--class", dest="cls", default="inj_ccsn")
    ap.add_argument("--detector-tag", default="H1_L1")
    ap.add_argument("--outdir", type=Path)
    ap.add_argument("--atlas-indices", default="0,2,16,22,70,74")
    args = ap.parse_args()

    root = args.campaign_root.resolve()
    outdir = args.outdir or root / "comparisons"
    outdir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(root, args.cls, args.detector_tag)
    paired = [row for row in inventory if row["matched"]]
    if len(paired) < 3:
        raise SystemExit(f"need at least three matched events; found {len(paired)}")
    summary = summarize(paired, inventory)
    _write_csv(outdir / "campaign_inventory.csv", inventory)
    _write_csv(outdir / "paired_evidence.csv", paired)
    plot_evidence(paired, summary, outdir / "evidence_comparison")
    indices = [int(v) for v in args.atlas_indices.split(",") if v.strip()]
    summary["posterior_predictive_atlas_indices"] = plot_pp_atlas(
        root, paired, outdir / "posterior_predictive_atlas", indices,
        args.cls, args.detector_tag)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote comparison artifacts to {outdir}")


if __name__ == "__main__":
    main()
