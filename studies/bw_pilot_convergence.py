"""Decide whether the production BayesWave settings were converged.

Reads the three-seed 4e6-iteration pilot written by slurm/bayeswave_seed_pilot.sh
and answers two separate questions per event:

1. **Is the pilot itself converged?**  Compare the seed-to-seed spread of each
   statistic against BayesWave's own reported evidence uncertainty.  Three seeds
   drawn from a converged sampler should scatter no more than that uncertainty
   allows; the test statistic is the reduced chi-square of the three seeds about
   their inverse-variance-weighted mean.  A large value means the reported
   uncertainty understates the real sampling error, which is itself a finding.

2. **Did the production run get the right answer?**  Compare the pilot's weighted
   mean against the 1e6-iteration production value, in units of the combined
   uncertainty.  A systematic shift -- especially one that moves the low-SNR
   disagreements -- means the production settings were inadequate.  Uncertainty
   that merely shrinks without the central value moving means they were fine.

Both the native lnB_S/G and the hypothesis-aligned lnB_S/(GvN) are reported,
because the two can move differently: an event whose noise evidence dominates has
an aligned statistic that barely depends on lnZ_glitch at all.

Usage:
    python studies/bw_pilot_convergence.py \
        --pilot-root /fred/oz303/.../bw_seed_pilot \
        --paired /fred/oz303/.../paired_lno_vs_bayeswave.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from starccato_lvk.bayeswave import (  # noqa: E402
    aligned_signal_vs_glitch_or_noise,
)

TRUE_SIGN = {"inj_ccsn": +1.0, "real_glitch": -1.0}

# A reduced chi-square this large over three seeds is not sampling noise: it says
# the seeds disagree by more than the reported evidence uncertainties allow. 3.0
# is roughly the 95th percentile of chi2_2/2, so a handful of exceedances across
# 40 events is expected and a systematic excess is not.
CHI2_CONVERGED = 3.0
# Shift of the pilot mean away from the production value, in sigma. Beyond this
# the production number was not measuring the same quantity the pilot is.
SHIFT_MATERIAL = 3.0


def _aligned(row: dict[str, float]) -> tuple[float, float]:
    value, w_glitch, w_noise = aligned_signal_vs_glitch_or_noise(
        row["logZ_signal"], row["logZ_glitch"], row["logZ_noise"]
    )
    unc = math.sqrt(
        row["unc_signal"] ** 2
        + (w_glitch * row["unc_glitch"]) ** 2
        + (w_noise * row["unc_noise"]) ** 2
    )
    return value, unc


def _load_seed(path: Path) -> dict[str, float] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    unc = data.get("evidence_uncertainty") or {}
    row = {
        "logZ_signal": data.get("logZ_signal"),
        "logZ_glitch": data.get("logZ_glitch"),
        "logZ_noise": data.get("logZ_noise"),
        "unc_signal": unc.get("signal"),
        "unc_glitch": unc.get("glitch"),
        "unc_noise": unc.get("noise"),
        "elapsed_seconds": data.get("elapsed_seconds") or math.nan,
    }
    if any(v is None or not math.isfinite(float(v)) for k, v in row.items()
           if k != "elapsed_seconds"):
        return None
    row = {k: float(v) for k, v in row.items()}
    row["native"] = row["logZ_signal"] - row["logZ_glitch"]
    row["native_unc"] = math.hypot(row["unc_signal"], row["unc_glitch"])
    row["aligned"], row["aligned_unc"] = _aligned(row)
    return row


def _weighted(values: np.ndarray, uncs: np.ndarray) -> tuple[float, float, float]:
    """Inverse-variance mean, its uncertainty, and the reduced chi-square."""
    weights = 1.0 / np.square(uncs)
    mean = float(np.sum(weights * values) / np.sum(weights))
    mean_unc = float(math.sqrt(1.0 / np.sum(weights)))
    dof = max(len(values) - 1, 1)
    chi2 = float(np.sum(weights * np.square(values - mean)) / dof)
    return mean, mean_unc, chi2


def _read_paired(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            key = (raw["cls"], int(float(raw["index"])))
            row = {}
            for name in ("lno_log_odds", "bw_lnbf_signal_glitch",
                         "bw_lnbf_signal_glitch_err", "bw_logZ_signal",
                         "bw_logZ_glitch", "bw_logZ_noise", "bw_target_snr"):
                try:
                    row[name] = float(raw[name])
                except (TypeError, ValueError):
                    row[name] = math.nan
            out[key] = row
    return out


def analyse(pilot_root: Path, cohort: dict, paired: dict) -> list[dict[str, Any]]:
    strata = {
        (member["cls"], int(member["index"])): name
        for name, block in cohort["strata"].items()
        for member in block["events"]
    }
    seeds = cohort["seeds"]
    rows: list[dict[str, Any]] = []
    for (cls, index), stratum in sorted(strata.items(), key=lambda kv: kv[1]):
        per_seed = {}
        for seed in seeds:
            row = _load_seed(
                pilot_root / f"e{index}" / cls / f"seed{seed}" / "result.json"
            )
            if row is not None:
                per_seed[seed] = row
        base = paired.get((cls, index), {})
        record: dict[str, Any] = {
            "stratum": stratum,
            "cls": cls,
            "index": index,
            "target_snr": base.get("bw_target_snr", math.nan),
            "lno_log_odds": base.get("lno_log_odds", math.nan),
            "n_seeds_complete": len(per_seed),
            "outcome": "",
        }
        if len(per_seed) < 2:
            # An explicit outcome category, never a silent drop: a cohort that
            # cannot be sampled is a result about BayesWave, not missing data.
            record["outcome"] = "incomplete"
            rows.append(record)
            continue

        for label in ("native", "aligned"):
            values = np.array([r[label] for r in per_seed.values()])
            uncs = np.array([r[f"{label}_unc"] for r in per_seed.values()])
            mean, mean_unc, chi2 = _weighted(values, uncs)
            record[f"{label}_pilot_mean"] = mean
            record[f"{label}_pilot_mean_err"] = mean_unc
            record[f"{label}_seed_spread"] = float(np.std(values, ddof=1))
            record[f"{label}_reported_unc"] = float(np.mean(uncs))
            record[f"{label}_reduced_chi2"] = chi2
            record[f"{label}_seed_values"] = [float(v) for v in values]

        production = base.get("bw_lnbf_signal_glitch", math.nan)
        production_err = base.get("bw_lnbf_signal_glitch_err", math.nan)
        combined = math.hypot(production_err, record["native_pilot_mean_err"])
        shift = record["native_pilot_mean"] - production
        record["production_native"] = production
        record["native_shift"] = shift
        record["native_shift_sigma"] = (
            shift / combined if combined > 0 and math.isfinite(combined) else math.nan
        )
        if math.isfinite(production):
            base_aligned, _, _ = aligned_signal_vs_glitch_or_noise(
                base["bw_logZ_signal"], base["bw_logZ_glitch"], base["bw_logZ_noise"]
            )
            record["production_aligned"] = base_aligned
            record["aligned_shift"] = record["aligned_pilot_mean"] - base_aligned

        truth = TRUE_SIGN[cls]
        record["production_native_correct"] = bool(
            math.copysign(1.0, production) == truth
        )
        record["pilot_native_correct"] = bool(
            math.copysign(1.0, record["native_pilot_mean"]) == truth
        )
        record["pilot_aligned_correct"] = bool(
            math.copysign(1.0, record["aligned_pilot_mean"]) == truth
        )
        record["verdict_flipped"] = bool(
            record["production_native_correct"] != record["pilot_native_correct"]
        )

        seed_ok = record["native_reduced_chi2"] <= CHI2_CONVERGED
        shifted = abs(record["native_shift_sigma"]) > SHIFT_MATERIAL
        if not seed_ok:
            record["outcome"] = "seed_inconsistent"
        elif shifted:
            record["outcome"] = "shifted"
        else:
            record["outcome"] = "stable"
        rows.append(record)
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        members = [r for r in rows if r["stratum"] == stratum]
        usable = [r for r in members if r["outcome"] != "incomplete"]
        block: dict[str, Any] = {
            "n": len(members),
            "outcomes": {
                name: sum(1 for r in members if r["outcome"] == name)
                for name in ("stable", "shifted", "seed_inconsistent", "incomplete")
            },
        }
        if usable:
            block.update(
                n_verdict_flipped=sum(1 for r in usable if r["verdict_flipped"]),
                median_abs_shift_sigma=float(np.median(
                    [abs(r["native_shift_sigma"]) for r in usable])),
                median_reduced_chi2=float(np.median(
                    [r["native_reduced_chi2"] for r in usable])),
                # Does the aligned statistic change BayesWave's verdict? If the
                # native and aligned columns disagree often, the two pipelines
                # were being compared on different hypotheses all along.
                n_native_correct=sum(1 for r in usable if r["pilot_native_correct"]),
                n_aligned_correct=sum(1 for r in usable if r["pilot_aligned_correct"]),
            )
        by_stratum[stratum] = block

    usable = [r for r in rows if r["outcome"] != "incomplete"]
    n_shifted = sum(1 for r in usable if r["outcome"] != "stable")
    decision = "insufficient_data"
    if usable:
        fraction = n_shifted / len(usable)
        if fraction <= 0.1:
            decision = "settings_adequate"
        elif fraction >= 0.25:
            decision = "rerun_full_cohort"
        else:
            decision = "ambiguous"
    return {
        "n_events": len(rows),
        "n_usable": len(usable),
        "n_incomplete": len(rows) - len(usable),
        "n_not_stable": n_shifted,
        "decision": decision,
        "decision_meaning": {
            "settings_adequate": "pilot reproduces production within uncertainty; "
                                 "the VAE advantage on this population is real",
            "rerun_full_cohort": "production settings materially misplaced the "
                                 "scores; re-run the whole matched cohort at 4e6",
            "ambiguous": "between 10% and 25% of the pilot moved; extend the "
                         "pilot before deciding",
            "insufficient_data": "not enough completed seeds to decide",
        },
        "thresholds": {
            "reduced_chi2_converged": CHI2_CONVERGED,
            "shift_material_sigma": SHIFT_MATERIAL,
        },
        "by_stratum": by_stratum,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: (json.dumps(v) if isinstance(v, list) else v)
                for k, v in row.items()
            })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-root", type=Path, required=True)
    ap.add_argument("--paired", type=Path, required=True)
    ap.add_argument("--cohort", type=Path,
                    help="pilot_cohort.json (default: <pilot-root>/pilot_cohort.json)")
    ap.add_argument("--outdir", type=Path)
    args = ap.parse_args()

    pilot_root = args.pilot_root.resolve()
    cohort = json.loads(
        (args.cohort or pilot_root / "pilot_cohort.json").read_text()
    )
    paired = _read_paired(args.paired)
    rows = analyse(pilot_root, cohort, paired)
    summary = summarise(rows)

    outdir = args.outdir or pilot_root / "convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "seed_convergence.csv", rows)
    (outdir / "seed_convergence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(f"{summary['n_usable']}/{summary['n_events']} events with >=2 seeds")
    for stratum, block in summary["by_stratum"].items():
        counts = ", ".join(f"{k}={v}" for k, v in block["outcomes"].items() if v)
        print(f"  {stratum:18s} n={block['n']:2d}  {counts}")
        if "n_native_correct" in block:
            print(f"    correct at 4e6: native={block['n_native_correct']}, "
                  f"aligned={block['n_aligned_correct']}; "
                  f"verdict flips vs production={block['n_verdict_flipped']}")
    print(f"\ndecision: {summary['decision']} -- "
          f"{summary['decision_meaning'][summary['decision']]}")
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
