"""Evaluate the pre-registered, three-seed BayesWave trust check.

The comparison is deliberately sign-level.  An event is seed-converged only if
all requested seeds finish with finite signal, glitch, and noise evidences and
the seeds agree on the sign of both BayesWave statistics: native S/G and the
hypothesis-aligned S/(G or N).  Reported evidence uncertainties are recorded but
never used to select the cohort or accept convergence.
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

TRUE_SIGN = {"inj_ccsn": 1, "real_glitch": -1}
QUANTITIES = (
    "logZ_signal",
    "logZ_glitch",
    "logZ_noise",
    "native",
    "aligned",
)


def _load_seed(path: Path) -> dict[str, float] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    uncertainty = data.get("evidence_uncertainty") or {}
    row = {
        "logZ_signal": data.get("logZ_signal"),
        "logZ_glitch": data.get("logZ_glitch"),
        "logZ_noise": data.get("logZ_noise"),
        "unc_signal": uncertainty.get("signal"),
        "unc_glitch": uncertainty.get("glitch"),
        "unc_noise": uncertainty.get("noise"),
    }
    if any(value is None or not math.isfinite(float(value)) for value in row.values()):
        return None
    row = {key: float(value) for key, value in row.items()}
    row["native"] = row["logZ_signal"] - row["logZ_glitch"]
    row["native_unc"] = math.hypot(row["unc_signal"], row["unc_glitch"])
    aligned, weight_glitch, weight_noise = aligned_signal_vs_glitch_or_noise(
        row["logZ_signal"], row["logZ_glitch"], row["logZ_noise"]
    )
    row["aligned"] = aligned
    row["aligned_unc"] = math.sqrt(
        row["unc_signal"] ** 2
        + (weight_glitch * row["unc_glitch"]) ** 2
        + (weight_noise * row["unc_noise"]) ** 2
    )
    return row


def _read_lno(path: Path) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                key = (raw["cls"], int(float(raw["index"])))
                value = float(raw["lno_log_odds"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                out[key] = value
    return out


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "values": [float(value) for value in array],
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "range": float(np.ptp(array)),
    }


def _cohort_events(cohort: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event_class in ("inj_ccsn", "real_glitch")
        for event in cohort["classes"][event_class]["events"]
    ]


def analyse(
    pilot_root: Path,
    cohort: dict[str, Any],
    lno: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in cohort["seeds"]]
    rows: list[dict[str, Any]] = []
    for event in _cohort_events(cohort):
        cls = event["cls"]
        index = int(event["index"])
        per_seed: dict[int, dict[str, float]] = {}
        for seed in seeds:
            result = _load_seed(
                pilot_root / f"e{index}" / cls / f"seed{seed}" / "result.json"
            )
            if result is not None:
                per_seed[seed] = result

        record: dict[str, Any] = {
            "cls": cls,
            "index": index,
            "requested_seeds": seeds,
            "completed_seeds": sorted(per_seed),
            "n_seeds_complete": len(per_seed),
            "lno_log_odds": lno.get((cls, index), math.nan),
        }
        if len(per_seed) != len(seeds):
            record["outcome"] = "incomplete"
            rows.append(record)
            continue

        for quantity in QUANTITIES:
            values = [per_seed[seed][quantity] for seed in seeds]
            for name, value in _stats(values).items():
                record[f"{quantity}_{name}"] = value
        for quantity in ("signal", "glitch", "noise", "native", "aligned"):
            key = f"unc_{quantity}" if quantity in {"signal", "glitch", "noise"} else f"{quantity}_unc"
            record[f"{quantity}_reported_unc_mean"] = float(
                np.mean([per_seed[seed][key] for seed in seeds])
            )

        native_signs = {_sign(per_seed[seed]["native"]) for seed in seeds}
        aligned_signs = {_sign(per_seed[seed]["aligned"]) for seed in seeds}
        record["native_seed_signs"] = sorted(native_signs)
        record["aligned_seed_signs"] = sorted(aligned_signs)
        record["outcome"] = (
            "seed_consistent"
            if len(native_signs) == 1
            and 0 not in native_signs
            and len(aligned_signs) == 1
            and 0 not in aligned_signs
            else "seed_inconsistent"
        )
        if record["outcome"] == "seed_consistent":
            truth = TRUE_SIGN[cls]
            record["vae_expected"] = _sign(record["lno_log_odds"]) == truth
            record["native_expected"] = next(iter(native_signs)) == truth
            record["aligned_expected"] = next(iter(aligned_signs)) == truth
        rows.append(record)
    return rows


def _contingency(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in ("inj_ccsn", "real_glitch", "total"):
        members = rows if cls == "total" else [row for row in rows if row["cls"] == cls]
        cells = {"both": 0, "vae_only": 0, "bw_only": 0, "neither": 0}
        for row in members:
            vae = bool(row["vae_expected"])
            bw = bool(row[f"{metric}_expected"])
            cell = "both" if vae and bw else "vae_only" if vae else "bw_only" if bw else "neither"
            cells[cell] += 1
        out[cls] = {"n": len(members), **cells}
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    converged = [row for row in rows if row["outcome"] == "seed_consistent"]
    complete = [row for row in rows if row["outcome"] != "incomplete"]
    seed_spread = {}
    for quantity in QUANTITIES:
        ranges = [row[f"{quantity}_range"] for row in complete]
        seed_spread[quantity] = {
            "median_range": float(np.median(ranges)) if ranges else math.nan,
            "max_range": float(np.max(ranges)) if ranges else math.nan,
        }
    return {
        "n_attempted": len(rows),
        "n_seed_consistent": len(converged),
        "n_seed_inconsistent": sum(
            row["outcome"] == "seed_inconsistent" for row in rows
        ),
        "n_incomplete": sum(row["outcome"] == "incomplete" for row in rows),
        "acceptance_criterion": (
            "all requested seeds complete with finite S/G/N evidences and have "
            "identical non-zero signs for native S/G and aligned S/(G or N)"
        ),
        "reported_uncertainty_used_for_acceptance": False,
        "seed_spread": seed_spread,
        "contingency": {
            "native": _contingency(converged, "native"),
            "aligned": _contingency(converged, "aligned"),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _write_latex(path: Path, summary: dict[str, Any]) -> None:
    labels = {"inj_ccsn": "Injected CCSN", "real_glitch": "Cataloged blip", "total": "Total"}
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\hline\hline",
        r"Statistic & Class & $N$ & Both & VAE only & BW only & Neither \\",
        r"\hline",
    ]
    for metric, metric_label in (("native", r"$\ln\mathcal{B}_{S/G}$"), ("aligned", r"$\ln\mathcal{B}_{S/(G\lor N)}$")):
        for position, cls in enumerate(("inj_ccsn", "real_glitch", "total")):
            block = summary["contingency"][metric][cls]
            label = metric_label if position == 0 else ""
            lines.append(
                f"{label} & {labels[cls]} & {block['n']} & {block['both']} & "
                f"{block['vae_only']} & {block['bw_only']} & {block['neither']} \\\\"
            )
        if metric == "native":
            lines.append(r"\hline")
    lines.extend([r"\hline\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()

    pilot_root = args.pilot_root.resolve()
    cohort = json.loads((args.cohort or pilot_root / "pilot_cohort.json").read_text())
    rows = analyse(pilot_root, cohort, _read_lno(args.paired))
    summary = summarise(rows)

    outdir = args.outdir or pilot_root / "convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "seed_convergence.csv", rows)
    (outdir / "seed_convergence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_latex(args.table or outdir / "bayeswave_sign_table.tex", summary)

    print(
        f"attempted={summary['n_attempted']}, "
        f"seed-consistent={summary['n_seed_consistent']}, "
        f"seed-inconsistent={summary['n_seed_inconsistent']}, "
        f"incomplete={summary['n_incomplete']}"
    )
    for metric in ("native", "aligned"):
        cells = summary["contingency"][metric]["total"]
        print(f"{metric:7s}: {cells}")
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
