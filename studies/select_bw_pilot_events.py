"""Select the controlled BayesWave pilot cohort from the paired lnO/BayesWave table.

The pilot asks one question: does BayesWave's disagreement with the VAE survive a
4x longer chain and three independent seeds, or was it a sampling artefact of the
1e6-iteration production settings?

To answer that without stacking the deck, the cohort is stratified into four
groups of ten:

* ``inj_disagree``  low-SNR injections where the two pipelines disagree in sign;
* ``inj_agree``     injections where they agree, SNR-matched one-to-one to the
                    disagreements so the two injection strata sit at the same
                    place on the SNR axis;
* ``glitch_disagree`` real blips where they disagree;
* ``glitch_agree``    real blips where they agree, SNR-matched the same way.

The SNR matching matters: without it the "agreement" control would sit at high
SNR where everything is easy, and a stable agreement stratum would tell us
nothing about whether the disagreements are stable.

Only full-quality rows are eligible (``bw_recovered=0`` and
``bw_degenerate_uncertainty=0``): a row whose evidence was salvaged from a log
has no trustworthy baseline to compare the pilot against.

Usage:
    python studies/select_bw_pilot_events.py \
        --paired /fred/oz303/.../paired_lno_vs_bayeswave.csv \
        --out /fred/oz303/.../bw_seed_pilot/pilot_cohort.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Sign convention: an injected CCSN should score positive, a real blip negative.
TRUE_SIGN = {"inj_ccsn": +1.0, "real_glitch": -1.0}

STRATA = (
    ("inj_disagree", "inj_ccsn", False),
    ("inj_agree", "inj_ccsn", True),
    ("glitch_disagree", "real_glitch", False),
    ("glitch_agree", "real_glitch", True),
)


def _read_paired(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict[str, Any] = {"cls": raw["cls"]}
            for key, value in raw.items():
                if key == "cls" or key == "bw_evidence_source":
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = math.nan
            row["bw_evidence_source"] = raw["bw_evidence_source"]
            rows.append(row)
    return rows


def _eligible(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = []
    for row in rows:
        if row["bw_recovered"] or row["bw_degenerate_uncertainty"]:
            continue
        if row["cls"] not in TRUE_SIGN:
            continue
        values = (
            row["lno_log_odds"],
            row["bw_lnbf_signal_glitch"],
            row["bw_target_snr"],
        )
        if not all(math.isfinite(v) for v in values):
            continue
        keep.append(row)
    return keep


def _annotate(row: dict[str, Any]) -> dict[str, Any]:
    """Attach the sign-level verdicts the strata are defined on."""
    truth = TRUE_SIGN[row["cls"]]
    lno_correct = math.copysign(1.0, row["lno_log_odds"]) == truth
    bw_correct = math.copysign(1.0, row["bw_lnbf_signal_glitch"]) == truth
    row["lno_correct"] = bool(lno_correct)
    row["bw_correct"] = bool(bw_correct)
    row["agree"] = bool(lno_correct == bw_correct)
    return row


def _snr_match(
    targets: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Greedy nearest-SNR matching without replacement.

    Targets are matched in order of how isolated they are on the SNR axis (the
    hardest to match first), so a crowded target does not consume the only
    partner an isolated one had.
    """
    available = list(pool)
    if not available:
        return []
    pool_snr = np.array([r["bw_target_snr"] for r in available])
    order = sorted(
        targets,
        key=lambda t: -float(np.min(np.abs(pool_snr - t["bw_target_snr"]))),
    )
    matched: list[dict[str, Any]] = []
    for target in order:
        if not available or len(matched) >= count:
            break
        snr = np.array([r["bw_target_snr"] for r in available])
        pick = int(np.argmin(np.abs(snr - target["bw_target_snr"])))
        chosen = available.pop(pick)
        chosen["snr_matched_to_index"] = int(target["index"])
        matched.append(chosen)
    return matched


def select(rows: list[dict[str, Any]], per_stratum: int) -> dict[str, list[dict]]:
    rows = [_annotate(row) for row in _eligible(rows)]
    cohort: dict[str, list[dict]] = {}
    for name, cls, agree in STRATA:
        candidates = [r for r in rows if r["cls"] == cls and r["agree"] is agree]
        if agree:
            # Match the agreement control to the disagreements from the same
            # class, so the two strata share an SNR distribution.
            targets = cohort[name.replace("agree", "disagree")]
            cohort[name] = _snr_match(targets, candidates, per_stratum)
        else:
            # Lowest SNR first: the low-SNR disagreements are the population the
            # whole pilot exists to interrogate.
            candidates.sort(key=lambda r: r["bw_target_snr"])
            cohort[name] = candidates[:per_stratum]
    return cohort


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-stratum", type=int, default=10)
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=[11, 22, 33],
        help="Independent chain/data seeds; one BayesWave run per event per seed",
    )
    args = ap.parse_args()

    rows = _read_paired(args.paired)
    cohort = select(rows, args.per_stratum)

    tasks: list[dict[str, Any]] = []
    for stratum, members in cohort.items():
        for member in members:
            for seed in args.seeds:
                tasks.append({
                    "stratum": stratum,
                    "cls": member["cls"],
                    "index": int(member["index"]),
                    "seed": int(seed),
                })

    payload = {
        "source_paired_table": str(args.paired.resolve()),
        "per_stratum": args.per_stratum,
        "seeds": list(args.seeds),
        "n_events": sum(len(v) for v in cohort.values()),
        "n_tasks": len(tasks),
        "strata": {
            name: {
                "n": len(members),
                "snr": [round(m["bw_target_snr"], 3) for m in members],
                "events": members,
            }
            for name, members in cohort.items()
        },
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # Flat task list the SLURM array indexes by $SLURM_ARRAY_TASK_ID.
    task_file = args.out.with_name("pilot_tasks.txt")
    task_file.write_text(
        "".join(f"{t['cls']} {t['index']} {t['seed']}\n" for t in tasks)
    )

    for name, members in cohort.items():
        snr = [m["bw_target_snr"] for m in members]
        span = f"{min(snr):.1f}-{max(snr):.1f}" if snr else "n/a"
        print(f"{name:18s} n={len(members):2d}  target SNR {span}")
        if len(members) < args.per_stratum:
            print(f"  WARNING: only {len(members)} of {args.per_stratum} available")
    print(f"\n{len(tasks)} BayesWave runs ({payload['n_events']} events x "
          f"{len(args.seeds)} seeds)")
    print(f"wrote {args.out}\nwrote {task_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
