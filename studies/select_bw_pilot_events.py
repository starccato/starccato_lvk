"""Pre-register a small, class-balanced BayesWave trust-check cohort.

Selection uses only prepared event manifests, before any BayesWave result is
read.  The default rule is deliberately mechanical: the first 12 numerical
indices from each class, for 24 events total.  Each event is assigned three
independent seeds.

Usage:
    python studies/select_bw_pilot_events.py \
        --ccsn-manifests /fred/.../bwcomp_nml_v044_ccsn/data/rn_H1_L1 \
        --glitch-manifests /fred/.../bwcomp_nml_v044_glitch/data/rn_H1_L1 \
        --out /fred/.../bw_seed_pilot/pilot_cohort.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

CLASS_ROOTS = {
    "inj_ccsn": "ccsn_manifests",
    "real_glitch": "glitch_manifests",
}


def _manifest_index(path: Path) -> int:
    match = re.fullmatch(r"e(\d+)", path.parent.name)
    if not match:
        raise ValueError(f"manifest parent is not an e<index> directory: {path}")
    return int(match.group(1))


def eligible_manifests(root: Path, event_class: str) -> list[dict[str, Any]]:
    """Return prepared H1--L1 manifests without inspecting BayesWave outputs."""

    events: list[dict[str, Any]] = []
    for path in root.glob("e*/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
            index = _manifest_index(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        detectors = {str(ifo).upper() for ifo in manifest.get("detectors", [])}
        bundles = manifest.get("bundles", {}).get(event_class, {})
        if not {"H1", "L1"}.issubset(detectors) or not {"H1", "L1"}.issubset(
            bundles
        ):
            continue
        events.append(
            {
                "cls": event_class,
                "index": index,
                "manifest": str(path.resolve()),
            }
        )
    return sorted(events, key=lambda row: row["index"])


def select(
    roots: dict[str, Path], per_class: int, seeds: list[int]
) -> dict[str, Any]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least three distinct seeds")

    classes: dict[str, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    for event_class in CLASS_ROOTS:
        eligible = eligible_manifests(roots[event_class], event_class)
        chosen = eligible[:per_class]
        if len(chosen) < per_class:
            raise ValueError(
                f"{event_class}: requested {per_class} events but only "
                f"{len(chosen)} eligible manifests were found"
            )
        classes[event_class] = {
            "n": len(chosen),
            "manifest_root": str(roots[event_class].resolve()),
            "indices": [row["index"] for row in chosen],
            "events": chosen,
        }
        for event in chosen:
            for seed in seeds:
                tasks.append({**event, "seed": seed})

    return {
        "selection_rule": (
            f"first {per_class} numerical event indices per class among prepared "
            "H1--L1 manifests; selected without reading BayesWave outputs"
        ),
        "per_class": per_class,
        "seeds": seeds,
        "n_events": 2 * per_class,
        "n_tasks": len(tasks),
        "classes": classes,
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccsn-manifests", type=Path, required=True)
    parser.add_argument("--glitch-manifests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    args = parser.parse_args()

    payload = select(
        {
            "inj_ccsn": args.ccsn_manifests,
            "real_glitch": args.glitch_manifests,
        },
        args.per_class,
        list(args.seeds),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    task_file = args.out.with_name("pilot_tasks.txt")
    task_file.write_text(
        "".join(
            f"{task['cls']} {task['index']} {task['seed']}\n"
            for task in payload["tasks"]
        )
    )
    for event_class, block in payload["classes"].items():
        print(f"{event_class:12s} n={block['n']:2d}  indices={block['indices']}")
    print(
        f"\n{payload['n_tasks']} BayesWave runs "
        f"({payload['n_events']} events x {len(payload['seeds'])} seeds)"
    )
    print(f"wrote {args.out}\nwrote {task_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
