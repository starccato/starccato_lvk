"""Merge per-event result JSONs into a single results.json for easy scp,
and report which (index, class) pairs are missing so failures can be rerun.

Run on the cluster against one external campaign cohort, for example:

    uv run python studies/collect_results.py /fred/.../CAMPAIGN/rn_L1 \
        --expected-start 0 --expected-stop 249

Then copy down that cohort's ``results.json`` and ``collection_summary.json``.
"""

import argparse
import glob
import json
from pathlib import Path

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")
PRODUCTION_CLASSES = ("noise", "inj_ccsn", "real_glitch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument(
        "--expected-start",
        type=int,
        help="First submitted array index (inclusive).",
    )
    parser.add_argument(
        "--expected-stop",
        type=int,
        help="Last submitted array index (inclusive).",
    )
    parser.add_argument(
        "--expected-classes",
        nargs="+",
        choices=CLASSES,
        default=list(PRODUCTION_CLASSES),
        help="Classes expected for every submitted index.",
    )
    args = parser.parse_args()
    if (args.expected_start is None) != (args.expected_stop is None):
        parser.error(
            "--expected-start and --expected-stop must be supplied together"
        )
    if (
        args.expected_start is not None
        and args.expected_stop < args.expected_start
    ):
        parser.error(
            "--expected-stop must be greater than or equal to --expected-start"
        )

    files = sorted(
        f
        for f in glob.glob(str(args.outdir / "results" / "e*_*.json"))
        # newSNR baseline rows live beside the odds rows; they are collected
        # separately by chisq_baseline.py --aggregate, not here
        if not f.endswith("_baseline.json")
    )
    if not files:
        raise FileNotFoundError(
            f"No per-event JSONs under {args.outdir / 'results'}; "
            "refusing to overwrite an existing collected results.json"
        )
    rows = [json.loads(Path(f).read_text()) for f in files]
    keys = [(int(row["index"]), row["cls"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(
            f"Duplicate (index, class) result files under {args.outdir}"
        )
    unknown = sorted({row["cls"] for row in rows} - set(CLASSES))
    if unknown:
        raise ValueError(
            f"Unexpected result classes under {args.outdir}: {unknown}"
        )
    campaign_ids = sorted(
        {row.get("campaign_id") for row in rows if row.get("campaign_id")}
    )
    if len(campaign_ids) > 1:
        raise RuntimeError(
            f"Collected results contain multiple campaign IDs: {campaign_ids}"
        )

    output = args.outdir / "results.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    done = set(keys)
    if args.expected_start is not None:
        indices = range(args.expected_start, args.expected_stop + 1)
    else:
        indices = sorted({index for index, _ in done})
    missing = [
        (index, cls)
        for index in indices
        for cls in args.expected_classes
        if (index, cls) not in done
    ]
    failure_files = sorted(
        glob.glob(str(args.outdir / "failures" / "e*_*.json"))
    )
    failures = [json.loads(Path(path).read_text()) for path in failure_files]

    print(f"{len(rows)} rows -> {output}")
    for cls in args.expected_classes:
        print(f"  {cls:>11s}: {sum(row['cls'] == cls for row in rows)}")
    if args.expected_start is None:
        print(
            "Expected range not supplied; entirely absent array indices cannot be detected."
        )
    if missing:
        print(f"{len(missing)} missing (index, cls) pairs:")
        print(" ".join(f"e{index}:{cls}" for index, cls in missing))
        print(
            "rerun indices: "
            + ",".join(
                str(index) for index in sorted({index for index, _ in missing})
            )
        )
    if failures:
        print(f"{len(failures)} recorded analysis exceptions:")
        for failure in failures:
            print(
                f"  e{failure['index']}:{failure['cls']} "
                f"{failure['error_type']}: {failure['error']}"
            )

    collection = {
        "campaign_ids": campaign_ids,
        "n_rows": len(rows),
        "expected_classes": args.expected_classes,
        "expected_start": args.expected_start,
        "expected_stop": args.expected_stop,
        "counts": {
            cls: sum(row["cls"] == cls for row in rows)
            for cls in args.expected_classes
        },
        "missing": [{"index": index, "cls": cls} for index, cls in missing],
        "n_failures": len(failures),
    }
    (args.outdir / "collection_summary.json").write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
