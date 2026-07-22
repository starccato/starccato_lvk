"""Find events whose coherent signal fit looks failed, and refit only those.

The v043 H1+L1 campaign showed ~13% of injections returning ``logZ_signal ~ 0``
with clean sampler diagnostics -- loud events (network SNR > 100) where the
multistart optimiser never located the coherent basin at all, so NUTS sampled a
spurious one and morphZ faithfully integrated it. Nothing in the existing status
fields catches this: the wrong answer is SMALL, while the morphZ cross-check
fires on LARGE evidences.

    # 1. list what looks broken (read-only)
    uv run python studies/refit_suspect_signal_fits.py scan <campaign_dir>

    # 2. refit one flagged event (SLURM array task; idempotent)
    uv run python studies/refit_suspect_signal_fits.py refit <campaign_dir> \
        --index 728 --cls inj_ccsn --lnz-method nested

Refits are written alongside the originals as ``e{i}_{cls}.refit.json`` and the
original row is never overwritten, so a refit campaign can be compared against
its parent before anything is adopted.

IMPORTANT -- what this is and is not. Selecting events by "the signal model lost
badly" and then fitting the signal model harder is an ASYMMETRIC escalation: it
can only ever move an event towards the signal hypothesis. That is legitimate
for removing an optimiser artifact, and illegitimate as a way to improve a
ranking statistic. So:

  * Report refit outcomes for EVERY flagged event, including the ones that come
    back unchanged -- those are the genuinely glitch-like events, and dropping
    them is what would turn this into cherry-picking.
  * ``--all-classes`` applies the identical rule to noise and real_glitch (where
    a recovered signal evidence makes the result WORSE). Run it. A fix that only
    ever helps the signal class has not been tested, it has been assumed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from starccato_lvk.analysis import run_bcr_posteriors
from starccato_lvk.analysis.main import _signal_fit_suspect

DEFAULT_MARGIN = 20.0


def _rows(campaign: Path) -> list[dict]:
    results = campaign / "results"
    out = []
    for path in sorted(results.glob("e*_*.json")):
        if path.name.endswith((".refit.json", "_baseline.json")):
            continue
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"WARNING: unreadable result {path}", flush=True)
    return out


def _is_suspect(row: dict, margin: float) -> bool:
    recorded = row.get("signal_fit_suspect")
    if recorded is not None:
        return bool(recorded)
    # Rows written before the flag existed: recompute from the stored evidences.
    return bool(
        _signal_fit_suspect(
            row.get("logZ_signal", float("nan")),
            row.get("logZ_glitch_by_detector", {}) or {},
            margin,
        )
    )


def _fitting_factor(row: dict, campaign: Path, cache: dict) -> float:
    """logZ_signal as a fraction of the ideal rho^2/2 for the INJECTED SNR.

    Only defined for injections, where the true amplitude is known from the
    manifest. This is a far sharper selector than the evidence-only flag: a
    coherent fit that found the signal lands at 0.4-0.7 even with an imperfect
    waveform, whereas a failed search lands near 0 regardless of how loud the
    injection was. Using injection truth is legitimate here because this is a
    validation study of the optimiser, not a detection statistic.
    """
    # Only inj_ccsn has a coherent-signal amplitude behind its recorded SNR. For
    # real_glitch that number is a single-detector catalogue trigger SNR, so a
    # low "signal fitting factor" is the CORRECT answer there, not a failure --
    # scoring those events this way would flag the whole class. They fall back
    # to the evidence-margin rule, which is the honest control.
    if row.get("cls") != "inj_ccsn":
        return float("nan")
    index = row["index"]
    if index not in cache:
        try:
            cache[index] = json.loads(
                (campaign / f"e{index}" / "manifest.json").read_text()
            )
        except OSError:
            cache[index] = None
    manifest = cache[index]
    if manifest is None:
        return float("nan")
    rho = (manifest.get("snr") or {}).get(row["cls"])
    if not rho or not np.isfinite(rho) or rho <= 0:
        return float("nan")
    return float(row["logZ_signal"]) / (rho**2 / 2.0)


def scan(
    campaign: Path,
    margin: float,
    classes: tuple[str, ...],
    ff_threshold: float | None = None,
    write_tasks: Path | None = None,
) -> list[dict]:
    rows = [r for r in _rows(campaign) if r.get("cls") in classes]
    cache: dict = {}

    def suspect(row: dict) -> bool:
        if ff_threshold is not None:
            ff = _fitting_factor(row, campaign, cache)
            if np.isfinite(ff):
                return bool(ff < ff_threshold)
        return _is_suspect(row, margin)

    flagged = [r for r in rows if suspect(r)]
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["cls"], []).append(row)
    print(f"campaign : {campaign}")
    criterion = (
        f"fitting factor logZ_s/(rho^2/2) < {ff_threshold} (injected SNR known)"
        if ff_threshold is not None
        else f"evidence margin {margin} nats (no injection truth used)"
    )
    print(f"criterion: {criterion}\n")
    print(f"{'class':>13s} {'n':>6s} {'flagged':>8s} {'%':>6s} {'median lnO (flagged)':>22s}")
    for cls, group in sorted(by_class.items()):
        hits = [r for r in group if suspect(r)]
        med = (
            np.median([r["log_odds"] for r in hits]) if hits else float("nan")
        )
        print(
            f"{cls:>13s} {len(group):6d} {len(hits):8d} "
            f"{100 * len(hits) / max(len(group), 1):5.1f}% {med:22.1f}"
        )
    print(f"\ntotal flagged: {len(flagged)}")
    if write_tasks is not None:
        lines = [
            f"{r['index']}\t{r['cls']}"
            for r in sorted(flagged, key=lambda r: (r["cls"], r["index"]))
        ]
        Path(write_tasks).write_text("\n".join(lines) + "\n")
        print(
            f"\nwrote {len(lines)} tasks -> {write_tasks}\n"
            f"submit with --array=0-{len(lines) - 1}"
        )
    if flagged:
        worst = sorted(flagged, key=lambda r: r["log_odds"])[:10]
        print("\nworst flagged events:")
        print(f"  {'index':>6s} {'cls':>12s} {'lnO':>10s} {'logZ_s':>9s} {'max logZ_g':>11s}")
        for row in worst:
            gl = row.get("logZ_glitch_by_detector", {}) or {}
            mx = max(gl.values()) if gl else float("nan")
            print(
                f"  {row['index']:6d} {row['cls']:>12s} {row['log_odds']:10.1f} "
                f"{row['logZ_signal']:9.1f} {mx:11.1f}"
            )
        print("\nindices:")
        print(",".join(str(r["index"]) for r in sorted(flagged, key=lambda r: r["index"])))
    return flagged


def _verify_rebuilt_manifest(original: dict, rebuilt: dict, cls: str) -> None:
    """Refuse to refit unless the re-prepped event reproduces the original.

    Campaign bundles are pruned once an event completes, so a refit must
    regenerate them. That is only meaningful if the regenerated strain is the
    strain the original analysis actually saw -- otherwise a "recovered" event
    could just be a different, easier injection. Preparation is seeded per
    index, so every one of these quantities must match bit-for-bit; any
    difference means the injection path changed since the campaign ran and the
    comparison is void.
    """
    checks = {
        "gps": (original.get("gps"), rebuilt.get("gps")),
        "sky": (original.get("sky"), rebuilt.get("sky")),
        "snr": (
            (original.get("snr") or {}).get(cls),
            (rebuilt.get("snr") or {}).get(cls),
        ),
        "snr_by_detector": (
            (original.get("snr_by_detector") or {}).get(cls),
            (rebuilt.get("snr_by_detector") or {}).get(cls),
        ),
        "injection.target_snr": (
            (original.get("injection") or {}).get("target_snr"),
            (rebuilt.get("injection") or {}).get("target_snr"),
        ),
        "injection.snr_normalisation": (
            (original.get("injection") or {}).get("snr_normalisation"),
            (rebuilt.get("injection") or {}).get("snr_normalisation"),
        ),
        "validation_indices": (
            (original.get("injection") or {}).get("validation_indices"),
            (rebuilt.get("injection") or {}).get("validation_indices"),
        ),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise RuntimeError(
            "Re-prepped event does not reproduce the original campaign event; "
            "refusing to refit. Differences (original vs rebuilt): "
            + json.dumps(bad, indent=2, default=str)
        )


def refit(
    campaign: Path,
    index: int,
    cls: str,
    *,
    lnz_method: str,
    bundles_from: Path | None = None,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    map_num_starts: int,
    nested_num_live_points: int,
    nested_max_samples: int,
) -> dict:
    manifest = json.loads((campaign / f"e{index}" / "manifest.json").read_text())
    original_path = campaign / "results" / f"e{index}_{cls}.json"
    original = json.loads(original_path.read_text())
    out_path = campaign / "results" / f"e{index}_{cls}.refit.json"
    if out_path.exists():
        print(f"refit exists, skipping: {out_path}", flush=True)
        return json.loads(out_path.read_text())
    if bundles_from is not None:
        rebuilt = json.loads(
            (Path(bundles_from) / f"e{index}" / "manifest.json").read_text()
        )
        _verify_rebuilt_manifest(manifest, rebuilt, cls)
        # Keep the ORIGINAL manifest for reporting; take only the bundle paths
        # from the rebuild, which the guard above just proved equivalent.
        manifest = {**manifest, "bundles": rebuilt["bundles"]}

    flow, fmax = manifest["band"]
    started = time.perf_counter()
    result = run_bcr_posteriors(
        detectors=manifest["detectors"],
        outdir=str(campaign / f"e{index}" / cls / "refit"),
        bundle_paths=manifest["bundles"][cls],
        extrinsic_params=manifest.get("sky"),
        signal_model="ccsne",
        glitch_model="blip",
        flow=flow,
        fmax=fmax,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        map_num_starts=map_num_starts,
        # A different seed from the original run: an identical seed would
        # reproduce the same failed search and prove nothing.
        rng_seed=int(original.get("rng_seed", 0)) + 777_000,
        save_artifacts=False,
        save_diagnostics=True,
        lnz_method=lnz_method,
        noise_scale_marginal=True,
        nested_num_live_points=nested_num_live_points,
        nested_max_samples=nested_max_samples,
    )
    gl = result.get("glitch", {})
    row = {
        "index": index,
        "cls": cls,
        "detectors": manifest["detectors"],
        "refit_of": str(original_path),
        "refit_method": lnz_method,
        "snr": manifest["snr"][cls],
        "snr_by_detector": (manifest.get("snr_by_detector") or {}).get(cls),
        "logZ_signal": float(result["signal"]["logZ"]),
        "logZ_glitch_by_detector": {k: float(v) for k, v in gl.items()},
        "log_odds": float(result["bcr_log"]),
        "signal_fit_suspect": result.get("signal_fit_suspect"),
        "evidence_status": result.get("evidence_status", {}),
        "nuts_diagnostics": result.get("nuts_diagnostics", {}),
        "map_initialization": result.get("map_initialization", {}),
        "original_logZ_signal": original.get("logZ_signal"),
        "original_log_odds": original.get("log_odds"),
        "delta_log_odds": float(result["bcr_log"]) - float(original["log_odds"]),
        "runtime_seconds": time.perf_counter() - started,
    }
    out_path.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
    print(
        f"[e{index} {cls}] refit ({lnz_method}): "
        f"lnO {original['log_odds']:.1f} -> {row['log_odds']:.1f} "
        f"(delta {row['delta_log_odds']:+.1f}), "
        f"logZ_s {original['logZ_signal']:.1f} -> {row['logZ_signal']:.1f}",
        flush=True,
    )
    return row


def report(campaign: Path) -> None:
    """Summarise every completed refit -- recovered AND unchanged alike."""
    refits = [
        json.loads(p.read_text())
        for p in sorted((campaign / "results").glob("e*_*.refit.json"))
    ]
    if not refits:
        raise SystemExit(f"no refits under {campaign / 'results'}")
    print(f"refits completed: {len(refits)}\n")
    print(f"{'class':>13s} {'n':>5s} {'recovered':>10s} {'unchanged':>10s} {'median dlnO':>12s}")
    by_class: dict[str, list[dict]] = {}
    for row in refits:
        by_class.setdefault(row["cls"], []).append(row)
    for cls, group in sorted(by_class.items()):
        delta = np.array([r["delta_log_odds"] for r in group])
        print(
            f"{cls:>13s} {len(group):5d} {(delta > 10).sum():10d} "
            f"{(np.abs(delta) <= 10).sum():10d} {np.median(delta):12.1f}"
        )
    print(
        "\nA fix that recovers signal events must leave noise/real_glitch "
        "events alone; large positive deltas there mean the refit is "
        "manufacturing signal evidence, not recovering it."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("campaign", type=Path, help="Campaign directory holding e*/ and results/")
    common.add_argument("--margin", type=float, default=DEFAULT_MARGIN)

    s = sub.add_parser("scan", parents=[common], help="List flagged events (read-only)")
    s.add_argument("--all-classes", action="store_true",
                   help="Scan noise and real_glitch too (the control classes).")
    s.add_argument("--write-tasks", type=Path,
                   help="Write flagged events as 'index<TAB>class' lines for a "
                        "SLURM array (see slurm/refit_suspect.sh).")
    s.add_argument("--ff-threshold", type=float, default=0.05,
                   help="Flag when logZ_signal/(rho^2/2) falls below this, using "
                        "the injected SNR from the manifest. Sharper than the "
                        "evidence-margin rule; set to a negative value to "
                        "disable and use the margin rule instead.")

    r = sub.add_parser("refit", parents=[common], help="Refit one flagged event")
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--cls", default="inj_ccsn")
    r.add_argument("--bundles-from", type=Path,
                   help="Directory holding re-prepped bundles for this event "
                        "(campaign bundles are pruned after completion). The "
                        "rebuild must reproduce the original manifest exactly.")
    r.add_argument("--lnz-method", choices=["morph", "nested"], default="nested",
                   help="Nested sampling explores the whole prior rather than "
                        "starting from a MAP point, so it does not inherit the "
                        "initialisation failure being investigated.")
    r.add_argument("--num-warmup", type=int, default=500)
    r.add_argument("--num-samples", type=int, default=1000)
    r.add_argument("--num-chains", type=int, default=2)
    r.add_argument("--map-num-starts", type=int, default=512)
    r.add_argument("--nested-num-live-points", type=int, default=1000)
    r.add_argument("--nested-max-samples", type=int, default=40000)

    sub.add_parser("report", parents=[common], help="Summarise completed refits")

    args = p.parse_args()
    if args.command == "scan":
        classes = (
            ("inj_ccsn", "noise", "real_glitch")
            if args.all_classes
            else ("inj_ccsn",)
        )
        scan(
            args.campaign,
            args.margin,
            classes,
            ff_threshold=(
                args.ff_threshold if args.ff_threshold >= 0 else None
            ),
            write_tasks=args.write_tasks,
        )
    elif args.command == "refit":
        refit(
            args.campaign,
            args.index,
            args.cls,
            lnz_method=args.lnz_method,
            bundles_from=args.bundles_from,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            map_num_starts=args.map_num_starts,
            nested_num_live_points=args.nested_num_live_points,
            nested_max_samples=args.nested_max_samples,
        )
    else:
        report(args.campaign)


if __name__ == "__main__":
    main()
