"""
Run a small simulation study for LVK Starccato:

- noise-only
- noise + injected signal
- blip glitch (one detector) + noise (other detector)

This uses the high-level workflow utilities so it reuses the same
data acquisition and analysis paths as the CLI/SLURM flows.

Usage:
    python lvk/studies/simulation_study.py \
        --index 0 \
        --config lvk/slurm/configs/analysis.yaml \
        --distance 1.0 \
        --stage both \
        [--force]

Results are written under the output_root configured in the YAML
(default: lvk/slurm/out/<scenario>/<gps>/...).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from starccato_lvk.workflows.run_event import (
    CONFIG_DEFAULT,
    TRIGGERS_CSV_DEFAULT,
    load_analysis_config,
    read_event_from_triggers_csv,
    run_event_workflow,
)
from starccato_lvk.acquisition.io.strain_loader import _detector_cat3_online


def _filter_detectors_for_gps(detectors: list[str], gps: float) -> list[str]:
    # Mirror analysis window in strain_loader.load_analysis_chunk_and_psd
    analysis_start = gps - 1
    gps_start = analysis_start - 65
    gps_end = gps + 1
    available: list[str] = []
    for det in detectors:
        ok = _detector_cat3_online(det, gps_start, gps_end)
        if ok:
            available.append(det)
    return available


def _run_one(scenario: str, index: int, config_path: Path, triggers_csv: Path, *, force: bool, distance: float, stage: str) -> None:
    cfg = load_analysis_config(config_path)
    gps = read_event_from_triggers_csv(scenario, index, csv_path=triggers_csv)
    # Pre-filter detectors to avoid GWOSC fetch failures for offlined instruments
    dets = _filter_detectors_for_gps(cfg.detectors, gps)
    if not dets:
        # Soft fallback: prefer L1 if present in config; otherwise keep original list and let it error
        if "L1" in [d.upper() for d in cfg.detectors]:
            dets = ["L1"]
        else:
            dets = [cfg.detectors[0]]
        print(f"[study] No detectors fully CAT3-online; falling back to {dets} for {gps}.")
    cfg.detectors = [d.upper() for d in dets]
    print(f"[study] Scenario={scenario} index={index} gps={gps} stage={stage} force={force} detectors={cfg.detectors}")
    res = run_event_workflow(
        cfg,
        scenario,
        gps,
        index,
        force=force,
        injection_distance=distance,
        stage=stage,
    )
    if res is None:
        print(f"[study] {scenario}: summary already exists; skipped analysis.")
    else:
        print(f"[study] {scenario}: done. BCR={res.get('bcr')} logZ_signal={res.get('logZ_signal')} logZ_glitch={res.get('logZ_glitch')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LVK Starccato simulation study across three scenarios.")
    parser.add_argument("--index", type=int, default=0, help="Row index into triggers CSV for GPS selection.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_DEFAULT),
        help="Analysis config YAML (controls detectors, priors, output_root, etc.).",
    )
    parser.add_argument(
        "--triggers-csv",
        type=Path,
        default=Path(TRIGGERS_CSV_DEFAULT),
        help="CSV with paired triggers (noise_trigger, blip_trigger).",
    )
    parser.add_argument("--distance", type=float, default=1.0, help="Injection distance scale for noise_inj.")
    parser.add_argument(
        "--stage",
        choices=["prep", "analysis", "both"],
        default="both",
        help="Which stages to run (bundle prep, analysis, or both).",
    )
    parser.add_argument("--force", action="store_true", help="Re-run analysis even if summary exists.")

    args = parser.parse_args()

    # Resolve to absolute for clarity in logs
    config_path = args.config.resolve()
    triggers_csv = args.triggers_csv.resolve()
    print(f"[study] Using config: {config_path}")
    print(f"[study] Using triggers CSV: {triggers_csv}")

    for scenario in ("noise", "noise_inj", "blip"):
        _run_one(
            scenario,
            args.index,
            config_path,
            triggers_csv,
            force=args.force,
            distance=args.distance,
            stage=args.stage,
        )


if __name__ == "__main__":
    main()
