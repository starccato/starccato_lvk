"""Paired one-vs-two-detector control for the network-degradation diagnosis.

The production campaigns (``real_noise_event.py``) normalise each injection to a
target *network* SNR over whichever detectors are analysed. A one-detector
campaign therefore puts the whole target into one interferometer while a
two-detector campaign splits it, so "1-det vs 2-det" across those campaigns is
not the same strain with a detector added. This control removes that confound.

Each event injects ONE coherent CCSN into H1+L1 exactly once and then analyses
those identical bundles three ways -- H1 only, L1 only, H1+L1 -- with no
renormalisation between analyses. Arms:

    noise    : the real-noise bundles, no injection (per-configuration LnO
               calibration + the single-detector noise-overfit check)
    sig_off  : held-out real CCSN waveform the VAE never trained on
    sig_on   : VAE prior draw (on-manifold), same noise/sky/target as sig_off

sig_on vs sig_off is the model-mismatch test: if the paired one-vs-two-detector
loss disappears on-manifold it is mismatch, if it survives it is not.

The event index, target SNR, sky and held-out waveform reproduce the production
draws for the same ``--index``/``--blip-ifo``, so control events map one-to-one
onto campaign events.

    uv run python studies/paired_detector_control.py --index 0 --blip-ifo H1 \
        --outdir out_pdc --stage both
    uv run python studies/paired_detector_control.py --summarize --outdir out_pdc
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from starccato_lvk.analysis import run_bcr_posteriors
from starccato_lvk.analysis.multidet_data_prep import (
    prepare_multi_detector_data,
)
from starccato_lvk.acquisition.io.glitch_catalog import get_blip_trigger_time

from real_noise_io import build_bundle, inject_into_bundle
from snr_vs_odds_roc import SAMPLE_RATE
from snr_vs_odds_roc_coherent import _inject_coherent, per_detector_snr

from starccato_jax.data.training_data import TrainValData
from starccato_jax.waveforms import get_model

DETECTORS = ("H1", "L1")
ARMS = ("noise", "sig_off", "sig_on")
ARM_SEED_OFFSET = {"noise": 1, "sig_off": 2, "sig_on": 3}
SEGMENT_SECONDS = 4.0


def _log(index: int, msg: str) -> None:
    print(f"[pdc e{index}] {msg}", flush=True)


def _configurations() -> dict[str, list[str]]:
    """H1, L1, then H1+L1 -- the three analyses of one injected dataset."""
    single = {det: [det] for det in DETECTORS}
    return {**single, "_".join(DETECTORS): list(DETECTORS)}


def _campaign_draws(index: int, snr_grid) -> dict:
    """Reproduce ``real_noise_event.prep_index`` draws for this index.

    The draw ORDER (target, ra, dec, psi, validation index) must match the
    production runner exactly or the control events stop corresponding to the
    campaign events they are meant to explain.
    """
    rng = np.random.default_rng(1000 + index)
    lo, hi = float(min(snr_grid)), float(max(snr_grid))
    target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    sky = {
        "ra": float(rng.uniform(0.0, 2.0 * np.pi)),
        "dec": float(np.arcsin(rng.uniform(-1.0, 1.0))),
        "psi": float(rng.uniform(0.0, np.pi)),
        "t_c": 0.0,
    }
    ccsn_pool = np.asarray(TrainValData.load(source="ccsne", seed=0).val)
    ccsn_index = int(rng.integers(ccsn_pool.shape[0]))
    return {
        "target_snr": target,
        "sky": sky,
        "ccsn_index": ccsn_index,
        "raw_wf": ccsn_pool[ccsn_index],
    }


def _on_manifold_waveform(index: int) -> np.ndarray:
    """A VAE prior draw, fed through the SAME raw-waveform injection path as the
    held-out waveform so the two arms differ only in the source morphology."""
    import jax

    model = get_model("ccsne")
    draw = model.generate(rng=jax.random.PRNGKey(700_000 + index), n=1)
    return np.asarray(draw, dtype=np.float64).reshape(-1)


def prep_index(index, band, snr_grid, noise_offset, outdir, blip_ifo) -> dict:
    flow, fmax = band
    dt = 1.0 / SAMPLE_RATE
    n_seg = int(round(SEGMENT_SECONDS * SAMPLE_RATE))
    edir = outdir / f"e{index}"
    draws = _campaign_draws(index, snr_grid)
    blip_gps = get_blip_trigger_time(index, ifo=blip_ifo)
    noise_gps = blip_gps - noise_offset

    _log(index, f"fetching H1+L1 noise @ gps={noise_gps:.1f} ...")
    noise_b = {
        det: build_bundle(noise_gps, edir / "noise" / det, det)
        for det in DETECTORS
    }
    prep = prepare_multi_detector_data(
        list(DETECTORS), bundle_paths=noise_b, flow=flow, fmax=fmax
    )

    bundles = {
        "noise": {det: str(path.resolve()) for det, path in noise_b.items()}
    }
    snr = {"noise": {det: 0.0 for det in DETECTORS}}
    net_snr = {"noise": 0.0}
    for arm, wf in (
        ("sig_off", draws["raw_wf"]),
        ("sig_on", _on_manifold_waveform(index)),
    ):
        _log(index, f"injecting {arm} coherently (net SNR={draws['target_snr']:.1f}) ...")
        injected, net = _inject_coherent(
            prep,
            draws["target_snr"],
            n_seg,
            dt,
            flow,
            fmax,
            wf,
            sky=draws["sky"],
        )
        arm_b = {
            det: inject_into_bundle(
                noise_b[det], injected[det], edir / arm / f"{det}.hdf5"
            )
            for det in DETECTORS
        }
        bundles[arm] = {
            det: str(path.resolve()) for det, path in arm_b.items()
        }
        snr[arm] = per_detector_snr(prep, injected, n_seg, dt, flow, fmax)
        net_snr[arm] = float(net)

    manifest = {
        "index": index,
        "blip_ifo": blip_ifo,
        "detectors": list(DETECTORS),
        "band": [flow, fmax],
        "gps": {"blip": blip_gps, "noise": noise_gps},
        "sky": draws["sky"],
        "target_snr": draws["target_snr"],
        "ccsne_validation_index": draws["ccsn_index"],
        "snr_by_detector": snr,
        "net_snr": net_snr,
        "bundles": bundles,
    }
    edir.mkdir(parents=True, exist_ok=True)
    (edir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def analyse_manifest(manifest, outdir, sampler, arms=ARMS) -> None:
    index = manifest["index"]
    flow, fmax = manifest["band"]
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    configurations = _configurations()
    failures = []

    for arm in arms:
        # One seed per arm, shared across configurations: within an arm the ONLY
        # difference between the three runs is which detectors are analysed.
        rng_seed = 200_000 + 10 * int(index) + ARM_SEED_OFFSET[arm]
        for cfg_name, cfg_dets in configurations.items():
            out_json = results_dir / f"e{index}_{arm}_{cfg_name}.json"
            if out_json.exists():
                _log(index, f"skip existing {out_json.name}")
                continue
            bundle_paths = {
                det: manifest["bundles"][arm][det] for det in cfg_dets
            }
            started = time.perf_counter()
            _log(index, f"analysing arm={arm} config={cfg_name} ...")
            try:
                r = run_bcr_posteriors(
                    detectors=cfg_dets,
                    outdir=str(outdir / f"e{index}" / arm / cfg_name),
                    bundle_paths=bundle_paths,
                    extrinsic_params=manifest["sky"],
                    signal_model="ccsne",
                    glitch_model="blip",
                    flow=flow,
                    fmax=fmax,
                    rng_seed=rng_seed,
                    save_artifacts=False,
                    save_diagnostics=True,
                    lnz_method="morph",
                    **sampler,
                )
                gl = r.get("glitch", {})
                row = {
                    "index": index,
                    "arm": arm,
                    "config": cfg_name,
                    "detectors": cfg_dets,
                    "blip_ifo": manifest["blip_ifo"],
                    "target_snr": manifest["target_snr"],
                    "net_snr": manifest["net_snr"][arm],
                    "snr_by_detector": manifest["snr_by_detector"][arm],
                    "snr_analysed": float(
                        np.sqrt(
                            sum(
                                manifest["snr_by_detector"][arm][det] ** 2
                                for det in cfg_dets
                            )
                        )
                    ),
                    "rng_seed": rng_seed,
                    "logZ_signal": float(r["signal"]["logZ"]),
                    "logZ_signal_err": float(r["signal"]["logZ_err"]),
                    "logZ_glitch_by_detector": {
                        k: float(v) for k, v in gl.items()
                    },
                    "logZ_noise_by_detector": r.get("noise", {}),
                    "log_odds": float(r["bcr_log"]),
                    "map_log_density": r.get("map_initialization", {}).get(
                        "log_density"
                    ),
                    "evidence_status": r.get("evidence_status", {}),
                    "evidence_failures": int(r.get("evidence_failures", 0)),
                    "evidence_fallbacks": int(r.get("evidence_fallbacks", 0)),
                    "nuts_diagnostics": r.get("nuts_diagnostics", {}),
                    "runtime_seconds": time.perf_counter() - started,
                }
                out_json.write_text(
                    json.dumps(row, indent=2, sort_keys=True, default=str)
                    + "\n"
                )
                _log(
                    index,
                    f"{arm:>7s}/{cfg_name:<5s} snr={row['snr_analysed']:6.1f} "
                    f"logZ_s={row['logZ_signal']:9.1f} lnO={row['log_odds']:9.1f}",
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{arm}/{cfg_name}")
                (results_dir / f"e{index}_{arm}_{cfg_name}.failure.json").write_text(
                    json.dumps(
                        {
                            "index": index,
                            "arm": arm,
                            "config": cfg_name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                _log(index, f"{arm}/{cfg_name} FAILED: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("Analysis failed for: " + ", ".join(failures))


def _median(values):
    return float(np.median(values)) if len(values) else float("nan")


def summarize(outdir: Path) -> dict:
    """Paired differences: every quantity compared WITHIN an event and arm."""
    rows = [
        json.loads(p.read_text())
        for p in sorted((outdir / "results").glob("e*_*.json"))
        if not p.name.endswith(".failure.json")
    ]
    if not rows:
        raise SystemExit(f"No result rows under {outdir / 'results'}")
    by_key = {(r["index"], r["arm"], r["config"]): r for r in rows}
    network = "_".join(DETECTORS)

    summary = {"n_rows": len(rows), "arms": {}}
    for arm in ARMS:
        indices = sorted(
            {i for (i, a, _c) in by_key if a == arm}
        )
        paired = {}
        for single in DETECTORS:
            deltas_lno, deltas_lnz, snr_single = [], [], []
            for i in indices:
                one = by_key.get((i, arm, single))
                two = by_key.get((i, arm, network))
                if one is None or two is None:
                    continue
                deltas_lno.append(two["log_odds"] - one["log_odds"])
                deltas_lnz.append(two["logZ_signal"] - one["logZ_signal"])
                snr_single.append(one["snr_analysed"])
            paired[single] = {
                "n_pairs": len(deltas_lno),
                "median_delta_log_odds": _median(deltas_lno),
                "median_delta_logZ_signal": _median(deltas_lnz),
                "median_single_detector_snr": _median(snr_single),
            }
        # Per-configuration LnO=0 calibration comes from the noise arm.
        summary["arms"][arm] = {
            "network_minus_single": paired,
            "median_log_odds_by_config": {
                cfg: _median(
                    [
                        r["log_odds"]
                        for r in rows
                        if r["arm"] == arm and r["config"] == cfg
                    ]
                )
                for cfg in _configurations()
            },
        }

    noise_by_cfg = {
        cfg: sorted(
            r["log_odds"] for r in rows if r["arm"] == "noise" and r["config"] == cfg
        )
        for cfg in _configurations()
    }
    summary["noise_lno_95th_percentile"] = {
        cfg: (float(np.percentile(v, 95)) if v else float("nan"))
        for cfg, v in noise_by_cfg.items()
    }
    (outdir / "control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def self_check(results_json: Path, snr_grid) -> None:
    """Verify the control still reproduces the production draw sequence.

    Pairing is the whole point of this study: if ``_campaign_draws`` drifts out
    of step with ``real_noise_event.prep_index`` the control events silently
    stop corresponding to the campaign events they explain. The campaign records
    the ACHIEVED injection SNR, which the injection normalises to the drawn
    target, so the drawn target must match it for every event.
    """
    rows = json.loads(results_json.read_text())
    lo, hi = float(min(snr_grid)), float(max(snr_grid))
    checked = 0
    for row in rows:
        if row.get("cls") != "inj_ccsn":
            continue
        rng = np.random.default_rng(1000 + row["index"])
        target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        achieved = row["snr"]
        assert abs(target - achieved) / achieved < 0.02, (
            f"draw sequence drifted at index {row['index']}: "
            f"target {target:.4f} vs achieved {achieved:.4f}"
        )
        checked += 1
    assert checked, f"No inj_ccsn rows in {results_json}"
    assert set(_configurations()) == {"H1", "L1", "H1_L1"}
    print(f"self-check OK: {checked} campaign injections reproduce their target SNR")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int)
    p.add_argument("--blip-ifo", default="H1", choices=["H1", "L1"])
    p.add_argument("--stage", choices=["prep", "analysis", "both"], default="both")
    p.add_argument("--outdir", type=Path, default=Path("out_pdc"))
    p.add_argument("--snr-grid", type=float, nargs="+", default=[10, 20, 40])
    p.add_argument("--noise-offset", type=float, default=100.0)
    p.add_argument("--flow", type=float, default=300.0)
    p.add_argument("--fmax", type=float, default=800.0)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-chains", type=int, default=2)
    p.add_argument(
        "--arm",
        choices=ARMS,
        help="Analyse only one arm (all three detector configurations). "
        "Use this for bounded SLURM array tasks.",
    )
    p.add_argument("--summarize", action="store_true")
    p.add_argument(
        "--self-check",
        type=Path,
        help="Campaign results.json to verify draw-sequence reproduction against.",
    )
    args = p.parse_args()

    if args.self_check:
        self_check(args.self_check, args.snr_grid)
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        print(json.dumps(summarize(args.outdir), indent=2, sort_keys=True))
        return
    if args.index is None:
        p.error("--index is required unless --summarize is given")
    if args.num_chains < 2:
        p.error("NUTS+MorphLnZ analyses require --num-chains >= 2")
    if args.flow >= args.fmax:
        p.error("--flow must be lower than --fmax")

    manifest_path = args.outdir / f"e{args.index}" / "manifest.json"
    if args.stage in ("prep", "both"):
        if manifest_path.exists():
            _log(args.index, f"prep skipped; manifest exists: {manifest_path}")
        else:
            prep_index(
                args.index,
                (args.flow, args.fmax),
                args.snr_grid,
                args.noise_offset,
                args.outdir,
                args.blip_ifo,
            )
            _log(args.index, f"prep done -> {manifest_path}")
    if args.stage in ("analysis", "both"):
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing manifest {manifest_path}; run the prep stage first."
            )
        analyse_manifest(
            json.loads(manifest_path.read_text()),
            args.outdir,
            sampler=dict(
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=args.num_chains,
                noise_scale_marginal=True,
                nested_num_live_points=300,
                nested_max_samples=6000,
            ),
            arms=(args.arm,) if args.arm else ARMS,
        )
    _log(args.index, "all stages complete")


if __name__ == "__main__":
    main()
