"""Single-event runner for the production real-noise SLURM campaign.

One analysis task evaluates one event class at one blip-catalogue index. The
production classes are noise, coherent injected CCSN, and a real catalogue
blip. The older injected-blip class remains available for sensitivity studies.

Two stages (matching the existing slurm/ design):
  --stage prep     : build bundles from the local strain mirror (or GWOSC when
                     no mirror is available), build the injected bundles, and
                     write a per-index manifest.json.
  --stage analysis : read the manifest and run the BCR for each class (compute node),
                     writing results/e{index}_{cls}.json.
  --stage both     : do both in one process (local / internet-capable node).

Aggregate the per-event JSONs with ``real_noise_aggregate.py``.

    uv run python studies/real_noise_event.py --index 0 --detectors L1 --stage both --outdir out_rn
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from starccato_lvk.analysis import run_bcr_posteriors
from starccato_lvk.analysis.multidet_data_prep import (
    prepare_multi_detector_data,
    whitened_band_power,
)
from starccato_lvk.acquisition.io.glitch_catalog import (
    get_blip_trigger_time,
    load_blip_glitch_catalog,
)
from starccato_lvk.acquisition.io.strain_loader import (
    StrainDataUnavailable,
    StrainFetchFailed,
)

from chisq_baseline import run_index as run_baseline_index
from real_noise_io import build_bundle, inject_into_bundle
from snr_vs_odds_roc import SAMPLE_RATE
from snr_vs_odds_roc_coherent import (
    _inject_coherent,
    _inject_single_glitch,
    per_detector_snr,
)

from starccato_jax.data.training_data import TrainValData
from starccato_jax.waveforms import get_model

class CleanSegmentRejected(RuntimeError):
    """Raised when a noise segment fails the whitening gate at preparation."""


CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")
PRODUCTION_CLASSES = ("noise", "inj_ccsn", "real_glitch")
# v3: injections are plus-polarized (h_x = 0, 2D-axisymmetric source) and the
# manifest records snr_by_detector. A v2 manifest was built with the old
# h_+ = h_x convention, so it MUST NOT be analysed with the current recovery
# model -- the version check below is what enforces that.
MANIFEST_SCHEMA_VERSION = 3
RESULT_SCHEMA_VERSION = 3
CLASS_SEED_OFFSET = {
    "noise": 1,
    "inj_ccsn": 2,
    "inj_glitch": 3,
    "real_glitch": 4,
}


def _json_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_provenance(root: Path) -> dict:
    commit = _git_commit(root)
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": commit, "dirty": None, "tracked_diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "tracked_diff_sha256": (
            hashlib.sha256(diff).hexdigest() if diff else None
        ),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_provenance(model_name: str) -> dict:
    model = get_model(model_name)
    model_data = getattr(model, "_data", None)
    return {
        "model_name": model_name,
        "model_dir": str(getattr(model, "model_dir", "")),
        "artifact_sha256": getattr(model_data, "artifact_sha256", None),
        "artifact_library_version": getattr(
            model_data, "library_version", None
        ),
        "artifact_metadata": getattr(model_data, "artifact_metadata", None),
    }


def _runtime_provenance() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "packages": {
            name: _package_version(name)
            for name in (
                "starccato-jax",
                "starccato-lvk",
                "morphZ",
                "jax",
                "numpyro",
            )
        },
        "git": {"starccato_lvk": _git_provenance(repo_root)},
        "models": {
            name: _model_provenance(name) for name in ("ccsne", "blip")
        },
    }


def _log(index: int, msg: str) -> None:
    print(f"[e{index}] {msg}", flush=True)


def _record_unavailable_trigger(
    *,
    index: int,
    outdir: Path,
    campaign_id: str,
    detectors: list[str],
    blip_ifo: str,
    error: StrainFetchFailed,
) -> Path:
    """Persist an auditable catalogue flag for a failed strain acquisition.

    No manifest is created because no analysis inputs exist.  The separate
    record lets the submitter distinguish a flagged trigger from work that has
    not yet run, without pretending the trigger produced a result.
    """
    path = outdir / f"e{index}" / "rejected.json"
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "status": "rejected",
            "reason": (
                "strain_data_unavailable"
                if isinstance(error, StrainDataUnavailable)
                else "strain_fetch_failed"
            ),
            "index": index,
            "campaign_id": campaign_id,
            "detectors": detectors,
            "blip_ifo": blip_ifo,
            "error_type": type(error).__name__,
            "error": str(error),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def prep_index(
    index,
    detectors,
    gdet,
    band,
    snr_grid,
    noise_offset,
    outdir,
    blip_ifo="L1",
    classes: Iterable[str] = PRODUCTION_CLASSES,
    campaign_id: str | None = None,
    snr_reference_det: str | None = None,
    require_clean_noise: bool = False,
    max_mean_whitened_power: float = 10.0,
) -> dict:
    """Build only the requested event classes and write a versioned manifest."""
    classes = tuple(dict.fromkeys(classes))
    unknown = sorted(set(classes) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown preparation classes: {unknown}")
    if not classes:
        raise ValueError("At least one preparation class is required.")
    flow, fmax = band
    dt = 1.0 / SAMPLE_RATE
    n_seg = int(round(4.0 * SAMPLE_RATE))
    edir = outdir / f"e{index}"
    _log(
        index,
        f"prep start: detectors={detectors} blip_ifo={blip_ifo} band=[{flow},{fmax}]",
    )
    cat = load_blip_glitch_catalog(ifo=blip_ifo)
    blip_gps = get_blip_trigger_time(index, ifo=blip_ifo)
    noise_gps = blip_gps - noise_offset
    cat_snr = float(cat.iloc[index]["snr"])
    rng = np.random.default_rng(1000 + index)  # reproducible per-event draws
    # Continuous injected SNR (log-uniform over the grid's span).
    lo, hi = float(min(snr_grid)), float(max(snr_grid))
    target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    # isotropic targeted sky (inject AND recover here -- known-direction analysis);
    # sampling the sky is slow, non-converging under HMC, and lets glitches find
    # blind-spot skies under nested -- so we fix a matched per-event direction.
    sky = {
        "ra": float(rng.uniform(0.0, 2.0 * np.pi)),
        "dec": float(np.arcsin(rng.uniform(-1.0, 1.0))),
        "psi": float(rng.uniform(0.0, np.pi)),
        "t_c": 0.0,
    }

    needs_noise = bool(
        {"noise", "inj_ccsn", "inj_glitch"}.intersection(classes)
    )
    noise_b = {}
    if needs_noise:
        _log(index, f"fetching noise bundle @ gps={noise_gps:.1f} ...")
        noise_b = {
            detector: build_bundle(
                noise_gps, edir / "noise" / detector, detector
            )
            for detector in detectors
        }

    real_g = {}
    if "real_glitch" in classes:
        _log(
            index,
            "fetching real-glitch bundle "
            f"@ gps={blip_gps:.1f} (catalogue snr={cat_snr:.1f}) ...",
        )
        real_g = {
            detector: build_bundle(
                blip_gps, edir / "realg" / detector, detector
            )
            for detector in detectors
        }

    # Data quality of the NOISE segment, which the noise and injected classes
    # share. Recorded for every event, and optionally used to reject the segment
    # outright: a chunk whose lines are not whitened by the off-source PSD makes
    # every evidence on it unreliable, and no downstream statistic can undo that.
    noise_dq = {}
    if needs_noise:
        dq_prep = prepare_multi_detector_data(
            detectors, bundle_paths=noise_b, flow=flow, fmax=fmax
        )
        noise_dq = {
            name: whitened_band_power(data)
            for name, data in dq_prep.detector_data.items()
        }
        bad = sorted(
            name
            for name, wp in noise_dq.items()
            if wp["mean"] > max_mean_whitened_power
        )
        if bad:
            message = (
                f"noise segment fails whitening in {', '.join(bad)} "
                + ", ".join(
                    f"{n} mean={noise_dq[n]['mean']:.1f}" for n in bad
                )
                + f" (limit {max_mean_whitened_power})"
            )
            if require_clean_noise:
                _log(index, f"SKIPPING: {message}")
                raise CleanSegmentRejected(message)
            _log(index, f"WARNING: {message}")

    prepared = None
    if {"inj_ccsn", "inj_glitch"}.intersection(classes):
        _log(index, "preparing multi-detector injection data ...")
        prepared = prepare_multi_detector_data(
            detectors, bundle_paths=noise_b, flow=flow, fmax=fmax
        )

    bundles: dict[str, dict[str, str]] = {}
    snr: dict[str, float] = {}
    snr_by_detector: dict[str, dict[str, float]] = {}
    injection_indices = {"ccsne_validation": None, "blip_validation": None}
    if "noise" in classes:
        bundles["noise"] = {
            detector: str(path.resolve()) for detector, path in noise_b.items()
        }
        snr["noise"] = 0.0
        snr_by_detector["noise"] = {detector: 0.0 for detector in detectors}

    if "inj_ccsn" in classes:
        ccsn_pool = np.asarray(TrainValData.load(source="ccsne", seed=0).val)
        ccsn_index = int(rng.integers(ccsn_pool.shape[0]))
        injection_indices["ccsne_validation"] = ccsn_index
        _log(index, f"building coherent CCSN injection (SNR={target:.1f}) ...")
        injected, net_snr = _inject_coherent(
            prepared,
            target,
            n_seg,
            dt,
            flow,
            fmax,
            ccsn_pool[ccsn_index],
            sky=sky,
            snr_reference_det=snr_reference_det,
        )
        ccsn_b = {
            detector: inject_into_bundle(
                noise_b[detector],
                injected[detector],
                edir / "ccsn" / f"{detector}.hdf5",
            )
            for detector in detectors
        }
        bundles["inj_ccsn"] = {
            detector: str(path.resolve()) for detector, path in ccsn_b.items()
        }
        snr["inj_ccsn"] = float(net_snr)
        snr_by_detector["inj_ccsn"] = per_detector_snr(
            prepared, injected, n_seg, dt, flow, fmax
        )

    if "inj_glitch" in classes:
        blip_pool = np.asarray(TrainValData.load(source="blip", seed=0).val)
        blip_index = int(rng.integers(blip_pool.shape[0]))
        injection_indices["blip_validation"] = blip_index
        _log(
            index, f"building incoherent blip injection (SNR={target:.1f}) ..."
        )
        injected, glitch_snr = _inject_single_glitch(
            prepared,
            gdet,
            target,
            n_seg,
            dt,
            flow,
            fmax,
            blip_pool[blip_index],
        )
        glitch_b = dict(noise_b)
        glitch_b[gdet] = inject_into_bundle(
            noise_b[gdet],
            injected,
            edir / "iglitch" / f"{gdet}.hdf5",
        )
        bundles["inj_glitch"] = {
            detector: str(path.resolve())
            for detector, path in glitch_b.items()
        }
        snr["inj_glitch"] = float(glitch_snr)
        # Incoherent by construction: all the power is in the host detector.
        snr_by_detector["inj_glitch"] = {
            detector: (float(glitch_snr) if detector == gdet else 0.0)
            for detector in detectors
        }

    # real_glitch deliberately has no snr_by_detector entry: the catalogue SNR is
    # a host-detector trigger value, and the non-host detectors hold whatever the
    # real strain happens to contain. Fabricating a split would be a lie.
    if "real_glitch" in classes:
        bundles["real_glitch"] = {
            detector: str(path.resolve()) for detector, path in real_g.items()
        }
        snr["real_glitch"] = cat_snr

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign_id": campaign_id or outdir.name,
        "index": index,
        "detectors": detectors,
        "glitch_det": gdet,
        "blip_ifo": blip_ifo,
        "prepared_classes": list(classes),
        "band": list(band),
        "gps": {"blip": blip_gps, "noise": noise_gps},
        "sky": sky,
        "snr": snr,
        "snr_by_detector": snr_by_detector,
        "noise_data_quality": noise_dq,
        "injection": {
            "target_snr": target,
            # "network": target_snr IS the network SNR (fixed-budget comparison).
            # A detector name: that detector reaches target_snr and the network
            # SNR is whatever the added detectors give (fixed-source comparison).
            "snr_normalisation": snr_reference_det or "network",
            "training_data_seed": 0,
            "validation_indices": injection_indices,
        },
        "preparation": {
            "snr_grid": [float(value) for value in snr_grid],
            "noise_offset_seconds": float(noise_offset),
            "sample_rate_hz": float(SAMPLE_RATE),
            "segment_duration_seconds": 4.0,
        },
        "bundles": bundles,
        "provenance": _runtime_provenance(),
    }
    manifest["manifest_fingerprint"] = _json_fingerprint(manifest)
    edir.mkdir(parents=True, exist_ok=True)
    (edir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(path)


def _validated_manifest_fingerprint(manifest: dict) -> str:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"This campaign requires a schema-v{MANIFEST_SCHEMA_VERSION} manifest "
            f"prepared with the current runner (found "
            f"v{manifest.get('schema_version')}). Use a fresh campaign output "
            "directory."
        )
    claimed = manifest.get("manifest_fingerprint")
    unsigned = dict(manifest)
    unsigned.pop("manifest_fingerprint", None)
    calculated = _json_fingerprint(unsigned)
    if not claimed or claimed != calculated:
        raise RuntimeError("Manifest fingerprint is missing or invalid.")
    return claimed


def analyse_manifest(
    manifest,
    outdir,
    num_warmup,
    num_samples,
    nsm,
    classes=PRODUCTION_CLASSES,
    *,
    num_chains=2,
    target_accept_prob=0.8,
    max_tree_depth=10,
    map_num_starts=128,
    map_maxiter=400,
    save_artifacts=False,
) -> None:
    detectors = manifest["detectors"]
    flow, fmax = manifest["band"]
    manifest_fingerprint = _validated_manifest_fingerprint(manifest)
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    index = manifest["index"]
    failures_dir = outdir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    analysis_provenance = _runtime_provenance()
    sampler_config = {
        "engine": "nuts",
        "lnz_method": "morph",
        "num_warmup": int(num_warmup),
        "num_samples_per_chain": int(num_samples),
        "num_chains": int(num_chains),
        "target_accept_prob": float(target_accept_prob),
        "max_tree_depth": int(max_tree_depth),
        "map_num_starts": int(map_num_starts),
        "map_maxiter": int(map_maxiter),
        "noise_scale_marginal": bool(nsm),
        "amplitude_marginal": True,
        "nested_fallback_num_live_points": 300,
        "nested_fallback_max_samples": 6000,
        "flow": float(flow),
        "fmax": float(fmax),
    }
    failed_classes = []
    for cls in classes:
        if cls not in manifest["bundles"]:
            raise RuntimeError(
                f"Class '{cls}' was not prepared in {outdir / f'e{index}' / 'manifest.json'}. "
                "Prepare it in a fresh campaign directory."
            )
        out_json = results_dir / f"e{index}_{cls}.json"
        failure_json = failures_dir / f"e{index}_{cls}.json"
        rng_seed = 100_000 + 10 * int(index) + CLASS_SEED_OFFSET[cls]
        analysis_signature = _json_fingerprint(
            {
                "manifest_fingerprint": manifest_fingerprint,
                "class": cls,
                "sampler": sampler_config,
                "analysis_packages": analysis_provenance["packages"],
                "analysis_models": analysis_provenance["models"],
            }
        )
        if out_json.exists():
            existing = json.loads(out_json.read_text())
            if existing.get("analysis_signature") == analysis_signature:
                _log(
                    index,
                    f"analysis skipped; matching result exists: {out_json}",
                )
                continue
            raise RuntimeError(
                f"Refusing to mix a stale result with this campaign: {out_json}. "
                "Use a fresh output directory."
            )
        _log(index, f"analysing class={cls} ...")
        started = time.perf_counter()
        analysis_dir = outdir / f"e{index}" / cls / "analysis"
        try:
            r = run_bcr_posteriors(
                detectors=detectors,
                outdir=str(analysis_dir),
                bundle_paths=manifest["bundles"][cls],
                extrinsic_params=manifest.get("sky"),
                signal_model="ccsne",
                glitch_model="blip",
                flow=flow,
                fmax=fmax,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                map_num_starts=map_num_starts,
                map_maxiter=map_maxiter,
                rng_seed=rng_seed,
                save_artifacts=save_artifacts,
                save_diagnostics=True,
                lnz_method="morph",
                noise_scale_marginal=nsm,
                nested_num_live_points=300,
                nested_max_samples=6000,
            )
            gl = r.get("glitch", {})
            row = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "campaign_id": manifest["campaign_id"],
                "manifest_fingerprint": manifest_fingerprint,
                "analysis_signature": analysis_signature,
                "index": index,
                "cls": cls,
                "detectors": detectors,
                "blip_ifo": manifest["blip_ifo"],
                "glitch_det": manifest["glitch_det"],
                "snr": manifest["snr"][cls],
                "rng_seed": rng_seed,
                "logZ_signal": float(r["signal"]["logZ"]),
                "logZ_glitch": (
                    float(np.nanmax(list(gl.values()))) if gl else float("nan")
                ),
                "logZ_glitch_by_detector": {
                    key: float(value) for key, value in gl.items()
                },
                "logZ_signal_err": float(r["signal"]["logZ_err"]),
                "logZ_glitch_err_by_detector": {
                    key: float(value)
                    for key, value in r.get("glitch_err", {}).items()
                },
                "logZ_noise_by_detector": r.get("noise", {}),
                "log_odds": float(r["bcr_log"]),
                "signal_fit_suspect": r.get("signal_fit_suspect"),
                "data_quality": r.get("data_quality", {}),
                "data_quality_failed": r.get("data_quality_failed", []),
                "evidence_failures": int(r.get("evidence_failures", 0)),
                "evidence_fallbacks": int(r.get("evidence_fallbacks", 0)),
                "evidence_status": r.get("evidence_status", {}),
                "nuts_diagnostics": r.get("nuts_diagnostics", {}),
                "map_initialization": r.get("map_initialization", {}),
                "sampler": sampler_config,
                "analysis_provenance": analysis_provenance,
                "runtime_seconds": time.perf_counter() - started,
            }
            _write_json_atomic(out_json, row)
            failure_json.unlink(missing_ok=True)
            _log(
                index,
                f"{cls:>11s} done: snr={row['snr']:6.1f} logBCR={row['log_odds']:9.1f}",
            )
        except Exception as exc:  # noqa: BLE001
            failed_classes.append(cls)
            _write_json_atomic(
                failure_json,
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "campaign_id": manifest["campaign_id"],
                    "manifest_fingerprint": manifest_fingerprint,
                    "analysis_signature": analysis_signature,
                    "index": index,
                    "cls": cls,
                    "detectors": detectors,
                    "rng_seed": rng_seed,
                    "sampler": sampler_config,
                    "analysis_provenance": analysis_provenance,
                    "runtime_seconds": time.perf_counter() - started,
                    "diagnostics_directory": str(analysis_dir.resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            _log(index, f"{cls} FAILED: {type(exc).__name__}: {exc}")
    if failed_classes:
        raise RuntimeError(
            f"Analysis failed for {len(failed_classes)} class(es): "
            + ", ".join(failed_classes)
            + ". See the per-class failure JSON and diagnostics directory."
        )


def prune_completed_index(manifest: dict, outdir: Path) -> None:
    """Delete an event's bundles and diagnostics once every prepared class has
    an odds result and a baseline result (inode hygiene on the cluster).

    The manifest and the per-class result JSONs stay. A failed class never
    prunes: its bundles and diagnostics directory are kept for debugging, and
    the failure JSON points at them.
    """
    index = manifest["index"]
    classes = manifest["prepared_classes"]
    results_dir = outdir / "results"
    complete = all(
        (results_dir / f"e{index}_{cls}.json").exists()
        and (results_dir / f"e{index}_{cls}_baseline.json").exists()
        for cls in classes
    )
    if not complete:
        return
    edir = outdir / f"e{index}"
    # ponytail: concurrent per-class tasks may race to prune; rmtree with
    # ignore_errors is idempotent, so the race is benign.
    for name in ("noise", "realg", "ccsn", "iglitch"):
        shutil.rmtree(edir / name, ignore_errors=True)
    for cls in classes:
        shutil.rmtree(edir / cls / "analysis", ignore_errors=True)
        try:
            (edir / cls).rmdir()
        except OSError:
            pass
    _log(index, "pruned bundles + diagnostics (all classes complete)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--detectors", nargs="+", default=["L1"])
    p.add_argument("--glitch-det", default="L1")
    p.add_argument(
        "--blip-ifo",
        default="L1",
        choices=["H1", "L1"],
        help="Gravity Spy catalogue for the real_glitch class (blip host detector).",
    )
    p.add_argument(
        "--stage", choices=["prep", "analysis", "both"], default="both"
    )
    p.add_argument("--outdir", type=Path, default=Path("out_rn"))
    p.add_argument("--snr-grid", type=float, nargs="+", default=[10, 20, 40])
    p.add_argument("--noise-offset", type=float, default=100.0)
    p.add_argument("--flow", type=float, default=300.0)
    p.add_argument("--fmax", type=float, default=800.0)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-chains", type=int, default=2)
    p.add_argument("--target-accept-prob", type=float, default=0.8)
    p.add_argument("--max-tree-depth", type=int, default=10)
    p.add_argument("--map-num-starts", type=int, default=128)
    p.add_argument("--map-maxiter", type=int, default=400)
    p.add_argument(
        "--class",
        dest="class_name",
        choices=CLASSES,
        help="Analyse only one class. Use this for bounded SLURM backfill tasks.",
    )
    p.add_argument(
        "--prep-classes",
        nargs="+",
        choices=CLASSES,
        help=(
            "Classes to prepare (default: noise inj_ccsn real_glitch). "
            "The injected-blip class is opt-in."
        ),
    )
    p.add_argument(
        "--campaign-id",
        help="Immutable campaign identifier stored in manifests and results.",
    )
    p.add_argument(
        "--snr-reference-det",
        help=(
            "Normalise the injected amplitude so THIS detector reaches the "
            "target SNR, letting the network SNR be whatever the added "
            "detectors give (fixed-source comparison: identical strain in the "
            "reference detector, plus a detector). Set it to the detector the "
            "one-detector campaign used. Omit to normalise on the network SNR, "
            "which instead redistributes a fixed budget."
        ),
    )
    p.add_argument("--no-marginal", action="store_true")
    p.add_argument(
        "--require-clean-noise",
        action="store_true",
        help="Reject an event at preparation when its noise segment fails the "
             "whitening check in ANY detector. A segment whose lines are not "
             "described by the off-source PSD makes every evidence on it "
             "unreliable, so for a methods campaign it is cleaner to select "
             "usable segments up front than to filter afterwards.",
    )
    p.add_argument(
        "--max-mean-whitened-power",
        type=float,
        default=10.0,
        help="Whitening limit for --require-clean-noise (mean per-bin whitened "
             "power; 1.0 is perfect).",
    )
    p.add_argument(
        "--keep-bundles",
        action="store_true",
        help="Skip the post-analysis pruning of bundles and diagnostics.",
    )
    p.add_argument(
        "--force-prep",
        action="store_true",
        help="Rebuild an existing manifest and its prepared bundles.",
    )
    p.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Write per-class signal/samples.npz posterior draws (needed by "
             "studies/plot_waveform_reconstruction.py). Off by default because "
             "a full population run would emit hundreds of these.",
    )
    args = p.parse_args()

    detectors = [d.upper() for d in args.detectors]
    glitch_det = args.glitch_det.upper()
    if args.index < 0:
        p.error("--index must be non-negative")
    if len(detectors) != len(set(detectors)) or not set(detectors).issubset(
        {"H1", "L1"}
    ):
        p.error("--detectors must contain unique H1 and/or L1 values")
    if glitch_det not in detectors:
        p.error(
            f"--glitch-det {glitch_det} is absent from --detectors {detectors}"
        )
    snr_reference_det = (
        args.snr_reference_det.upper() if args.snr_reference_det else None
    )
    if snr_reference_det is not None and snr_reference_det not in detectors:
        p.error(
            f"--snr-reference-det {snr_reference_det} is absent from "
            f"--detectors {detectors}"
        )
    if args.flow >= args.fmax:
        p.error("--flow must be lower than --fmax")
    if args.num_chains < 2:
        p.error("NUTS+MorphLnZ production analyses require --num-chains >= 2")
    if (
        min(
            args.num_warmup,
            args.num_samples,
            args.max_tree_depth,
            args.map_num_starts,
            args.map_maxiter,
        )
        < 1
    ):
        p.error("sampler counts and limits must all be positive")
    if not 0.0 < args.target_accept_prob < 1.0:
        p.error("--target-accept-prob must be strictly between 0 and 1")
    if not args.snr_grid or min(args.snr_grid) <= 0:
        p.error("--snr-grid values must all be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / f"e{args.index}" / "manifest.json"
    campaign_id = args.campaign_id or args.outdir.name
    prep_classes = tuple(args.prep_classes or PRODUCTION_CLASSES)
    if args.stage == "both" and args.class_name and not args.prep_classes:
        prep_classes = (args.class_name,)

    _log(
        args.index,
        f"launched: stage={args.stage} detectors={detectors} "
        f"glitch_det={glitch_det} blip_ifo={args.blip_ifo} "
        f"campaign={campaign_id} outdir={args.outdir}",
    )
    if args.stage in ("prep", "both"):
        if manifest_path.exists() and not args.force_prep:
            existing_manifest = json.loads(manifest_path.read_text())
            _validated_manifest_fingerprint(existing_manifest)
            expected = {
                "campaign_id": campaign_id,
                "detectors": detectors,
                "glitch_det": glitch_det,
                "blip_ifo": args.blip_ifo,
                "band": [args.flow, args.fmax],
                "preparation": {
                    "snr_grid": [float(value) for value in args.snr_grid],
                    "noise_offset_seconds": float(args.noise_offset),
                    "sample_rate_hz": float(SAMPLE_RATE),
                    "segment_duration_seconds": 4.0,
                },
            }
            mismatched = {
                key: (existing_manifest.get(key), value)
                for key, value in expected.items()
                if existing_manifest.get(key) != value
            }
            # Normalisation decides what target_snr MEANS, so a manifest prepared
            # under the other convention is not interchangeable with this one.
            expected_normalisation = snr_reference_det or "network"
            existing_normalisation = (
                existing_manifest.get("injection") or {}
            ).get("snr_normalisation")
            if existing_normalisation != expected_normalisation:
                mismatched["injection.snr_normalisation"] = (
                    existing_normalisation,
                    expected_normalisation,
                )
            missing_classes = sorted(
                set(prep_classes)
                - set(existing_manifest.get("prepared_classes", []))
            )
            if mismatched or missing_classes:
                raise RuntimeError(
                    "Existing manifest does not match this submission: "
                    f"mismatched={mismatched}, missing_classes={missing_classes}. "
                    "Use a fresh campaign output directory."
                )
            _log(
                args.index,
                f"prep skipped; matching manifest exists: {manifest_path}",
            )
        else:
            existing_results = sorted(
                (args.outdir / "results").glob(f"e{args.index}_*.json")
            )
            if args.force_prep and existing_results:
                raise RuntimeError(
                    "Refusing to rebuild a manifest with existing analysis "
                    f"results: {existing_results}. Use a fresh campaign output directory."
                )
            try:
                prep_index(
                    args.index,
                    detectors,
                    glitch_det,
                    (args.flow, args.fmax),
                    args.snr_grid,
                    args.noise_offset,
                    args.outdir,
                    blip_ifo=args.blip_ifo,
                    classes=prep_classes,
                    campaign_id=campaign_id,
                    snr_reference_det=snr_reference_det,
                    require_clean_noise=args.require_clean_noise,
                    max_mean_whitened_power=args.max_mean_whitened_power,
                )
            except StrainFetchFailed as exc:
                rejected = _record_unavailable_trigger(
                    index=args.index,
                    outdir=args.outdir,
                    campaign_id=campaign_id,
                    detectors=detectors,
                    blip_ifo=args.blip_ifo,
                    error=exc,
                )
                _log(
                    args.index,
                    f"REJECTED: strain acquisition failed -> {rejected}",
                )
                # Keep the task failed so aftercorr-dependent analyses do not
                # run against a missing manifest.  Future submission scans see
                # rejected.json and do not requeue this trigger.
                raise
            _log(args.index, f"prep done -> {manifest_path}")
    if args.stage in ("analysis", "both"):
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing manifest {manifest_path}; run the prep stage first."
            )
        manifest = json.loads(manifest_path.read_text())
        _validated_manifest_fingerprint(manifest)
        expected = {
            "campaign_id": campaign_id,
            "detectors": detectors,
            "glitch_det": glitch_det,
            "blip_ifo": args.blip_ifo,
            "band": [args.flow, args.fmax],
        }
        mismatched = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                "Manifest does not match this analysis submission: "
                f"{mismatched}. Use the original submission settings or a "
                "fresh campaign output directory."
            )
        classes = (args.class_name,) if args.class_name else PRODUCTION_CLASSES
        analyse_manifest(
            manifest,
            args.outdir,
            args.num_warmup,
            args.num_samples,
            not args.no_marginal,
            classes=classes,
            num_chains=args.num_chains,
            target_accept_prob=args.target_accept_prob,
            max_tree_depth=args.max_tree_depth,
            map_num_starts=args.map_num_starts,
            map_maxiter=args.map_maxiter,
            save_artifacts=args.save_artifacts,
        )
        # newSNR/chi^2 baseline: seconds per class, no sampling. Runs on every
        # prepared class (skips existing rows), so the last class task to
        # finish completes the set and prunes the event's heavy files.
        run_baseline_index(manifest["index"], args.outdir, 128, 8)
        # --save-artifacts writes samples.npz under <edir>/<cls>/analysis/, which
        # is exactly what prune_completed_index deletes; asking for artifacts
        # must not silently destroy them.
        if not args.keep_bundles and not args.save_artifacts:
            prune_completed_index(manifest, args.outdir)
    _log(args.index, "all stages complete")


if __name__ == "__main__":
    main()
