"""Real-data (single-detector L1) SNR-vs-Odds ROC on GWOSC strain.

The honest real-data counterpart of ``snr_vs_odds_roc.py``. For each catalogue
blip we build three events from REAL GWOSC L1 strain:

    noise   : a quiet stretch before the blip          (background)
    signal  : that noise + a held-out CCSN injection    (foreground)
              at a target optimal SNR
    blip     : the real GravitySpy blip                 (background)

Recovery is the single-detector BCR with the off-source Welch PSD already baked
into the bundle and (by default) the PSD-amplitude marginal for robustness. We
record the odds (log BCR) and an SNR-like loudness for each event, then compare
ROC/AUC for the odds vs SNR -- the demonstration that the VAE odds rejects loud
real glitches that SNR ranks as signal-like.

SNR axis: injected optimal SNR for signals, GravitySpy catalogue SNR for blips,
0 for noise -- realistic loudnesses (real blips are loud, real CCSNe faint).

    uv run python studies/real_data_roc.py --n-blips 8 --outdir out_real_roc
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from starccato_lvk.analysis import run_bcr_posteriors
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
from starccato_lvk.acquisition.io.glitch_catalog import get_blip_trigger_time, load_blip_glitch_catalog

from real_data_smoke_test import _build_bundle
from snr_vs_odds_roc import SAMPLE_RATE, EventRow, _inject_through_recovery, _roc_auc

try:
    from starccato_jax.data.training_data import TrainValData
except Exception:  # pragma: no cover
    TrainValData = None


def _inject_into_bundle(noise_bundle: Path, inj_td: np.ndarray, dest: Path) -> Path:
    """Add a time-domain injection to a real noise bundle's strain (keeps the PSD)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(noise_bundle, "r") as src:
        strain = np.array(src["strain"]["values"])
        injected = strain + np.asarray(inj_td)[: strain.shape[0]]
        with h5py.File(dest, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            src.copy("psd", dst)
            if "full_strain" in src:
                src.copy("full_strain", dst)
            grp = dst.create_group("strain")
            grp.create_dataset("values", data=injected)
            for k, v in src["strain"].attrs.items():
                grp.attrs[k] = v
    return dest


def _run_bcr(bundle: Path, outdir: Path, flow, fmax, nsm, num_warmup, num_samples):
    return run_bcr_posteriors(
        detectors=["L1"], outdir=str(outdir), bundle_paths={"L1": str(bundle)},
        signal_model="ccsne", glitch_model="blip", flow=flow, fmax=fmax,
        num_warmup=num_warmup, num_samples=num_samples, save_artifacts=False,
        lnz_method="morph", noise_scale_marginal=nsm,
        # widened amplitude prior (default 5) + morphZ-vs-nested cross-check on loud
        # events (verify_logz_threshold default 50); keep the cross-check's nested run small.
        nested_num_live_points=300, nested_max_samples=6000,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("out_real_roc"))
    p.add_argument("--n-blips", type=int, default=8)
    p.add_argument("--noise-offset", type=float, default=100.0)
    p.add_argument("--snr-grid", type=float, nargs="+", default=[10, 20, 40])
    p.add_argument("--flow", type=float, default=300.0)
    p.add_argument("--fmax", type=float, default=800.0)
    p.add_argument("--num-warmup", type=int, default=200)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--no-marginal", action="store_true", help="Use the Gaussian likelihood (default: PSD-amplitude marginal).")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    nsm = not args.no_marginal
    dt = 1.0 / SAMPLE_RATE

    cat = load_blip_glitch_catalog()
    held_ccsn = np.asarray(TrainValData.load(source="ccsne", seed=0).val) if TrainValData else None
    wf_rng = np.random.default_rng(7)
    print(f"[real-roc] n_blips={args.n_blips} band=[{args.flow},{args.fmax}] marginal={nsm}")

    rows: list[EventRow] = []
    eid = 0
    for i in range(args.n_blips):
        blip_gps = get_blip_trigger_time(i)
        noise_gps = blip_gps - args.noise_offset
        cat_snr = float(cat.iloc[i]["snr"])
        target_snr = float(args.snr_grid[i % len(args.snr_grid)])
        try:
            noise_bundle = _build_bundle(noise_gps, args.outdir / f"e{i}/noise")
            blip_bundle = _build_bundle(blip_gps, args.outdir / f"e{i}/blip")
        except Exception as exc:  # noqa: BLE001
            print(f"[real-roc] blip#{i} bundle FAILED: {type(exc).__name__}: {exc}")
            continue

        # signal injection: held-out CCSN into the real noise, at target SNR
        n_seg = int(round(4.0 * SAMPLE_RATE))
        prep = prepare_multi_detector_data(["L1"], bundle_paths={"L1": noise_bundle}, flow=args.flow, fmax=args.fmax)
        raw = held_ccsn[wf_rng.integers(held_ccsn.shape[0])] if held_ccsn is not None else None
        inj_td, inj_snr = _inject_through_recovery(prep, "signal", target_snr, n_seg, dt, args.flow, args.fmax, raw_wf=raw)
        inj_bundle = _inject_into_bundle(noise_bundle, inj_td, args.outdir / f"e{i}/signal/inj.hdf5")

        for cls, bundle, snr in [("noise", noise_bundle, 0.0), ("signal", inj_bundle, inj_snr), ("blip", blip_bundle, cat_snr)]:
            try:
                r = _run_bcr(bundle, args.outdir / f"e{i}/{cls}/a", args.flow, args.fmax, nsm, args.num_warmup, args.num_samples)
                row = EventRow(event_id=eid, cls=cls, target_snr=snr, injected_snr=snr,
                               logZ_signal=float(r["signal"]["logZ"]),
                               logZ_glitch=float(r["glitch"].get("L1", np.nan)),
                               logZ_noise=0.0, log_odds=float(r["bcr_log"]),
                               evidence_failures=int(r.get("evidence_failures", 0)))
                print(f"[{cls:>6s} blip#{i}] snr={snr:6.1f} logZ_s={row.logZ_signal:8.1f} "
                      f"logZ_g={row.logZ_glitch:8.1f} logBCR={row.log_odds:8.1f}")
                rows.append(row)
                # Save incrementally so progress survives interruptions (the nested
                # cross-checks on loud events make a full run long).
                (args.outdir / "real_roc_rows.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
            except Exception as exc:  # noqa: BLE001
                print(f"[real-roc] {cls} blip#{i} FAILED: {type(exc).__name__}: {exc}")
            eid += 1

    (args.outdir / "real_roc_rows.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
    snr = {c: np.array([r.injected_snr for r in rows if r.cls == c]) for c in ("noise", "signal", "glitch")}
    snr["glitch"] = np.array([r.injected_snr for r in rows if r.cls == "blip"])
    odds = {c: np.array([r.log_odds for r in rows if r.cls == c]) for c in ("noise", "signal", "blip")}
    bg_snr = np.concatenate([snr["noise"], snr["glitch"]]) if rows else np.array([])
    bg_odds = np.concatenate([odds["noise"], odds["blip"]]) if rows else np.array([])
    aucs = {
        "auc_snr_signal_vs_background": _roc_auc(snr["signal"], bg_snr),
        "auc_odds_signal_vs_background": _roc_auc(odds["signal"], bg_odds),
        "auc_odds_signal_vs_blip": _roc_auc(odds["signal"], odds["blip"]),
        "n_blips_misclassified": int(np.sum(odds["blip"] > 0)) if odds["blip"].size else 0,
        "n_blips": int(odds["blip"].size),
    }
    (args.outdir / "real_roc_summary.json").write_text(json.dumps(aucs, indent=2))
    print("\nREAL-DATA ROC summary:")
    for k, v in aucs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
