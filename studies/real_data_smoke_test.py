"""Single-event real-data smoke test (single-detector L1).

Goal: prove the real-data path runs end-to-end on REAL, non-Gaussian GWOSC L1
strain before scaling to a population, and confirm the BCR behaves qualitatively:

    scenario      expected ranking behaviour
    --------      --------------------------
    noise         logZ_signal ~ 0, logZ_glitch ~ 0  -> BCR ~ neutral/negative
    noise_inj     logZ_signal  > 0 (injected CCSN)  -> BCR strongly positive
    blip          logZ_glitch  > logZ_signal        -> BCR negative

It also gives the first REAL-data check on whether logZ_noise ~ 0 holds (the
hardcoded analytic noise reference), which feeds the BCR derivation we still owe.

DATA-QUALITY NOTE
-----------------
The standard GWOSC-fallback path in ``strain_loader`` requires the analysis
window to *pass* the CBC CAT3 veto. A blip glitch is precisely what that veto
flags, so a window containing a blip never passes (verified: 0/40 catalogue
blips loadable that way). We therefore fetch strain via an explicit GWOSC
``data_fetcher``, which bypasses the CAT3 gate by design -- the right behaviour
for the blip scenario, where we *want* the transient. Noise times are placed
well away from any catalogue blip so the noise chunk is clean.

Trigger times come from the GravitySpy O3b blip catalogue (L1-only): a real
high-confidence blip for the blip scenario, and a quiet stretch 100 s before it
for the noise / noise_inj scenarios.

Run:
    uv run python studies/real_data_smoke_test.py --outdir out_smoke
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from starccato_lvk.workflows.run_event import CONFIG_DEFAULT, load_analysis_config, inject_signal
from starccato_lvk.analysis import run_bcr_posteriors
from starccato_lvk.acquisition.io.strain_loader import strain_loader, _remote_data_fetcher
from starccato_lvk.acquisition.io.glitch_catalog import get_blip_trigger_time


def _build_bundle(t: float, outdir: Path, detector: str = "L1") -> Path:
    """Build an analysis bundle at ``t`` from LOCAL strain if mirrored, else GWOSC.

    ``data_fetcher=None`` tries the local mirror first (``config.DATA_DIRS`` -> no
    internet, no CAT3 gate; ideal on OzSTAR), and only falls back to a GWOSC fetch
    when no local file is found. ``require_cat3=False`` so the GWOSC fallback also
    works for blip times (a blip fails the CAT3 veto by construction).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    existing = list(outdir.glob("analysis_bundle_*.hdf5"))
    if existing:
        return existing[0]
    strain_loader(
        trigger_time=t,
        outdir=str(outdir),
        data_fetcher=None,
        detector=detector,
        require_cat3=False,
    )
    bundles = list(outdir.glob("analysis_bundle_*.hdf5"))
    if not bundles:
        raise FileNotFoundError(f"No bundle written for {detector} at gps {t}")
    return bundles[0]


def _bcr(cfg, bundle_paths: dict, outdir: Path, *, num_warmup: int, num_samples: int, seed: int,
        flow: float, fmax: float, noise_scale_marginal: bool = False) -> dict:
    return run_bcr_posteriors(
        detectors=["L1"],
        outdir=str(outdir),
        bundle_paths={d: str(p) for d, p in bundle_paths.items()},
        signal_model=cfg.signal_model,
        glitch_model=cfg.glitch_model,
        latent_sigma_signal=cfg.signal_latent_sigma,
        log_amp_sigma_signal=cfg.signal_log_amp_sigma,
        latent_sigma_glitch=cfg.glitch_latent_sigma,
        log_amp_sigma_glitch=cfg.glitch_log_amp_sigma,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=1,
        rng_seed=seed,
        save_artifacts=True,
        ci=cfg.ci,
        alpha=cfg.alpha,
        beta=cfg.beta,
        lnz_method="morph",
        flow=flow,
        fmax=fmax,
        noise_scale_marginal=noise_scale_marginal,
    )


def _summarise(scenario: str, result: dict | None) -> dict:
    if result is None:
        return {"scenario": scenario, "status": "failed"}
    return {
        "scenario": scenario,
        "status": "ok",
        "logZ_signal": result.get("signal", {}).get("logZ"),
        "logZ_glitch": result.get("glitch", {}),
        "logZ_noise": result.get("noise", {}),
        "log_bcr": result.get("bcr_log"),
        "bcr": result.get("bcr"),
        "evidence_failures": result.get("evidence_failures"),
        "evidence_fallbacks": result.get("evidence_fallbacks"),
        "evidence_status": result.get("evidence_status"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("out_smoke"))
    p.add_argument("--blip-index", type=int, default=0, help="Row in the blip catalogue for the blip scenario.")
    p.add_argument("--noise-offset", type=float, default=100.0, help="Seconds before the blip to place the (clean) noise trigger.")
    p.add_argument("--distance", type=float, default=1.0, help="Injection distance scale for noise_inj.")
    p.add_argument("--num-warmup", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=800)
    p.add_argument("--flow", type=float, default=256.0, help="Low-frequency cutoff (avoid the seismic/scattered-light wall).")
    p.add_argument("--fmax", type=float, default=896.0, help="High-frequency cutoff (avoid violin modes).")
    p.add_argument("--scenarios", nargs="+", default=["noise", "noise_inj", "blip"])
    p.add_argument("--noise-scale-marginal", action="store_true",
                   help="Use the PSD-amplitude-marginal likelihood (robust to non-stationary / "
                        "mis-estimated real-data PSDs).")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_analysis_config(CONFIG_DEFAULT)
    cfg.detectors = ["L1"]

    blip_gps = get_blip_trigger_time(args.blip_index)
    noise_gps = blip_gps - args.noise_offset
    print(f"[smoke] blip_gps={blip_gps:.3f}  noise_gps={noise_gps:.3f} "
          f"(={args.noise_offset:.0f}s before blip#{args.blip_index})")

    rows = []
    noise_bundle = None
    for scenario in args.scenarios:
        print("\n" + "=" * 70)
        print(f"SCENARIO={scenario}  detector=L1")
        print("=" * 70)
        try:
            if scenario in ("noise", "noise_inj") and noise_bundle is None:
                noise_bundle = _build_bundle(noise_gps, args.outdir / "noise" / "bundle")

            if scenario == "noise":
                bundles = {"L1": noise_bundle}
            elif scenario == "noise_inj":
                inj = inject_signal(
                    {"L1": noise_bundle}, noise_gps, cfg,
                    args.outdir / "noise_inj" / "bundle_injected",
                    distance_scale=args.distance,
                )
                bundles = {"L1": inj["L1"]}
            elif scenario == "blip":
                bundles = {"L1": _build_bundle(blip_gps, args.outdir / "blip" / "bundle")}
            else:
                raise ValueError(f"unknown scenario {scenario}")

            result = _bcr(
                cfg, bundles, args.outdir / scenario / "analysis",
                num_warmup=args.num_warmup, num_samples=args.num_samples,
                seed=args.blip_index, flow=args.flow, fmax=args.fmax,
                noise_scale_marginal=args.noise_scale_marginal,
            )
        except Exception as exc:  # noqa: BLE001 - smoke test: report, don't abort
            print(f"[smoke] scenario '{scenario}' FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            result = None
        rows.append(_summarise(scenario, result))

    print("\n" + "#" * 70)
    print("SMOKE TEST SUMMARY (single-detector L1, real GWOSC data)")
    print("#" * 70)
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['scenario']:>10s}: FAILED")
            continue
        print(
            f"  {r['scenario']:>10s}: logZ_sig={_fmt(r['logZ_signal'])}  "
            f"logZ_glitch={_fmt_dict(r['logZ_glitch'])}  "
            f"logBCR={_fmt(r['log_bcr'])}  fails={r['evidence_failures']} fallbacks={r['evidence_fallbacks']}"
        )

    (args.outdir / "smoke_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.outdir/'smoke_summary.json'}")


def _fmt(x) -> str:
    try:
        return f"{float(x):+8.2f}"
    except (TypeError, ValueError):
        return f"{x}"


def _fmt_dict(d) -> str:
    if not isinstance(d, dict):
        return _fmt(d)
    return "{" + ", ".join(f"{k}:{_fmt(v)}" for k, v in d.items()) + "}"


if __name__ == "__main__":
    main()
