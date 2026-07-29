"""Recover result.json for BayesWave runs that finished sampling but were rejected.

Two distinct failure modes leave a completed run without a ``result.json``:

1. **Placeholder summary.** ``evidence.dat`` holds sentinels (``signal 0 0``,
   ``glitch 10000 1``) because BayesWavePost never ran, so the file was never
   filled in. The sampling itself finished and BayesWave's own thermodynamic
   integration is recorded in ``bayeswave.log``; parsing it reproduces
   ``evidence.dat`` to <0.5 nat on runs where both exist.

2. **Degenerate uncertainty.** ``evidence.dat`` holds real log-evidences but the
   uncertainty on a model underflowed to 0 (or a tiny negative, e.g. -8e-05),
   typically when a model finds no wavelets so the likelihood is flat across
   temperature rungs. ``parse_evidence`` rejects those as "placeholder/unsampled",
   which is right for case 1 but discards a usable log-evidence here.

Recovered results are **evidence-only**: without post-processing there are no
posterior waveform draws, so they support the lnO-vs-lnBF evidence comparison
but not the morphology overlay. Every recovered file is tagged (``recovered``,
``evidence_source``, ``degenerate_uncertainty``, ``posteriors_available``) so it
can be filtered out of any analysis that needs the full product.

Never overwrites an existing result.json. Dry run unless --apply is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

MODEL_FROM_LOG = {"signal": "signal", "glitch": "glitch", "Gaussian noise": "noise"}
SENTINELS = (0.0, 10000.0)
TI_RE = re.compile(
    r"Thermodynamic Integration logZ\s*=\s*(-?[\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)"
)


def parse_evidence_dat(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                out[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return out


def parse_log_evidence(path: Path) -> dict[str, tuple[float, float]]:
    """BayesWave's own TI evidence per model, keyed off the 'characterizing' headers.

    A model may be characterized more than once (restarts); the last block wins,
    which is the attempt that actually carried through to the end of the run.
    """
    out: dict[str, tuple[float, float]] = {}
    if not path.is_file():
        return out
    current: str | None = None
    for line in path.read_text().splitlines():
        header = re.search(r"characterizing (.+?) model", line)
        if header:
            current = MODEL_FROM_LOG.get(header.group(1).strip())
            continue
        found = TI_RE.search(line)
        if found and current:
            try:
                out[current] = (float(found.group(1)), float(found.group(2)))
            except ValueError:
                continue
    return out


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def recover_event(event_dir: Path, cls: str) -> dict | None:
    evidence = parse_evidence_dat(event_dir / "evidence.dat")
    is_placeholder = any(z in SENTINELS for z, _ in evidence.values())

    if is_placeholder or not {"signal", "glitch"} <= set(evidence):
        evidence = parse_log_evidence(event_dir / "bayeswave.log")
        source = "bayeswave_log"
    else:
        source = "evidence_dat"

    if not {"signal", "glitch"} <= set(evidence):
        return None
    if any(z in SENTINELS for z, _ in evidence.values()):
        return None  # log did not supply real numbers either

    z_sig, u_sig = evidence["signal"]
    z_gl, u_gl = evidence["glitch"]
    if not (_finite(z_sig) and _finite(z_gl)):
        return None

    degenerate = not (u_sig > 0 and u_gl > 0)
    lnbf = z_sig - z_gl
    lnbf_unc = (
        math.sqrt(u_sig**2 + u_gl**2) if (u_sig > 0 and u_gl > 0) else float("nan")
    )

    meta_path = event_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    z_noise, u_noise = evidence.get("noise", (float("nan"), float("nan")))

    return {
        "cls": cls,
        "index": int(event_dir.parent.name[1:]),
        "detectors": meta.get("detectors", ["H1", "L1"]),
        "logZ_signal": z_sig,
        "logZ_glitch": z_gl,
        "logZ_noise": z_noise,
        "evidence_uncertainty": {
            "signal": u_sig,
            "glitch": u_gl,
            "noise": u_noise,
        },
        "log_bayeswave_signal_glitch": lnbf,
        "log_bayeswave_signal_glitch_uncertainty": lnbf_unc,
        "target_snr": meta.get("target_snr", float("nan")),
        "signal_reconstructed_snr_median": float("nan"),
        "signal_reconstructed_snr_per_detector": {},
        "manifest": meta.get("manifest", ""),
        "output_dir": str(event_dir.resolve()),
        "settings": meta.get("settings", {}),
        # provenance -- these results are NOT interchangeable with a full run
        "recovered": True,
        "evidence_source": source,
        "degenerate_uncertainty": degenerate,
        "posteriors_available": (
            event_dir / "post" / "signal"
        ).is_dir(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--bw-campaigns", nargs="+", required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    totals = {"from_log": 0, "from_dat": 0, "degenerate": 0, "unrecoverable": 0}
    for spec in args.bw_campaigns:
        campaign, _, cls = spec.partition(":")
        base = args.results_root / campaign / "bw_fixedsky"
        for event_dir in sorted(base.glob(f"e*/{cls}")):
            if (event_dir / "result.json").is_file():
                continue  # never touch a run that already succeeded
            result = recover_event(event_dir, cls)
            if result is None:
                totals["unrecoverable"] += 1
                continue
            totals["from_log" if result["evidence_source"] == "bayeswave_log" else "from_dat"] += 1
            totals["degenerate"] += bool(result["degenerate_uncertainty"])
            if args.apply:
                (event_dir / "result.json").write_text(json.dumps(result, indent=2))

    mode = "WROTE" if args.apply else "would write (dry run)"
    print(f"{mode}: {totals['from_log']} from bayeswave.log, "
          f"{totals['from_dat']} from evidence.dat "
          f"({totals['degenerate']} flagged degenerate_uncertainty); "
          f"{totals['unrecoverable']} unrecoverable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
