"""Append BayesWave residual diagnostics to an existing combined-results HDF5.

Fast companion to ``build_combined_results.py``: it only walks the BayesWave
campaign directories, so it does not repeat the slow lnO scan. Adds the small
per-event diagnostic tables that the main builder skips -- Anderson-Darling
residual p-values, per-detector signal stats and whitened moments -- which are
the residual-gaussianity evidence a referee is most likely to ask for. Run this
before deleting any ``post/`` directory so the prune stays lossless.

Usage:
    python studies/append_bw_diagnostics.py \
        --results-root /fred/oz303/avajpeyi/results/starccato_lvk \
        --bw-campaigns bwcomp_nml_v044_ccsn:inj_ccsn bwcomp_nml_v044_glitch:real_glitch \
        --h5 /fred/oz303/avajpeyi/results/starccato_lvk/combined_results_nuts_morphlnz_v044.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def _load(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        arr = np.loadtxt(path, dtype=np.float32)
    except Exception:
        return None
    return arr if arr.size else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--bw-campaigns", nargs="+", required=True)
    p.add_argument("--h5", type=Path, required=True)
    args = p.parse_args()

    n_events = n_arrays = 0
    with h5py.File(args.h5, "a") as h5f:
        for spec in args.bw_campaigns:
            campaign, _, cls = spec.partition(":")
            base = args.results_root / campaign / "bw_fixedsky"
            post_group = f"bayeswave/{campaign}/{cls}/posteriors"
            if post_group not in h5f:
                print(f"  skip {campaign}/{cls}: no posteriors group", file=sys.stderr)
                continue
            for event_dir in sorted(base.glob(f"e*/{cls}")):
                if not (event_dir / "result.json").is_file():
                    continue
                idx = event_dir.parent.name  # "e123"
                if idx not in h5f[post_group]:
                    continue
                grp = h5f[f"{post_group}/{idx}"]
                post = event_dir / "post"
                candidates = {
                    "anderson_darling_p_values": post / "anderson_darling_p_values.dat",
                }
                for ifo in ("H1", "L1"):
                    candidates[f"signal_stats_{ifo}"] = post / "signal" / f"signal_stats_{ifo}.dat"
                    candidates[f"signal_whitened_moments_{ifo}"] = (
                        post / "signal" / f"signal_whitened_moments_{ifo}.dat"
                    )
                wrote = False
                for name, path in candidates.items():
                    if name in grp:
                        continue
                    arr = _load(path)
                    if arr is None:
                        continue
                    grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
                    n_arrays += 1
                    wrote = True
                n_events += bool(wrote)
    print(f"appended diagnostics for {n_events} events ({n_arrays} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
