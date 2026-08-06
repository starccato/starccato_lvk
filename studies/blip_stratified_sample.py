"""Phase 0a: stratified sample of real cataloged blips, fetched and cored.

The FF study earlier in this investigation sampled the loudest-N blips by
catalog order, which is a high-frequency-biased sample (peak frequency rises
steeply with SNR). This script instead stratifies by GravitySpy
`peak_frequency`, so Phase 0b/0c see the population gengli was actually built
for as well as the population that motivated this investigation.

    ./.venv/bin/python studies/blip_stratified_sample.py --ifo L1 --run O3b \
        --per-bin 8 --outdir ../blip_stratified/L1_O3b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from real_noise_io import build_bundle

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "starccato_lvk" / "acquisition" / "io" / "data"

# Same peak-frequency bin edges used throughout this investigation (matches
# the population breakdown in the rebuild plan).
FREQ_BINS = [(0, 150), (150, 250), (250, 350), (350, 550), (550, 900), (900, 4096)]


def load_catalog(ifo: str, run: str, min_confidence: float = 0.9,
                 min_duration: float = 0.125) -> pd.DataFrame:
    """Raw GravitySpy CSV, filtered to confident blips -- keeps peak_frequency.

    `glitch_catalog.load_blip_glitch_catalog` drops that column, so this reads
    the cached run-level CSV directly (same filter, superset of columns).
    """
    path = DATA_DIR / f"{ifo}_{run}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not cached. Fetch it first: "
            f"curl -sL -o {path} https://zenodo.org/records/5649212/files/{ifo}_{run}.csv"
        )
    df = pd.read_csv(path, usecols=["event_time", "duration", "snr", "peak_frequency",
                                    "ml_label", "ml_confidence"])
    return df[(df.ml_label == "Blip") & (df.ml_confidence >= min_confidence)
              & (df.duration >= min_duration)].reset_index(drop=True)


def stratified_indices(catalog: pd.DataFrame, per_bin: int, seed: int) -> list[int]:
    """Row indices spanning the peak-frequency bins, drawn WITHOUT SNR bias.

    Within each bin, sample uniformly at random rather than by loudness --
    the whole point is to stop over-representing the loud tail.
    """
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for lo, hi in FREQ_BINS:
        mask = (catalog.peak_frequency >= lo) & (catalog.peak_frequency < hi)
        pool = catalog.index[mask].to_numpy()
        if pool.size == 0:
            continue
        take = min(per_bin, pool.size)
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    return sorted(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ifo", required=True, choices=["H1", "L1"])
    p.add_argument("--run", required=True, choices=["O2", "O3a", "O3b"])
    p.add_argument("--per-bin", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    catalog = load_catalog(args.ifo, args.run)
    indices = stratified_indices(catalog, args.per_bin, args.seed)
    print(f"catalog: {len(catalog)} blips; sampling {len(indices)} across "
          f"{len(FREQ_BINS)} peak-frequency bins")

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row_idx in indices:
        row = catalog.iloc[row_idx]
        gps = float(row.event_time)
        try:
            bundle = build_bundle(gps, args.outdir / f"e{row_idx}" / args.ifo, args.ifo)
        except Exception as exc:
            print(f"[e{row_idx}] FETCH FAILED @ gps={gps:.2f}: {type(exc).__name__}: {exc}")
            continue
        entry = {
            "catalog_row": row_idx,
            "gps": gps,
            "snr": float(row.snr),
            "peak_frequency": float(row.peak_frequency),
            "duration": float(row.duration),
            "bundle": str(bundle),
        }
        manifest.append(entry)
        print(f"[e{row_idx}] snr={row.snr:6.1f} peak_f={row.peak_frequency:6.1f} Hz -> {bundle}")

    (args.outdir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(indices)} fetched -> {args.outdir / 'sample_manifest.json'}")


if __name__ == "__main__":
    main()
