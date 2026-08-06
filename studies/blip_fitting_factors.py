"""Fitting factors of BOTH decoders against real cataloged blips.

For each of the first N cataloged blips in the host detector, fetch the real
strain at its GPS time, whiten with the production off-source PSD and
300-800 Hz band, and maximize the whitened match over the 5-D latent space for
the CCSN decoder and the blip decoder -- with the template additionally free to
slide in time (grid over +-25 ms), because this study asks a morphology
question, not a timing question.

Motivated by event 0, where at the blip's own arrival time the CCSN decoder
reached FF 0.72 against a real L1 blip while the blip decoder reached only
0.44: the blip prior carries essentially no power inside the 300-800 Hz
analysis band. This script decides whether that is the population rule or a
one-off.

Selection is mechanical (the first N catalogue indices, the same rule family
as the manuscript's three-class figure) and the outputs are one CSV row per
blip plus summary statistics.

    ./.venv/bin/python studies/blip_fitting_factors.py --n-events 32 \
        --outdir ../blip_ff_study
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from starccato_jax.waveforms import get_model
from starccato_lvk.acquisition.io.glitch_catalog import load_blip_glitch_catalog
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data

from real_noise_io import build_bundle

SAMPLE_RATE = 4096.0
FLOW, FMAX = 300.0, 800.0
WAVEFORM_N = 512
# +-25 ms covers the timing freedom SNR* enjoys on the scale of the burst
# duration; the FFT bins of the 4 s segment are ~0.24 ms apart at 4096 Hz.
TIME_SHIFT_MS = 25.0
N_STARTS = 48
N_OPT_STEPS = 500
LEARNING_RATE = 3e-2


def _whitened_target(bundle: Path, detector: str):
    """Whitened, unit-norm data plus everything needed to whiten templates."""
    prep = prepare_multi_detector_data(
        (detector,), bundle_paths={detector: bundle}, flow=FLOW, fmax=FMAX
    )
    d = prep.detector_data[detector]
    psd = np.asarray(d.psd_likelihood)
    x = np.asarray(d.windowed_strain)
    n = x.size
    dt = float(d.dt)
    df = 1.0 / (n * dt)
    w = jnp.asarray(np.where(np.isfinite(psd) & (psd > 0), psd, np.inf))

    def whiten(v):
        return jnp.fft.irfft(jnp.fft.rfft(v) * dt / jnp.sqrt(w / (4.0 * df)), n=n)

    target = whiten(jnp.asarray(x))
    target = target / jnp.linalg.norm(target)
    return whiten, target, n


def _fitting_factor(model_name: str, whiten, target, n: int, seed: int) -> float:
    """max over latent AND a time-shift grid of the whitened match."""
    decoder = get_model(model_name)
    pad = (n - WAVEFORM_N) // 2
    max_shift = int(TIME_SHIFT_MS * 1e-3 * SAMPLE_RATE)

    def match(z):
        wf = decoder.generate(z=z.reshape(1, 5))[0]
        h = whiten(jnp.pad(wf, (pad, n - WAVEFORM_N - pad)))
        hn = h / jnp.linalg.norm(h)
        # Correlate against the target over the shift grid; jnp.correlate with
        # mode="same" would correlate the full 16k series, but only +-max_shift
        # matters, so slide the target instead of the template.
        shifts = jnp.arange(-max_shift, max_shift + 1)
        rolled = jax.vmap(lambda s: jnp.roll(target, s))(shifts)
        return jnp.max(rolled @ hn)

    neg = jax.jit(jax.value_and_grad(lambda z: -match(z)))
    rng = np.random.default_rng(seed)
    best = -1.0
    for start in rng.normal(size=(N_STARTS, 5)).astype(np.float32):
        z = jnp.asarray(start)
        opt = optax.adam(LEARNING_RATE)
        state = opt.init(z)
        for _ in range(N_OPT_STEPS):
            _, g = neg(z)
            upd, state = opt.update(g, state)
            z = optax.apply_updates(z, upd)
        best = max(best, float(match(z)))
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ifo", default="L1", choices=["H1", "L1"])
    p.add_argument("--n-events", type=int, default=32)
    p.add_argument("--outdir", type=Path, default=Path("../blip_ff_study"))
    args = p.parse_args()

    catalog = load_blip_glitch_catalog(ifo=args.ifo)
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / f"blip_ff_{args.ifo}.csv"

    done = {}
    if csv_path.is_file():
        with open(csv_path) as f:
            done = {int(r["index"]): r for r in csv.DictReader(f)}

    rows = list(done.values())
    for index in range(min(args.n_events, len(catalog))):
        if index in done:
            continue
        gps = float(catalog.iloc[index]["event_time"])
        cat_snr = float(catalog.iloc[index]["snr"])
        try:
            bundle = build_bundle(
                gps, args.outdir / f"bundles/e{index}/{args.ifo}", args.ifo
            )
            whiten, target, n = _whitened_target(bundle, args.ifo)
            row = {
                "index": index,
                "gps": gps,
                "catalog_snr": cat_snr,
                "ff_ccsne": round(_fitting_factor("ccsne", whiten, target, n, seed=index), 4),
                "ff_blip": round(_fitting_factor("blip", whiten, target, n, seed=index), 4),
            }
        except Exception as exc:  # acquisition failures are data, not crashes
            print(f"[e{index}] FAILED: {exc}")
            row = {"index": index, "gps": gps, "catalog_snr": cat_snr,
                   "ff_ccsne": "", "ff_blip": ""}
        rows.append(row)
        print(f"[e{index}] snr={cat_snr:6.1f} ff_ccsne={row['ff_ccsne']} ff_blip={row['ff_blip']}")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    ok = [r for r in rows if r["ff_ccsne"] != ""]
    ffc = np.array([float(r["ff_ccsne"]) for r in ok])
    ffb = np.array([float(r["ff_blip"]) for r in ok])
    summary = {
        "ifo": args.ifo,
        "n_attempted": len(rows),
        "n_ok": len(ok),
        "ff_ccsne": {"median": float(np.median(ffc)),
                     "p5": float(np.percentile(ffc, 5)),
                     "p95": float(np.percentile(ffc, 95))},
        "ff_blip": {"median": float(np.median(ffb)),
                    "p5": float(np.percentile(ffb, 5)),
                    "p95": float(np.percentile(ffb, 95))},
        "n_ccsn_fits_better": int(np.sum(ffc > ffb)),
    }
    (args.outdir / f"summary_{args.ifo}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
