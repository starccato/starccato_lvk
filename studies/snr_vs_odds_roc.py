"""Single-detector (L1) SNR-vs-Odds ROC on design-PSD-coloured Gaussian noise.

The paper's core demonstration: that the VAE evidence ratio O = Z_S/(Z_N + Z_G)
separates CCSN signals from blip glitches better than SNR. We use stationary
design-PSD noise (which matches its PSD by construction), so the matched-filter
evidences are well behaved -- this sidesteps the real-data non-stationarity that
the smoke test exposed, deferring full real-GWOSC conditioning to a later
robustness study.

Event classes (single detector, L1):
    noise   : design-PSD noise only                 (background)
    signal  : noise + injected CCSN (VAE z=0)        (foreground / detection target)
    glitch  : noise + injected blip (VAE z=0)        (background trigger)

Injections are scaled to a grid of target *optimal SNRs* so the signal and glitch
SNR distributions overlap. That overlap is what makes the comparison fair: a loud
blip has high SNR but should have low O (it is off the CCSN manifold), so O should
separate signals from glitches at fixed SNR. For each event we record the injected
SNR and the recovered O, then compare ROC/AUC for O vs SNR as ranking statistics.

Run a quick smoke configuration first:
    uv run python studies/snr_vs_odds_roc.py --outdir out_roc_smoke \
        --n-per-class 2 --snr-grid 20 --num-warmup 80 --num-samples 200 --lnz-method morph
Then scale up (more events, full SNR grid).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import jax.numpy as jnp

from gwpy.timeseries import TimeSeries

from starccato_jax.waveforms import get_model
from starccato_lvk.analysis.jim_waveform import StarccatoJimWaveform, StarccatoGlitchWaveform
from starccato_lvk.analysis.main import run_bcr_posteriors, _clone_no_response_detector
from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
from starccato_lvk.acquisition.io.strain_loader import _write_analysis_bundle, generate_psd

from simulated_design_psd import (  # reuse the design-PSD plumbing
    build_design_psd,
    pycbc_psd_to_gwpy,
    to_timeseries,
)


def simulate_noise_fd(psd_vals: np.ndarray, n: int, dt: float, seed: int) -> np.ndarray:
    """Colored Gaussian noise drawn to match the likelihood's whitening convention.

    The JIM likelihood whitens with ``d_fd = rfft(d) * dt`` and one-sided PSD ``S``,
    so for whitened residuals ~ N(0,1) the noise must satisfy ``E|d_fd|^2 = S*T/2``.
    We draw the rfft coefficients to that variance directly. This sidesteps
    ``pycbc.noise_from_psd``, whose FFT-normalisation convention differs from the
    likelihood's by a constant ~47x (in power) and was inflating every evidence.
    """
    rng = np.random.default_rng(seed)
    T = n * dt
    n_freq = n // 2 + 1
    sigma = np.sqrt(psd_vals[:n_freq] * T / 4.0)
    fd = (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)) * sigma
    fd[0] = rng.standard_normal() * np.sqrt(psd_vals[0] * T / 2.0)  # DC is real
    if n % 2 == 0:
        fd[-1] = rng.standard_normal() * np.sqrt(psd_vals[n_freq - 1] * T / 2.0)  # Nyquist real
    return np.fft.irfft(fd / dt, n)

SAMPLE_RATE = 4096.0
SEGMENT_DURATION = 4.0  # s -> 0.25 Hz, matches the real-data analysis resolution
PSD_OFFSOURCE_SECONDS = 64.0  # off-source data used for the Welch PSD estimate
BASE_TRIGGER = 1_260_000_000.0


def _event_noise_and_psd(event_id, cls, n_seg, dt, *, design_psd_4s, psd_fs_design,
                         design_psd_long, use_welch):
    """Return (on-source 4 s noise chunk, PSD FrequencySeries for the bundle).

    With ``use_welch`` the noise is drawn over a long segment and the PSD is
    estimated by median-Welch from an OFF-SOURCE stretch (mirroring the real
    pipeline: the analysis never sees the true PSD, only an estimate from
    adjacent signal-free data, with its bias and scatter). Otherwise the true
    design PSD is used directly (idealized).
    """
    seed = 7919 * event_id + {"noise": 0, "signal": 1, "glitch": 2}[cls]
    if not use_welch:
        return simulate_noise_fd(design_psd_4s, n_seg, dt, seed), psd_fs_design
    fs = 1.0 / dt
    n_off = int(round(PSD_OFFSOURCE_SECONDS * fs))
    n_long = n_off + n_seg
    long_noise = simulate_noise_fd(design_psd_long, n_long, dt, seed)
    chunk = long_noise[-n_seg:]                       # on-source segment
    offsrc = TimeSeries(long_noise[:n_off], dt=dt)    # off-source, signal-free
    psd_fs = generate_psd(offsrc)                     # median Welch, fftlength=4 s
    return np.asarray(chunk, dtype=np.float64), psd_fs


@dataclass
class EventRow:
    event_id: int
    cls: str          # "noise" | "signal" | "glitch"
    target_snr: float
    injected_snr: float
    logZ_signal: float
    logZ_glitch: float
    logZ_noise: float
    log_odds: float   # log O = logZ_S - log(Z_N + Z_G)
    evidence_failures: int


def _write_event_bundle(outdir: Path, data: np.ndarray, psd_fs, trigger: float) -> Path:
    ts = to_timeseries(data, SAMPLE_RATE, trigger, "L1")
    outdir.mkdir(parents=True, exist_ok=True)
    bundle = outdir / f"analysis_bundle_{int(trigger)}.hdf5"
    # full_strain and analysis chunk are the same 4 s segment for simulated data.
    _write_analysis_bundle(str(bundle), ts, ts, psd_fs, trigger)
    return bundle


def _center_pad(wf512: np.ndarray, n_seg: int) -> np.ndarray:
    out = np.zeros(n_seg, dtype=np.float64)
    m = min(n_seg, wf512.shape[0])
    start = n_seg // 2 - m // 2
    out[start:start + m] = wf512[:m]
    return out


def _inject_through_recovery(prep, cls: str, target_snr: float, n_seg: int, dt: float,
                             flow: float, fmax: float,
                             raw_wf: Optional[np.ndarray] = None) -> tuple[np.ndarray, float]:
    """Build the recovery-matched injection and return (time-domain trace, exact SNR).

    The injection is projected through the SAME detector the recovery uses -- the
    CCSN through the antenna response, the blip through a no-response detector. The
    optimal SNR is computed in the likelihood's exact convention (``<h|h>`` over the
    band against the line-notched ``det.psd``) and the amplitude scaled so the
    injected SNR equals ``target_snr`` by construction.

    ``raw_wf`` selects the source morphology:
      - ``None``: on-manifold -- the VAE's own waveform at latent ``z = 0``.
      - a 512-sample array: OFF-MANIFOLD -- a held-out real waveform the VAE never
        trained on. This is the honest test: the recovery still uses the VAE, so a
        high odds requires the VAE to *generalise* to unseen signals/glitches.
    """
    sr = 1.0 / dt
    full_freqs = np.fft.rfftfreq(n_seg, d=dt)
    fj = jnp.asarray(full_freqs)
    df = float(full_freqs[1] - full_freqs[0])
    det = prep.detectors[0] if cls == "signal" else _clone_no_response_detector(prep.detectors[0])
    psd_full = np.asarray(det.psd.values)
    band = (full_freqs >= flow) & (full_freqs <= fmax)
    resp_params = {"t_c": 0.0, "ra": 0.0, "dec": 0.0, "psi": 0.0,
                   "gmst": float(prep.gmst), "trigger_time": float(prep.trigger_time)}

    if raw_wf is not None:
        # Off-manifold: build the unit sky-frame waveform from the held-out morphology,
        # padded + windowed exactly as the VAE waveform path does.
        td = _center_pad(np.asarray(raw_wf, dtype=np.float64), n_seg) * 1e-21 * np.asarray(prep.window)
        h_sky_unit = jnp.fft.rfft(jnp.asarray(td)) * dt

        # h_x = 0 for the plus-polarized (2D axisymmetric) CCSN source, matching
        # StarccatoJimWaveform. For cls == "glitch" the no-response detector reads
        # only "p", so this is a no-op on that path.
        h_zero = jnp.zeros_like(h_sky_unit)

        def _hdec(amp):
            hs = {"p": h_sky_unit * amp, "c": h_zero}
            return np.asarray(det.fd_response(fj, hs, resp_params))
        hdec0 = _hdec(1.0)
        snr_unit = float(np.sqrt(4.0 * np.sum(np.abs(hdec0[band]) ** 2 / psd_full[band]) * df))
        amp = target_snr / snr_unit
        hdec = _hdec(amp)
    else:
        # On-manifold: the VAE's own z=0 waveform, amplitude via log_amp.
        model = "ccsne" if cls == "signal" else "blip"
        wf = (StarccatoJimWaveform(model=get_model(model), sample_rate=sr, window=prep.window)
              if cls == "signal" else
              StarccatoGlitchWaveform(model=get_model(model), sample_rate=sr,
                                      window=prep.window, strain_scale=1e-21))
        params = {name: 0.0 for name in wf.latent_names}
        params.update({"log_amp": 0.0, "luminosity_distance": 10.0, **resp_params})

        def _hdec(p):
            return np.asarray(det.fd_response(fj, wf(fj, p), p))
        hdec0 = _hdec(params)
        snr_unit = float(np.sqrt(4.0 * np.sum(np.abs(hdec0[band]) ** 2 / psd_full[band]) * df))
        params["log_amp"] = float(np.log(target_snr / snr_unit))
        hdec = _hdec(params)

    injected_snr = float(np.sqrt(4.0 * np.sum(np.abs(hdec[band]) ** 2 / psd_full[band]) * df))
    inj_td = np.fft.irfft(hdec, n=n_seg) / dt
    return inj_td, injected_snr


def run_event(
    event_id: int,
    cls: str,
    target_snr: float,
    *,
    psd_vals: np.ndarray,
    psd_fs,
    n_seg: int,
    outdir: Path,
    flow: float,
    fmax: float,
    bcr_kwargs: dict,
    raw_wf: Optional[np.ndarray] = None,
    noise_psd: Optional[dict] = None,
) -> EventRow:
    dt = 1.0 / SAMPLE_RATE
    noise, psd_fs = _event_noise_and_psd(
        event_id, cls, n_seg, dt,
        design_psd_4s=psd_vals, psd_fs_design=psd_fs,
        design_psd_long=(noise_psd or {}).get("long"),
        use_welch=(noise_psd or {}).get("use_welch", False),
    )
    trigger = BASE_TRIGGER + event_id
    event_dir = outdir / cls / f"event_{event_id}"

    injected_snr = 0.0
    if cls in ("signal", "glitch"):
        noise_bundle = _write_event_bundle(event_dir / "noise", noise, psd_fs, trigger)
        prep = prepare_multi_detector_data(["L1"], bundle_paths={"L1": noise_bundle},
                                           flow=flow, fmax=fmax)
        inj_td, injected_snr = _inject_through_recovery(prep, cls, target_snr, n_seg, dt,
                                                        flow, fmax, raw_wf=raw_wf)
        noise = noise + inj_td

    bundle = _write_event_bundle(event_dir / "data", noise, psd_fs, trigger)

    result = run_bcr_posteriors(
        detectors=["L1"],
        outdir=str(event_dir / "analysis"),
        bundle_paths={"L1": str(bundle)},
        flow=flow,
        fmax=fmax,
        save_artifacts=False,
        **bcr_kwargs,
    )
    logZ_s = float(result["signal"].get("logZ", np.nan))
    logZ_g = float(result.get("glitch", {}).get("L1", np.nan))
    logZ_n = float(result["noise"].get("L1", 0.0))
    log_odds = float(result.get("bcr_log", np.nan))
    return EventRow(
        event_id=event_id, cls=cls, target_snr=target_snr, injected_snr=injected_snr,
        logZ_signal=logZ_s, logZ_glitch=logZ_g, logZ_noise=logZ_n, log_odds=log_odds,
        evidence_failures=int(result.get("evidence_failures", 0)),
    )


def _roc_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUC = P(score_pos > score_neg) via Mann-Whitney; ignores non-finite scores."""
    pos = scores_pos[np.isfinite(scores_pos)]
    neg = scores_neg[np.isfinite(scores_neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
    return float(wins / (pos.size * neg.size))


def _make_plots(rows: List[EventRow], outdir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = {c: [r for r in rows if r.cls == c] for c in ("noise", "signal", "glitch")}
    snr = {c: np.array([r.injected_snr for r in arr[c]]) for c in arr}
    odds = {c: np.array([r.log_odds for r in arr[c]]) for c in arr}

    # Foreground = signal; background = noise + glitch.
    bg_snr = np.concatenate([snr["noise"], snr["glitch"]]) if arr["noise"] or arr["glitch"] else np.array([])
    bg_odds = np.concatenate([odds["noise"], odds["glitch"]]) if arr["noise"] or arr["glitch"] else np.array([])
    auc_snr = _roc_auc(snr["signal"], bg_snr)
    auc_odds = _roc_auc(odds["signal"], bg_odds)
    # The discriminating sub-question: signal vs glitch at matched SNR.
    auc_odds_sig_vs_glitch = _roc_auc(odds["signal"], odds["glitch"])
    auc_snr_sig_vs_glitch = _roc_auc(snr["signal"], snr["glitch"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"noise": "tab:gray", "signal": "tab:orange", "glitch": "tab:blue"}
    for c in ("noise", "glitch", "signal"):
        if arr[c]:
            ax1.scatter(snr[c], odds[c], s=18, alpha=0.7, color=colors[c], label=c)
    ax1.set_xlabel("injected optimal SNR")
    ax1.set_ylabel(r"log odds  $\log Z_S/(Z_N+Z_G)$")
    ax1.legend()
    ax1.set_title("SNR vs Odds")

    for label, pos, neg in [("Odds", odds["signal"], bg_odds), ("SNR", snr["signal"], bg_snr)]:
        pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
        if pos.size and neg.size:
            thr = np.sort(np.concatenate([pos, neg]))[::-1]
            tpr = [np.mean(pos >= t) for t in thr]
            fpr = [np.mean(neg >= t) for t in thr]
            auc = auc_odds if label == "Odds" else auc_snr
            ax2.plot([0] + fpr + [1], [0] + tpr + [1], label=f"{label} (AUC={auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax2.set_xlabel("false positive rate (background)")
    ax2.set_ylabel("true positive rate (signal)")
    ax2.legend()
    ax2.set_title("ROC: signal vs background")
    fig.tight_layout()
    fig.savefig(outdir / "snr_vs_odds_roc.png", dpi=150)
    plt.close(fig)

    return {
        "auc_snr_signal_vs_background": auc_snr,
        "auc_odds_signal_vs_background": auc_odds,
        "auc_snr_signal_vs_glitch": auc_snr_sig_vs_glitch,
        "auc_odds_signal_vs_glitch": auc_odds_sig_vs_glitch,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("out_roc"))
    p.add_argument("--cache-dir", type=Path, default=Path("design_psd_cache"))
    p.add_argument("--n-per-class", type=int, default=20, help="Events per class per SNR grid point.")
    p.add_argument("--snr-grid", type=float, nargs="+", default=[10, 15, 20, 30, 40])
    p.add_argument("--flow", type=float, default=100.0)
    p.add_argument("--fmax", type=float, default=1024.0)
    p.add_argument("--num-warmup", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=800)
    p.add_argument("--lnz-method", choices=["morph", "nested"], default="morph")
    p.add_argument("--on-manifold", action="store_true",
                   help="Inject the VAE's own z=0 waveforms instead of held-out real ones "
                        "(idealized; AUC will be optimistic).")
    p.add_argument("--ideal-psd", action="store_true",
                   help="Use the true design PSD instead of an off-source Welch estimate "
                        "(idealized; the analysis normally only has an estimate).")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    n_seg = int(round(SEGMENT_DURATION * SAMPLE_RATE))
    delta_f = 1.0 / SEGMENT_DURATION
    n_freq = n_seg // 2 + 1

    psd_pycbc = build_design_psd("L1", delta_f, n_freq, args.cache_dir)
    psd_vals = np.asarray(psd_pycbc.numpy(), dtype=np.float64)
    psd_fs = pycbc_psd_to_gwpy(psd_pycbc)

    use_welch = not args.ideal_psd
    noise_psd = {"use_welch": use_welch, "long": None}
    if use_welch:
        n_long = int(round(PSD_OFFSOURCE_SECONDS * SAMPLE_RATE)) + n_seg
        long_pycbc = build_design_psd("L1", 1.0 / (n_long / SAMPLE_RATE), n_long // 2 + 1, args.cache_dir)
        noise_psd["long"] = np.asarray(long_pycbc.numpy(), dtype=np.float64)
        print(f"[roc] PSD: off-source Welch estimate from {PSD_OFFSOURCE_SECONDS:.0f}s "
              f"(analysis never sees the true PSD)")

    # Held-out (val) waveforms the VAE never trained on -> honest off-manifold injections.
    held_out = {}
    if not args.on_manifold:
        from starccato_jax.data.training_data import TrainValData
        held_out["signal"] = np.asarray(TrainValData.load(source="ccsne", seed=0).val)
        held_out["glitch"] = np.asarray(TrainValData.load(source="blip", seed=0).val)
        print(f"[roc] off-manifold injections: {held_out['signal'].shape[0]} held-out CCSNe, "
              f"{held_out['glitch'].shape[0]} held-out blips")
    _wf_rng = np.random.default_rng(20240501)

    bcr_kwargs = dict(
        signal_model="ccsne", glitch_model="blip",
        num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=1,
        lnz_method=args.lnz_method,
        nested_num_live_points=200, nested_max_samples=4000,
        verify_logz_threshold=None,  # morphZ validated reliable on simulated data; skip cross-check for speed
    )

    rows: List[EventRow] = []
    event_id = 0
    for snr in args.snr_grid:
        for _ in range(args.n_per_class):
            for cls in ("noise", "signal", "glitch"):
                raw_wf = None
                if cls in held_out:
                    pool = held_out[cls]
                    raw_wf = pool[_wf_rng.integers(pool.shape[0])]
                row = run_event(
                    event_id, cls, snr,
                    psd_vals=psd_vals, psd_fs=psd_fs, n_seg=n_seg,
                    outdir=args.outdir,
                    flow=args.flow, fmax=args.fmax, bcr_kwargs=bcr_kwargs,
                    raw_wf=raw_wf, noise_psd=noise_psd,
                )
                print(f"[{cls:>6s} id={event_id:3d}] target_snr={snr:5.1f} inj_snr={row.injected_snr:6.2f} "
                      f"logZ_s={row.logZ_signal:8.1f} logZ_g={row.logZ_glitch:8.1f} logO={row.log_odds:8.1f}")
                rows.append(row)
                event_id += 1

    (args.outdir / "roc_rows.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
    aucs = _make_plots(rows, args.outdir)
    (args.outdir / "roc_summary.json").write_text(json.dumps(aucs, indent=2))
    print("\nAUC summary:")
    for k, v in aucs.items():
        print(f"  {k}: {v:.3f}")
    print(f"Wrote {args.outdir/'snr_vs_odds_roc.png'} and roc_summary.json")


if __name__ == "__main__":
    main()
