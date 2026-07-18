"""Classic-search baseline for the real-noise study: reweighted matched-filter SNR.

Raw SNR loses to the odds by construction (glitches are loud), which invites the
"straw-man baseline" review. This computes what real searches actually rank by:
matched-filter SNR against a template bank (held-in CCSN training waveforms),
vetoed by an Allen frequency-band chi-squared and reweighted CBC-style,

    newSNR = rho / [(1 + chisq_r^3) / 2]^(1/6)   (for chisq_r > 1, else rho).

No sampling -- it re-reads the per-event bundles written by real_noise_event.py
(prep stage) and adds results/e{i}_{cls}_baseline.json rows next to the odds rows.

    # per-event (SLURM-array friendly, idempotent):
    uv run python studies/chisq_baseline.py --outdir slurm/out/rn_L1 --index 0
    # everything under an outdir:
    uv run python studies/chisq_baseline.py --outdir slurm/out/rn_L1 --all
    # AUCs of newSNR (block bootstrap), for the manuscript comparison:
    uv run python studies/chisq_baseline.py --outdir slurm/out/rn_L1 --aggregate
    # synthetic self-check (no data needed):
    uv run python studies/chisq_baseline.py --self-test
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Core statistics (pure numpy; exercised by --self-test)
# ---------------------------------------------------------------------------


def matched_filter(data_fd: np.ndarray, h_fd: np.ndarray, psd: np.ndarray,
                   df: float, dt: float, max_shift_s: float):
    """Time-maximized matched filter of one real template against one detector.

    Returns (rho, k_hat): the SNR |z(t)|/sqrt(<h|h>) maximized over shifts
    |t| <= max_shift_s, and the argmax shift in samples (wrapped).
    """
    w = np.where(np.isfinite(psd), 1.0 / psd, 0.0)
    hh = 4.0 * df * np.sum(np.abs(h_fd) ** 2 * w)
    if hh <= 0:
        return 0.0, 0
    x = data_fd * np.conj(h_fd) * w
    z = (2.0 / dt) * np.fft.irfft(x)  # z[k] = <d|h shifted by k samples>
    n = z.size
    k_win = max(1, int(round(max_shift_s / dt)))
    allowed = np.r_[0:k_win + 1, n - k_win:n]
    k_hat = allowed[np.argmax(np.abs(z[allowed]))]
    return float(np.abs(z[k_hat]) / np.sqrt(hh)), int(k_hat)


def allen_chisq_r(data_fd: np.ndarray, h_fd: np.ndarray, psd: np.ndarray,
                  df: float, k_hat: int, p: int = 8) -> float:
    """Reduced Allen chi-squared (p frequency bands, dof = p-1) at shift k_hat."""
    w = np.where(np.isfinite(psd), 1.0 / psd, 0.0)
    hw = np.abs(h_fd) ** 2 * w
    hh = 4.0 * df * np.sum(hw)
    live = np.flatnonzero(hw > 0)
    if hh <= 0 or live.size < 2 * p:
        return np.inf
    # band edges: equal cumulative template power
    cum = np.cumsum(hw[live])
    edges = np.searchsorted(cum, np.linspace(0, cum[-1], p + 1)[1:-1])
    bands = np.split(live, edges)
    n = (data_fd.size - 1) * 2
    j = np.arange(data_fd.size)
    x = data_fd * np.conj(h_fd) * w * np.exp(2j * np.pi * j * k_hat / n)
    z_l = np.array([4.0 * df * np.sum(x[b]).real for b in bands])
    hh_l = np.array([4.0 * df * np.sum(hw[b]) for b in bands])
    z = z_l.sum()
    chisq = np.sum((z_l - z * hh_l / hh) ** 2 / hh_l)
    return float(chisq / (p - 1))


def reweight(rho: float, chisq_r: float) -> float:
    """CBC-style reweighted SNR."""
    if not np.isfinite(chisq_r):
        return 0.0
    return rho / ((1.0 + chisq_r ** 3) / 2.0) ** (1.0 / 6.0) if chisq_r > 1.0 else rho


def bank_newsnr(data_fd: np.ndarray, bank_fd: np.ndarray, psd: np.ndarray,
                df: float, dt: float, max_shift_s: float = 0.1, p: int = 8):
    """Best-template (max rho) newSNR for one detector. Returns (newsnr, rho, chisq_r)."""
    best = (0.0, 0, None)  # rho, k_hat, h_fd
    for h_fd in bank_fd:
        rho, k = matched_filter(data_fd, h_fd, psd, df, dt, max_shift_s)
        if rho > best[0]:
            best = (rho, k, h_fd)
    rho, k_hat, h_fd = best
    if h_fd is None:
        return 0.0, 0.0, np.inf
    cr = allen_chisq_r(data_fd, h_fd, psd, df, k_hat, p=p)
    return reweight(rho, cr), rho, cr


# ---------------------------------------------------------------------------
# Event runner (reuses the real_noise_event bundle/manifest layout)
# ---------------------------------------------------------------------------

CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")


def _template_bank_fd(n_seg: int, dt: float, n_templates: int, seed: int = 42) -> np.ndarray:
    """FD bank from the CCSN *training* split (the search knows only its templates)."""
    from starccato_jax.data.training_data import TrainValData
    from snr_vs_odds_roc import _center_pad

    pool = np.asarray(TrainValData.load(source="ccsne", seed=0).train)
    rng = np.random.default_rng(seed)
    sel = rng.choice(pool.shape[0], min(n_templates, pool.shape[0]), replace=False)
    return np.stack([np.fft.rfft(_center_pad(pool[i], n_seg)) * dt for i in sel])


def run_index(index: int, outdir: Path, n_templates: int, p: int) -> None:
    from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data
    from snr_vs_odds_roc import SAMPLE_RATE

    manifest = json.loads((outdir / f"e{index}" / "manifest.json").read_text())
    detectors = manifest["detectors"]
    flow, fmax = manifest["band"]
    dt = 1.0 / SAMPLE_RATE
    n_seg = int(round(4.0 * SAMPLE_RATE))
    bank = _template_bank_fd(n_seg, dt, n_templates)
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        if cls not in manifest["bundles"]:
            continue  # production manifests prepare only 3 of the 4 classes
        out_json = results_dir / f"e{index}_{cls}_baseline.json"
        if out_json.exists():
            continue  # idempotent (SLURM re-runs)
        prep = prepare_multi_detector_data(
            detectors, bundle_paths=manifest["bundles"][cls], flow=flow, fmax=fmax)
        per_det = {}
        for d in detectors:
            det = prep.detector_data[d]
            ns, rho, cr = bank_newsnr(
                np.asarray(det.data_fd_likelihood), bank,
                np.asarray(det.psd_likelihood), det.df, det.dt, p=p)
            per_det[d] = {"new_snr": ns, "mf_snr": rho, "chisq_r": cr if np.isfinite(cr) else None}
        row = {
            "index": index, "cls": cls, "detectors": detectors,
            "new_snr": float(np.sqrt(sum(v["new_snr"] ** 2 for v in per_det.values()))),
            "mf_snr": float(np.sqrt(sum(v["mf_snr"] ** 2 for v in per_det.values()))),
            "per_det": per_det,
        }
        tmp = out_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row, indent=2))
        tmp.replace(out_json)  # atomic: per-class SLURM tasks may race here
        print(f"[e{index} {cls:>11s}] mf_snr={row['mf_snr']:6.2f} new_snr={row['new_snr']:6.2f}")


def aggregate(outdir: Path) -> None:
    from snr_vs_odds_roc import _roc_auc
    from real_noise_aggregate import _boot_auc_err

    rows = [json.loads(Path(f).read_text())
            for f in glob.glob(str(outdir / "results" / "e*_baseline.json"))]
    if not rows:
        print(f"No baseline JSONs under {outdir/'results'}.")
        return
    val = {c: np.array([r["new_snr"] for r in rows if r["cls"] == c]) for c in CLASSES}
    idx = {c: np.array([r["index"] for r in rows if r["cls"] == c], dtype=int) for c in CLASSES}
    bg = np.concatenate([val["noise"], val["inj_glitch"], val["real_glitch"]])
    bg_idx = np.concatenate([idx["noise"], idx["inj_glitch"], idx["real_glitch"]])
    summary = {
        "n": {c: int(val[c].size) for c in CLASSES},
        "auc_newsnr_signal_vs_background": _roc_auc(val["inj_ccsn"], bg),
        "auc_newsnr_signal_vs_background_err": _boot_auc_err(val["inj_ccsn"], bg, idx["inj_ccsn"], bg_idx),
        "auc_newsnr_signal_vs_real_glitch": _roc_auc(val["inj_ccsn"], val["real_glitch"]),
        "auc_newsnr_signal_vs_real_glitch_err": _boot_auc_err(val["inj_ccsn"], val["real_glitch"],
                                                              idx["inj_ccsn"], idx["real_glitch"]),
    }
    (outdir / "summary_baseline.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Self-test: synthetic bank + white PSD, no external data
# ---------------------------------------------------------------------------


def self_test() -> None:
    rng = np.random.default_rng(0)
    fs, T = 4096.0, 4.0
    n = int(fs * T)
    dt, df = 1.0 / fs, 1.0 / T
    freq = np.fft.rfftfreq(n, dt)
    psd = np.where((freq >= 300) & (freq <= 800), 1.0, np.inf)

    def damped_sine(f0, tau):  # crude CCSN-ish ring-down, centered
        t = (np.arange(n) - n // 2) * dt
        return np.exp(-np.abs(t) / tau) * np.sin(2 * np.pi * f0 * t)

    bank_td = [damped_sine(f0, 0.01) for f0 in (450, 550, 650, 750)]
    bank = np.stack([np.fft.rfft(h) * dt for h in bank_td])

    def noise_fd():
        sig = np.sqrt(np.where(np.isfinite(psd), psd, 0.0) / (4 * df))
        return sig * (rng.standard_normal(freq.size) + 1j * rng.standard_normal(freq.size))

    # 1) on-bank injection at SNR 20: recovered, chi2 ~ 1, newSNR ~ rho
    h = bank[1]
    hh = 4 * df * np.sum(np.abs(h) ** 2 / np.where(np.isfinite(psd), psd, np.inf))
    d = noise_fd() + (20.0 / np.sqrt(hh)) * h
    ns, rho, cr = bank_newsnr(d, bank, psd, df, dt)
    assert 15 < rho < 25, f"on-bank rho={rho}"
    assert cr < 2.5, f"on-bank chisq_r={cr}"
    assert ns > 0.9 * rho, f"on-bank newSNR={ns} vs rho={rho}"

    # 2) broadband white burst at the same amplitude scale: chi2 kills it
    burst = np.zeros(n)
    burst[n // 2 - 64:n // 2 + 64] = rng.standard_normal(128)
    b = np.fft.rfft(burst) * dt
    bb = 4 * df * np.sum(np.abs(b) ** 2 / np.where(np.isfinite(psd), psd, np.inf))
    d2 = noise_fd() + (20.0 / np.sqrt(bb)) * b
    ns2, rho2, cr2 = bank_newsnr(d2, bank, psd, df, dt)
    assert cr2 > 2.5, f"off-bank chisq_r={cr2} (expected >> 1)"
    assert ns2 < 0.8 * rho2, f"off-bank newSNR={ns2} not down-weighted vs rho={rho2}"

    # 3) pure noise: small rho
    ns3, rho3, _ = bank_newsnr(noise_fd(), bank, psd, df, dt)
    assert rho3 < 6, f"noise rho={rho3}"
    print(f"self-test OK: on-bank (rho={rho:.1f}, chi2r={cr:.2f}, new={ns:.1f}); "
          f"burst (rho={rho2:.1f}, chi2r={cr2:.2f}, new={ns2:.1f}); noise rho={rho3:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=Path("out_rn"))
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="Run every e*/manifest.json under outdir.")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--n-templates", type=int, default=128)
    ap.add_argument("--chisq-bands", type=int, default=8)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
    elif args.aggregate:
        aggregate(args.outdir)
    elif args.all:
        for m in sorted(glob.glob(str(args.outdir / "e*" / "manifest.json"))):
            run_index(int(Path(m).parent.name[1:]), args.outdir, args.n_templates, args.chisq_bands)
    elif args.index is not None:
        run_index(args.index, args.outdir, args.n_templates, args.chisq_bands)
    else:
        ap.error("one of --index / --all / --aggregate / --self-test required")


if __name__ == "__main__":
    main()
