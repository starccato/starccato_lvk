"""Consolidate lnO (VAE) and BayesWave campaign results into one HDF5 file.

Scans the per-event JSON/`.dat` outputs scattered across
``RESULTS_ROOT/<lno_campaign>/rn_*/results/*.json`` and
``RESULTS_ROOT/<bw_campaign>/bw_fixedsky/e*/<class>/`` and writes a single
queryable HDF5 file with:

  /lno/<cohort>/<class>/            columnar evidence/QC arrays + raw_json
  /lno_baseline/<cohort>/<class>/   matched-filter SNR baseline (separate
                                     schema from the morphZ evidence results)
  /bayeswave/<campaign>/<class>/    columnar evidence arrays + raw_json
  /bayeswave/<campaign>/<class>/posteriors/e<index>/
                                     posterior waveform draws, whitened data,
                                     median reconstruction, time/frequency axes
                                     and the median signal-model PSD (kept only
                                     because plot_waveform_reconstruction.py
                                     whitens our VAE waveform through it)

Every row-oriented group also gets a ``raw_json`` variable-length-string
dataset holding the untouched source JSON, so nothing is lost even though
only a curated subset of fields is promoted to typed columns.

Usage:
    python studies/build_combined_results.py \
        --results-root /fred/oz303/avajpeyi/results/starccato_lvk \
        --lno-campaign nuts_morphlnz_v044 \
        --bw-campaigns bwcomp_nml_v044_ccsn:inj_ccsn bwcomp_nml_v044_glitch:real_glitch \
        --out /fred/oz303/avajpeyi/results/starccato_lvk/combined_results_nuts_morphlnz_v044.h5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

NAN = float("nan")


def _git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get(d: dict, *path, default=NAN):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def _write_vlen_json(group: h5py.Group, name: str, raw_jsons: list[str]) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.array(raw_jsons, dtype=object), dtype=dt,
                          compression="gzip", compression_opts=4)


def _write_cols(group: h5py.Group, cols: dict[str, list]) -> None:
    for key, values in cols.items():
        arr = np.asarray(values)
        if arr.dtype.kind in ("U", "S", "O"):
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(key, data=np.array(values, dtype=object), dtype=dt,
                                  compression="gzip", compression_opts=4)
        else:
            group.create_dataset(key, data=arr, compression="gzip", compression_opts=4)


# --------------------------------------------------------------------------
# lnO (VAE / morphZ evidence) results
# --------------------------------------------------------------------------

def collect_lno(results_root: Path, campaign: str, h5f: h5py.File) -> int:
    campaign_dir = results_root / campaign
    n_written = 0
    for results_dir in sorted(campaign_dir.glob("*/results")):
        cohort = results_dir.parent.name  # e.g. rn_H1_L1, rn_H1_blipH1
        by_class: dict[str, list[Path]] = {}
        for f in sorted(results_dir.glob("e*_*.json")):
            # filename: e<index>_<class>.json  (class may itself contain
            # underscores, e.g. inj_ccsn, inj_ccsn_baseline, real_glitch)
            stem = f.stem
            idx_str, _, cls = stem.partition("_")
            if not idx_str.startswith("e") or not idx_str[1:].isdigit():
                continue
            by_class.setdefault(cls, []).append(f)

        for cls, files in sorted(by_class.items()):
            is_baseline = cls.endswith("_baseline")
            rows: list[dict] = []
            raw_jsons: list[str] = []
            for f in sorted(files, key=lambda p: int(p.stem.split("_")[0][1:])):
                text = f.read_text()
                try:
                    d = json.loads(text)
                except json.JSONDecodeError:
                    continue
                rows.append(d)
                raw_jsons.append(text)
            if not rows:
                continue

            group_root = "lno_baseline" if is_baseline else "lno"
            grp = h5f.require_group(f"{group_root}/{cohort}/{cls}")

            if is_baseline:
                cols = {
                    "index": [int(_get(d, "index", default=-1)) for d in rows],
                    "mf_snr": [float(_get(d, "mf_snr")) for d in rows],
                    "new_snr": [float(_get(d, "new_snr")) for d in rows],
                    "mf_snr_H1": [float(_get(d, "per_det", "H1", "mf_snr")) for d in rows],
                    "mf_snr_L1": [float(_get(d, "per_det", "L1", "mf_snr")) for d in rows],
                }
            else:
                cols = {
                    "index": [int(_get(d, "index", default=-1)) for d in rows],
                    "blip_ifo": [str(_get(d, "blip_ifo", default="")) for d in rows],
                    "logZ_signal": [float(_get(d, "logZ_signal")) for d in rows],
                    "logZ_signal_err": [float(_get(d, "logZ_signal_err")) for d in rows],
                    "logZ_glitch": [float(_get(d, "logZ_glitch")) for d in rows],
                    "logZ_glitch_H1": [float(_get(d, "logZ_glitch_by_detector", "H1")) for d in rows],
                    "logZ_glitch_L1": [float(_get(d, "logZ_glitch_by_detector", "L1")) for d in rows],
                    "logZ_glitch_err_H1": [float(_get(d, "logZ_glitch_err_by_detector", "H1")) for d in rows],
                    "logZ_glitch_err_L1": [float(_get(d, "logZ_glitch_err_by_detector", "L1")) for d in rows],
                    "logZ_noise_H1": [float(_get(d, "logZ_noise_by_detector", "H1")) for d in rows],
                    "logZ_noise_L1": [float(_get(d, "logZ_noise_by_detector", "L1")) for d in rows],
                    "log_odds": [float(_get(d, "log_odds")) for d in rows],
                    "evidence_failures": [int(_get(d, "evidence_failures", default=0)) for d in rows],
                    "evidence_fallbacks": [int(_get(d, "evidence_fallbacks", default=0)) for d in rows],
                    "data_quality_H1_mean": [float(_get(d, "data_quality", "H1", "mean")) for d in rows],
                    "data_quality_L1_mean": [float(_get(d, "data_quality", "L1", "mean")) for d in rows],
                    "nuts_divergences_signal": [
                        int(_get(d, "nuts_diagnostics", "signal", "divergences", default=0)) for d in rows
                    ],
                    "manifest_fingerprint": [str(_get(d, "manifest_fingerprint", default="")) for d in rows],
                }
            _write_cols(grp, cols)
            _write_vlen_json(grp, "raw_json", raw_jsons)
            n_written += len(rows)
    return n_written


# --------------------------------------------------------------------------
# BayesWave results + posterior waveform draws
# --------------------------------------------------------------------------

def _load_dat(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return np.loadtxt(path, dtype=np.float32)
    except Exception:
        return None


def collect_bayeswave(results_root: Path, campaign: str, cls: str, h5f: h5py.File,
                       keep_posteriors: bool = True) -> int:
    campaign_dir = results_root / campaign / "bw_fixedsky"
    if not campaign_dir.is_dir():
        return 0
    event_dirs = sorted(
        (p for p in campaign_dir.glob(f"e*/{cls}") if (p / "result.json").is_file()),
        key=lambda p: int(p.parent.name[1:]),
    )
    if not event_dirs:
        return 0

    rows: list[dict] = []
    raw_jsons: list[str] = []
    indices: list[int] = []
    for d in event_dirs:
        text = (d / "result.json").read_text()
        try:
            r = json.loads(text)
        except json.JSONDecodeError:
            continue
        rows.append(r)
        raw_jsons.append(text)
        indices.append(int(d.parent.name[1:]))

    grp = h5f.require_group(f"bayeswave/{campaign}/{cls}")
    cols = {
        "index": indices,
        "logZ_signal": [float(_get(r, "logZ_signal")) for r in rows],
        "logZ_glitch": [float(_get(r, "logZ_glitch")) for r in rows],
        "logZ_noise": [float(_get(r, "logZ_noise")) for r in rows],
        "evidence_unc_signal": [float(_get(r, "evidence_uncertainty", "signal")) for r in rows],
        "evidence_unc_glitch": [float(_get(r, "evidence_uncertainty", "glitch")) for r in rows],
        "evidence_unc_noise": [float(_get(r, "evidence_uncertainty", "noise")) for r in rows],
        "log_bayeswave_signal_glitch": [float(_get(r, "log_bayeswave_signal_glitch")) for r in rows],
        "log_bayeswave_signal_glitch_unc": [
            float(_get(r, "log_bayeswave_signal_glitch_uncertainty")) for r in rows
        ],
        "target_snr": [float(_get(r, "target_snr")) for r in rows],
        "signal_reconstructed_snr_median": [
            float(_get(r, "signal_reconstructed_snr_median")) for r in rows
        ],
        "signal_reconstructed_snr_H1": [
            float(_get(r, "signal_reconstructed_snr_per_detector", "H1")) for r in rows
        ],
        "signal_reconstructed_snr_L1": [
            float(_get(r, "signal_reconstructed_snr_per_detector", "L1")) for r in rows
        ],
        "elapsed_seconds": [float(_get(r, "elapsed_seconds")) for r in rows],
    }
    _write_cols(grp, cols)
    _write_vlen_json(grp, "raw_json", raw_jsons)

    if keep_posteriors:
        post_grp = grp.require_group("posteriors")
        for d, idx in zip(event_dirs, indices):
            post_signal = d / "post" / "signal"
            post_root = d / "post"
            if not post_signal.is_dir():
                continue
            ev_grp = post_grp.require_group(f"e{idx}")
            for ifo in ("H1", "L1"):
                draws = _load_dat(post_signal / f"signal_recovered_whitened_waveform_{ifo}.dat")
                if draws is not None:
                    ev_grp.create_dataset(
                        f"whitened_waveform_draws_{ifo}", data=draws,
                        compression="gzip", compression_opts=4,
                    )
                data = _load_dat(post_root / f"whitened_data_{ifo}.dat")
                if data is not None:
                    ev_grp.create_dataset(
                        f"whitened_data_{ifo}", data=data,
                        compression="gzip", compression_opts=4,
                    )
                # The median signal-model PSD is not kept for its own sake: it is
                # what studies/plot_waveform_reconstruction.py whitens OUR VAE
                # waveform with, so the morphology overlay compares like with
                # like. Archiving it here is what makes it safe to delete the
                # post/ directories afterwards.
                psd = _load_dat(post_signal / f"signal_median_PSD_{ifo}.dat")
                if psd is not None:
                    ev_grp.create_dataset(
                        f"signal_median_psd_{ifo}", data=psd,
                        compression="gzip", compression_opts=4,
                    )
                # Median reconstruction: recoverable from the draws, but cheap
                # and it is what most summary figures actually plot.
                med = _load_dat(post_signal / f"signal_median_time_domain_waveform_{ifo}.dat")
                if med is not None:
                    ev_grp.create_dataset(
                        f"median_time_domain_waveform_{ifo}", data=med,
                        compression="gzip", compression_opts=4,
                    )
            for axis in ("timesamp", "freqsamp"):
                arr = _load_dat(post_root / f"{axis}.dat")
                if arr is not None:
                    ev_grp.create_dataset(axis, data=arr,
                                          compression="gzip", compression_opts=4)
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--lno-campaign", required=True)
    p.add_argument("--bw-campaigns", nargs="*", default=[],
                    help="campaign_dir:class pairs, e.g. bwcomp_nml_v044_ccsn:inj_ccsn")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-posteriors", action="store_true",
                    help="skip BayesWave posterior waveform draws (evidence tables only)")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_suffix(".h5.tmp")

    with h5py.File(tmp_out, "w") as h5f:
        h5f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5f.attrs["git_commit"] = _git_commit(repo)
        h5f.attrs["results_root"] = str(args.results_root)
        h5f.attrs["lno_campaign"] = args.lno_campaign
        h5f.attrs["bw_campaigns"] = json.dumps(args.bw_campaigns)
        h5f.attrs["schema"] = (
            "/lno/<cohort>/<class>/{index,logZ_*,log_odds,...,raw_json}; "
            "/lno_baseline/<cohort>/<class>/{index,mf_snr,new_snr,...}; "
            "/bayeswave/<campaign>/<class>/{index,logZ_*,log_bayeswave_signal_glitch,...,raw_json}; "
            "/bayeswave/<campaign>/<class>/posteriors/e<index>/"
            "{whitened_waveform_draws_*,whitened_data_*,signal_median_psd_*,"
            "median_time_domain_waveform_*,timesamp,freqsamp}"
        )

        n_lno = collect_lno(args.results_root, args.lno_campaign, h5f)
        print(f"lnO: {n_lno} rows written", file=sys.stderr)

        n_bw = 0
        for spec in args.bw_campaigns:
            campaign, _, cls = spec.partition(":")
            n = collect_bayeswave(args.results_root, campaign, cls, h5f,
                                   keep_posteriors=not args.no_posteriors)
            print(f"BayesWave {campaign}/{cls}: {n} rows written", file=sys.stderr)
            n_bw += n

        h5f.attrs["n_lno_rows"] = n_lno
        h5f.attrs["n_bayeswave_rows"] = n_bw

    tmp_out.replace(args.out)
    print(f"wrote {args.out} (lno={n_lno} rows, bayeswave={n_bw} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
