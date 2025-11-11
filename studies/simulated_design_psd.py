"""Simulate design-PSD noise and run Starccato analysis scenarios.

This script downloads the official O3 design ASDs, uses them to color
Gaussian noise for each detector, optionally injects coherent CCSN signals
or incoherent blip glitches, writes Starccato analysis bundles, and runs
the standard BCR analysis on the synthetic data.

It mirrors the three scenarios used on real data:

1. noise-only
2. noise with coherent CCSN injection across detectors
3. noise with a blip glitch injected in a single detector

Example:
    PYTHONPATH=lvk/src ./starccato_venv/bin/python lvk/studies/simulated_design_psd.py \
        --outdir lvk/studies/out_sim_psd \
        --detector H1 --detector L1 \
        --signal-distance 2.0 \
        --glitch-detector H1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import requests
from astropy import units as u
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from pycbc.noise.gaussian import noise_from_psd
from pycbc.types import FrequencySeries as PycbcFrequencySeries

from starccato_jax.waveforms import get_model

from starccato_lvk.analysis.jim_waveform import StarccatoJimWaveform
from starccato_lvk.analysis.main import run_bcr_posteriors
from starccato_lvk.acquisition.io.strain_loader import _write_analysis_bundle


ASD_URLS = {
    "H1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt",
    "L1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
    "V1": "https://dcc.ligo.org/public/0169/P2000251/001/O3-V1_sensitivity_strain_asd.txt",
}

DEFAULT_SAMPLE_RATE = 4096.0
DEFAULT_FULL_DURATION = 8.0  # seconds (sets delta_f = 0.125 Hz)
DEFAULT_CHUNK_SAMPLES = 512  # matches real-data analysis window
BASE_TRIGGER = 1_260_000_000.0


def download_asd(det: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = ASD_URLS[det]
    dest = cache_dir / Path(url).name
    if dest.exists():
        return dest
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def build_design_psd(det: str, delta_f: float, n_freq: int, cache_dir: Path) -> PycbcFrequencySeries:
    asd_path = download_asd(det, cache_dir)
    freq, asd = np.loadtxt(asd_path, unpack=True)
    psd_vals = np.square(asd)
    freq_grid = np.arange(n_freq) * delta_f
    interp = np.interp(freq_grid, freq, psd_vals, left=psd_vals[0], right=psd_vals[-1])
    return PycbcFrequencySeries(interp.astype(np.float64), delta_f=delta_f)


def pycbc_psd_to_gwpy(psd: PycbcFrequencySeries) -> FrequencySeries:
    data = np.asarray(psd.numpy())
    return FrequencySeries(
        data,
        f0=0.0,
        df=float(psd.delta_f),
        unit=(u.Hz ** -1),
    )


def simulate_noise(psd: PycbcFrequencySeries, length: int, delta_t: float, seed: int) -> np.ndarray:
    noise_ts = noise_from_psd(length, delta_t, psd, seed=seed)
    return np.asarray(noise_ts.numpy(), dtype=np.float64)


def build_waveform(model_name: str, sample_rate: float, log_amp: float) -> np.ndarray:
    model = get_model(model_name)
    waveform = StarccatoJimWaveform(model=model, sample_rate=sample_rate)
    params = {name: 0.0 for name in waveform.latent_names}
    params["log_amp"] = log_amp
    params.update(
        {
            "t_c": 0.0,
            "ra": 0.0,
            "dec": 0.0,
            "psi": 0.0,
            "luminosity_distance": float(np.exp(-log_amp)),
            "gmst": 0.0,
            "trigger_time": 0.0,
        }
    )
    return waveform.time_domain_waveform_numpy(params)


def center_injection(data: np.ndarray, injection: np.ndarray) -> None:
    if injection.size == 0:
        return
    n = min(len(data), len(injection))
    inj = injection[:n]
    start = len(data) // 2 - n // 2
    end = start + n
    data[start:end] += inj


def to_timeseries(values: np.ndarray, sample_rate: float, trigger_time: float, label: str) -> TimeSeries:
    dt = 1.0 / sample_rate
    duration = len(values) * dt
    epoch = trigger_time - duration / 2.0
    return TimeSeries(
        values,
        sample_rate=sample_rate,
        epoch=epoch,
        unit=u.dimensionless_unscaled,
        name=label,
    )


def write_bundle(
    bundle_path: Path,
    full_ts: TimeSeries,
    chunk_ts: TimeSeries,
    psd_fs: FrequencySeries,
    trigger_time: float,
) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    _write_analysis_bundle(str(bundle_path), chunk_ts, full_ts, psd_fs, trigger_time)


def crop_analysis_chunk(full_ts: TimeSeries, trigger_time: float, chunk_samples: int) -> TimeSeries:
    dt = full_ts.dt.value
    span = chunk_samples * dt / 2.0
    return full_ts.crop(trigger_time - span, trigger_time + span)


def generate_scenario_bundles(
    scenario: str,
    detectors: Iterable[str],
    *,
    outdir: Path,
    psd_cache: Dict[str, PycbcFrequencySeries],
    sample_rate: float,
    full_duration: float,
    chunk_samples: int,
    base_seed: int,
    signal_log_amp: float,
    glitch_log_amp: float,
    glitch_detector: str,
    signal_model: str,
    glitch_model: str,
) -> Dict[str, Path]:
    delta_t = 1.0 / sample_rate
    delta_f = psd_cache[next(iter(psd_cache))].delta_f
    n_full = int(full_duration * sample_rate)
    trigger_time = BASE_TRIGGER + {"noise": 0, "signal": 10, "glitch": 20}.get(scenario, 0)

    bundle_paths: Dict[str, Path] = {}
    signal_waveform = build_waveform(signal_model, sample_rate, signal_log_amp) if scenario == "signal" else None
    glitch_waveform = build_waveform(glitch_model, sample_rate, glitch_log_amp) if scenario == "glitch" else None

    for det_index, det in enumerate(detectors):
        psd = psd_cache[det]
        seed = base_seed + 100 * det_index + {"noise": 0, "signal": 1000, "glitch": 2000}.get(scenario, 0)
        noise_data = simulate_noise(psd, n_full, delta_t, seed)

        if scenario == "signal" and signal_waveform is not None:
            center_injection(noise_data, signal_waveform)
        if scenario == "glitch" and det.upper() == glitch_detector.upper() and glitch_waveform is not None:
            center_injection(noise_data, glitch_waveform)

        full_ts = to_timeseries(noise_data, sample_rate, trigger_time, f"{det}_{scenario}")
        chunk_ts = crop_analysis_chunk(full_ts, trigger_time, chunk_samples)
        psd_fs = pycbc_psd_to_gwpy(psd)
        det_dir = outdir / "bundles" / det.upper()
        bundle_path = det_dir / f"analysis_bundle_{int(trigger_time)}.hdf5"
        write_bundle(bundle_path, full_ts, chunk_ts, psd_fs, trigger_time)
        bundle_paths[det.upper()] = bundle_path

    return bundle_paths


def run_analysis_on_scenario(
    scenario: str,
    bundle_paths: Dict[str, Path],
    outdir: Path,
    detectors: Iterable[str],
    *,
    signal_model: str,
    glitch_model: str,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    latent_sigma_signal: float,
    latent_sigma_glitch: float,
    log_amp_sigma_signal: float,
    log_amp_sigma_glitch: float,
    rng_seed: int,
) -> Path:
    scenario_out = outdir / "analysis"
    scenario_out.mkdir(parents=True, exist_ok=True)
    result = run_bcr_posteriors(
        detectors=[d.upper() for d in detectors],
        outdir=str(scenario_out),
        bundle_paths={k: str(v) for k, v in bundle_paths.items()},
        signal_model=signal_model,
        glitch_model=glitch_model,
        latent_sigma_signal=latent_sigma_signal,
        latent_sigma_glitch=latent_sigma_glitch,
        log_amp_sigma_signal=log_amp_sigma_signal,
        log_amp_sigma_glitch=log_amp_sigma_glitch,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        rng_seed=rng_seed,
    )
    summary_path = scenario_out / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate design-PSD datasets and run Starccato analysis.")
    parser.add_argument("--outdir", type=Path, default=Path("lvk/studies/out_sim_psd"), help="Output root for synthetic bundles and analyses.")
    parser.add_argument("--cache-dir", type=Path, default=Path("lvk/studies/design_psd_cache"), help="Directory to cache downloaded ASD files.")
    parser.add_argument("--detector", action="append", dest="detectors", default=["H1", "L1"], help="Detector to include (repeat).")
    parser.add_argument("--scenarios", nargs="*", default=["noise", "signal", "glitch"], choices=["noise", "signal", "glitch"], help="Scenarios to generate.")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE, help="Sample rate in Hz for simulated data.")
    parser.add_argument("--duration", type=float, default=DEFAULT_FULL_DURATION, help="Full noise duration in seconds (sets PSD delta_f).")
    parser.add_argument("--chunk-samples", type=int, default=DEFAULT_CHUNK_SAMPLES, help="Number of samples in the analysis window.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for noise draws.")
    parser.add_argument("--signal-distance", type=float, default=1.0, help="Distance scale for coherent CCSN injection (lower is louder).")
    parser.add_argument("--glitch-distance", type=float, default=0.3, help="Distance-like scale for blip glitch injection amplitude.")
    parser.add_argument("--glitch-detector", type=str, default="H1", help="Detector to host the glitch in the 'glitch' scenario.")
    parser.add_argument("--signal-model", type=str, default="ccsne", help="Waveform model for coherent injection.")
    parser.add_argument("--glitch-model", type=str, default="blip", help="Waveform model for glitch injection.")
    parser.add_argument("--num-samples", type=int, default=500, help="NumPyro samples for BCR analysis.")
    parser.add_argument("--num-warmup", type=int, default=250, help="NumPyro warmup steps.")
    parser.add_argument("--num-chains", type=int, default=1, help="NumPyro chains.")
    parser.add_argument("--latent-sigma-signal", type=float, default=1.0, help="Latent prior sigma for coherent model.")
    parser.add_argument("--latent-sigma-glitch", type=float, default=0.5, help="Latent prior sigma for glitch model.")
    parser.add_argument("--log-amp-sigma-signal", type=float, default=1.0, help="log_amp sigma for coherent model.")
    parser.add_argument("--log-amp-sigma-glitch", type=float, default=0.2, help="log_amp sigma for glitch model.")
    parser.add_argument("--rng-seed", type=int, default=1234, help="Seed controlling sampling order in BCR analysis.")
    parser.add_argument("--skip-analysis", action="store_true", help="Only generate bundles; skip inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detectors = [d.upper() for d in args.detectors]
    if args.duration <= 0:
        raise ValueError("Duration must be positive.")
    delta_t = 1.0 / args.sample_rate
    delta_f = 1.0 / args.duration
    n_full = int(args.sample_rate * args.duration)
    n_freq = n_full // 2 + 1

    psd_cache: Dict[str, PycbcFrequencySeries] = {}
    for det in detectors:
        if det not in ASD_URLS:
            raise ValueError(f"No ASD URL configured for detector {det}.")
        psd_cache[det] = build_design_psd(det, delta_f, n_freq, args.cache_dir)

    for scenario in args.scenarios:
        scenario_root = args.outdir / scenario
        bundle_paths = generate_scenario_bundles(
            scenario,
            detectors,
            outdir=scenario_root,
            psd_cache=psd_cache,
            sample_rate=args.sample_rate,
            full_duration=args.duration,
            chunk_samples=args.chunk_samples,
            base_seed=args.seed,
            signal_log_amp=-np.log(max(args.signal_distance, 1e-3)),
            glitch_log_amp=-np.log(max(args.glitch_distance, 1e-3)),
            glitch_detector=args.glitch_detector,
            signal_model=args.signal_model,
            glitch_model=args.glitch_model,
        )

        if args.skip_analysis:
            continue
        summary_path = run_analysis_on_scenario(
            scenario,
            bundle_paths,
            scenario_root,
            detectors,
            signal_model=args.signal_model,
            glitch_model=args.glitch_model,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            latent_sigma_signal=args.latent_sigma_signal,
            latent_sigma_glitch=args.latent_sigma_glitch,
            log_amp_sigma_signal=args.log_amp_sigma_signal,
            log_amp_sigma_glitch=args.log_amp_sigma_glitch,
            rng_seed=args.rng_seed,
        )
        print(f"[{scenario}] analysis summary: {summary_path}")


if __name__ == "__main__":
    main()

