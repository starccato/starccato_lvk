"""Load strain data from HDF5 files."""

import os
from typing import Callable, Optional

import h5py
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from .plotting import plot
from .utils import _get_fnames_for_range
from .. import config

DataFetcher = Callable[[float, float], TimeSeries]
ANALYSIS_SAMPLES = 512


def strain_loader(
    trigger_time: float,
    outdir: str = None,
    data_fetcher: Optional[DataFetcher] = None,
    detector: Optional[str] = None,
) -> tuple[TimeSeries, FrequencySeries]:
    """Load strain data and compute PSD around a trigger time, optionally saving to outdir.

    Args:
        trigger_time: GPS time to centre the analysis window on.
        outdir: Directory in which to save the analysis/PSD products.
        data_fetcher: Custom callable returning a `TimeSeries` for (start, end) GPS times.
        detector: Detector code (e.g. "H1"/"L1") to use when falling back to GWOSC fetches.
    """
    analysis_chunk, psd, full_data = load_analysis_chunk_and_psd(
        trigger_time,
        data_fetcher=data_fetcher,
        detector=detector,
    )
    if outdir:
        _save_analysis_chunk_and_psd(analysis_chunk, psd, full_data, trigger_time, outdir)
    return analysis_chunk, psd


def load_analysis_chunk_and_psd(
    trigger_time: float,
    data_fetcher: Optional[DataFetcher] = None,
    detector: Optional[str] = None,
) -> tuple[TimeSeries, FrequencySeries, TimeSeries]:
    """Load strain data and compute PSD around a trigger time."""
    analysis_start = trigger_time - 1
    gps_start = analysis_start - 65
    gps_end = trigger_time + 1
    full_data = load_strain_segment(
        gps_start,
        gps_end,
        data_fetcher=data_fetcher,
        detector=detector,
    )

    times = full_data.times.value
    values = full_data.value
    sample_rate = full_data.sample_rate.value

    center_idx = int(np.argmin(np.abs(times - trigger_time)))
    half = ANALYSIS_SAMPLES // 2
    start_idx = center_idx - half
    end_idx = start_idx + ANALYSIS_SAMPLES

    if start_idx < 0 or end_idx > len(values):
        raise ValueError(
            "Requested analysis window exceeds available data. "
            "Increase segment buffer or reduce analysis length."
        )

    analysis_values = values[start_idx:end_idx]
    analysis_times = times[start_idx:end_idx]
    analysis_chunk = TimeSeries(
        analysis_values,
        times=analysis_times,
        unit=full_data.unit,
    )

    psd_chunk = full_data.crop(gps_start, analysis_start)
    psd = generate_psd(psd_chunk)

    return analysis_chunk, psd, full_data


def _save_analysis_chunk_and_psd(
    analysis_chunk: TimeSeries,
    psd: FrequencySeries,
    full_data: TimeSeries,
    trigger_time: float,
    outdir: str,
    injection=None,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Save diagnostic plot (optional artefact)
    plot_fname = os.path.join(outdir, f"analysis_chunk_{int(trigger_time)}.png")
    plot(full_data, psd, trigger_time, plot_fname)

    bundle_path = os.path.join(outdir, f"analysis_bundle_{int(trigger_time)}.hdf5")
    _write_analysis_bundle(bundle_path, analysis_chunk, full_data, psd, trigger_time, injection)

    return bundle_path


def generate_psd(data: TimeSeries) -> FrequencySeries:
    """
    See https://lscsoft.docs.ligo.org/bilby_pipe/0.3.12/_modules/bilby_pipe/data_generation.html#DataGenerationInput.__generate_psd
    """
    roll_off = 0.4
    duration = 4.0
    fractional_overlap = 0.5
    overlap = fractional_overlap * duration
    psd_alpha = 2 * roll_off / duration
    return data.psd(
        fftlength=duration,
        overlap=overlap,
        window=("tukey", psd_alpha),
        method="median",
    )


def _local_data_fetcher(gps_start: float, gps_end: float) -> TimeSeries:
    files = _get_fnames_for_range(gps_start, gps_end)
    files_avail = [f for f in files if os.path.exists(f)]
    if len(files_avail) == 0:
        raise FileNotFoundError(
            f"No data files found for GPS range {gps_start} to {gps_end}. Checked files: {files}"
        )
    return TimeSeries.read(files_avail, format="hdf5.gwosc", start=gps_start, end=gps_end)


def _remote_data_fetcher(detector: str) -> DataFetcher:
    def fetch(gps_start: float, gps_end: float) -> TimeSeries:
        return TimeSeries.fetch_open_data(detector, gps_start, gps_end, verbose=False)

    return fetch


def load_strain_segment(
    gps_start: float,
    gps_end: float,
    data_fetcher: Optional[DataFetcher] = None,
    detector: Optional[str] = None,
) -> TimeSeries:
    """Load strain data segment using a provided fetcher or local files."""
    fetcher = data_fetcher or _local_data_fetcher
    try:
        return fetcher(gps_start, gps_end)
    except FileNotFoundError as err:
        if fetcher is not data_fetcher:
            det = detector or getattr(config, "DEFAULT_DETECTOR", None)
            if det is None:
                raise err
            remote_fetch = _remote_data_fetcher(det)
            print(
                "Local strain files not found; fetching data from GWOSC for "
                f"detector {det} between {gps_start} and {gps_end}."
            )
            try:
                return remote_fetch(gps_start, gps_end)
            except Exception as fetch_err:
                raise RuntimeError(
                    "Failed to fetch strain data from GWOSC. "
                    "Check your network connection or set `DEFAULT_DETECTOR` "
                    "to a valid instrument."
                ) from fetch_err
        raise


def _write_analysis_bundle(
    bundle_path: str,
    analysis_chunk: TimeSeries,
    full_data: TimeSeries,
    psd: FrequencySeries,
    trigger_time: float,
    injection=None,
) -> None:
    with h5py.File(bundle_path, "w") as f:
        f.attrs["trigger_time"] = float(trigger_time)

        short_grp = f.create_group("strain")
        short_grp.create_dataset("values", data=analysis_chunk.value)
        short_grp.attrs["t0"] = float(analysis_chunk.times.value[0])
        short_grp.attrs["dt"] = float(analysis_chunk.dt.value)
        short_grp.attrs["unit"] = str(analysis_chunk.unit)
        short_grp.attrs["sample_rate"] = float(analysis_chunk.sample_rate.value)

        full_grp = f.create_group("full_strain")
        full_grp.create_dataset("values", data=full_data.value)
        full_grp.attrs["t0"] = float(full_data.times.value[0])
        full_grp.attrs["dt"] = float(full_data.dt.value)
        full_grp.attrs["unit"] = str(full_data.unit)
        full_grp.attrs["sample_rate"] = float(full_data.sample_rate.value)

        psd_grp = f.create_group("psd")
        psd_grp.create_dataset("values", data=psd.value)
        psd_grp.create_dataset("frequencies", data=psd.frequencies.value)
        psd_grp.attrs["unit"] = str(psd.unit)

        if injection is not None:
            extras = f.create_group("extras")
            extras.create_dataset("injection", data=np.asarray(injection))


def load_analysis_bundle(bundle_path: str) -> tuple[TimeSeries, FrequencySeries, dict]:
    with h5py.File(bundle_path, "r") as f:
        metadata = dict(f.attrs)

        strain_grp = f["strain"]
        strain_values = strain_grp["values"][...]
        t0 = float(strain_grp.attrs["t0"])
        dt = float(strain_grp.attrs["dt"])
        unit = strain_grp.attrs.get("unit")
        times = t0 + np.arange(strain_values.shape[0]) * dt
        strain = TimeSeries(strain_values, times=times, unit=unit)

        psd_grp = f["psd"]
        psd_values = psd_grp["values"][...]
        freqs = psd_grp["frequencies"][...]
        psd_unit = psd_grp.attrs.get("unit")
        psd = FrequencySeries(psd_values, frequencies=freqs, unit=psd_unit)

        if "full_strain" in f:
            full_grp = f["full_strain"]
            full_vals = full_grp["values"][...]
            full_t0 = float(full_grp.attrs["t0"])
            full_dt = float(full_grp.attrs["dt"])
            full_unit = full_grp.attrs.get("unit")
            full_times = full_t0 + np.arange(full_vals.shape[0]) * full_dt
            metadata["full_strain"] = TimeSeries(full_vals, times=full_times, unit=full_unit)

        if "extras" in f and "injection" in f["extras"]:
            metadata["injection"] = f["extras"]["injection"][...]

    return strain, psd, metadata
