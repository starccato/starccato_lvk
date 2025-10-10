"""Load strain data from HDF5 files."""

import os
from typing import Callable, Optional

import h5py
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from .plotting import plot
from .utils import _get_fnames_for_range
from .. import config


DataFetcher = Callable[[float, float], TimeSeries]


def strain_loader(
    trigger_time: float,
    outdir: str = None,
    data_fetcher: Optional[DataFetcher] = None,
    detector: Optional[str] = None,
) -> None:
    """Load strain data and compute PSD around a trigger time, optionally saving to outdir.

    Args:
        trigger_time: GPS time to centre the analysis window on.
        outdir: Directory in which to save the analysis/PSD products.
        data_fetcher: Custom callable returning a `TimeSeries` for (start, end) GPS times.
        detector: Detector code (e.g. "H1"/"L1") to use when falling back to GWOSC fetches.
    """
    data, psd = load_analysis_chunk_and_psd(
        trigger_time,
        data_fetcher=data_fetcher,
        detector=detector,
    )
    if outdir:
        _save_analysis_chunk_and_psd(data, psd, trigger_time, outdir)
    return data, psd


def load_analysis_chunk_and_psd(
    trigger_time: float,
    data_fetcher: Optional[DataFetcher] = None,
    detector: Optional[str] = None,
) -> (TimeSeries, FrequencySeries):
    """Load strain data and compute PSD around a trigger time."""
    analysis_start = trigger_time - 1
    gps_start = analysis_start - 65
    gps_end = trigger_time + 1
    data = load_strain_segment(
        gps_start,
        gps_end,
        data_fetcher=data_fetcher,
        detector=detector,
    )
    analysis_chunk = data.crop(analysis_start, gps_end)
    psd_chunk = data.crop(gps_start, analysis_start)
    psd = generate_psd(psd_chunk)

    return analysis_chunk, psd


def _save_analysis_chunk_and_psd(analysis_chunk, psd, trigger_time: float, outdir: str, injection=None) -> None:
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"analysis_chunk_{int(trigger_time)}.png")
    plot(analysis_chunk, psd, trigger_time, fname)
    # save the analysis chunk and psd
    analysis_fn = os.path.join(outdir, f"analysis_chunk_{int(trigger_time)}.hdf5")
    psd_fn = os.path.join(outdir, f"psd_{int(trigger_time)}.hdf5")
    analysis_chunk.write(analysis_fn, format='hdf5', overwrite=True)
    psd.write(psd_fn, format='hdf5', overwrite=True)

    if injection is not None:
        injection_fn = os.path.join(outdir, f"injection_{int(trigger_time)}.hdf5")
        with h5py.File(injection_fn, 'w') as f:
            f.create_dataset('injection', data=injection)


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
