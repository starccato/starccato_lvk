"""Load strain data from HDF5 files."""

from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

import os
from .plotting import plot
from .utils import _get_fnames_for_range

def load_analysis_chunk_and_psd(trigger_time: float, outdir: str = None) -> (TimeSeries, FrequencySeries):
    """Load strain data and compute PSD around a trigger time."""
    analysis_start = trigger_time - 1
    gps_start = analysis_start - 65
    gps_end = trigger_time + 1
    data = load_strain_segment(gps_start, gps_end)
    analysis_chunk = data.crop(analysis_start, gps_end)
    psd_chunk = data.crop(gps_start, analysis_start)
    psd = generate_psd(psd_chunk)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f"analysis_chunk_{int(trigger_time)}.png")
        plot(analysis_chunk, psd, trigger_time, fname)
        # save the analysis chunk and psd
        analysis_fn = os.path.join(outdir, f"analysis_chunk_{int(trigger_time)}.hdf5")
        psd_fn = os.path.join(outdir, f"psd_{int(trigger_time)}.hdf5")
        if not os.path.exists(analysis_fn):
            analysis_chunk.write(analysis_fn, format='hdf5')
        if not os.path.exists(psd_fn):
            psd.write(psd_fn, format='hdf5')
    return analysis_chunk, psd


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


def load_strain_segment(gps_start: float, gps_end: float) -> TimeSeries:
    """Load strain data segment from HDF5 files."""
    files = _get_fnames_for_range(gps_start, gps_end)
    return TimeSeries.read(files, format='hdf5.gwosc', start=gps_start, end=gps_end)
