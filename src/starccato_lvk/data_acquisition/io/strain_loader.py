"""Load strain data from HDF5 files."""
import glob
import re
from typing import Dict

from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries


from gwpy.timeseries import StateVector
file = "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3b/strain.4k/hdf.v1/V1/1259339776/V-V1_GWOSC_O3b_4KHZ_R1-1259835392-4096.hdf5"
state = StateVector.read(file, format='hdf5.gwosc')


from starccato_lvk.data_acquisition import config
import os
from .plotting import plot


def load_analysis_chunk_and_psd(trigger_time: float, outdir:str=None) -> (TimeSeries, FrequencySeries):
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
    gps_to_files = _get_data_files_and_gps_times()
    gps_start_file = _get_fname_for_gps(gps_start, gps_to_files)
    gps_end_file = _get_fname_for_gps(gps_end, gps_to_files)
    files = list(set([gps_start_file, gps_end_file]))
    data = TimeSeries.read(gps_start_file, format='hdf5.gwosc', start=gps_start, end=gps_end)
    return data


def load_state_vector(gps_start: float, gps_end: float) -> StateVector:
    """Load state vector data segment from HDF5 files."""
    gps_to_files = _get_data_files_and_gps_times()
    gps_start_file = _get_fname_for_gps(gps_start, gps_to_files)
    gps_end_file = _get_fname_for_gps(gps_end, gps_to_files)

    all_files = list(gps_to_files.values())
    start_index = all_files.index(gps_start_file)
    end_index = all_files.index(gps_end_file) + 1  # include the end file
    files = all_files[start_index:end_index]


    state_vector = StateVector.read(files, format='hdf5.gwosc')


    return state_vector


def _get_data_files_and_gps_times() -> Dict[int, str]:
    """Get a mapping of GPS times (start times) to HDF5 files."""
    files = glob.glob(f"{config.DATA_DIR}/*/*.hdf5")
    gps_starts = [int(re.search(r"R1-(\d+)-\d+\.hdf5", p).group(1)) for p in files]
    return {gps: f for gps, f in zip(gps_starts, files)}


def _get_fname_for_gps(gps: float, gps_to_files: Dict[int, str]) -> str:
    """Get the filename for a given GPS time."""
    gps = int(gps)
    all_start_times = sorted(gps_to_files.keys())
    for i in range(len(all_start_times) - 1):
        if all_start_times[i] <= gps < all_start_times[i + 1]:
            return gps_to_files[all_start_times[i]]
