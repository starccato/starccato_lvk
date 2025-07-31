"""Load strain data from HDF5 files."""

from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
import h5py

import os
from .plotting import plot
from .utils import _get_fnames_for_range

from starccato_jax.waveforms import StarccatoCCSNe

import numpy as np


def strain_loader(trigger_time: float, outdir: str = None, add_injection: bool = False, distance: float = None,
                  rng=None) -> None:
    data, psd = load_analysis_chunk_and_psd(trigger_time)

    injection = None
    if add_injection:
        # 2. Create the CCSNe signal
        injection = np.array(StarccatoCCSNe().generate(rng=rng, n=1)[0], dtype=np.float64)
        injection = injection * (100 / distance)  # scale to the desired distance
        data = _inject(data, injection, trigger_time)

    if outdir:
        _save_analysis_chunk_and_psd(data, psd, trigger_time, outdir, injection=injection)


def load_analysis_chunk_and_psd(trigger_time: float) -> (TimeSeries, FrequencySeries):
    """Load strain data and compute PSD around a trigger time."""
    analysis_start = trigger_time - 1
    gps_start = analysis_start - 65
    gps_end = trigger_time + 1
    data = load_strain_segment(gps_start, gps_end)
    analysis_chunk = data.crop(analysis_start, gps_end)
    psd_chunk = data.crop(gps_start, analysis_start)
    psd = generate_psd(psd_chunk)

    return analysis_chunk, psd


def _save_analysis_chunk_and_psd(analysis_chunk, psd, trigger_time: float, outdir: str, injection=None) -> None:
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"analysis_chunk_{int(trigger_time)}.png")
    plot(analysis_chunk, psd, trigger_time, fname, injection=injection)
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


def load_strain_segment(gps_start: float, gps_end: float) -> TimeSeries:
    """Load strain data segment from HDF5 files."""
    files = _get_fnames_for_range(gps_start, gps_end)
    return TimeSeries.read(files, format='hdf5.gwosc', start=gps_start, end=gps_end)


def _inject(data: TimeSeries, injection: np.ndarray, trigger_time: float) -> TimeSeries:

    ## TODO: I THINK THIS IS Fed -- see the plot of the injection in the test_sampler.py -- doesnt match trigger time
    n_inj = len(injection)
    fs = data.sample_rate.value  # sampling frequency in Hz
    dt = 1 / fs  # time step between samples

    # Calculate half the duration of the injection in seconds
    t_offset = n_inj / (2 * fs)

    # Define the start and end times of the injection
    t0 = trigger_time - t_offset
    t1 = trigger_time + t_offset

    # Convert t0 and t1 to sample indices
    start_time = data.times.value[0]  # absolute start time of the TimeSeries in GPS
    t0_idx = int(round((t0 - start_time) * fs))
    t1_idx = t0_idx + n_inj  # inject exactly n_inj samples

    # Safety check to avoid IndexError
    if t0_idx < 0 or t1_idx > len(data):
        raise ValueError("Injection exceeds bounds of TimeSeries.")

    # Inject the signal
    data.value[t0_idx:t1_idx] += injection
    return data
