from gwpy.timeseries import StateVector
import numpy as np
import os

import matplotlib.pyplot as plt

from .utils import _get_fnames_for_range


def generate_times_for_valid_data(
        gps_start: float, gps_end: float, segment_length: int = 130, min_gap: int = 10,
        outdir: str = 'outdir'
) -> np.ndarray:
    """Generate a list of GPS times for valid data segments that pass CAT3 CBC test

    Args:
        gps_start: Start GPS time
        gps_end: End GPS time
        segment_length: Length of each data segment in seconds (default 130)
        min_gap: Minimum gap between segments in seconds (default 10)

    Returns:
        2D numpy array of shape (N, 2) with start-stop GPS times for valid segments
    """

    files = _get_fnames_for_range(gps_start, gps_end)

    valid_times = [
        _get_valid_start_stops_for_one_file(file, segment_length, min_gap)
        for file in files
    ]

    # Concatenate all valid segments from all files
    if len(valid_times) == 0:
        raise ValueError("No valid segments found in the provided files.")
    valid_start_stop = np.concatenate(valid_times, axis=0)
    # Save the valid segments to a file if outdir is provided
    print(f"Identified {len(valid_start_stop)} valid segments.")

    os.makedirs(outdir, exist_ok=True)
    plot_valid_segments(valid_start_stop, outdir=outdir)
    # save txt with [start, stop] pairs
    np.savetxt(
        os.path.join(outdir, f"valid_segments_{int(gps_start)}-{int(gps_end)}.txt"),
        valid_start_stop, fmt='%d',
        header='Start GPS, Stop GPS',
        comments=''
    )


def _get_valid_start_stops_for_one_file(file, segment_length, min_gap):
    state_vec = StateVector.read(file, format='hdf5.gwosc')

    # Parse state vector into data quality flags
    flags = state_vec.to_dqflags()
    cbc_cat3_flag = flags["passes cbc CAT3 test"]

    valid_segments = []
    for seg in cbc_cat3_flag.active:
        duration = seg[1] - seg[0]
        if duration >= segment_length:
            valid_segments.append([int(seg[0]), int(seg[1])])

    # print(f"Found {len(valid_segments)} valid segments that "
    #       f"pass CAT3 CBC test and have duration >= {segment_length} seconds.")

    # Chunk up each valid segment
    valid_start_stop = []
    for start, stop in valid_segments:
        # Start with min_gap from the beginning of this valid segment
        current_start = start + min_gap

        # Create consecutive segments with min_gap between them
        while current_start + segment_length <= stop:
            seg_stop = current_start + segment_length
            valid_start_stop.append([current_start, seg_stop])
            current_start = seg_stop + min_gap  # Next segment starts after min_gap

    # Convert to numpy array
    if len(valid_start_stop) == 0:
        raise ValueError("No valid segments found that pass CAT3 CBC test with the given parameters.")

    valid_start_stop = np.array(valid_start_stop)
    return valid_start_stop


def load_state_vector(gps_start: float, gps_end: float) -> StateVector:
    """Load state vector data segment from HDF5 files."""
    files = _get_fnames_for_range(gps_start, gps_end)
    return StateVector.read(files, format='hdf5.gwosc')


def plot_valid_segments(valid_segments: np.ndarray, outdir: str = None):
    plt.figure(figsize=(10, 5))
    for start, stop in valid_segments:
        plt.plot(
            [start, stop], [0, 0], color='green', linewidth=30,
            solid_capstyle='butt', marker='|', markersize=35,
        )
        # draw a vertical line at the start and stop of each segment
        plt.axvline(x=start, color='red', linestyle='--', linewidth=1, label='Start Time')
        plt.axvline(x=stop, color='blue', linestyle='--', linewidth=1, label='Start Time')

    plt.yticks([])
    plt.grid(False)
    plt.xlabel('GPS Time')
    plt.ylabel('Valid Segment')
    plt.title('Valid Segments for Data Acquisition')
    plt.savefig(os.path.join(outdir, f"valid_segments_{valid_segments[0, 0]}-{valid_segments[-1, -1]}.png"))
