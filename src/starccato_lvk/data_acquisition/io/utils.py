import glob
import re
from typing import Dict

from starccato_lvk.data_acquisition import config


def _get_data_files_and_gps_times() -> Dict[int, str]:
    """Get a mapping of GPS times (start times) to HDF5 files.

    Returns:
        Dict[int, str]: A dictionary mapping GPS start times to HDF5 file paths.
        {
        100: "path/to/file.hdf5",
        200: "path/to/another_file.hdf5",
        ...
        999: "path/to/yet_another_file.hdf5"
        }
    """
    search_str = f"{config.DATA_DIR}/*/*.hdf5"
    files = glob.glob(search_str)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found at {search_str}")
    gps_starts = [int(re.search(r"R1-(\d+)-\d+\.hdf5", p).group(1)) for p in files]
    path_dict = {gps: f for gps, f in zip(gps_starts, files)}
    path_dict = dict(sorted(path_dict.items()))
    return path_dict


def _get_fnames_for_range(gps_start: float, gps_end: float) -> (str, str):
    """Get the filenames for the start and end GPS times."""
    gps_start = int(gps_start)
    gps_end = int(gps_end)

    gps_files = _get_data_files_and_gps_times()
    start_times = sorted(gps_files.keys())

    files = []

    for i in range(len(start_times)):
        t0 = start_times[i]
        t1 = start_times[i + 1] if i + 1 < len(start_times) else float('inf')

        # Does [gps_start, gps_end] intersect with [interval_start, interval_end]?
        if gps_end > t0 and gps_start < t1:
            files.append(gps_files[t0])

    return files
