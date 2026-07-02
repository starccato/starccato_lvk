import glob
import re
from typing import Dict, Optional

from starccato_lvk.acquisition import config


def _get_data_files_and_gps_times(detector: Optional[str] = None) -> Dict[int, tuple]:
    """Get a mapping of GPS start times to (file path, duration) tuples.

    The duration is parsed from the GWOSC filename (``R1-<start>-<duration>``)
    so each file's true coverage ``[start, start + duration)`` is known. This
    matters because the local mirror can have GAPS (missing chunks): coverage
    must be bounded by the file's own length, not by the next file's start.

    Returns:
        Dict[int, tuple]: ``{gps_start: (path, duration), ...}`` sorted by start.
    """

    det = (detector or config.DEFAULT_DETECTOR).upper()
    base = config.DATA_DIRS.get(det, config.DATA_DIR)
    print(f"Looking for HDF5 files in {base} for detector {det}...")
    search_str = f"{base}/*/*.hdf5"
    files = glob.glob(search_str)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found at {search_str}")
    path_dict = {}
    for p in files:
        m = re.search(r"R1-(\d+)-(\d+)\.hdf5", p)
        if m is None:
            continue
        path_dict[int(m.group(1))] = (p, int(m.group(2)))
    return dict(sorted(path_dict.items()))


def _get_fnames_for_range(gps_start: float, gps_end: float, detector: Optional[str] = None) -> list:
    """Return local files whose true coverage intersects ``[gps_start, gps_end)``.

    Each file covers ``[start, start + duration)``; gaps in the mirror yield no
    file (rather than the previous file being wrongly returned), so a caller
    reading the result gets exactly the covering files or an empty list.
    """
    gps_start = int(gps_start)
    gps_end = int(gps_end)

    gps_files = _get_data_files_and_gps_times(detector)

    files = []
    for t0, (path, dur) in gps_files.items():
        t1 = t0 + dur  # true end of this file's coverage
        if gps_end > t0 and gps_start < t1:
            files.append(path)

    return files
