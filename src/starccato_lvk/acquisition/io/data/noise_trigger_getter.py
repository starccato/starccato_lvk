#!/usr/bin/env python3
"""
Find overlapping CAT3-valid noise segments for both H1 and L1
in local GWOSC O3b strain data, merge short gaps, and save to text file.
"""

import os
import re
import glob
import logging
import numpy as np
from gwpy.timeseries import StateVector

# ==============================================================
# CONFIGURATION
# ==============================================================
BASE_DATA_DIR = "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3b/strain.4k/hdf.v1"
DATA_DIRS = {"H1": f"{BASE_DATA_DIR}/H1", "L1": f"{BASE_DATA_DIR}/L1"}

OUT_FILE = "only_noise_segments.txt"
LOG_FILE = "cat3_segment_scan.log"
MERGE_GAP = 10  # seconds: merge adjacent segments with small gaps

# ==============================================================
# LOGGING SETUP
# ==============================================================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console)

# ==============================================================
# HELPERS
# ==============================================================
def get_data_files(detector: str) -> dict[int, str]:
    """Return mapping from GPS start → file path (handles H-H1/L-L1 naming)."""
    base = DATA_DIRS[detector]
    search_pattern = os.path.join(base, "*", "*.hdf5")
    files = sorted(glob.glob(search_pattern))
    gps_starts = []

    for f in files:
        m = re.search(r"R1-(\d+)-\d+\.hdf5$", os.path.basename(f))
        if not m:
            logging.warning(f"Could not parse GPS start from filename: {f}")
            continue
        gps_starts.append(int(m.group(1)))

    mapping = dict(sorted(zip(gps_starts, files)))
    logging.info(f"{detector}: Found {len(mapping)} files.")
    return mapping


def get_cat3_segments(file_path: str) -> list[tuple[int, int]]:
    """Return CAT3-active intervals for one file."""
    try:
        sv = StateVector.read(file_path, format="hdf5.gwosc")
        dq = sv.to_dqflags()
        if "passes cbc CAT3 test" not in dq:
            return []
        return dq["passes cbc CAT3 test"].active
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []


def find_joint_segments() -> np.ndarray:
    """Find overlapping CAT3 segments for both detectors."""
    h1_files = get_data_files("H1")
    l1_files = get_data_files("L1")

    # Only consider GPS chunks present in both
    common_gps = sorted(set(h1_files.keys()) & set(l1_files.keys()))
    logging.info(f"Found {len(common_gps)} overlapping GPS files between H1 and L1.")

    joint_segments = []

    for gps in common_gps:
        h1_seg = get_cat3_segments(h1_files[gps])
        l1_seg = get_cat3_segments(l1_files[gps])
        if not h1_seg or not l1_seg:
            continue

        for s1, e1 in h1_seg:
            for s2, e2 in l1_seg:
                start, stop = max(s1, s2), min(e1, e2)
                if stop > start:
                    joint_segments.append((start, stop))

    logging.info(f"Total raw overlapping CAT3 segments: {len(joint_segments)}")
    return np.array(joint_segments, dtype=int)


def merge_segments(segments: np.ndarray, gap: int = 10) -> np.ndarray:
    """Merge adjacent/overlapping segments separated by <= gap seconds."""
    if len(segments) == 0:
        return segments

    segments = segments[np.argsort(segments[:, 0])]  # sort by start
    merged = [segments[0]]

    for start, stop in segments[1:]:
        last_start, last_stop = merged[-1]
        if start - last_stop <= gap:  # merge
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))

    logging.info(f"Merged to {len(merged)} segments (gap ≤ {gap}s).")
    return np.array(merged, dtype=int)


def save_segments(segments: np.ndarray, path: str):
    """Save start/stop GPS pairs to text file."""
    if len(segments) == 0:
        logging.warning("No CAT3 segments to save.")
        return
    header = "StartGPS StopGPS"
    np.savetxt(path, segments, fmt="%d", header=header, comments="")
    logging.info(f"Saved {len(segments)} segments to {path}")


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    logging.info("=== Starting CAT3 noise segment scan ===")

    segments = find_joint_segments()
    merged = merge_segments(segments, gap=MERGE_GAP)
    save_segments(merged, OUT_FILE)

    logging.info("=== Completed successfully ===")
