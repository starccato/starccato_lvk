#!/usr/bin/env python3
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import StateVector

# -------------------------------
# CONFIGURATION
# -------------------------------
BASE_DATA_DIR = "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3b/strain.4k/hdf.v1"
DATA_DIRS = {"H1": f"{BASE_DATA_DIR}/H1", "L1": f"{BASE_DATA_DIR}/L1"}
N_SEGMENTS = 5  # number of valid segments to show
DETECTOR_PAIR = ("H1", "L1")


# -------------------------------
# HELPERS
# -------------------------------
def get_data_files(detector):
    """Return sorted mapping from GPS start time → file path."""
    base = DATA_DIRS[detector]
    files = glob.glob(f"{base}/*/*.hdf5")
    gps_starts = [int(f.split("-")[1]) for f in files]
    return dict(sorted(zip(gps_starts, files)))


def get_cat3_segments(file_path):
    """Return list of (start, stop) tuples for CAT3-valid times."""
    sv = StateVector.read(file_path, format="hdf5.gwosc")
    dq = sv.to_dqflags()
    if "passes cbc CAT3 test" not in dq:
        return []
    cat3 = dq["passes cbc CAT3 test"]
    return cat3.active


def find_joint_cat3_segments():
    """Find times where both detectors have CAT3 valid."""
    print("Scanning for CAT3-valid overlap segments...")

    # Get H1 and L1 file maps
    h1_files = get_data_files("H1")
    l1_files = get_data_files("L1")

    # Only consider overlapping GPS chunks
    common_gps = sorted(set(h1_files.keys()) & set(l1_files.keys()))

    joint_segments = []

    for gps_start in common_gps:
        h1_seg = get_cat3_segments(h1_files[gps_start])
        l1_seg = get_cat3_segments(l1_files[gps_start])
        if not h1_seg or not l1_seg:
            continue

        # Compute overlaps
        for s1, e1 in h1_seg:
            for s2, e2 in l1_seg:
                start = max(s1, s2)
                stop = min(e1, e2)
                if stop - start > 2:  # require >2s overlap
                    joint_segments.append((int(start), int(stop)))

    print(f"Found {len(joint_segments)} overlapping CAT3 segments.")
    return np.array(joint_segments)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    segments = find_joint_cat3_segments()

    if len(segments) == 0:
        raise RuntimeError("No overlapping CAT3 segments found.")

    # Randomly select N segments
    chosen = random.sample(list(segments), min(N_SEGMENTS, len(segments)))

    print("Example CAT3-valid segments (both detectors):")
    for start, stop in chosen:
        print(f"  Start: {start}  Stop: {stop}  Duration: {stop - start}s")

    # -------------------------------
    # Plot CAT3 flags
    # -------------------------------
    det_colors = {"H1": "tab:blue", "L1": "tab:orange"}
    fig, ax = plt.subplots(figsize=(10, 3))

    for i, det in enumerate(DETECTOR_PAIR):
        files = get_data_files(det)
        first_file = list(files.values())[0]
        sv = StateVector.read(first_file, format="hdf5.gwosc")
        dq = sv.to_dqflags()
        cat3 = dq["passes cbc CAT3 test"]

        for seg in cat3.active:
            ax.barh(
                y=i,
                width=seg[1] - seg[0],
                left=seg[0],
                height=0.4,
                color=det_colors[det],
                alpha=0.6,
                label=det if i == 0 else ""
            )

    ax.set_xlabel("GPS Time [s]")
    ax.set_yticks(range(len(DETECTOR_PAIR)))
    ax.set_yticklabels(DETECTOR_PAIR)
    ax.legend()
    plt.title("CAT3-valid segments (H1 & L1)")
    plt.tight_layout()
    plt.savefig("cat3_valid_segments.png", dpi=150)
    print("Saved plot: cat3_valid_segments.png")



