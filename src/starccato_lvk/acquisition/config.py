FS = 4096
MIN_SEGMENT_DURATION = 129  # seconds
SIGNAL_DURATION = 1  # seconds

# Default instrument used when a specific detector is not provided
DEFAULT_DETECTOR = "L1"

# Base directory pointing to local GWOSC mirrors (adjust per system)
BASE_DATA_DIR = "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3b/strain.4k/hdf.v1"

# Per-detector data directories. Update BASE_DATA_DIR above if path differs.
DATA_DIRS = {
    "H1": f"{BASE_DATA_DIR}/H1",
    "L1": f"{BASE_DATA_DIR}/L1",
}

# Backward compatibility alias (kept for older helpers expecting a single path)
DATA_DIR = DATA_DIRS.get(DEFAULT_DETECTOR, f"{BASE_DATA_DIR}/{DEFAULT_DETECTOR}")
