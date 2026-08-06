import os

FS = 4096
MIN_SEGMENT_DURATION = 129  # seconds
SIGNAL_DURATION = 1  # seconds

# Default instrument used when a specific detector is not provided
DEFAULT_DETECTOR = "L1"

# Which observing run's local mirror to read from. The production campaign is
# O3b-only, so that stays the default; override with
# STARCCATO_LVK_GWOSC_RUN=O3a (etc.) BEFORE the interpreter starts (e.g. in a
# SLURM script's #SBATCH --export or a shell export) when fetching a
# different run on a cluster with a local mirror -- this module reads the
# env var once at import time, not per call.
GWOSC_RUN = os.environ.get("STARCCATO_LVK_GWOSC_RUN", "O3b")

# Base directory pointing to local GWOSC mirrors (adjust per system)
BASE_DATA_DIR = f"/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/{GWOSC_RUN}/strain.4k/hdf.v1"

# Per-detector data directories. Update BASE_DATA_DIR above if path differs.
DATA_DIRS = {
    "H1": f"{BASE_DATA_DIR}/H1",
    "L1": f"{BASE_DATA_DIR}/L1",
}

# Backward compatibility alias (kept for older helpers expecting a single path)
DATA_DIR = DATA_DIRS.get(DEFAULT_DETECTOR, f"{BASE_DATA_DIR}/{DEFAULT_DETECTOR}")
