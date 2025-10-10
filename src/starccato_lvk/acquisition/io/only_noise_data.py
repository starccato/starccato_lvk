import os
from typing import Optional

import numpy as np

from .determine_valid_segments import _get_valid_start_stops_for_one_file
from .utils import _get_data_files_and_gps_times

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE_DATA_CACHE = f"{HERE}/data/only_noise_segments.txt"
DEFAULT_SEGMENT_LENGTH = 130  # seconds
DEFAULT_MIN_GAP = 10  # seconds
MAX_SEGMENTS = 2000  # cap to avoid huge caches


def _load_cached_segments(cache_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(cache_path):
        return None

    noise_segments = np.loadtxt(
        cache_path,
        dtype=int,
        comments='#',
        delimiter=' ',
        skiprows=1,  # Skip the header line
    )

    if noise_segments.ndim == 1:
        noise_segments = noise_segments.reshape(-1, 2)

    return noise_segments


def _cache_segments(cache_path: str, segments: np.ndarray) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    header = "Start GPS, Stop GPS"
    np.savetxt(cache_path, segments.astype(int), fmt="%d", header=header, comments="")


def load_only_noise_segments() -> np.ndarray:
    """
    Load CAT3-valid noise segments.

    Returns
    -------
    np.ndarray
        Each tuple contains (start_time, end_time) of noise segments.
    """

    segments = _load_cached_segments(NOISE_DATA_CACHE)
    if segments is not None:
        return segments

    segments = _generate_noise_segments_from_local_files()
    if segments.size == 0:
        raise RuntimeError(
            "Unable to find CAT3-valid noise segments in the local data store. "
            "Ensure DATA_DIR points to valid LVK strain files."
        )

    _cache_segments(NOISE_DATA_CACHE, segments)
    return segments


def get_noise_trigger_time(idx) -> float:
    noise_segments = load_only_noise_segments()
    if len(noise_segments) == 0:
        raise RuntimeError("No CAT3-valid noise segments available.")
    # Allow indices beyond the list length by cycling (helps for large batches)
    seg = noise_segments[idx % len(noise_segments)]
    # choose a time safely inside the segment (1 second before end by default)
    return float(seg[1] - 1)


def _generate_noise_segments_from_local_files(
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    min_gap: int = DEFAULT_MIN_GAP,
) -> np.ndarray:
    """Scan local GWOSC/OzSTAR files and extract CAT3-valid noise segments."""
    files_map = _get_data_files_and_gps_times()
    if not files_map:
        return np.empty((0, 2), dtype=int)

    segments = []
    for gps_start, file_path in files_map.items():
        try:
            valid = _get_valid_start_stops_for_one_file(
                file_path,
                segment_length=segment_length,
                min_gap=min_gap,
            )
        except (ValueError, FileNotFoundError, OSError):
            continue

        if valid.size > 0:
            segments.append(valid)

        if segments and sum(len(seg) for seg in segments) >= MAX_SEGMENTS:
            break

    if not segments:
        return np.empty((0, 2), dtype=int)

    all_segments = np.vstack(segments)
    # Shuffle so that repeated requests cycle through varied times
    rng = np.random.default_rng(seed=0)
    rng.shuffle(all_segments)
    return all_segments
