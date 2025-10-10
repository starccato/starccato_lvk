import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE_DATA_CACHE = f"{HERE}/data/only_noise_segments.txt"


def load_only_noise_segments() -> np.ndarray:
    """
    Load segments of only noise data from a text file.

    Start GPS, Stop GPS
    1256655676 1256655806
    1256655816 1256655946
    1256655956 1256656086

    Returns
    -------
    np.ndarray
        Each tuple contains (start_time, end_time) of noise segments.
    """
    if not os.path.exists(NOISE_DATA_CACHE):
        raise FileNotFoundError(f"Noise data file not found: {NOISE_DATA_CACHE}")

    # Load the noise segments from the text file
    noise_segments = np.loadtxt(
        NOISE_DATA_CACHE,
        dtype=int,
        comments='#',
        delimiter=' ',
        skiprows=1,  # Skip the header line
    )

    if noise_segments.ndim == 1:
        noise_segments = noise_segments.reshape(-1, 2)

    return noise_segments


def get_noise_trigger_time(idx) -> float:
    noise_segments = load_only_noise_segments()
    assert idx < len(noise_segments), f"Index {idx} out of bounds for noise segments."
    return float(noise_segments[idx, 1] - 1)
