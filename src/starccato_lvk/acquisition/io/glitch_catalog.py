"""Load and filter Gravity Spy glitch catalogs (Zenodo record 5649212, O3b)."""

import os

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CSV_URL_TEMPLATE = "https://zenodo.org/records/5649212/files/{ifo}_O3b.csv"
FS = 4096.0
N = 512
DURATION = N / FS  # 0.125 seconds


def _blip_csv_file(ifo: str) -> str:
    # ponytail: L1 keeps the legacy "blip.csv" name so existing OzSTAR caches
    # (compute nodes have no internet) stay valid.
    name = "blip.csv" if ifo == "L1" else f"blip_{ifo}.csv"
    return os.path.join(DATA_DIR, name)


def load_full_glitch_catalog(ifo: str = "L1") -> pd.DataFrame:
    """Download the full Gravity Spy O3b CSV for `ifo` if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_file = os.path.join(DATA_DIR, f"{ifo}_O3b.csv")
    if not os.path.exists(csv_file):
        csv_url = CSV_URL_TEMPLATE.format(ifo=ifo)
        print(f"Downloading CSV from {csv_url}...")
        response = requests.get(csv_url)
        response.raise_for_status()
        with open(csv_file, "wb") as f:
            f.write(response.content)
    return pd.read_csv(csv_file)


def load_blip_glitch_catalog(min_confidence=0.9, min_duration=DURATION, ifo: str = "L1") -> pd.DataFrame:
    """Filter glitches for 'Blip' label, minimum confidence and duration."""
    os.makedirs(DATA_DIR, exist_ok=True)
    blip_csv = _blip_csv_file(ifo)
    if not os.path.exists(blip_csv):
        df = load_full_glitch_catalog(ifo)
        print(f"Filtering {ifo} blip glitches...")
        df = df[df['ml_label'] == "Blip"]
        filtered = df[
            (df['ml_confidence'] >= min_confidence) &
            (df['duration'] >= min_duration)
            ].sort_values(by='snr', ascending=False).reset_index(drop=True)
        filtered = filtered[['event_time', 'duration', 'snr']]
        filtered.to_csv(blip_csv, index=False)
        print(f"Filtered blip glitches saved to {blip_csv}")
    return pd.read_csv(blip_csv)


def get_blip_trigger_time(idx: int, ifo: str = "L1") -> float:
    """Get the trigger time for a specific blip glitch."""
    blip_df = load_blip_glitch_catalog(ifo=ifo)
    if idx < len(blip_df):
        return float(blip_df.iloc[idx]['event_time'])
    else:
        raise IndexError(f"Index {idx} out of bounds for {ifo} blip glitches.")
