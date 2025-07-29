"""Load and filter glitch catalogs."""

import os

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CSV_URL = "https://zenodo.org/records/5649212/files/L1_O3b.csv"
FS = 4096.0
N = 512
DURATION = N / FS  # 0.125 seconds

FULL_CSV_FILE = os.path.join(DATA_DIR, "L1_O3b.csv")
BLIP_CSV_FILE = os.path.join(DATA_DIR, "blip.csv")

def load_full_glitch_catalog(csv_url=CSV_URL, csv_file=FULL_CSV_FILE):
    """Download CSV file if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(csv_file):
        print(f"Downloading CSV from {csv_url}...")
        response = requests.get(csv_url)
        response.raise_for_status()
        with open(csv_file, "wb") as f:
            f.write(response.content)
    return pd.read_csv(csv_file)

def load_blip_glitch_catalog(min_confidence=0.9, min_duration=DURATION):
    """Filter glitches for 'Blip' label, minimum confidence and duration."""
    df = load_full_glitch_catalog()
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(BLIP_CSV_FILE):
        print("Filtering blip glitches...")
        df = df[df['ml_label'] == "Blip"]
        filtered = df[
            (df['ml_confidence'] >= min_confidence) &
            (df['duration'] >= min_duration)
            ].sort_values(by='snr', ascending=False).reset_index(drop=True)
        filtered = filtered[['event_time', 'duration', 'snr']]
        filtered.to_csv(BLIP_CSV_FILE, index=False)
        print(f"Filtered blip glitches saved to {BLIP_CSV_FILE}")
    return pd.read_csv(BLIP_CSV_FILE)

def get_blip_trigger_time(idx:int)->float:
    """Get the trigger time for a specific blip glitch."""
    blip_df = load_blip_glitch_catalog()
    if idx < len(blip_df):
        return float(blip_df.iloc[idx]['event_time'])
    else:
        raise IndexError(f"Index {idx} out of bounds for blip glitches.")

