import os
import pandas as pd
import requests
from gwpy.timeseries import TimeSeries
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from gwpy.timeseries import TimeSeries

OUTDIR = "glitch_data"
os.makedirs(OUTDIR, exist_ok=True)

CSV_URL = "https://zenodo.org/records/5649212/files/L1_O3b.csv"
CSV_FILE = "L1_O3b.csv"

FS = 4096.0  # Sample rate in Hz, used for plotting and analysis
N = 512
DURATION = N / FS  # Duration of each segment in seconds




# Step 1: Download CSV if not present
if not os.path.exists(CSV_FILE):
    print("Downloading CSV...")
    response = requests.get(CSV_URL)
    with open(CSV_FILE, "wb") as f:
        f.write(response.content)

# Step 2: Filter CSV
df = pd.read_csv(CSV_FILE)
df = df[df['ml_label'] == "Blip"]



filtered = df[
    (df['ml_confidence'] >= 0.9) &
    (df['duration'] < DURATION)
]
# sort by SNR
filtered = filtered.sort_values(by='snr', ascending=False)

