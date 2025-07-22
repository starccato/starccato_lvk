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


def plot_gravityspy_qtransform(ts, fname, event_time):
    """
    Perform Q-transform and plot spectrogram in Gravity Spy style.

    Parameters:
        data (array-like): Input time series data.
        srate (float): Sample rate of the data.
        tlim (tuple): Time limits for x-axis (default: (1.5, 2.5)).
    """
    q_scan = ts.q_transform(qrange=[4, 64], frange=[10, 2048],
                            tres=0.002, fres=0.5, whiten=True
                            )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=120)
    xlims = [
        (event_time - 0.25, event_time + 0.25),
        (event_time - 0.5, event_time + 0.5),
        (event_time - 1, event_time + 1),
        (event_time - 2, event_time + 2),
    ]
    for ax, xlim in zip(axes, xlims):
        im = ax.imshow(q_scan, cmap='viridis')
        ax.set_xlim(xlim)
        ax.set_yscale('log', base=2)
        ax.set_ylabel('Frequency (Hz)', fontsize=14)
        ax.set_xlabel('Time (s)', labelpad=0.1, fontsize=14)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=14)
    plt.suptitle(f"Q-transform ({event_time})")
    plt.tight_layout()
    plt.savefig(fname)


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


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(df.duration, df.snr, c=df.ml_confidence)
axes[0].set_xlabel("Duration (s)")
axes[0].set_ylabel("SNR")
axes[0].set_title(f"{len(df)} L1-O3b Blip Glitches")
cm = axes[1].scatter(filtered.duration, filtered.snr, c=filtered.ml_confidence)
axes[1].set_xlabel("Duration (s)")
axes[1].set_ylabel("SNR")
axes[1].set_title(f"{len(filtered)} Filtered Blip Glitches")
# add colorbar shared
cbar = fig.colorbar(cm, ax=axes[1], orientation='vertical', fraction=0.02, pad=0.04, )
cbar.set_label("ML confidence")
plt.tight_layout()
plt.savefig("snr_vs_duration_filtered.png")




# Step 3: Download data for first 100 entries
for idx, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Downloading glitch data"):
    # Assuming 'time' column exists and is GPS time
    gps_time = row['peak_time']
    # Download 1 second of data around the event
    filename = os.path.join(OUTDIR, f"glitch_{gps_time}.hdf5")

    if os.path.exists(filename):
        ts = TimeSeries.read(filename, format='hdf5')
    else:
        ts = TimeSeries.fetch_open_data('L1', gps_time - 10, gps_time + 10)
        ts.write(filename, format='hdf5', overwrite=True)
        print(f"Downloaded data for GPS {gps_time}")

    # center the time series around the glitch
    event_time = row['event_time']
    plot_gravityspy_qtransform(ts, os.path.join(OUTDIR, f"glitch_{gps_time}_spectrogram.png"), event_time)
