#!/usr/bin/env python3
"""
LVK Data Fetcher — download and extract LVK strain data (blip-glitch or noise)
using GWPy and PyCBC.

Features:
- Downloads glitch catalogs (H1/L1 O3a + O3b) from Zenodo.
- Downloads GWTC-2 & GWTC-3 event catalogs from PyCBC.
- For noise: randomly samples O3a+O3b, skips known events/glitches,
  and checks CBC CAT3 data-quality flag.
- For glitches: centers segment on glitch peak, filters by duration,
  and records SNR, glitch duration, and Q-factor.
- Saves strain, metadata, and a multi-panel Q-transform plot with
  adaptive color scale and Q-range.
"""

import os
import json
import random
import click
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from gwpy.timeseries import TimeSeries, StateVector
from pycbc.catalog import Catalog
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import logging
import warnings
from typing import Optional

# ---------------------------------------------------------------------
# Constants and Directories
# ---------------------------------------------------------------------
ZENODO_BASE = "https://zenodo.org/records/5649212/files"
GLITCH_FILES = ["L1_O3a.csv", "L1_O3b.csv", "H1_O3a.csv", "H1_O3b.csv"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GLITCH_DIR = os.path.join(DATA_DIR, "glitch_catalogs")
EVENT_CSV = os.path.join(DATA_DIR, "event_catalog.csv")

# O3 run GPS boundaries
O3A_START, O3A_END = 1238166018, 1253977218
O3B_START, O3B_END = 1256655618, 1269363618

# Set up logger
logger = logging.getLogger("lvk_data_fetcher")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

# Suppress specific GWPy warning
warnings.filterwarnings(
    "ignore",
    message=r"upper frequency of .* is too high for the given Q range, resetting to",
    category=UserWarning,
)

# ---------------------------------------------------------------------
# Catalog downloaders
# ---------------------------------------------------------------------
def _download_glitch_catalogs(verbose: bool = False) -> pd.DataFrame:
    """Download all Zenodo glitch CSVs and combine into one DataFrame."""
    os.makedirs(GLITCH_DIR, exist_ok=True)
    glitch_fn = os.path.join(GLITCH_DIR, "all_glitches.csv")
    if not os.path.exists(glitch_fn):
        if verbose:
            logger.info("Downloading glitch catalogs from Zenodo...")
        dfs = []
        for fname in GLITCH_FILES:
            url = f"{ZENODO_BASE}/{fname}"
            local = os.path.join(GLITCH_DIR, fname)
            if not os.path.exists(local):
                if verbose:
                    logger.info(f"Downloading {url}")
                r = requests.get(url)
                r.raise_for_status()
                with open(local, "wb") as f:
                    f.write(r.content)
            df = pd.read_csv(local)
            det, run = fname.split("_")
            run = run.replace(".csv", "")
            df["detector"], df["run"] = det, run
            dfs.append(df)
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.to_csv(os.path.join(GLITCH_DIR, "all_glitches.csv"), index=False)
        if verbose:
            logger.info(f"Combined glitch catalogs in {GLITCH_DIR}/all_glitches.csv")
    return pd.read_csv(glitch_fn)


def _download_event_catalog(verbose: bool = False) -> pd.DataFrame:
    """Download GWTC-2 & GWTC-3 event catalogs from PyCBC."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EVENT_CSV):
        return pd.read_csv(EVENT_CSV)
    if verbose:
        logger.info("Downloading GWTC-2 and GWTC-3 catalogs from PyCBC...")
    events = []
    for cat in [Catalog(source="gwtc-2"), Catalog(source="gwtc-3")]:
        for name in cat:
            ev = cat[name]
            events.append((name, float(ev.time)))
    df = pd.DataFrame(events, columns=["event_name", "event_time"])
    df.to_csv(EVENT_CSV, index=False)
    if verbose:
        logger.info(f"Saved event catalog to {EVENT_CSV}")
    return df


# ---------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------
def _is_near_event(event_df, gps_start, gps_end, margin=8.0):
    return any((gps_start - margin) <= t <= (gps_end + margin)
               for t in event_df["event_time"].values)


def _is_near_glitch(glitch_df, detector, gps_start, gps_end, margin=0.5):
    df = glitch_df[glitch_df["detector"] == detector]
    return any((gps_start - margin) <= t <= (gps_end + margin)
               for t in df["event_time"].values)


# ---------------------------------------------------------------------
# Randomized noise finder
# ---------------------------------------------------------------------
def _find_noise_segment(
    detector: str,
    idx: int,
    event_df: pd.DataFrame,
    glitch_df: pd.DataFrame,
    segment_len: float = 4.0,
    max_attempts: int = 50,
    verbose: bool = False
) -> float:
    run_ranges = [(O3A_START, O3A_END), (O3B_START, O3B_END)]
    rng = np.random.default_rng(seed=idx)
    attempts = 0
    while attempts < max_attempts:
        run_start, run_end = random.choice(run_ranges)
        gps_time = rng.uniform(run_start, run_end - segment_len)
        gps_start, gps_end = gps_time, gps_time + segment_len
        if _is_near_event(event_df, gps_start, gps_end) or \
           _is_near_glitch(glitch_df, detector, gps_start, gps_end):
            attempts += 1
            continue
        try:
            sv = StateVector.fetch_open_data(detector, gps_time - 65, gps_time + 65, verbose=False)
            dqflags = sv.to_dqflags()
            if "passes cbc CAT3 test" not in dqflags:
                attempts += 1
                continue
            cat3 = dqflags["passes cbc CAT3 test"]
            if any((gps_start >= s) and (gps_end <= e) for s, e in cat3.active):
                if verbose:
                    logger.info(f"Found CAT3-valid noise segment at GPS {gps_time:.0f} for {detector}")
                return gps_time
        except Exception as e:
            if verbose:
                logger.warning(f"Exception during CAT3 check: {e}")
            pass
        attempts += 1
    raise RuntimeError(f"Failed to find clean noise after {max_attempts} attempts.")


# ---------------------------------------------------------------------
# Q-transform plot helper
# ---------------------------------------------------------------------
def plot_qtransform(ts: TimeSeries, event_time: float, fname: str, qrange=(4, 64)):
    """Generate a multi-panel Q-transform plot with adaptive scaling."""
    q_scan = ts.q_transform(
        qrange=qrange, frange=[10, 2048],
        tres=0.002, fres=0.5, whiten=True
    )

    data_vals = q_scan.value.flatten()
    vmin = np.percentile(data_vals, 5)
    vmax = np.percentile(data_vals, 99.5)

    times = q_scan.times.value - event_time
    freqs = q_scan.frequencies.value
    duration = times[-1] - times[0]
    extent = [times[0], times[-1], freqs[0], freqs[-1]]

    # offsets are based on duration (most zoomed, till least zoomed (full duration for last))
    offsets = [0.06 * (duration / 4), 0.1 * (duration / 4), 0.25 * (duration / 4), duration / 2]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 4, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for ax, offset in zip(axes, offsets):
        im = ax.imshow(
            q_scan.value.T, aspect="auto", extent=extent,
            origin="lower", cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        ax.set_xlim(-offset, offset)
        ax.set_yscale("log", base=2)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_title(f"t0 ± {offset:.2f}s")
        ax.grid(False)

    fig.suptitle(f"Q-transform around {event_time:.3f} (relative to event)", fontsize=16)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------
def fetch_data(
    data_type: str,
    index: int,
    detector: str,
    outdir: str,
    duration: float,
    verbose: bool = False
) -> dict:
    base_dir = os.path.join(outdir, data_type)
    os.makedirs(base_dir, exist_ok=True)
    event_df = _download_event_catalog(verbose=verbose)
    glitch_df = _download_glitch_catalogs(verbose=verbose)
    if data_type == "blip-glitch":
        df = glitch_df[
            (glitch_df["detector"] == detector)
            & (glitch_df["ml_label"] == "Blip")
            & (glitch_df["duration"] <= duration)
        ].sort_values("snr", ascending=False).reset_index(drop=True)
        if index >= len(df):
            raise IndexError(f"Index {index} out of {len(df)} filtered blip glitches.")
        row = df.iloc[index]
        event_time = float(row["event_time"])
        glitch_dur = float(row["duration"])
        snr = float(row.get("snr", np.nan))
        q_factor = float(row.get("q", np.nan)) if "q" in row else None
        if q_factor and np.isfinite(q_factor):
            qrange = (max(4, q_factor / 2), min(64, q_factor * 2))
        else:
            qrange = (4, 64)
        start_time = event_time - duration / 2
        end_time = event_time + duration / 2
        source = "blip-glitch"
    else:
        gps_time = _find_noise_segment(detector, index, event_df, glitch_df, duration, verbose=verbose)
        start_time = gps_time
        end_time = gps_time + duration
        event_time = gps_time + duration / 2
        snr = None
        glitch_dur = None
        qrange = (4, 64)
        source = "random-cat3-noise"
    if verbose:
        logger.info(f"Fetching {detector} strain data from {start_time:.3f} to {end_time:.3f} (center={event_time:.3f})")
    data = TimeSeries.fetch_open_data(detector, start_time, end_time, verbose=verbose)
    prefix = os.path.join(base_dir, f"{int(event_time)}_{detector}")
    txt_path = prefix + ".txt"
    np.savetxt(txt_path, np.column_stack((data.times.value, data.value)))
    png_path = prefix + "_spectrogram.png"
    plot_qtransform(data, event_time, png_path, qrange=qrange)
    meta = dict(
        event_time=event_time,
        gps_start=start_time,
        gps_end=end_time,
        duration=duration,
        glitch_duration=glitch_dur,
        snr=snr,
        q_factor=q_factor if data_type == "blip-glitch" else None,
        detector=detector,
        data_type=data_type,
        source=source,
    )
    meta_path = prefix + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return dict(gps_time=event_time, txt=txt_path, png=png_path, meta=meta_path)


# ---------------------------------------------------------------------
# Multi-sample helper
# ---------------------------------------------------------------------
def fetch_multiple(
    data_type: str,
    n: int,
    base_outdir: str,
    duration: float,
    verbose: bool = False
):
    detectors = ["H1", "L1"]
    logger.info(f"Fetching {n} {data_type} samples for detectors: {detectors}")
    for det in detectors:
        for idx in tqdm(range(n), desc=f"{det} {data_type}"):
            try:
                fetch_data(data_type, idx, det, base_outdir, duration, verbose=verbose)
            except Exception as e:
                logger.warning(f"Warning: failed idx={idx} for {det}: {e}")
                continue


@click.command("lvk_fetch")
@click.option("--data-type", type=click.Choice(["blip-glitch", "noise"]), required=True)
@click.option("--index", type=int, default=None)
@click.option("--n", type=int, default=None)
@click.option("--detector", type=click.Choice(["H1", "L1"]), default=None)
@click.option("--outdir", type=click.Path(file_okay=False), default="./lvk_data", show_default=True)
@click.option("--duration", type=float, default=4.0, show_default=True)
@click.option("--verbose/--quiet", default=False, show_default=True, help="Enable verbose logging.")
def cli(data_type, index, n, detector, outdir, duration, verbose):
    """Fetch LVK strain data (single or multiple)."""
    logger.info("-- LVK Data Fetcher --")
    if n is not None:
        fetch_multiple(data_type, n, outdir, duration, verbose=verbose)
    else:
        if detector is None or index is None:
            raise click.UsageError("--detector and --index required for single fetch")
        fetch_data(data_type, index, detector, outdir, duration, verbose=verbose)
    logger.info("-- Done --")


if __name__ == "__main__":
    cli()
