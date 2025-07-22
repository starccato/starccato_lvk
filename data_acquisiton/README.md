# Data Acquisition 


We require data for three primary purposes:
- Analysis segments – time intervals containing either a signal, a glitch, or background noise.
- Glitches – labeled transient noise artifacts.
- Noise PSD estimation – LVK-style median-Welch PSDs for realistic noise modeling.

1. Glitch Data

We use publicly available glitch data from LIGO Livingston (L1) during O3b. 
Specifically, we download the CSV files from the Gravity Spy Glitch Catalog on Zenodo, 
as described in Zevin et al. (2017).
https://zenodo.org/records/5649212#.YrHBzC0lNQL
https://arxiv.org/pdf/1611.04596


We apply the following filters to isolate high-confidence blip glitches:
- ml_label == "Blip"
- ml_confidence ≥ 0.9
- duration ≥ 0.125 s (512 samples at 4096 Hz, matching the duration of our starccato signal)

2. Signal Injection Data
To simulate signals in realistic conditions, we grab background noise segments from L1 O3b data.
We need to ensure:
- Detector must be in observing mode (STATE_VECTOR == science)
- No known glitches or signals in the segment

We then inject Richer's CCSNe waveforms into these noise segments.


3. PSD Estimation
To compute an LVK-style noise Power Spectral Density (PSD):
- Extract 32 pre-event chunks from each analysis segment.
- Compute overlapping median Welch PSDs using standard LVK methodology.
- ensure data is the locked state (STATE_VECTOR == science).




We need data for 
1. analysis segments (section with signal/glitch/noise)
2. LVK styled median-welch PSD

## Glitches 
We access the public Glitch DB and download CSV files containing the information regarding from L1 O3b:


We filter this to only include:
- ml_label = "Blip"
- ml_confidence ≥ 0.9
- duration >= 0.125 seconds (ie 512 samples at 4096 Hz) -- the length of a CCSNe signal


## Signals 

We need to 
1. get data from L1 O3b noise when there is no signal/glitches and when the detector is in lock and
in observing mode
2. inject signal into this noise

"Open data from the third observing run of LIGO, Virgo, KAGRA and GEO"
https://dcc.ligo.org/public/0184/P2200316/015/main.pdf

We'll use flag 3: CBC_CAT3

For step 1
- Use data quality flags (DQ flags) to find times when L1 was in science/observing mode.
- Use your glitch list (e.g., full Gravity Spy CSV) to mask out glitch times.
- Use a catalog (e.g. GWTC-3) to mask out known signal times.


```python
import glob
import h5py
import numpy as np
import os
from gwpy.segments import SegmentList, segment

def get_noise_segments_from_file(h5file, bad_times=None, fs=4096, dq_flag="CBC_CAT3", min_duration=65):
    with h5py.File(h5file, 'r') as f:
        dqmask = f["quality/simple/DQmask"][:]
        shortnames = [s.decode() for s in f["quality/simple/DQShortnames"][:]]
        gps_start = f["meta/GPSstart"][()]

    if dq_flag not in shortnames:
        raise ValueError(f"{dq_flag} not found in DQShortnames: {shortnames}")

    idx = shortnames.index(dq_flag)
    good_mask = ((dqmask >> idx) & 1).astype(bool)

    # Find good intervals
    onsets = np.flatnonzero(np.diff(good_mask.astype(int)) == 1) + 1
    offsets = np.flatnonzero(np.diff(good_mask.astype(int)) == -1) + 1
    if good_mask[0]:
        onsets = np.insert(onsets, 0, 0)
    if good_mask[-1]:
        offsets = np.append(offsets, len(good_mask))

    science_segments = SegmentList([
        segment(gps_start + start/fs, gps_start + stop/fs)
        for start, stop in zip(onsets, offsets)
    ])

    if bad_times:
        bad_segments = SegmentList([
            segment(start, start + dur) for start, dur in bad_times
        ])
        science_segments -= bad_segments

    return SegmentList([s for s in science_segments if s.duration >= min_duration])


def get_random_noise_triggers_from_files(data_dir, N, seed=None):
    """
    Collect N random trigger times t0 from GWOSC O3b files such that:
        (t0 - 64, t0 + 1) lies entirely within a noise-only segment.
    
    Parameters
    ----------
    data_dir : str
        Path to root directory of GWOSC O3b HDF5 files.
    N : int
        Number of trigger times to return.
    seed : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    List[float] : list of GPS trigger times
    """
    rng = np.random.default_rng(seed)
    all_files = glob.glob(os.path.join(data_dir, "*", "*.hdf5"))
    rng.shuffle(all_files)

    collected_triggers = []
    for h5file in all_files:
        try:
            clean_segments = get_noise_segments_from_file(h5file, min_duration=65)
        except Exception as e:
            print(f"[warn] Skipping {os.path.basename(h5file)} due to: {e}")
            continue

        for seg in clean_segments:
            # Slide t0 across segment, ensure (t0-64, t0+1) in seg
            start = seg[0] + 64
            end = seg[1] - 1
            if end <= start:
                continue

            n_possible = int((end - start) // 1)
            if n_possible <= 0:
                continue

            t0s = rng.uniform(start, end, size=n_possible)
            rng.shuffle(t0s)

            for t0 in t0s:
                collected_triggers.append(t0)
                if len(collected_triggers) >= N:
                    return sorted(collected_triggers)

    raise RuntimeError(f"Only found {len(collected_triggers)} valid triggers. Needed {N}.")

```

Example Usage
```python
data_dir = "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3b/strain.4k/hdf.v1/L1"
triggers = get_random_noise_triggers_from_files(data_dir, N=10, seed=1234)

for i, t in enumerate(triggers):
    print(f"Trigger {i}: t0 = {t:.2f} (window: [{t - 64:.2f}, {t + 1:.2f}])")
```



TO GET TIMES FOR TRANSIENTS:
https://zenodo.org/records/5546665

/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/zenodo/ligo-virgo-kagra/2021/5546664/1/


## PSD 

Get 32 chunks of the analysis segments before the event, 
compute overlapping median Welch PSDs to get an estimate of the noise PSD.






## STRUCTURE

data_acquisition/
│
├── __init__.py
├── config.py                  # Constants (fs, segment length, thresholds, paths)
├── segment_model.py           # Defines AnalysisSegment dataclass
│
├── glitch_catalog/
│   ├── __init__.py
│   ├── download.py           # Download Gravity Spy CSVs
│   ├── filter.py             # Apply blip filters
│   ├── extract_segments.py   # Extract time series for each glitch
│   └── generate_dataset.py   # Orchestrates full glitch segment dataset creation
│
├── signal_injection/
│   ├── __init__.py
│   ├── segment_selector.py   # Select clean noise segments (science mode, no glitches)
│   ├── inject_signal.py      # Inject Richer CCSNe waveforms
│   └── generate_dataset.py   # Orchestrates signal + PSD generation
│
├── psd_estimation/
│   ├── __init__.py
│   ├── compute_psd.py        # LVK-style median Welch PSD
│   └── chunking.py           # Handles pre-event segment chunking
│
├── io/
|   ├── strain_loader.py      # loads time series from local GWOSC HDF5 files
│   ├── save.py               # Save `AnalysisSegment` to HDF5 or NPZ
│   ├── load.py               # Load `AnalysisSegment` from file
│   └── utils.py              # Filename templates, versioning, path logic
│
└── scripts/                  # Top-level orchestration scripts
    ├── generate_blip_dataset.py
    ├── generate_injected_dataset.py
    ├── inspect_segments.py
    └── visualize_examples.py




def get_science_segments(filename, fs=4096):
    with h5py.File(filename, 'r') as f:
        dqmask = f["quality/simple/DQmask"][:]
        shortnames = [s.decode() for s in f["quality/simple/DQShortnames"][:]]
        gps_start = f["meta/GPSstart"][()]
    # Find science mode bit (e.g., 'OBSERVATION')
    try:
        sci_idx = shortnames.index("OBSERVATION")
    except ValueError:
        raise RuntimeError("OBSERVATION flag not found in DQShortnames")
    # Find where bit is set
    is_science = ((dqmask >> sci_idx) & 1).astype(bool)
    # Convert to SegmentList
    onsets = np.flatnonzero(np.diff(is_science.astype(int)) == 1) + 1
    offsets = np.flatnonzero(np.diff(is_science.astype(int)) == -1) + 1
    # Edge cases: if science starts at beginning or ends at end
    if is_science[0]:
        onsets = np.insert(onsets, 0, 0)
    if is_science[-1]:
        offsets = np.append(offsets, len(is_science))
    times = SegmentList([
        segment(gps_start + onset / fs, gps_start + offset / fs)
        for onset, offset in zip(onsets, offsets)
    ])
    return times