# data_acquisition

This module handles:

- Extraction of CAT3-valid noise segments and glitch triggers
- Generation of analysis-ready strain chunks and median-Welch PSDs
- Optional diagnostic plotting for quick inspection

The primary entry point is the acquisition CLI:

```bash
# Acquire N samples (noise + blip) into ./lvk_data
uv run starccato-lvk acquire batch 100 --outdir ./lvk_data
```

Each trigger produces:

- `analysis_bundle_<t>.hdf5` – single file packaging the strain chunk, PSD, and metadata
- `analysis_chunk_<t>.png` – diagnostic plot (time series, PSD, Q-transform)

The bundle format is consumed directly by `run_starccato_analysis`, so no separate PSD path is required.
