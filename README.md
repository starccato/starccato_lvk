# starccato_lvk

Tools to help work with LVK data. The project covers two primary workflows:

1. **Data acquisition** – fetching GWOSC/OzSTAR strain segments and preparing analysis-ready chunks/PSDs.
2. **Data analysis** – running Starccato signal inference (supernova, blip, noise models) on prepared data.

Typical usage:

```bash
# Acquire 100 segments (noise + blip) into ./lvk_data
collect_lvk_data 100

# Run Starccato analysis on a bundled strain/PSD file
python -m starccato_lvk.cli run analysis_bundle.hdf5 ./outdir

# (Optional) run with separate strain/PSD files
python -m starccato_lvk.cli run analysis_chunk.hdf5 ./outdir --psd-path psd.hdf5

# Generate prior diagnostics (time/PSD overlays) before sampling
python -m starccato_lvk.cli run analysis_bundle.hdf5 ./outdir --diagnostics

The diagnostics PNG overlays the analysis data with prior samples in both time and frequency domains so you can tune priors before expensive runs.
```

See `docs/` and `src/starccato_lvk/acquisition/README.md` for more details on configuration and advanced options.
