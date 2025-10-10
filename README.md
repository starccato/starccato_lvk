# starccato_lvk

Tools to help work with LVK data. The project covers two primary workflows:

1. **Data acquisition** – fetching GWOSC/OzSTAR strain segments and preparing analysis-ready chunks/PSDs.
2. **Data analysis** – running Starccato signal inference (supernova, blip, noise models) on prepared data.

Typical usage:

```bash
# Acquire 100 segments (noise + blip) into ./lvk_data
collect_lvk_data 100

# Run Starccato analysis on an analysis/PSD pair
run_starccato_analysis analysis_chunk.hdf5 psd.hdf5 ./outdir
```

See `docs/` and `src/starccato_lvk/acquisition/README.md` for more details on configuration and advanced options.
