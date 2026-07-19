# starccato_lvk

Tools to help work with LVK data. The project covers two primary workflows:

1. **Data acquisition** – fetching GWOSC/OzSTAR strain segments and preparing analysis-ready chunks/PSDs.
2. **Data analysis** – running Starccato signal inference (supernova, blip, noise models) on prepared data.

Typical usage:

```bash
# Acquire 100 segments (noise + blip) into ./lvk_data
uv run starccato-lvk acquire batch 100 --outdir ./lvk_data

# Run Starccato analysis on an existing detector bundle
uv run starccato-lvk run ./outdir \
    --bundle H1=/path/to/analysis_bundle.hdf5 \
    --model ccsne

# Or acquire data around a GPS trigger before analysis
uv run starccato-lvk run ./outdir \
    --detector H1 \
    --trigger-time 1263743076 \
    --model ccsne
```

See `docs/` and `src/starccato_lvk/acquisition/README.md` for more details on configuration and advanced options.
