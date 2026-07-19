# Simulation study

Generate or provide a CSV containing paired `noise_trigger` and `blip_trigger`
columns, then pass it explicitly to the study:

```bash
uv run generate_trigger_csv --help
uv run python studies/simulation_study.py \
    --config slurm/configs/analysis.yaml \
    --triggers-csv /path/to/triggers.csv \
    --index 0
```

The study runs the `noise`, `noise_inj`, and `blip` scenarios through the same
workflow used by the command-line and SLURM entry points.



