# NUTS+MorphLnZ real-noise campaign

The production runner evaluates three classes for each detector cohort:

- `noise`: real detector noise;
- `inj_ccsn`: a held-out CCSN injected coherently into the selected network;
- `real_glitch`: a Gravity Spy blip in `BLIP_IFO` and coincident real noise in
  the other detector, when present.

The injected-blip class is retained as an opt-in sensitivity study but is not
part of the main campaign.

## 1. Synchronize and verify the environment

From the `starccato_lvk` checkout on an OzSTAR login node:

```bash
module load gcc/12.3.0 python/3.11.3
sacctmgr show assoc user="${USER}" format=Account,Partition,QOS
uv sync --frozen
uv run python -c \
  'import importlib.metadata as m; print(m.version("starccato-jax"), m.version("starccato-lvk"), m.version("morphZ"))'
```

The production lock requires `starccato-jax>=0.4.0`. Do not submit if the
verification prints an older model package. The helper defaults to account
`oz303`; set `SLURM_ACCOUNT` and `RESULTS_ROOT` if the association command shows
that this allocation or storage path has changed.

## 2. Pre-cache both Gravity Spy catalogues

Run this once on an internet-capable node. Compute nodes use the local O3
strain mirror and should not need network access.

```bash
uv run python - <<'PY'
from starccato_lvk.acquisition.io.glitch_catalog import load_blip_glitch_catalog
for ifo in ("H1", "L1"):
    print(ifo, len(load_blip_glitch_catalog(ifo=ifo)))
PY
```

The printed lengths define the valid upper bound for each catalogue. Do not
submit an index past the selected catalogue length.

## 3. Run a pilot

The submission helper launches three preparation arrays and nine analysis
arrays, one class per task. Outputs default to
`/fred/oz303/avajpeyi/results/starccato_lvk/<CAMPAIGN_ID>` and never enter Git.

```bash
N_EVENTS=2 MAX_CONCURRENT=1 \
  slurm/submit_nuts_morphlnz.sh nuts_morphlnz_v040_20260718
```

Inspect resource use and failures after the pilot:

```bash
sacct -X --starttime today \
  --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

Every successful result records the manifest fingerprint, package versions,
model-artifact hashes, injection identity, sampler configuration, per-detector
evidences, NUTS diagnostics, MAP initialization summary, MorphLnZ/fallback
status, and runtime. Failed convergence retains lightweight diagnostics below
the event's `analysis/` directory and writes a failure JSON.

## 4. Scale

For 250 catalogue indices, the nine analysis cohorts produce 2,250 analysis
tasks plus 750 preparation tasks:

```bash
N_EVENTS=250 MAX_CONCURRENT=10 \
  slurm/submit_nuts_morphlnz.sh nuts_morphlnz_v040_20260718
```

Use a new `CAMPAIGN_ID` whenever code, weights, frequency band, priors, or
sampler settings change. Matching tasks are idempotent. A stale result or
manifest causes a hard failure rather than being silently mixed into a new
campaign.

`MAX_CONCURRENT` is a per-array throttle. There are nine analysis arrays, so a
value of 10 allows at most 90 analysis tasks to run simultaneously, subject to
the cluster's own limits.

The default main-analysis band is 300--800 Hz. Band sensitivity campaigns must
use a distinct campaign ID and set `FLOW`/`FMAX` consistently for preparation
and analysis.

## 5. Collect

For each cohort, use its actual catalogue upper bound:

```bash
ROOT=/fred/oz303/avajpeyi/results/starccato_lvk/nuts_morphlnz_v040_20260718
uv run python studies/collect_results.py "${ROOT}/rn_L1" \
  --expected-start 0 --expected-stop 249
uv run python studies/collect_results.py "${ROOT}/rn_H1_blipH1" \
  --expected-start 0 --expected-stop 249
uv run python studies/collect_results.py "${ROOT}/rn_H1_L1" \
  --expected-start 0 --expected-stop 249
```

`collection_summary.json` reports class counts, missing pairs, recorded
exceptions, and the campaign ID. Selection for publication must use the saved
convergence/evidence diagnostics rather than simply taking the first N files.
