# BayesWave baseline for the real-noise population

This workflow evaluates BayesWave's signal-versus-glitch log Bayes factor on
the exact H1-L1 events prepared by `studies/real_noise_event.py`.  The primary
comparison statistic is

```text
log_bayeswave_signal_glitch = logZ_signal - logZ_glitch
```

The recovered signal SNR is saved as a diagnostic, not used as the primary
ranking statistic.

## What is required

- The existing H1-L1 `manifest.json` files and their referenced HDF5 bundles,
  normally under `slurm/out/rn_H1_L1/e*/` on OzSTAR.
- BayesWave and BayesWavePost 1.1.3.
- Python in the same environment with `numpy`, `h5py`, and `gwpy`.  GWPy writes
  the short HDF5 strain arrays as GWF frames for BayesWave.
- The runner downsamples the 4096 Hz event bundles to 2048 Hz. BayesLine is fit
  over 300--1024 Hz while signal/glitch wavelets are restricted to the paper's
  300--800 Hz band. BayesLine's initial transform requires a power-of-two sample
  count, so using 800 Hz itself as the detector cutoff is not supported. The
  extra 800--1024 Hz bins contain no transient wavelets in either hypothesis and
  therefore enter both evidences through the same noise model.
- Four CPUs per run by default.  The supplied configuration uses 20 parallel
  tempering chains in groups of four threads, 10 GB RAM, and a 24-hour SLURM
  limit with hourly BayesWave checkpoints.
- Shared output storage for chains and post-processing products. Keep the
  eight-run pilot until its per-run disk usage is measured; budget tens of GB
  for the 100-run campaign. The supplied SLURM script defaults to
  `/fred/oz980/avajpeyi/results/starccato_lvk/bayeswave_H1_L1`, outside the Git
  checkout; override `OUTPUT_ROOT` if needed.

The Starccato JAX environment is not used for sampling.  The runner only needs
the repository on `PYTHONPATH` so it can read the event manifest and bundle.

## Install on OzSTAR

Create an isolated prefix; do not modify the Starccato virtual environment:

```bash
module load mamba
mamba create -y \
  -p /fred/oz980/avajpeyi/envs/bayeswave \
  -c conda-forge --strict-channel-priority \
  'bayeswave=1.1.3' 'bayeswaveutils=1.1.3' \
  gwpy h5py numpy
```

Validate the executable and frame-writing dependencies:

```bash
/fred/oz980/avajpeyi/envs/bayeswave/bin/BayesWave --help
/fred/oz980/avajpeyi/envs/bayeswave/bin/python -c \
  'import gwpy, h5py, numpy; print("BayesWave Python dependencies OK")'
```

If that prefix differs, export it before submission:

```bash
export BAYESWAVE_ENV=/path/to/bayeswave/environment
```

## Inspect one command without writing anything

From the `starccato_lvk` repository:

```bash
PYTHONPATH=src /path/to/bayeswave/bin/python \
  -m starccato_lvk.bayeswave \
  slurm/out/rn_H1_L1/e0/manifest.json \
  --class inj_ccsn \
  --output slurm/out/bayeswave_H1_L1/e0/inj_ccsn
```

This validates the manifest and bundles and prints the exact BayesWave and
BayesWavePost commands.  It does not create files or start sampling.

To test only HDF5-to-GWF conversion, add `--prepare-only`.  To run a deliberately
short end-to-end software smoke test, use settings such as
`--iterations 1000 --burnin 100 --chains 4 --threads 2 --execute`.  Those short
settings are not scientifically valid and their evidences must not be used.

## Run the timing pilot

The checked-in SLURM array runs four event indices, with one injected CCSN and
one real glitch per index (eight runs):

```bash
sbatch slurm/bayeswave_pilot.sh
```

Each run writes:

- `run_metadata.json`: input fingerprint, commands, settings, and elapsed time;
- `bayeswave.log` and `bayeswave_post.log`;
- `evidence.dat` plus the native BayesWave chains/checkpoints;
- `post/`: BayesWavePost reconstruction products;
- `result.json`: compact evidences, signal/glitch log Bayes factor, recovered
  signal SNR, uncertainties, runtime, and provenance.

Inspect all eight `result.json` files and logs before scaling.  In particular,
check that both signal and glitch evidences are finite, post-processing produced
`signal_stats.dat.geo`, checkpoints are working, and the 24-hour request is
adequate.

## Scale to the planned 100-event comparison

After the pilot passes:

```bash
sbatch --array=0-99 \
  slurm/bayeswave_pilot.sh
```

This maps 50 existing event indices to 100 runs: 50 injected CCSN signals and
the 50 paired real blips.  Rerunning the same array is safe: completed evidence
and post-processing products are reused, while incomplete BayesWave jobs can
resume from their checkpoints.

Do not reduce the production iteration, burn-in, or chain counts merely to make
the run finish.  If the timing pilot shows that the defaults are impractical,
record the failure and tune settings only with an explicit convergence study.

## BayesWave references

- [Installation](https://lscsoft.docs.ligo.org/bayeswave/install.html)
- [Running analyses](https://lscsoft.docs.ligo.org/bayeswave/running.html)
- [Output data products](https://lscsoft.docs.ligo.org/bayeswave/bayeswave-output.html)
