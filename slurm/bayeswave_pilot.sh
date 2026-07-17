#!/bin/bash
# BayesWave signal-versus-glitch baseline for the existing H1-L1 real-noise
# population.  Each array task runs one class from one existing event manifest.
#
# The checked-in default is an 8-run timing/validation pilot:
#   indices 0-3 x {inj_ccsn, real_glitch}
#
# After validating those outputs, scale to the planned 100-run comparison:
#   sbatch --array=0-99 slurm/bayeswave_pilot.sh
#
# Required environment variables:
#   BAYESWAVE_ENV  conda/mamba prefix containing BayesWave and BayesWavePost
# Optional overrides:
#   MANIFEST_ROOT  existing rn_H1_L1 output (default below)
#   OUTPUT_ROOT    BayesWave output root
#   BW_NITER, BW_BURNIN, BW_NCHAIN, BW_THREADS
#
# Install command and validation are documented in docs/bayeswave_baseline.md.

#SBATCH --job-name=starccato_bw
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/bw_%A_%a.out
#SBATCH --error=slurm/logs/bw_%A_%a.err

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
EVENT_INDEX=$((TASK_ID / 2))
if (( TASK_ID % 2 == 0 )); then
  EVENT_CLASS=inj_ccsn
else
  EVENT_CLASS=real_glitch
fi

BAYESWAVE_ENV=${BAYESWAVE_ENV:-/fred/oz303/avajpeyi/envs/bayeswave}
MANIFEST_ROOT=${MANIFEST_ROOT:-slurm/out/rn_H1_L1}
OUTPUT_ROOT=${OUTPUT_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk/bayeswave_H1_L1}
BW_NITER=${BW_NITER:-1000000}
BW_BURNIN=${BW_BURNIN:-100000}
BW_NCHAIN=${BW_NCHAIN:-20}
BW_THREADS=${BW_THREADS:-${SLURM_CPUS_PER_TASK:-4}}

MANIFEST=${MANIFEST_ROOT}/e${EVENT_INDEX}/manifest.json
OUTPUT=${OUTPUT_ROOT}/e${EVENT_INDEX}/${EVENT_CLASS}
PYTHON=${BAYESWAVE_ENV}/bin/python
BAYESWAVE=${BAYESWAVE_ENV}/bin/BayesWave
BAYESWAVE_POST=${BAYESWAVE_ENV}/bin/BayesWavePost

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing event manifest: ${MANIFEST}" >&2
  exit 2
fi
for executable in "${PYTHON}" "${BAYESWAVE}" "${BAYESWAVE_POST}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Missing executable: ${executable}" >&2
    exit 2
  fi
done

export OMP_NUM_THREADS=${BW_THREADS}
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

srun "${PYTHON}" -m starccato_lvk.bayeswave \
  "${MANIFEST}" \
  --class "${EVENT_CLASS}" \
  --output "${OUTPUT}" \
  --bayeswave-executable "${BAYESWAVE}" \
  --post-executable "${BAYESWAVE_POST}" \
  --iterations "${BW_NITER}" \
  --burnin "${BW_BURNIN}" \
  --chains "${BW_NCHAIN}" \
  --threads "${BW_THREADS}" \
  --seed "$((1234 + TASK_ID))" \
  --execute
