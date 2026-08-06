#!/bin/bash
# Fetch the stratified O3a blip training sample (studies/blip_stratified_sample.py)
# using OzSTAR's local GWOSC mirror instead of the network GWOSC fetch.
#
# On this machine, fetching ~150 bundles over the network (GWOSC) ran at
# ~3.4 bundles/min combined across both detectors -- a ~480-bundle target
# would take ~2.5 hours. If OzSTAR mirrors O3a strain the same way it mirrors
# O3b (same gwosc.osgstorage.org layout, sibling run directory -- NOT
# verified from this session, check before relying on it:
#   ls /datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3a/strain.4k/hdf.v1/
# ), local reads should be disk-speed instead of network-speed.
#
# This is a plain batch job, not an array job: blip_stratified_sample.py
# already loops over many events in one process, and the whole point of local
# reads is that no per-event parallelism is needed to get the speed benefit.
#
# Required submission variables:
#   IFO   H1 or L1
#
# Common optional variables:
#   PER_BIN (default 40), RUN (default O3a), RESULTS_ROOT, VENV
#
# Example:
#   sbatch --export=ALL,IFO=L1 slurm/blip_o3a_fetch.sh
#   sbatch --export=ALL,IFO=H1 slurm/blip_o3a_fetch.sh

#SBATCH --job-name=starccato_blip_fetch
#SBATCH --account=oz303
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/blipfetch_%j.out
#SBATCH --error=slurm/logs/blipfetch_%j.err

set -euo pipefail

IFO=${IFO:?set IFO to H1 or L1}
RUN=${RUN:-O3a}
PER_BIN=${PER_BIN:-40}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
VENV=${VENV:-/fred/oz303/avajpeyi/codes/starccato_lvk/.venv}
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}

OUTDIR=${RESULTS_ROOT}/blip_train_${RUN,,}/${IFO}

case "${OUTDIR}" in
  "${REPO_ROOT}"/*)
    echo "Output must be outside the Git checkout: ${OUTDIR}" >&2
    exit 2
    ;;
esac
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing environment interpreter: ${VENV}/bin/python" >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p slurm/logs "${OUTDIR}"

# The one thing this script exists to set: read local mirrored strain for
# ${RUN} instead of O3b (config.py's hardcoded default) and instead of the
# network GWOSC fetch that dominated runtime locally.
export STARCCATO_LVK_GWOSC_RUN=${RUN}

"${VENV}/bin/python" studies/blip_stratified_sample.py \
  --ifo "${IFO}" --run "${RUN}" --per-bin "${PER_BIN}" --outdir "${OUTDIR}"

echo "Done. Bundle count: $(find "${OUTDIR}" -name 'analysis_bundle_*.hdf5' | wc -l)"
