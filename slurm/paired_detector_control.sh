#!/bin/bash
# Paired one-vs-two-detector control (studies/paired_detector_control.py).
# One array task is one catalogue index and, for analysis, exactly one arm
# (which is three BCR runs: H1, L1, H1+L1 on the SAME injected data).
#
# Required submission variables:
#   CAMPAIGN_ID   e.g. pdc_v041_20260721
#
# Common optional variables:
#   STAGE (prep|analysis|both), ARM (noise|sig_off|sig_on), BLIP_IFO,
#   RESULTS_ROOT, VENV, FLOW, FMAX, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS
#
# The loss under investigation is concentrated in the H1-host cohort, so
# BLIP_IFO defaults to H1 (this selects the event GPS times, matching the
# rn_*_blipH1 campaign indices).
#
# Example:
#   sbatch --array=0-29%30 \
#     --export=ALL,CAMPAIGN_ID=pdc_v041_20260721,STAGE=prep \
#     slurm/paired_detector_control.sh
#   for arm in noise sig_off sig_on; do
#     sbatch --array=0-29%30 \
#       --export=ALL,CAMPAIGN_ID=pdc_v041_20260721,STAGE=analysis,ARM=$arm \
#       slurm/paired_detector_control.sh
#   done

#SBATCH --job-name=starccato_pdc
#SBATCH --account=oz303
#SBATCH --array=0-29%30
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/pdc_%A_%a.out
#SBATCH --error=slurm/logs/pdc_%A_%a.err

set -euo pipefail

CAMPAIGN_ID=${CAMPAIGN_ID:?set an immutable CAMPAIGN_ID}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
VENV=${VENV:-/fred/oz303/avajpeyi/codes/starccato_lvk/.venv}
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${INDEX_OFFSET:-0} ))
STAGE=${STAGE:-analysis}
ARM=${ARM:-}
BLIP_IFO=${BLIP_IFO:-H1}
FLOW=${FLOW:-300}
FMAX=${FMAX:-800}
NUM_WARMUP=${NUM_WARMUP:-500}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
NUM_CHAINS=${NUM_CHAINS:-2}

OUTDIR=${RESULTS_ROOT}/${CAMPAIGN_ID}/pdc_blip${BLIP_IFO}

case "${OUTDIR}" in
  "${REPO_ROOT}"/*)
    echo "Production output must be outside the Git checkout: ${OUTDIR}" >&2
    exit 2
    ;;
esac
if [[ "${STAGE}" == "analysis" && -z "${ARM}" ]]; then
  echo "Analysis submissions must set ARM to one arm" >&2
  exit 2
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing environment interpreter: ${VENV}/bin/python" >&2
  exit 2
fi

cd "${REPO_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing a production run from a dirty Git checkout" >&2
  git status --short >&2
  exit 2
fi
mkdir -p slurm/logs "${OUTDIR}"
export OMP_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}/studies:${PYTHONPATH:-}"
module load gcc/12.3.0 python/3.11.3
source "${VENV}/bin/activate"

RUNNER_ARGS=(
  --index "${INDEX}"
  --blip-ifo "${BLIP_IFO}"
  --stage "${STAGE}"
  --outdir "${OUTDIR}"
  --flow "${FLOW}"
  --fmax "${FMAX}"
  --num-warmup "${NUM_WARMUP}"
  --num-samples "${NUM_SAMPLES}"
  --num-chains "${NUM_CHAINS}"
)
if [[ -n "${ARM}" ]]; then
  RUNNER_ARGS+=(--arm "${ARM}")
fi

srun "${VENV}/bin/python" studies/paired_detector_control.py "${RUNNER_ARGS[@]}"
