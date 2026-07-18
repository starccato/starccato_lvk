#!/bin/bash
# Submit the H1, L1, and H1-L1 NUTS+MorphLnZ production cohorts.
#
# Usage:
#   N_EVENTS=2 MAX_CONCURRENT=1 slurm/submit_nuts_morphlnz.sh CAMPAIGN_ID
#   N_EVENTS=250 MAX_CONCURRENT=10 slurm/submit_nuts_morphlnz.sh CAMPAIGN_ID

set -euo pipefail

CAMPAIGN_ID=${1:?usage: $0 CAMPAIGN_ID}
N_EVENTS=${N_EVENTS:-10}
MAX_CONCURRENT=${MAX_CONCURRENT:-10}
NETWORK_BLIP_IFO=${NETWORK_BLIP_IFO:-L1}
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-oz303}

if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]+$ ]]; then
  echo "CAMPAIGN_ID contains unsupported characters: ${CAMPAIGN_ID}" >&2
  exit 2
fi
if [[ ! "${N_EVENTS}" =~ ^[0-9]+$ || ! "${MAX_CONCURRENT}" =~ ^[0-9]+$ ]]; then
  echo "N_EVENTS and MAX_CONCURRENT must be integers" >&2
  exit 2
fi
if (( N_EVENTS < 1 || MAX_CONCURRENT < 1 )); then
  echo "N_EVENTS and MAX_CONCURRENT must be positive integers" >&2
  exit 2
fi
if [[ "${NETWORK_BLIP_IFO}" != "H1" && "${NETWORK_BLIP_IFO}" != "L1" ]]; then
  echo "NETWORK_BLIP_IFO must be H1 or L1" >&2
  exit 2
fi

ARRAY_SPEC="0-$((N_EVENTS - 1))%${MAX_CONCURRENT}"
CLASSES=(noise inj_ccsn real_glitch)

submit_prep() {
  local detectors=$1
  local blip_ifo=$2
  local glitch_det=$3
  sbatch --parsable --account="${SLURM_ACCOUNT}" --array="${ARRAY_SPEC}" \
    --export="ALL,CAMPAIGN_ID=${CAMPAIGN_ID},RESULTS_ROOT=${RESULTS_ROOT},STAGE=prep,DETECTORS=${detectors},BLIP_IFO=${blip_ifo},GLITCH_DET=${glitch_det}" \
    slurm/real_noise.sh
}

submit_analysis() {
  local dependency=$1
  local detectors=$2
  local blip_ifo=$3
  local glitch_det=$4
  local event_class=$5
  sbatch --parsable --account="${SLURM_ACCOUNT}" --array="${ARRAY_SPEC}" \
    --dependency="afterany:${dependency}" \
    --export="ALL,CAMPAIGN_ID=${CAMPAIGN_ID},RESULTS_ROOT=${RESULTS_ROOT},STAGE=analysis,CLASS=${event_class},DETECTORS=${detectors},BLIP_IFO=${blip_ifo},GLITCH_DET=${glitch_det}" \
    slurm/real_noise.sh
}

H1_PREP=$(submit_prep "H1" "H1" "H1")
L1_PREP=$(submit_prep "L1" "L1" "L1")
NETWORK_PREP=$(submit_prep "H1 L1" "${NETWORK_BLIP_IFO}" "${NETWORK_BLIP_IFO}")

printf "prep H1=%s L1=%s network=%s\n" "${H1_PREP}" "${L1_PREP}" "${NETWORK_PREP}"
for event_class in "${CLASSES[@]}"; do
  job=$(submit_analysis "${H1_PREP}" "H1" "H1" "H1" "${event_class}")
  printf "analysis H1 %-11s %s\n" "${event_class}" "${job}"
  job=$(submit_analysis "${L1_PREP}" "L1" "L1" "L1" "${event_class}")
  printf "analysis L1 %-11s %s\n" "${event_class}" "${job}"
  job=$(submit_analysis "${NETWORK_PREP}" "H1 L1" "${NETWORK_BLIP_IFO}" "${NETWORK_BLIP_IFO}" "${event_class}")
  printf "analysis H1-L1 %-8s %s\n" "${event_class}" "${job}"
done

printf "\nCampaign root: %s/%s\n" "${RESULTS_ROOT}" "${CAMPAIGN_ID}"
printf "Inspect the pilot with sacct before increasing N_EVENTS.\n"
