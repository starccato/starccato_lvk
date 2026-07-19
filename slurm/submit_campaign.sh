#!/bin/bash
# Idempotent submitter for the full real-noise campaign (all cohorts).
#
#   CAMPAIGN_ID=nuts_morphlnz_v040 bash slurm/submit_campaign.sh
#
# Scans RESULTS_ROOT for existing manifests/results and submits ONLY the
# missing work: one prep array per cohort, then one analysis array per event
# class (dependent on prep). Safe to re-run at any time — completed work is
# never resubmitted, so this is also the resubmission tool after failures.
#
# Cohorts (paper table): L1, H1, H1+L1 with L1-hosted blips, H1+L1 with
# H1-hosted blips. The two network runs merge at aggregation time into the
# "blip host H1 or L1" cohort.
#
# Optional: RESULTS_ROOT, THROTTLE (concurrent tasks per array, default 50),
# COHORTS (override the cohort list, entries "DETECTORS|BLIP_IFO"),
# N_EVENTS (cap events per cohort, e.g. a pilot; default: full catalogue).

set -euo pipefail

CAMPAIGN_ID=${CAMPAIGN_ID:?set an immutable CAMPAIGN_ID}
if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]+$ ]]; then
  echo "CAMPAIGN_ID contains unsupported characters: ${CAMPAIGN_ID}" >&2
  exit 2
fi
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-oz303}
THROTTLE=${THROTTLE:-50}
CLASSES=(noise inj_ccsn real_glitch)
DATA_DIR=src/starccato_lvk/acquisition/io/data
DEFAULT_COHORTS=("L1|L1" "H1|H1" "H1 L1|L1" "H1 L1|H1")
read -r -a COHORT_LIST <<< "${COHORTS:-}"
(( ${#COHORT_LIST[@]} )) || COHORT_LIST=("${DEFAULT_COHORTS[@]}")

cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to submit from a dirty Git checkout (jobs re-check this too)" >&2
  exit 2
fi

n_events() {
  local csv=${DATA_DIR}/blip.csv
  [[ $1 != L1 ]] && csv=${DATA_DIR}/blip_$1.csv
  if [[ ! -f ${csv} ]]; then
    echo "Missing cached blip catalogue ${csv}; commit it before submitting" >&2
    exit 2
  fi
  echo $(( $(wc -l < "${csv}") - 1 ))
}

# submit_missing STAGE CLASS DETECTORS BLIP_IFO DEPENDENCY IDX...
# Windows the (ascending) indices so array values stay < 1000 (MaxArraySize),
# one sbatch per window with INDEX_OFFSET. Echoes job ids as a :-joined list.
submit_missing() {
  local stage=$1 cls=$2 dets=$3 blip=$4 dep=$5
  shift 5
  local max=${!#}
  local jobids="" exports list off idx jid
  for ((off = 0; off <= max; off += 1000)); do
    list=""
    for idx in "$@"; do
      (( idx >= off && idx < off + 1000 )) && list+="$(( idx - off )),"
    done
    [[ -n ${list} ]] || continue
    exports="ALL,CAMPAIGN_ID=${CAMPAIGN_ID},RESULTS_ROOT=${RESULTS_ROOT}"
    exports+=",DETECTORS=${dets},BLIP_IFO=${blip},STAGE=${stage},INDEX_OFFSET=${off}"
    [[ -n ${cls} ]] && exports+=",CLASS=${cls}"
    # command substitution does not inherit errexit, so failures (typically
    # QOSMaxSubmitJobPerUserLimit) must be caught explicitly or the script
    # would submit dependent arrays against empty job ids
    if [[ -n ${dep} ]]; then
      jid=$(sbatch --parsable --account="${SLURM_ACCOUNT}" \
        --dependency="afterany:${dep}" \
        --array="${list%,}%${THROTTLE}" --export="${exports}" slurm/real_noise.sh) \
        || exit 3
    else
      jid=$(sbatch --parsable --account="${SLURM_ACCOUNT}" \
        --array="${list%,}%${THROTTLE}" --export="${exports}" slurm/real_noise.sh) \
        || exit 3
    fi
    jobids+="${jid}:"
  done
  echo "${jobids%:}"
}

for spec in "${COHORT_LIST[@]}"; do
  dets=${spec%|*}
  blip=${spec#*|}
  dettag=${dets// /_}
  outdir=${RESULTS_ROOT}/${CAMPAIGN_ID}/rn_${dettag}
  [[ ${blip} != L1 ]] && outdir=${outdir}_blip${blip}
  n=$(n_events "${blip}")
  [[ -n ${N_EVENTS:-} ]] && (( N_EVENTS < n )) && n=${N_EVENTS}
  label="[${dettag} blip=${blip}]"

  prep_missing=()
  for ((i = 0; i < n; i++)); do
    [[ -f ${outdir}/e${i}/manifest.json ]] || prep_missing+=("${i}")
  done
  prep_dep=""
  if (( ${#prep_missing[@]} )); then
    if ! prep_dep=$(submit_missing prep "" "${dets}" "${blip}" "" "${prep_missing[@]}"); then
      echo "Submission stopped (sbatch failed — usually the QOS per-user job" >&2
      echo "submit limit). Everything already submitted stands; re-run this" >&2
      echo "script once the queue drains to submit the remainder." >&2
      exit 3
    fi
    echo "${label} prep: ${#prep_missing[@]}/${n} events -> job(s) ${prep_dep}"
  else
    echo "${label} prep: complete (${n} events)"
  fi

  for cls in "${CLASSES[@]}"; do
    missing=()
    for ((i = 0; i < n; i++)); do
      [[ -f ${outdir}/results/e${i}_${cls}.json ]] || missing+=("${i}")
    done
    if (( ${#missing[@]} == 0 )); then
      echo "${label} ${cls}: complete"
      continue
    fi
    if ! jid=$(submit_missing analysis "${cls}" "${dets}" "${blip}" "${prep_dep}" "${missing[@]}"); then
      echo "Submission stopped (sbatch failed — usually the QOS per-user job" >&2
      echo "submit limit). Everything already submitted stands; re-run this" >&2
      echo "script once the queue drains to submit the remainder." >&2
      exit 3
    fi
    echo "${label} ${cls}: ${#missing[@]}/${n} tasks -> job(s) ${jid}"
  done
done
