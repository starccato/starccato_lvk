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
# N_EVENTS (cap events per cohort, e.g. a pilot; default: full catalogue),
# TASK_BUDGET (max array tasks to submit this invocation; default: detected
# from the QOS per-user submit limit minus what is already queued).

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

# Per-user submit-limit budget: whole cohort-class chunks that do not fit are
# deferred to a later invocation instead of slamming into sbatch rejections.
detect_budget() {
  local qos limit queued
  qos=$(sacctmgr -n show assoc user="${USER}" account="${SLURM_ACCOUNT}" \
    format=qos 2>/dev/null | awk 'NR==1{print $1}')
  limit=$(sacctmgr -n show qos "${qos}" format=maxsubmitpu 2>/dev/null \
    | awk 'NR==1{print $1}')
  queued=$(squeue -u "${USER}" -h -r 2>/dev/null | wc -l)
  if [[ ${limit} =~ ^[0-9]+$ ]]; then
    echo $(( limit - queued - 200 ))  # 200-task safety margin
  else
    echo 1000000  # no detectable limit; fail-fast on sbatch is the backstop
  fi
}
BUDGET=${TASK_BUDGET:-$(detect_budget)}
if (( BUDGET < 1 )); then
  echo "No submission budget left (queue is full against the QOS limit)." >&2
  echo "Re-run once jobs drain: completed work is skipped automatically." >&2
  exit 0
fi
echo "Task budget this invocation: ${BUDGET}"
DEFERRED=0

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
  if (( ${#prep_missing[@]} > BUDGET )); then
    echo "${label} prep: ${#prep_missing[@]} tasks exceed remaining budget (${BUDGET}) — deferred with its analyses"
    DEFERRED=1
    continue  # analyses need the manifests, so defer the whole cohort
  fi
  if (( ${#prep_missing[@]} )); then
    if ! prep_dep=$(submit_missing prep "" "${dets}" "${blip}" "" "${prep_missing[@]}"); then
      echo "Submission stopped (sbatch failed — usually the QOS per-user job" >&2
      echo "submit limit). Everything already submitted stands; re-run this" >&2
      echo "script once the queue drains to submit the remainder." >&2
      exit 3
    fi
    echo "${label} prep: ${#prep_missing[@]}/${n} events -> job(s) ${prep_dep}"
    BUDGET=$(( BUDGET - ${#prep_missing[@]} ))
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
    if (( ${#missing[@]} > BUDGET )); then
      echo "${label} ${cls}: ${#missing[@]} tasks exceed remaining budget (${BUDGET}) — deferred"
      DEFERRED=1
      continue
    fi
    if ! jid=$(submit_missing analysis "${cls}" "${dets}" "${blip}" "${prep_dep}" "${missing[@]}"); then
      echo "Submission stopped (sbatch failed — usually the QOS per-user job" >&2
      echo "submit limit). Everything already submitted stands; re-run this" >&2
      echo "script once the queue drains to submit the remainder." >&2
      exit 3
    fi
    echo "${label} ${cls}: ${#missing[@]}/${n} tasks -> job(s) ${jid}"
    BUDGET=$(( BUDGET - ${#missing[@]} ))
  done
done

if (( DEFERRED )); then
  echo
  echo "Some work was deferred to respect the QOS submit limit."
  echo "Re-run this script once the queue drains; it submits only what is missing."
fi
