#!/bin/bash
# Idempotent submitter for the full real-noise campaign (all cohorts).
#
#   CAMPAIGN_ID=nuts_morphlnz_v044 SNR_REFERENCE_DET=H1 REQUIRE_CLEAN_NOISE=1 \
#     bash slurm/submit_campaign.sh
#
# Scans RESULTS_ROOT for existing manifests/results and submits ONLY the
# missing work. Safe to re-run at any time — completed work is never
# resubmitted, so this is also the resubmission tool after failures.
#
# Prep -> analysis dependency is PER EVENT, not per array: analysis task N
# depends on prep task N via SLURM's --dependency=aftercorr (task-to-task
# array correspondence), not on the whole prep array finishing. Concretely:
# for each <=1000-index window of newly-prepped events, the prep window is
# submitted first, and each class's analysis array for THAT SAME window of
# indices is submitted immediately after with aftercorr against that prep
# job -- so an event's analyses start as soon as its own prep finishes
# (rather than waiting for the slowest event in the whole array), and only
# run at all if prep actually SUCCEEDED (unlike afterany, which would let
# analysis run against a manifest that was never written). Events whose
# manifest already exists (from an earlier invocation, or because one class
# already ran) skip prep and submit analysis immediately with no dependency.
#
# Cohorts (paper table): L1, H1, H1+L1 with L1-hosted blips, H1+L1 with
# H1-hosted blips. The two network runs merge at aggregation time into the
# "blip host H1 or L1" cohort.
#
# Optional: RESULTS_ROOT, THROTTLE (concurrent tasks per array, default 50),
# COHORTS (override the cohort list, entries "DETECTORS|BLIP_IFO"),
# N_EVENTS (cap events per cohort, e.g. a pilot; default: full catalogue),
# TASK_BUDGET (max array tasks to submit this invocation; default: detected
# from the QOS per-user submit limit minus what is already queued),
# SNR_REFERENCE_DET, REQUIRE_CLEAN_NOISE, MAX_MEAN_WHITENED_POWER (forwarded
# to slurm/real_noise.sh; see that script's header for what they do).

set -euo pipefail

CAMPAIGN_ID=${CAMPAIGN_ID:?set an immutable CAMPAIGN_ID}
if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]+$ ]]; then
  echo "CAMPAIGN_ID contains unsupported characters: ${CAMPAIGN_ID}" >&2
  exit 2
fi
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-oz303}
THROTTLE=${THROTTLE:-50}
SNR_REFERENCE_DET=${SNR_REFERENCE_DET:-}
REQUIRE_CLEAN_NOISE=${REQUIRE_CLEAN_NOISE:-}
MAX_MEAN_WHITENED_POWER=${MAX_MEAN_WHITENED_POWER:-10.0}
CLASSES=(noise inj_ccsn real_glitch)
DATA_DIR=src/starccato_lvk/acquisition/io/data
DEFAULT_COHORTS=("L1|L1" "H1|H1" "H1 L1|L1" "H1 L1|H1")
# One cohort per line ("DETECTORS|BLIP_IFO"), so a network cohort's internal
# space (e.g. "H1 L1|L1") survives: splitting on spaces would corrupt it.
COHORT_LIST=()
while IFS= read -r _cohort; do
  [[ -n ${_cohort} ]] && COHORT_LIST+=("${_cohort}")
done <<< "${COHORTS:-}"
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

# sbatch_array STAGE CLASS DETECTORS BLIP_IFO INDEX_OFFSET LOCAL_LIST [DEPENDENCY]
# LOCAL_LIST is an already-windowed, already-0-offset comma list (i.e. actual
# index minus INDEX_OFFSET), matching what --array expects. Prints the job id.
sbatch_array() {
  local stage=$1 cls=$2 dets=$3 blip=$4 off=$5 list=$6 dep=${7:-}
  local exports dep_arg=()
  exports="ALL,CAMPAIGN_ID=${CAMPAIGN_ID},RESULTS_ROOT=${RESULTS_ROOT}"
  exports+=",DETECTORS=${dets},BLIP_IFO=${blip},STAGE=${stage},INDEX_OFFSET=${off}"
  [[ -n ${cls} ]] && exports+=",CLASS=${cls}"
  [[ -n ${SNR_REFERENCE_DET} ]] && exports+=",SNR_REFERENCE_DET=${SNR_REFERENCE_DET}"
  [[ -n ${REQUIRE_CLEAN_NOISE} ]] && exports+=",REQUIRE_CLEAN_NOISE=${REQUIRE_CLEAN_NOISE},MAX_MEAN_WHITENED_POWER=${MAX_MEAN_WHITENED_POWER}"
  [[ -n ${dep} ]] && dep_arg=(--dependency="${dep}")
  # command substitution does not inherit errexit, so a failure (typically
  # QOSMaxSubmitJobPerUserLimit) must be caught explicitly here or a dependent
  # array below would submit against an empty job id.
  # "${dep_arg[@]+...}" (not just "${dep_arg[@]}") because referencing an
  # EMPTY array under `set -u` is an unbound-variable error on bash < 4.4.
  sbatch --parsable --account="${SLURM_ACCOUNT}" ${dep_arg[@]+"${dep_arg[@]}"} \
    --array="${list}%${THROTTLE}" --export="${exports}" slurm/real_noise.sh
}

# windows IDX... : prints "off list" lines, one per <=1000-wide MaxArraySize
# window (list already offset-adjusted, comma-joined). Pure function of the
# (ascending) index list, so calling it twice on the SAME list reproduces the
# SAME off-boundaries -- the property the prep/analysis pairing relies on.
windows() {
  local max=${!#} off idx list
  for ((off = 0; off <= max; off += 1000)); do
    list=""
    for idx in "$@"; do
      (( idx >= off && idx < off + 1000 )) && list+="$(( idx - off )),"
    done
    [[ -n ${list} ]] && echo "${off} ${list%,}"
  done
}

on_sbatch_failure() {
  echo "Submission stopped (sbatch failed — usually the QOS per-user job" >&2
  echo "submit limit). Everything already submitted stands; re-run this" >&2
  echo "script once the queue drains to submit the remainder." >&2
  exit 3
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
  rejected=()
  for ((i = 0; i < n; i++)); do
    if [[ -f ${outdir}/e${i}/rejected.json ]]; then
      rejected+=("${i}")
    elif [[ ! -f ${outdir}/e${i}/manifest.json ]]; then
      prep_missing+=("${i}")
    fi
  done
  (( ${#rejected[@]} )) && echo "${label} excluded: ${#rejected[@]} trigger(s) flagged during strain acquisition"
  if (( ${#prep_missing[@]} > BUDGET )); then
    echo "${label} prep: ${#prep_missing[@]} tasks exceed remaining budget (${BUDGET}) — deferred with its analyses"
    DEFERRED=1
    continue  # analyses need the manifests, so defer the whole cohort
  fi

  # Prep, and its dependent analyses, window by window: submit the prep
  # window, then immediately submit every class's analysis array over the
  # SAME window's indices with aftercorr against that prep job. No state is
  # threaded between windows or out of this loop -- each window is entirely
  # self-contained, which is what keeps this correct without needing to track
  # job ids by offset across separate loops.
  if (( ${#prep_missing[@]} )); then
    prep_jobids="" analysis_jobids=""
    while read -r off list; do
      [[ -n ${off} ]] || continue
      prep_jid=$(sbatch_array prep "" "${dets}" "${blip}" "${off}" "${list}") || on_sbatch_failure
      prep_jobids+="${prep_jid}:"
      for cls in "${CLASSES[@]}"; do
        jid=$(sbatch_array analysis "${cls}" "${dets}" "${blip}" "${off}" "${list}" "aftercorr:${prep_jid}") \
          || on_sbatch_failure
        analysis_jobids+="${jid}:"
      done
    done < <(windows "${prep_missing[@]}")
    echo "${label} prep: ${#prep_missing[@]}/${n} events -> job(s) ${prep_jobids%:}"
    echo "${label} (all classes, newly-prepped events): -> job(s) ${analysis_jobids%:}"
    BUDGET=$(( BUDGET - ${#prep_missing[@]} * (1 + ${#CLASSES[@]}) ))
  else
    echo "${label} prep: complete (${n} events)"
  fi

  # Events that already had a manifest before this invocation (an earlier
  # run, or another class already completed for them): no prep dependency.
  # "has a manifest on disk" is exactly the complement of prep_missing (which
  # was built from the same test moments ago, and nothing has run since --
  # prep jobs are only QUEUED above), so test the file rather than searching
  # prep_missing.
  for cls in "${CLASSES[@]}"; do
    already_prepped=()
    for ((i = 0; i < n; i++)); do
      [[ -f ${outdir}/results/e${i}_${cls}.json ]] && continue
      [[ -f ${outdir}/e${i}/manifest.json ]] && already_prepped+=("${i}")
    done
    if (( ${#already_prepped[@]} == 0 )); then
      # Nothing left that is not already covered by the prep-dependent
      # submissions above; only genuinely say "complete" when there were none.
      (( ${#prep_missing[@]} )) || echo "${label} ${cls}: complete"
      continue
    fi
    if (( ${#already_prepped[@]} > BUDGET )); then
      echo "${label} ${cls} (already-prepped events): ${#already_prepped[@]} tasks exceed remaining budget (${BUDGET}) — deferred"
      DEFERRED=1
      continue
    fi
    jobids=""
    while read -r off list; do
      [[ -n ${off} ]] || continue
      jid=$(sbatch_array analysis "${cls}" "${dets}" "${blip}" "${off}" "${list}") || on_sbatch_failure
      jobids+="${jid}:"
    done < <(windows "${already_prepped[@]}")
    echo "${label} ${cls} (already-prepped events): ${#already_prepped[@]}/${n} tasks -> job(s) ${jobids%:}"
    BUDGET=$(( BUDGET - ${#already_prepped[@]} ))
  done

done

if (( DEFERRED )); then
  echo
  echo "Some work was deferred to respect the QOS submit limit."
  echo "Re-run this script once the queue drains; it submits only what is missing."
fi
