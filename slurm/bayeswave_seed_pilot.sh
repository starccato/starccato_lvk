#!/bin/bash
# Controlled BayesWave trust check: a pre-registered, class-balanced 24-event
# cohort from studies/select_bw_pilot_events.py, each run under three independent
# seeds with the matched 64 s off-source PSD window.
#
# The question this answers is deliberately sign-level: do the three seeds agree
# on whether BayesWave favors the class-expected hypothesis? Reported evidence
# uncertainties are recorded, but they do not select events or decide acceptance.
#
# The two methods use the same strain, trigger, 300-800 Hz wavelet band, fixed sky,
# and 64 s off-source PSD stretch stored in each manifest's analysis bundles.
#
# Prerequisite (writes pilot_tasks.txt next to pilot_cohort.json):
#   python studies/select_bw_pilot_events.py \
#     --ccsn-manifests ${RESULTS_ROOT}/${CCSN_CAMPAIGN}/data/rn_H1_L1 \
#     --glitch-manifests ${RESULTS_ROOT}/${GLITCH_CAMPAIGN}/data/rn_H1_L1 \
#     --out ${PILOT_ROOT}/pilot_cohort.json
#
# Submit the whole cohort (one array task per event x seed):
#   sbatch --array=0-71%24 slurm/bayeswave_seed_pilot.sh
#
# Optional overrides:
#   RESULTS_ROOT, PILOT_ROOT, TASK_FILE, CCSN_CAMPAIGN, GLITCH_CAMPAIGN
#   BW_NITER, BW_BURNIN, BW_NCHAIN, BW_THREADS, BW_CKPT_HRS, BAYESWAVE_ENV

#SBATCH --job-name=starccato_bwseed
#SBATCH --account=oz303
#SBATCH --array=0-71%24
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
# 1e6 iterations took ~1 h wall on the earlier cohort, so 4e6 needs ~4-5 h.
# The request is generous because BayesWave checkpoints hourly: a job that hits
# the wall is resumed by resubmitting the same array, but a job that fits first
# time saves a whole round trip.
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bwseed_%A_%a.out
#SBATCH --error=slurm/logs/bwseed_%A_%a.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
# Inputs (manifests, bundles) are read from RESULTS_ROOT on oz303, but the pilot
# WRITES to oz980. A BayesWave run holds ~600 files while sampling, so 72 runs
# at 24-way concurrency need ~15k inodes of headroom, and oz303 sits at 97% of
# its 1M-file group quota. Exhausting it does not just fail this job: every job
# in the group that cannot create its output file is killed at startup with
# RaisedSignal:53. Point PILOT_ROOT back at oz303 only after checking
# `lfs quota -g oz303 /fred`.
PILOT_ROOT=${PILOT_ROOT:-/fred/oz980/avajpeyi/results/starccato_lvk/bw_seed_pilot}
TASK_FILE=${TASK_FILE:-${PILOT_ROOT}/pilot_tasks.txt}
CCSN_CAMPAIGN=${CCSN_CAMPAIGN:-bwcomp_nml_v044_ccsn}
GLITCH_CAMPAIGN=${GLITCH_CAMPAIGN:-bwcomp_nml_v044_glitch}
BAYESWAVE_ENV=${BAYESWAVE_ENV:-/fred/oz980/avajpeyi/envs/bayeswave}

BW_NITER=${BW_NITER:-4000000}
BW_BURNIN=${BW_BURNIN:-400000}
BW_NCHAIN=${BW_NCHAIN:-20}
BW_THREADS=${BW_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
BW_CKPT_HRS=${BW_CKPT_HRS:-1.0}

[[ -f "${TASK_FILE}" ]] || {
  echo "No task list at ${TASK_FILE}; run studies/select_bw_pilot_events.py first" >&2
  exit 2
}
# Array task N runs line N+1. sed -n is exact: a task id past the end prints
# nothing and the emptiness check below catches it, rather than silently
# re-running the last line.
LINE=$(sed -n "$((TASK_ID + 1))p" "${TASK_FILE}")
[[ -n "${LINE}" ]] || { echo "No task ${TASK_ID} in ${TASK_FILE}" >&2; exit 2; }
read -r CLASS INDEX SEED <<<"${LINE}"

case "${CLASS}" in
  inj_ccsn)    CAMPAIGN=${CCSN_CAMPAIGN} ;;
  real_glitch) CAMPAIGN=${GLITCH_CAMPAIGN} ;;
  *) echo "Unsupported class ${CLASS} in ${TASK_FILE}" >&2; exit 2 ;;
esac

MANIFEST=${RESULTS_ROOT}/${CAMPAIGN}/data/rn_H1_L1/e${INDEX}/manifest.json
# One directory per seed. bayeswave.py refuses to reuse a directory whose
# recorded settings differ, so seeds must not share an output tree -- and this
# keeps the three evidences independently inspectable.
OUTPUT=${PILOT_ROOT}/e${INDEX}/${CLASS}/seed${SEED}

PYTHON=${BAYESWAVE_ENV}/bin/python
BAYESWAVE=${BAYESWAVE_ENV}/bin/BayesWave
BAYESWAVE_POST=${BAYESWAVE_ENV}/bin/BayesWavePost

[[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}" >&2; exit 2; }
for exe in "${PYTHON}" "${BAYESWAVE}" "${BAYESWAVE_POST}"; do
  [[ -x "${exe}" ]] || { echo "Missing executable: ${exe}" >&2; exit 2; }
done

cd "${REPO_ROOT}"
# The pilot's conclusion is a statement about a specific code state; a dirty
# checkout makes that statement unverifiable.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing a pilot run from a dirty Git checkout" >&2
  git status --short >&2; exit 2
fi
mkdir -p slurm/logs "${PILOT_ROOT}"
export OMP_NUM_THREADS=${BW_THREADS}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# BayesWavePost allocates large arrays on the stack and SIGSEGVs under SLURM's
# default stack ulimit while running fine interactively.
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null || true

echo "[task ${TASK_ID}] e${INDEX} ${CLASS} seed=${SEED} niter=${BW_NITER} -> ${OUTPUT}"
srun "${PYTHON}" -m starccato_lvk.bayeswave \
  "${MANIFEST}" \
  --class "${CLASS}" \
  --output "${OUTPUT}" \
  --bayeswave-executable "${BAYESWAVE}" \
  --post-executable "${BAYESWAVE_POST}" \
  --iterations "${BW_NITER}" \
  --burnin "${BW_BURNIN}" \
  --chains "${BW_NCHAIN}" \
  --threads "${BW_THREADS}" \
  --checkpoint-interval-hours "${BW_CKPT_HRS}" \
  --seed "${SEED}" \
  --execute

# Prune sampler scratch only once result.json proves the run completed. These
# campaigns exhaust the project's 1M-file inode quota otherwise, which kills
# unrelated jobs at startup with RaisedSignal:53. A failed run keeps everything,
# including checkpoint/, so a resubmit resumes and can be debugged.
if [[ "${KEEP_SCRATCH:-0}" != "1" && -f "${OUTPUT}/result.json" ]]; then
  rm -f  "${OUTPUT}"/core.*
  rm -rf "${OUTPUT}/checkpoint" "${OUTPUT}/chains" "${OUTPUT}/frames"
  rm -f  "${OUTPUT}"/*_PSD.dat "${OUTPUT}"/*_psd.dat "${OUTPUT}"/*_asd.dat
  rm -f  "${OUTPUT}"/*_priorpsd.dat "${OUTPUT}"/*_fairdraw_res.dat
  echo "[task ${TASK_ID}] pruned scratch in ${OUTPUT}"
fi
echo "[task ${TASK_ID}] done"
