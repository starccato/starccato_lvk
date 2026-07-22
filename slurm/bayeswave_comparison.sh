#!/bin/bash
# BayesWave-vs-lnO comparison for ONE event: BayesWave (+ optionally our NUTS
# posterior) plus a comparison plot. Per-index events are deterministic
# (seeded by index), so e<N> here is the same physical event as e<N> in any
# lnO campaign on this code.
#
# Run slurm/bayeswave_prep.sh FIRST. Prep (fetching/building the strain
# bundles) OOM'd when it shared a job with this script at --mem=4G -- it is
# I/O-heavy in a way that does not fit a small memory budget, so it is now a
# separate, dedicated job. This script assumes the manifest + bundles already
# exist and errors clearly if they do not.
#
# Stages (each idempotent -- skipped when its output exists, so a resubmit
# resumes rather than redoing work; BayesWave itself resumes from checkpoint):
#   1. (optional, DATA_STAGE=analysis) our NUTS posterior, for the comparison
#      plot -- --save-artifacts keeps analysis/ instead of pruning it.
#   2. BayesWave, fixed sky (the production setting -- matches lnO, which also
#      analyses at the known per-event sky). RUN_FREESKY=1 adds the free-sky
#      diagnostic in parallel.
#   3. Comparison plot (our VAE posterior vs BayesWave reconstruction, both
#      whitened) and a one-line lnBF summary.
#
# Required submission variable:
#   CAMPAIGN_ID   immutable tag, matching the bayeswave_prep.sh run
#
# Pilot (want the comparison plot -- runs our NUTS posterior too):
#   PREP=$(sbatch --parsable --array=0-3 \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260723 slurm/bayeswave_prep.sh)
#   sbatch --array=0-3 --dependency=aftercorr:${PREP} \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260723,DATA_STAGE=analysis \
#     slurm/bayeswave_comparison.sh          # and again with CLASS=real_glitch
#
# Population (~100/class, after validating the pilot): BayesWave only, no NUTS
# re-run and no plots -- lnO comes from the existing campaign at the same
# indices, paired offline by (index, class):
#   PREP=$(sbatch --parsable --array=0-99%50 \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260723 slurm/bayeswave_prep.sh)
#   sbatch --array=0-99%50 --dependency=aftercorr:${PREP} \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260723,RUN_DATA=0,RUN_PLOTS=0 \
#     slurm/bayeswave_comparison.sh          # and again with CLASS=real_glitch
#
# (aftercorr starts each analysis task as soon as its OWN prep task finishes;
# afterok:${PREP} waits for the whole prep array first, if preferred.)
#
# Common optional variables:
#   CLASS (inj_ccsn), DETECTORS ("H1 L1"), RUN_FREESKY (0), PARALLEL (1)
#   RUN_DATA (1)/RUN_BW/RUN_PLOTS, DATA_STAGE (both), INDEX_OFFSET
#   BW_NITER, BW_BURNIN, BW_NCHAIN, BW_THREADS, BW_CKPT_HRS, PLOT_IFO
#   VENV, BAYESWAVE_ENV, RESULTS_ROOT, SNR_REFERENCE_DET

#SBATCH --job-name=starccato_bwcmp
#SBATCH --account=oz303
#SBATCH --array=0-3
# 4 CPUs = one BayesWave at BW_THREADS=4. With RUN_FREESKY=1 submit
# --cpus-per-task=8 so the two parallel runs do not share cores.
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/bwcmp_%A_%a.out
#SBATCH --error=slurm/logs/bwcmp_%A_%a.err

set -euo pipefail

CAMPAIGN_ID=${CAMPAIGN_ID:?set an immutable CAMPAIGN_ID}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${INDEX_OFFSET:-0} ))

CLASS=${CLASS:-inj_ccsn}                 # inj_ccsn / inj_glitch / real_glitch
DETECTORS=${DETECTORS:-"H1 L1"}          # BayesWave sig-vs-glitch needs >=2 IFOs
PARALLEL=${PARALLEL:-1}                   # run the two sky settings concurrently
# Free sky is a diagnostic, OFF by default: the e0 pilot showed its evidence
# does not converge at 1e6 iters (lnZ_S-N=-0.78 while reconstructing SNR 18.8,
# and lnZ_glitch shifted 16 nats with no sky parameter in that model). Fixed
# sky also matches our lnO configuration. RUN_FREESKY=1 re-enables it (then
# submit with --cpus-per-task=8 so both runs get ${BW_THREADS} threads).
RUN_FREESKY=${RUN_FREESKY:-0}
RUN_DATA=${RUN_DATA:-1}; RUN_BW=${RUN_BW:-1}; RUN_PLOTS=${RUN_PLOTS:-1}
# both = prep + our NUTS posterior (needed for reconstruction PLOTS; ~30 min).
# prep = bundles/manifest only (~minutes) -- the population mode: BayesWave only
# needs the strain bundles, and the paired lnO values come from the existing
# campaign's results at the same indices (events are index-seeded).
DATA_STAGE=${DATA_STAGE:-both}
PLOT_IFO=${PLOT_IFO:-H1}

VENV=${VENV:-/fred/oz303/avajpeyi/codes/starccato_lvk/.venv}
BAYESWAVE_ENV=${BAYESWAVE_ENV:-/fred/oz980/avajpeyi/envs/bayeswave}
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
SNR_REFERENCE_DET=${SNR_REFERENCE_DET:-}  # empty = normalise on network SNR

BW_NITER=${BW_NITER:-1000000}; BW_BURNIN=${BW_BURNIN:-100000}
BW_NCHAIN=${BW_NCHAIN:-20}; BW_THREADS=${BW_THREADS:-4}; BW_CKPT_HRS=${BW_CKPT_HRS:-1.0}
# BayesWave requires burnin < iterations; keep small-Niter smoke tests valid.
if (( BW_BURNIN >= BW_NITER )); then BW_BURNIN=$(( BW_NITER / 10 )); fi

case "${CLASS}" in inj_ccsn|inj_glitch|real_glitch) ;; *)
  echo "CLASS=${CLASS} is not a transient class (noise has nothing to reconstruct)" >&2
  exit 2 ;;
esac

DETTAG=$(echo "${DETECTORS}" | tr ' ' '_')
CAMP_ROOT=${RESULTS_ROOT}/${CAMPAIGN_ID}
DATA_ROOT=${CAMP_ROOT}/data/rn_${DETTAG}          # our pipeline output (kept artifacts)
MANIFEST=${DATA_ROOT}/e${INDEX}/manifest.json
OUR_SAMPLES=${DATA_ROOT}/e${INDEX}/${CLASS}/analysis/signal/samples.npz
BW_FIXED=${CAMP_ROOT}/bw_fixedsky/e${INDEX}/${CLASS}
BW_FREE=${CAMP_ROOT}/bw_freesky/e${INDEX}/${CLASS}
PLOT_DIR=${CAMP_ROOT}/plots

case "${CAMP_ROOT}" in "${REPO_ROOT}"/*)
  echo "Output must be outside the Git checkout: ${CAMP_ROOT}" >&2; exit 2 ;;
esac
# Two interpreters: the project venv runs our JAX pipeline (stage 1) and the
# decoder-backed plotter (stage 3); the BayesWave env runs bayeswave.py, which
# writes GWF frames with gwpy that the project venv does not carry.
PYTHON=${VENV}/bin/python
BW_PYTHON=${BAYESWAVE_ENV}/bin/python
BAYESWAVE=${BAYESWAVE_ENV}/bin/BayesWave
BAYESWAVE_POST=${BAYESWAVE_ENV}/bin/BayesWavePost
for exe in "${PYTHON}" "${BW_PYTHON}" "${BAYESWAVE}" "${BAYESWAVE_POST}"; do
  [[ -x "${exe}" ]] || { echo "Missing executable: ${exe}" >&2; exit 2; }
done

cd "${REPO_ROOT}"
# A dirty checkout means the code that produced these results is unrecorded; the
# whole point of the comparison is reproducibility, so refuse it.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing a comparison run from a dirty Git checkout" >&2
  git status --short >&2; exit 2
fi
mkdir -p slurm/logs "${CAMP_ROOT}" "${PLOT_DIR}"
export OMP_NUM_THREADS=${BW_THREADS}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
module load gcc/12.3.0 python/3.11.3
source "${VENV}/bin/activate"
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null || true

# --- Stage 1: generate data (and, in "both" mode, our posterior)
STAGE1_DONE=${OUR_SAMPLES}
[[ "${DATA_STAGE}" == "prep" ]] && STAGE1_DONE=${MANIFEST}
if [[ "${RUN_DATA}" == "1" && ! -f "${STAGE1_DONE}" ]]; then
  echo "[e${INDEX}] stage 1 (${DATA_STAGE}): generating data"
  # Injected classes need the noise bundle to inject into; real_glitch does not.
  PREP_CLASSES=("${CLASS}")
  [[ "${CLASS}" == inj_* ]] && PREP_CLASSES=(noise "${CLASS}")
  DATA_ARGS=(
    --index "${INDEX}" --detectors ${DETECTORS} --stage "${DATA_STAGE}"
    --class "${CLASS}" --prep-classes "${PREP_CLASSES[@]}"
    --campaign-id "${CAMPAIGN_ID}" --outdir "${DATA_ROOT}" --save-artifacts
  )
  [[ -n "${SNR_REFERENCE_DET}" ]] && DATA_ARGS+=(--snr-reference-det "${SNR_REFERENCE_DET}")
  srun "${PYTHON}" studies/real_noise_event.py "${DATA_ARGS[@]}"
else
  echo "[e${INDEX}] stage 1: skipped (have ${STAGE1_DONE} or RUN_DATA=0)"
fi
[[ -f "${MANIFEST}" ]] || { echo "No manifest at ${MANIFEST}; cannot run BayesWave" >&2; exit 2; }

# --- Stage 2: BayesWave, fixed sky (setting 1) and free sky (setting 2)
run_bayeswave() {  # $1=output_dir  $2..=extra flags (e.g. --free-sky)
  local out=$1; shift
  "${BW_PYTHON}" -m starccato_lvk.bayeswave "${MANIFEST}" \
    --class "${CLASS}" --output "${out}" \
    --bayeswave-executable "${BAYESWAVE}" --post-executable "${BAYESWAVE_POST}" \
    --iterations "${BW_NITER}" --burnin "${BW_BURNIN}" \
    --chains "${BW_NCHAIN}" --threads "${BW_THREADS}" \
    --checkpoint-interval-hours "${BW_CKPT_HRS}" \
    --seed "$((1234 + INDEX))" --execute "$@"
}
if [[ "${RUN_BW}" == "1" ]]; then
  echo "[e${INDEX}] stage 2: BayesWave fixed-sky -> ${BW_FIXED}, free-sky -> ${BW_FREE}"
  # Stage logs live in CAMP_ROOT (already created); the bw_* trees do not exist
  # until bayeswave.py makes them, so redirecting into them would fail.
  LOG_FIXED=${CAMP_ROOT}/e${INDEX}_${CLASS}_fixed.log
  LOG_FREE=${CAMP_ROOT}/e${INDEX}_${CLASS}_free.log
  if [[ "${RUN_FREESKY}" != "1" ]]; then
    run_bayeswave "${BW_FIXED}" 2>&1 | tee "${LOG_FIXED}"
  elif [[ "${PARALLEL}" == "1" ]]; then
    run_bayeswave "${BW_FIXED}"           > "${LOG_FIXED}" 2>&1 & pid_fixed=$!
    run_bayeswave "${BW_FREE}" --free-sky > "${LOG_FREE}"  2>&1 & pid_free=$!
    fail=0; wait "${pid_fixed}" || fail=1; wait "${pid_free}" || fail=1
    if [[ "${fail}" == "1" ]]; then
      echo "A BayesWave run failed; log tails:" >&2
      tail -8 "${LOG_FIXED}" "${LOG_FREE}" >&2
      exit 1
    fi
  else
    run_bayeswave "${BW_FIXED}"
    run_bayeswave "${BW_FREE}" --free-sky
  fi
else
  echo "[e${INDEX}] stage 2: skipped (RUN_BW=0)"
fi

# --- Stage 3: comparison plots + lnBF summary
LABELS=(fixedsky); [[ "${RUN_FREESKY}" == "1" ]] && LABELS+=(freesky)
if [[ "${RUN_PLOTS}" == "1" ]]; then
  for label in "${LABELS[@]}"; do
    bw=${BW_FIXED}; [[ "${label}" == "freesky" ]] && bw=${BW_FREE}
    post=${bw}/post/signal
    out=${PLOT_DIR}/e${INDEX}_${CLASS}_${label}.pdf
    if [[ -f "${OUR_SAMPLES}" && -d "${post}" ]]; then
      "${PYTHON}" studies/plot_waveform_reconstruction.py \
        --our-samples "${OUR_SAMPLES}" --bayeswave-post "${post}" \
        --ifo "${PLOT_IFO}" --out "${out}" || echo "plot ${label} failed (see above)"
    else
      echo "[e${INDEX}] plot ${label} skipped: missing ${OUR_SAMPLES} or ${post}"
    fi
  done
  echo "[e${INDEX}] lnBF(signal-glitch) summary:"
  for label in "${LABELS[@]}"; do
    r=${BW_FIXED}/result.json; [[ "${label}" == "freesky" ]] && r=${BW_FREE}/result.json
    [[ -f "${r}" ]] && "${PYTHON}" - "${r}" "${label}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  {sys.argv[2]:9s} lnBF={d['log_bayeswave_signal_glitch']:+7.2f} "
      f"+/-{d['log_bayeswave_signal_glitch_uncertainty']:.2f}  "
      f"recon_snr={d.get('signal_reconstructed_snr_median')}  "
      f"target_snr={d['target_snr']:.2f}")
PY
  done
fi
echo "[e${INDEX}] done"
