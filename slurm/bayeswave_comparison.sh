#!/bin/bash
# End-to-end BayesWave-vs-lnO comparison for ONE injected-signal event.
#
# One array task == one catalogue index, and runs three stages in order:
#   1. starccato_lvk pipeline: generate the event data AND our posterior
#      (real_noise_event.py --save-artifacts, so analysis/ is kept not pruned).
#   2. BayesWave twice on that data -- fixed sky (setting 1) and free sky
#      (setting 2) -- into separate output dirs. Run in parallel by default.
#   3. Comparison plots (our VAE posterior vs each BayesWave reconstruction)
#      and a one-line lnBF summary for both settings.
#
# Each stage is idempotent: it is skipped when its output already exists, so a
# resubmit resumes rather than redoing work (BayesWave itself resumes from its
# checkpoint).
#
# Required submission variable:
#   CAMPAIGN_ID   immutable tag, e.g. bwcomp_20260722
# Example (4 injected events, both sky settings each):
#   sbatch --array=0-3 \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260722 slurm/bayeswave_comparison.sh
#
# Common optional variables:
#   CLASS (inj_ccsn), DETECTORS ("H1 L1"), PARALLEL (1), RUN_DATA/RUN_BW/RUN_PLOTS
#   BW_NITER, BW_BURNIN, BW_NCHAIN, BW_THREADS, BW_CKPT_HRS, PLOT_IFO
#   VENV, BAYESWAVE_ENV, RESULTS_ROOT, SNR_REFERENCE_DET

#SBATCH --job-name=starccato_bwcmp
#SBATCH --account=oz303
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/bwcmp_%A_%a.out
#SBATCH --error=slurm/logs/bwcmp_%A_%a.err

set -euo pipefail

CAMPAIGN_ID=${CAMPAIGN_ID:?set an immutable CAMPAIGN_ID}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${INDEX_OFFSET:-0} ))

CLASS=${CLASS:-inj_ccsn}                 # this comparison is about injected signals
DETECTORS=${DETECTORS:-"H1 L1"}          # BayesWave sig-vs-glitch needs >=2 IFOs
PARALLEL=${PARALLEL:-1}                   # run the two sky settings concurrently
RUN_DATA=${RUN_DATA:-1}; RUN_BW=${RUN_BW:-1}; RUN_PLOTS=${RUN_PLOTS:-1}
PLOT_IFO=${PLOT_IFO:-H1}

VENV=${VENV:-/fred/oz303/avajpeyi/codes/starccato_lvk/.venv}
BAYESWAVE_ENV=${BAYESWAVE_ENV:-/fred/oz980/avajpeyi/envs/bayeswave}
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
SNR_REFERENCE_DET=${SNR_REFERENCE_DET:-}  # empty = normalise on network SNR

BW_NITER=${BW_NITER:-1000000}; BW_BURNIN=${BW_BURNIN:-100000}
BW_NCHAIN=${BW_NCHAIN:-20}; BW_THREADS=${BW_THREADS:-4}; BW_CKPT_HRS=${BW_CKPT_HRS:-1.0}

case "${CLASS}" in inj_ccsn|inj_glitch) ;; *)
  echo "CLASS=${CLASS} is not an injected class; this script compares injections" >&2
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
PYTHON=${VENV}/bin/python
BAYESWAVE=${BAYESWAVE_ENV}/bin/BayesWave
BAYESWAVE_POST=${BAYESWAVE_ENV}/bin/BayesWavePost
for exe in "${PYTHON}" "${BAYESWAVE}" "${BAYESWAVE_POST}"; do
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

# --- Stage 1: generate data + our posterior (skip if samples.npz already exists)
if [[ "${RUN_DATA}" == "1" && ! -f "${OUR_SAMPLES}" ]]; then
  echo "[e${INDEX}] stage 1: generating data + posterior (--save-artifacts)"
  DATA_ARGS=(
    --index "${INDEX}" --detectors ${DETECTORS} --stage both
    --class "${CLASS}" --prep-classes noise "${CLASS}"
    --campaign-id "${CAMPAIGN_ID}" --outdir "${DATA_ROOT}" --save-artifacts
  )
  [[ -n "${SNR_REFERENCE_DET}" ]] && DATA_ARGS+=(--snr-reference-det "${SNR_REFERENCE_DET}")
  srun "${PYTHON}" studies/real_noise_event.py "${DATA_ARGS[@]}"
else
  echo "[e${INDEX}] stage 1: skipped (have ${OUR_SAMPLES} or RUN_DATA=0)"
fi
[[ -f "${MANIFEST}" ]] || { echo "No manifest at ${MANIFEST}; cannot run BayesWave" >&2; exit 2; }

# --- Stage 2: BayesWave, fixed sky (setting 1) and free sky (setting 2)
run_bayeswave() {  # $1=output_dir  $2..=extra flags (e.g. --free-sky)
  local out=$1; shift
  "${PYTHON}" -m starccato_lvk.bayeswave "${MANIFEST}" \
    --class "${CLASS}" --output "${out}" \
    --bayeswave-executable "${BAYESWAVE}" --post-executable "${BAYESWAVE_POST}" \
    --iterations "${BW_NITER}" --burnin "${BW_BURNIN}" \
    --chains "${BW_NCHAIN}" --threads "${BW_THREADS}" \
    --checkpoint-interval-hours "${BW_CKPT_HRS}" \
    --seed "$((1234 + INDEX))" --execute "$@"
}
if [[ "${RUN_BW}" == "1" ]]; then
  echo "[e${INDEX}] stage 2: BayesWave fixed-sky -> ${BW_FIXED}, free-sky -> ${BW_FREE}"
  if [[ "${PARALLEL}" == "1" ]]; then
    run_bayeswave "${BW_FIXED}"            > "${BW_FIXED%/*}_fixed.stage.log" 2>&1 &
    pid_fixed=$!
    run_bayeswave "${BW_FREE}" --free-sky  > "${BW_FREE%/*}_free.stage.log"  2>&1 &
    pid_free=$!
    wait "${pid_fixed}"; wait "${pid_free}"
  else
    run_bayeswave "${BW_FIXED}"
    run_bayeswave "${BW_FREE}" --free-sky
  fi
else
  echo "[e${INDEX}] stage 2: skipped (RUN_BW=0)"
fi

# --- Stage 3: comparison plots + lnBF summary
if [[ "${RUN_PLOTS}" == "1" ]]; then
  for label in fixedsky freesky; do
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
  for label in fixedsky freesky; do
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
