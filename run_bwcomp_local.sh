#!/bin/bash
# Local (no-SLURM) BayesWave-vs-lnO comparison for ONE injected-signal event.
# Same three stages as slurm/bayeswave_comparison.sh, driven directly on this
# machine with the two local conda/venv environments.
#
#   Usage:  ./run_bwcomp_local.sh [EVENT_INDEX]        # default index 0
#
#   Quick plumbing test (~30 s, numbers are NOT converged):
#       BW_NITER=4000 ./run_bwcomp_local.sh 0
#   Real run (~1 h; the two sky settings run in parallel):
#       ./run_bwcomp_local.sh 0
#
# Stages, each skipped when its output already exists (resubmit = resume):
#   1. starccato_lvk pipeline -> event data + our posterior (--save-artifacts)
#   2. BayesWave twice: fixed sky (setting 1) and free sky (setting 2)
#   3. comparison plots + a one-line lnBF summary for both settings
#
# Override any of the CAPS variables below from the environment.

set -euo pipefail

INDEX=${1:-0}
CLASS=${CLASS:-inj_ccsn}
DETECTORS=${DETECTORS:-"H1 L1"}
PARALLEL=${PARALLEL:-1}
PLOT_IFO=${PLOT_IFO:-H1}
RUN_DATA=${RUN_DATA:-1}; RUN_BW=${RUN_BW:-1}; RUN_PLOTS=${RUN_PLOTS:-1}

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
# outdir* is gitignored, so results never land in a commit.
OUTROOT=${OUTROOT:-${REPO_ROOT}/outdir_bwcomp}
VENV_PY=${VENV_PY:-${REPO_ROOT}/.venv/bin/python}             # project env: JAX pipeline + plotter
BW_ENV=${BW_ENV:-/Users/avi/micromamba/envs/bayeswave_env}    # BayesWave env: gwpy + binaries
BW_PYTHON=${BW_ENV}/bin/python
BAYESWAVE=${BW_ENV}/bin/BayesWave
BAYESWAVE_POST=${BW_ENV}/bin/BayesWavePost

# Full run is 1e6 iters; set BW_NITER=4000 for a fast plumbing test.
BW_NITER=${BW_NITER:-1000000}; BW_BURNIN=${BW_BURNIN:-100000}
BW_NCHAIN=${BW_NCHAIN:-20}; BW_THREADS=${BW_THREADS:-4}; BW_CKPT_HRS=${BW_CKPT_HRS:-1.0}
# BayesWave requires burnin < iterations; keep a small-Niter plumbing test valid.
if (( BW_BURNIN >= BW_NITER )); then BW_BURNIN=$(( BW_NITER / 10 )); fi

case "${CLASS}" in inj_ccsn|inj_glitch) ;; *)
  echo "CLASS=${CLASS} is not an injected class; this compares injections" >&2; exit 2 ;;
esac
for exe in "${VENV_PY}" "${BW_PYTHON}" "${BAYESWAVE}" "${BAYESWAVE_POST}"; do
  [[ -x "${exe}" ]] || { echo "Missing executable: ${exe} (set VENV_PY / BW_ENV)" >&2; exit 2; }
done

DETTAG=$(echo "${DETECTORS}" | tr ' ' '_')
DATA_ROOT=${OUTROOT}/data/rn_${DETTAG}
MANIFEST=${DATA_ROOT}/e${INDEX}/manifest.json
OUR_SAMPLES=${DATA_ROOT}/e${INDEX}/${CLASS}/analysis/signal/samples.npz
BW_FIXED=${OUTROOT}/bw_fixedsky/e${INDEX}/${CLASS}
BW_FREE=${OUTROOT}/bw_freesky/e${INDEX}/${CLASS}
PLOT_DIR=${OUTROOT}/plots

cd "${REPO_ROOT}"
mkdir -p "${PLOT_DIR}"
export OMP_NUM_THREADS=${BW_THREADS}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# --- Stage 1: data + our posterior -----------------------------------------
if [[ "${RUN_DATA}" == "1" && ! -f "${OUR_SAMPLES}" ]]; then
  echo "[e${INDEX}] stage 1: generating data + posterior (--save-artifacts)"
  "${VENV_PY}" studies/real_noise_event.py \
    --index "${INDEX}" --detectors ${DETECTORS} --stage both \
    --class "${CLASS}" --prep-classes noise "${CLASS}" \
    --campaign-id localcomp --outdir "${DATA_ROOT}" --save-artifacts
else
  echo "[e${INDEX}] stage 1: skipped (have ${OUR_SAMPLES} or RUN_DATA=0)"
fi
[[ -f "${MANIFEST}" ]] || { echo "No manifest at ${MANIFEST}; cannot run BayesWave" >&2; exit 2; }

# --- Stage 2: BayesWave, fixed sky (setting 1) and free sky (setting 2) -----
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
  echo "[e${INDEX}] stage 2: BayesWave fixed-sky + free-sky (Niter=${BW_NITER}, parallel=${PARALLEL})"
  if [[ "${PARALLEL}" == "1" ]]; then
    run_bayeswave "${BW_FIXED}"           > "${OUTROOT}/e${INDEX}_fixed.log" 2>&1 & pf=$!
    run_bayeswave "${BW_FREE}" --free-sky > "${OUTROOT}/e${INDEX}_free.log"  2>&1 & pr=$!
    fail=0; wait "${pf}" || fail=1; wait "${pr}" || fail=1
    if [[ "${fail}" == "1" ]]; then
      echo "A BayesWave run failed; tails:" >&2
      tail -5 "${OUTROOT}/e${INDEX}_fixed.log" "${OUTROOT}/e${INDEX}_free.log" >&2; exit 1
    fi
  else
    run_bayeswave "${BW_FIXED}"
    run_bayeswave "${BW_FREE}" --free-sky
  fi
else
  echo "[e${INDEX}] stage 2: skipped (RUN_BW=0)"
fi

# --- Stage 3: plots + lnBF summary -----------------------------------------
if [[ "${RUN_PLOTS}" == "1" ]]; then
  for label in fixedsky freesky; do
    bw=${BW_FIXED}; [[ "${label}" == "freesky" ]] && bw=${BW_FREE}
    post=${bw}/post/signal
    out=${PLOT_DIR}/e${INDEX}_${CLASS}_${label}.pdf
    if [[ -f "${OUR_SAMPLES}" && -d "${post}" ]]; then
      "${VENV_PY}" studies/plot_waveform_reconstruction.py \
        --our-samples "${OUR_SAMPLES}" --bayeswave-post "${post}" \
        --ifo "${PLOT_IFO}" --out "${out}" || echo "plot ${label} failed (see above)"
    else
      echo "[e${INDEX}] plot ${label} skipped: missing ${OUR_SAMPLES} or ${post}"
    fi
  done
  echo "[e${INDEX}] lnBF(signal-glitch) summary:"
  for label in fixedsky freesky; do
    r=${BW_FIXED}/result.json; [[ "${label}" == "freesky" ]] && r=${BW_FREE}/result.json
    [[ -f "${r}" ]] && "${VENV_PY}" - "${r}" "${label}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  {sys.argv[2]:9s} lnBF={d['log_bayeswave_signal_glitch']:+8.2f} "
      f"+/-{d['log_bayeswave_signal_glitch_uncertainty']:.2f}  "
      f"recon_snr={d.get('signal_reconstructed_snr_median')}  "
      f"target_snr={d['target_snr']:.2f}")
PY
  done
fi
echo "[e${INDEX}] done -> ${OUTROOT}"
