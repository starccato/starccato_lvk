#!/bin/bash
# Prep-only stage for the BayesWave-vs-lnO comparison: fetch/build the strain
# bundles and manifest for one event, nothing else. Split out from
# slurm/bayeswave_comparison.sh because "fetching noise bundle" (a local-mirror
# read via gwpy, see acquisition/io/strain_loader.py + utils.py) OOM'd the
# combined job at --mem=4G, --cpus-per-task=4/8 (dies within ~90s, 14K+ IOPS,
# consistent with the full-tree glob in _get_data_files_and_gps_times scanning
# every O3b HDF5 file under the mirror on every call -- worth caching/fixing
# later, not done here). This job is I/O-heavy, not CPU-heavy: 1 core is
# enough, and 10 min / 10G is ample headroom over the ~90s failure point.
#
# Chain into slurm/bayeswave_comparison.sh once this array completes. Use
# aftercorr (not afterok) so each analysis task starts as soon as its OWN
# index's prep finishes, rather than the whole prep array:
#
#   PREP_JOB=$(sbatch --parsable --array=0-3 \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260723 slurm/bayeswave_prep.sh)
#   sbatch --array=0-3 --dependency=aftercorr:${PREP_JOB} \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260723,DATA_STAGE=analysis \
#     slurm/bayeswave_comparison.sh
#
# For the ~100-event population run (BayesWave only; lnO already exists from
# the earlier campaign at the same indices), skip the NUTS re-analysis
# entirely with RUN_DATA=0 -- job 2 then only needs prep's manifest+bundles:
#
#   PREP_JOB=$(sbatch --parsable --array=0-99%50 \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260723 slurm/bayeswave_prep.sh)
#   sbatch --array=0-99%50 --dependency=aftercorr:${PREP_JOB} \
#     --export=ALL,CAMPAIGN_ID=bwcomp_20260723,RUN_DATA=0,RUN_PLOTS=0 \
#     slurm/bayeswave_comparison.sh
#
# (afterok:${PREP_JOB} also works if you would rather wait for the WHOLE prep
# array before any analysis starts -- simpler semantics, less concurrency.)
#
# Required submission variable:
#   CAMPAIGN_ID   immutable tag, matching the one bayeswave_comparison.sh uses
# Common optional variables (must match the paired bayeswave_comparison.sh
# submission so both scripts agree on paths):
#   CLASS (inj_ccsn), DETECTORS ("H1 L1"), VENV, RESULTS_ROOT, INDEX_OFFSET
#   SNR_REFERENCE_DET

#SBATCH --job-name=starccato_bwprep
#SBATCH --account=oz303
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/bwprep_%A_%a.out
#SBATCH --error=slurm/logs/bwprep_%A_%a.err

set -euo pipefail

CAMPAIGN_ID=${CAMPAIGN_ID:?set an immutable CAMPAIGN_ID}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${INDEX_OFFSET:-0} ))

CLASS=${CLASS:-inj_ccsn}
DETECTORS=${DETECTORS:-"H1 L1"}
VENV=${VENV:-/fred/oz303/avajpeyi/codes/starccato_lvk/.venv}
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
SNR_REFERENCE_DET=${SNR_REFERENCE_DET:-}

case "${CLASS}" in inj_ccsn|inj_glitch|real_glitch) ;; *)
  echo "CLASS=${CLASS} is not a transient class" >&2; exit 2 ;;
esac

DETTAG=$(echo "${DETECTORS}" | tr ' ' '_')
CAMP_ROOT=${RESULTS_ROOT}/${CAMPAIGN_ID}
DATA_ROOT=${CAMP_ROOT}/data/rn_${DETTAG}
MANIFEST=${DATA_ROOT}/e${INDEX}/manifest.json

case "${CAMP_ROOT}" in "${REPO_ROOT}"/*)
  echo "Output must be outside the Git checkout: ${CAMP_ROOT}" >&2; exit 2 ;;
esac
PYTHON=${VENV}/bin/python
[[ -x "${PYTHON}" ]] || { echo "Missing executable: ${PYTHON}" >&2; exit 2; }

cd "${REPO_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing a comparison run from a dirty Git checkout" >&2
  git status --short >&2; exit 2
fi
mkdir -p slurm/logs "${CAMP_ROOT}"
export OMP_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
module load gcc/12.3.0 python/3.11.3
source "${VENV}/bin/activate"

if [[ -f "${MANIFEST}" ]]; then
  echo "[e${INDEX}] prep: skipped, manifest already exists: ${MANIFEST}"
  exit 0
fi

echo "[e${INDEX}] prep: fetching bundles for class=${CLASS} detectors=${DETECTORS}"
# Injected classes need the noise bundle to inject into; real_glitch does not.
PREP_CLASSES=("${CLASS}")
[[ "${CLASS}" == inj_* ]] && PREP_CLASSES=(noise "${CLASS}")
PREP_ARGS=(
  --index "${INDEX}" --detectors ${DETECTORS} --stage prep
  --prep-classes "${PREP_CLASSES[@]}"
  --campaign-id "${CAMPAIGN_ID}" --outdir "${DATA_ROOT}"
)
[[ -n "${SNR_REFERENCE_DET}" ]] && PREP_ARGS+=(--snr-reference-det "${SNR_REFERENCE_DET}")
srun "${PYTHON}" studies/real_noise_event.py "${PREP_ARGS[@]}"
echo "[e${INDEX}] prep done -> ${MANIFEST}"
