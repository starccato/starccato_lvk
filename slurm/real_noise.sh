#!/bin/bash
# Production NUTS+MorphLnZ runner. One array task is one catalogue index and,
# for analysis, exactly one event class. Preparation and results live outside
# the Git checkout under a required immutable CAMPAIGN_ID.
#
# Required submission variables:
#   CAMPAIGN_ID   e.g. nuts_morphlnz_v040_20260718
#   DETECTORS     "H1", "L1", or "H1 L1"
#
# Common optional variables:
#   STAGE, CLASS, BLIP_IFO, GLITCH_DET, SNR_REFERENCE_DET, RESULTS_ROOT, VENV
#   FLOW, FMAX, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS
#   TARGET_ACCEPT_PROB, MAX_TREE_DEPTH, MAP_NUM_STARTS, MAP_MAXITER
#
# Example pilot:
#   sbatch --array=0-9%10 \
#     --export=ALL,CAMPAIGN_ID=nuts_morphlnz_v040_20260718,STAGE=prep,DETECTORS=L1 \
#     slurm/real_noise.sh
#   sbatch --array=0-9%10 \
#     --export=ALL,CAMPAIGN_ID=nuts_morphlnz_v040_20260718,STAGE=analysis,CLASS=noise,DETECTORS=L1 \
#     slurm/real_noise.sh

#SBATCH --job-name=starccato_nml
#SBATCH --account=oz303
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=1
# Measured actual (job 14474010, 1-detector L1, single class 'noise', analysis
# stage): 1.7 GB peak / 4m25s, against a 10G/12h request -- ~6x and ~160x
# headroom. These values are conservative for the untested 2-detector case
# (H1+L1 analysis runs one coherent signal model + a glitch model PER
# detector, so cost is somewhat but not double the 1-detector run) rather than
# tight to that single 1-detector data point. Re-check with seff/job-report
# after the first H1+L1 pilot batch and tighten further before the full array.
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/nml_%A_%a.out
#SBATCH --error=slurm/logs/nml_%A_%a.err

set -euo pipefail

CAMPAIGN_ID=${CAMPAIGN_ID:?set an immutable CAMPAIGN_ID}
DETECTORS=${DETECTORS:?set DETECTORS to H1, L1, or "H1 L1"}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
VENV=${VENV:-/fred/oz303/avajpeyi/codes/starccato_lvk/.venv}
RESULTS_ROOT=${RESULTS_ROOT:-/fred/oz303/avajpeyi/results/starccato_lvk}
# INDEX_OFFSET keeps array values < MaxArraySize for catalogues > ~1000 events.
INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${INDEX_OFFSET:-0} ))
STAGE=${STAGE:-analysis}
CLASS=${CLASS:-}
BLIP_IFO=${BLIP_IFO:-L1}
GLITCH_DET=${GLITCH_DET:-${BLIP_IFO}}
# Normalise the injected amplitude on ONE detector so the network SNR is an
# output, not an input (fixed-source comparison against the one-detector
# campaign). Set it to the detector that campaign analysed. Empty = normalise on
# the network SNR, which redistributes a fixed budget instead of adding signal.
SNR_REFERENCE_DET=${SNR_REFERENCE_DET:-}
PREP_CLASSES=${PREP_CLASSES:-"noise inj_ccsn real_glitch"}
FLOW=${FLOW:-300}
FMAX=${FMAX:-800}
NUM_WARMUP=${NUM_WARMUP:-500}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
NUM_CHAINS=${NUM_CHAINS:-2}
TARGET_ACCEPT_PROB=${TARGET_ACCEPT_PROB:-0.8}
MAX_TREE_DEPTH=${MAX_TREE_DEPTH:-10}
MAP_NUM_STARTS=${MAP_NUM_STARTS:-128}
MAP_MAXITER=${MAP_MAXITER:-400}

DETTAG=$(echo "${DETECTORS}" | tr ' ' '_')
OUTDIR=${RESULTS_ROOT}/${CAMPAIGN_ID}/rn_${DETTAG}
if [[ "${BLIP_IFO}" != "L1" ]]; then
  OUTDIR=${OUTDIR}_blip${BLIP_IFO}
fi
if [[ "${FLOW}" != "300" || "${FMAX}" != "800" ]]; then
  OUTDIR=${OUTDIR}_band${FLOW}_${FMAX}
fi

case " ${DETECTORS} " in
  *" ${GLITCH_DET} "*) ;;
  *)
    echo "GLITCH_DET=${GLITCH_DET} is absent from DETECTORS=${DETECTORS}" >&2
    exit 2
    ;;
esac
case "${OUTDIR}" in
  "${REPO_ROOT}"/*)
    echo "Production output must be outside the Git checkout: ${OUTDIR}" >&2
    exit 2
    ;;
esac
if [[ "${STAGE}" == "analysis" && -z "${CLASS}" ]]; then
  echo "Analysis submissions must set CLASS to one event class" >&2
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
module load gcc/12.3.0 python/3.11.3
source "${VENV}/bin/activate"
"${VENV}/bin/python" -c \
  'from importlib.metadata import version; from packaging.version import Version; assert Version(version("starccato-jax")) >= Version("0.4.0"), version("starccato-jax")'

RUNNER_ARGS=(
  --index "${INDEX}"
  --detectors ${DETECTORS}
  --glitch-det "${GLITCH_DET}"
  --blip-ifo "${BLIP_IFO}"
  --stage "${STAGE}"
  --campaign-id "${CAMPAIGN_ID}"
  --outdir "${OUTDIR}"
  --flow "${FLOW}"
  --fmax "${FMAX}"
  --num-warmup "${NUM_WARMUP}"
  --num-samples "${NUM_SAMPLES}"
  --num-chains "${NUM_CHAINS}"
  --target-accept-prob "${TARGET_ACCEPT_PROB}"
  --max-tree-depth "${MAX_TREE_DEPTH}"
  --map-num-starts "${MAP_NUM_STARTS}"
  --map-maxiter "${MAP_MAXITER}"
)
if [[ "${STAGE}" == "prep" || "${STAGE}" == "both" ]]; then
  read -r -a PREP_CLASS_ARRAY <<< "${PREP_CLASSES}"
  RUNNER_ARGS+=(--prep-classes "${PREP_CLASS_ARRAY[@]}")
fi
if [[ -n "${CLASS}" ]]; then
  RUNNER_ARGS+=(--class "${CLASS}")
fi
if [[ -n "${SNR_REFERENCE_DET}" ]]; then
  RUNNER_ARGS+=(--snr-reference-det "${SNR_REFERENCE_DET}")
fi

srun "${VENV}/bin/python" studies/real_noise_event.py "${RUNNER_ARGS[@]}"
