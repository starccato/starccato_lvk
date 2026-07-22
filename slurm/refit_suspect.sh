#!/bin/bash
# Re-prep and refit the events whose coherent signal fit looks like it never
# found the signal (see studies/refit_suspect_signal_fits.py).
#
# Campaign bundles are pruned once an event completes, so the strain must be
# regenerated before it can be refitted. Preparation is seeded per index, so the
# rebuild reproduces the original event exactly -- and the refit REFUSES to run
# unless the rebuilt manifest matches the original (gps, sky, snr,
# snr_by_detector, target_snr, normalisation, waveform index).
#
# Build the task list first (one line per event: "index<TAB>class"):
#
#   python studies/refit_suspect_signal_fits.py scan <CAMPAIGN> \
#       --write-tasks slurm/refit_tasks.txt
#   # add the falsification control (real glitches must NOT recover):
#   python studies/refit_suspect_signal_fits.py scan <CAMPAIGN> --all-classes \
#       --write-tasks slurm/refit_tasks_with_control.txt
#
# Then, with N = number of lines in the task file:
#
#   sbatch --array=0-$((N-1))%50 \
#     --export=ALL,CAMPAIGN=<CAMPAIGN>,TASKFILE=slurm/refit_tasks.txt,STAGE=prep \
#     slurm/refit_suspect.sh
#   sbatch --array=0-$((N-1))%50 \
#     --export=ALL,CAMPAIGN=<CAMPAIGN>,TASKFILE=slurm/refit_tasks.txt,STAGE=refit \
#     slurm/refit_suspect.sh
#
# Results land beside the originals as results/e{i}_{cls}.refit.json; the
# original rows are never overwritten. Summarise with:
#
#   python studies/refit_suspect_signal_fits.py report <CAMPAIGN>

#SBATCH --job-name=starccato_refit
#SBATCH --account=oz303
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=1
# Production analysis (--mem=4G --time=00:30:00, measured 1.7 GB / 4m25s for a
# 1-detector NUTS+morphZ class) already exercises nested sampling as an
# in-process fallback on a meaningful fraction of events and completes cleanly
# within that budget, so full --lnz-method nested is not expected to need
# materially more. Small headroom over that budget for the 2-detector,
# both-models-nested refit; tighten from seff once the first tasks land.
#SBATCH --mem=6G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/refit_%A_%a.out
#SBATCH --error=slurm/logs/refit_%A_%a.err

set -euo pipefail

CAMPAIGN=${CAMPAIGN:?set CAMPAIGN to the campaign directory holding e*/ and results/}
TASKFILE=${TASKFILE:?set TASKFILE to the task list written by the scan subcommand}
STAGE=${STAGE:-refit}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
VENV=${VENV:-/fred/oz303/avajpeyi/codes/starccato_lvk/.venv}
# Rebuilt bundles live OUTSIDE the campaign: the runner refuses to re-prep over
# an existing manifest that already has analysis results, and the campaign's own
# rows must stay untouched so the refit can be compared against them.
BUNDLE_ROOT=${BUNDLE_ROOT:-${CAMPAIGN}_refit_bundles}
LNZ_METHOD=${LNZ_METHOD:-nested}
FLOW=${FLOW:-300}
FMAX=${FMAX:-800}
DETECTORS=${DETECTORS:-"H1 L1"}
BLIP_IFO=${BLIP_IFO:-H1}
GLITCH_DET=${GLITCH_DET:-${BLIP_IFO}}
SNR_REFERENCE_DET=${SNR_REFERENCE_DET:-H1}
NESTED_LIVE=${NESTED_LIVE:-1000}
NESTED_MAX=${NESTED_MAX:-40000}
MAP_NUM_STARTS=${MAP_NUM_STARTS:-512}

if [[ ! -f "${TASKFILE}" ]]; then
  echo "Task file not found: ${TASKFILE}" >&2
  exit 2
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing environment interpreter: ${VENV}/bin/python" >&2
  exit 2
fi

# One array task is one line of the task file.
LINE_NO=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
TASK_LINE=$(sed -n "${LINE_NO}p" "${TASKFILE}")
if [[ -z "${TASK_LINE}" ]]; then
  echo "No task on line ${LINE_NO} of ${TASKFILE}; array range exceeds the task list" >&2
  exit 2
fi
INDEX=$(echo "${TASK_LINE}" | awk '{print $1}')
CLASS=$(echo "${TASK_LINE}" | awk '{print $2}')
if [[ -z "${INDEX}" || -z "${CLASS}" ]]; then
  echo "Malformed task line ${LINE_NO}: '${TASK_LINE}' (expected 'index<TAB>class')" >&2
  exit 2
fi

cd "${REPO_ROOT}"
mkdir -p slurm/logs "${BUNDLE_ROOT}"
export OMP_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}/studies:${PYTHONPATH:-}"
module load gcc/12.3.0 python/3.11.3
source "${VENV}/bin/activate"

echo "[refit] task ${SLURM_ARRAY_TASK_ID:-0}: index=${INDEX} class=${CLASS} stage=${STAGE}"

if [[ "${STAGE}" == "prep" ]]; then
  # --keep-bundles is belt-and-braces: prep alone never prunes (pruning runs
  # after analysis), but it documents that these bundles must survive.
  srun "${VENV}/bin/python" studies/real_noise_event.py \
    --index "${INDEX}" \
    --detectors ${DETECTORS} \
    --glitch-det "${GLITCH_DET}" \
    --blip-ifo "${BLIP_IFO}" \
    --snr-reference-det "${SNR_REFERENCE_DET}" \
    --prep-classes "${CLASS}" \
    --stage prep \
    --campaign-id "$(basename "${CAMPAIGN}")_refit" \
    --outdir "${BUNDLE_ROOT}" \
    --flow "${FLOW}" \
    --fmax "${FMAX}" \
    --keep-bundles
elif [[ "${STAGE}" == "refit" ]]; then
  srun "${VENV}/bin/python" studies/refit_suspect_signal_fits.py refit \
    "${CAMPAIGN}" \
    --index "${INDEX}" \
    --cls "${CLASS}" \
    --bundles-from "${BUNDLE_ROOT}" \
    --lnz-method "${LNZ_METHOD}" \
    --map-num-starts "${MAP_NUM_STARTS}" \
    --nested-num-live-points "${NESTED_LIVE}" \
    --nested-max-samples "${NESTED_MAX}"
else
  echo "STAGE must be 'prep' or 'refit', got '${STAGE}'" >&2
  exit 2
fi
