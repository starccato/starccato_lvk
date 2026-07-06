#!/bin/bash
# P-P calibration test -- one array task per injection (idempotent: skips existing
# results/inj_{i}.json). Purely simulated (design-PSD noise), no internet needed.
#
#   sbatch slurm/pp_test.sh
#   # then aggregate + plot:
#   python studies/pp_test.py --plot-only --outdir slurm/out/pp
#
#SBATCH --job-name=starccato_pp
#SBATCH --array=0-299
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/pp_%A_%a.out
#SBATCH --error=slurm/logs/pp_%A_%a.err

VENV=/fred/oz303/avajpeyi/codes/starccato_lvk/.venv
INDEX=${SLURM_ARRAY_TASK_ID:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OMP_NUM_THREADS=1
export STARCCATO_PSD_CACHE="${REPO_ROOT}/studies/design_psd_cache"
module load gcc/12.3.0 python/3.11.3
source ${VENV}/bin/activate

srun ${VENV}/bin/python studies/pp_test.py \
  --index ${INDEX} \
  --outdir slurm/out/pp \
  --snr-ref 20 \
  --num-warmup 500 --num-samples 1000 --num-chains 4
