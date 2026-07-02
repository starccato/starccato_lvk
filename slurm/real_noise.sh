#!/bin/bash
# Real-noise analysis study on OzSTAR -- one array task per blip-catalogue index,
# four event classes each (noise, inj_ccsn, inj_glitch, real_glitch).
#
# STRAIN IS READ FROM THE LOCAL LVK MIRROR (config.BASE_DATA_DIR =
# /datasets/LIGO/.../O3b/strain.4k/hdf.v1), so compute nodes need NO internet and
# there is no CAT3 gate -> a single-stage 'both' run per index is enough.
#
# ONE-TIME PRE-CACHE (on an internet-capable 'inode'), to populate the venv cache
# with the VAE weights + held-out training data + blip catalogue:
#   python studies/real_noise_event.py --index 0 --detectors L1 --stage both --outdir slurm/out/rn_L1
#
# Then launch the arrays (no dependency needed -- strain is local):
#   sbatch --export=DETECTORS="L1"     slurm/real_noise.sh          # 1-detector
#   sbatch --export=DETECTORS="H1 L1"  slurm/real_noise.sh          # 2-detector
#
# The runner is IDEMPOTENT: each class writes results/e{i}_{cls}.json and is
# skipped if it already exists. So re-submitting the SAME indices does no new
# work (it only backfills classes that previously FAILED). To grow the sample,
# point the array at a FRESH index range (overrides the #SBATCH --array below):
#   sbatch --array=200-599 --export=DETECTORS="L1"    slurm/real_noise.sh   # new events
#   sbatch --array=0-199   --export=DETECTORS="H1 L1" slurm/real_noise.sh   # backfill failures
# The blip catalogue has ~1178 rows, so indices up to ~1177 are valid.
#
#   # aggregate + plot when done:
#   python studies/real_noise_aggregate.py --outdir slurm/out/rn_L1
#   python studies/real_noise_plots.py     --l1 slurm/out/rn_L1 --h1l1 slurm/out/rn_H1_L1
#
# (If run on a system WITHOUT the local mirror, each task falls back to a GWOSC
#  download, so use STAGE=prep on a data-mover node then STAGE=analysis on compute.)
#
#SBATCH --job-name=starccato_rn
#SBATCH --array=0-199
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/rn_%A_%a.out
#SBATCH --error=slurm/logs/rn_%A_%a.err

VENV=/fred/oz303/avajpeyi/codes/starccato_lvk/.venv
INDEX=${SLURM_ARRAY_TASK_ID:-0}
STAGE=${STAGE:-both}
DETECTORS=${DETECTORS:-L1}
GLITCH_DET=${GLITCH_DET:-L1}
DETTAG=$(echo "${DETECTORS}" | tr ' ' '_')
OUTDIR=slurm/out/rn_${DETTAG}

export OMP_NUM_THREADS=1
module load gcc/12.3.0 python/3.11.3
source ${VENV}/bin/activate

# data-mover nodes lack a GPU/heavy compute; compute nodes may lack internet -> split stages.
srun ${VENV}/bin/python studies/real_noise_event.py \
  --index ${INDEX} \
  --detectors ${DETECTORS} \
  --glitch-det ${GLITCH_DET} \
  --stage ${STAGE} \
  --outdir ${OUTDIR} \
  --flow 300 --fmax 800 \
  --num-warmup 500 --num-samples 1000
