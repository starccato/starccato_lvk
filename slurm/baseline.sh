#!/bin/bash
# Reweighted-SNR baseline (matched filter + Allen chi-squared + CBC newSNR) over
# an EXISTING real-noise population. Reads the per-event bundles under OUTDIR and
# writes results/e*_*_baseline.json + summary_baseline.json. No sampling.
#
#   sbatch --export=OUTDIR=slurm/out/rn_L1    slurm/baseline.sh
#   sbatch --export=OUTDIR=slurm/out/rn_H1_L1 slurm/baseline.sh   # 2-det: slower
#
# Idempotent: each class writes its own JSON and is skipped if present, so a
# resubmit after a timeout just continues where it stopped.

#SBATCH --job-name=starccato_base
#SBATCH --account=oz303
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/base_%j.out
#SBATCH --error=slurm/logs/base_%j.err

set -euo pipefail
VENV=/fred/oz303/avajpeyi/codes/starccato_lvk/.venv
OUTDIR=${OUTDIR:?set OUTDIR=slurm/out/rn_L1}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# The venv interpreter links against the module-provided libpython3.11.so, so the
# module MUST be loaded before activating -- omitting it fails with
# "libpython3.11.so.1.0: cannot open shared object file".
module load gcc/12.3.0 python/3.11.3
source ${VENV}/bin/activate

srun python studies/chisq_baseline.py --outdir "${OUTDIR}" --all
srun python studies/chisq_baseline.py --outdir "${OUTDIR}" --aggregate
