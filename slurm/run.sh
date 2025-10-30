#!/bin/bash
#SBATCH --job-name=starccato_run
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/run_%A_%a.out
#SBATCH --error=slurm/logs/run_%A_%a.err

VENV=/fred/oz303/avajpeyi/venvs/jax_cpu_venv/
EVENT_INDEX=${SLURM_ARRAY_TASK_ID:-0}
CONFIG=lvk/slurm/configs/analysis.yaml

export OMP_NUM_THREADS=1

module --force purge && ml gcc/12.3.0 python/3.11.3 && source ${VENV}/bin/activate || true

srun ${VENV}/bin/starccato_lvk_run_event \
  --scenario blip \
  --index ${EVENT_INDEX} \
  --config ${CONFIG} \
  --stage analysis


srun ${VENV}/bin/starccato_lvk_run_event \
  --scenario noise \
  --index ${EVENT_INDEX} \
  --config ${CONFIG} \
  --stage analysis


srun ${VENV}/bin/starccato_lvk_run_event \
  --scenario noise_inj \
  --index ${EVENT_INDEX} \
  --config ${CONFIG} \
  --stage analysis
