# Slurm runners  

1. ONE-TIME pre-cache on an inode (populates the shared venv with VAE weights, held-out training data, and the blip catalogue — the only things that download):
python studies/real_noise_event.py --index 0 --detectors L1 --stage both --outdir slurm/out/rn_L1

2. launch the arrays on compute nodes — reads local strain, no internet, no dependency:
sbatch --export=DETECTORS="L1"    slurm/real_noise.sh      # 1-detector
sbatch --export=DETECTORS="H1 L1" slurm/real_noise.sh      # 2-detector
(scale with #SBATCH --array=0-N; catalog has ~1178 blips)

3. aggregate:
python studies/real_noise_aggregate.py --outdir slurm/out/rn_L1