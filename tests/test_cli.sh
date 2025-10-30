starccato_lvk_run_event --scenario blip --index 1 --config ../slurm/configs/analysis.yaml --force
starccato_lvk_run_event --scenario noise --index 1 --config ../slurm/configs/analysis.yaml --force
starccato_lvk_run_event --scenario noise_inj --index 1 --config ../slurm/configs/analysis.yaml --distance 10 --force
