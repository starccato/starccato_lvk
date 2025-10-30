from __future__ import annotations

import glob
import os

from starccato_lvk.acquisition.main import strain_loader
from starccato_lvk.analysis import run_starccato_analysis, run_bcr_posteriors


def test_analysis(outdir, mock_data_dir, noise_trigger_time):
    out = f"{outdir}/starccato_analysis"
    os.makedirs(out, exist_ok=True)
    strain_loader(noise_trigger_time, outdir=out)
    bundle_path = glob.glob(f"{out}/analysis_bundle_*.hdf5")[0]

    nuts_results = run_starccato_analysis(
        detectors=["H1"],
        outdir=f"{out}/nuts_single",
        bundle_paths={"H1": bundle_path},
        model_types=["ccsne"],
        sampler="nuts",
        num_samples=10,
        num_warmup=10,
        num_chains=1,
        latent_sigma=1.0,
        log_amp_sigma=1.0,
        save_artifacts=True,
    )
    assert "ccsne" in nuts_results

    nested_results = run_starccato_analysis(
        detectors=["H1"],
        outdir=f"{out}/nested_single",
        bundle_paths={"H1": bundle_path},
        model_types=["ccsne"],
        sampler="nested",
        num_samples=10,
        num_live_points=10,
        max_samples=50,
        latent_sigma=1.0,
        log_amp_sigma=1.0,
        save_artifacts=True,
    )
    assert "ccsne" in nested_results

    bcr_results = run_bcr_posteriors(
        detectors=["H1"],
        outdir=f"{out}/bcr_single",
        bundle_paths={"H1": bundle_path},
        signal_model="ccsne",
        glitch_model="blip",
        num_samples=5,
        num_warmup=5,
        num_chains=1,
        latent_sigma_signal=1.0,
        log_amp_sigma_signal=1.0,
        latent_sigma_glitch=0.5,
        log_amp_sigma_glitch=0.1,
        save_artifacts=False,
    )
    assert "logZ" in bcr_results["signal"]
    assert "H1" in bcr_results["noise"]
    assert "bcr" in bcr_results
