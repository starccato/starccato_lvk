import bilby
import jax.numpy as jnp
import numpy as np
import os
from starccato_jax.waveforms import MODELS, get_model
from starccato_lvk.lvk_data_prep import LvkDataPrep
from starccato_lvk.post_proc import plot_posterior_predictive, plot_diagnostics, plot_posterior_comparison
from starccato_lvk.likelihood import run_inference
import arviz as az
from typing import Dict, Any


def run_starccato_analysis(
        data_path: str,
        psd_path: str,
        injection_params: Dict[str, Any],
        outdir: str,
        injection_model_type: str = 'ccsne',
        num_samples: int = 2000,
        force_rerun: bool = False,
        test_mode:bool = False,
        verbose: bool = False,
) -> Dict[str, az.InferenceData]:
    """
    Run comparison analysis: create injection once, then analyze with multiple models.

    Parameters:
    -----------
    data_path : str
        Path to the HDF5 data file
    psd_path : str
        Path to the HDF5 PSD file
    injection_params : Dict[str, Any]
        Dictionary containing injection parameters (amplitude, rng_key, z)
    outdir : str
        Output directory for results
    injection_model_type : str, optional
        Model to use for creating injection (default: 'ccsne')
    analysis_models : list, optional
        List of models to use for analysis (default: ['ccsne', 'blip'])
    label_suffix : str, optional
        Additional suffix for output files
    num_samples : int, optional
        Number of MCMC samples (default: 2000)
    num_warmup : int, optional
        Number of warmup samples (default: 1000)
    force_rerun : bool, optional
        If True, rerun even if results exist (default: False)

    Returns:
    --------
    Dict[str, az.InferenceData]
        Dictionary with model names as keys containing results
    """

    # Helper function to get model instance

    print(f"{'=' * 60}")
    print("STARCCATO ANALYSIS")
    print(f"Injection model: {injection_model_type.upper()}")
    print(f"Analysis models: {[m.upper() for m in MODELS]}")
    print(f"Output directory: {outdir}")
    print(f"{'=' * 60}")

    # Step 1: Create injection ONCE
    print(f"Creating injection using {injection_model_type.upper()} model...")
    injection_model = get_model(injection_model_type)

    data = LvkDataPrep.load(
        data_path, psd_path,
        waveform_model=injection_model,
        injection_params=injection_params
    )

    # Print injection SNRs
    snrs = data.get_snrs()
    print(f"Injection created:")
    print(f"  Optimal SNR: {snrs['optimal_snr']:.3f}")
    print(f"  Matched Filter SNR: {snrs['matched_snr']:.3f}")

    # Step 2: Loop through analysis models and run inference on same data
    os.makedirs(outdir, exist_ok=True)
    results = {}

    for model_type in MODELS:
        print(f"\n{'-' * 50}")
        print(f"ANALYZING WITH {model_type.upper()} MODEL")
        print(f"{'-' * 50}")

        # Setup for this model
        analysis_model = get_model(model_type)
        label = model_type.lower()
        out_model = f"{outdir}/{label}"
        inference_fname = os.path.join(out_model, "inference.nc")
        if os.path.exists(inference_fname) and not force_rerun:
            print(f"Loading existing results from {inference_fname}")
            arviz_res = az.from_netcdf(inference_fname)
            print("Loaded existing results.")
        else:
            # Run MCMC inference
            print(f"Starting MCMC inference ({num_samples} samples)...")

            kwgs = {}
            if test_mode:
                kwgs = dict(nlive=50, max_samples=100)

            arviz_res = run_inference(
                data, analysis_model,
                num_samples=num_samples,
                verbose=verbose,
                **kwgs
            )

            # Save results
            os.makedirs(out_model, exist_ok=True)
            arviz_res.to_netcdf(inference_fname)
            print("MCMC inference complete and saved.")

        # Generate plots
        print("Generating plots...")
        plot_posterior_predictive(arviz_res, f"{out_model}/posterior_predictive.png")
        plot_diagnostics(arviz_res, outdir=out_model)
        print(f"Analysis complete for {model_type.upper()}")

        results[model_type] = arviz_res

    plot_posterior_comparison(
        results['ccsne'], results['blip'],
        fname=os.path.join(outdir, "comparison_posterior.png")
    )

    print(f"\n{'=' * 60}")
    print("COMPARISON ANALYSIS COMPLETE")
    print(f"Injection: {injection_model_type.upper()} model")
    print(f"Analyzed with: {[m.upper() for m in MODELS]}")
    print(f"Results saved in: {outdir}")
    print(f"{'=' * 60}")

    # Access results
    ccsne_results = results['ccsne']
    blip_results = results['blip']
    ccsne_lnz = ccsne_results.attrs.get('log_evidence', 0)
    blip_lnz = blip_results.attrs.get('log_evidence', 0)
    noise_lnz = ccsne_results.attrs.get('log_evidence_noise', 0)
    match_snr = ccsne_results.constant_data.snr_quantiles.values[1,1]
    # z_ccsn / (z_ccsn + z_blip)
    lnbf = ccsne_lnz - (noise_lnz + blip_lnz)

    print("\nFinal Summary:")
    print(f"CCSNE log_evidence: {ccsne_lnz:.4f}")
    print(f"BLIP log_evidence: {blip_lnz:.4f}")
    print(f"Noise log_evidence: {noise_lnz:.4f}")
    print(f"Log Bayes Factor (CCSNE / (Noise + BLIP)): {lnbf:.4f}")
    print(f"CCSNE-Matched Filter SNR: {match_snr:.3f}")

    # write summary into text file
    summary_fname = os.path.join(outdir, "comparison_summary.txt")
    np.savetxt(
        summary_fname,
        np.array([[ccsne_lnz, blip_lnz, noise_lnz, lnbf, match_snr]]),
        header="ccsne_lnz blip_lnz noise_lnz lnbf snr",
        fmt="%.8f"
    )
    return results

