import csv
import os
from typing import Dict, Any

import arviz as az
import numpy as np
from starccato_jax.waveforms import MODELS, get_model
from .lvk_data_prep import LvkDataPrep
from .post_proc import plot_posterior_predictive, plot_diagnostics, plot_posterior_comparison
from .likelihood import run_inference


def run_starccato_analysis(
        data_path: str,
        psd_path: str | None = None,
        outdir: str = "outdir",
        injection_model_type: str = None,
        injection_params: Dict[str, Any]={},
        num_samples: int = 2000,
        force_rerun: bool = False,
        test_mode: bool = False,
        verbose: bool = False,
        save_artifacts: bool = True,
) -> Dict[str, az.InferenceData]:
    """
    Run comparison analysis: create injection once, then analyze with multiple models.

    Parameters:
    -----------
    data_path : str
        Path to the HDF5 data file or bundled analysis file
    psd_path : str, optional
        Path to the HDF5 PSD file (omit when using bundled data)
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
    save_artifacts : bool, optional
        If False, skip saving intermediate products (inference files and plots) and only write the summary CSV.

    Returns:
    --------
    Dict[str, az.InferenceData]
        Dictionary with model names as keys containing results
    """

    # Helper function to get model instance

    print(f"{'=' * 60}")
    print("STARCCATO ANALYSIS")
    print(f"Analysis models: {[m.upper() for m in MODELS]}")
    print(f"Output directory: {outdir}")
    if not save_artifacts:
        print("Will skip saving inference files and plots (save_artifacts=False).")
    injection_model = None # Default: no injection
    if injection_model_type is not None:
        print(f"Injection model: {injection_model_type.upper()}")
        injection_model = get_model(injection_model_type)
    else:
        print("No injection model specified; proceeding without injection.")

    print(f"{'=' * 60}")

    data = LvkDataPrep.load(
        data_path,
        psd_path,
        waveform_model=injection_model,
        injection_params=injection_params
    )

    # Print injection SNRs
    snrs = data.get_snrs()
    if snrs.get('optimal_snr') is not None:
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

            if save_artifacts:
                os.makedirs(out_model, exist_ok=True)
                arviz_res.to_netcdf(inference_fname)
                print("MCMC inference complete and saved.")
            else:
                print("Skipping inference file save (save_artifacts=False).")

        if save_artifacts:
            print("Generating plots...")
            os.makedirs(out_model, exist_ok=True)
            plot_posterior_predictive(arviz_res, f"{out_model}/posterior_predictive.png")
            plot_diagnostics(arviz_res, outdir=out_model)
            print(f"Analysis complete for {model_type.upper()}")
        else:
            print("Skipping diagnostic plot generation (save_artifacts=False).")
            print(f"Analysis complete for {model_type.upper()} (artifacts skipped)")

        results[model_type] = arviz_res

    if save_artifacts:
        plot_posterior_comparison(
            results['ccsne'], results['blip'],
            fname=os.path.join(outdir, "comparison_posterior.png")
        )
    else:
        print("Skipping posterior comparison plot (save_artifacts=False).")

    print(f"\n{'=' * 60}")
    print("COMPARISON ANALYSIS COMPLETE")
    if injection_model_type is not None:
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

    lnbf_signal_vs_noise = ccsne_lnz - noise_lnz
    lnbf_blip_vs_noise = blip_lnz - noise_lnz
    lnbf_signal_vs_blip = ccsne_lnz - blip_lnz

    # Uniform priors across CCSNE, BLIP, Noise -> alternative = average of BLIP + Noise
    logZ_alt = np.logaddexp(blip_lnz, noise_lnz) - np.log(2.0)
    lnbf_signal_vs_alt = ccsne_lnz - logZ_alt

    print("\nFinal Summary:")
    print(f"CCSNE log_evidence: {ccsne_lnz:.4f}")
    print(f"BLIP log_evidence: {blip_lnz:.4f}")
    print(f"Noise log_evidence: {noise_lnz:.4f}")
    print(f"log_BF(CCSNE vs Noise): {lnbf_signal_vs_noise:.4f}")
    print(f"log_BF(BLIP vs Noise): {lnbf_blip_vs_noise:.4f}")
    print(f"log_BF(CCSNE vs BLIP): {lnbf_signal_vs_blip:.4f}")
    print(f"log_BF(CCSNE vs avg(BLIP, Noise)): {lnbf_signal_vs_alt:.4f}")
    print(f"CCSNE-Matched Filter SNR: {match_snr:.3f}")

    # write summary CSV (main analysis product)
    summary_fname = os.path.join(outdir, "comparison_summary.csv")
    headers = [
        "ccsne_lnz",
        "blip_lnz",
        "noise_lnz",
        "logBF_ccsne_noise",
        "logBF_blip_noise",
        "logBF_ccsne_blip",
        "logBF_ccsne_alt",
        "ccsne_snr_matched",
    ]
    row = [
        float(ccsne_lnz),
        float(blip_lnz),
        float(noise_lnz),
        float(lnbf_signal_vs_noise),
        float(lnbf_blip_vs_noise),
        float(lnbf_signal_vs_blip),
        float(lnbf_signal_vs_alt),
        float(match_snr),
    ]
    with open(summary_fname, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(row)
    print(f"Summary CSV written to {summary_fname}")
    return results
