import bilby
import jax.numpy as jnp
import numpy as np
import os
from starccato_jax.waveforms import StarccatoCCSNe
from starccato_lvk.lvk_data_prep import LvkDataPrep
from starccato_lvk.post_proc import plot_posterior_predictive, plot_diagnostics
from starccato_lvk.likelihood import run_inference
import arviz as az

np.random.seed(170801)


def main():
    """Main analysis pipeline with NumPyro inference."""
    HERE = os.path.dirname(os.path.abspath(__file__))
    OUTDIR = "outdir"
    LABEL = "supernova_numpyro"

    bilby.core.utils.setup_logger(outdir=OUTDIR, label=LABEL)

    # Setup waveform model and injection parameters
    starccato_model = StarccatoCCSNe()
    injection_params = {
        'amplitude': 5e-22,
        'rng_key': 0,
        'z': np.random.normal(0, 1, size=32)
    }

    # Load data with injection
    data_fn = f"{HERE}/test_data/analysis_chunk_1256676910.hdf5"
    psd_fn = f"{HERE}/test_data/psd_1256676910.hdf5"

    data = LvkDataPrep.load(
        data_fn, psd_fn,
        waveform_model=starccato_model,
        injection_params=injection_params
    )
    snrs = data.get_snrs()
    print(f"\nAnalysis Complete - {LABEL}")
    print(f"Optimal SNR: {snrs['optimal_snr']:.3f}")
    print(f"Matched Filter SNR: {snrs['matched_snr']:.3f}")

    # # Test likelihood computation
    # likelihood_computation_check(data_prep.rescaled_data)

    fname = f"{OUTDIR}/{LABEL}_inference.nc"

    if os.path.exists(fname):
        print(f"Loading existing results from {fname}")
        arviz_res = az.from_netcdf(fname)
        print("Loaded existing results.")
    else:
        # Run MCMC inference
        print("Starting MCMC inference...")
        arviz_res = run_inference(data, starccato_model, num_samples=2000, num_warmup=1000)
        # save results
        arviz_res.to_netcdf(f"{OUTDIR}/{LABEL}_inference.nc")
        print("MCMC inference complete.")

    # Generate plots with posterior predictive
    fig = plot_posterior_predictive(arviz_res)
    fig.savefig("supernova_analysis_starccato_posterior.png", bbox_inches="tight", dpi=300)

    plot_diagnostics(arviz_res, outdir=OUTDIR)

    # Results summary


if __name__ == "__main__":
    main()