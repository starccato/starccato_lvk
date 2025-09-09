from starccato_lvk.main import run_starccato_analysis
import numpy as np
import os


def main():
    """Example usage of the comparison analysis."""
    # Set random seed for reproducibility
    np.random.seed(170801)

    # Define paths and parameters
    HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = f"{HERE}/test_data/analysis_chunk_1256676910.hdf5"
    psd_path = f"{HERE}/test_data/psd_1256676910.hdf5"
    outdir = "outdir_comparison"

    # Injection parameters
    injection_params = {
        'amplitude': 5e-22,
        'rng_key': 0,
        'z': np.random.normal(0, 1, size=32)
    }

    # Run comparison analysis
    run_starccato_analysis(
        data_path=data_path,
        psd_path=psd_path,
        injection_params=injection_params,
        outdir=outdir,
        injection_model_type='ccsne',  # Create injection with CCSNE
        num_samples=2000,
        test_mode=False,
        force_rerun=True,
    )


if __name__ == "__main__":
    main()
