from starccato_lvk.analysis import run_starccato_analysis
import os
import jax
from starccato_lvk.acquisition.main import strain_loader
import glob


def test_analysis(outdir, mock_data_dir, noise_trigger_time):
    out = f"{outdir}/starccato_analysis"
    os.makedirs(out, exist_ok=True)
    strain_loader(noise_trigger_time, outdir=out)
    data_path = glob.glob(f"{out}/analysis_chunk_*.hdf5")[0]
    psd_path = glob.glob(f"{out}/psd_*.hdf5")[0]


    out_noise = f"{out}/noise"
    run_starccato_analysis(
        data_path=data_path,
        psd_path=psd_path,
        outdir = out_noise,
        test_mode = True,
    )

    out_noise = f"{out}/injection"
    run_starccato_analysis(
        data_path=data_path,
        psd_path=psd_path,
        outdir = out_noise,
        test_mode = True,
        injection_model_type='ccsne',  # Create injection with CCSNE
    )

    out_blip = f"{out}/blip"
    run_starccato_analysis(
        data_path=data_path,
        psd_path=psd_path,
        outdir = out_blip,
        test_mode = True,
        injection_model_type='blip',  # Create injection with BLIP
    )
