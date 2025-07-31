import os

import jax

# set float64 precision for JAX
jax.config.update("jax_enable_x64", True)

from starccato_jax.waveforms import StarccatoBlip
from starccato_lvk.likelihood import StarccatoLVKLikelihood
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


def test_lnl(outdir, mock_data_dir, analysis_data):
    paths = analysis_data
    assert os.path.exists(paths["strain_file"])

    model = StarccatoBlip()
    lnl_obj = StarccatoLVKLikelihood.from_hdf5_files(
        starccato_model=model,
        **paths,
    )

    rng = jax.random.PRNGKey(0)

    theta = jnp.array([0 for _ in range(lnl_obj.starccato_model.latent_dim)])
    ln_pri = float(lnl_obj.log_prior(theta))
    assert not np.isnan(ln_pri)
    assert ln_pri > -np.inf
    assert ln_pri < 0

    ln_lik = float(lnl_obj.log_likelihood(theta, rng=rng))
    assert not np.isnan(ln_lik)
    assert ln_lik > -np.inf
    assert ln_lik < np.inf


    plt.plot(lnl_obj.whitened_data)
    plt.title("Whitened Data")
    plt.savefig(os.path.join(outdir, "whitened_data.png"))
    plt.close("all")

    plt.plot(lnl_obj.strain_data)
    plt.title("Original Strain Data")
    plt.savefig(os.path.join(outdir, "original_strain_data.png"))

    plt.figure()
    plt.loglog(lnl_obj.freq_array, lnl_obj.psd_interp, label="Interpolated PSD")
    plt.plot(lnl_obj.psd_freq, lnl_obj.psd_values, alpha=0.4, label="Original PSD")
    plt.plot(lnl_obj.freq_array, np.abs(lnl_obj.data_fft)**2, label="Data FFT", alpha=0.5)
    plt.legend()
    plt.title("Power Spectral Density (PSD)")
    plt.savefig(os.path.join(outdir, "psd.png"))




