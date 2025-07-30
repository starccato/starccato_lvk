import os

import jax

# set float64 precision for JAX
jax.config.update("jax_enable_x64", True)

from starccato_jax.waveforms import StarccatoBlip
from starccato_lvk.likelihood import StarccatoLVKLikelihood
import jax.numpy as jnp
import numpy as np


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


