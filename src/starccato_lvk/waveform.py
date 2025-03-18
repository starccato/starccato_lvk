from abc import ABC

from jaxtyping import Array, Float
from starccato_jax import StarccatoVAE
from jax.random import PRNGKey
import jax.numpy as jnp

__all__ = ["StarccatoWaveform"]

class Waveform(ABC):

    def __init__(self):
        return NotImplemented

    def __call__(
        self, axis: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        return NotImplemented


class StarccatoWaveform(Waveform):

    vae: StarccatoVAE
    rng: PRNGKey

    def __init__(self, key: PRNGKey):
        self.vae = StarccatoVAE()
        self.rng = key

    def __call__(
        self, axis: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:

        # get z from {z_1, z_2, z_3...
        z = jnp.array([params[f"z_{i}"] for i in range(self.vae.latent_dim)])
        dl = params["luminosity_distance"]
        signal = self.vae.generate(z, self.rng)

        # waveforms generated at 10kpc, so scale to the luminosity distance
        scaling = 1e-21 * (10.0 / dl)

        # TODO: the signals are just + pol, but we duplicate for x pol...
        return dict(
            p=signal * scaling,
            c=signal * scaling
        )


    def __repr__(self):
        return self.vae.__repr__()