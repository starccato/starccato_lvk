from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from ..jimgw.core.single_event.waveform import Waveform

from starccato_jax.starccato_model import StarccatoModel


def _to_ndarray64(x) -> np.ndarray:
    """Best-effort conversion to NumPy float64 array without unnecessary copies."""
    if isinstance(x, np.ndarray) and x.dtype == np.float64:
        return x
    try:
        x = jax.device_get(x)
    except Exception:  # pragma: no cover - safe guard for non-device arrays
        pass
    return np.asarray(x, dtype=np.float64)


@dataclass
class StarccatoJimWaveform(Waveform):
    """JIM-compatible frequency-domain waveform wrapper for Starccato VAEs."""

    model: StarccatoModel
    sample_rate: float
    strain_scale: float = 1e-21
    reference_distance: float = 10.0
    window: Optional[np.ndarray] = None
    latent_names: List[str] = field(init=False)
    latent_dim: int = field(init=False)
    num_samples: int = field(init=False)
    original_num_samples: int = field(init=False)
    dt: float = field(init=False)
    _window_jnp: Optional[jnp.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.latent_dim = int(self.model.latent_dim)
        self.latent_names = [f"z_{i}" for i in range(self.latent_dim)]
        self.dt = 1.0 / float(self.sample_rate)
        zero_latent = jnp.zeros((1, self.latent_dim), dtype=jnp.float32)
        test_waveform = self.model.generate(z=zero_latent, rng=jax.random.PRNGKey(0))[0]
        self.original_num_samples = int(test_waveform.shape[0])
        self.num_samples = self.original_num_samples
        if self.window is not None:
            window_np = _to_ndarray64(self.window)
            self.num_samples = int(window_np.shape[0])
            self._window_jnp = jnp.asarray(window_np, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _latent_vector(self, params: Dict[str, float]) -> jnp.ndarray:
        """Return latents as a (1, latent_dim) JAX array in model order."""
        values = [params[name] for name in self.latent_names]
        return jnp.asarray(values, dtype=jnp.float32)[None, :]

    def _time_domain_waveform(self, params: Dict[str, float]) -> jnp.ndarray:
        """Return time-domain waveform scaled by extrinsic amplitude terms."""
        latents = self._latent_vector(params)
        waveform = self.model.generate(z=latents)[0].astype(jnp.float64)

        if waveform.shape[0] > self.num_samples:
            waveform = waveform[: self.num_samples]
        elif waveform.shape[0] < self.num_samples:
            pad_total = self.num_samples - waveform.shape[0]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = jnp.pad(waveform, (pad_left, pad_right))

        distance = params.get("luminosity_distance", self.reference_distance)
        log_amp = params.get("log_amp", 0.0)
        intrinsic_amp = jnp.exp(log_amp)
        amp_scale = intrinsic_amp * self.strain_scale * (self.reference_distance / distance)
        waveform = waveform * amp_scale

        if self._window_jnp is not None:
            waveform = waveform * self._window_jnp

        return waveform

    # ------------------------------------------------------------------
    # Waveform API
    # ------------------------------------------------------------------
    def __call__(
        self,
        frequency: jnp.ndarray,
        params: Dict[str, float],
    ) -> Dict[str, jnp.ndarray]:
        """
        Return frequency-domain waveform at supplied frequencies.

        Parameters
        ----------
        frequency
            JAX array of positive frequencies (rfft grid).
        params
            Dictionary containing latent coordinates (`z_i`), `log_amp`, and optional
            extrinsic parameters such as `luminosity_distance`.
        """
        expected = self.num_samples // 2 + 1
        if frequency.shape[0] != expected:
            raise ValueError(
                f"Frequency grid mismatch: got {frequency.shape[0]}, expected {expected}."
            )

        waveform_td = self._time_domain_waveform(params)
        waveform_fd = jnp.fft.rfft(waveform_td) * self.dt
        return {"p": waveform_fd, "c": waveform_fd}

    # ------------------------------------------------------------------
    # Convenience helpers for diagnostics / plotting
    # ------------------------------------------------------------------
    def time_domain_waveform_numpy(self, params: Dict[str, float]) -> np.ndarray:
        """Return waveform in NumPy float64 form for downstream utilities."""
        return _to_ndarray64(self._time_domain_waveform(params))


@dataclass
class StarccatoGlitchWaveform(Waveform):
    """Frequency-domain glitch waveform evaluated independently per detector."""

    model: StarccatoModel
    sample_rate: float
    strain_scale: float = 1.0
    window: Optional[np.ndarray] = None
    latent_names: List[str] = field(init=False)
    latent_dim: int = field(init=False)
    num_samples: int = field(init=False)
    dt: float = field(init=False)
    _window_jnp: Optional[jnp.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.latent_dim = int(self.model.latent_dim)
        self.latent_names = [f"z_{i}" for i in range(self.latent_dim)]
        self.dt = 1.0 / float(self.sample_rate)
        zero_latent = jnp.zeros((1, self.latent_dim), dtype=jnp.float32)
        test_waveform = self.model.generate(z=zero_latent, rng=jax.random.PRNGKey(0))[0]
        self.num_samples = int(test_waveform.shape[0])
        if self.window is not None:
            window_np = _to_ndarray64(self.window)
            self.num_samples = int(window_np.shape[0])
            self._window_jnp = jnp.asarray(window_np, dtype=jnp.float64)

    def _latent_vector(self, params: Dict[str, float]) -> jnp.ndarray:
        values = [params[name] for name in self.latent_names]
        return jnp.asarray(values, dtype=jnp.float32)[None, :]

    def _time_domain_waveform(self, params: Dict[str, float]) -> jnp.ndarray:
        latents = self._latent_vector(params)
        waveform = self.model.generate(z=latents)[0].astype(jnp.float64)

        if waveform.shape[0] > self.num_samples:
            waveform = waveform[: self.num_samples]
        elif waveform.shape[0] < self.num_samples:
            pad_total = self.num_samples - waveform.shape[0]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            waveform = jnp.pad(waveform, (pad_left, pad_right))

        log_amp = params.get("log_amp", 0.0)
        waveform = waveform * jnp.exp(log_amp) * self.strain_scale

        if self._window_jnp is not None:
            waveform = waveform * self._window_jnp

        return waveform

    def __call__(
        self,
        frequency: jnp.ndarray,
        params: Dict[str, float],
    ) -> Dict[str, jnp.ndarray]:
        expected = self.num_samples // 2 + 1
        if frequency.shape[0] != expected:
            raise ValueError(
                f"Frequency grid mismatch: got {frequency.shape[0]}, expected {expected}."
            )

        waveform_td = self._time_domain_waveform(params)
        waveform_fd = jnp.fft.rfft(waveform_td) * self.dt
        return {"p": waveform_fd, "c": waveform_fd}

    def time_domain_waveform_numpy(self, params: Dict[str, float]) -> np.ndarray:
        return _to_ndarray64(self._time_domain_waveform(params))
