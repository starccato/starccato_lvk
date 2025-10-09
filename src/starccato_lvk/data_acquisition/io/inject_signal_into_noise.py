# DEPRECATED: injection made via lvk_data_prep
from typing import Union

import jax
import numpy as np

from starccato_jax.waveforms import StarccatoCCSNe

from .strain_loader import load_analysis_chunk_and_psd, _save_analysis_chunk_and_psd
from .only_noise_data import get_noise_trigger_time


_CCSNE_MODEL: StarccatoCCSNe | None = None


def _get_waveform_model() -> StarccatoCCSNe:
    global _CCSNE_MODEL
    if _CCSNE_MODEL is None:
        _CCSNE_MODEL = StarccatoCCSNe()
    return _CCSNE_MODEL


def _normalize_rng(rng: Union[int, np.integer, jax.Array]) -> jax.Array:
    if isinstance(rng, (int, np.integer)):
        return jax.random.PRNGKey(int(rng))
    if isinstance(rng, jax.Array):
        return rng
    raise TypeError("rng must be an int seed or a jax.random.PRNGKey")


def create_injection_signal(trigger_time, rng, distance: float, outdir: str = None):
    """Inject a CCSNe signal around an index or explicit trigger time."""
    # Normalize trigger input: allow passing an index or a GPS time directly.
    if isinstance(trigger_time, (int, np.integer)):
        noise_trigger_time = get_noise_trigger_time(int(trigger_time))
    else:
        noise_trigger_time = float(trigger_time)

    # 1. Load the noise data (and PSD) around the trigger time
    data, psd = load_analysis_chunk_and_psd(noise_trigger_time)

    # 2. Create the CCSNe signal
    model = _get_waveform_model()
    rng_key = _normalize_rng(rng)
    ccsne_model = np.array(model.generate(rng=rng_key, n=1)[0], dtype=np.float64)
    ccsne_model /= distance

    # 3. Inject the CCSNe signal into the noise data (ensure that the signal PEAK is at the trigger time)
    ccsne_start_time = noise_trigger_time - len(ccsne_model) // 2
    ccsne_end_time = ccsne_start_time + len(ccsne_model)
    data.value[ccsne_start_time:ccsne_end_time] += ccsne_model

    # 4. Save the modified noise data and PSD (and true CCSNe signal)
    if outdir:
        _save_analysis_chunk_and_psd(data, psd, noise_trigger_time, outdir, injection=ccsne_model)

    return data, psd, ccsne_model
