from starccato_jax.waveforms import StarccatoCCSNe
from .strain_loader import load_analysis_chunk_and_psd, _save_analysis_chunk_and_psd
from .only_noise_data import get_noise_trigger_time
import numpy as np


def create_injection_signal(trigger_time: float, rng: int, distance: float, outdir: str = None):
    # 1. Load the noise data (and PSD) around the trigger time
    noise_trigger_time = get_noise_trigger_time(trigger_time)
    data, psd = load_analysis_chunk_and_psd(noise_trigger_time)

    # 2. Create the CCSNe signal
    ccsne_model = np.array(StarccatoCCSNe.generate(rng, n=1)[0], dtype=np.float64)
    ccsne_model /= distance

    # 3. Inject the CCSNe signal into the noise data (ensure that the signal PEAK is at the trigger time)
    ccsne_start_time = noise_trigger_time - len(ccsne_model) // 2
    ccsne_end_time = ccsne_start_time + len(ccsne_model)
    data.value[ccsne_start_time:ccsne_end_time] += ccsne_model

    # 4. Save the modified noise data and PSD (and true CCSNe signal)
    if outdir:
        _save_analysis_chunk_and_psd(data, psd, noise_trigger_time, outdir, injection=ccsne_model)

    return data, psd, ccsne_model
