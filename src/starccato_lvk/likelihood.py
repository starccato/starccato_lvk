
import jax.numpy as jnp
from astropy.time import Time
from jaxtyping import Array, Float

from .detector import Detector, GroundBased2G
from .waveform import Waveform
import logging


class SingleEventLikelihood:
    detectors: list[Detector]
    waveform: Waveform

    def __init__(self, detectors: list[Detector], waveform: Waveform) -> None:
        self.detectors = detectors
        self.waveform = waveform



class TransientLikelihoodFD(SingleEventLikelihood):
    def __init__(
            self,
            detectors: list[Detector],
            waveform: Waveform,
            f_min: Float = 0,
            f_max: Float = float("inf"),
            trigger_time: Float = 0,
            post_trigger_duration: Float = 2,
            **kwargs,
    ) -> None:
        # NOTE: having 'kwargs' here makes it very difficult to diagnose
        # errors and keep track of what's going on, would be better to list
        # explicitly what the arguments are accepted
        self.detectors = detectors

        # make sure data has a Fourier representation
        for det in detectors:
            if not det.data.has_fd:
                logging.info("Computing FFT with default window")
                det.data.fft()

        # collect the data, psd and frequencies for the requested band
        freqs = []
        datas = []
        psds = []
        for detector in detectors:
            data, freq_0 = detector.data.frequency_slice(f_min, f_max)
            psd, freq_1 = detector.psd.frequency_slice(f_min, f_max)
            freqs.append(freq_0)
            datas.append(data)
            psds.append(psd)
            # make sure the psd and data are consistent
            assert (freq_0 == freq_1).all(), \
                f"The {detector.name} data and PSD must have same frequencies"

        # make sure all detectors are consistent
        assert all([(freqs[0] == freq).all() for freq in freqs]), \
            "The detectors must have the same frequency grid"

        self.frequencies = freqs[0]  # type: ignore
        self.datas = [d.data.frequency_slice(f_min, f_max)[0] for d in detectors]
        self.psds = [d.psd.frequency_slice(f_min, f_max)[0] for d in detectors]

        self.waveform = waveform
        self.trigger_time = trigger_time
        self.gmst = (
            Time(trigger_time, format="gps").sidereal_time("apparent",
                                                           "greenwich").rad
        )

        self.trigger_time = trigger_time
        self.duration = duration = self.detectors[0].data.duration
        self.post_trigger_duration = post_trigger_duration
        self.kwargs = kwargs



        # the fixing_parameters is expected to be a dictionary
        # with key as parameter name and value is the fixed value
        # e.g. {'M_c': 1.1975, 't_c': 0}
        if "fixing_parameters" in self.kwargs:
            fixing_parameters = self.kwargs["fixing_parameters"]
            print(f"Parameters are fixed {fixing_parameters}")
            self.fixing_func = lambda x: {**x, **fixing_parameters}
        else:
            self.fixing_func = lambda x: x

    @property
    def epoch(self):
        """The epoch of the data.
        """
        return self.duration - self.post_trigger_duration

    @property
    def ifos(self):
        """The interferometers for the likelihood.
        """
        return [detector.name for detector in self.detectors]

    def evaluate(self, params: dict[str, Float]) -> Float:
        # TODO: Test whether we need to pass data in or with class changes is fine.
        """Evaluate the likelihood for a given set of parameters.
        """
        frequencies = self.frequencies
        params["gmst"] = self.gmst

        # adjust the params due to fixing parameters
        params = self.fixing_func(params)
        # evaluate the waveform as usual
        waveform_sky = self.waveform(frequencies, params)
        align_time = jnp.exp(
            -1j * 2 * jnp.pi * frequencies * (self.epoch + params["t_c"])
        )
        log_likelihood = original_likelihood(
            params,
            waveform_sky,
            self.detectors,
            frequencies,
            self.datas,
            self.psds,
            align_time,
            **self.kwargs,
        )
        return log_likelihood


likelihood_presets = {
    "TransientLikelihoodFD": TransientLikelihoodFD,
}


def original_likelihood(
        params: dict[str, Float],
        h_sky: dict[str, Float[Array, " n_dim"]],
        detectors: list[Detector],
        freqs: Float[Array, " n_dim"],
        datas: list[Float[Array, " n_dim"]],
        psds: list[Float[Array, " n_dim"]],
        align_time: Float,
        **kwargs,
) -> Float:
    log_likelihood = 0.0
    df = freqs[1] - freqs[0]
    for detector, data, psd in zip(detectors, datas, psds):
        h_dec = detector.fd_response(freqs, h_sky, params) * align_time
        # NOTE: do we want to take the slide outside the likelihood?
        match_filter_SNR = (
                4 * jnp.sum((jnp.conj(h_dec) * data) / psd * df).real
        )
        optimal_SNR = 4 * jnp.sum(jnp.conj(h_dec) * h_dec / psd * df).real
        log_likelihood += match_filter_SNR - optimal_SNR / 2

    return log_likelihood

