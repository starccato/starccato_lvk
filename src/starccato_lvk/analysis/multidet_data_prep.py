from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import types

import jax.numpy as jnp
import numpy as np
from astropy.time import Time
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from jimgw.core.single_event import detector as jim_detector
from jimgw.core.single_event import data as jim_data

from .constants import FLOW, FMAX
from ..acquisition.io.strain_loader import (
    load_analysis_bundle,
    load_analysis_chunk_and_psd,
)


class InterferometerData(jim_data.Data):
    """Concrete Data container for interferometer time-frequency pairs."""

    def __init__(
        self,
        *,
        name: str,
        time_domain: np.ndarray,
        window: np.ndarray,
        delta_t: float,
        epoch: float,
    ) -> None:
        self.name = name
        self.delta_t = float(delta_t)
        self.epoch = float(epoch)
        self.window = jnp.asarray(window, dtype=jnp.float64)
        self.td = jnp.asarray(time_domain, dtype=jnp.float64)
        self.fd = jnp.fft.rfft(self.td * self.window) * self.delta_t


@dataclass
class DetectorTimeseries:
    """Container holding per-detector data and associated PSD objects."""

    name: str
    time: np.ndarray
    dt: float
    frequency: np.ndarray
    df: float
    data: InterferometerData
    psd: jim_data.PowerSpectrum
    psd_likelihood: np.ndarray
    data_fd_likelihood: np.ndarray
    band_mask: np.ndarray

    @property
    def strain(self) -> np.ndarray:
        return np.asarray(self.data.td)

    @property
    def windowed_strain(self) -> np.ndarray:
        return np.asarray(self.data.td * self.data.window)


@dataclass
class MultiDetPreparedData:
    """Prepared data ready for JIM likelihood construction."""

    detectors: List[jim_detector.GroundBased2G]
    detector_data: Dict[str, DetectorTimeseries]
    trigger_time: float
    duration: float
    post_trigger_duration: float
    df: float
    gmst: float
    window: np.ndarray
    roll_off: float


DETECTOR_PRESETS = {name.upper(): det for name, det in jim_detector.get_detector_preset().items()}


def _clone_detector(detector_name: str) -> jim_detector.GroundBased2G:
    try:
        template = DETECTOR_PRESETS[detector_name.upper()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown detector '{detector_name}' in jimgw.core.single_event.detector.") from exc
    return copy.deepcopy(template)


def _apply_no_response(detector_obj: jim_detector.GroundBased2G) -> jim_detector.GroundBased2G:
    det_copy = copy.deepcopy(detector_obj)

    def _fd_response_noresp(self, frequency, h_sky, params, **kwargs):
        if "p" in h_sky:
            return h_sky["p"]
        if "c" in h_sky:
            return h_sky["c"]
        return jnp.zeros_like(frequency, dtype=jnp.complex64)

    det_copy.fd_response = types.MethodType(_fd_response_noresp, det_copy)
    return det_copy


DETECTOR_PRESETS = {name.upper(): det for name, det in jim_detector.get_detector_preset().items()}


def _clone_detector(detector_name: str) -> jim_detector.GroundBased2G:
    """Return a deep copy of the named JIM detector template."""
    try:
        template = DETECTOR_PRESETS[detector_name.upper()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown detector '{detector_name}' in jimgw.core.single_event.detector.") from exc
    return copy.deepcopy(template)


def _analysis_from_trigger(trigger_time: float, detector: str) -> Tuple[TimeSeries, FrequencySeries, TimeSeries]:
    """Load analysis chunk, PSD, and surrounding strain for a detector."""
    return load_analysis_chunk_and_psd(trigger_time=trigger_time, detector=detector)


def _analysis_from_bundle(bundle_path: Path) -> Tuple[TimeSeries, FrequencySeries, Optional[TimeSeries], float]:
    """Load analysis products from a Starccato analysis bundle file."""
    strain, psd, metadata = load_analysis_bundle(str(bundle_path))
    trigger_time = metadata.get("trigger_time")
    full_strain = metadata.get("full_strain")
    if trigger_time is None:
        raise ValueError(f"Bundle '{bundle_path}' is missing 'trigger_time' metadata.")
    return strain, psd, full_strain, float(trigger_time)


def _frequency_domain_representation(
    strain: TimeSeries,
    psd: FrequencySeries,
    trigger_time: float,
    flow: float,
    fmax: float,
    roll_off: float,
    detector_name: str,
) -> DetectorTimeseries:
    """Convert time series and PSD into aligned frequency-domain arrays."""
    strain_values = np.asarray(strain.value, dtype=np.float64)
    times = np.asarray(strain.times.value, dtype=np.float64)
    dt = float(strain.dt.value)
    sample_rate = float(strain.sample_rate.value)
    duration = strain_values.shape[0] * dt
    rel_time = times - trigger_time

    freq = np.fft.rfftfreq(strain_values.shape[0], d=dt)
    if freq.size < 2:
        raise ValueError("Frequency array must contain at least two samples.")

    df = float(freq[1] - freq[0])
    zero_mean = strain_values - np.mean(strain_values)
    alpha = min(1.0, max(0.0, 2.0 * roll_off / duration)) if duration > 0 else 0.0
    window = jim_data.tukey(strain_values.shape[0], alpha)
    windowed = zero_mean * window
    data_fd_full = np.fft.rfft(windowed) * dt

    psd_interp = np.interp(freq, psd.frequencies.value, psd.value, left=psd.value[0], right=psd.value[-1])
    psd_interp = np.clip(psd_interp, np.finfo(np.float64).tiny, None)
    eps = np.finfo(np.float64).tiny
    psd_interp = np.where(psd_interp > 0, psd_interp, eps)

    band_mask = (freq >= flow) & (freq <= fmax)
    psd_likelihood = np.where(band_mask, psd_interp, np.inf)
    data_fd_likelihood = np.where(band_mask, data_fd_full, 0.0)

    data_obj = InterferometerData(
        name=detector_name,
        time_domain=zero_mean,
        window=window,
        delta_t=dt,
        epoch=rel_time[0],
    )
    power_spectrum = jim_data.PowerSpectrum(
        values=jnp.asarray(psd_interp, dtype=jnp.float64),
        frequencies=jnp.asarray(freq, dtype=jnp.float64),
        name=detector_name,
    )

    return DetectorTimeseries(
        name=detector_name,
        time=rel_time,
        dt=dt,
        frequency=freq,
        df=df,
        data=data_obj,
        psd=power_spectrum,
        psd_likelihood=psd_likelihood,
        data_fd_likelihood=data_fd_likelihood,
        band_mask=band_mask,
    )


def prepare_multi_detector_data(
    detectors: Sequence[str],
    *,
    trigger_time: Optional[float] = None,
    bundle_paths: Optional[Mapping[str, Path]] = None,
    flow: float = FLOW,
    fmax: float = FMAX,
    roll_off: float = 0.01,
) -> MultiDetPreparedData:
    """
    Prepare multi-detector data products for JIM analysis.

    Parameters
    ----------
    detectors
        Iterable of detector identifiers understood by ``jimgw.detector`` (e.g. ``["H1", "L1"]``).
    trigger_time
        GPS time of the analysis window centre. Required when ``bundle_paths`` is not provided.
    bundle_paths
        Optional mapping ``{detector: Path}`` pointing to Starccato analysis bundles. When
        supplied, the trigger time is inferred from the bundle metadata if not provided.
    flow, fmax
        Frequency band limits used to mask the likelihood.
    """
    if not detectors:
        raise ValueError("At least one detector must be specified.")

    detector_data: Dict[str, DetectorTimeseries] = {}
    trigger_time_inferred: Optional[float] = trigger_time
    reference_window: Optional[np.ndarray] = None

    # Load data for each detector
    for det in detectors:
        det_upper = det.upper()
        if bundle_paths and det_upper in bundle_paths:
            strain, psd, _, bundle_trigger = _analysis_from_bundle(bundle_paths[det_upper])
            if trigger_time_inferred is None:
                trigger_time_inferred = bundle_trigger
        elif trigger_time is not None:
            strain, psd, _ = _analysis_from_trigger(trigger_time, det_upper)
        else:
            raise ValueError(
                f"Missing data source for detector '{det_upper}'. "
                "Provide a trigger_time or bundle path."
            )

        if trigger_time_inferred is None:
            raise ValueError("Trigger time could not be determined from inputs.")

        data = _frequency_domain_representation(
            strain=strain,
            psd=psd,
            trigger_time=trigger_time_inferred,
            flow=flow,
            fmax=fmax,
            roll_off=roll_off,
            detector_name=det_upper,
        )
        if reference_window is None:
            reference_window = np.asarray(data.data.window)
        else:
            if data.data.window.shape[0] != reference_window.shape[0] or not np.allclose(
                np.asarray(data.data.window), reference_window, atol=1e-6
            ):
                raise ValueError("Window mismatch between detectors; ensure consistent data preparation.")
        detector_data[det_upper] = data

    # Consistency checks across detectors
    sample_rates = {d.name: 1.0 / d.dt for d in detector_data.values()}
    if len(set(round(v, 9) for v in sample_rates.values())) != 1:
        raise ValueError(f"Sample rates differ across detectors: {sample_rates}")

    freq_shapes = {d.name: d.frequency.shape[0] for d in detector_data.values()}
    if len(set(freq_shapes.values())) != 1:
        raise ValueError(f"Frequency grid lengths differ across detectors: {freq_shapes}")

    dfs = {d.name: d.df for d in detector_data.values()}
    if len(set(round(v, 12) for v in dfs.values())) != 1:
        raise ValueError(f"Frequency resolution differs across detectors: {dfs}")

    duration = detector_data[next(iter(detector_data))].data.duration
    post_trigger_duration = duration / 2.0
    df = next(iter(detector_data.values())).df

    gmst = Time(trigger_time_inferred, format="gps").sidereal_time("apparent", "greenwich").rad

    prepared_detectors: List[jim_detector.GroundBased2G] = []
    for det_name, data in detector_data.items():
        det_obj = _clone_detector(det_name)
        det_obj.data = data.data
        det_obj.psd = data.psd
        det_obj.set_frequency_bounds(flow, fmax)
        prepared_detectors.append(det_obj)

    return MultiDetPreparedData(
        detectors=prepared_detectors,
        detector_data=detector_data,
        trigger_time=float(trigger_time_inferred),
        duration=float(duration),
        post_trigger_duration=float(post_trigger_duration),
        df=float(df),
        gmst=float(gmst),
        window=np.asarray(reference_window, dtype=np.float64) if reference_window is not None else None,
        roll_off=float(roll_off),
    )
