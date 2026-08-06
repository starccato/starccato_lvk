"""Shared blip-core extraction: whiten -> locate peak -> cut -> PCA -> un-whiten.

Used by the Phase 0 diagnostics (studies/blip_stratified_sample.py,
studies/blip_gengli_ff_by_freq.py, studies/blip_denoiser_injection_test.py) and
by the Phase 1 training-set builder. See
starccato_lvk/docs/blip_prior_rebuild_plan.md for the plan this supports.

Extraction band is wide (30-1024 Hz) by design -- not the 300-800 Hz analysis
band -- so a trained decoder is not structurally band-limited and a future band
ablation stays possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from starccato_lvk.analysis.multidet_data_prep import prepare_multi_detector_data

EXTRACT_FLOW = 30.0
EXTRACT_FMAX = 1024.0
CORE_HALF_WIDTH = 256  # samples either side of the peak, at 4096 Hz -> 512 total
SAMPLE_RATE = 4096.0


@dataclass
class WhitenedSegment:
    whitened: np.ndarray  # time-domain, whitened, dimensionless
    psd: np.ndarray  # one-sided PSD on the rfft grid, +inf outside [flow,fmax]
    dt: float
    n: int


# A low `flow` gives the whitening filter a long impulse response; FFT-based
# whitening of a finite segment is circular, so that response wraps around
# and concentrates a boundary artifact at the segment edges unless the Tukey
# taper is wide enough to suppress it. The `prepare_multi_detector_data`
# default (roll_off=0.01, ~20 ms/edge on a 4 s segment) is nowhere near
# enough for flow=30 Hz: measured directly, the edge peaked at 7.37 sigma
# against a true in-segment peak of 0.78 sigma. roll_off=0.1 (~400 ms/edge)
# suppresses it to 0.04 sigma with no change to the recovered peak. This
# does NOT matter for the narrow 300-800 Hz band (short filter response,
# edge artifact never observed there) but is required for any EXTRACT-band
# (wide) whitening.
WIDE_BAND_ROLL_OFF = 0.1


def whiten_bundle(bundle_path: Path, detector: str, flow: float = EXTRACT_FLOW,
                  fmax: float = EXTRACT_FMAX, roll_off: float = WIDE_BAND_ROLL_OFF) -> WhitenedSegment:
    """Whiten one strain bundle with its own off-source PSD, band-limited."""
    d = prepare_multi_detector_data(
        (detector,), bundle_paths={detector: bundle_path}, flow=flow, fmax=fmax,
        roll_off=roll_off,
    ).detector_data[detector]
    psd = np.asarray(d.psd_likelihood)
    x = np.asarray(d.windowed_strain)
    n = x.size
    dt = float(d.dt)
    df = 1.0 / (n * dt)
    w = np.where(np.isfinite(psd) & (psd > 0), psd, np.inf)
    xw = np.fft.irfft(np.fft.rfft(x) * dt / np.sqrt(w / (4.0 * df)), n=n)
    return WhitenedSegment(whitened=xw, psd=psd, dt=dt, n=n)


def locate_peak(seg: WhitenedSegment, search_half_width_ms: float = 250.0) -> int:
    """Sample index of the largest |whitened strain| within the search window."""
    mid = seg.n // 2
    half = int(search_half_width_ms * 1e-3 / seg.dt)
    window = seg.whitened[mid - half: mid + half]
    return int(np.argmax(np.abs(window))) + mid - half


def peak_to_floor(seg: WhitenedSegment, peak_idx: int) -> float:
    """Ratio of the whitened peak to the RMS floor away from the transient.

    The floor is measured in the segment's INTERIOR quarters, not the outer
    ones. A wide, low-flow analysis band (e.g. EXTRACT_FLOW=30 Hz) has a long
    filter impulse response; FFT-based whitening of a finite segment is
    circular, so that long response wraps around and concentrates a boundary
    artifact right at the segment edges (indices near 0 and near n-1). An
    outer-quarter floor estimate captures that artifact, not genuine noise,
    and deflates the ratio -- measured directly on one real event: 1.44 with
    the outer-quarter floor vs 13.65 with an interior floor, a 9.5x error.
    The Tukey taper (`roll_off`, default 0.01 in `prepare_multi_detector_data`)
    is far too short (~20 ms per edge on a 4 s segment) to suppress this for a
    30 Hz high-pass; do not lower the analysis flow without also widening the
    taper.
    """
    quarter = seg.n // 4
    interior = seg.whitened[quarter: seg.n - quarter]
    floor = float(np.std(interior))
    return abs(float(seg.whitened[peak_idx])) / floor if floor > 0 else 0.0


def extract_core(seg: WhitenedSegment, peak_idx: int,
                 half_width: int = CORE_HALF_WIDTH, sign_align: bool = True) -> np.ndarray:
    """512-sample whitened core centred on the peak, optionally sign-aligned."""
    core = seg.whitened[peak_idx - half_width: peak_idx + half_width].copy()
    if core.size != 2 * half_width:
        raise ValueError(
            f"peak too close to segment edge: got {core.size} samples, "
            f"need {2 * half_width}"
        )
    if sign_align:
        core *= np.sign(core[half_width])
    return core


def pca_denoise(cores: np.ndarray, k: int, basis: np.ndarray | None = None,
                mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project each row of `cores` onto the top-k PCA components.

    If `basis`/`mean` are supplied (fit on a DIFFERENT set), this scores
    out-of-sample reconstruction rather than fitting to itself -- what the
    Phase 0c injection-recovery test and any leave-one-out check both need.
    Returns (denoised, basis_used, mean_used).
    """
    if mean is None:
        mean = cores.mean(axis=0)
    if basis is None:
        _, _, vt = np.linalg.svd(cores - mean, full_matrices=False)
        basis = vt[:k]
    centred = cores - mean
    denoised = mean + centred @ basis[:k].T @ basis[:k]
    return denoised, basis, mean


def unwhiten(core: np.ndarray, seg: WhitenedSegment, centre_idx: int | None = None) -> np.ndarray:
    """Invert `whiten_bundle`'s transform on a (possibly denoised) core.

    The core lives on a shorter time axis than the analysis segment, so this
    pads back to the segment length, applies the INVERSE of the whitening
    filter (multiply by sqrt(S(f)) instead of divide), and returns raw strain
    -- the physically meaningful quantity for a template.

    `centre_idx` MUST be the same sample index `extract_core` was cut around
    (typically the value `locate_peak` returned). Padding around the segment
    midpoint instead, when the peak sits elsewhere, misplaces every sample by
    the offset -- with the long, sinc-like impulse response a hard line notch
    produces, even a 1-2 sample misplacement destroys the phase and collapses
    the match. There is no safe default for this reason; the caller must know
    where the core came from.
    """
    n = seg.n
    half = core.size // 2
    if centre_idx is None:
        centre_idx = n // 2
    padded = np.zeros(n)
    padded[centre_idx - half: centre_idx + half] = core
    df = 1.0 / (n * seg.dt)
    w = np.where(np.isfinite(seg.psd) & (seg.psd > 0), seg.psd, np.inf)
    finite = np.isfinite(w)
    spec = np.fft.rfft(padded)
    raw_spec = np.zeros_like(spec)
    raw_spec[finite] = spec[finite] * np.sqrt(w[finite] / (4.0 * df)) / seg.dt
    return np.fft.irfft(raw_spec, n=n)
