"""Explicit Whittle noise evidence and a noise-scale (PSD-amplitude) marginal.

The production BCR works in the JIM *noise-relative* convention: the likelihood
evaluates ``log L_jim(h) = <h|d> - <h|h>/2``, dropping the ``-<d|d>/2`` and the
Gaussian normalisation. Those dropped terms are exactly the absolute Whittle
noise log-evidence ``log Z_N``, so morphZ/nested return ``log(Z_X / Z_N)`` and
``Z_N`` cancels in the odds -- hence ``logZ_noise = 0`` is a correct *convention*.

These functions compute ``Z_N`` explicitly, for two reasons:

1. **Validation** of that convention: ``log L_full(h) = log L_jim(h) + log Z_N``
   and, at the evidence level, ``log Z_S_absolute = log Z_N + log Z_S_jim``.
2. **Robustness for real data**: a noise hypothesis with a free PSD-amplitude
   scale ``eta`` (PSD = ``eta * S``) has a *marginal* ``Z_N(eta)`` that does NOT
   cancel. When the data carry more power than the estimated PSD (non-stationary
   noise, mis-estimated PSD), the noise model can absorb it instead of the signal
   model overfitting -- exactly the failure mode the real-data smoke test showed.

Convention (matching JIM): inner product ``<a|b> = 4 df Re sum a* b / S`` with
one-sided PSD ``S``. The per-bin circular-complex-normal variance is ``S/(2 df)``,
giving the Gaussian normalisation ``C = -sum_i log(pi S_i / (2 df))`` over the
in-band bins (PSD finite; out-of-band / notched bins have ``S = +inf`` and drop out).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass
class BandQuantities:
    """Band-summed quantities for one prepared detector (JIM convention)."""

    dd: float        # <d|d> over the analysis band
    n_bins: int      # number of in-band (complex) frequency bins
    log_norm: float  # C = -sum log(pi S_i / (2 df))


def band_quantities(detector) -> BandQuantities:
    """Extract ``<d|d>``, the in-band bin count, and the Gaussian normalisation.

    Uses ``detector.psd``/``detector.data.fd`` directly (full rfft grid). The
    analysis band and line notches are encoded as ``+inf`` PSD entries, so the
    finite-PSD mask selects exactly the bins the likelihood actually uses.
    """
    psd = np.asarray(detector.psd.values, dtype=np.float64)
    d = np.asarray(detector.data.fd)
    freqs = np.asarray(detector.psd.frequencies, dtype=np.float64)
    df = float(freqs[1] - freqs[0])
    finite = np.isfinite(psd) & (psd > 0)
    dd = float(4.0 * df * np.sum((np.abs(d) ** 2 / psd)[finite]))
    n_bins = int(finite.sum())
    log_norm = float(-np.sum(np.log(np.pi * psd[finite] / (2.0 * df))))
    return BandQuantities(dd=dd, n_bins=n_bins, log_norm=log_norm)


def full_gaussian_loglike(detector, residual_dd: float) -> float:
    """Full Whittle log-likelihood for a residual ``<d-h|d-h>`` at fixed PSD.

    With ``residual_dd = <d|d>`` (no signal) this is the noise log-evidence
    :func:`whittle_log_zn`.
    """
    return -0.5 * float(residual_dd) + band_quantities(detector).log_norm


def whittle_log_zn(detector) -> float:
    """Absolute Whittle log-evidence of the noise-only hypothesis (fixed PSD)."""
    q = band_quantities(detector)
    return -0.5 * q.dd + q.log_norm


def whittle_log_zn_marginal(detector, *, a: float = 100.0, b: float | None = None) -> float:
    """Noise-scale-marginalised log Z_N: ``PSD = eta * S``, ``eta ~ InvGamma(a, b)``.

    Closed form (inverse-gamma is conjugate to the Gaussian scale):

        log Z_N = C + a log b - logGamma(a) + logGamma(n + a) - (n + a) log(<d|d>/2 + b)

    The default prior has mean ``b/(a-1) = 1`` (``b = a-1``) and standard deviation
    ``~1/sqrt(a-2)``; ``a = 100`` allows ~10% PSD-amplitude uncertainty. Unlike the
    fixed-PSD ``Z_N``, this marginal does not cancel in the BCR.
    """
    q = band_quantities(detector)
    b = (a - 1.0) if b is None else float(b)
    return float(
        q.log_norm
        + a * np.log(b)
        - gammaln(a)
        + gammaln(q.n_bins + a)
        - (q.n_bins + a) * np.log(q.dd / 2.0 + b)
    )


def noise_scale_marginal_loglike(detector, residual_dd: float, *,
                                 a: float = 100.0, b: float | None = None) -> float:
    """eta-marginalised log-likelihood for a residual ``<d-h|d-h>``.

    This is the Student-t-like likelihood obtained by analytically marginalising the
    PSD-amplitude ``eta`` at fixed waveform parameters. Replacing the Gaussian
    likelihood with this in the sampler makes the signal/glitch/noise evidences all
    PSD-amplitude-robust; ``residual_dd = <d|d>`` recovers :func:`whittle_log_zn_marginal`.
    """
    q = band_quantities(detector)
    b = (a - 1.0) if b is None else float(b)
    return float(
        q.log_norm
        + a * np.log(b)
        - gammaln(a)
        + gammaln(q.n_bins + a)
        - (q.n_bins + a) * np.log(float(residual_dd) / 2.0 + b)
    )
