"""Robust log-evidence estimation built on top of morphZ.

This module wraps the morphZ bridge-sampling estimator with the safety net the
BCR ranking pipeline needs: a finite-value guard, bounded retries with jittered
proposal settings, and an optional fallback estimator (e.g. nested sampling) so
that no event silently collapses to ``NaN``.

morphZ is the production estimator for the ranking statistic, so its failures
must be observable. Every call returns an :class:`EvidenceResult` carrying the
method that actually produced the number, a status flag, and a human-readable
message; callers aggregate these to report a per-run failure rate instead of
discovering holes in the ROC after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from morphZ import evidence as _morphz_evidence
except ImportError:  # pragma: no cover - optional dependency
    _morphz_evidence = None


# Status constants for EvidenceResult.status.
STATUS_OK = "ok"
STATUS_FALLBACK = "fallback"
STATUS_FAILED = "failed"


@dataclass
class EvidenceResult:
    """Outcome of a single log-evidence estimation.

    Attributes
    ----------
    logZ, logZ_err
        Log-evidence and its estimated uncertainty. ``NaN`` only when every
        attempt (including any fallback) failed.
    method
        Which estimator produced the value: ``"morph"``, ``"nested"``, or
        ``"none"`` when nothing succeeded.
    status
        One of :data:`STATUS_OK`, :data:`STATUS_FALLBACK`, :data:`STATUS_FAILED`.
    n_attempts
        Number of morphZ attempts made before succeeding or giving up.
    message
        Diagnostic detail, e.g. the exception text from the last failure.
    """

    logZ: float
    logZ_err: float
    method: str
    status: str
    n_attempts: int = 1
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status != STATUS_FAILED and np.isfinite(self.logZ)

    @classmethod
    def failed(cls, message: str, n_attempts: int) -> "EvidenceResult":
        return cls(
            logZ=float("nan"),
            logZ_err=float("nan"),
            method="none",
            status=STATUS_FAILED,
            n_attempts=n_attempts,
            message=message,
        )


def morphz_log_evidence(
    post_samples: np.ndarray,
    log_posterior_function: Callable[[np.ndarray], float],
    *,
    log_posterior_values: Optional[np.ndarray] = None,
    param_names: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
    label: str = "evidence",
    n_resamples: int = 2000,
    n_estimations: int = 2,
    morph_type: str = "indep",
    n_retries: int = 2,
    verbose: bool = False,
) -> EvidenceResult:
    """Estimate ``log Z`` with morphZ, retrying on failure or non-finite output.

    morphZ uses internal randomness (``shuffle=True``), so each retry draws a
    fresh KDE/bridge split; retries also bump ``n_resamples`` to stabilise the
    bridge estimate. A retry is triggered both by an exception and by a
    non-finite estimate (the silent-NaN failure mode this guard exists for).

    Parameters mirror :func:`morphZ.evidence`; ``n_retries`` is the number of
    *additional* attempts beyond the first (so total attempts = ``n_retries+1``).
    """
    if _morphz_evidence is None:
        return EvidenceResult.failed("morphZ is not installed", n_attempts=0)

    post_samples = np.asarray(post_samples, dtype=np.float64)
    if post_samples.ndim != 2 or post_samples.shape[0] < 2:
        return EvidenceResult.failed(
            f"need a 2-D sample array with >=2 rows, got shape {post_samples.shape}",
            n_attempts=0,
        )
    logpost_values = (
        None if log_posterior_values is None else np.asarray(log_posterior_values)
    )
    param_names = list(param_names) if param_names is not None else None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    last_message = ""
    total_attempts = max(1, n_retries + 1)
    for attempt in range(total_attempts):
        # Grow the proposal budget on each retry for a steadier bridge estimate.
        resamples = int(n_resamples * (1.0 + 0.5 * attempt))
        try:
            res = _morphz_evidence(
                post_samples=post_samples,
                log_posterior_values=logpost_values,
                log_posterior_function=log_posterior_function,
                n_resamples=resamples,
                morph_type=morph_type,
                param_names=param_names,
                output_path=None if output_path is None else str(output_path / f"morphZ_{label}"),
                n_estimations=n_estimations,
                overwrite_path=True,
                verbose=verbose,
                show_progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface morphZ failure mode
            last_message = f"{type(exc).__name__}: {exc}"
            continue

        logZ, logZ_err = _reduce_morphz_estimates(res)
        if np.isfinite(logZ):
            return EvidenceResult(
                logZ=float(logZ),
                logZ_err=float(logZ_err),
                method="morph",
                status=STATUS_OK,
                n_attempts=attempt + 1,
                message="" if attempt == 0 else f"succeeded on attempt {attempt + 1}",
            )
        last_message = "morphZ returned a non-finite estimate"

    return EvidenceResult.failed(
        last_message or "morphZ failed for an unknown reason",
        n_attempts=total_attempts,
    )


def evidence_with_fallback(
    post_samples: np.ndarray,
    log_posterior_function: Callable[[np.ndarray], float],
    *,
    fallback: Optional[Callable[[], "EvidenceResult"]] = None,
    **morphz_kwargs,
) -> EvidenceResult:
    """Run morphZ; if it fails, defer to ``fallback`` (typically nested sampling).

    ``fallback`` is a zero-argument callable returning an :class:`EvidenceResult`
    so the caller controls the (potentially expensive) alternative estimator and
    its configuration. When morphZ fails and no fallback is supplied, the failed
    morphZ result is returned unchanged.
    """
    result = morphz_log_evidence(post_samples, log_posterior_function, **morphz_kwargs)
    if result.ok or fallback is None:
        return result

    try:
        fb = fallback()
    except Exception as exc:  # noqa: BLE001
        return EvidenceResult.failed(
            f"morphZ failed ({result.message}); fallback raised {type(exc).__name__}: {exc}",
            n_attempts=result.n_attempts,
        )

    if fb is None or not np.isfinite(fb.logZ):
        return EvidenceResult.failed(
            f"morphZ failed ({result.message}); fallback produced no finite estimate",
            n_attempts=result.n_attempts,
        )

    return EvidenceResult(
        logZ=float(fb.logZ),
        logZ_err=float(fb.logZ_err),
        method=fb.method if fb.method not in ("none", "") else "nested",
        status=STATUS_FALLBACK,
        n_attempts=result.n_attempts,
        message=f"morphZ failed ({result.message}); used fallback",
    )


def _reduce_morphz_estimates(res) -> tuple[float, float]:
    """Collapse morphZ's list-of-[logz, err] estimates into a single (logz, err).

    With ``n_estimations > 1`` morphZ returns one row per independent bridge run.
    We average the finite log-evidence estimates and report the larger of the
    averaged internal error and the scatter across estimates, so the quoted
    uncertainty reflects run-to-run variability when it dominates.
    """
    if res is None or len(res) == 0:
        return float("nan"), float("nan")

    rows = np.asarray(res, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    logz = rows[:, 0]
    err = rows[:, 1] if rows.shape[1] > 1 else np.full_like(logz, np.nan)

    finite = np.isfinite(logz)
    if not finite.any():
        return float("nan"), float("nan")
    logz = logz[finite]
    err = err[finite]

    logz_mean = float(np.mean(logz))
    internal_err = float(np.sqrt(np.nanmean(err**2))) if np.isfinite(err).any() else float("nan")
    scatter = float(np.std(logz, ddof=1)) if logz.size > 1 else 0.0
    combined_err = np.nanmax([internal_err, scatter])
    return logz_mean, float(combined_err)
