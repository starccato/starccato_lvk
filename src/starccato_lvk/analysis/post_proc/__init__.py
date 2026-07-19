"""Posterior plotting helpers.

The production plotting path lives in :mod:`.jim_plots`. Older ArviZ-based
helpers remain lazily importable for one compatibility cycle without pulling
their optional plotting stack into normal analysis imports.
"""

from importlib import import_module
import warnings

__all__ = [
    "plot_diagnostics",
    "plot_posterior_comparison",
    "plot_posterior_predictive",
]

_LEGACY_EXPORTS = {
    "plot_diagnostics": (".plot_diagnostics", "plot_diagnostics"),
    "plot_posterior_comparison": (
        ".plot_posterior_predictive",
        "plot_posterior_comparison",
    ),
    "plot_posterior_predictive": (
        ".plot_posterior_predictive",
        "plot_posterior_predictive",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LEGACY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    warnings.warn(
        f"starccato_lvk.analysis.post_proc.{name} is deprecated; "
        "use starccato_lvk.analysis.post_proc.jim_plots instead",
        DeprecationWarning,
        stacklevel=2,
    )
    module = import_module(module_name, package=__name__)
    return getattr(module, attribute_name)
