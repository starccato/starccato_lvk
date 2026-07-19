import sys

import pytest


def test_production_plots_do_not_eagerly_import_legacy_modules():
    from starccato_lvk.analysis.post_proc import jim_plots  # noqa: F401

    assert (
        "starccato_lvk.analysis.post_proc.plot_diagnostics" not in sys.modules
    )
    assert (
        "starccato_lvk.analysis.post_proc.plot_posterior_predictive"
        not in sys.modules
    )


def test_legacy_plot_export_warns_before_lazy_import():
    import starccato_lvk.analysis.post_proc as post_proc

    with pytest.warns(DeprecationWarning, match="jim_plots"):
        plot_diagnostics = post_proc.plot_diagnostics
    assert callable(plot_diagnostics)
