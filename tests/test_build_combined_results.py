from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np


def _load_study_module():
    path = Path(__file__).resolve().parents[1] / "studies" / "build_combined_results.py"
    spec = importlib.util.spec_from_file_location("build_combined_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_psd_is_loaded_and_archived_as_float64(tmp_path):
    module = _load_study_module()
    source = tmp_path / "signal_median_PSD_H1.dat"
    source.write_text("300 1e-46\n301 2e-46\n")

    psd = module._load_dat(source, dtype=np.float64)
    assert psd is not None
    assert psd.dtype == np.float64
    assert np.all(psd[:, 1] > 0.0)

    archive = tmp_path / "combined.h5"
    with h5py.File(archive, "w") as h5f:
        group = h5f.create_group("posteriors/e1")
        group.create_dataset(
            "signal_median_psd_H1",
            data=np.asarray([[300.0, 0.0], [301.0, 0.0]], dtype=np.float32),
        )
        changed = module._add_or_upgrade_physical_psd(
            group, "signal_median_psd_H1", psd
        )
        stored = np.asarray(group["signal_median_psd_H1"])

    assert changed
    assert stored.dtype == np.float64
    np.testing.assert_array_equal(stored, psd)
