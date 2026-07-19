from __future__ import annotations

import glob
import os

from starccato_lvk.acquisition.main import strain_loader
from starccato_lvk.analysis.multidet_data_prep import (
    prepare_multi_detector_data,
)


def test_real_bundle_prepares_for_analysis(
    outdir, strain_data_fetcher, noise_trigger_time
):
    out = f"{outdir}/starccato_analysis"
    os.makedirs(out, exist_ok=True)
    strain_loader(
        noise_trigger_time,
        outdir=out,
        data_fetcher=strain_data_fetcher,
        detector="L1",
        require_cat3=False,
    )
    bundle_path = glob.glob(f"{out}/analysis_bundle_*.hdf5")[0]

    prepared = prepare_multi_detector_data(
        detectors=["L1"],
        bundle_paths={"L1": bundle_path},
    )
    assert [det.name for det in prepared.detectors] == ["L1"]
    assert prepared.duration == 4.0
    assert prepared.detector_data["L1"].band_mask.any()
