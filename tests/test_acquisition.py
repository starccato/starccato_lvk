from starccato_lvk.acquisition.io.strain_loader import (
    ANALYSIS_DURATION,
    strain_loader,
    load_analysis_bundle,
    load_analysis_chunk_and_psd,
)
from starccato_lvk.acquisition.io import plotting
from starccato_lvk.acquisition.io.utils import _get_fnames_for_range
from starccato_lvk.acquisition.io.glitch_catalog import get_blip_trigger_time
from starccato_lvk.acquisition.io.determine_valid_segments import (
    load_state_vector,
    generate_times_for_valid_data,
    plot_valid_segments,
)
import os
import pytest
import numpy as np
from gwpy.timeseries import TimeSeries

OUT = "data_acquition"

"""
├── L-L1_GWOSC_O3b_4KHZ_R1-1256652800-4096.hdf5
├── L-L1_GWOSC_O3b_4KHZ_R1-1263747072-4096.hdf5
└── L-L1_GWOSC_O3b_4KHZ_R1-9999999999-4096.hdf5
"""


def test_data_loader(outdir, strain_data_fetcher, glitch_trigger_time):
    outdir = os.path.join(outdir, OUT)
    strain_loader(
        glitch_trigger_time,
        outdir=outdir,
        data_fetcher=strain_data_fetcher,
        detector="L1",
        require_cat3=False,
    )
    assert os.path.exists(
        os.path.join(outdir, f"analysis_chunk_{int(glitch_trigger_time)}.png")
    )
    assert os.path.exists(
        os.path.join(
            outdir, f"analysis_bundle_{int(glitch_trigger_time)}.hdf5"
        )
    )
    bundle_path = os.path.join(
        outdir, f"analysis_bundle_{int(glitch_trigger_time)}.hdf5"
    )
    analysis_chunk, _, metadata = load_analysis_bundle(bundle_path)
    expected_samples = int(
        ANALYSIS_DURATION * analysis_chunk.sample_rate.value
    )
    assert len(analysis_chunk.value) == expected_samples
    assert "full_strain" in metadata
    assert len(metadata["full_strain"].value) > len(analysis_chunk.value)


def test_load_state_vector(outdir, strain_data_fetcher, glitch_trigger_time):
    if strain_data_fetcher is not None:
        pytest.skip("state-vector test requires local O3b HDF5 fixtures")
    outdir = os.path.join(outdir, OUT)
    gps_start = glitch_trigger_time - 65
    gps_end = glitch_trigger_time + 1
    state_vector = load_state_vector(gps_start, gps_end)
    assert state_vector is not None
    plot = state_vector.plot(insetlabels=True)
    plot.savefig(
        os.path.join(outdir, f"state_vector_{int(glitch_trigger_time)}.png")
    )

    T0 = 1263743000
    T1 = 1263750000
    times = generate_times_for_valid_data(
        T0, T1, segment_length=130, min_gap=10, outdir=outdir
    )
    assert len(times) > 0


def test_utils(mock_get_data_files_and_gps_times):
    files = _get_fnames_for_range(110, 150)
    assert len(files) > 0
    files_2 = _get_fnames_for_range(110, 151)
    assert len(files) < len(files_2)
    assert len(_get_fnames_for_range(10, 11)) == 0


def test_load_blips():
    blips_times = [get_blip_trigger_time(i) for i in range(10)]
    assert len(blips_times) == 10


def test_qtransform_nan_in_diagnostic_window_is_skipped(monkeypatch):
    data = TimeSeries(np.ones(4096))
    data.value[0] = np.nan

    class Axis:
        def __init__(self):
            self.title = None

        def axis(self, *_args):
            pass

        def set_title(self, title):
            self.title = title

    axes = [Axis() for _ in range(4)]

    def fail_q_transform(**_kwargs):
        raise AssertionError("q-transform should not run for non-finite data")

    monkeypatch.setattr(data, "q_transform", fail_q_transform)
    plotting.plot_qtransform(data, event_time=0.0, axes=axes)

    assert axes[0].title == "Q-transform unavailable (non-finite samples in full window)"


def test_plot_proceeds_when_qtransform_raises(monkeypatch, tmp_path):
    data = TimeSeries(np.ones(4096))

    def fail_q_transform(**_kwargs):
        raise ValueError("GWpy diagnostic failure")

    monkeypatch.setattr(data, "q_transform", fail_q_transform)
    plotting.plot(data, data.psd(), event_time=0.0, fname=tmp_path / "diagnostic.png")

    assert (tmp_path / "diagnostic.png").exists()


@pytest.mark.parametrize("bad_region", ["analysis", "PSD"])
def test_nonfinite_inference_strain_is_rejected(bad_region):
    trigger = 1_000.0
    sample_rate = 128.0

    def fetch(start, end):
        times = np.arange(start, end, 1 / sample_rate)
        values = np.ones(times.size)
        if bad_region == "analysis":
            values[np.argmin(np.abs(times - trigger))] = np.nan
        else:
            values[np.argmin(np.abs(times - (trigger - 3.0)))] = np.nan
        return TimeSeries(values, times=times)

    with pytest.raises(ValueError, match=rf"{bad_region} strain contains 1 non-finite"):
        load_analysis_chunk_and_psd(trigger, data_fetcher=fetch)
