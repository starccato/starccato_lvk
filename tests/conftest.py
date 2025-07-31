
from unittest.mock import patch
import os
import pytest
from starccato_lvk.data_acquisition.io.determine_valid_segments import _get_valid_start_stops_for_one_file
from starccato_lvk.data_acquisition.io.strain_loader import strain_loader
from starccato_lvk.data_acquisition.io.utils import _get_data_files_and_gps_times
import jax


HERE = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(HERE, "test_data/L1")


@pytest.fixture()
def outdir():
    outdir = os.path.join(HERE, "test_outdir")
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture
def mock_data_dir():
    with patch('starccato_lvk.data_acquisition.config.DATA_DIR', new=TEST_DIR):
        yield


@pytest.fixture
def mock_get_data_files_and_gps_times(monkeypatch):
    def fake_func(*args, **kwargs):
        return {
            i: f"file_{i}.hdf5" for i in range(100, 200, 10)
        }
    monkeypatch.setattr('starccato_lvk.data_acquisition.io.utils._get_data_files_and_gps_times', fake_func)





def get_trigger_time(clean=False):
    """Get a trigger time for testing."""
    if clean:
        files = _get_data_files_and_gps_times()
        file = list(files.values())[0]
        start_stops = _get_valid_start_stops_for_one_file(file, segment_length=130, min_gap=10)
        return start_stops[0][1] - 1
    else:
        return 1256655805


@pytest.fixture
def analysis_data(outdir, mock_data_dir):
    t0 = get_trigger_time(clean=False)
    files = dict(
        strain_file=f"{outdir}/analysis_chunk_{int(t0)}.hdf5",
        psd_file=f"{outdir}/psd_{int(t0)}.hdf5",
    )

    if not os.path.exists(files["strain_file"]) or not os.path.exists(files["psd_file"]):
        rng = jax.random.PRNGKey(0)
        strain_loader(t0, outdir=outdir, add_injection=True, distance=1e23, rng=rng)
    return files
