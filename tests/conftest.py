from unittest.mock import patch
import os
import pytest
from starccato_lvk.acquisition.io.determine_valid_segments import _get_valid_start_stops_for_one_file
from starccato_lvk.acquisition.io.strain_loader import strain_loader
from starccato_lvk.acquisition.io.utils import _get_data_files_and_gps_times
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
    with patch('starccato_lvk.acquisition.config.DATA_DIR', new=TEST_DIR):
        yield


@pytest.fixture
def mock_get_data_files_and_gps_times(monkeypatch):
    def fake_func(*args, **kwargs):
        return {
            i: f"file_{i}.hdf5" for i in range(100, 200, 10)
        }

    monkeypatch.setattr('starccato_lvk.acquisition.io.utils._get_data_files_and_gps_times', fake_func)


@pytest.fixture
def noise_trigger_time():
    """Get a trigger time for testing."""
    clean = False
    if clean:
        files = _get_data_files_and_gps_times()
        file = list(files.values())[0]
        start_stops = _get_valid_start_stops_for_one_file(file, segment_length=130, min_gap=10)
        return start_stops[0][1] - 1
    else:
        return 1263742976 + 100  # A time with valid data in test files


@pytest.fixture
def glitch_trigger_time():
    """Get a trigger time for testing."""
    return 1263748255.33508
