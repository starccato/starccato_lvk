import os
import contextlib
import pytest
from unittest.mock import patch

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
