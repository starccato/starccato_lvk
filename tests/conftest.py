from unittest.mock import patch
import os
import pytest
from gwpy.timeseries import TimeSeries
from starccato_lvk.acquisition import config
from starccato_lvk.acquisition.io.determine_valid_segments import (
    _get_valid_start_stops_for_one_file,
)
from starccato_lvk.acquisition.io.utils import _get_data_files_and_gps_times

HERE = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(HERE, "test_data/L1")


@pytest.fixture()
def outdir():
    outdir = os.path.join(HERE, "test_outdir")
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope="session")
def strain_data_fetcher():
    files = []
    for root, _, names in os.walk(TEST_DIR):
        files.extend(
            os.path.join(root, name)
            for name in names
            if name.endswith(".hdf5")
        )
    if files:
        with (
            patch.object(config, "DATA_DIR", TEST_DIR),
            patch.dict(config.DATA_DIRS, {"L1": TEST_DIR}, clear=False),
        ):
            yield None
        return

    cache = {}

    def fetch(gps_start: float, gps_end: float) -> TimeSeries:
        key = (float(gps_start), float(gps_end))
        if key not in cache:
            try:
                cache[key] = TimeSeries.fetch_open_data(
                    "L1", gps_start, gps_end, verbose=False
                )
            except Exception as exc:
                pytest.skip(f"GWOSC strain download unavailable: {exc}")
        return cache[key]

    yield fetch


@pytest.fixture
def mock_get_data_files_and_gps_times(monkeypatch):
    def fake_func(*args, **kwargs):
        # {gps_start: (path, duration)} with contiguous 10 s files
        return {i: (f"file_{i}.hdf5", 10) for i in range(100, 200, 10)}

    monkeypatch.setattr(
        "starccato_lvk.acquisition.io.utils._get_data_files_and_gps_times",
        fake_func,
    )


@pytest.fixture
def noise_trigger_time():
    """Get a trigger time for testing."""
    clean = False
    if clean:
        files = _get_data_files_and_gps_times()
        file = list(files.values())[0]
        start_stops = _get_valid_start_stops_for_one_file(
            file, segment_length=130, min_gap=10
        )
        return start_stops[0][1] - 1
    else:
        return 1263742976 + 100  # A time with valid data in test files


@pytest.fixture
def glitch_trigger_time():
    """Get a trigger time for testing."""
    return 1263748255.33508
