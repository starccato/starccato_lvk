import os

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