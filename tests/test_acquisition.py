from starccato_lvk.acquisition.io.strain_loader import strain_loader
from starccato_lvk.acquisition.io.utils import _get_fnames_for_range
from starccato_lvk.acquisition.io.glitch_catalog import get_blip_trigger_time
from starccato_lvk.acquisition.io.determine_valid_segments import load_state_vector, generate_times_for_valid_data, plot_valid_segments
import os

OUT = 'data_acquition'

"""
├── L-L1_GWOSC_O3b_4KHZ_R1-1256652800-4096.hdf5
├── L-L1_GWOSC_O3b_4KHZ_R1-1263747072-4096.hdf5
└── L-L1_GWOSC_O3b_4KHZ_R1-9999999999-4096.hdf5
"""


def test_data_loader(outdir, mock_data_dir, glitch_trigger_time):
    outdir = os.path.join(outdir, OUT)
    strain_loader(glitch_trigger_time, outdir=outdir)
    assert os.path.exists(os.path.join(outdir, f"analysis_chunk_{int(glitch_trigger_time)}.png"))
    assert os.path.exists(os.path.join(outdir, f"analysis_bundle_{int(glitch_trigger_time)}.hdf5"))


def test_load_state_vector(outdir, mock_data_dir,glitch_trigger_time):
    outdir = os.path.join(outdir, OUT)
    gps_start = glitch_trigger_time - 65
    gps_end = glitch_trigger_time + 1
    state_vector = load_state_vector(gps_start, gps_end)
    assert state_vector is not None
    plot = state_vector.plot(insetlabels=True)
    plot.savefig(os.path.join(outdir, f"state_vector_{int(glitch_trigger_time)}.png"))

    T0 = 1263743000
    T1 = 1263750000
    times = generate_times_for_valid_data(T0, T1, segment_length=130, min_gap=10, outdir=outdir)
    assert len(times) > 0

def test_utils(mock_get_data_files_and_gps_times):
    files = _get_fnames_for_range(110, 150)
    assert len(files) > 0
    files_2 = _get_fnames_for_range(110, 151)
    assert len(files) < len(files_2)
    assert len(_get_fnames_for_range(10, 11)) == 0


def test_load_blips():
    blips_times = [get_blip_trigger_time(i) for i in range(10)]
    assert len(blips_times)==10
