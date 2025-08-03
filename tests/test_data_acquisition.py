from starccato_lvk.data_acquisition.io.strain_loader import load_analysis_chunk_and_psd, strain_loader
from starccato_lvk.data_acquisition.io.determine_valid_segments import load_state_vector, generate_times_for_valid_data, plot_valid_segments
from conftest import get_trigger_time
import os
import jax

GLITCH_TIME = 1263748255.33508

"""
├── L-L1_GWOSC_O3b_4KHZ_R1-1256652800-4096.hdf5
├── L-L1_GWOSC_O3b_4KHZ_R1-1263747072-4096.hdf5
└── L-L1_GWOSC_O3b_4KHZ_R1-9999999999-4096.hdf5
"""


def test_data_loader(outdir, mock_data_dir):
    strain_loader(GLITCH_TIME, outdir=outdir)
    assert os.path.exists(os.path.join(outdir, f"analysis_chunk_{int(GLITCH_TIME)}.png"))
    assert os.path.exists(os.path.join(outdir, f"analysis_chunk_{int(GLITCH_TIME)}.hdf5"))


def test_load_state_vector(outdir, mock_data_dir):
    gps_start = GLITCH_TIME - 65
    gps_end = GLITCH_TIME + 1
    state_vector = load_state_vector(gps_start, gps_end)
    assert state_vector is not None
    plot = state_vector.plot(insetlabels=True)
    plot.savefig(os.path.join(outdir, f"state_vector_{int(GLITCH_TIME)}.png"))

    T0 = 1263743000
    T1 = 1263750000
    times = generate_times_for_valid_data(T0, T1, segment_length=130, min_gap=10, outdir=outdir)
    assert len(times) > 0

def test_utils(mock_get_data_files_and_gps_times):
    from starccato_lvk.data_acquisition.io.utils import _get_fnames_for_range
    files = _get_fnames_for_range(110, 150)
    assert len(files) > 0
    files_2 = _get_fnames_for_range(110, 151)
    assert len(files) < len(files_2)
    assert len(_get_fnames_for_range(10, 11)) == 0


def test_load_blips():
    from starccato_lvk.data_acquisition.io.glitch_catalog import get_blip_trigger_time
    blips_times = [get_blip_trigger_time(i) for i in range(10)]
    assert len(blips_times)==10

def test_injection(outdir, mock_data_dir):
    t0 = get_trigger_time()
    rng = jax.random.PRNGKey(0)
    strain_loader(t0, outdir=outdir, add_injection=True, distance=1e20, rng=rng)