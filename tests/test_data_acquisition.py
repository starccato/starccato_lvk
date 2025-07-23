from starccato_lvk.data_acquisition.io.strain_loader import load_analysis_chunk_and_psd
import os

GLITCH_TIME = 1263748255.33508

def test_data_loader(outdir, mock_data_dir):
    load_analysis_chunk_and_psd(GLITCH_TIME, outdir=outdir)
    assert os.path.exists(os.path.join(outdir, f"analysis_chunk_{int(GLITCH_TIME)}.png"))
    assert os.path.exists(os.path.join(outdir, f"analysis_chunk_{int(GLITCH_TIME)}.hdf5"))


def test_load_state_vector(outdir, mock_data_dir):
    from starccato_lvk.data_acquisition.io.strain_loader import load_state_vector
    gps_start = GLITCH_TIME - 65
    gps_end = GLITCH_TIME + 1
    state_vector = load_state_vector(gps_start, gps_end)
    assert state_vector is not None
    plot = state_vector.plot(insetlabels=True)
    plot.savefig(os.path.join(outdir, f"state_vector_{int(GLITCH_TIME)}.png"))