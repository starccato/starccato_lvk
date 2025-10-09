import click
import jax

from tqdm import tqdm

from .io.determine_valid_segments import generate_times_for_valid_data
from .io.glitch_catalog import get_blip_trigger_time
from .io.only_noise_data import get_noise_trigger_time
from .io.strain_loader import strain_loader


DEFAULT_SIGNAL_DISTANCE = 1e23


def get_analysis_data(idx: int, trigger_type: str = "blip", outdir: str = "."):
    """Dispatch helper that backs both the CLI command and internal calls."""
    if trigger_type not in {"blip", "noise", "signal"}:
        raise ValueError(f"Unknown trigger_type '{trigger_type}'")

    resolved_outdir = outdir if outdir != "." else f"outdir_{trigger_type}"

    if trigger_type == "blip":
        trigger_time = get_blip_trigger_time(idx)
        strain_loader(trigger_time, outdir=resolved_outdir)
    elif trigger_type == "noise":
        trigger_time = get_noise_trigger_time(idx)
        strain_loader(trigger_time, outdir=resolved_outdir)
    else:  # signal
        trigger_time = get_noise_trigger_time(idx)
        rng_key = jax.random.PRNGKey(idx)
        strain_loader(
            trigger_time,
            outdir=resolved_outdir,
            add_injection=True,
            distance=DEFAULT_SIGNAL_DISTANCE,
            rng=rng_key,
        )


@click.command("get_analysis_data")
@click.argument('idx', type=int, default=0)
@click.option('--trigger_type', type=str, default='blip', help='Type of trigger: blip, noise, or signal')
@click.option('--outdir', type=str, default='.', help='Output directory for analysis chunk and PSD')
def cli_get_analysis_data(idx, trigger_type, outdir):
    get_analysis_data(idx, trigger_type=trigger_type, outdir=outdir)


@click.command("get_valid_times")
@click.option('--gps_start', type=float, default=1256656000, help='Start GPS time')
@click.option('--gps_end', type=float, default=1256900000, help='End GPS time')
@click.option('--segment_length', type=int, default=130, help='Length of each data segment in seconds')
@click.option('--min_gap', type=int, default=10, help='Minimum gap between segments in seconds')
@click.option('--outdir', type=str, default='outdir', help='Output directory for valid segments')
def cli_get_valid_times(gps_start, gps_end, segment_length, min_gap, outdir):
    generate_times_for_valid_data(
        gps_start=gps_start,
        gps_end=gps_end,
        segment_length=segment_length,
        min_gap=min_gap,
        outdir='outdir'
    )


@click.command("collect_lvk_data")
@click.argument('num', type=int, default=100)
def cli_collect_lvk_data(num: int):
    for i in tqdm(range(num)):
        get_analysis_data(i, trigger_type='blip')
        get_analysis_data(i, trigger_type='noise')
        get_analysis_data(i, trigger_type='signal')
