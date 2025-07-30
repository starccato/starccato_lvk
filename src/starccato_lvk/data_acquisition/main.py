import click
from .io.determine_valid_segments import generate_times_for_valid_data
from .io.strain_loader import strain_loader
from .io.glitch_catalog import get_blip_trigger_time
from .io.only_noise_data import get_noise_trigger_time

from tqdm import tqdm


@click.command("get_analysis_data")
@click.argument('idx', type=int, default=0)
@click.option('--trigger_type', type=str, default='blip', help='Type of trigger: blip, noise, or signal')
@click.option('--outdir', type=str, default='.', help='Output directory for analysis chunk and PSD')
def cli_get_analysis_data(idx, trigger_type, outdir):
    kwargs = {}
    if trigger_type == 'blip':
        kwargs = dict(
            trigger_time=get_blip_trigger_time(idx)
        )
    elif trigger_type == 'noise':
        kwargs = dict(
            trigger_time=get_noise_trigger_time(idx)
        )
    elif trigger_type == 'signal':
        kwargs = dict(
            add_injection=True,
            trigger_time=get_noise_trigger_time(idx),
            rng=idx,
            distance=kwargs['distance'],
        )
    if outdir == '.':
        outdir = f'outdir_{trigger_type}'
    strain_loader(**kwargs, outdir=outdir)


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
        cli_get_analysis_data(i, trigger_type='blip')
        cli_get_analysis_data(i, trigger_type='noise')
        cli_get_analysis_data(i, trigger_type='signal')
