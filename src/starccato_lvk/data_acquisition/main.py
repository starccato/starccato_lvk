import click
from .io.determine_valid_segments import generate_times_for_valid_data
from .io.strain_loader import load_analysis_chunk_and_psd


@click.command("get_analysis_data")
@click.argument('trigger_time', type=float)
@click.option('--outdir', type=str, default='out', help='Output directory for analysis chunk and PSD')
def cli_get_analysis_data(trigger_time, outdir):
    load_analysis_chunk_and_psd(
        trigger_time=trigger_time,
        outdir=outdir
    )


@click.command("get_valid_times")
@click.option('--gps_start', type=float, default=1256656000, help='Start GPS time')
@click.option('--gps_end', type=float, default=1256660000, help='End GPS time')
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
