"""
Starccato LVK Command Line Interface

This module provides a command-line interface for the Starccato LVK toolkit,
offering tools for gravitational-wave data acquisition and analysis. Commands are
grouped under **acquire** (data pulls) and a top-level **run** analysis command.

Usage Examples:
    # Acquire data for a specific trigger
    python -m starccato_lvk.cli acquire data 123 --trigger-type blip

    # Run analysis on data files
    python -m starccato_lvk.cli run data.hdf5 psd.hdf5 ./output

    # View help for any command
    python -m starccato_lvk.cli --help
    python -m starccato_lvk.cli acquire --help
"""

import os

import click
from .analysis.main import run_starccato_analysis
from .acquisition.main import cli_collect_lvk_data, cli_get_analysis_data


@click.group()
@click.version_option("1.0.0")
def cli():
    """Starccato LVK - Gravitational Wave Analysis Toolkit

    A comprehensive toolkit for gravitational wave data acquisition and analysis
    using supernova and blip signal models.
    """
    pass


@cli.group(name="acquire")
def acquire_group():
    """Data acquisition commands for fetching and processing LVK data."""
    pass


@acquire_group.command("data")
@click.argument('index', type=int)
@click.option('--trigger-type', type=click.Choice(['blip', 'noise']),
              default='blip', help='Type of trigger to analyze')
@click.option('--outdir', type=str, default=None,
              help='Output directory (default: outdir_<trigger_type>)')
def acquire_data(index, trigger_type, outdir):
    """Acquire LVK data for a specific trigger index.

    Downloads and processes gravitational wave strain data for the specified
    trigger index. Can handle both blip triggers and noise triggers.

    Args:
        index: Trigger index to process
        trigger_type: Type of trigger ('blip' or 'noise')
        outdir: Output directory for results
    """
    if outdir is None:
        outdir = f'outdir_{trigger_type}'

    click.echo(f"Acquiring {trigger_type} data for index {index}...")
    click.echo(f"Output directory: {outdir}")

    # Call the existing function
    cli_get_analysis_data(index, trigger_type, outdir)

    click.echo(f"Data acquisition complete for index {index}")


@acquire_group.command("batch")
@click.argument('num_samples', type=int)
@click.option('--outdir', type=str, default='outdir_batch',
              help='Output directory for batch acquisition')
def acquire_batch(num_samples, outdir):
    """Acquire multiple LVK data samples in batch.

    Downloads and processes gravitational wave strain data for multiple
    trigger indices. Processes both blip and noise triggers for each index.

    Args:
        num_samples: Number of trigger indices to process
        outdir: Base output directory for results
    """
    click.echo(f"Starting batch acquisition of {num_samples} samples...")
    click.echo(f"Output directory: {outdir}")

    # Call the existing batch function
    cli_collect_lvk_data(num_samples, outdir)

    click.echo(f"Batch acquisition complete for {num_samples} samples")


@cli.command(name="run")
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("psd_path", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--injection-model",
    type=click.Choice(["ccsne", "blip"]),
    default=None,
    help="Model to use for signal injection.",
)
@click.option(
    "--num-samples",
    type=int,
    default=2000,
    show_default=True,
    help="Number of posterior samples to draw from the nested sampler.",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    help="Force rerun even if results already exist in the output directory.",
)
@click.option(
    "--test-mode",
    is_flag=True,
    help="Run with reduced nested-sampling settings for quick tests.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose nested-sampling output.",
)
@click.option(
    "--save-artifacts/--skip-artifacts",
    default=True,
    show_default=True,
    help="Save (or skip) inference files and diagnostic plots.",
)
def run_command(
    data_path: str,
    psd_path: str,
    outdir: str,
    injection_model: str | None,
    num_samples: int,
    force_rerun: bool,
    test_mode: bool,
    verbose: bool,
    save_artifacts: bool,
) -> None:
    """Run supernova signal analysis on LVK data."""

    run_starccato_analysis(
        data_path=data_path,
        psd_path=psd_path,
        outdir=outdir,
        injection_model_type=injection_model,
        num_samples=num_samples,
        force_rerun=force_rerun,
        test_mode=test_mode,
        verbose=verbose,
        save_artifacts=save_artifacts,
    )
    summary_path = os.path.join(outdir, "comparison_summary.csv")
    click.echo(f"Analysis complete. Summary written to {summary_path}")



if __name__ == "__main__":
    cli()
