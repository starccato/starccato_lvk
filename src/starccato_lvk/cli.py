"""
Starccato LVK Command Line Interface

This CLI provides data acquisition utilities and a JIM-based multi-detector
analysis workflow for Starccato waveform models.

Usage Examples:
    # Acquire data for a specific trigger
    python -m starccato_lvk.cli acquire data 123 --trigger-type blip

    # Run analysis using bundles for H1 and L1
    python -m starccato_lvk.cli run ./outdir --bundle H1=path/to/H1.hdf5 --bundle L1=path/to/L1.hdf5

    # Run analysis by fetching live data around a trigger time
    python -m starccato_lvk.cli run ./outdir --detector H1 --detector L1 --trigger-time 1126259642.4

    # View detailed help
    python -m starccato_lvk.cli --help
    python -m starccato_lvk.cli run --help
"""

import json
from pathlib import Path
from typing import Sequence

import click
from starccato_jax.waveforms import MODELS

from .analysis.main import run_starccato_analysis, run_bcr_posteriors
from .acquisition.main import cli_collect_lvk_data, cli_get_analysis_data
from .workflows import load_analysis_config, run_event_workflow
from .workflows.run_event import (
    CONFIG_DEFAULT,
    EVENTS_DIR_DEFAULT,
    prepare_event_lists_from_files,
    read_event_list,
)


@click.group()
@click.version_option("1.0.0")
def cli():
    """Starccato LVK: VAE PE toolkit for CCSNe LnZ computation.

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


@cli.group(name="events")
def events_group():
    """Event list management and batch-analysis helpers."""
    pass


@events_group.command("generate")
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=str(EVENTS_DIR_DEFAULT),
    show_default=True,
    help="Directory to write scenario event lists.",
)
@click.option(
    "--blip-count",
    type=int,
    default=200,
    show_default=True,
    help="Number of blip glitch GPS times to include.",
)
@click.option(
    "--noise-file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Text file containing noise GPS times (one per line).",
)
@click.option(
    "--noise-inj-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional text file of noise+injection GPS times (defaults to noise file).",
)
def events_generate(outdir, blip_count, noise_file, noise_inj_file):
    """Generate scenario GPS lists used by SLURM workflows."""
    try:
        prepare_event_lists_from_files(
            Path(outdir),
            blip_count=blip_count,
            noise_file=Path(noise_file),
            noise_inj_file=Path(noise_inj_file) if noise_inj_file else None,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote event lists to {outdir}")


@events_group.command("run")
@click.option("--scenario", type=click.Choice(["blip", "noise", "noise_inj"]), required=True)
@click.option("--index", type=int, required=True, help="Event index (e.g. SLURM_ARRAY_TASK_ID).")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=str(CONFIG_DEFAULT),
    show_default=True,
    help="Analysis configuration YAML.",
)
@click.option(
    "--events-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=str(EVENTS_DIR_DEFAULT),
    show_default=True,
    help="Directory containing events_<scenario>.txt files.",
)
@click.option("--force", is_flag=True, help="Re-run analysis even if summary exists.")
@click.option("--distance", type=float, default=1.0, show_default=True, help="Injection distance scale for noise_inj scenario.")
@click.option(
    "--stage",
    type=click.Choice(["prep", "analysis", "both"]),
    default="both",
    show_default=True,
    help="Which workflow stage to execute.",
)
def events_run(scenario, index, config, events_dir, force, distance, stage):
    """Run the high-level event workflow for a single GPS trigger."""
    cfg = load_analysis_config(Path(config))
    gps = read_event_list(scenario, index, root=Path(events_dir))
    result = run_event_workflow(
        cfg,
        scenario,
        gps,
        index,
        force=force,
        injection_distance=distance,
        stage=stage,
    )
    if result is None:
        click.echo("Workflow completed with no new analysis (results already present).")
    else:
        click.echo(f"Analysis complete for {scenario} @ {gps}. BCR={result.get('bcr')}")


@cli.command(name="run")
@click.argument("outdir", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--detector",
    "detectors",
    multiple=True,
    default=("H1",),
    help="Detector(s) to include (repeatable). Default: H1.",
)
@click.option(
    "--bundle",
    "bundle_pairs",
    multiple=True,
    help="Detector bundle mapping of the form DETECTOR=PATH. Repeat for multiple detectors.",
)
@click.option(
    "--trigger-time",
    type=float,
    default=None,
    help="GPS trigger time for data acquisition (required if bundles are not supplied).",
)
@click.option(
    "--model",
    "model_types",
    type=click.Choice(MODELS),
    multiple=True,
    help="Model(s) to analyse. Defaults to all available models.",
)
@click.option(
    "--sampler",
    type=click.Choice(["nuts", "nested"]),
    default="nuts",
    show_default=True,
    help="Inference engine to use.",
)
@click.option("--num-samples", type=int, default=1000, show_default=True, help="NumPyro samples (NUTS).")
@click.option("--num-warmup", type=int, default=500, show_default=True, help="NumPyro warmup steps (NUTS).")
@click.option("--num-chains", type=int, default=1, show_default=True, help="NumPyro chains (NUTS).")
@click.option("--num-live-points", type=int, default=500, show_default=True, help="Nested sampling live points.")
@click.option("--max-samples", type=int, default=20000, show_default=True, help="Nested sampling max samples.")
@click.option("--latent-sigma", type=float, default=1.0, show_default=True, help="Latent prior standard deviation.")
@click.option("--log-amp-sigma", type=float, default=1.0, show_default=True, help="Log-amplitude prior sigma.")
@click.option("--rng-seed", type=int, default=0, show_default=True, help="Random seed for reproducibility.")
@click.option(
    "--extrinsic",
    "extrinsic_pairs",
    multiple=True,
    help="Extrinsic parameter override in the form key=value (e.g., ra=1.0).",
)
@click.option(
    "--save-artifacts/--no-save-artifacts",
    default=True,
    show_default=True,
    help="Save inference summaries and samples.",
)
def run_command(
    outdir: str,
    detectors: Sequence[str],
    bundle_pairs: Sequence[str],
    trigger_time: float | None,
    model_types: Sequence[str],
    sampler: str,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    num_live_points: int,
    max_samples: int,
    latent_sigma: float,
    log_amp_sigma: float,
    rng_seed: int,
    extrinsic_pairs: Sequence[str],
    save_artifacts: bool,
) -> None:
    """Run Starccato analysis with the JIM transient likelihood."""

    detector_list = list(detectors) if detectors else ["H1"]
    bundle_map = {}
    for pair in bundle_pairs:
        if "=" not in pair:
            raise click.BadOptionUsage("--bundle", f"Bundle specification '{pair}' must be DETECTOR=PATH.")
        det, path = pair.split("=", 1)
        bundle_map[det.upper()] = path

    if not bundle_map and trigger_time is None:
        raise click.BadOptionUsage(
            "--trigger-time",
            "Provide a trigger time when bundles are not supplied.",
        )

    extrinsics = {}
    for pair in extrinsic_pairs:
        if "=" not in pair:
            raise click.BadOptionUsage("--extrinsic", f"Extrinsic specification '{pair}' must be key=value.")
        key, value = pair.split("=", 1)
        try:
            extrinsics[key] = float(value)
        except ValueError as exc:
            raise click.BadOptionUsage("--extrinsic", f"Value '{value}' for '{key}' is not a float.") from exc

    models_to_run = list(model_types) if model_types else list(MODELS)

    run_starccato_analysis(
        detectors=detector_list,
        outdir=outdir,
        bundle_paths=bundle_map if bundle_map else None,
        trigger_time=trigger_time,
        model_types=models_to_run,
        sampler=sampler,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        num_live_points=num_live_points,
        max_samples=max_samples,
        latent_sigma=latent_sigma,
        log_amp_sigma=log_amp_sigma,
        rng_seed=rng_seed,
        extrinsic_params=extrinsics if extrinsics else None,
        save_artifacts=save_artifacts,
    )
    click.echo(f"Analysis complete. Results stored in {outdir}")


@cli.command(name="run-bcr")
@click.argument("outdir", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--bundle",
    "bundle_pairs",
    multiple=True,
    required=True,
    help="Detector bundle mapping of the form DETECTOR=PATH. Repeat for multiple detectors.",
)
@click.option(
    "--detector",
    "detectors",
    multiple=True,
    default=(),
    help="Optional explicit detector order (defaults to bundle keys).",
)
@click.option("--signal-model", type=click.Choice(MODELS), default="ccsne", show_default=True)
@click.option("--glitch-model", type=click.Choice(MODELS), default="blip", show_default=True)
@click.option("--num-warmup", type=int, default=500, show_default=True)
@click.option("--num-samples", type=int, default=1000, show_default=True)
@click.option("--num-chains", type=int, default=1, show_default=True)
@click.option("--signal-latent-sigma", type=float, default=1.0, show_default=True)
@click.option("--signal-log-amp-sigma", type=float, default=1.0, show_default=True)
@click.option("--glitch-latent-sigma", type=float, default=0.5, show_default=True)
@click.option("--glitch-log-amp-sigma", type=float, default=0.1, show_default=True)
@click.option("--rng-seed", type=int, default=0, show_default=True)
@click.option("--alpha", type=float, default=1.0, show_default=True, help="Prior odds factor α for BCR.")
@click.option("--beta", type=float, default=0.5, show_default=True, help="Glitch prior weight β for BCR.")
@click.option("--ci", nargs=2, type=int, default=(5, 95), show_default=True, help="Credible interval percentiles for predictive plots.")
@click.option(
    "--extrinsic",
    "extrinsic_pairs",
    multiple=True,
    help="Extrinsic parameter override key=value (e.g., ra=1.0).",
)
@click.option(
    "--save-artifacts/--no-save-artifacts",
    default=True,
    show_default=True,
    help="Save predictive plots and samples to disk.",
)
def run_bcr_command(
    outdir: str,
    bundle_pairs: Sequence[str],
    detectors: Sequence[str],
    signal_model: str,
    glitch_model: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    signal_latent_sigma: float,
    signal_log_amp_sigma: float,
    glitch_latent_sigma: float,
    glitch_log_amp_sigma: float,
    rng_seed: int,
    alpha: float,
    beta: float,
    ci: tuple[int, int],
    extrinsic_pairs: Sequence[str],
    save_artifacts: bool,
) -> None:
    """Run coherent CCSNe vs per-detector glitch analysis and compute BCR."""

    bundle_map: dict[str, str] = {}
    for pair in bundle_pairs:
        if "=" not in pair:
            raise click.BadOptionUsage("--bundle", f"Bundle specification '{pair}' must be DET=PATH.")
        det, path = pair.split("=", 1)
        bundle_map[det.upper()] = path

    if not bundle_map:
        raise click.UsageError("At least one --bundle DET=PATH is required.")

    detector_list = [d.upper() for d in detectors] if detectors else sorted(bundle_map.keys())

    extrinsics: dict[str, float] = {}
    for pair in extrinsic_pairs:
        if "=" not in pair:
            raise click.BadOptionUsage("--extrinsic", f"Extrinsic specification '{pair}' must be key=value.")
        key, value = pair.split("=", 1)
        try:
            extrinsics[key] = float(value)
        except ValueError as exc:
            raise click.BadOptionUsage("--extrinsic", f"Value '{value}' for '{key}' is not a float.") from exc

    result = run_bcr_posteriors(
        detectors=detector_list,
        outdir=outdir,
        bundle_paths=bundle_map,
        signal_model=signal_model,
        glitch_model=glitch_model,
        extrinsic_params=extrinsics if extrinsics else None,
        latent_sigma_signal=signal_latent_sigma,
        log_amp_sigma_signal=signal_log_amp_sigma,
        latent_sigma_glitch=glitch_latent_sigma,
        log_amp_sigma_glitch=glitch_log_amp_sigma,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        rng_seed=rng_seed,
        save_artifacts=save_artifacts,
        ci=ci,
        alpha=alpha,
        beta=beta,
    )

    summary_path = Path(outdir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    click.echo(f"BCR analysis complete. Summary written to {summary_path}")



if __name__ == "__main__":
    cli()
