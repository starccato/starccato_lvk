from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import h5py
import numpy as np
import yaml
import pandas as pd
from astropy.time import Time
import click

from starccato_jax.waveforms import get_model

from ..acquisition.io.strain_loader import strain_loader
from ..analysis import run_bcr_posteriors
from ..analysis.jim_waveform import StarccatoJimWaveform
from ..acquisition.io.glitch_catalog import get_blip_trigger_time

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_SLURM_ROOT = _PACKAGE_ROOT / "slurm"
if not _SLURM_ROOT.exists():  # pragma: no cover - fallback for non-repo installs
    _SLURM_ROOT = Path("slurm")

CONFIG_DEFAULT = _SLURM_ROOT / "configs/analysis.yaml"
EVENTS_DIR_DEFAULT = _SLURM_ROOT / "configs"
TRIGGERS_CSV_DEFAULT = _PACKAGE_ROOT / "src/starccato_lvk/acquisition/io/data/triggers.csv"


@dataclass
class AnalysisConfig:
    detectors: List[str]
    flow: float
    fmax: float
    roll_off: float
    num_warmup: int
    num_samples: int
    num_chains: int
    signal_latent_sigma: float
    signal_log_amp_sigma: float
    glitch_latent_sigma: float
    glitch_log_amp_sigma: float
    signal_model: str
    glitch_model: str
    alpha: float
    beta: float
    save_artifacts: bool
    ci: tuple[int, int]
    output_root: Path


def load_analysis_config(path: Path) -> AnalysisConfig:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    detectors = [det.upper() for det in cfg["detectors"]]
    analysis_cfg = cfg["analysis"]
    sampler_cfg = cfg["sampler"]
    priors_sig = cfg["priors"]["signal"]
    priors_glitch = cfg["priors"]["glitch"]
    models = cfg["models"]

    return AnalysisConfig(
        detectors=detectors,
        flow=float(analysis_cfg["flow"]),
        fmax=float(analysis_cfg["fmax"]),
        roll_off=float(cfg["data"]["roll_off"]),
        num_warmup=int(sampler_cfg["num_warmup"]),
        num_samples=int(sampler_cfg["num_samples"]),
        num_chains=int(sampler_cfg["num_chains"]),
        signal_latent_sigma=float(priors_sig["latent_sigma"]),
        signal_log_amp_sigma=float(priors_sig["log_amp_sigma"]),
        glitch_latent_sigma=float(priors_glitch["latent_sigma"]),
        glitch_log_amp_sigma=float(priors_glitch["log_amp_sigma"]),
        signal_model=str(models["signal"]),
        glitch_model=str(models["glitch"]),
        alpha=float(analysis_cfg["alpha"]),
        beta=float(analysis_cfg["beta"]),
        save_artifacts=bool(analysis_cfg["save_artifacts"]),
        ci=tuple(int(x) for x in analysis_cfg["ci"]),
        output_root=Path(cfg["output_root"]).resolve(),
    )


def generate_event_lists(
    outdir: Path,
    *,
    blip_gps: Sequence[float],
    noise_gps: Sequence[float],
    noise_inj_gps: Optional[Sequence[float]] = None,
) -> None:
    """Persist GPS trigger lists for each scenario into text files."""
    outdir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, values: Iterable[float]) -> None:
        if not values:
            raise ValueError(f"No GPS times provided for '{name}'.")
        path = outdir / f"events_{name}.txt"
        path.write_text("\n".join(str(v) for v in values) + "\n")

    _write("blip", blip_gps)
    _write("noise", noise_gps)

    if noise_inj_gps is None or len(noise_inj_gps) == 0:
        noise_inj_gps = noise_gps
    _write("noise_inj", noise_inj_gps)


def prepare_event_lists_from_files(
    outdir: Path,
    *,
    blip_count: int,
    noise_file: Path,
    noise_inj_file: Optional[Path] = None,
) -> None:
    def _read_list(path: Path) -> list[float]:
        with path.open("r") as f:
            values = [float(line.strip()) for line in f if line.strip()]
        if not values:
            raise ValueError(f"No GPS entries found in {path}.")
        return values

    blip_times = [get_blip_trigger_time(i) for i in range(blip_count)]
    noise_times = _read_list(noise_file)
    noise_inj_times = _read_list(noise_inj_file) if noise_inj_file else None

    generate_event_lists(
        outdir,
        blip_gps=blip_times,
        noise_gps=noise_times,
        noise_inj_gps=noise_inj_times,
    )


def read_event_list(scenario: str, index: int, *, root: Path = EVENTS_DIR_DEFAULT) -> float:
    list_path = root / f"events_{scenario}.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Event list {list_path} missing.")
    with list_path.open("r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if index < 0 or index >= len(lines):
        raise IndexError(f"Index {index} out of range for {scenario} list (len={len(lines)}).")
    return float(lines[index])


def read_event_from_triggers_csv(
    scenario: str, index: int, *, csv_path: Path
) -> float:
    if not csv_path.exists():
        raise FileNotFoundError(f"Triggers CSV {csv_path} missing.")
    df = pd.read_csv(csv_path)
    required_cols = {"noise_trigger", "blip_trigger"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Triggers CSV must contain columns {required_cols}, found {set(df.columns)}"
        )
    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of range for triggers CSV (len={len(df)}).")
    if scenario == "blip":
        return float(df.loc[index, "blip_trigger"])
    else:  # "noise" or "noise_inj"
        return float(df.loc[index, "noise_trigger"])


def prepare_bundles(detectors: Sequence[str], gps: float, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_map: Dict[str, Path] = {}
    for det in detectors:
        det_name = det.upper()
        det_dir = output_dir / det_name
        det_dir.mkdir(parents=True, exist_ok=True)
        bundles = list(det_dir.glob("analysis_bundle_*.hdf5"))
        if bundles:
            bundle_map[det_name] = bundles[0]
            continue
        strain_loader(trigger_time=gps, outdir=det_dir, detector=det_name)
        bundles = list(det_dir.glob("analysis_bundle_*.hdf5"))
        if not bundles:
            raise FileNotFoundError(f"No bundle found for {det_name} at gps {gps}")
        bundle_map[det_name] = bundles[0]
    return bundle_map


def _find_existing_noise_bundles(
    detectors: Sequence[str], gps: float, output_root: Path
) -> Optional[Dict[str, Path]]:
    noise_dir = output_root / "noise" / f"{int(gps)}" / "bundles"
    if not noise_dir.exists():
        return None
    bundle_map: Dict[str, Path] = {}
    for det in detectors:
        det_dir = noise_dir / det.upper()
        if not det_dir.exists():
            return None
        bundles = list(det_dir.glob("analysis_bundle_*.hdf5"))
        if not bundles:
            return None
        bundle_map[det.upper()] = bundles[0]
    return bundle_map if bundle_map else None


def inject_signal(
    bundle_paths: Mapping[str, Path],
    gps: float,
    cfg: AnalysisConfig,
    outdir: Path,
    *,
    distance_scale: float = 1.0,
) -> Dict[str, Path]:
    model = get_model(cfg.signal_model)
    outdir.mkdir(parents=True, exist_ok=True)

    injected: Dict[str, Path] = {}
    for det, bundle in bundle_paths.items():
        dest = outdir / f"{bundle.stem}_inj.hdf5"
        if dest.exists():
            injected[det] = dest
            continue

        with h5py.File(bundle, "r") as src:
            strain_vals = np.array(src["strain"]["values"])
            dt = float(src["strain"].attrs["dt"])
            attrs = dict(src["strain"].attrs)

        sample_rate = 1.0 / dt
        waveform = StarccatoJimWaveform(model=model, sample_rate=sample_rate)
        params = {name: 0.0 for name in waveform.latent_names}
        params["log_amp"] = -np.log(distance_scale)
        gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad
        params.update(
            {
                "t_c": 0.0,
                "ra": 0.0,
                "dec": 0.0,
                "psi": 0.0,
                "luminosity_distance": distance_scale,
                "gmst": float(gmst),
                "trigger_time": gps,
            }
        )

        wf_td = waveform.time_domain_waveform_numpy(params)
        if len(wf_td) > len(strain_vals):
            wf_td = wf_td[: len(strain_vals)]
        pad_left = (len(strain_vals) - len(wf_td)) // 2
        pad_right = len(strain_vals) - len(wf_td) - pad_left
        wf_td = np.pad(wf_td, (pad_left, pad_right))
        injected_strain = strain_vals + wf_td

        with h5py.File(dest, "w") as dst:
            with h5py.File(bundle, "r") as src:
                src.copy("psd", dst)
                if "full_strain" in src:
                    src.copy("full_strain", dst)
            strain_grp = dst.create_group("strain")
            strain_grp.create_dataset("values", data=injected_strain)
            for key, value in attrs.items():
                strain_grp.attrs[key] = value

        injected[det] = dest
    return injected


def run_event_workflow(
    cfg: AnalysisConfig,
    scenario: str,
    gps: float,
    idx: int,
    *,
    force: bool = False,
    injection_distance: float = 1.0,
    stage: str = "both",
) -> Dict[str, Dict[str, float]] | None:
    scenario_dir = cfg.output_root / scenario / f"{int(gps)}"
    bundles_dir = scenario_dir / "bundles"
    bundle_paths: Optional[Dict[str, Path]] = None

    if stage in ("prep", "both"):
        if scenario == "noise_inj":
            # Prefer reusing existing noise bundles to avoid duplicate downloads
            bundle_paths = _find_existing_noise_bundles(cfg.detectors, gps, cfg.output_root)
            if bundle_paths is None:
                bundle_paths = prepare_bundles(cfg.detectors, gps, bundles_dir)
            injected_dir = scenario_dir / "bundles_injected"
            bundle_paths = inject_signal(bundle_paths, gps, cfg, injected_dir, distance_scale=injection_distance)
        else:
            bundle_paths = prepare_bundles(cfg.detectors, gps, bundles_dir)

    if stage in ("analysis", "both"):
        if bundle_paths is None:
            if scenario == "noise_inj":
                bundle_paths = _find_existing_noise_bundles(cfg.detectors, gps, cfg.output_root)
                if bundle_paths is None:
                    bundle_paths = prepare_bundles(cfg.detectors, gps, bundles_dir)
                injected_dir = scenario_dir / "bundles_injected"
                bundle_paths = inject_signal(bundle_paths, gps, cfg, injected_dir, distance_scale=injection_distance)
            else:
                bundle_paths = prepare_bundles(cfg.detectors, gps, bundles_dir)

        outdir = scenario_dir / "analysis"
        outdir.mkdir(parents=True, exist_ok=True)
        summary_path = outdir / "summary.json"
        if not force and summary_path.exists():
            return None

        result = run_bcr_posteriors(
            detectors=cfg.detectors,
            outdir=str(outdir),
            bundle_paths={det: str(path) for det, path in bundle_paths.items()},
            signal_model=cfg.signal_model,
            glitch_model=cfg.glitch_model,
            extrinsic_params=None,
            latent_sigma_signal=cfg.signal_latent_sigma,
            log_amp_sigma_signal=cfg.signal_log_amp_sigma,
            latent_sigma_glitch=cfg.glitch_latent_sigma,
            log_amp_sigma_glitch=cfg.glitch_log_amp_sigma,
            num_samples=cfg.num_samples,
            num_warmup=cfg.num_warmup,
            num_chains=cfg.num_chains,
            rng_seed=idx,
            save_artifacts=cfg.save_artifacts,
            ci=cfg.ci,
            alpha=cfg.alpha,
            beta=cfg.beta,
        )
        with summary_path.open("w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        return result

    return None


@click.command(name="starccato_lvk_run_event")
@click.option("--scenario", type=click.Choice(["blip", "noise", "noise_inj"]), required=True)
@click.option("--index", "event_index", type=int, required=True, help="Event index (e.g. SLURM_ARRAY_TASK_ID).")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=CONFIG_DEFAULT,
    show_default=True,
    help="Analysis configuration YAML.",
)
@click.option(
    "--events-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=EVENTS_DIR_DEFAULT,
    show_default=True,
    help="Directory containing events_<scenario>.txt files.",
)
@click.option(
    "--triggers-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional CSV with paired triggers (noise_trigger, blip_trigger). Overrides events-dir if given.",
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
def cli_run_event(
    scenario: str,
    event_index: int,
    config: Path,
    events_dir: Path,
    triggers_csv: Optional[Path],
    force: bool,
    distance: float,
    stage: str,
) -> None:
    """Run Starccato LVK analysis for a single event."""
    cfg = load_analysis_config(config)
    # Resolve GPS from triggers CSV if provided or available at default path
    if triggers_csv is None and TRIGGERS_CSV_DEFAULT.exists():
        triggers_csv = TRIGGERS_CSV_DEFAULT
    if triggers_csv is not None:
        gps = read_event_from_triggers_csv(scenario, event_index, csv_path=triggers_csv)
    else:
        gps = read_event_list(scenario, event_index, root=events_dir)
    result = run_event_workflow(
        cfg,
        scenario,
        gps,
        event_index,
        force=force,
        injection_distance=distance,
        stage=stage,
    )
    if result is None:
        click.echo("Workflow complete: summary already existed; no new analysis performed.")
    else:
        click.echo(f"Workflow complete: BCR={result.get('bcr')}")


@click.command(name="starccato_lvk_generate_events")
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=EVENTS_DIR_DEFAULT,
    show_default=True,
    help="Directory to write scenario event lists.",
)
@click.option("--blip-count", type=int, default=200, show_default=True, help="Number of blip glitches to record.")
@click.option(
    "--noise-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Text file containing noise GPS times (one per line).",
)
@click.option(
    "--noise-inj-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional text file with noise+injection GPS times (defaults to noise file).",
)
def cli_generate_events(outdir: Path, blip_count: int, noise_file: Path, noise_inj_file: Optional[Path]) -> None:
    """Generate event lists used by the workflow CLI."""
    prepare_event_lists_from_files(
        outdir,
        blip_count=blip_count,
        noise_file=noise_file,
        noise_inj_file=noise_inj_file,
    )
    click.echo(f"Wrote event lists to {outdir}")


def generate_events_main(argv: Optional[Sequence[str]] = None) -> None:
    cli_generate_events.main(args=list(argv) if argv is not None else None, prog_name="starccato_lvk_generate_events")


def main(argv: Optional[Sequence[str]] = None) -> None:
    cli_run_event.main(args=list(argv) if argv is not None else None, prog_name="starccato_lvk_run_event")


if __name__ == "__main__":  # pragma: no cover - module execution helper
    main()
