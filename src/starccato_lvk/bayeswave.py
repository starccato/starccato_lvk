"""Run a BayesWave signal-versus-glitch baseline on a prepared event.

The input is a ``manifest.json`` written by ``studies/real_noise_event.py``.
Each invocation handles one event class so that expensive runs can be mapped
cleanly onto a SLURM array.  Without ``--execute`` the command is a read-only
dry run that prints the BayesWave and BayesWavePost commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

EVENT_CLASSES = ("noise", "inj_ccsn", "inj_glitch", "real_glitch")


@dataclass(frozen=True)
class DetectorInput:
    """Frame/cache information for one detector."""

    ifo: str
    bundle: Path
    frame: Path
    cache: Path
    channel: str
    t0: float
    source_dt: float
    source_samples: int
    sample_rate: float

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def duration(self) -> float:
        return self.source_samples * self.source_dt

    @property
    def samples(self) -> int:
        return int(round(self.duration * self.sample_rate))


@dataclass(frozen=True)
class RunSettings:
    """BayesWave sampling settings recorded with every result."""

    iterations: int = 1_000_000
    burnin: int = 100_000
    chains: int = 20
    threads: int = 4
    seed: int = 1234
    checkpoint_interval_hours: float = 1.0
    sample_rate: float = 2048.0

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.burnin < 0 or self.burnin >= self.iterations:
            raise ValueError("burnin must satisfy 0 <= burnin < iterations")
        if self.chains <= 0 or self.threads <= 0:
            raise ValueError("chains and threads must be positive")
        if self.chains % self.threads:
            raise ValueError("threads must divide chains")
        if self.checkpoint_interval_hours <= 0:
            raise ValueError("checkpoint interval must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample rate must be positive")


def load_event_manifest(path: Path) -> dict:
    """Load and minimally validate a real-noise event manifest."""

    path = Path(path)
    manifest = json.loads(path.read_text())
    required = {"index", "detectors", "band", "snr", "bundles"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"manifest is missing keys: {sorted(missing)}")
    if len(manifest["band"]) != 2:
        raise ValueError("manifest band must contain [flow, fmax]")
    detectors = [str(ifo).upper() for ifo in manifest["detectors"]]
    if len(detectors) < 2:
        raise ValueError(
            "BayesWave signal-versus-glitch comparison requires at least two "
            "detectors; use an H1-L1 manifest"
        )
    manifest["detectors"] = detectors
    return manifest


def _resolve_bundle(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_working_directory = path.resolve()
    if from_working_directory.is_file():
        return from_working_directory
    return (manifest_path.parent / path).resolve()


def _bundle_header(path: Path) -> tuple[float, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"analysis bundle not found: {path}")
    with h5py.File(path, "r") as h5:
        if "strain/values" not in h5:
            raise ValueError(f"bundle has no strain/values dataset: {path}")
        group = h5["strain"]
        values = group["values"]
        try:
            t0 = float(group.attrs["t0"])
            dt = float(group.attrs["dt"])
        except KeyError as exc:
            raise ValueError(
                f"bundle is missing strain timing metadata: {path}"
            ) from exc
        count = int(values.shape[0])
    if not math.isfinite(t0) or not math.isfinite(dt) or dt <= 0 or count <= 0:
        raise ValueError(f"bundle has invalid strain timing or shape: {path}")
    return t0, dt, count


def detector_inputs(
    manifest: Mapping,
    manifest_path: Path,
    event_class: str,
    output_dir: Path,
    sample_rate: float = 2048.0,
) -> list[DetectorInput]:
    """Resolve bundle paths and the deterministic frame/cache filenames."""

    if event_class not in EVENT_CLASSES:
        raise ValueError(f"unknown event class: {event_class}")
    if event_class not in manifest["bundles"]:
        raise ValueError(f"manifest has no bundles for class {event_class}")

    inputs: list[DetectorInput] = []
    reference: tuple[float, float, int] | None = None
    index = int(manifest["index"])
    output_dir = Path(output_dir).resolve()
    for ifo in manifest["detectors"]:
        try:
            raw_path = manifest["bundles"][event_class][ifo]
        except KeyError as exc:
            raise ValueError(
                f"manifest has no {event_class}/{ifo} bundle"
            ) from exc
        bundle = _resolve_bundle(Path(manifest_path).resolve(), raw_path)
        t0, source_dt, source_count = _bundle_header(bundle)
        header = (t0, source_dt, source_count)
        if reference is None:
            reference = header
        elif not (
            np.isclose(t0, reference[0], rtol=0.0, atol=1e-8)
            and np.isclose(source_dt, reference[1], rtol=0.0, atol=1e-12)
            and source_count == reference[2]
        ):
            raise ValueError(
                "detector bundles do not share the same time grid"
            )

        source_rate = 1.0 / source_dt
        if sample_rate > source_rate and not np.isclose(
            sample_rate, source_rate
        ):
            raise ValueError(
                "BayesWave sample rate cannot exceed the bundle sample rate"
            )
        duration = source_count * source_dt
        output_samples = duration * sample_rate
        if not np.isclose(output_samples, round(output_samples), atol=1e-8):
            raise ValueError(
                "duration times BayesWave sample rate must be an integer"
            )
        if int(round(output_samples)) & (int(round(output_samples)) - 1):
            raise ValueError(
                "BayesLine requires a power-of-two sample count; "
                "choose a compatible sample rate"
            )

        frame_start = math.floor(t0)
        frame_end = math.ceil(t0 + duration)
        frame_duration = frame_end - frame_start
        tag = f"STARCCATO_E{index}_{event_class.upper()}_SR{int(sample_rate)}"
        frame = (
            output_dir
            / "frames"
            / (f"{ifo[0]}-{tag}-{frame_start}-{frame_duration}.gwf")
        )
        cache = output_dir / "frames" / f"{ifo}.cache"
        inputs.append(
            DetectorInput(
                ifo=ifo,
                bundle=bundle,
                frame=frame,
                cache=cache,
                channel=f"{ifo}:STARCCATO_STRAIN",
                t0=t0,
                source_dt=source_dt,
                source_samples=source_count,
                sample_rate=sample_rate,
            )
        )
    return inputs


def prepare_frames(
    inputs: Iterable[DetectorInput], *, overwrite: bool = False
) -> None:
    """Write GWF files and LAL cache files for BayesWave."""

    from gwpy.timeseries import TimeSeries

    for item in inputs:
        item.frame.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not item.frame.is_file():
            with h5py.File(item.bundle, "r") as h5:
                values = np.asarray(h5["strain/values"], dtype=float)
            if values.size != item.source_samples or not np.all(
                np.isfinite(values)
            ):
                raise ValueError(
                    f"bundle strain is incomplete or non-finite: {item.bundle}"
                )
            series = TimeSeries(
                values,
                t0=item.t0,
                dt=item.source_dt,
                unit="strain",
                channel=item.channel,
            )
            source_rate = 1.0 / item.source_dt
            if not np.isclose(source_rate, item.sample_rate):
                series = series.resample(item.sample_rate)
            if series.size != item.samples:
                sizes = f"{series.size} samples, expected {item.samples}"
                raise ValueError(f"resampled frame has {sizes}")
            series.write(item.frame, format="gwf")

        cache_start = math.floor(item.t0)
        cache_end = math.ceil(item.t0 + item.duration)
        cache_duration = cache_end - cache_start
        description = item.frame.name.split("-", 2)[1]
        item.cache.write_text(
            f"{item.ifo[0]} {description} {cache_start} {cache_duration} "
            f"file://localhost{item.frame.resolve()}\n"
        )


def bayeswave_command(
    executable: str,
    inputs: Sequence[DetectorInput],
    output_dir: Path,
    flow: float,
    fmax: float,
    sky: Mapping[str, float] | None,
    settings: RunSettings,
) -> list[str]:
    """Construct the scientific BayesWave signal/glitch command."""

    settings.validate()
    first = inputs[0]
    command = [executable]
    for item in inputs:
        command.extend(["--ifo", item.ifo])
    for item in inputs:
        command.extend(
            [
                f"--{item.ifo}-flow",
                str(flow),
                f"--{item.ifo}-fhigh",
                str(item.sample_rate / 2.0),
                f"--{item.ifo}-cache",
                str(item.cache.resolve()),
                f"--{item.ifo}-channel",
                item.channel,
            ]
        )
    command.extend(
        [
            "--trigtime",
            str(first.t0 + first.duration / 2.0),
            "--segment-start",
            str(first.t0),
            "--srate",
            str(round(first.sample_rate)),
            "--seglen",
            str(first.duration),
            "--psdstart",
            str(first.t0),
            "--psdlength",
            str(first.duration),
            "--Niter",
            str(settings.iterations),
            "--Nburnin",
            str(settings.burnin),
            "--Nchain",
            str(settings.chains),
            "--Nthreads",
            str(settings.threads),
            "--chainseed",
            str(settings.seed),
            "--outputDir",
            str(Path(output_dir).resolve()),
            "--bayesLine",
            "--waveletFmin",
            str(flow),
            "--waveletFmax",
            str(fmax),
            "--checkpoint",
            "--checkpointIntervalHrs",
            str(settings.checkpoint_interval_hours),
        ]
    )
    if sky is not None and "ra" in sky and "dec" in sky:
        command.extend(
            [
                "--fixSky",
                "--fixRA",
                str(float(sky["ra"])),
                "--fixDEC",
                str(float(sky["dec"])),
            ]
        )
    return command


def bayeswave_post_command(
    executable: str,
    inputs: Sequence[DetectorInput],
    output_dir: Path,
    flow: float,
    fmax: float,
    settings: RunSettings,
) -> list[str]:
    """Construct BayesWavePost command using BayesWave's fair-draw ASDs."""

    first = inputs[0]
    command = [executable]
    for item in inputs:
        command.extend(["--ifo", item.ifo])
    for item in inputs:
        asd = Path(output_dir).resolve() / f"{item.ifo}_fairdraw_asd.dat"
        command.extend(
            [
                f"--{item.ifo}-flow",
                str(flow),
                f"--{item.ifo}-fhigh",
                str(item.sample_rate / 2.0),
                f"--{item.ifo}-cache",
                f"interp:{asd}",
            ]
        )
    command.extend(
        [
            "--trigtime",
            str(first.t0 + first.duration / 2.0),
            "--segment-start",
            str(first.t0),
            "--srate",
            str(round(first.sample_rate)),
            "--seglen",
            str(first.duration),
            "--psdstart",
            str(first.t0),
            "--psdlength",
            str(first.duration),
            "--dataseed",
            str(settings.seed),
            "--outputDir",
            str(Path(output_dir).resolve()),
            "--bayesLine",
            "--0noise",
        ]
    )
    return command


def parse_evidence(path: Path) -> dict[str, tuple[float, float]]:
    """Parse ``evidence.dat`` into model -> (logZ, uncertainty)."""

    evidence: dict[str, tuple[float, float]] = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            evidence[fields[0]] = (float(fields[1]), float(fields[2]))
        except ValueError:
            continue
    required = {"signal", "glitch", "noise"}
    missing = required.difference(evidence)
    if missing:
        raise ValueError(f"evidence file is missing models: {sorted(missing)}")
    return evidence


def _stats_row(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    header: list[str] | None = None
    values: list[float] | None = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped[1:].split()
        elif values is None:
            values = [float(value) for value in stripped.split()]
    if header is None or values is None:
        return None
    return dict(zip(header, values))


def collect_result(
    manifest: Mapping,
    manifest_path: Path,
    event_class: str,
    output_dir: Path,
    settings: RunSettings,
    elapsed_seconds: float | None = None,
) -> dict:
    """Collect evidences and SNR into one machine-readable row."""

    output_dir = Path(output_dir).resolve()
    evidence = parse_evidence(output_dir / "evidence.dat")
    signal_logz, signal_unc = evidence["signal"]
    glitch_logz, glitch_unc = evidence["glitch"]
    noise_logz, noise_unc = evidence["noise"]
    signal_stats = _stats_row(output_dir / "post/signal/signal_stats.dat.geo")
    if elapsed_seconds is None:
        metadata_path = output_dir / "run_metadata.json"
        if metadata_path.is_file():
            elapsed_seconds = json.loads(metadata_path.read_text()).get(
                "elapsed_seconds"
            )
    return {
        "index": int(manifest["index"]),
        "cls": event_class,
        "detectors": list(manifest["detectors"]),
        "target_snr": float(manifest["snr"][event_class]),
        "logZ_signal": signal_logz,
        "logZ_glitch": glitch_logz,
        "logZ_noise": noise_logz,
        "log_bayeswave_signal_glitch": signal_logz - glitch_logz,
        "log_bayeswave_signal_glitch_uncertainty": math.hypot(
            signal_unc, glitch_unc
        ),
        "signal_reconstructed_snr_median": (
            signal_stats.get("snr") if signal_stats is not None else None
        ),
        "evidence_uncertainty": {
            "signal": signal_unc,
            "glitch": glitch_unc,
            "noise": noise_unc,
        },
        "elapsed_seconds": elapsed_seconds,
        "manifest": str(Path(manifest_path).resolve()),
        "output_dir": str(output_dir),
        "settings": asdict(settings),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_executable(value: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = str(candidate.resolve())
    if resolved is None:
        raise FileNotFoundError(f"executable not found: {value}")
    return resolved


def _run(command: Sequence[str], cwd: Path, log_name: str) -> None:
    log_path = cwd / log_name
    with log_path.open("a") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        subprocess.run(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--class", dest="event_class", choices=EVENT_CLASSES, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bayeswave-executable", default="BayesWave")
    parser.add_argument("--post-executable", default="BayesWavePost")
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--burnin", type=int, default=100_000)
    parser.add_argument("--chains", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=2048.0,
        help="BayesWave frame rate; 2048 Hz gives a 1024 Hz Nyquist cutoff",
    )
    parser.add_argument("--checkpoint-interval-hours", type=float, default=1.0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write frame/cache inputs but do not run BayesWave",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="parse an already completed output directory",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write inputs and execute BayesWave plus BayesWavePost",
    )
    parser.add_argument("--skip-post", action="store_true")
    parser.add_argument("--overwrite-frames", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = int(args.prepare_only) + int(args.collect_only) + int(args.execute)
    if modes > 1:
        raise ValueError(
            "choose only one of --prepare-only, --collect-only, --execute"
        )

    settings = RunSettings(
        iterations=args.iterations,
        burnin=args.burnin,
        chains=args.chains,
        threads=args.threads,
        seed=args.seed,
        checkpoint_interval_hours=args.checkpoint_interval_hours,
        sample_rate=args.sample_rate,
    )
    settings.validate()
    manifest_path = args.manifest.resolve()
    manifest = load_event_manifest(manifest_path)
    output_dir = args.output.resolve()
    inputs = detector_inputs(
        manifest,
        manifest_path,
        args.event_class,
        output_dir,
        sample_rate=settings.sample_rate,
    )
    flow, fmax = (float(value) for value in manifest["band"])
    if flow <= 0 or fmax <= flow:
        raise ValueError("manifest band must satisfy 0 < flow < fmax")
    if fmax >= settings.sample_rate / 2.0:
        raise ValueError(
            "manifest fmax must be below the BayesWave Nyquist frequency"
        )

    bayeswave_value = args.bayeswave_executable
    post_value = args.post_executable
    if args.execute:
        bayeswave_value = _resolve_executable(bayeswave_value)
        post_value = _resolve_executable(post_value)
    run_command = bayeswave_command(
        bayeswave_value,
        inputs,
        output_dir,
        flow,
        fmax,
        manifest.get("sky"),
        settings,
    )
    post_command = bayeswave_post_command(
        post_value, inputs, output_dir, flow, fmax, settings
    )

    print(f"BayesWave:     {shlex.join(run_command)}", flush=True)
    print(f"BayesWavePost: {shlex.join(post_command)}", flush=True)
    if not modes:
        print("dry run only; pass --prepare-only or --execute", flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.collect_only:
        result = collect_result(
            manifest, manifest_path, args.event_class, output_dir, settings
        )
        _write_json(output_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0

    prepare_frames(inputs, overwrite=args.overwrite_frames)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "event_class": args.event_class,
        "detectors": list(manifest["detectors"]),
        "band": [flow, fmax],
        "inputs": [
            {
                **asdict(item),
                "bundle": str(item.bundle),
                "frame": str(item.frame),
                "cache": str(item.cache),
            }
            for item in inputs
        ],
        "settings": asdict(settings),
        "bayeswave_command": run_command,
        "bayeswave_post_command": post_command,
    }
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text())
        for key in (
            "sampling_elapsed_seconds",
            "post_elapsed_seconds",
            "elapsed_seconds",
        ):
            if key in previous:
                metadata[key] = previous[key]
    _write_json(metadata_path, metadata)
    if args.prepare_only:
        print(f"prepared inputs under {output_dir}", flush=True)
        return 0

    evidence_path = output_dir / "evidence.dat"
    sampling_elapsed = 0.0
    post_elapsed = 0.0
    if not evidence_path.is_file():
        started = time.perf_counter()
        _run(run_command, output_dir, "bayeswave.log")
        sampling_elapsed = time.perf_counter() - started
        metadata["sampling_elapsed_seconds"] = sampling_elapsed
        _write_json(metadata_path, metadata)
    else:
        sampling_elapsed = float(metadata.get("sampling_elapsed_seconds", 0.0))
    if not evidence_path.is_file():
        raise RuntimeError(f"BayesWave did not create {evidence_path}")
    if not args.skip_post:
        signal_stats = output_dir / "post/signal/signal_stats.dat.geo"
        if not signal_stats.is_file():
            started = time.perf_counter()
            _run(post_command, output_dir, "bayeswave_post.log")
            post_elapsed = time.perf_counter() - started
        else:
            post_elapsed = float(metadata.get("post_elapsed_seconds", 0.0))
        if not signal_stats.is_file():
            raise RuntimeError(
                "BayesWavePost exited without creating "
                f"{signal_stats}; inspect bayeswave_post.log"
            )
    elapsed = sampling_elapsed + post_elapsed
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["sampling_elapsed_seconds"] = sampling_elapsed
    metadata["post_elapsed_seconds"] = post_elapsed
    metadata["elapsed_seconds"] = elapsed
    _write_json(metadata_path, metadata)
    result = collect_result(
        manifest,
        manifest_path,
        args.event_class,
        output_dir,
        settings,
        elapsed_seconds=elapsed,
    )
    _write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
