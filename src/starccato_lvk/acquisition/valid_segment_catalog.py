"""Deterministically catalogue usable local GWOSC strain segments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import click
import numpy as np
import pandas as pd
from gwpy.timeseries import TimeSeries
from scipy.signal import welch

from .io.utils import _get_data_files_and_gps_times


@dataclass(frozen=True)
class CatalogSettings:
    analysis_duration: float = 4.0
    psd_duration: float = 64.0
    psd_gap: float = 0.5
    welch_duration: float = 4.0
    min_frequency: float = 20.0
    max_frequency: float = 1024.0
    max_median_log_psd_distance: float = 0.35
    max_window_log_psd_distance: float = 1.0
    max_abs_robust_z: float = 15.0
    glitch_guard: float = 1.0
    candidate_spacing: float = 128.0

    def required_interval(self, trigger: float) -> tuple[float, float]:
        half = self.analysis_duration / 2
        return trigger - half - self.psd_gap - self.psd_duration, trigger + half

    def psd_interval(self, trigger: float) -> tuple[float, float]:
        start, _ = self.required_interval(trigger)
        return start, trigger - self.analysis_duration / 2 - self.psd_gap


def _covering_paths(detector: str, start: float, end: float) -> list[str] | None:
    """Return local paths only when they continuously cover [start, end)."""
    try:
        files = _get_data_files_and_gps_times(detector).items()
    except FileNotFoundError:
        return None
    selected = sorted(
        (item for item in files if end > item[0] and start < item[0] + item[1][1])
    )
    cursor, paths = start, []
    for file_start, (path, duration) in selected:
        if file_start > cursor + 1e-6:
            return None
        if file_start + duration > cursor:
            paths.append(path)
            cursor = file_start + duration
        if cursor >= end:
            return paths
    return None


def stationarity_metrics(values: np.ndarray, rate: float, settings: CatalogSettings) -> dict[str, float] | None:
    """Compare fixed Welch spectra plus a robust transient statistic."""
    n, nwin = int(round(settings.psd_duration * rate)), int(round(settings.welch_duration * rate))
    if nwin < 8 or len(values) < n or n % nwin:
        return None
    data = values[:n]
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    robust_z = np.inf if mad == 0 else float(np.max(np.abs(data - median)) / (1.4826 * mad))
    psds = []
    for chunk in data.reshape(-1, nwin):
        frequencies, power = welch(chunk, fs=rate, nperseg=nwin, noverlap=0, detrend="constant")
        psds.append(power)
    psds = np.asarray(psds)
    band = (frequencies >= settings.min_frequency) & (frequencies <= min(settings.max_frequency, rate / 2))
    if not band.any() or np.any(psds[:, band] <= 0):
        return None
    logs = np.log(psds[:, band])
    distances = np.median(np.abs(logs - np.median(logs, axis=0)), axis=1)
    return {"median_log_psd_distance": float(np.median(distances)), "max_log_psd_distance": float(np.max(distances)), "max_abs_robust_z": robust_z}


def _validate(detector: str, trigger: float, settings: CatalogSettings) -> dict[str, object] | None:
    start, end = settings.required_interval(trigger)
    paths = _covering_paths(detector, start, end)
    if not paths:
        return None
    try:
        series = TimeSeries.read(paths, format="hdf5.gwosc", start=start, end=end)
    except (OSError, ValueError, KeyError):
        return None
    values, rate = np.asarray(series.value, dtype=float), float(series.sample_rate.value)
    expected = int(round((end - start) * rate))
    if len(values) < expected or not np.isfinite(values[:expected]).all():
        return None
    metrics = stationarity_metrics(values[:expected], rate, settings)
    if metrics is None or metrics["median_log_psd_distance"] > settings.max_median_log_psd_distance or metrics["max_log_psd_distance"] > settings.max_window_log_psd_distance or metrics["max_abs_robust_z"] > settings.max_abs_robust_z:
        return None
    psd_start, psd_end = settings.psd_interval(trigger)
    return {"paths": paths, "psd_start": psd_start, "psd_end": psd_end, **metrics}


def _overlaps(trigger: float, events: np.ndarray, settings: CatalogSettings) -> bool:
    start, end = settings.required_interval(trigger)
    return bool(np.any((events >= start - settings.glitch_guard) & (events <= end + settings.glitch_guard)))


def _noise_candidates(segments: Iterable[Sequence[float]], settings: CatalogSettings) -> Iterable[float]:
    half = settings.analysis_duration / 2
    for start, stop in sorted((float(a), float(b)) for a, b in segments):
        first, last = start + settings.psd_duration + settings.psd_gap + half, stop - half
        if last >= first:
            yield from np.arange(first, last + 1e-9, settings.candidate_spacing)


def build_catalog(*, blips: Mapping[str, pd.DataFrame], noise_segments: Iterable[Sequence[float]], settings: CatalogSettings = CatalogSettings(), max_per_category: int | None = None) -> pd.DataFrame:
    """Build all requested categories in a deterministic sort order."""
    events = {det: np.sort(df["event_time"].to_numpy(float)) for det, df in blips.items()}
    all_events = np.sort(np.concatenate(list(events.values())))
    settings_json = json.dumps(asdict(settings), sort_keys=True)
    settings_hash = hashlib.sha256(settings_json.encode()).hexdigest()
    rows: list[dict[str, object]] = []
    def add(category: str, trigger: float, glitch: str | None, data: Mapping[str, dict[str, object]]) -> None:
        if max_per_category is not None and sum(row["category"] == category for row in rows) >= max_per_category:
            return
        start, end = settings.required_interval(trigger)
        row: dict[str, object] = {"category": category, "trigger_gps": f"{trigger:.6f}", "glitch_detector": glitch or "", "required_start_gps": f"{start:.6f}", "required_end_gps": f"{end:.6f}", "settings_sha256": settings_hash, "settings_json": settings_json}
        for detector in ("H1", "L1"):
            item = data.get(detector)
            for key in ("paths", "psd_start", "psd_end", "median_log_psd_distance", "max_log_psd_distance", "max_abs_robust_z"):
                row[f"{detector}_{key}"] = json.dumps(item[key]) if item and key == "paths" else (item.get(key, "") if item else "")
        rows.append(row)
    for glitch in sorted(blips):
        other = "L1" if glitch == "H1" else "H1"
        for trigger in events[glitch]:
            trigger = float(trigger)
            own, clean = _validate(glitch, trigger, settings), _validate(other, trigger, settings)
            if own:
                add("glitch_single", trigger, glitch, {glitch: own})
            if own and clean and not _overlaps(trigger, events.get(other, np.array([])), settings):
                add("glitch_coincident_clean", trigger, glitch, {glitch: own, other: clean})
    for trigger in _noise_candidates(noise_segments, settings):
        if _overlaps(trigger, all_events, settings):
            continue
        h1, l1 = _validate("H1", trigger, settings), _validate("L1", trigger, settings)
        if h1:
            add("noise_single", trigger, None, {"H1": h1})
        if l1:
            add("noise_single", trigger, None, {"L1": l1})
        if h1 and l1:
            add("noise_coincident", trigger, None, {"H1": h1, "L1": l1})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["category", "trigger_gps", "glitch_detector"], kind="stable").reset_index(drop=True)


@click.command("build_valid_segment_catalog")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--h1-blips", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path(__file__).parent / "io/data/blip_H1.csv", show_default=True)
@click.option("--l1-blips", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path(__file__).parent / "io/data/blip.csv", show_default=True)
@click.option("--noise-segments", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path(__file__).parent / "io/data/only_noise_segments.txt", show_default=True)
@click.option("--max-per-category", type=click.IntRange(1), default=None)
@click.option("--candidate-spacing", type=click.FloatRange(min=1), default=128.0, show_default=True)
def cli(out: Path, h1_blips: Path, l1_blips: Path, noise_segments: Path, max_per_category: int | None, candidate_spacing: float) -> None:
    """Write a deterministic CSV of locally available stationary segments."""
    blips = {"H1": pd.read_csv(h1_blips), "L1": pd.read_csv(l1_blips)}
    if any("event_time" not in frame for frame in blips.values()):
        raise click.ClickException("blip catalogues require an event_time column")
    segments = np.loadtxt(noise_segments, comments="#", skiprows=1).reshape(-1, 2)
    catalog = build_catalog(blips=blips, noise_segments=segments, settings=CatalogSettings(candidate_spacing=candidate_spacing), max_per_category=max_per_category)
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out, index=False)
    click.echo(f"Wrote {len(catalog)} rows to {out}")
