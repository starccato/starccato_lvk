from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from .io.only_noise_data import load_only_noise_segments
from .io.glitch_catalog import load_blip_glitch_catalog, DURATION as CCSN_DURATION
from .io.strain_loader import _detector_cat3_online

HERE = Path(__file__).parent
DATA_DIR = HERE / "io/data"


def _sample_noise_triggers(
    n: int,
    *,
    margin: float = 1.0,
    seed: int = 0,
    require_both_detectors: bool = True,
    detectors: tuple[str, str] = ("H1", "L1"),
    verbose: bool = False,
    log_every: int = 50,
) -> pd.DataFrame:
    """Sample `n` trigger times uniformly within valid noise segments.

    Returns a DataFrame with columns: event_time, segment_start, segment_stop
    """
    segments = load_only_noise_segments()
    if segments.size == 0:
        raise RuntimeError("No CAT3-valid noise segments available.")

    rng = np.random.default_rng(seed)
    out_times: list[float] = []
    seg_starts: list[float] = []
    seg_stops: list[float] = []

    i = 0
    max_attempts = n * 50 if require_both_detectors else n * 10
    last_log_accept = 0
    last_log_attempt = 0
    while len(out_times) < n and i < max_attempts:  # hard cap to avoid infinite loops
        s, e = segments[i % len(segments)]
        i += 1
        if e - s <= 2 * margin:
            continue
        t = rng.uniform(s + margin, e - margin)
        if require_both_detectors:
            # Match the loader window: [t-66, t+1]
            gps_start = float(t) - 66.0
            gps_end = float(t) + 1.0
            ok = all(_dq_online_cached(det, gps_start, gps_end) for det in detectors)
            if not ok:
                continue
        out_times.append(float(t))
        seg_starts.append(float(s))
        seg_stops.append(float(e))

        if verbose and (len(out_times) - last_log_accept) >= log_every:
            click.echo(
                f"[noise] accepted={len(out_times)}/{n} attempts={i}/{max_attempts}",
                err=True,
            )
            last_log_accept = len(out_times)
        if verbose and (i - last_log_attempt) >= (log_every * 5):
            click.echo(
                f"[noise] progress: attempts={i}/{max_attempts} accepted={len(out_times)}",
                err=True,
            )
            last_log_attempt = i

    if len(out_times) < n:
        click.echo(
            f"Warning: only produced {len(out_times)} noise triggers (requested {n}).",
            err=True,
        )

    return pd.DataFrame(
        {
            "event_time": out_times,
            "segment_start": seg_starts,
            "segment_stop": seg_stops,
        }
    )


def _select_blip_triggers(
    max_count: int,
    *,
    around_seconds: float,
    tol_frac: float = 0.5,
    require_both_detectors: bool = True,
    detectors: tuple[str, str] = ("H1", "L1"),
    verbose: bool = False,
    log_every: int = 50,
) -> pd.DataFrame:
    """Select up to `max_count` blip glitches with durations ~ CCSN length.

    Returns a DataFrame with columns: event_time, duration, snr
    """
    df = load_blip_glitch_catalog()
    if not {"event_time", "duration", "snr"}.issubset(df.columns):
        raise RuntimeError("blip glitch catalog missing required columns.")

    lo = max(0.0, around_seconds * (1.0 - tol_frac))
    hi = around_seconds * (1.0 + tol_frac)
    sel = df[(df["duration"] >= lo) & (df["duration"] <= hi)].copy()
    if require_both_detectors:
        # Incrementally test highest-SNR glitches until enough pass DQ
        passed_rows = []
        checked = 0
        for _, row in sel.iterrows():
            if len(passed_rows) >= max_count:
                break
            t = float(row["event_time"])  # peak/center time
            gps_start, gps_end = t - 66.0, t + 1.0
            ok = all(_dq_online_cached(det, gps_start, gps_end) for det in detectors)
            if ok:
                passed_rows.append(row)
            checked += 1
            if verbose and ((len(passed_rows) and len(passed_rows) % log_every == 0) or (checked % (log_every * 5) == 0)):
                click.echo(
                    f"[blip] accepted={len(passed_rows)}/{max_count} checked={checked}",
                    err=True,
                )
        if not passed_rows:
            sel = sel.iloc[0:0]
        else:
            sel = pd.DataFrame(passed_rows)
    # prefer higher SNR entries
    sel.sort_values(by=["snr"], ascending=False, inplace=True)
    if len(sel) > max_count:
        sel = sel.iloc[:max_count]
    if sel.empty:
        raise RuntimeError(
            f"No blip glitches within duration window [{lo:.3f}, {hi:.3f}] s."
        )
    return sel[["event_time", "duration", "snr"]].reset_index(drop=True)


@click.command("generate_trigger_csv")
@click.option(
    "--outdir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=DATA_DIR,
    show_default=True,
    help="Directory to write CSV files.",
)
@click.option("--noise-count", type=int, default=1000, show_default=True)
@click.option("--blip-count", type=int, default=1000, show_default=True)
@click.option(
    "--blip-tol",
    type=float,
    default=0.5,
    show_default=True,
    help="Fractional tolerance around CCSN duration for blip selection.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--both-detectors/--single-detector",
    default=True,
    show_default=True,
    help="For noise triggers, require CAT3-online for both H1 and L1.",
)
@click.option("--verbose/--quiet", default=False, show_default=True, help="Print progress logs during selection.")
@click.option("--log-every", type=int, default=50, show_default=True, help="Log frequency for progress updates.")
def cli(
    outdir: Path,
    noise_count: int,
    blip_count: int,
    blip_tol: float,
    seed: int,
    both_detectors: bool,
    verbose: bool,
    log_every: int,
) -> None:
    """Generate CSVs of noise and blip trigger times, plus a merged list."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Build single output with paired triggers and blip metadata
    if verbose:
        click.echo(
            f"Selecting noise triggers: target={noise_count} both_detectors={both_detectors}",
            err=True,
        )
    noise_df = _sample_noise_triggers(
        noise_count,
        seed=seed,
        require_both_detectors=both_detectors,
        detectors=("H1", "L1"),
        verbose=verbose,
        log_every=log_every,
    )
    if verbose:
        click.echo(
            f"Selecting blip triggers: target={blip_count} tol={blip_tol} both_detectors={both_detectors}",
            err=True,
        )
    blip_df = _select_blip_triggers(
        blip_count,
        around_seconds=CCSN_DURATION,
        tol_frac=blip_tol,
        require_both_detectors=both_detectors,
        detectors=("H1", "L1"),
        verbose=verbose,
        log_every=log_every,
    )

    k = min(len(noise_df), len(blip_df))
    if k == 0:
        raise RuntimeError("No overlapping entries to build triggers CSV.")

    triggers = pd.DataFrame(
        {
            "noise_trigger": noise_df["event_time"].values[:k],
            "blip_trigger": blip_df["event_time"].values[:k],
            "blip_snr": blip_df["snr"].values[:k],
            "blip_duration": blip_df["duration"].values[:k],
        }
    )
    out_path = outdir / "triggers.csv"
    triggers.to_csv(out_path, index=False)

    if verbose:
        click.echo(
            f"Completed: wrote {len(triggers)} rows to {out_path}",
            err=True,
        )
    click.echo(f"Wrote: {out_path}")


def main(argv: Optional[list[str]] = None) -> None:  # pragma: no cover
    cli.main(args=argv, prog_name="generate_trigger_csv")
_DQ_CACHE: dict[tuple[str, int, int], bool] = {}


def _dq_online_cached(det: str, gps_start: float, gps_end: float) -> bool:
    key = (det, int(np.floor(gps_start)), int(np.ceil(gps_end)))
    hit = _DQ_CACHE.get(key)
    if hit is not None:
        return hit
    ok = _detector_cat3_online(det, gps_start, gps_end)
    _DQ_CACHE[key] = ok
    return ok
