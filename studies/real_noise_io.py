"""Shared bundle I/O for the production real-noise studies."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from starccato_lvk.acquisition.io.strain_loader import strain_loader


def build_bundle(trigger_time: float, outdir: Path, detector: str) -> Path:
    """Build or reuse an analysis bundle from the local mirror or GWOSC."""
    outdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(outdir.glob("analysis_bundle_*.hdf5"))
    if existing:
        return existing[0]
    strain_loader(
        trigger_time=trigger_time,
        outdir=str(outdir),
        data_fetcher=None,
        detector=detector,
        # Catalogue blips fail CAT3 by construction. The campaign deliberately
        # selects its trigger times and therefore must be able to read them.
        require_cat3=False,
    )
    bundles = sorted(outdir.glob("analysis_bundle_*.hdf5"))
    if not bundles:
        raise FileNotFoundError(
            f"No bundle written for {detector} at GPS {trigger_time}"
        )
    return bundles[0]


def inject_into_bundle(
    noise_bundle: Path, injection_td: np.ndarray, destination: Path
) -> Path:
    """Add a time-domain injection while preserving PSD and bundle metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(noise_bundle, "r") as source:
        strain = np.asarray(source["strain"]["values"])
        injection = np.asarray(injection_td)
        if injection.shape[0] < strain.shape[0]:
            raise ValueError(
                "Injection is shorter than the analysis strain: "
                f"{injection.shape[0]} < {strain.shape[0]}"
            )
        injected = strain + injection[: strain.shape[0]]
        with h5py.File(destination, "w") as output:
            for key, value in source.attrs.items():
                output.attrs[key] = value
            source.copy("psd", output)
            if "full_strain" in source:
                source.copy("full_strain", output)
            strain_group = output.create_group("strain")
            strain_group.create_dataset("values", data=injected)
            for key, value in source["strain"].attrs.items():
                strain_group.attrs[key] = value
    return destination
