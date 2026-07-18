import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from starccato_lvk.bayeswave import (
    RunSettings,
    bayeswave_command,
    bayeswave_post_command,
    collect_result,
    detector_inputs,
    load_event_manifest,
    parse_evidence,
    prepare_frames,
)


def _write_bundle(path: Path, *, t0=1_260_000_000.0, dt=1 / 4096, n=16384):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        strain = h5.create_group("strain")
        strain.create_dataset("values", data=np.zeros(n))
        strain.attrs["t0"] = t0
        strain.attrs["dt"] = dt


def _write_manifest(tmp_path: Path, *, detectors=("H1", "L1")) -> Path:
    bundles = {}
    for event_class in ("noise", "inj_ccsn", "inj_glitch", "real_glitch"):
        bundles[event_class] = {}
        for ifo in detectors:
            bundle = tmp_path / event_class / f"{ifo}.hdf5"
            _write_bundle(bundle)
            bundles[event_class][ifo] = str(bundle)
    manifest = {
        "index": 7,
        "detectors": list(detectors),
        "band": [300.0, 800.0],
        "sky": {"ra": 1.2, "dec": -0.3, "psi": 0.4},
        "snr": {
            "noise": 0.0,
            "inj_ccsn": 17.5,
            "inj_glitch": 17.5,
            "real_glitch": 42.0,
        },
        "bundles": bundles,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_manifest_requires_network(tmp_path):
    path = _write_manifest(tmp_path, detectors=("L1",))
    with pytest.raises(ValueError, match="at least two detectors"):
        load_event_manifest(path)


def test_repo_relative_bundle_paths_resolve_from_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    for classes in manifest["bundles"].values():
        for ifo, path in classes.items():
            classes[ifo] = str(Path(path).relative_to(tmp_path))
    manifest_path.write_text(json.dumps(manifest))

    loaded = load_event_manifest(manifest_path)
    inputs = detector_inputs(
        loaded, manifest_path, "inj_ccsn", tmp_path / "output"
    )
    assert all(item.bundle.is_file() for item in inputs)


def test_commands_match_manifest_and_fixed_sky(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    manifest = load_event_manifest(manifest_path)
    output = tmp_path / "out"
    inputs = detector_inputs(manifest, manifest_path, "inj_ccsn", output)
    settings = RunSettings(iterations=1000, burnin=100, chains=4, threads=2)

    command = bayeswave_command(
        "BayesWave", inputs, output, 300.0, 800.0, manifest["sky"], settings
    )
    assert command.count("--ifo") == 2
    assert command[command.index("--H1-fhigh") + 1] == "1024.0"
    assert command[command.index("--L1-fhigh") + 1] == "1024.0"
    assert command[command.index("--segment-start") + 1] == "1260000000.0"
    assert command[command.index("--seglen") + 1] == "4.0"
    assert command[command.index("--srate") + 1] == "2048"
    assert command[command.index("--trigtime") + 1] == "1260000002.0"
    assert command[command.index("--fixRA") + 1] == "1.2"
    assert command[command.index("--fixDEC") + 1] == "-0.3"
    # Checkpoint resume fails without --dataseed; it must match Post's dataseed.
    assert command[command.index("--dataseed") + 1] == str(settings.seed)

    post = bayeswave_post_command(
        "BayesWavePost", inputs, output, 300.0, 800.0, settings
    )
    h1_cache = post[post.index("--H1-cache") + 1]
    assert h1_cache == f"interp:{output.resolve()}/H1_fairdraw_asd.dat"
    assert "--dataseed" in post
    assert "--0noise" in post


def test_detector_inputs_reject_mismatched_grids(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    _write_bundle(Path(manifest["bundles"]["inj_ccsn"]["L1"]), dt=1 / 2048)
    manifest_path.write_text(json.dumps(manifest))
    loaded = load_event_manifest(manifest_path)
    with pytest.raises(ValueError, match="same time grid"):
        detector_inputs(loaded, manifest_path, "inj_ccsn", tmp_path / "out")


def test_prepare_frames_downsamples_to_power_of_two(tmp_path):
    from gwpy.timeseries import TimeSeries

    manifest_path = _write_manifest(tmp_path)
    manifest = load_event_manifest(manifest_path)
    inputs = detector_inputs(
        manifest, manifest_path, "inj_ccsn", tmp_path / "out"
    )
    prepare_frames(inputs)

    for item in inputs:
        frame = TimeSeries.read(item.frame, channel=item.channel)
        assert frame.size == 8192
        assert frame.sample_rate.value == pytest.approx(2048.0)
        assert item.cache.read_text().endswith(f"{item.frame.resolve()}\n")


def test_parse_and_collect_result(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    manifest = load_event_manifest(manifest_path)
    output = tmp_path / "out"
    output.mkdir()
    evidence_path = output / "evidence.dat"
    evidence_path.write_text(
        "signal 12.5 0.3\n" "glitch 8.0 0.4\n" "noise -1.0 0.2\n"
    )
    stats = output / "post/signal/signal_stats.dat.geo"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        "# map_D bayesfactor snr time duration frequency bandwidth h_max "
        "t_at_h_max f_at_max_amp\n"
        "3 10 14.25 1260000002 0.1 500 200 1e-22 1260000002 510\n"
    )

    parsed = parse_evidence(evidence_path)
    assert parsed["signal"] == (12.5, 0.3)
    result = collect_result(
        manifest,
        manifest_path,
        "inj_ccsn",
        output,
        RunSettings(),
        elapsed_seconds=123.0,
    )
    assert result["log_bayeswave_signal_glitch"] == 4.5
    assert result["log_bayeswave_signal_glitch_uncertainty"] == 0.5
    assert result["signal_reconstructed_snr_median"] == 14.25
    assert result["target_snr"] == 17.5


def test_settings_require_thread_factor():
    with pytest.raises(ValueError, match="divide chains"):
        RunSettings(chains=20, threads=3).validate()
