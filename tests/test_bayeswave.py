import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import math

from starccato_lvk.bayeswave import (
    RunSettings,
    aligned_signal_vs_glitch_or_noise,
    bayeswave_command,
    bayeswave_post_command,
    collect_result,
    detector_inputs,
    load_event_manifest,
    parse_evidence,
    prepare_frames,
)


def _write_bundle(
    path: Path, *, t0=1_260_000_000.0, dt=1 / 4096, n=16384, offsource=True
):
    """Write a bundle laid out as the campaign writes them.

    ``full_strain`` holds the off-source PSD stretch followed by the analysis
    segment: 64 s of PSD data, a 0.5 s gap, then the 4 s segment. Pass
    ``offsource=False`` for a legacy bundle with the analysis segment only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = 64.5
    with h5py.File(path, "w") as h5:
        strain = h5.create_group("strain")
        strain.create_dataset("values", data=np.zeros(n))
        strain.attrs["t0"] = t0
        strain.attrs["dt"] = dt
        if offsource:
            full = h5.create_group("full_strain")
            full.create_dataset(
                "values", data=np.zeros(n + int(round(lead / dt)))
            )
            full.attrs["t0"] = t0 - lead
            full.attrs["dt"] = dt


def _write_manifest(tmp_path: Path, *, detectors=("H1", "L1"), offsource=True) -> Path:
    bundles = {}
    for event_class in ("noise", "inj_ccsn", "inj_glitch", "real_glitch"):
        bundles[event_class] = {}
        for ifo in detectors:
            bundle = tmp_path / event_class / f"{ifo}.hdf5"
            _write_bundle(bundle, offsource=offsource)
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


def test_free_sky_omits_fixed_sky_arguments(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    manifest = load_event_manifest(manifest_path)
    output = tmp_path / "out"
    inputs = detector_inputs(manifest, manifest_path, "inj_ccsn", output)
    settings = RunSettings(
        iterations=1000, burnin=100, chains=4, threads=2, fix_sky=False
    )

    command = bayeswave_command(
        "BayesWave", inputs, output, 300.0, 800.0, manifest["sky"], settings
    )
    assert "--fixSky" not in command
    assert "--fixRA" not in command
    assert "--fixDEC" not in command


def test_fixed_sky_rejects_degrees(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    manifest = load_event_manifest(manifest_path)
    output = tmp_path / "out"
    inputs = detector_inputs(manifest, manifest_path, "inj_ccsn", output)
    settings = RunSettings(iterations=1000, burnin=100, chains=4, threads=2)

    with pytest.raises(ValueError, match="radians"):
        bayeswave_command(
            "BayesWave",
            inputs,
            output,
            300.0,
            800.0,
            {"ra": 68.8, "dec": -17.2},  # the same direction, in degrees
            settings,
        )


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
        # The frame carries the off-source PSD stretch as well as the segment,
        # so read the two spans BayesWave actually asks for rather than the
        # whole (deliberately wider, integer-second) frame span.
        segment = TimeSeries.read(
            item.frame, channel=item.channel,
            start=item.t0, end=item.t0 + item.duration,
        )
        assert segment.size == 8192
        assert segment.sample_rate.value == pytest.approx(2048.0)
        psd = TimeSeries.read(
            item.frame, channel=item.channel,
            start=item.frame_t0, end=item.frame_t0 + item.psd_length,
        )
        assert psd.size == int(64.0 * 2048)
        assert item.cache.read_text().endswith(f"{item.frame.resolve()}\n")


def test_offsource_psd_matches_the_vae_window(tmp_path):
    """psdlength must be 64 s of off-source data, not the analysis segment.

    psdlength == seglen leaves BayesLine fitting a single periodogram of the very
    data being ranked; the resulting noise evidence moved ~100 nats between chain
    seeds in the v0.44 campaign.
    """
    manifest_path = _write_manifest(tmp_path)
    manifest = load_event_manifest(manifest_path)
    inputs = detector_inputs(manifest, manifest_path, "inj_ccsn", tmp_path / "out")
    first = inputs[0]
    assert first.has_offsource_psd
    assert first.psd_length == 64.0
    # 16 averaged FFTs of the analysis-segment length, and the stretch must stop
    # before the segment starts so the PSD never sees the event.
    assert first.psd_length / first.duration == 16
    assert first.frame_t0 + first.psd_length <= first.t0

    command = bayeswave_command(
        "BayesWave", inputs, tmp_path / "out", 300.0, 800.0,
        manifest["sky"], RunSettings(),
    )
    assert command[command.index("--psdlength") + 1] == "64.0"
    assert command[command.index("--psdstart") + 1] == str(first.frame_t0)
    assert command[command.index("--seglen") + 1] == "4.0"


def test_bundle_without_offsource_stretch_is_rejected(tmp_path):
    manifest_path = _write_manifest(tmp_path, offsource=False)
    manifest = load_event_manifest(manifest_path)
    with pytest.raises(ValueError, match="no full_strain"):
        detector_inputs(manifest, manifest_path, "inj_ccsn", tmp_path / "out")
    # ...but reproducing a pre-fix run stays possible when asked for explicitly.
    inputs = detector_inputs(
        manifest, manifest_path, "inj_ccsn", tmp_path / "out",
        allow_onsource_psd=True,
    )
    assert not inputs[0].has_offsource_psd
    assert inputs[0].psd_length == inputs[0].duration


def test_parse_and_collect_result(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    manifest = load_event_manifest(manifest_path)
    output = tmp_path / "out"
    output.mkdir()
    evidence_path = output / "evidence.dat"
    evidence_path.write_text(
        "signal 12.5 0.3\n" "glitch 8.0 0.4\n" "noise -1.0 0.2\n"
    )
    # Reconstructed SNR is read per-detector (the network .geo file writes
    # snr=0) and combined in quadrature: hypot(3, 4) = 5.
    stats_dir = output / "post/signal"
    stats_dir.mkdir(parents=True)
    header = ("# map_D bayesfactor snr time duration frequency bandwidth h_max "
              "t_at_h_max f_at_max_amp\n")
    (stats_dir / "signal_stats.dat.geo").write_text(
        header + "1 1000 0 1260000002 0 0 0 0 1260000002 0\n"
    )
    (stats_dir / "signal_stats_H1.dat").write_text(
        header + "3 10 3.0 1260000002 0.1 500 200 1e-22 1260000002 510\n"
    )
    (stats_dir / "signal_stats_L1.dat").write_text(
        header + "3 10 4.0 1260000002 0.1 500 200 1e-22 1260000002 510\n"
    )

    parsed = parse_evidence(evidence_path)
    assert parsed["signal"] == (12.5, 0.3)

    # Placeholder rows BayesWave writes for models it never sampled must not be
    # reported as a real (huge) Bayes factor -- both seen in the H1-L1 pilot.
    for label, text in (
        ("sentinel", "signal 0 0\nglitch 10000 1\nnoise 0 0\n"),
        ("negative uncertainty",
         "signal 604415.0 4.38\nglitch 604465.0 -6.03e-05\nnoise 604513.7 3.3e-05\n"),
    ):
        bad = evidence_path.parent / f"evidence_{label.split()[0]}.dat"
        bad.write_text(text)
        with pytest.raises(ValueError, match="placeholder/unsampled"):
            parse_evidence(bad)
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
    assert result["signal_reconstructed_snr_median"] == pytest.approx(5.0)
    assert result["signal_reconstructed_snr_per_detector"] == {"H1": 3.0, "L1": 4.0}
    assert result["target_snr"] == 17.5
    # Aligned statistic: glitch (8.0) dominates noise (-1.0) in the mixture, so
    # the denominator is lnZ_glitch + ln(beta) and the aligned value is the
    # native factor plus ln 2. Its uncertainty is dominated by the same two
    # terms, with the noise term suppressed by its ~0 mixture weight.
    assert result["log_bayeswave_signal_glitch_or_noise"] == pytest.approx(
        4.5 + math.log(2.0), abs=1e-3
    )
    assert result["aligned_mixture"]["weight_glitch"] == pytest.approx(1.0, abs=1e-3)
    assert result["log_bayeswave_signal_glitch_or_noise_uncertainty"] == pytest.approx(
        0.5, abs=1e-3
    )


def test_aligned_statistic_tracks_the_dominant_denominator_model():
    # Noise-dominated: the aligned statistic must follow lnZ_noise, NOT lnZ_glitch.
    # This is the whole point -- a quiet event's native lnB_S/G is the difference
    # of two rejected models and can sit anywhere.
    aligned, w_glitch, w_noise = aligned_signal_vs_glitch_or_noise(2.0, -40.0, 1.0)
    assert w_noise == pytest.approx(1.0, abs=1e-6)
    assert aligned == pytest.approx(2.0 - 1.0 + math.log(2.0), abs=1e-6)
    # Equal evidences: the mixture collapses to that common value exactly, and
    # the weights split evenly.
    aligned, w_glitch, w_noise = aligned_signal_vs_glitch_or_noise(5.0, 3.0, 3.0)
    assert aligned == pytest.approx(2.0)
    assert w_glitch == pytest.approx(0.5)
    assert w_noise == pytest.approx(0.5)


def test_settings_require_thread_factor():
    with pytest.raises(ValueError, match="divide chains"):
        RunSettings(chains=20, threads=3).validate()
