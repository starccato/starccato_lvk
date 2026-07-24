from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from starccato_lvk.analysis import main as analysis_main

STUDIES = Path(__file__).resolve().parents[1] / "studies"
sys.path.insert(0, str(STUDIES))

import real_noise_event as campaign  # noqa: E402
import real_noise_aggregate  # noqa: E402
import real_noise_plots  # noqa: E402
import collect_results  # noqa: E402
from real_noise_io import inject_into_bundle  # noqa: E402


def test_confirmed_missing_remote_data_is_distinguished_from_network_failure():
    missing = FileNotFoundError("no archive file")
    assert campaign.StrainDataUnavailable.__name__ == "StrainDataUnavailable"
    from starccato_lvk.acquisition.io.strain_loader import (
        _is_confirmed_missing_data,
    )

    assert _is_confirmed_missing_data(ExceptionGroup("gwpy", [missing]))
    assert not _is_confirmed_missing_data(ConnectionError("DNS unavailable"))
    assert issubclass(
        campaign.StrainDataUnavailable, campaign.StrainFetchFailed
    )


def test_unavailable_trigger_is_recorded_and_excluded_from_collection(
    tmp_path, monkeypatch
):
    rejected = campaign._record_unavailable_trigger(
        index=2,
        outdir=tmp_path,
        campaign_id="campaign-test",
        detectors=["L1"],
        blip_ifo="L1",
        error=campaign.StrainDataUnavailable("No GWOSC strain data"),
    )
    assert rejected.exists()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for event_class in campaign.PRODUCTION_CLASSES:
        (results_dir / f"e0_{event_class}.json").write_text(
            json.dumps(
                {
                    "campaign_id": "campaign-test",
                    "index": 0,
                    "cls": event_class,
                }
            )
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_results.py",
            str(tmp_path),
            "--expected-start",
            "0",
            "--expected-stop",
            "2",
        ],
    )
    collect_results.main()
    summary = json.loads((tmp_path / "collection_summary.json").read_text())
    assert {row["index"] for row in summary["missing"]} == {1}
    assert summary["rejected_triggers"][0]["index"] == 2


def test_manifest_fingerprint_detects_changes():
    manifest = {
        "schema_version": campaign.MANIFEST_SCHEMA_VERSION,
        "campaign_id": "test",
        "index": 3,
    }
    manifest["manifest_fingerprint"] = campaign._json_fingerprint(manifest)

    assert (
        campaign._validated_manifest_fingerprint(manifest)
        == manifest["manifest_fingerprint"]
    )

    manifest["index"] = 4
    with pytest.raises(RuntimeError, match="fingerprint"):
        campaign._validated_manifest_fingerprint(manifest)


def test_inject_into_bundle_preserves_metadata(tmp_path):
    source = tmp_path / "source.hdf5"
    destination = tmp_path / "injected.hdf5"
    with h5py.File(source, "w") as handle:
        handle.attrs["trigger_time"] = 123.0
        strain = handle.create_group("strain")
        strain.create_dataset("values", data=np.arange(4.0))
        strain.attrs["dt"] = 0.25
        psd = handle.create_group("psd")
        psd.create_dataset("values", data=np.ones(3))

    inject_into_bundle(source, np.full(4, 2.0), destination)

    with h5py.File(destination, "r") as handle:
        assert handle.attrs["trigger_time"] == 123.0
        assert handle["strain"].attrs["dt"] == 0.25
        np.testing.assert_allclose(
            handle["strain"]["values"][:], [2.0, 3.0, 4.0, 5.0]
        )
        np.testing.assert_allclose(handle["psd"]["values"][:], 1.0)


def test_lightweight_nuts_diagnostics_are_json_serializable(tmp_path):
    rng = np.random.default_rng(4)
    result = SimpleNamespace(
        extra={
            "samples_grouped": {"z0": rng.normal(size=(2, 40))},
            "diverging": np.zeros(80, dtype=bool),
            "accept_prob": np.full(80, 0.9),
            "num_steps": np.full(80, 7),
            "energy": rng.normal(size=80),
            "num_chains": 2,
            "num_warmup": 20,
            "num_samples": 40,
            "target_accept_prob": 0.8,
            "max_tree_depth": 10,
            "chain_method": "vectorized",
        }
    )

    analysis_main._save_lightweight_nuts_diagnostics(tmp_path, result)

    payload = json.loads((tmp_path / "nuts_diagnostics.json").read_text())
    assert payload["divergences"] == 0
    assert payload["num_chains"] == 2
    assert payload["chain_method"] == "vectorized"
    assert "max_rhat" in payload


def test_three_class_aggregation_and_table(tmp_path, monkeypatch):
    rows = []
    for index in range(5):
        for event_class, score, snr in (
            ("noise", -2.0, 0.0),
            ("inj_ccsn", 5.0 + index, 10.0 + index),
            ("real_glitch", -4.0, 20.0 + index),
        ):
            rows.append(
                {
                    "campaign_id": "campaign-test",
                    "index": index,
                    "cls": event_class,
                    "log_odds": score,
                    "snr": snr,
                }
            )
    (tmp_path / "results.json").write_text(json.dumps(rows))

    monkeypatch.setattr(
        sys,
        "argv",
        ["real_noise_aggregate.py", "--outdir", str(tmp_path)],
    )
    real_noise_aggregate.main()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["classes"] == ["noise", "inj_ccsn", "real_glitch"]
    assert summary["n_complete_class_groups"] == 5
    assert summary["auc_odds_signal_vs_inj_glitch"] is None

    run = real_noise_plots.load_run(tmp_path)
    assert run is not None
    np.testing.assert_array_equal(
        real_noise_plots._background(run, "log_odds"),
        np.array([-2.0] * 5 + [-4.0] * 5),
    )
    table = tmp_path / "summary_table.tex"
    real_noise_plots.write_table({"L1": run}, table)
    assert "inj.\\ blip" not in table.read_text()


def test_collect_results_defaults_to_production_classes(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for event_class in campaign.PRODUCTION_CLASSES:
        (results_dir / f"e0_{event_class}.json").write_text(
            json.dumps(
                {
                    "campaign_id": "campaign-test",
                    "index": 0,
                    "cls": event_class,
                }
            )
        )
        (results_dir / f"e0_{event_class}_baseline.json").write_text(
            json.dumps(
                {
                    "index": 0,
                    "cls": event_class,
                    "mf_snr": 1.0,
                }
            )
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_results.py",
            str(tmp_path),
            "--expected-start",
            "0",
            "--expected-stop",
            "0",
        ],
    )
    collect_results.main()

    summary = json.loads((tmp_path / "collection_summary.json").read_text())
    assert summary["expected_classes"] == list(campaign.PRODUCTION_CLASSES)
    assert summary["missing"] == []
    assert summary["campaign_ids"] == ["campaign-test"]


def test_analysis_failure_is_recorded_and_propagated(tmp_path, monkeypatch):
    manifest = {
        "schema_version": campaign.MANIFEST_SCHEMA_VERSION,
        "campaign_id": "campaign-test",
        "index": 0,
        "detectors": ["L1"],
        "glitch_det": "L1",
        "blip_ifo": "L1",
        "band": [300.0, 800.0],
        "sky": {},
        "snr": {"noise": 0.0},
        "bundles": {"noise": {"L1": "unused.hdf5"}},
    }
    manifest["manifest_fingerprint"] = campaign._json_fingerprint(manifest)
    provenance = {"packages": {}, "models": {}}
    monkeypatch.setattr(campaign, "_runtime_provenance", lambda: provenance)

    def fail_analysis(**_kwargs):
        raise ValueError("deliberate test failure")

    monkeypatch.setattr(campaign, "run_bcr_posteriors", fail_analysis)

    with pytest.raises(RuntimeError, match="noise"):
        campaign.analyse_manifest(
            manifest,
            tmp_path,
            num_warmup=10,
            num_samples=20,
            nsm=True,
            classes=("noise",),
        )

    failure = json.loads((tmp_path / "failures" / "e0_noise.json").read_text())
    assert failure["error_type"] == "ValueError"
    assert failure["error"] == "deliberate test failure"
