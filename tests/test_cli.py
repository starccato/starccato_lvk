import pytest
from click.testing import CliRunner
from starccato_lvk import cli

import sys

@pytest.fixture(autouse=True)
def patch_sys_argv(monkeypatch):
    # Prevent click from picking up pytest's argv
    monkeypatch.setattr(sys, "argv", ["cli"])


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_acquire_data_command(monkeypatch, outdir):
    runner = CliRunner()
    called = {}
    outdir = f"{outdir}/test_cli"
    def fake_get_analysis_data(index, trigger_type, outdir):
        called["index"] = index
        called["trigger_type"] = trigger_type
        called["outdir"] = outdir

    monkeypatch.setattr("starccato_lvk.acquisition.main.cli_get_analysis_data", fake_get_analysis_data)
    result = runner.invoke(cli.cli, ["acquire", "data", "5", "--trigger-type", "blip", "--outdir", "foo"])
    assert result.exit_code == 0
    assert "Acquiring blip data for index 5" in result.output
    assert called == {"index": 5, "trigger_type": "blip", "outdir": "foo"}


def test_acquire_batch_command(monkeypatch):
    runner = CliRunner()
    called = {}

    def fake_collect_lvk_data(num_samples, outdir):
        called["num_samples"] = num_samples
        called["outdir"] = outdir

    monkeypatch.setattr("starccato_lvk.acquisition.main.cli_collect_lvk_data", fake_collect_lvk_data)
    result = runner.invoke(cli.cli, ["acquire", "batch", "3", "--outdir", "batchdir"])
    assert result.exit_code == 0
    assert "Starting batch acquisition of 3 samples" in result.output
    assert called == {"num_samples": 3, "outdir": "batchdir"}


def test_run_command(monkeypatch, tmp_path):
    runner = CliRunner()
    called = {}

    def fake_run_starccato_analysis(
        data_path,
        psd_path,
        outdir,
        injection_model_type,
        num_samples,
        force_rerun,
        test_mode,
        verbose,
        save_artifacts,
    ):
        called.update(locals())

    monkeypatch.setattr("starccato_lvk.analysis.main.run_starccato_analysis", fake_run_starccato_analysis)
    data_file = tmp_path / "data.hdf5"
    psd_file = tmp_path / "psd.hdf5"
    outdir = tmp_path / "output"
    data_file.write_text("dummy")
    psd_file.write_text("dummy")
    outdir.mkdir()
    result = runner.invoke(
        cli.cli,
        [
            "run",
            str(data_file),
            str(outdir),
            "--psd-path",
            str(psd_file),
            "--injection-model",
            "ccsne",
            "--num-samples",
            "10",
            "--force-rerun",
            "--test-mode",
            "--verbose",
            "--skip-artifacts",
        ],
    )
    assert result.exit_code == 0
    assert "Analysis complete." in result.output
    assert called["data_path"] == str(data_file)
    assert called["psd_path"] == str(psd_file)
    assert called["outdir"] == str(outdir)
    assert called["injection_model_type"] == "ccsne"
    assert called["num_samples"] == 10
    assert called["force_rerun"] is True
    assert called["test_mode"] is True
    assert called["verbose"] is True
    assert called["save_artifacts"] is False


def test_run_command_bundle(monkeypatch, tmp_path):
    runner = CliRunner()
    called = {}

    def fake_run_starccato_analysis(
        data_path,
        psd_path,
        outdir,
        injection_model_type,
        num_samples,
        force_rerun,
        test_mode,
        verbose,
        save_artifacts,
    ):
        called.update(locals())

    monkeypatch.setattr("starccato_lvk.analysis.main.run_starccato_analysis", fake_run_starccato_analysis)

    bundle_file = tmp_path / "analysis_bundle.hdf5"
    bundle_file.write_text("bundle")
    outdir = tmp_path / "output"
    outdir.mkdir()

    result = runner.invoke(
        cli.cli,
        [
            "run",
            str(bundle_file),
            str(outdir),
            "--skip-artifacts",
        ],
    )
    assert result.exit_code == 0
    assert called["data_path"] == str(bundle_file)
    assert called["psd_path"] is None
    assert called["save_artifacts"] is False
