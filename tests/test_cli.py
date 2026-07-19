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

    monkeypatch.setattr(cli, "cli_get_analysis_data", fake_get_analysis_data)
    result = runner.invoke(
        cli.cli,
        [
            "acquire",
            "data",
            "5",
            "--trigger-type",
            "blip",
            "--outdir",
            "foo",
        ],
    )
    assert result.exit_code == 0
    assert "Acquiring blip data for index 5" in result.output
    assert called == {"index": 5, "trigger_type": "blip", "outdir": "foo"}


def test_acquire_batch_command(monkeypatch):
    runner = CliRunner()
    called = {}

    def fake_collect_lvk_data(num_samples, outdir):
        called["num_samples"] = num_samples
        called["outdir"] = outdir

    monkeypatch.setattr(cli, "cli_collect_lvk_data", fake_collect_lvk_data)
    result = runner.invoke(
        cli.cli,
        ["acquire", "batch", "3", "--outdir", "batchdir"],
    )
    assert result.exit_code == 0
    assert "Starting batch acquisition of 3 samples" in result.output
    assert called == {"num_samples": 3, "outdir": "batchdir"}


def test_run_command(monkeypatch, tmp_path):
    runner = CliRunner()
    called = {}

    def fake_run_starccato_analysis(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(
        cli, "run_starccato_analysis", fake_run_starccato_analysis
    )
    bundle_file = tmp_path / "analysis_bundle.hdf5"
    outdir = tmp_path / "output"
    bundle_file.write_text("bundle")
    outdir.mkdir()
    result = runner.invoke(
        cli.cli,
        [
            "run",
            str(outdir),
            "--bundle",
            f"H1={bundle_file}",
            "--model",
            "ccsne",
            "--num-samples",
            "10",
            "--no-save-artifacts",
        ],
    )
    assert result.exit_code == 0
    assert "Analysis complete." in result.output
    assert called["outdir"] == str(outdir)
    assert called["detectors"] == ["H1"]
    assert called["bundle_paths"] == {"H1": str(bundle_file)}
    assert called["trigger_time"] is None
    assert called["model_types"] == ["ccsne"]
    assert called["num_samples"] == 10
    assert called["save_artifacts"] is False


def test_run_command_trigger_time(monkeypatch, tmp_path):
    runner = CliRunner()
    called = {}

    def fake_run_starccato_analysis(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(
        cli, "run_starccato_analysis", fake_run_starccato_analysis
    )

    outdir = tmp_path / "output"
    outdir.mkdir()

    result = runner.invoke(
        cli.cli,
        [
            "run",
            str(outdir),
            "--trigger-time",
            "1234567890",
            "--no-save-artifacts",
        ],
    )
    assert result.exit_code == 0
    assert called["bundle_paths"] is None
    assert called["trigger_time"] == 1234567890.0
    assert called["save_artifacts"] is False
