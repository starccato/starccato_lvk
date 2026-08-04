import json
import importlib.util
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "studies" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convergence = _load("bw_pilot_convergence")
selection = _load("select_bw_pilot_events")


def _manifest(root: Path, index: int, event_class: str) -> None:
    path = root / f"e{index}" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "index": index,
                "detectors": ["H1", "L1"],
                "bundles": {
                    event_class: {"H1": "H1.hdf5", "L1": "L1.hdf5"}
                },
            }
        )
    )


def test_selection_is_first_index_per_class_without_bayeswave_outputs(tmp_path):
    ccsn = tmp_path / "ccsn"
    glitch = tmp_path / "glitch"
    for index in (9, 2, 5):
        _manifest(ccsn, index, "inj_ccsn")
    for index in (8, 1, 4):
        _manifest(glitch, index, "real_glitch")

    cohort = selection.select(
        {"inj_ccsn": ccsn, "real_glitch": glitch},
        per_class=2,
        seeds=[11, 22, 33],
    )
    assert cohort["classes"]["inj_ccsn"]["indices"] == [2, 5]
    assert cohort["classes"]["real_glitch"]["indices"] == [1, 4]
    assert cohort["n_events"] == 4
    assert cohort["n_tasks"] == 12


def _result(path: Path, signal: float, glitch: float, noise: float, unc: float) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "logZ_signal": signal,
                "logZ_glitch": glitch,
                "logZ_noise": noise,
                "evidence_uncertainty": {
                    "signal": unc,
                    "glitch": unc,
                    "noise": unc,
                },
            }
        )
    )


def test_convergence_uses_three_seed_sign_consistency_not_reported_sigma(tmp_path):
    cohort = {
        "seeds": [11, 22, 33],
        "classes": {
            "inj_ccsn": {
                "events": [{"cls": "inj_ccsn", "index": 2}]
            },
            "real_glitch": {
                "events": [{"cls": "real_glitch", "index": 1}]
            },
        },
    }
    # Huge reported uncertainties do not reject a sign-stable injection.
    for seed, signal in zip(cohort["seeds"], (10.0, 12.0, 14.0)):
        _result(
            tmp_path / "e2" / "inj_ccsn" / f"seed{seed}" / "result.json",
            signal,
            1.0,
            0.0,
            unc=1000.0,
        )
    # The blip flips native and aligned signs between seeds and is reported as
    # non-converged rather than removed from the attempted count.
    for seed, signal in zip(cohort["seeds"], (2.0, -2.0, 2.0)):
        _result(
            tmp_path / "e1" / "real_glitch" / f"seed{seed}" / "result.json",
            signal,
            0.0,
            0.0,
            unc=0.01,
        )

    rows = convergence.analyse(
        tmp_path,
        cohort,
        {("inj_ccsn", 2): 5.0, ("real_glitch", 1): -5.0},
    )
    summary = convergence.summarise(rows)
    assert [row["outcome"] for row in rows] == [
        "seed_consistent",
        "seed_inconsistent",
    ]
    assert summary["n_attempted"] == 2
    assert summary["n_seed_consistent"] == 1
    assert summary["n_seed_inconsistent"] == 1
    assert not summary["reported_uncertainty_used_for_acceptance"]
    assert summary["contingency"]["native"]["total"]["both"] == 1
