from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from construct_benchmark.manifests import file_sha256
from construct_benchmark.prompts import PromptRecord, write_prompt_records
from scripts import plan_construct_steering


ROOT = Path(__file__).resolve().parents[1]
CONSTRUCT_ID = "realization_account_closure"
SPEC = ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json"
RUN_CONFIG = ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json"


def _write_fixture(
    tmp_path: Path,
    *,
    direction: np.ndarray | None = None,
    pair_differences: np.ndarray | None = None,
) -> dict[str, Path]:
    readout_dir = tmp_path / "readout"
    direction_path = readout_dir / "arrays" / "mean_direction.npy"
    pair_differences_path = readout_dir / "arrays" / "pair_differences.npy"
    direction_path.parent.mkdir(parents=True)
    np.save(direction_path, direction if direction is not None else np.asarray([1.0, 0.0, 0.0]))
    np.save(
        pair_differences_path,
        pair_differences
        if pair_differences is not None
        else np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )

    summary_path = readout_dir / "summary.json"
    summary = {
        "construct_id": CONSTRUCT_ID,
        "selected_layer": 20,
        "layer_selection": {
            "rule": "validation_max_margin",
            "selection_split": "direction_validation",
        },
        "direction": {
            "source_split": "direction_train",
            "path": "arrays/mean_direction.npy",
            "pair_differences_path": "arrays/pair_differences.npy",
            "direction_sha256": file_sha256(direction_path),
            "pair_differences_sha256": file_sha256(pair_differences_path),
        },
        "calibration": {
            "method": "neutral",
            "construct_id": CONSTRUCT_ID,
            "split": "calibration",
            "sample_count": 2,
            "group_count": 1,
            "projection_scale": 1.0,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    inventory_path = tmp_path / "prompt_inventory.csv"
    write_prompt_records(
        [
            PromptRecord(
                prompt_id="steering_fixture_001",
                construct_id=CONSTRUCT_ID,
                split="steering_eval",
                prompt_role="steering",
                prompt_text="Return the registered fixture response.",
                prompt_family="fixture_steering",
                task_id="fixture_task",
                expected_output_format="single_integer",
                parser_id="fixture_parser",
            )
        ],
        inventory_path,
    )
    return {
        "summary": summary_path,
        "direction": direction_path,
        "pair_differences": pair_differences_path,
        "inventory": inventory_path,
        "controls": tmp_path / "controls",
        "output": tmp_path / "steering_plan.json",
    }


def _run_plan(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
    *,
    direction: Path | None = None,
    pair_differences: Path | None = None,
) -> None:
    argv = [
        "plan_construct_steering.py",
        "--prompt-inventory",
        str(paths["inventory"]),
        "--construct-spec",
        str(SPEC),
        "--run-config",
        str(RUN_CONFIG),
        "--readout-summary",
        str(paths["summary"]),
        "--direction",
        str(direction or paths["direction"]),
        "--pair-differences",
        str(pair_differences or paths["pair_differences"]),
        "--direction-output-dir",
        str(paths["controls"]),
        "--mode",
        "test",
        "--output",
        str(paths["output"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    plan_construct_steering.main()


def test_matching_selected_train_artifacts_write_a_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_fixture(tmp_path)

    _run_plan(monkeypatch, paths)

    payload = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert payload["mode"] == "test"
    assert payload["confirmatory"] is False
    assert payload["direction_paths"]["target"] == str(paths["direction"].resolve())
    assert payload["provenance"]["direction_sha256"] == file_sha256(paths["direction"])
    assert payload["provenance"]["pair_differences_sha256"] == file_sha256(paths["pair_differences"])


@pytest.mark.parametrize("artifact", ["direction", "pair_differences"])
def test_tampered_selected_artifact_is_rejected_before_plan_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    paths = _write_fixture(tmp_path)
    if artifact == "direction":
        np.save(paths[artifact], np.asarray([2.0, 0.0, 0.0]))
    else:
        np.save(paths[artifact], np.asarray([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    with pytest.raises(SystemExit, match="hash"):
        _run_plan(monkeypatch, paths)

    assert not paths["output"].exists()
    assert not paths["controls"].exists()


@pytest.mark.parametrize("artifact", ["direction", "pair_differences"])
def test_supplied_artifact_path_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    paths = _write_fixture(tmp_path)
    mismatched_path = tmp_path / f"other_{artifact}.npy"
    value = np.asarray([1.0, 0.0, 0.0]) if artifact == "direction" else np.ones((2, 3))
    np.save(mismatched_path, value)

    with pytest.raises(SystemExit, match="artifact path"):
        _run_plan(
            monkeypatch,
            paths,
            direction=mismatched_path if artifact == "direction" else None,
            pair_differences=mismatched_path if artifact == "pair_differences" else None,
        )

    assert not paths["output"].exists()
    assert not paths["controls"].exists()


@pytest.mark.parametrize(
    ("artifact", "value", "message"),
    [
        ("direction", np.asarray([1.0, np.nan, 0.0]), "finite"),
        ("direction", np.zeros(3), "all zero"),
        ("direction", np.ones((1, 3)), "1-dimensional"),
        ("direction", np.asarray(["x", "y", "z"], dtype=object), "could not be loaded"),
        ("pair_differences", np.asarray([1.0, 0.0, 0.0]), "2-dimensional"),
        ("pair_differences", np.zeros((2, 3)), "all zero"),
        ("pair_differences", np.asarray([[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0]]), "finite"),
    ],
)
def test_invalid_numeric_artifacts_are_rejected_before_plan_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    value: np.ndarray,
    message: str,
) -> None:
    kwargs = {artifact: value}
    paths = _write_fixture(tmp_path, **kwargs)

    with pytest.raises(SystemExit, match=message):
        _run_plan(monkeypatch, paths)

    assert not paths["output"].exists()
    assert not paths["controls"].exists()


def test_incompatible_pair_hidden_size_is_rejected_before_plan_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_fixture(tmp_path, pair_differences=np.ones((2, 4)))

    with pytest.raises(SystemExit, match="incompatible shapes"):
        _run_plan(monkeypatch, paths)

    assert not paths["output"].exists()
    assert not paths["controls"].exists()


def test_non_training_selected_summary_artifacts_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_fixture(tmp_path)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["direction"]["source_split"] = "direction_heldout"
    paths["summary"].write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="direction_train"):
        _run_plan(monkeypatch, paths)

    assert not paths["output"].exists()
    assert not paths["controls"].exists()
