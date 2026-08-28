from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fake_benchmark_runs_one_construct_without_external_services(tmp_path: Path) -> None:
    output_dir = tmp_path / "fake_run"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_fake_benchmark.py"),
        "--construct-spec",
        str(ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json"),
        "--run-config",
        str(ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json"),
        "--analysis-spec",
        str(ROOT / "configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json"),
        "--output-dir",
        str(output_dir),
        "--bootstrap-resamples",
        "40",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    summary = json.loads((output_dir / "fake_summary.json").read_text(encoding="utf-8"))
    construct = summary["constructs"]["realization_account_closure"]
    assert completed.returncode == 0
    assert summary["external_calls"] == {"openrouter": False, "runpod": False, "model_weights": False}
    assert construct["selected_layer"] == 20
    assert construct["steering"]["doses"] == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert construct["steering"]["condition_counts"] == {"random": 72, "shuffled": 24, "target": 30}
    assert construct["steering"]["tracking_layers"] == [20, 30]
    assert construct["steering"]["manipulation_checks"]["injection_record_count"] == 126
    assert construct["steering"]["manipulation_checks"]["missing_or_unscorable_records"] == 0
    assert (output_dir / "prompt_inventory.csv").exists()


def test_fake_benchmark_runs_wave1_four_constructs(tmp_path: Path) -> None:
    output_dir = tmp_path / "fake_wave1_run"
    construct_dir = ROOT / "configs/construct_benchmark/constructs"
    command = [sys.executable, str(ROOT / "scripts/run_fake_benchmark.py")]
    command.extend(
        item
        for construct_id in (
            "realization_account_closure",
            "evidence_diagnosticity",
            "source_reliability",
            "persistence_continuation",
        )
        for item in (
            "--construct-spec",
            str(construct_dir / f"{construct_id}_v1.json"),
        )
    )
    command.extend(
        [
            "--run-config",
            str(ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_smoke_v1.json"),
            "--analysis-spec",
            str(ROOT / "configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json"),
            "--output-dir",
            str(output_dir),
            "--bootstrap-resamples",
            "40",
        ]
    )
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    summary = json.loads((output_dir / "fake_summary.json").read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert summary["construct_ids"] == [
        "realization_account_closure",
        "evidence_diagnosticity",
        "source_reliability",
        "persistence_continuation",
    ]
    assert summary["inventory"]["total_prompts"] == 4 * (2 * sum((8, 4, 6)) + 3 * 6)
    assert summary["constructs"]["source_reliability"]["fake_fixture"]["empirical_result"] is False
    assert (output_dir / "prompt_inventory.csv").exists()
