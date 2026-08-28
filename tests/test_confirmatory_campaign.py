from __future__ import annotations

import json
from pathlib import Path

import pytest

from construct_benchmark.config import load_run_config
from construct_benchmark.campaign import release_wave_prompt_inventory, wave_construct_ids


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/construct_benchmark/construct_registry_v1.json"
CAMPAIGN_PATH = ROOT / "configs/construct_benchmark/confirmatory_campaigns/waves2_4_confirmatory_v1.json"


def test_confirmatory_campaign_keeps_one_balanced_run_per_wave() -> None:
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    assert campaign["status"] == "repaired_v2_prompt_inputs_ready_execution_gated"
    assert campaign["confirmatory_release"] is False
    assert campaign["prompt_input_release"]["status"] == "validated_nonconfirmatory"
    assert campaign["prompt_input_release"]["confirmatory"] is False
    assert campaign["prompt_input_release"]["audit_status"] == "passed"
    assert [entry["wave"] for entry in campaign["waves"]] == [2, 3, 4]

    for entry in campaign["waves"]:
        assert tuple(entry["construct_ids"]) == wave_construct_ids(REGISTRY_PATH, entry["wave"])
        run_config_path = CAMPAIGN_PATH.parent / entry["run_config_path"]
        run_config = load_run_config(run_config_path)
        assert tuple(run_config.construct_ids) == tuple(entry["construct_ids"])
        assert run_config.execution["max_constructs_per_run"] == 4
        assert run_config.execution["parallel_construct_analysis"] is True
        assert run_config.execution["shared_activation_pass"] is True
        assert run_config.execution["run_modes"]["full"]["confirmatory"] is True
        assert run_config.execution["run_modes"]["test"]["confirmatory"] is False


def test_confirmatory_campaign_has_explicit_release_blockers() -> None:
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    blockers = {item["id"]: item for item in campaign["required_confirmatory_prerequisites"]}
    assert set(blockers) == {
        "wave1_measurement_gate",
        "repaired_prompt_inventory_audit",
        "precision_simulation",
        "downstream_inventory_release",
    }
    assert blockers["wave1_measurement_gate"]["status"] == "pending"
    assert blockers["repaired_prompt_inventory_audit"]["status"] == "satisfied"
    assert blockers["precision_simulation"]["status"] == "pending"
    assert blockers["downstream_inventory_release"]["status"] == "pending"
    assert blockers["downstream_inventory_release"]["audit_paths"]


def test_prompt_release_refuses_inventory_without_passing_audit(tmp_path: Path) -> None:
    source_manifest_path = (
        ROOT
        / "results/benchmark/prompt_inventories/wave2_four_construct_full_luna_v1/inventory_manifest.json"
    )
    source_manifest_before = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_before)
    output_dir = tmp_path / "wave2_release"

    with pytest.raises(ValueError, match="prompt_audit"):
        release_wave_prompt_inventory(
            wave=2,
            registry_path=REGISTRY_PATH,
            source_manifest_path=source_manifest_path,
            output_dir=output_dir,
            released_by="test authority",
            release_statement="Release prompt inputs only; model execution remains gated.",
            release_date="2026-08-27",
        )

    assert source_manifest_path.read_bytes() == source_manifest_before
    assert source_manifest["confirmatory"] is False
