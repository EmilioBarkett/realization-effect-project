from __future__ import annotations

import json
from pathlib import Path

from construct_benchmark.config import load_construct_spec, load_run_config
from construct_benchmark.generation import load_generation_plan
from construct_benchmark.registry import load_construct_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/construct_benchmark"
REGISTRY_PATH = CONFIG_ROOT / "construct_registry_repaired_v2.json"


def test_repaired_registry_loads_all_versioned_specs_and_plans() -> None:
    registry = load_construct_registry(REGISTRY_PATH)
    raw_registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    raw_entries = {entry["construct_id"]: entry for entry in raw_registry["entries"]}

    assert len(registry.entries) == 16
    for entry in registry.entries:
        raw_entry = raw_entries[entry.construct_id]
        spec = load_construct_spec(CONFIG_ROOT / raw_entry["spec_path"])
        plan = load_generation_plan(CONFIG_ROOT / raw_entry["generation_plan_path"], spec)
        assert spec.version == "v2"
        if entry.wave == 1:
            assert plan["plan_id"].endswith("v2")
        else:
            assert plan["plan_id"].endswith("repaired_v2")
        assert plan["models"] == [{"alias": "luna", "model": "gpt-5.6-luna"}]
        assert plan["generation"]["max_items_per_request"] == 20
        if entry.wave > 1:
            assert plan["generation"]["retries"] == 0
        for cell in plan["cells"]:
            if cell["split"] in {"behavior_eval", "steering_eval", "calibration"}:
                assert cell["task_id"] == spec.independent_behavior_task["task_id"]


def test_qwen38_replication_config_is_pinned_and_storage_efficient() -> None:
    config = load_run_config(
        CONFIG_ROOT / "run_configs/wave1_four_construct_qwen38_27b_repaired_v2.json"
    )

    assert config.model == {
        "model_id": "Qwen/Qwen3.8-27B",
        "tokenizer_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    }
    assert config.activation["layers"] == [16, 32, 48]
    assert config.activation["storage_dtype"] == "float16"
    topology = config.execution["parallel_executor"]["topology"]
    assert topology == {
        "provider": "runpod",
        "pod_count": 1,
        "gpu_type": "NVIDIA B300",
        "gpu_count": 1,
        "model_replica_rollout": [1, 3, 4],
    }


def test_engineering_full_execution_config_preserves_nonconfirmatory_identity() -> None:
    config = load_run_config(
        CONFIG_ROOT / "run_configs/wave1_four_construct_repaired_v2_engineering_full_v1.json"
    )
    full = config.execution["run_modes"]["full"]
    assert config.execution["default_run_mode"] == "full"
    assert full["purpose"] == "full_coverage_engineering"
    assert full["confirmatory"] is False
    assert full["engineering_only"] is True


def test_all_active_run_configs_use_fp16_activation_storage() -> None:
    run_config_paths = sorted((CONFIG_ROOT / "run_configs").glob("*.json"))

    assert run_config_paths
    for path in run_config_paths:
        config = load_run_config(path)
        assert config.activation["storage_dtype"] == "float16", path


def test_repaired_wave2_to_wave4_task_families_avoid_v1_independence_blockers() -> None:
    registry = load_construct_registry(REGISTRY_PATH)
    raw_registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    raw_entries = {entry["construct_id"]: entry for entry in raw_registry["entries"]}
    blocked_v1_families = {
        "reference_dependent_risk_choice",
        "prior_evidence_probability_judgment",
        "authority_evidence_conflict_choice",
        "two_option_bandit_choice",
    }

    for entry in registry.entries:
        if entry.wave == 1:
            continue
        raw_entry = raw_entries[entry.construct_id]
        spec = load_construct_spec(CONFIG_ROOT / raw_entry["spec_path"])
        assert spec.independent_behavior_task["task_family"] not in blocked_v1_families
