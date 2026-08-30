from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage_model_steering_preflight.py"
RUNNER = ROOT / "scripts/_ssh_preflight_runner.py"
WAVE1_SPECS = [
    ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v4.json",
    ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v5.json",
    ROOT / "configs/construct_benchmark/constructs/source_reliability_v3.json",
    ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v4.json",
]
RUN_CONFIGS = {
    "mistral": ROOT
    / "configs/construct_benchmark/run_configs/wave1_four_construct_mistral_model_preflight_repaired_v4.json",
    "qwen": ROOT
    / "configs/construct_benchmark/run_configs/wave1_four_construct_qwen_model_preflight_repaired_v4.json",
}
INVENTORY = ROOT / "results/benchmark/prompt_inventories/wave1_preflight_v4_luna_v2/combined.csv"


def _load_stage_module():
    module_name = "stage_model_steering_preflight_fixture"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_revision(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["model"]["revision"])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_bundle(tmp_path: Path, alias: str) -> tuple[Path, Path, dict]:
    stage = _load_stage_module()
    from construct_benchmark.config import load_construct_specs, load_run_config
    from construct_benchmark.manifests import canonical_hash

    run_config_path = RUN_CONFIGS[alias]
    run_config = load_run_config(run_config_path)
    specs = load_construct_specs(WAVE1_SPECS)
    source_root = tmp_path / alias / "source"
    source_root.mkdir(parents=True)
    plan_root = source_root / "plans"
    array_root = source_root / "arrays"
    inventory_hash = _sha(INVENTORY)
    config_hash = canonical_hash(run_config.to_mapping())
    spec_entries = []
    for spec_path in WAVE1_SPECS:
        construct_id = stage.load_construct_spec(spec_path).construct_id
        spec_entries.append(
            {
                "construct_id": construct_id,
                "path": str(spec_path),
                "sha256": _sha(spec_path),
                "construct_spec_hash": canonical_hash(specs[construct_id].to_mapping()),
            }
        )

    plans = []
    for construct_index, construct_id in enumerate(run_config.construct_ids):
        construct_dir = array_root / construct_id
        construct_dir.mkdir(parents=True)
        array_paths: dict[str, Path] = {}
        layers = [int(layer) for layer in run_config.activation["layers"]]
        roles = ("target", "shuffled", "random_00", "random_01", "random_02") + tuple(
            f"track_{layer}" for layer in layers[1:]
        )
        for index, role in enumerate(roles):
            path = construct_dir / f"{role}.npy"
            np.save(path, np.arange(8, dtype=np.float16) + construct_index + index + 1)
            array_paths[role] = path
        tracking = {
            str(layers[0]): {
                "layer": layers[0],
                "direction_id": f"{construct_id}__injected_direction__layer_{layers[0]}",
                "path": str(array_paths["target"]),
                "source": "injection_direction_train_only",
                "role": "injection_immediate",
                "source_split": "direction_train",
                "direction_sha256": _sha(array_paths["target"]),
                "calibration": {"projection_scale": 1.0},
            },
            str(layers[1]): {
                "layer": layers[1],
                "direction_id": f"{construct_id}__construct_state__layer_{layers[1]}",
                "path": str(array_paths[f"track_{layers[1]}"]),
                "source": "construct_state_direction_train_only",
                "role": "independent_later_layer_readout",
                "source_split": "direction_train",
                "direction_sha256": _sha(array_paths[f"track_{layers[1]}"]),
                "calibration": {"projection_scale": 1.0},
            },
            str(layers[2]): {
                "layer": layers[2],
                "direction_id": f"{construct_id}__construct_state__layer_{layers[2]}",
                "path": str(array_paths[f"track_{layers[2]}"]),
                "source": "construct_state_direction_train_only",
                "role": "independent_later_layer_readout",
                "source_split": "direction_train",
                "direction_sha256": _sha(array_paths[f"track_{layers[2]}"]),
                "calibration": {"projection_scale": 1.0},
            },
        }
        plan = {
            "schema_version": run_config.schema_version,
            "plan_type": "construct_steering_conditions",
            "run_id": run_config.run_id,
            "mode": "test",
            "purpose": "model_behavior_accessibility_preflight",
            "confirmatory": False,
            "construct_id": construct_id,
            "model": dict(run_config.model),
            "candidate_layers": list(run_config.activation["layers"]),
            "layer": layers[0],
            "tracking_layers": layers,
            "tracking_directions": tracking,
            "activation_site": "resid_post",
            "position_mode": "last",
            "intervention_timing": "prefill_only",
            "calibration": {"projection_scale": 1.0},
            "direction_paths": {
                "target": str(array_paths["target"]),
                "shuffled": str(array_paths["shuffled"]),
                "random": [str(array_paths[f"random_{index:02d}"]) for index in range(3)],
            },
            "condition_count": 3,
            "conditions": [
                {"condition_id": f"{construct_id}__target", "prompt_id": f"{construct_id}__item", "direction_kind": "target", "direction_index": 0, "dose": 0.0},
                {"condition_id": f"{construct_id}__shuffled", "prompt_id": f"{construct_id}__item", "direction_kind": "shuffled", "direction_index": 0, "dose": 1.0},
                {"condition_id": f"{construct_id}__random", "prompt_id": f"{construct_id}__item", "direction_kind": "random", "direction_index": 0, "dose": 1.0},
            ],
            "provenance": {
                "run_config_hash": config_hash,
                "construct_spec_hash": canonical_hash(specs[construct_id].to_mapping()),
                "prompt_inventory_sha256": inventory_hash,
                "direction_sha256": _sha(array_paths["target"]),
                "control_direction_hashes": {
                    "shuffled": _sha(array_paths["shuffled"]),
                    "random": [_sha(array_paths[f"random_{index:02d}"]) for index in range(3)],
                },
                "tracking_direction_hashes": {
                    layer: entry["direction_sha256"] for layer, entry in tracking.items()
                },
            },
        }
        plan_path = plan_root / f"{construct_id}.json"
        _write_json(plan_path, plan)
        plans.append({"construct_id": construct_id, "path": str(plan_path), "sha256": _sha(plan_path)})

    manifest_path = tmp_path / alias / "source_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "0.1.0",
            "manifest_type": "model_steering_preflight_source_bundle",
            "source_root": str(source_root),
            "model": {
                "alias": alias,
                "model_id": run_config.model["model_id"],
                "revision": run_config.model["revision"],
            },
            "run_config": {
                "path": str(run_config_path),
                "sha256": _sha(run_config_path),
                "run_config_hash": config_hash,
            },
            "prompt_inventory": {"path": str(INVENTORY), "sha256": inventory_hash},
            "construct_specs": spec_entries,
            "plans": plans,
        },
    )
    return manifest_path, tmp_path / alias / "staged", run_config.to_mapping()


@pytest.mark.parametrize("alias", ["mistral", "qwen"])
def test_stage_bundle_rebases_all_v4_direction_controls_and_is_idempotent(tmp_path: Path, alias: str) -> None:
    stage = _load_stage_module()
    manifest, output, _ = _fixture_bundle(tmp_path, alias)
    staged = stage.stage_steering_bundle(manifest, output)
    assert staged["model"]["alias"] == alias
    assert staged["candidate_count"] == 4
    assert staged["policy"]["synthetic_directions"] is False
    assert staged["run_config"]["run_config_hash"]
    for entry in staged["plans"]:
        plan = json.loads(Path(entry["staged_path"]).read_text(encoding="utf-8"))
        for raw in stage._plan_artifact_paths(plan, plan_label="fixture"):
            path = Path(raw[1])
            assert path.is_absolute()
            assert output.resolve() in path.parents
            assert path.is_file()
    assert stage.validate_staged_bundle(output)["candidate_count"] == 4
    assert stage.stage_steering_bundle(manifest, output) == staged


@pytest.mark.parametrize("alias", ["mistral", "qwen"])
def test_staged_bundle_is_discoverable_as_exactly_four_construct_plans(tmp_path: Path, alias: str, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = _load_stage_module()
    manifest, output, _ = _fixture_bundle(tmp_path, alias)
    stage.stage_steering_bundle(manifest, output)

    values = {
        "RSC_RUN_ID": "fixture-run",
        "RSC_MODEL_ALIAS": alias,
        "RSC_MODEL_ID": "Qwen/Qwen3.8-27B" if alias == "qwen" else "mistralai/Mistral-Small-24B-Instruct-2501",
        "RSC_MODEL_REVISION": load_revision(RUN_CONFIGS[alias]),
        "RSC_EXPECTED_REPO_SHA": "fixture-sha",
        "RSC_REPO_URL": "https://example.invalid/repo.git",
        "RSC_WORK_ROOT": str(tmp_path / "runner"),
        "RSC_STORAGE_KIND": "ephemeral_container_disk",
        "RSC_EXPECTED_STORAGE_GB": "160",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(f"runner_fixture_{alias}", RUNNER)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.CHECKOUT = ROOT
    runner.RUN_ROOT = tmp_path / "runner" / "fixture-run" / alias
    runner.VOLUME = tmp_path / "runner"
    runner.STEERING_PLAN_ROOT_ENV = "RSC_STEERING_PLAN_ROOT"
    monkeypatch.setenv(runner.STEERING_PLAN_ROOT_ENV, str(output))
    config = RUN_CONFIGS[alias]
    report_path = tmp_path / alias / "discovery.json"
    report = runner.discover_plans(config, WAVE1_SPECS, report_path)
    assert report["pass"] is True
    assert report["candidate_count"] == 4
    assert report["missing"] == []
    assert report["duplicates"] == {}
    assert set(report["selected"]) == {
        "realization_account_closure",
        "evidence_diagnosticity",
        "source_reliability",
        "persistence_continuation",
    }


def test_stage_bundle_rejects_model_revision_mismatch(tmp_path: Path) -> None:
    stage = _load_stage_module()
    manifest, output, _ = _fixture_bundle(tmp_path, "qwen")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["model"]["revision"] = "stale-revision"
    _write_json(manifest, payload)
    with pytest.raises(stage.BundleValidationError, match="model metadata does not match"):
        stage.stage_steering_bundle(manifest, output)


def test_stage_bundle_rejects_non_train_direction_provenance(tmp_path: Path) -> None:
    stage = _load_stage_module()
    manifest, output, _ = _fixture_bundle(tmp_path, "qwen")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    first_plan = Path(payload["plans"][0]["path"])
    plan = json.loads(first_plan.read_text(encoding="utf-8"))
    plan["tracking_directions"]["16"]["source_split"] = "direction_heldout"
    _write_json(first_plan, plan)
    payload["plans"][0]["sha256"] = _sha(first_plan)
    _write_json(manifest, payload)
    with pytest.raises(stage.BundleValidationError, match="direction_train"):
        stage.stage_steering_bundle(manifest, output)


def test_stage_bundle_refuses_partial_nonempty_output(tmp_path: Path) -> None:
    stage = _load_stage_module()
    manifest, output, _ = _fixture_bundle(tmp_path, "qwen")
    output.mkdir(parents=True)
    (output / "unrelated.txt").write_text("do not replace\n", encoding="utf-8")
    with pytest.raises(stage.BundleValidationError, match="non-empty directory"):
        stage.stage_steering_bundle(manifest, output)
