from __future__ import annotations

import copy
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_preflight_artifact_index import file_sha256, validate_index


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "configs/construct_benchmark/preflight_campaigns/waves1_4_preflight_artifact_index_v1.json"
WAVE1_INVENTORY_PATH = ROOT / "results/benchmark/prompt_inventories/wave1_preflight_v4_luna_v2/combined.csv"
WAVE1_SELECTION_PATHS = {
    "mistral": ROOT / "results/benchmark/model_preflight_v4/wave1_preflight_v4_luna_v2_normalized_v1_mistral_selection.json",
    "qwen": ROOT / "results/benchmark/model_preflight_v4/wave1_preflight_v4_luna_v2_normalized_v1_qwen_selection.json",
}
WAVE1_NON_COLLATERAL_DIGESTS = {
    "mistral": "ea13eba099789c68a0d1b787fea7db79097546bef6afa16e7ae08fa4fef99333",
    "qwen": "0d7cf97db831f037babd4767a538859af719fc94785dda63fe71a5ac6747791e",
}
WAVE1_COLLATERAL_SWAPS = {
    "mistral": {
        "evidence_diagnosticity": (
            "evidence_diagnosticity_collateral_v1__evidence_diagnosticity__luna__collateral_eval__part_001__item_13",
            "evidence_diagnosticity_collateral_v1__evidence_diagnosticity__luna__collateral_eval__part_001__item_02",
        ),
        "realization_account_closure": (
            "realization_account_closure_collateral_v1__realization_account_closure__luna__collateral_eval__part_001__variant_20",
            "realization_account_closure_collateral_v1__realization_account_closure__luna__collateral_eval__part_002__collateral_11",
        ),
        "source_reliability": (
            "source_reliability_collateral_v1__source_reliability__luna__collateral_eval__part_001__v06",
            "source_reliability_collateral_v1__source_reliability__luna__collateral_eval__part_001__v15",
        ),
    },
    "qwen": {
        "evidence_diagnosticity": (
            "evidence_diagnosticity_collateral_v1__evidence_diagnosticity__luna__collateral_eval__part_001__item_17",
            "evidence_diagnosticity_collateral_v1__evidence_diagnosticity__luna__collateral_eval__part_001__item_14",
        ),
        "persistence_continuation": (
            "persistence_continuation_collateral_v1__persistence_continuation__luna__collateral_eval__part_002__item_03",
            "persistence_continuation_collateral_v1__persistence_continuation__luna__collateral_eval__part_001__v02",
        ),
        "realization_account_closure": (
            "realization_account_closure_collateral_v1__realization_account_closure__luna__collateral_eval__part_001__variant_14",
            "realization_account_closure_collateral_v1__realization_account_closure__luna__collateral_eval__part_001__variant_01",
        ),
        "realization_account_closure__second": (
            "realization_account_closure_collateral_v1__realization_account_closure__luna__collateral_eval__part_001__variant_20",
            "realization_account_closure_collateral_v1__realization_account_closure__luna__collateral_eval__part_001__variant_13",
        ),
    },
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ready_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    inventory = tmp_path / "results" / "inventory.csv"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text("prompt_id,construct_id\nitem-1,c1\n", encoding="utf-8")
    inventory_hash = file_sha256(inventory)

    manifest = inventory.parent / "inventory_manifest.json"
    _write_json(
        manifest,
        {
            "manifest_type": "wave_execution_prompt_inventory",
            "status": "frozen",
            "frozen": True,
            "wave": 1,
            "construct_ids": ["c1"],
            "output_path": "combined.csv",
            "output_sha256": inventory_hash,
        },
    )

    gate = tmp_path / "configs" / "gate.json"
    _write_json(
        gate,
        {
            "gate_id": "gate-c1",
            "models": [{"model_id": "model-c1", "revision": "revision-c1"}],
            "construct_ids": ["c1"],
        },
    )
    gate_hash = file_sha256(gate)

    selection = tmp_path / "results" / "selection.json"
    _write_json(
        selection,
        {
            "manifest_type": "model_behavior_accessibility_preflight_selection",
            "model": {"model_id": "model-c1", "revision": "revision-c1"},
            "construct_ids": ["c1"],
            "source_inventory": "results/inventory.csv",
            "source_inventory_sha256": inventory_hash,
            "gate_id": "gate-c1",
            "gate_config_sha256": gate_hash,
        },
    )

    run_config = tmp_path / "configs" / "run.json"
    _write_json(
        run_config,
        {
            "run_id": "run-c1",
            "run_kind": "preflight_only",
            "construct_ids": ["c1"],
            "model": {"model_id": "model-c1", "revision": "revision-c1"},
            "preflight": {"preflight_only": True, "gate_id": "gate-c1"},
        },
    )

    index_payload: dict[str, object] = {
        "schema_version": "0.1.0",
        "index_type": "model_preflight_artifact_index",
        "index_id": "fixture",
        "execution_allowed": False,
        "entries": [
            {
                "wave": 1,
                "model": {"alias": "fixture", "model_id": "model-c1", "revision": "revision-c1"},
                "construct_ids": ["c1"],
                "artifacts": {
                    "inventory": {
                        "status": "ready",
                        "path": "results/inventory.csv",
                        "sha256": inventory_hash,
                        "manifest": {
                            "status": "ready",
                            "path": "results/inventory_manifest.json",
                            "sha256": file_sha256(manifest),
                        },
                    },
                    "selection": {
                        "status": "ready",
                        "path": "results/selection.json",
                        "sha256": file_sha256(selection),
                    },
                    "gate": {
                        "status": "ready",
                        "id": "gate-c1",
                        "path": "configs/gate.json",
                        "sha256": gate_hash,
                    },
                    "run_config": {
                        "status": "ready",
                        "id": "run-c1",
                        "path": "configs/run.json",
                        "sha256": file_sha256(run_config),
                    },
                },
            }
        ],
    }
    index_path = tmp_path / "configs" / "preflight_index.json"
    _write_json(index_path, index_payload)
    return index_path, index_payload


def test_canonical_index_is_complete_and_hash_verified() -> None:
    report = validate_index(INDEX_PATH, repo_root=ROOT)

    assert report["ready"] is True
    assert report["summary"] == {
        "entry_count": 8,
        "ready_entries": 8,
        "blocked_entries": 0,
        "issue_count": 0,
    }
    assert all(entry["status"] == "ready" for entry in report["entries"])
    assert report["issues"] == []


def test_wave1_collateral_rebalance_is_balanced_and_inventory_backed() -> None:
    with WAVE1_INVENTORY_PATH.open(newline="", encoding="utf-8") as handle:
        inventory = {row["prompt_id"]: row for row in csv.DictReader(handle)}

    for alias, selection_path in WAVE1_SELECTION_PATHS.items():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        assert selection["selection_informed_by_outcomes"] is False
        non_collateral = {
            construct_id: {
                split: selection["selected"][construct_id][split]["prompt_ids"]
                for split in ("behavior_eval", "steering_eval")
            }
            for construct_id in sorted(selection["selected"])
        }
        digest = hashlib.sha256(
            json.dumps(non_collateral, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert digest == WAVE1_NON_COLLATERAL_DIGESTS[alias]

        for construct_id, pools in selection["selected"].items():
            collateral = pools["collateral_eval"]
            prompt_ids = collateral["prompt_ids"]
            assert collateral["item_count"] == 16
            assert len(prompt_ids) == 16
            assert len(set(prompt_ids)) == 16
            rows = [inventory[prompt_id] for prompt_id in prompt_ids]
            assert all(
                row["construct_id"] == construct_id
                and row["split"] == "collateral_eval"
                and row["prompt_role"] == "collateral"
                for row in rows
            )
            assert Counter(row["correct_option"] for row in rows) == Counter({"1": 8, "2": 8})
            assert len({row["prompt_family"] for row in rows}) == 1
            assert len({row["task_id"] for row in rows}) == 1

        for swap_key, (removed, added) in WAVE1_COLLATERAL_SWAPS[alias].items():
            construct_id = swap_key.split("__second", 1)[0]
            prompt_ids = set(selection["selected"][construct_id]["collateral_eval"]["prompt_ids"])
            assert removed not in prompt_ids
            assert added in prompt_ids


def test_pending_artifact_blocks_readiness(tmp_path: Path) -> None:
    index_path, payload = _ready_fixture(tmp_path)
    payload = copy.deepcopy(payload)
    payload["entries"][0]["artifacts"]["run_config"] = {
        "status": "pending",
        "path": None,
        "sha256": None,
        "reason": "Waiting for the finalized run config.",
    }
    _write_json(index_path, payload)

    report = validate_index(index_path, repo_root=tmp_path)

    assert report["ready"] is False
    assert any(issue["code"] == "pending" for issue in report["issues"])


def test_complete_fixture_passes_hash_and_cross_reference_checks(tmp_path: Path) -> None:
    index_path, _ = _ready_fixture(tmp_path)

    report = validate_index(index_path, repo_root=tmp_path)

    assert report["ready"] is True
    assert report["issues"] == []
    assert report["entries"][0]["status"] == "ready"
    assert report["entries"][0]["artifacts"] == {
        "inventory": "pass",
        "selection": "pass",
        "gate": "pass",
        "run_config": "pass",
    }


@pytest.mark.parametrize(
    ("replacement", "expected_code"),
    (("/tmp/outside.csv", "absolute_path"), ("../outside.csv", "path_traversal")),
)
def test_artifact_paths_must_be_repo_relative(tmp_path: Path, replacement: str, expected_code: str) -> None:
    index_path, payload = _ready_fixture(tmp_path)
    payload = copy.deepcopy(payload)
    payload["entries"][0]["artifacts"]["inventory"]["path"] = replacement
    _write_json(index_path, payload)

    report = validate_index(index_path, repo_root=tmp_path)

    assert report["ready"] is False
    assert any(issue["code"] == expected_code for issue in report["issues"])


def test_hash_mismatch_and_marked_failed_root_block_readiness(tmp_path: Path) -> None:
    index_path, payload = _ready_fixture(tmp_path)
    payload = copy.deepcopy(payload)
    payload["entries"][0]["artifacts"]["inventory"]["sha256"] = "0" * 64
    _write_json(index_path, payload)

    hash_report = validate_index(index_path, repo_root=tmp_path)

    failed_root = tmp_path / "results" / "failed-run"
    failed_root.mkdir(parents=True)
    (failed_root / "FAILED.json").write_text("{}\n", encoding="utf-8")
    payload["entries"][0]["artifacts"]["inventory"]["path"] = "results/failed-run/inventory.csv"
    _write_json(index_path, payload)

    report = validate_index(index_path, repo_root=tmp_path)

    assert report["ready"] is False
    assert any(issue["code"] == "hash_mismatch" for issue in hash_report["issues"])
    assert any(issue["code"] == "forbidden_root" for issue in report["issues"])


def test_inventory_manifest_and_selection_cross_references_are_checked(tmp_path: Path) -> None:
    index_path, payload = _ready_fixture(tmp_path)
    payload = copy.deepcopy(payload)
    selection_path = tmp_path / "results" / "selection.json"
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_payload["source_inventory"] = "results/inventory_manifest.json"
    _write_json(selection_path, selection_payload)
    payload["entries"][0]["artifacts"]["selection"]["sha256"] = file_sha256(selection_path)
    _write_json(index_path, payload)

    report = validate_index(index_path, repo_root=tmp_path)

    assert report["ready"] is False
    assert any(issue["code"] == "cross_reference_mismatch" for issue in report["issues"])
