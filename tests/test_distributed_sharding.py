from __future__ import annotations

import json
from pathlib import Path

import pytest

from construct_benchmark.sharding import (
    build_shard_plan,
    validate_shard_manifests,
    write_shard_outputs,
)


CONSTRUCTS = ("construct_a", "construct_b", "construct_c", "construct_d")


def _inventory_rows(*, version_by_construct: dict[str, str] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    versions = version_by_construct or {}
    for construct_id in CONSTRUCTS:
        for unit_index in range(3):
            pair_id = f"{construct_id}_pair_{unit_index}"
            cell_id = f"{construct_id}_cell_{unit_index}"
            for condition in ("negative", "positive"):
                rows.append(
                    {
                        "request_id": f"{pair_id}_{condition}",
                        "construct_id": construct_id,
                        "split": "direction_train",
                        "prompt_role": "probe",
                        "pair_id": pair_id,
                        "pair_role": condition,
                        "factor_cell_id": cell_id,
                        "stage_id": "probe_train",
                        "observation_id": f"{pair_id}_{condition}_observation",
                        "prompt_version": versions.get(construct_id, "v1"),
                    }
                )
    return rows


def _plan(rows: list[dict[str, object]], *, workers: int) -> object:
    return build_shard_plan(
        rows,
        worker_count=workers,
        seed=41,
        run_config_hash="run-config-sha256",
        run_mode="test",
        confirmatory=False,
    )


def test_three_four_and_five_worker_layouts_are_construct_pure() -> None:
    rows = _inventory_rows()
    for worker_count in (3, 4, 5):
        plan = _plan(rows, workers=worker_count)
        assert len(plan.manifests) == (5 if worker_count == 5 else 4)
        assert plan.worker_count == worker_count
        assert all(len(manifest["construct_ids"]) == 1 for manifest in plan.manifests.values())
        assert set(plan.assignment) == {str(row["request_id"]) for row in rows}
        if worker_count == 3:
            assert sorted(len(shards) for shards in plan.worker_schedule.values()) == [1, 1, 2]
        elif worker_count == 4:
            assert all(len(shards) == 1 for shards in plan.worker_schedule.values())
        else:
            split_shards = [
                manifest for manifest in plan.manifests.values() if "_part_" in str(manifest["shard_id"])
            ]
            assert len(split_shards) == 2
            assert len({manifest["construct_id"] for manifest in split_shards}) == 1


def test_three_worker_shorthand_promotes_to_four_physical_shards() -> None:
    plan = build_shard_plan(
        _inventory_rows(),
        shard_count=3,
        worker_count=3,
        seed=41,
        run_config_hash="run-config-sha256",
    )
    assert len(plan.manifests) == 4
    assert plan.worker_count == 3
    assert all(manifest["shard_count"] == 4 for manifest in plan.manifests.values())


def test_assignment_is_repeatable_and_preserves_pairs_and_cells() -> None:
    rows = _inventory_rows()
    first = _plan(rows, workers=5)
    second = _plan(list(reversed(rows)), workers=5)
    assert first.assignment == second.assignment
    assert first.to_mapping() == second.to_mapping()

    for construct_id in CONSTRUCTS:
        request_to_shard = {
            request_id: shard_id
            for request_id, shard_id in first.assignment.items()
            if request_id.startswith(construct_id)
        }
        for unit_index in range(3):
            pair_requests = [
                f"{construct_id}_pair_{unit_index}_{condition}"
                for condition in ("negative", "positive")
            ]
            assert len({request_to_shard[request_id] for request_id in pair_requests}) == 1
            cell_requests = [
                row["request_id"]
                for row in rows
                if row["construct_id"] == construct_id and row["factor_cell_id"] == f"{construct_id}_cell_{unit_index}"
            ]
            assert len({first.assignment[str(request_id)] for request_id in cell_requests}) == 1


def test_materialized_shards_are_hashable_immutable_and_complete(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _inventory_rows()),
        encoding="utf-8",
    )
    plan = build_shard_plan(inventory, worker_count=3, seed=9, run_config_hash="cfg")
    output_dir = tmp_path / "shards"
    report = write_shard_outputs(plan, output_dir, inventory_suffix=".jsonl")
    assert report["shard_count"] == 4
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_dir.glob("*.manifest.json"))
    ]
    summary = validate_shard_manifests(
        manifests,
        expected_shard_count=4,
        expected_worker_count=3,
        expected_request_ids=[row["request_id"] for row in _inventory_rows()],
        expected_observation_ids=[row["observation_id"] for row in _inventory_rows()],
        expected_construct_ids=CONSTRUCTS,
    )
    assert summary["request_ids"] == sorted(row["request_id"] for row in _inventory_rows())
    with pytest.raises(FileExistsError, match="overwrite"):
        write_shard_outputs(plan, output_dir, inventory_suffix=".jsonl")


def test_sharding_rejects_duplicate_missing_unknown_incomplete_and_pooled_ids() -> None:
    rows = _inventory_rows()
    duplicate = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate request"):
        build_shard_plan(duplicate, worker_count=4)
    with pytest.raises(ValueError, match="unknown=.*missing"):
        build_shard_plan(rows, worker_count=4, expected_request_ids=[row["request_id"] for row in rows[:-1]])
    with pytest.raises(ValueError, match="unknown=.*missing"):
        build_shard_plan(rows, worker_count=4, expected_request_ids=[row["request_id"] for row in rows] + ["unknown"])

    incomplete = [row for row in rows if row["request_id"] != "construct_a_pair_0_positive"]
    with pytest.raises(ValueError, match="Incomplete pair"):
        build_shard_plan(incomplete, worker_count=4)

    pooled = [dict(row) for row in rows]
    pooled[0]["pair_id"] = "shared_pair"
    pooled[1]["pair_id"] = "shared_pair"
    pooled[1]["construct_id"] = "construct_b"
    with pytest.raises(ValueError, match="Construct pooling"):
        build_shard_plan(pooled, worker_count=4)


def test_sharding_rejects_version_contamination_and_incompatible_roles() -> None:
    mixed = _inventory_rows(version_by_construct={"construct_d": "v2"})
    with pytest.raises(ValueError, match="v1/v2|versions"):
        build_shard_plan(mixed, worker_count=4)

    incompatible = _inventory_rows()
    incompatible[1]["prompt_role"] = "behavior"
    with pytest.raises(ValueError, match="incompatible prompt role"):
        build_shard_plan(incompatible, worker_count=4)


def test_v2_prompt_inventory_allows_v1_task_and_parser_schema_labels() -> None:
    rows = _inventory_rows(version_by_construct={construct_id: "v2" for construct_id in CONSTRUCTS})
    for row in rows:
        row["task_id"] = "independent_task_v1"
        row["parser_id"] = "parser_schema_v1"
        row["expected_output_format"] = "format_v1"
        row["metadata"] = {
            "task_id": "nested_task_v1",
            "parser_id": "nested_parser_v1",
            "expected_output_format": "nested_format_v1",
        }
        row["metadata_json"] = json.dumps(
            {
                "task_id": "serialized_task_v1",
                "parser_id": "serialized_parser_v1",
                "expected_output_format": "serialized_format_v1",
            }
        )

    plan = _plan(rows, workers=4)

    assert {family for manifest in plan.manifests.values() for family in manifest["version_families"]} == {"v2"}


def test_four_constructs_cannot_be_pooled_into_three_physical_shards_without_worker_slots() -> None:
    with pytest.raises(ValueError, match="construct pooling"):
        build_shard_plan(_inventory_rows(), shard_count=3, seed=1)
