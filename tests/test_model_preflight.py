from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from construct_benchmark.behavior_baseline import select_preflight_behavior_records
from construct_benchmark.config import load_construct_spec
from construct_benchmark.manifests import file_sha256
from construct_benchmark.model_preflight import (
    prepare_selection_manifest,
    validate_preflight,
)
from construct_benchmark.prompts import PromptRecord


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v3.json"


def _records(spec_id: str) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for split, role in (
        ("behavior_eval", "behavior"),
        ("steering_eval", "steering"),
        ("collateral_eval", "collateral"),
    ):
        for index in range(16):
            metadata = (
                {"task_metadata": {"correct_option": 1 if index % 2 == 0 else 2}}
                if split == "collateral_eval"
                else {"task_metadata": {"outcome_valence": "gain" if index % 2 == 0 else "loss"}}
            )
            task = (
                "collateral_factual_choice_realization_account_closure_v1"
                if split == "collateral_eval"
                else "realization_risk_allocation_v2"
            )
            parser = (
                "single_integer_choice_1_or_2_v1"
                if split == "collateral_eval"
                else "single_integer_allocation_0_to_100_v1"
            )
            response = (
                "single_integer_1_or_2"
                if split == "collateral_eval"
                else "single_integer_allocation_0_to_100"
            )
            records.append(
                PromptRecord(
                    prompt_id=f"{split}__{index:02d}",
                    construct_id=spec_id,
                    split=split,
                    prompt_role=role,
                    prompt_text=f"{split} item {index}.",
                    prompt_family=f"{spec_id}_{role}",
                    task_id=task,
                    parser_id=parser,
                    expected_output_format=response,
                    metadata=metadata,
                )
            )
    return records


def _write_output(path: Path, rows: list[dict], manifest_type: str, model: dict) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    manifest = {
        "manifest_type": manifest_type,
        "complete": True,
        "model": model,
        "raw_generations_sha256": file_sha256(path),
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_preflight_selection_is_bounded_and_outcome_independent(tmp_path: Path) -> None:
    spec = load_construct_spec(SPEC_PATH)
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("frozen inventory\n", encoding="utf-8")
    manifest = prepare_selection_manifest(
        _records(spec.construct_id),
        source_inventory=inventory,
        model={"model_id": "mistralai/Mistral-Small-24B-Instruct-2501", "revision": "rev"},
        construct_ids=[spec.construct_id],
    )

    assert manifest["selection_informed_by_outcomes"] is False
    for split in ("behavior_eval", "steering_eval", "collateral_eval"):
        assert manifest["selected"][spec.construct_id][split]["item_count"] == 16


def test_selection_manifest_serializes_repository_paths_relative(tmp_path: Path) -> None:
    spec = load_construct_spec(SPEC_PATH)
    repository_root = tmp_path / "repository"
    inventory = repository_root / "results" / "inventory.csv"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("frozen inventory\n", encoding="utf-8")

    manifest = prepare_selection_manifest(
        _records(spec.construct_id),
        source_inventory=inventory,
        model={
            "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
            "revision": "rev",
        },
        construct_ids=[spec.construct_id],
        repository_root=repository_root,
    )

    assert manifest["source_inventory"] == "results/inventory.csv"
    assert not Path(manifest["source_inventory"]).is_absolute()


def test_preflight_runner_selection_uses_frozen_ids(tmp_path: Path) -> None:
    spec = load_construct_spec(SPEC_PATH)
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("frozen inventory\n", encoding="utf-8")
    model = {
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "revision": "rev",
    }
    records = _records(spec.construct_id)
    selection = prepare_selection_manifest(
        records,
        source_inventory=inventory,
        model=model,
        construct_ids=[spec.construct_id],
    )

    selected, manifest = select_preflight_behavior_records(
        records,
        run_config=SimpleNamespace(
            model=model,
            construct_ids=(spec.construct_id,),
            schema_version="0.1.0",
            run_id="test_preflight_run",
        ),
        construct_specs={spec.construct_id: spec},
        preflight_selection=selection,
        split="behavior_eval",
    )
    assert [record.prompt_id for record in selected] == selection["selected"][spec.construct_id]["behavior_eval"]["prompt_ids"]
    assert manifest["preflight_selection_sha256"] == selection["selection_sha256"]


def test_preflight_requires_complete_usable_behavior_and_steering(tmp_path: Path) -> None:
    spec = load_construct_spec(SPEC_PATH)
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("frozen inventory\n", encoding="utf-8")
    model = {
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "revision": "rev",
    }
    selection = prepare_selection_manifest(
        _records(spec.construct_id),
        source_inventory=inventory,
        model=model,
        construct_ids=[spec.construct_id],
    )
    behavior_rows = []
    collateral_rows = []
    for index in range(16):
        metadata = {"outcome_valence": "gain" if index % 2 == 0 else "loss"}
        behavior_rows.append(
            {
                "record_id": f"behavior_eval__{index:02d}__prompt_only",
                "prompt_id": f"behavior_eval__{index:02d}",
                "construct_id": spec.construct_id,
                "split": "behavior_eval",
                "model": model,
                "parser_id": "single_integer_allocation_0_to_100_v1",
                "task_id": "realization_risk_allocation_v2",
                "task_metadata": metadata,
                "output_text": str(20 + index * 3),
            }
        )
        collateral_rows.append(
            {
                "record_id": f"collateral_eval__{index:02d}__prompt_only",
                "prompt_id": f"collateral_eval__{index:02d}",
                "construct_id": spec.construct_id,
                "split": "collateral_eval",
                "model": model,
                "parser_id": "single_integer_choice_1_or_2_v1",
                "task_id": "collateral_factual_choice_realization_account_closure_v1",
                "task_metadata": {"correct_option": 1},
                "output_text": "1",
            }
        )
    behavior = tmp_path / "behavior.jsonl"
    collateral = tmp_path / "collateral.jsonl"
    _write_output(behavior, behavior_rows, "construct_behavior_output", model)
    _write_output(collateral, collateral_rows, "construct_behavior_output", model)

    steering_rows = []
    for prompt_index in range(16):
        prompt_id = f"steering_eval__{prompt_index:02d}"
        metadata = {"outcome_valence": "gain" if prompt_index % 2 == 0 else "loss"}
        for direction_kind, doses in (
            ("target", (-1.0, 0.0, 1.0)),
            ("shuffled", (0.0,)),
            ("random", (0.0,)),
        ):
            for dose in doses:
                steering_rows.append(
                    {
                        "record_id": f"{prompt_id}__{direction_kind}__{dose}",
                        "prompt_id": prompt_id,
                        "construct_id": spec.construct_id,
                        "model": model,
                        "direction_kind": direction_kind,
                        "dose": dose,
                        "injection_applied": direction_kind == "target" and dose != 0.0,
                        "intervention_timing": "prefill_only",
                        "parser_id": "single_integer_allocation_0_to_100_v1",
                        "task_id": "realization_risk_allocation_v2",
                        "task_metadata": metadata,
                        "output_text": "50",
                    }
                )
    steering = tmp_path / "steering.jsonl"
    _write_output(steering, steering_rows, "construct_steering_output", model)

    report = validate_preflight(
        selection_manifest=selection,
        construct_specs={spec.construct_id: spec},
        behavior_output=behavior,
        collateral_output=collateral,
        steering_outputs={spec.construct_id: steering},
    )

    assert report["release_decision"] == "pass_preflight"
    assert report["constructs"][spec.construct_id]["pass"] is True
