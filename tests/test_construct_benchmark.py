from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from construct_benchmark.config import (
    load_analysis_spec,
    load_construct_specs,
    load_run_config,
)
from construct_benchmark.manifests import build_run_plan, load_run_plan
from construct_benchmark.prompts import PromptRecord, combine_prompt_files, load_prompt_records, validate_prompt_records, write_prompt_records
from construct_benchmark.schemas import ConstructSpec, RunConfig
from activation_analysis.vector_analysis import (
    PromptActivation,
    build_directions_by_construct,
    build_pair_directions,
)
from scripts.build_activation_vectors import _validated_include_splits, build_argument_parser


ROOT = Path(__file__).resolve().parents[1]
CONSTRUCT_PATHS = [
    ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json",
    ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v1.json",
]
RUN_CONFIG_PATH = ROOT / "configs/construct_benchmark/run_configs/two_construct_smoke_v1.json"
ANALYSIS_PATH = ROOT / "configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json"


def _configs() -> tuple[dict[str, ConstructSpec], RunConfig, object]:
    specs = load_construct_specs(CONSTRUCT_PATHS)
    run_config = load_run_config(RUN_CONFIG_PATH)
    analysis_spec = load_analysis_spec(ANALYSIS_PATH)
    return specs, run_config, analysis_spec


def _prompt_inventory(specs: dict[str, ConstructSpec]) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for construct_id, spec in specs.items():
        for split in spec.paired_splits:
            for condition_id in spec.condition_ids:
                records.append(
                    PromptRecord(
                        prompt_id=f"{construct_id}__{split}__{condition_id}",
                        construct_id=construct_id,
                        split=split,
                        prompt_role="probe",
                        prompt_text=f"{construct_id} {split} {condition_id} scenario.",
                        condition_id=condition_id,
                        pair_id=f"{construct_id}__{split}__pair_000",
                        pair_role=condition_id,
                        prompt_family=f"{construct_id}_probe",
                    )
                )
        for split, prompt_role in (
            ("behavior_eval", "behavior"),
            ("steering_eval", "steering"),
            ("calibration", "calibration"),
        ):
            records.append(
                PromptRecord(
                    prompt_id=f"{construct_id}__{split}__neutral",
                    construct_id=construct_id,
                    split=split,
                    prompt_role=prompt_role,
                    prompt_text=f"{construct_id} {split} independent task.",
                    condition_id="neutral",
                    prompt_family=f"{construct_id}_behavior",
                    task_id=spec.independent_behavior_task["task_id"],
                    expected_output_format=spec.independent_behavior_task["response_format"],
                    parser_id=spec.parsing_rules["parser_id"],
                )
            )
    return records


def test_two_construct_configs_are_valid_and_share_execution() -> None:
    specs, run_config, analysis_spec = _configs()
    records = _prompt_inventory(specs)

    summary = validate_prompt_records(records, specs)
    plan = build_run_plan(
        run_config,
        specs,
        analysis_spec,
        prompt_inventory_path="results/benchmark/two_construct_smoke_v1/prompts/combined.csv",
        prompt_records=records,
    )

    assert summary["construct_ids"] == sorted(specs)
    assert plan["construct_count"] == 2
    assert plan["shared_execution"]["construct_ids"] == list(run_config.construct_ids)
    assert [stage["scope"] for stage in plan["execution_graph"][:2]] == ["shared", "shared"]
    assert len([stage for stage in plan["execution_graph"] if stage["scope"] == "construct"]) == 10
    baseline_stages = [
        stage for stage in plan["execution_graph"] if stage["stage_id"].startswith("evaluate_behavior_baseline:")
    ]
    steered_stages = [
        stage for stage in plan["execution_graph"] if stage["stage_id"].startswith("evaluate_behavior_steered:")
    ]
    assert len(baseline_stages) == len(steered_stages) == 2
    assert all(stage["prompt_only_baseline"] for stage in baseline_stages)
    assert all(stage["intervention"] == "steered" for stage in steered_stages)
    assert {entry["construct_id"] for entry in plan["constructs"]} == set(specs)
    assert all(
        stage["group_key"] == ["construct_id", "pair_id"]
        for stage in plan["execution_graph"]
        if stage["stage_id"].startswith("build_direction:")
    )


def test_same_execution_plan_scales_to_four_constructs() -> None:
    specs, run_config, analysis_spec = _configs()
    for construct_id, family in (
        ("authority_verification", "social"),
        ("persistence_abandonment", "agentic"),
    ):
        payload = copy.deepcopy(next(iter(specs.values())).to_mapping())
        payload["construct_id"] = construct_id
        payload["family"] = family
        payload["title"] = construct_id.replace("_", " ").title()
        payload["description"] = "Synthetic fixture used to verify four-construct scheduling."
        specs[construct_id] = ConstructSpec.from_mapping(payload)

    run_payload = run_config.to_mapping()
    run_payload["construct_ids"] = list(specs)
    four_construct_config = RunConfig.from_mapping(run_payload)
    plan = build_run_plan(four_construct_config, specs, analysis_spec)

    assert plan["construct_count"] == 4
    assert len(plan["shared_execution"]["construct_ids"]) == 4
    assert len([stage for stage in plan["execution_graph"] if stage["scope"] == "shared"]) == 2
    assert len([stage for stage in plan["execution_graph"] if stage["scope"] == "construct"]) == 20


def test_prompt_inventory_round_trips_and_combines_without_id_collisions(tmp_path: Path) -> None:
    specs, _, _ = _configs()
    records = _prompt_inventory(specs)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    combined_path = tmp_path / "combined.csv"
    write_prompt_records(records[: len(records) // 2], first_path)
    write_prompt_records(records[len(records) // 2 :], second_path)

    assert combine_prompt_files([first_path, second_path], combined_path) == len(records)
    combined = load_prompt_records(combined_path)
    assert [record.prompt_id for record in combined] == [record.prompt_id for record in records]
    assert combined[0].construct_id in specs


def test_prompt_validation_rejects_duplicate_global_ids() -> None:
    specs, _, _ = _configs()
    records = _prompt_inventory(specs)
    duplicate = copy.copy(records[-1])
    duplicate = PromptRecord(
        prompt_id=records[0].prompt_id,
        construct_id=duplicate.construct_id,
        split=duplicate.split,
        prompt_role=duplicate.prompt_role,
        prompt_text=duplicate.prompt_text,
        condition_id=duplicate.condition_id,
    )

    with pytest.raises(ValueError, match="Duplicate prompt_id"):
        validate_prompt_records([*records, duplicate], specs)


def test_prompt_validation_rejects_role_mismatch_and_missing_task_fields() -> None:
    specs, _, _ = _configs()
    records = _prompt_inventory(specs)
    bad_role = PromptRecord(
        prompt_id="bad_role",
        construct_id="realization_account_closure",
        split="direction_train",
        prompt_role="behavior",
        prompt_text="bad role",
        condition_id="open_pending",
        pair_id="bad_pair",
        pair_role="open_pending",
        prompt_family="realization_account_closure_probe",
    )
    with pytest.raises(ValueError, match="requires role='probe'"):
        validate_prompt_records([*records, bad_role], specs)

    bad_task = PromptRecord(
        prompt_id="bad_task",
        construct_id="realization_account_closure",
        split="behavior_eval",
        prompt_role="behavior",
        prompt_text="missing parser fields",
        condition_id="neutral",
        prompt_family="realization_account_closure_behavior",
    )
    with pytest.raises(ValueError, match="role-specific fields"):
        validate_prompt_records([*records, bad_task], specs)


def test_structured_prompt_metadata_preserves_decoded_types(tmp_path: Path) -> None:
    record = PromptRecord(
        prompt_id="structured_metadata",
        construct_id="realization_account_closure",
        split="behavior_eval",
        prompt_role="behavior",
        prompt_text="A behavior prompt.",
        condition_id="neutral",
        prompt_family="realization_account_closure_behavior",
        task_id="realization_risk_choice_v1",
        expected_output_format="two_integers_on_separate_lines",
        parser_id="two_integers_risk_choice_v1",
        metadata={"nested": [1, 2]},
    )
    path = tmp_path / "structured.csv"
    write_prompt_records([record], path)
    loaded = load_prompt_records(path)
    assert loaded[0].metadata["nested"] == [1, 2]


def test_schema_version_is_rejected_when_unsupported() -> None:
    specs, _, _ = _configs()
    payload = specs["realization_account_closure"].to_mapping()
    payload["schema_version"] = "999.0"
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        ConstructSpec.from_mapping(payload)


def test_run_plan_loader_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    specs, run_config, analysis_spec = _configs()
    plan = build_run_plan(run_config, specs, analysis_spec)
    plan["schema_version"] = "999.0"
    path = tmp_path / "invalid_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        load_run_plan(path)


def test_direction_builder_defaults_to_train_split() -> None:
    args = build_argument_parser().parse_args(["--activation-run", "run", "--output-dir", "out"])
    assert args.include_splits == "direction_train"
    assert _validated_include_splits(args.include_splits, allow_nontrain=args.allow_nontrain_splits) == {
        "direction_train"
    }
    with pytest.raises(ValueError, match="train-only"):
        _validated_include_splits("direction_validation", allow_nontrain=False)
    assert _validated_include_splits("direction_validation", allow_nontrain=True) == {
        "direction_validation"
    }


def test_run_config_uses_canonical_intervention_timing() -> None:
    _, run_config, _ = _configs()
    assert run_config.steering["intervention_timing"] == "prefill_only"
    payload = run_config.to_mapping()
    payload["steering"]["intervention_timing"] = "behavior_prompt"
    with pytest.raises(ValueError, match="intervention_timing must be one of"):
        RunConfig.from_mapping(payload)


def test_combined_activation_directions_are_never_pooled_across_constructs() -> None:
    activations = []
    for construct_id, positive_role, negative_role in (
        ("realization_account_closure", "closed_settled", "open_pending"),
        ("evidence_diagnosticity", "high_diagnosticity", "low_diagnosticity"),
    ):
        activations.extend(
            [
                PromptActivation(
                    prompt_id=f"{construct_id}__positive",
                    metadata={
                        "construct_id": construct_id,
                        "pair_id": f"{construct_id}__pair_000",
                        "pair_role": positive_role,
                        "split": "direction_train",
                    },
                    vector=np.array([2.0, 0.0], dtype=np.float32),
                    token_count=1,
                ),
                PromptActivation(
                    prompt_id=f"{construct_id}__negative",
                    metadata={
                        "construct_id": construct_id,
                        "pair_id": f"{construct_id}__pair_000",
                        "pair_role": negative_role,
                        "split": "direction_train",
                    },
                    vector=np.array([0.0, 1.0], dtype=np.float32),
                    token_count=1,
                ),
            ]
        )

    with pytest.raises(ValueError, match="Multiple construct_id"):
        build_pair_directions(activations, positive_role="closed_settled", negative_role="open_pending")

    directions = build_directions_by_construct(
        activations,
        positive_roles={
            "realization_account_closure": "closed_settled",
            "evidence_diagnosticity": "high_diagnosticity",
        },
        negative_roles={
            "realization_account_closure": "open_pending",
            "evidence_diagnosticity": "low_diagnosticity",
        },
    )
    assert set(directions) == {"realization_account_closure", "evidence_diagnosticity"}
    assert all(len(rows) == 1 and direction is not None for rows, direction in directions.values())
