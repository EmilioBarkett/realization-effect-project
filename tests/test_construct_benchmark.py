from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from construct_benchmark.config import (
    load_analysis_spec,
    load_construct_spec,
    load_construct_specs,
    load_run_config,
)
from construct_benchmark.manifests import build_run_plan, load_run_plan
from construct_benchmark.generation import (
    dry_run_summary,
    generate_prompt_records,
    iter_generation_jobs,
    load_generation_plan,
    normalize_probe_prompt_wrapper,
)
from construct_benchmark.prompts import PromptRecord, combine_prompt_files, load_prompt_records, validate_prompt_records, write_prompt_records
from construct_benchmark.registry import ConstructRegistry, load_construct_registry, validate_registry_against_specs
from construct_benchmark.run_modes import select_prompt_records
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
REGISTRY_PATH = ROOT / "configs/construct_benchmark/construct_registry_v1.json"
WAVE1_PLAN_PATHS = [
    ROOT / "configs/construct_benchmark/generation_plans/wave1_realization_account_closure_v1.json",
    ROOT / "configs/construct_benchmark/generation_plans/wave1_evidence_diagnosticity_v1.json",
    ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json",
    ROOT / "configs/construct_benchmark/generation_plans/wave1_persistence_continuation_v1.json",
]
WAVE1_CONSTRUCT_PATHS = [
    ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json",
    ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v1.json",
    ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json",
    ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v1.json",
]
WAVE1_RUN_CONFIG_PATH = ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_smoke_v1.json"
ALL_GENERATION_PLAN_PATHS = sorted(
    (ROOT / "configs/construct_benchmark/generation_plans").glob("wave[1-4]_*.json")
)


def _configs() -> tuple[dict[str, ConstructSpec], RunConfig, object]:
    specs = load_construct_specs(CONSTRUCT_PATHS)
    run_config = load_run_config(RUN_CONFIG_PATH)
    analysis_spec = load_analysis_spec(ANALYSIS_PATH)
    return specs, run_config, analysis_spec


def _mock_generation_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
    del model_id
    payload = json.loads(messages[1]["content"])
    generation_job_id = options["generation_job_id"]
    job_nonce = hashlib.sha256(generation_job_id.encode("utf-8")).hexdigest()[:12]
    count = int(payload["count"])
    assigned_domains = payload["assigned_content_domains"]
    if payload["generation_mode"] == "paired":
        condition_ids = [condition["condition_id"] for condition in payload["condition_definitions"]]
        paired_schema = payload.get("paired_item_metadata_schema")
        required_category_assignments = payload.get("required_category_assignments", [])
        pairs = []
        for index in range(count):
            paired_task_metadata = {}
            if paired_schema:
                for field_name, field_schema in paired_schema["properties"].items():
                    if field_schema.get("enum"):
                        paired_task_metadata[field_name] = field_schema["enum"][0]
                    elif field_schema["type"] == "integer":
                        paired_task_metadata[field_name] = int(field_schema.get("minimum", 0))
                    elif field_schema["type"] == "number":
                        paired_task_metadata[field_name] = float(field_schema.get("minimum", 0))
                    elif field_schema["type"] == "boolean":
                        paired_task_metadata[field_name] = True
                    else:
                        paired_task_metadata[field_name] = "mock_value"
                paired_task_metadata.update(
                    required_category_assignments[index]
                    if index < len(required_category_assignments)
                    else {}
                )
            pairs.append(
                {
                    "pair_id": f"mock_pair_{index}",
                    "content_domain": assigned_domains[index],
                    "prompts": [
                        {
                            "condition_id": condition_id,
                            "prompt_text": (
                                f"A person reviews item {job_nonce}_{index}_{condition_ids.index(condition_id)} "
                                "in an unfamiliar setting."
                            ),
                            **(
                                {"task_metadata": dict(paired_task_metadata)}
                                if paired_schema
                                else {}
                            ),
                        }
                        for condition_id in condition_ids
                    ],
                    "notes": "deterministic mock response",
                }
            )
        data = {"pairs": pairs}
    else:
        task_schema = payload["item_metadata_schema"]
        required_category_assignments = payload.get("required_category_assignments", [])
        task_metadata = {}
        for field_name, field_schema in task_schema["properties"].items():
            if field_schema.get("enum"):
                task_metadata[field_name] = field_schema["enum"][0]
            elif field_schema["type"] == "integer":
                task_metadata[field_name] = int(field_schema.get("minimum", 0))
            elif field_schema["type"] == "number":
                task_metadata[field_name] = float(field_schema.get("minimum", 0))
            elif field_schema["type"] == "boolean":
                task_metadata[field_name] = True
            else:
                task_metadata[field_name] = "mock_value"
        for field_name, field_value in (required_category_assignments[0] if required_category_assignments else {}).items():
            task_metadata[field_name] = field_value
        data = {
            "prompts": [
                {
                    "variant_id": f"mock_variant_{index}",
                    "content_domain": assigned_domains[index],
                    "task_metadata": {
                        **task_metadata,
                        **(
                            required_category_assignments[index]
                            if index < len(required_category_assignments)
                            else {}
                        ),
                    },
                    "prompt_text": (
                        f"Independent task item {job_nonce}_{index} asks for a response."
                    ),
                    "notes": "deterministic mock response",
                }
                for index in range(count)
            ]
        }
    return {"choices": [{"message": {"content": json.dumps(data)}}]}


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
                    prompt_family=f"{construct_id}_{prompt_role}",
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
    assert plan["run_mode"]["mode"] == "test"
    assert plan["run_mode"]["confirmatory"] is False
    assert plan["shared_execution"]["construct_ids"] == list(run_config.construct_ids)
    assert [stage["scope"] for stage in plan["execution_graph"][:2]] == ["shared", "shared"]
    assert len([stage for stage in plan["execution_graph"] if stage["scope"] == "construct"]) == 12
    baseline_stages = [
        stage for stage in plan["execution_graph"] if stage["stage_id"].startswith("evaluate_behavior_baseline:")
    ]
    steered_stages = [
        stage for stage in plan["execution_graph"] if stage["stage_id"].startswith("evaluate_behavior_steered:")
    ]
    zero_dose_stages = [
        stage for stage in plan["execution_graph"] if stage["stage_id"].startswith("evaluate_zero_dose_behavior:")
    ]
    assert len(baseline_stages) == len(zero_dose_stages) == len(steered_stages) == 2
    assert all(stage["prompt_only_baseline"] for stage in baseline_stages)
    assert all(stage["intervention"] == "target_zero_dose_only" for stage in zero_dose_stages)
    assert all(stage["intervention"] == "steered" for stage in steered_stages)
    assert all(stage["behavior_split"] == "steering_eval" for stage in steered_stages)
    assert all("zero_dose" in stage["comparison"] for stage in steered_stages)
    assert {entry["construct_id"] for entry in plan["constructs"]} == set(specs)
    assert all(
        stage["group_key"] == ["construct_id", "pair_id"]
        for stage in plan["execution_graph"]
        if stage["stage_id"].startswith("build_direction:")
    )


def test_construct_registry_freezes_all_waves_and_agrees_with_specs() -> None:
    registry = load_construct_registry(REGISTRY_PATH)
    specs = load_construct_specs(
        [REGISTRY_PATH.parent / entry.spec_path for entry in registry.entries]
    )
    summary = validate_registry_against_specs(registry, specs)
    assert summary["construct_count"] == 16
    assert summary["specified_count"] == 16
    assert summary["planned_count"] == 0
    assert summary["construct_ids_by_wave"]["1"] == [
        "realization_account_closure",
        "evidence_diagnosticity",
        "source_reliability",
        "persistence_continuation",
    ]

    malformed = registry.to_mapping()
    malformed["entries"][-1]["wave"] = 1
    with pytest.raises(ValueError, match="one construct from each registry family"):
        ConstructRegistry.from_mapping(malformed)


def test_wave_one_four_construct_run_config_fans_out_without_pooling() -> None:
    specs = load_construct_specs(WAVE1_CONSTRUCT_PATHS)
    run_config = load_run_config(WAVE1_RUN_CONFIG_PATH)
    analysis_spec = load_analysis_spec(ANALYSIS_PATH)
    plan = build_run_plan(run_config, specs, analysis_spec)

    assert plan["construct_count"] == 4
    assert plan["shared_execution"]["construct_ids"] == list(run_config.construct_ids)
    assert len([stage for stage in plan["execution_graph"] if stage["scope"] == "construct"]) == 24
    assert all(
        stage["group_key"] == ["construct_id", "pair_id"]
        for stage in plan["execution_graph"]
        if stage["stage_id"].startswith("build_direction:")
    )


def test_wave_one_generation_emits_canonical_records_and_is_deterministic(tmp_path: Path) -> None:
    specs = load_construct_specs(
        [
            ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json",
            ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v1.json",
            ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json",
            ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v1.json",
        ]
    )
    for plan_path in WAVE1_PLAN_PATHS:
        plan_raw = json.loads(plan_path.read_text(encoding="utf-8"))
        spec = specs[plan_raw["construct_id"]]
        plan = load_generation_plan(plan_path, spec)
        first_jobs = [job.job_id for job in iter_generation_jobs(plan)]
        second_jobs = [job.job_id for job in iter_generation_jobs(plan)]
        assert first_jobs == second_jobs

        result = generate_prompt_records(plan, spec, api_key="test", request_fn=_mock_generation_response)
        assert result.complete is True
        expected = dry_run_summary(plan)
        assert len(result.records) == expected["expected_record_count"]
        assert result.request_count == expected["request_count"]
        validate_prompt_records(result.records, {spec.construct_id: spec})
        assert {record.split for record in result.records} == set(spec.required_splits)
        assert all(record.metadata["generation_plan_id"] == plan["plan_id"] for record in result.records)
        assert all(record.metadata["generation_batch_id"] == record.metadata["generation_job_id"] for record in result.records)
        assert all(
            record.metadata["content_domain"]
            in plan["content_pools"][record.metadata["content_pool"]]["domains"]
            for record in result.records
        )
        downstream_records = [record for record in result.records if record.prompt_role != "probe"]
        assert all(record.metadata["task_metadata"] for record in downstream_records)
        assert all(
            set(record.metadata["task_metadata"])
            == set(spec.independent_behavior_task["item_metadata_schema"]["required"])
            for record in downstream_records
        )
        assert len({job.seed for job in iter_generation_jobs(plan)}) == len(first_jobs)

        chunked_plan = copy.deepcopy(plan)
        for cell in chunked_plan["cells"]:
            cell["count_per_model"] = 4
        chunked_result = generate_prompt_records(
            chunked_plan,
            spec,
            api_key="test",
            request_fn=_mock_generation_response,
        )
        assert len({record.metadata["generation_seed"] for record in chunked_result.records}) == chunked_result.request_count

        csv_path = tmp_path / f"{spec.construct_id}.csv"
        jsonl_path = tmp_path / f"{spec.construct_id}.jsonl"
        write_prompt_records(result.records, csv_path)
        write_prompt_records(result.records, jsonl_path)
        csv_records = load_prompt_records(csv_path)
        jsonl_records = load_prompt_records(jsonl_path)
        assert csv_records[0].metadata["generation_seed"] == result.records[0].metadata["generation_seed"]
        assert [record.prompt_id for record in jsonl_records] == [record.prompt_id for record in result.records]


def test_generation_parallelizes_request_jobs_but_keeps_plan_order() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json",
        spec,
    )
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def concurrent_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        response = _mock_generation_response(model_id, messages, options)
        with lock:
            active -= 1
        return response

    result = generate_prompt_records(
        plan,
        spec,
        api_key="test",
        request_fn=concurrent_response,
        workers=3,
        count_per_model_override=1,
        splits={"behavior_eval", "steering_eval", "calibration"},
    )
    assert maximum_active == 3
    assert result.request_count == 3
    assert [record.metadata["generation_job_id"] for record in result.records] == [
        "wave1_source_reliability_prompts_v1__source_reliability__sonnet__behavior_eval",
        "wave1_source_reliability_prompts_v1__source_reliability__sonnet__steering_eval",
        "wave1_source_reliability_prompts_v1__source_reliability__sonnet__calibration",
    ]


def test_source_reliability_materializes_balanced_paired_position_metadata() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json",
        spec,
    )
    result = generate_prompt_records(plan, spec, api_key="test", request_fn=_mock_generation_response)
    vector_records = [record for record in result.records if record.split in spec.paired_splits]
    assert vector_records
    for split in spec.paired_splits:
        split_records = [record for record in vector_records if record.split == split]
        positions = [record.metadata["minority_report_position"] for record in split_records]
        expected_count = next(cell for cell in plan["cells"] if cell["split"] == split)["count_per_model"]
        assert len(split_records) == expected_count * 2
        assert all(positions.count(position) == expected_count * 2 // 5 for position in range(1, 6))
        by_pair: dict[str, set[object]] = {}
        for record in split_records:
            by_pair.setdefault(str(record.pair_id), set()).add(record.metadata["minority_report_position"])
        assert by_pair
        assert all(values and len(values) == 1 for values in by_pair.values())

    missing_metadata = list(result.records)
    first = missing_metadata[0]
    first_metadata = dict(first.metadata)
    first_metadata.pop("minority_report_position")
    nested = dict(first_metadata.get("task_metadata", {}))
    nested.pop("minority_report_position", None)
    first_metadata["task_metadata"] = nested
    missing_metadata[0] = replace(first, metadata=first_metadata)
    with pytest.raises(ValueError, match="missing required metadata field"):
        validate_prompt_records(missing_metadata, {spec.construct_id: spec}, require_all_splits=False)



def test_generation_rejects_wrong_fields_missing_pairs_duplicate_ids_and_forbidden_text() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    plan_path = ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json"
    plan = load_generation_plan(plan_path, spec)

    def missing_pair_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
        del model_id, messages, options
        return {"choices": [{"message": {"content": json.dumps({"pairs": []})}}]}

    with pytest.raises(ValueError, match=r"exactly \d+ pairs"):
        generate_prompt_records(plan, spec, api_key="test", request_fn=missing_pair_response)

    def duplicate_pair_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
        response = _mock_generation_response(model_id, messages, options)
        payload = json.loads(response["choices"][0]["message"]["content"])
        if payload.get("pairs"):
            payload["pairs"][1]["pair_id"] = payload["pairs"][0]["pair_id"]
        response["choices"][0]["message"]["content"] = json.dumps(payload)
        return response

    with pytest.raises(ValueError, match="duplicate pair_id"):
        generate_prompt_records(plan, spec, api_key="test", request_fn=duplicate_pair_response)

    def forbidden_text_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
        response = _mock_generation_response(model_id, messages, options)
        payload = json.loads(response["choices"][0]["message"]["content"])
        if payload.get("pairs"):
            payload["pairs"][0]["prompts"][0]["prompt_text"] = "source_reliability"
        response["choices"][0]["message"]["content"] = json.dumps(payload)
        return response

    with pytest.raises(ValueError, match="forbidden term"):
        generate_prompt_records(plan, spec, api_key="test", request_fn=forbidden_text_response)

    def wrong_field_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
        response = _mock_generation_response(model_id, messages, options)
        payload = json.loads(response["choices"][0]["message"]["content"])
        if payload.get("pairs"):
            payload["pairs"][0]["unexpected_field"] = "reject me"
        response["choices"][0]["message"]["content"] = json.dumps(payload)
        return response

    with pytest.raises(ValueError, match="unexpected field"):
        generate_prompt_records(plan, spec, api_key="test", request_fn=wrong_field_response)

    def wrong_paired_metadata_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
        response = _mock_generation_response(model_id, messages, options)
        payload = json.loads(response["choices"][0]["message"]["content"])
        if payload.get("pairs"):
            payload["pairs"][0]["prompts"][0]["task_metadata"]["minority_report_position"] = 99
        response["choices"][0]["message"]["content"] = json.dumps(payload)
        return response

    with pytest.raises(ValueError, match="outside its registered enum"):
        generate_prompt_records(plan, spec, api_key="test", request_fn=wrong_paired_metadata_response)

    def missing_task_metadata_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
        response = _mock_generation_response(model_id, messages, options)
        payload = json.loads(response["choices"][0]["message"]["content"])
        if payload.get("prompts"):
            payload["prompts"][0].pop("task_metadata")
        response["choices"][0]["message"]["content"] = json.dumps(payload)
        return response

    with pytest.raises(ValueError, match="missing required field"):
        generate_prompt_records(plan, spec, api_key="test", request_fn=missing_task_metadata_response)


def test_generation_reports_provider_max_output_token_incomplete_response() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json",
        spec,
    )

    def incomplete_response(model_id: str, messages: list[dict[str, str]], options: dict) -> dict:
        del model_id, messages, options
        return {
            "choices": [{"message": {"content": '{"pairs": []}'}}],
            "_generation_metadata": {
                "provider": "openai",
                "status": "incomplete",
                "incomplete": True,
                "incomplete_reason": "max_output_tokens",
            },
        }

    with pytest.raises(ValueError, match=r"response incomplete.*max_output_tokens"):
        generate_prompt_records(plan, spec, api_key="test", request_fn=incomplete_response)


def test_generation_plan_requires_all_wave_one_splits(tmp_path: Path) -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    plan_path = ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["cells"] = [cell for cell in payload["cells"] if cell["split"] != "calibration"]
    incomplete_path = tmp_path / "incomplete_generation_plan.json"
    incomplete_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required splits"):
        load_generation_plan(incomplete_path, spec)

    missing_composition = json.loads(plan_path.read_text(encoding="utf-8"))
    missing_composition.pop("task_composition")
    missing_composition_path = tmp_path / "missing_composition_plan.json"
    missing_composition_path.write_text(json.dumps(missing_composition), encoding="utf-8")
    with pytest.raises(ValueError, match="task_composition must be an object"):
        load_generation_plan(missing_composition_path, spec)


def test_single_generator_plan_and_pilot_override_are_explicit() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    plan_path = ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json"
    plan = load_generation_plan(plan_path, spec)

    full = dry_run_summary(plan)
    assert full["complete_plan"] is True
    assert full["selected_model_aliases"] == ["sonnet"]
    assert full["expected_record_count"] == 456

    pilot = dry_run_summary(plan, count_per_model_override=1)
    assert pilot["complete_plan"] is False
    assert pilot["count_per_model_override"] == 1
    assert pilot["expected_record_count"] == 9
    assert pilot["request_count"] == 6
    assert pilot["estimated_output_tokens"] < full["estimated_output_tokens"]

    priced = dry_run_summary(
        plan,
        count_per_model_override=1,
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=2.0,
    )
    assert priced["estimated_cost_usd"] is not None

    result = generate_prompt_records(
        plan,
        spec,
        api_key="test",
        request_fn=_mock_generation_response,
        count_per_model_override=1,
    )
    assert result.complete is False
    assert len(result.records) == 9


def test_vector_scope_generates_and_validates_only_paired_splits() -> None:
    spec = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json"
    )
    plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave1_realization_account_closure_v1.json",
        spec,
    )
    vector_splits = {"direction_train", "direction_validation", "direction_heldout"}
    summary = dry_run_summary(plan, splits=vector_splits)
    assert summary["complete_plan"] is False
    assert summary["selected_splits"] == sorted(vector_splits)
    assert set(summary["records_by_split"]) == vector_splits

    result = generate_prompt_records(
        plan,
        spec,
        api_key="test",
        request_fn=_mock_generation_response,
        splits=vector_splits,
    )
    assert result.complete is False
    assert {record.split for record in result.records} == vector_splits
    validate_prompt_records(
        result.records,
        {spec.construct_id: spec},
        require_all_splits=False,
    )
    with pytest.raises(ValueError, match="missing required splits"):
        validate_prompt_records(result.records, {spec.construct_id: spec})


def test_cell_instructions_do_not_override_named_mode_counts() -> None:
    for plan_path in ALL_GENERATION_PLAN_PATHS:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if "base_plan_path" in plan:
            continue
        for cell in plan["cells"]:
            instructions = str(cell.get("instructions", ""))
            assert re.search(r"\bgenerate\s+\d+\b", instructions, flags=re.IGNORECASE) is None, (
                plan_path,
                cell["cell_id"],
            )


def test_crossed_wave_one_factor_schedules_are_complete() -> None:
    for plan_path in WAVE1_PLAN_PATHS[-2:]:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        schedule = plan["behavior_factor_schedule"]
        fields = schedule["required_item_fields"]
        required = {
            tuple(combination[field] for field in fields)
            for combination in schedule["required_combinations"]
        }
        assert len(required) == len(schedule["required_combinations"])
        for cell in plan["cells"]:
            if cell["split"] not in {"behavior_eval", "steering_eval"}:
                continue
            observed = set(zip(*(cell["category_balance"][field] for field in fields), strict=True))
            assert cell["count_per_model"] == len(required)
            assert observed == required


def test_named_generation_and_model_run_modes_are_explicit_and_deterministic() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json")
    plan_path = ROOT / "configs/construct_benchmark/generation_plans/wave1_realization_account_closure_v1.json"
    plan = load_generation_plan(plan_path, spec)
    assert plan["run_modes"]["review"]["partial"] is True
    assert plan["run_modes"]["full"]["partial"] is False

    run_config = load_run_config(
        ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json"
    )
    result = generate_prompt_records(plan, spec, api_key="test", request_fn=_mock_generation_response)
    selected_a, manifest_a = select_prompt_records(
        result.records,
        run_config=run_config,
        construct_specs={spec.construct_id: spec},
        mode="test",
    )
    selected_b, manifest_b = select_prompt_records(
        result.records,
        run_config=run_config,
        construct_specs={spec.construct_id: spec},
        mode="test",
    )
    assert [record.prompt_id for record in selected_a] == [record.prompt_id for record in selected_b]
    assert manifest_a == manifest_b
    assert manifest_a["confirmatory"] is False
    assert manifest_a["max_runtime_minutes"] == 60
    assert manifest_a["selected_prompt_count"] == 18
    assert manifest_a["selected_counts_by_construct_split"] == {
        "realization_account_closure": {
            "behavior_eval": 2,
            "calibration": 2,
            "direction_heldout": 4,
            "direction_train": 4,
            "direction_validation": 4,
            "steering_eval": 2,
        }
    }
    for split in spec.paired_splits:
        pair_counts = {}
        for record in selected_a:
            if record.split == split:
                pair_counts[record.pair_id] = pair_counts.get(record.pair_id, 0) + 1
        assert pair_counts
        assert set(pair_counts.values()) == {2}

    selected_full, full_manifest = select_prompt_records(
        result.records,
        run_config=run_config,
        construct_specs={spec.construct_id: spec},
        mode="full",
    )
    assert len(selected_full) == len(result.records)
    assert len(result.records) == dry_run_summary(plan)["expected_record_count"]
    assert full_manifest["complete_inventory"] is True
    assert full_manifest["confirmatory"] is True
    full_plan = build_run_plan(
        load_run_config(
            ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json"
        ),
        {spec.construct_id: spec},
        load_analysis_spec(ANALYSIS_PATH),
        run_mode="full",
    )
    assert full_plan["run_mode"]["mode"] == "full"
    assert full_plan["run_mode"]["confirmatory"] is True


def test_limited_generation_is_explicitly_incomplete() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    plan_path = ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json"
    plan = load_generation_plan(plan_path, spec)
    result = generate_prompt_records(
        plan,
        spec,
        api_key="test",
        request_fn=_mock_generation_response,
        limit_jobs=1,
    )
    assert result.complete is False
    assert result.summary()["complete"] is False


def test_generated_records_cannot_bypass_canonical_role_validation() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    plan_path = ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json"
    plan = load_generation_plan(plan_path, spec)
    result = generate_prompt_records(plan, spec, api_key="test", request_fn=_mock_generation_response)
    altered = replace(result.records[0], prompt_role="behavior")
    with pytest.raises(ValueError, match="requires role='probe'"):
        validate_prompt_records([altered, *result.records[1:]], {spec.construct_id: spec})


def test_probe_wrapper_normalization_repairs_only_registered_whitespace() -> None:
    spec = load_construct_spec(ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json")
    prefix, suffix = spec.probe_prompt_template.split("{scenario}", maxsplit=1)
    malformed = f"{prefix.rstrip()}Example scenario. {suffix.strip()}"

    normalized, changed = normalize_probe_prompt_wrapper(
        malformed,
        probe_prompt_template=spec.probe_prompt_template,
    )

    assert changed is True
    assert normalized == f"{prefix}Example scenario.{suffix}"
    unchanged, changed_again = normalize_probe_prompt_wrapper(
        "A different opening that is not the registered wrapper.",
        probe_prompt_template=spec.probe_prompt_template,
    )
    assert unchanged == "A different opening that is not the registered wrapper."
    assert changed_again is False


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
    assert len([stage for stage in plan["execution_graph"] if stage["scope"] == "construct"]) == 24


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


def test_prompt_validation_rejects_reused_text_across_independent_roles() -> None:
    specs, _, _ = _configs()
    records = _prompt_inventory(specs)
    behavior = next(record for record in records if record.prompt_role == "behavior")
    steering = next(record for record in records if record.prompt_role == "steering")
    altered = replace(steering, prompt_text=behavior.prompt_text)
    with pytest.raises(ValueError, match="reuses normalized prompt text"):
        validate_prompt_records([record for record in records if record is not steering] + [altered], specs)


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
