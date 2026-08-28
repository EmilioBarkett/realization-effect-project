from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from construct_benchmark.prompts import load_prompt_records  # noqa: E402
from construct_benchmark.generation import (  # noqa: E402
    _complete_downstream_prompt_text,
    _registered_response_instruction,
    _response_instruction_is_present,
)
from scripts.generate_downstream_prompts import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_REASONING_EFFORT,
    DOWNSTREAM_SPLITS,
    _calibration_cue_hits,
    _calibration_text_issues,
    _allocate_request_workers,
    _downstream_pool_text_issues,
    _effective_entries,
    _parser_instruction_issues,
    _validate_calibration_plan,
    _validate_quality_gate,
    audit_downstream_inventory,
    orchestrate_downstream_prompts,
)


REGISTRY = ROOT / "configs/construct_benchmark/construct_registry_v1.json"


def test_allocation_response_format_cannot_fall_back_to_probability_wording() -> None:
    task = {
        "task_id": "goal_renewal_allocation_v2",
        "prompt_template": "Describe the independent allocation scenario.",
    }
    instruction = _registered_response_instruction(
        task,
        "single_integer_allocation_0_to_100",
    )
    assert instruction == "Return exactly one integer from 0 to 100: the points assigned to option A."
    assert "probability" not in instruction.casefold()
    assert _response_instruction_is_present(
        instruction,
        "single_integer_allocation_0_to_100",
    )


def test_downstream_workers_are_a_global_request_budget() -> None:
    entries = [
        type("Entry", (), {"construct_id": construct_id})()
        for construct_id in ("alpha", "beta", "gamma", "delta")
    ]
    assert _allocate_request_workers(entries, 4) == {
        "alpha": 1,
        "beta": 1,
        "gamma": 1,
        "delta": 1,
    }
    assert _allocate_request_workers(entries[:2], 4) == {"alpha": 2, "beta": 2}
    assert _allocate_request_workers(entries[:1], 4) == {"alpha": 4}


def _deterministic_response(model_id, messages, options):
    del model_id
    payload = json.loads(messages[1]["content"].split("\n\nCORRECTIVE RETRY", 1)[0])
    count = int(payload["count"])
    domains = payload["assigned_content_domains"]
    assignments = payload["required_category_assignments"]
    task_schema = payload["item_metadata_schema"]
    task_template = payload["independent_task"]["prompt_template"]
    nonce = hashlib.sha256(options["generation_job_id"].encode("utf-8")).hexdigest()[:10]
    prompts = []
    for index in range(count):
        metadata = dict(assignments[index]) if index < len(assignments) else {}
        for field, schema in task_schema["properties"].items():
            if field not in metadata:
                if schema.get("enum"):
                    metadata[field] = schema["enum"][0]
                elif schema.get("type") == "integer":
                    metadata[field] = int(schema.get("minimum", 0))
                elif schema.get("type") == "number":
                    metadata[field] = float(schema.get("minimum", 0))
                elif schema.get("type") == "boolean":
                    metadata[field] = True
                else:
                    metadata[field] = "mock"
        # The final downstream prompt must end with its response contract;
        # generation-only domain/variant notes belong before the task text and
        # the probe-context instruction is not part of the end-user prompt.
        task_template = task_template.replace(" Do not mention or reuse the earlier scenario.", "")
        task_template = task_template.replace(" do not mention the earlier scenario.", "")
        if task_template.rstrip()[-1:] not in ".!?":
            task_template = task_template.rstrip() + "."
        prompt_text = (
            f"Use the {domains[index]} setting for this fixture; "
            f"variant {payload['prompt_role']} {nonce} {index}. {task_template}"
        )
        if payload["prompt_role"] == "calibration" and "outcome_valence" in task_schema["properties"]:
            prompt_text = (
                "The sure option produces exactly 10 neutral outcome units with certainty. "
                "The risky option produces exactly 20 neutral outcome units with probability one-half "
                "and 0 neutral outcome units otherwise. These are abstract units with no external meaning. "
                + prompt_text
            )
        prompts.append(
            {
                "variant_id": f"item_{index}_{nonce}",
                "content_domain": domains[index],
                "task_metadata": metadata,
                "prompt_text": prompt_text,
                "notes": "deterministic downstream fixture",
            }
        )
    return {
        "choices": [{"message": {"content": json.dumps({"prompts": prompts})}}],
        "_generation_metadata": {
            "actual_cost_usd": 0.0001,
            "input_tokens": 100,
            "output_tokens": 100,
            "total_tokens": 200,
        },
    }


def _wrong_count_response(model_id, messages, options):
    response = _deterministic_response(model_id, messages, options)
    payload = json.loads(response["choices"][0]["message"]["content"])
    payload["prompts"] = []
    response["choices"][0]["message"]["content"] = json.dumps(payload)
    return response


def test_wave1_full_dry_run_has_registered_downstream_counts(tmp_path: Path) -> None:
    manifest = orchestrate_downstream_prompts(
        registry_path=REGISTRY,
        waves=[1],
        output_dir=tmp_path,
        mode="full",
        batch_size=20,
        max_output_tokens=12000,
        dry_run=True,
    )
    assert manifest["counts"]["record_count"] == 210
    assert manifest["counts"]["split_counts"] == {
        "behavior_eval": 70,
        "calibration": 70,
        "steering_eval": 70,
    }
    assert manifest["counts"]["request_count"] == 15
    assert not list(tmp_path.iterdir())


def test_full_generation_rejects_subminimum_request_batches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"full generation requires batch_size >= 20"):
        orchestrate_downstream_prompts(
            registry_path=REGISTRY,
            waves=[1],
            construct_ids=["source_reliability"],
            output_dir=tmp_path,
            mode="full",
            batch_size=10,
            max_output_tokens=12000,
            dry_run=True,
        )


def test_wave1_calibration_contracts_use_separate_neutral_schedules() -> None:
    entries = _effective_entries(
        REGISTRY,
        waves=[1],
        construct_ids=None,
        batch_size=20,
        max_output_tokens=12000,
    )
    by_id = {entry.construct_id: entry for entry in entries}
    for entry in entries:
        _validate_calibration_plan(entry)
        calibration = next(cell for cell in entry.plan["cells"] if cell["split"] == "calibration")
        assert calibration["factor_schedule"] == "calibration_factor_schedule"
    realization = by_id["realization_account_closure"]
    realization_calibration = next(cell for cell in realization.plan["cells"] if cell["split"] == "calibration")
    assert set(realization_calibration["category_balance"]["outcome_valence"]) == {"neutral"}
    evidence = by_id["evidence_diagnosticity"]
    evidence_calibration = next(cell for cell in evidence.plan["cells"] if cell["split"] == "calibration")
    assert set(evidence_calibration["category_balance"]["evidence_valence"]) == {"neutral"}
    assert set(evidence_calibration["category_balance"]["likelihood_separation"]) == {"none"}
    source = by_id["source_reliability"]
    source_calibration = next(cell for cell in source.plan["cells"] if cell["split"] == "calibration")
    assert set(source_calibration["category_balance"]["source_track_record"]) == {"midpoint"}
    persistence = by_id["persistence_continuation"]
    assert persistence.plan["calibration_factor_schedule"]["neutral_fields"] == {}


def test_wave2_to_wave4_calibration_contracts_use_separate_neutral_schedules() -> None:
    entries = _effective_entries(
        REGISTRY,
        waves=[2, 3, 4],
        construct_ids=None,
        batch_size=20,
        max_output_tokens=30000,
    )
    assert len(entries) == 12
    for entry in entries:
        _validate_calibration_plan(entry)
        calibration = next(cell for cell in entry.plan["cells"] if cell["split"] == "calibration")
        assert calibration["factor_schedule"] == "calibration_factor_schedule"
        assert set(entry.plan["calibration_factor_schedule"]) >= {
            "purpose",
            "nuisance_fields",
            "neutral_fields",
            "forbidden_terms",
            "required_response_format",
        }


def test_missing_downstream_response_instruction_is_completed_from_registered_task() -> None:
    entry = _effective_entries(
        REGISTRY,
        waves=[2],
        construct_ids=["reference_frame"],
        batch_size=20,
        max_output_tokens=30000,
    )[0]
    task = entry.spec.independent_behavior_task
    completed, changed = _complete_downstream_prompt_text(
        "A sure option pays 10 units with certainty, while a risky option pays 20 units with probability 0.5.",
        task=task,
        expected_format=task["response_format"],
        prompt_role="behavior",
    )
    assert changed is True
    assert "Return exactly one integer: 1 or 2" in completed


def test_materialized_downstream_record_keeps_registered_response_completion(tmp_path: Path) -> None:
    def request(model_id, messages, options):
        response = _deterministic_response(model_id, messages, options)
        payload = json.loads(response["choices"][0]["message"]["content"])
        for prompt in payload["prompts"]:
            prompt["prompt_text"] = prompt["prompt_text"].split(" Return exactly", 1)[0].rstrip() + "."
        response["choices"][0]["message"]["content"] = json.dumps(payload)
        return response

    manifest = orchestrate_downstream_prompts(
        registry_path=REGISTRY,
        waves=[2],
        construct_ids=["reference_frame"],
        output_dir=tmp_path,
        mode="review",
        workers=1,
        batch_size=20,
        max_output_tokens=12000,
        api_key="test-only",
        request_fn=request,
        provider="openai",
        model="gpt-5.6-luna",
        reasoning_effort="xhigh",
        max_estimated_cost_usd=1.0,
        input_usd_per_million_tokens=0.2,
        output_usd_per_million_tokens=1.2,
        vector_reference=None,
    )
    assert manifest["status"] == "complete_review"
    records = load_prompt_records(tmp_path / "combined.csv")
    assert len(records) == 3
    assert all(
        record.metadata["response_instruction_completion"] == "appended_registered_task_instruction"
        for record in records
    )
    assert all("Return exactly one integer: 1 or 2;" in record.prompt_text for record in records)


def test_registered_neutral_calibration_payoff_is_completed_from_plan() -> None:
    entry = _effective_entries(
        REGISTRY,
        waves=[1],
        construct_ids=["realization_account_closure"],
        batch_size=50,
        max_output_tokens=60000,
    )[0]
    task = entry.spec.independent_behavior_task
    completed, response_changed = _complete_downstream_prompt_text(
        "Choose between a sure option and a risky option in this neutral calibration item.",
        task=task,
        expected_format=task["response_format"],
        prompt_role="calibration",
        plan=entry.plan,
    )
    assert response_changed is True
    assert "sure option produces exactly 10 neutral outcome units with certainty" in completed
    assert "risky option produces exactly 20 neutral outcome units with probability one-half" in completed
    assert "0 neutral outcome units otherwise" in completed
    assert "abstract outcome units have no external meaning" in completed
    assert len(completed) < 2000


def test_calibration_validator_rejects_reused_target_schedule() -> None:
    entry = next(
        entry
        for entry in _effective_entries(
            REGISTRY,
            waves=[1],
            construct_ids=["evidence_diagnosticity"],
            batch_size=20,
            max_output_tokens=12000,
        )
    )
    bad_plan = deepcopy(entry.plan)
    calibration = next(cell for cell in bad_plan["cells"] if cell["split"] == "calibration")
    calibration["factor_schedule"] = "behavior_factor_schedule"
    from dataclasses import replace
    from scripts.generate_downstream_prompts import _canonical_sha256

    bad_entry = replace(entry, plan=bad_plan, plan_sha256=_canonical_sha256(bad_plan))
    with pytest.raises(ValueError, match="separate calibration_factor_schedule"):
        _validate_calibration_plan(bad_entry)


def test_realization_calibration_rejects_missing_parser_request_and_status_cue() -> None:
    from construct_benchmark.prompts import PromptRecord

    record = PromptRecord(
        prompt_id="bad_realization_calibration",
        construct_id="realization_account_closure",
        split="calibration",
        prompt_role="calibration",
        prompt_text="A community ledger records an unresolved provisional entry. Continue processing the scenario.",
        condition_id="neutral",
        expected_output_format="two_integers_on_separate_lines",
        parser_id="two_integers_risk_choice_v1",
    )
    assert _parser_instruction_issues(record)
    entry = next(
        entry
        for entry in _effective_entries(
            REGISTRY,
            waves=[1],
            construct_ids=["realization_account_closure"],
            batch_size=20,
            max_output_tokens=12000,
        )
    )
    assert {"unresolved", "provisional"}.issubset(set(_calibration_cue_hits(entry, record.prompt_text)))


def test_realization_calibration_requires_coherent_neutral_abstract_payoff() -> None:
    from construct_benchmark.prompts import PromptRecord

    entry = next(
        entry
        for entry in _effective_entries(
            REGISTRY,
            waves=[1],
            construct_ids=["realization_account_closure"],
            batch_size=20,
            max_output_tokens=12000,
        )
    )
    base = (
        "Consider a separate choice between a sure option and a risky option. "
        "The sure option produces exactly 10 neutral outcome units with certainty. "
        "The risky option produces exactly 20 neutral outcome units with probability one-half and 0 neutral outcome units otherwise. "
        "These are abstract units with no external meaning. Return exactly two integers on separate lines: first the allocation to the risky option from 1 to 1000, with the remainder assigned to the sure option; second a risk preference from 1 to 5."
    )
    good = PromptRecord(
        prompt_id="good_neutral_risk_calibration",
        construct_id=entry.construct_id,
        split="calibration",
        prompt_role="calibration",
        prompt_text=base,
        condition_id="neutral",
        parser_id="two_integers_risk_choice_v1",
        expected_output_format="two_integers_on_separate_lines",
        metadata={"content_pool": "realization_calibration", "task_metadata": {"outcome_valence": "neutral"}},
    )
    assert not _calibration_text_issues(entry, good)
    bad = PromptRecord(
        prompt_id="meaningless_neutral_risk_calibration",
        construct_id=entry.construct_id,
        split="calibration",
        prompt_role="calibration",
        prompt_text=(
            "Tokens assigned to the sure option stay in a tray. Tokens assigned to the risky option are sorted into two equally likely trays. "
            "Neither tray changes the tokens. Return exactly two integers on separate lines: first the allocation to the risky option from 1 to 1000, with the remainder assigned to the sure option; second a risk preference from 1 to 5."
        ),
        condition_id="neutral",
        parser_id="two_integers_risk_choice_v1",
        expected_output_format="two_integers_on_separate_lines",
        metadata={"content_pool": "realization_calibration", "task_metadata": {"outcome_valence": "neutral"}},
    )
    assert _calibration_text_issues(entry, bad)


def test_persistence_pool_rules_reject_semantically_shared_topics() -> None:
    entries = _effective_entries(
        REGISTRY,
        waves=[1],
        construct_ids=["persistence_continuation"],
        batch_size=20,
        max_output_tokens=12000,
    )
    entry = entries[0]
    from construct_benchmark.prompts import PromptRecord

    shared_text = (
        "A city library maintains an existing literacy program and launches a new digital-skills program. "
        "Return exactly two integers on separate lines: first the existing-program allocation and second the new-program allocation, with the two integers summing to 100."
    )
    steering = PromptRecord(
        prompt_id="shared_steering_topic",
        construct_id=entry.construct_id,
        split="steering_eval",
        prompt_role="steering",
        prompt_text=shared_text,
        condition_id="neutral",
        prompt_family="persistence_steering_eval",
        task_id="program_renewal_allocation_v1",
        parser_id="two_integers_sum_100_v1",
        expected_output_format="two_integers_sum_100",
        metadata={"content_pool": "persistence_steering", "task_metadata": {"existing_program_value": "high", "alternative_value": "high", "implementation_feasibility": "viable"}},
    )
    assert _downstream_pool_text_issues(entry, steering)


def test_semantic_retry_corrects_wrong_prompt_count_and_logs_attempts(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    seen: dict[str, int] = {}

    def request(model_id, messages, options):
        job_id = str(options["generation_job_id"])
        seen[job_id] = seen.get(job_id, 0) + 1
        calls.append((job_id, messages[-1]["content"]))
        if seen[job_id] == 1:
            return _wrong_count_response(model_id, messages, options)
        return _deterministic_response(model_id, messages, options)

    manifest = orchestrate_downstream_prompts(
        registry_path=REGISTRY,
        waves=[1],
        construct_ids=["realization_account_closure"],
        output_dir=tmp_path,
        mode="review",
        workers=1,
        batch_size=20,
        max_output_tokens=12000,
        api_key="test-only",
        request_fn=request,
        provider="openai",
        model="gpt-5.6-luna",
        reasoning_effort="xhigh",
        max_estimated_cost_usd=1.0,
        input_usd_per_million_tokens=0.2,
        output_usd_per_million_tokens=1.2,
        vector_reference=None,
    )
    assert manifest["status"] == "complete_review"
    assert manifest["counts"]["record_count"] == 3
    assert manifest["counts"]["attempt_count"] == 6
    assert manifest["counts"]["rejected_attempt_count"] == 3
    assert manifest["counts"]["actual_cost_usd"] == pytest.approx(0.0006)
    assert manifest["counts"]["materialized_record_cost_usd"] == pytest.approx(0.0003)
    assert manifest["accounting"]["new_attempt_count"] == 6
    assert manifest["accounting"]["cumulative_attempt_count"] == 6
    retry_messages = [content for _, content in calls if "CORRECTIVE RETRY 2" in content]
    assert len(retry_messages) == 3
    assert all("must return exactly 1 prompts" in content for content in retry_messages)
    state = json.loads((tmp_path / "downstream_prompt_run_state.json").read_text())
    assert sum(len(history) for history in state["job_attempts"].values()) == 6
    assert sum(
        item["status"] == "rejected"
        for history in state["job_attempts"].values()
        for item in history
    ) == 3
    checkpoint = json.loads((tmp_path / "checkpoints/realization_account_closure.json").read_text())
    assert sum(len(history) for history in checkpoint["attempts"].values()) == 6
    assert all(
        attempt["response_metadata"]["actual_cost_usd"] == 0.0001
        for history in checkpoint["attempts"].values()
        for attempt in history
    )


def test_semantic_retry_fails_closed_after_bounded_retry_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"failed after 6 semantic attempt\(s\)"):
        orchestrate_downstream_prompts(
            registry_path=REGISTRY,
            waves=[1],
            construct_ids=["realization_account_closure"],
            output_dir=tmp_path,
            mode="review",
            workers=1,
            batch_size=20,
            max_output_tokens=12000,
            api_key="test-only",
            request_fn=_wrong_count_response,
            provider="openai",
            model="gpt-5.6-luna",
            reasoning_effort="xhigh",
            max_estimated_cost_usd=1.0,
            input_usd_per_million_tokens=0.2,
            output_usd_per_million_tokens=1.2,
            vector_reference=None,
        )
    state = json.loads((tmp_path / "downstream_prompt_run_state.json").read_text())
    assert state["status"] == "failed"
    histories = list(state["job_attempts"].values())
    assert len(histories) == 1
    assert len(histories[0]) == 6
    assert all(item["status"] == "rejected" for item in histories[0])
    assert all("must return exactly 1 prompts" in item["rejection_reason"] for item in histories[0])
    assert state["budget_state"]["completed_request_count"] == 6
    checkpoint = json.loads((tmp_path / "checkpoints/realization_account_closure.json").read_text())
    assert len(checkpoint["attempts"]) == 1
    assert len(next(iter(checkpoint["attempts"].values()))) == 6


def test_failed_phase_resume_preserves_attempt_usage_and_rejection_history(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    phase = "failed_phase"

    def request(model_id, messages, options):
        job_id = str(options["generation_job_id"])
        seen[job_id] = seen.get(job_id, 0) + 1
        calls.append((job_id, messages[-1]["content"]))
        if phase == "failed_phase" and "persistence_continuation" in job_id and "behavior_eval" in job_id:
            if seen[job_id] == 1:
                return _wrong_count_response(model_id, messages, options)
            raise RuntimeError("simulated interruption after the rejected response")
        if phase == "resume" and "persistence_continuation" in job_id and "steering_eval" in job_id and seen[job_id] == 1:
            return _wrong_count_response(model_id, messages, options)
        return _deterministic_response(model_id, messages, options)

    common = {
        "registry_path": REGISTRY,
        "waves": [1],
        "construct_ids": ["realization_account_closure", "persistence_continuation"],
        "output_dir": tmp_path,
        "mode": "review",
        "workers": 1,
        "batch_size": 20,
        "max_output_tokens": 12000,
        "api_key": "test-only",
        "request_fn": request,
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "max_estimated_cost_usd": 1.0,
        "input_usd_per_million_tokens": 0.2,
        "output_usd_per_million_tokens": 1.2,
        "vector_reference": None,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        orchestrate_downstream_prompts(**common)
    failed_state = json.loads((tmp_path / "downstream_prompt_run_state.json").read_text())
    assert failed_state["status"] == "failed"
    assert failed_state["cumulative_attempt_count"] == 5
    assert failed_state["cumulative_rejected_attempt_count"] == 1
    assert failed_state["cumulative_failed_request_count"] == 1
    assert failed_state["cumulative_actual_input_tokens"] == 400
    assert failed_state["cumulative_actual_output_tokens"] == 400
    assert failed_state["cumulative_actual_total_tokens"] == 800
    assert failed_state["unattributed_attempt_count"] == 1
    assert sum(len(history) for history in failed_state["job_attempts"].values()) == 4

    phase = "resume"
    calls.clear()
    resumed = orchestrate_downstream_prompts(**(common | {"resume": True}))
    assert resumed["status"] == "complete_review"
    assert resumed["counts"]["record_count"] == 6
    assert resumed["counts"]["attempt_count"] == 9
    assert resumed["counts"]["rejected_attempt_count"] == 2
    assert resumed["counts"]["failed_request_count"] == 1
    assert resumed["counts"]["actual_cost_usd"] == pytest.approx(0.0008)
    assert resumed["counts"]["materialized_record_cost_usd"] == pytest.approx(0.0006)
    assert resumed["counts"]["actual_input_tokens"] == 800
    assert resumed["counts"]["actual_output_tokens"] == 800
    assert resumed["accounting"]["prior_attempt_count"] == 5
    assert resumed["accounting"]["new_attempt_count"] == 4
    assert resumed["accounting"]["cumulative_attempt_count"] == 9
    assert resumed["accounting"]["prior_actual_spent_usd"] == pytest.approx(0.0004)
    assert resumed["accounting"]["new_actual_spent_usd"] == pytest.approx(0.0004)
    assert resumed["accounting"]["cumulative_actual_spent_usd"] == pytest.approx(0.0008)
    assert len(calls) == 4
    retry_messages = [content for _, content in calls if "CORRECTIVE RETRY 2" in content]
    assert len(retry_messages) == 2
    assert all("must return exactly 1 prompts" in content for content in retry_messages)
    final_state = json.loads((tmp_path / "downstream_prompt_run_state.json").read_text())
    assert final_state["cumulative_attempt_count"] == 9
    assert final_state["cumulative_rejected_attempt_count"] == 2
    assert final_state["cumulative_failed_request_count"] == 1
    assert sum(len(history) for history in final_state["job_attempts"].values()) == 8


def test_downstream_validator_rejects_ambiguous_risk_allocation_mapping() -> None:
    from construct_benchmark.prompts import PromptRecord

    record = PromptRecord(
        prompt_id="ambiguous_risk_allocation",
        construct_id="realization_account_closure",
        split="behavior_eval",
        prompt_role="behavior",
        prompt_text=(
            "Choose between a sure option and a risky option. Return exactly two integers on separate lines: "
            "first an allocation from 1 to 1000; second a risk preference from 1 to 5."
        ),
        condition_id="neutral",
        parser_id="two_integers_risk_choice_v1",
        expected_output_format="two_integers_on_separate_lines",
    )
    assert any("risky-option allocation" in issue for issue in _parser_instruction_issues(record))


def test_downstream_validator_rejects_truncated_prompt_tail() -> None:
    from construct_benchmark.prompts import PromptRecord

    record = PromptRecord(
        prompt_id="truncated_source_prompt",
        construct_id="source_reliability",
        split="steering_eval",
        prompt_role="steering",
        prompt_text="Report the probability from 0 to 100 as one integer for the claim that the",
        condition_id="neutral",
        parser_id="single_integer_probability_v1",
        expected_output_format="single_integer_0_to_100",
    )
    issues = _parser_instruction_issues(record)
    assert "prompt does not end with terminal punctuation" in issues
    assert "prompt appears truncated at an incomplete phrase" in issues


def test_downstream_workflow_defaults_and_artifacts_are_luna_only(tmp_path: Path) -> None:
    assert DEFAULT_PROVIDER == "openai"
    assert DEFAULT_MODEL == "gpt-5.6-luna"
    assert DEFAULT_REASONING_EFFORT == "xhigh"
    result = orchestrate_downstream_prompts(
        registry_path=REGISTRY,
        waves=[1],
        output_dir=tmp_path,
        mode="review",
        workers=1,
        batch_size=20,
        max_output_tokens=4000,
        api_key="test-only",
        request_fn=_deterministic_response,
        max_estimated_cost_usd=10.0,
        input_usd_per_million_tokens=0.2,
        output_usd_per_million_tokens=1.2,
        vector_reference=None,
    )
    assert result["provider"] == "openai"
    assert result["requested_model"] == "gpt-5.6-luna"
    assert result["reasoning_effort"] == "xhigh"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8").lower()
            assert "sonnet" not in text
            assert "claude" not in text
    records = load_prompt_records(tmp_path / "combined.csv")
    assert {record.metadata["source_model_alias"] for record in records} == {"luna"}
    assert {record.metadata["source_model"] for record in records} == {"gpt-5.6-luna"}


def test_downstream_workflow_rejects_non_luna_provider_model_or_reasoning(tmp_path: Path) -> None:
    common = {
        "registry_path": REGISTRY,
        "waves": [1],
        "output_dir": tmp_path,
        "mode": "review",
        "api_key": "test-only",
        "request_fn": _deterministic_response,
        "max_estimated_cost_usd": 10.0,
        "input_usd_per_million_tokens": 0.2,
        "output_usd_per_million_tokens": 1.2,
        "vector_reference": None,
    }
    with pytest.raises(ValueError, match="provider must be 'openai'"):
        orchestrate_downstream_prompts(**common, provider="openrouter")
    with pytest.raises(ValueError, match="model must be 'gpt-5.6-luna'"):
        orchestrate_downstream_prompts(**(common | {"output_dir": tmp_path / "model"}), model="gpt-5.5")
    with pytest.raises(ValueError, match="reasoning_effort='xhigh'"):
        orchestrate_downstream_prompts(**(common | {"output_dir": tmp_path / "reason"}), reasoning_effort="high")


def test_composite_quality_gate_selects_construct_specific_review_manifests(tmp_path: Path) -> None:
    entries = _effective_entries(
        REGISTRY,
        waves=[1],
        construct_ids=None,
        batch_size=20,
        max_output_tokens=12000,
        model=DEFAULT_MODEL,
    )

    def write_manifest(name: str, selected_entries: list[object]) -> tuple[Path, str]:
        constructs = []
        for entry in selected_entries:
            output_path = tmp_path / f"{name}_{entry.construct_id}.csv"
            output_path.write_text(f"fixture,{entry.construct_id}\n", encoding="utf-8")
            constructs.append({
                "construct_id": entry.construct_id,
                "source_plan_sha256": entry.source_plan_sha256,
                "plan_sha256": entry.plan_sha256,
                "spec_sha256": entry.spec_sha256,
                "output_path": str(output_path),
                "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            })
        manifest = {
            "manifest_type": "downstream_prompt_generation",
            "status": "complete_review",
            "run_mode": "review",
            "partial": True,
            "dry_run": False,
            "frozen": False,
            "provider": "openai",
            "requested_model": DEFAULT_MODEL,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
            "audit": {"severe_flag_count": 0},
            "constructs": constructs,
        }
        path = tmp_path / f"{name}_review_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    by_id = {entry.construct_id: entry for entry in entries}
    source_manifest, source_hash = write_manifest(
        "source_evidence",
        [by_id["evidence_diagnosticity"], by_id["source_reliability"]],
    )
    realization_manifest, realization_hash = write_manifest(
        "realization_persistence",
        [by_id["realization_account_closure"], by_id["persistence_continuation"]],
    )
    gate_path = tmp_path / "wave1_composite_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "quality_gate_version": "1",
                "status": "approved",
                "approved": True,
                "reviewer": "test-reviewer",
                "components": [
                    {
                        "label": "source_evidence",
                        "review_manifest_path": str(source_manifest),
                        "review_manifest_sha256": source_hash,
                        "construct_ids": ["evidence_diagnosticity", "source_reliability"],
                    },
                    {
                        "label": "realization_persistence",
                        "review_manifest_path": str(realization_manifest),
                        "review_manifest_sha256": realization_hash,
                        "construct_ids": ["realization_account_closure", "persistence_continuation"],
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    validated = _validate_quality_gate(gate_path, entries)
    assert validated["composite"] is True
    assert set(validated["construct_provenance"]) == {entry.construct_id for entry in entries}
    assert [item["construct_ids"] for item in validated["components"]] == [
        ["evidence_diagnosticity", "source_reliability"],
        ["realization_account_closure", "persistence_continuation"],
    ]


def test_composite_quality_gate_rejects_duplicate_construct_selection(tmp_path: Path) -> None:
    entries = _effective_entries(
        REGISTRY,
        waves=[1],
        construct_ids=None,
        batch_size=20,
        max_output_tokens=12000,
        model=DEFAULT_MODEL,
    )
    manifest = {
        "manifest_type": "downstream_prompt_generation",
        "status": "complete_review",
        "run_mode": "review",
        "partial": True,
        "dry_run": False,
        "frozen": False,
        "provider": "openai",
        "requested_model": DEFAULT_MODEL,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "audit": {"severe_flag_count": 0},
        "constructs": [],
    }
    manifest_path = tmp_path / "empty_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    gate_path = tmp_path / "bad_composite_gate.json"
    gate_path.write_text(
        json.dumps({
            "quality_gate_version": "1",
            "status": "approved",
            "approved": True,
            "reviewer": "test-reviewer",
            "components": [
                {
                    "label": "duplicate",
                    "review_manifest_path": str(manifest_path),
                    "review_manifest_sha256": manifest_hash,
                    "construct_ids": [entries[0].construct_id, entries[0].construct_id],
                }
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate construct_ids"):
        _validate_quality_gate(gate_path, entries)


def test_source_recovery_quality_gate_pins_review_hash_across_batch_override() -> None:
    gate_path = ROOT / "configs/construct_benchmark/quality_gates/source_reliability_downstream_luna_recovery_v1.json"
    reviewed_entry = _effective_entries(
        REGISTRY,
        waves=[1],
        construct_ids=["source_reliability"],
        batch_size=20,
        max_output_tokens=12000,
        model=DEFAULT_MODEL,
    )[0]
    recovery_entry = _effective_entries(
        REGISTRY,
        waves=[1],
        construct_ids=["source_reliability"],
        batch_size=10,
        max_output_tokens=12000,
        model=DEFAULT_MODEL,
    )[0]
    assert reviewed_entry.plan_sha256 != recovery_entry.plan_sha256
    validated = _validate_quality_gate(gate_path, [recovery_entry])
    assert validated["composite"] is True
    provenance = validated["construct_provenance"]["source_reliability"]
    assert provenance["source_plan_sha256"] == recovery_entry.source_plan_sha256
    assert provenance["spec_sha256"] == recovery_entry.spec_sha256
    assert provenance["reviewed_plan_sha256"] == reviewed_entry.plan_sha256


def test_all16_full_dry_run_has_registered_downstream_counts(tmp_path: Path) -> None:
    manifest = orchestrate_downstream_prompts(
        registry_path=REGISTRY,
        waves=["all"],
        output_dir=tmp_path,
        mode="full",
        batch_size=20,
        max_output_tokens=12000,
        dry_run=True,
    )
    assert manifest["counts"]["record_count"] == 594
    assert manifest["counts"]["split_counts"] == {
        "behavior_eval": 198,
        "calibration": 198,
        "steering_eval": 198,
    }
    assert manifest["counts"]["request_count"] == 51


def test_all16_full_non_dry_run_requires_an_approved_quality_gate_before_api(tmp_path: Path) -> None:
    def should_not_run(*args, **kwargs):
        raise AssertionError("ungated all-16 generation must fail before its first API request")

    with pytest.raises(ValueError, match="quality-gate-file"):
        orchestrate_downstream_prompts(
            registry_path=REGISTRY,
            waves=["all"],
            output_dir=tmp_path,
            mode="full",
            api_key="test-only",
            request_fn=should_not_run,
            max_estimated_cost_usd=100.0,
            input_usd_per_million_tokens=0.2,
            output_usd_per_million_tokens=1.2,
            vector_reference=None,
        )


def test_review_generation_checkpoints_and_resumes_without_new_requests(tmp_path: Path) -> None:
    calls: list[str] = []

    def request(model_id, messages, options):
        calls.append(options["generation_job_id"])
        return _deterministic_response(model_id, messages, options)

    first = orchestrate_downstream_prompts(
        registry_path=REGISTRY,
        waves=[1],
        output_dir=tmp_path,
        mode="review",
        workers=4,
        batch_size=20,
        max_output_tokens=12000,
        api_key="test-only",
        request_fn=request,
        max_estimated_cost_usd=10.0,
        input_usd_per_million_tokens=3.0,
        output_usd_per_million_tokens=15.0,
        vector_reference=None,
    )
    assert first["status"] == "complete_review"
    assert first["partial"] is True
    assert first["frozen"] is False
    assert first["counts"]["record_count"] == 12
    assert set(load_prompt_records(tmp_path / "combined.csv")[0].metadata) >= {
        "generation_plan_sha256",
        "generation_plan_id",
    }
    assert {record.split for record in load_prompt_records(tmp_path / "combined.csv")} == DOWNSTREAM_SPLITS
    assert len(calls) == 12

    calls.clear()
    resumed = orchestrate_downstream_prompts(
        registry_path=REGISTRY,
        waves=[1],
        output_dir=tmp_path,
        mode="review",
        workers=2,
        batch_size=20,
        max_output_tokens=12000,
        resume=True,
        api_key="test-only",
        request_fn=lambda *args, **kwargs: pytest.fail("a valid resume must not make new requests"),
        max_estimated_cost_usd=10.0,
        input_usd_per_million_tokens=3.0,
        output_usd_per_million_tokens=15.0,
        vector_reference=None,
    )
    assert calls == []
    assert resumed["combined_sha256"] == first["combined_sha256"]


def test_non_dry_run_requires_explicit_cap_and_prices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max-estimated-cost-usd"):
        orchestrate_downstream_prompts(
            registry_path=REGISTRY,
            waves=[1],
            output_dir=tmp_path,
            mode="review",
            api_key="test-only",
            request_fn=_deterministic_response,
        )
    with pytest.raises(ValueError, match="explicit input/output token prices"):
        orchestrate_downstream_prompts(
            registry_path=REGISTRY,
            waves=[1],
            output_dir=tmp_path / "priced",
            mode="review",
            api_key="test-only",
            request_fn=_deterministic_response,
            max_estimated_cost_usd=10.0,
        )


def test_audit_fails_on_cross_pool_duplicate() -> None:
    from construct_benchmark.config import load_construct_spec
    from construct_benchmark.generation import load_generation_plan
    from scripts.generate_downstream_prompts import _PlanEntry, _canonical_sha256
    from construct_benchmark.manifests import canonical_hash

    def make_entry(construct_id: str) -> object:
        spec_path = ROOT / f"configs/construct_benchmark/constructs/{construct_id}_v1.json"
        plan_path = ROOT / f"configs/construct_benchmark/generation_plans/wave1_{construct_id}_v1.json"
        spec = load_construct_spec(spec_path)
        plan = load_generation_plan(plan_path, spec)
        return _PlanEntry(
            construct_id=spec.construct_id,
            wave=1,
            spec_path=spec_path,
            plan_path=plan_path,
            spec=spec,
            plan=plan,
            source_plan_sha256=_canonical_sha256(plan),
            spec_sha256=canonical_hash(spec.to_mapping()),
            plan_sha256=_canonical_sha256(plan),
        )

    first = make_entry("realization_account_closure")
    second = make_entry("persistence_continuation")
    from construct_benchmark.prompts import PromptRecord

    records = [
        PromptRecord(
            prompt_id=f"p{index}",
            construct_id=construct_id,
            split="behavior_eval",
            prompt_role="behavior",
            prompt_text="identical downstream task text",
            condition_id="neutral",
            prompt_family=family,
            task_id=task_id,
            parser_id=parser_id,
            expected_output_format=output_format,
            metadata={
                "generation_plan_sha256": plan_hash,
                "generation_plan_id": plan_id,
                "content_pool": pool,
                "source_model_alias": "sonnet",
                "task_metadata": task_metadata,
            },
        )
        for index, (construct_id, family, pool, task_id, parser_id, output_format, task_metadata, plan_id, plan_hash) in enumerate(
            (
                (
                    "realization_account_closure",
                    "realization_behavior_eval",
                    "realization_behavior",
                    "realization_risk_choice_v1",
                    "two_integers_risk_choice_v1",
                    "two_integers_on_separate_lines",
                    {"outcome_valence": "gain", "stake_level": "low", "risk_probability": "even"},
                    first.plan["plan_id"],
                    first.plan_sha256,
                ),
                (
                    "persistence_continuation",
                    "persistence_behavior_eval",
                    "persistence_behavior",
                    "program_renewal_allocation_v1",
                    "two_integers_sum_100_v1",
                    "two_integers_on_separate_lines",
                    {"existing_program_value": "high", "alternative_value": "high", "implementation_feasibility": "viable"},
                    second.plan["plan_id"],
                    second.plan_sha256,
                ),
            )
        )
    ]
    # The duplicate is across construct namespaces, so the canonical validator
    # permits it and the cross-pool audit must flag it.
    flags = audit_downstream_inventory(records, [first, second], vector_reference=None)
    assert flags["severe_flag_count"] >= 1


def test_audit_does_not_escalate_shared_calibration_scaffolding() -> None:
    from construct_benchmark.config import load_construct_spec
    from construct_benchmark.generation import load_generation_plan
    from scripts.generate_downstream_prompts import _PlanEntry, _canonical_sha256
    from construct_benchmark.manifests import canonical_hash
    from construct_benchmark.prompts import PromptRecord

    construct_id = "realization_account_closure"
    spec_path = ROOT / f"configs/construct_benchmark/constructs/{construct_id}_v2.json"
    plan_path = ROOT / f"configs/construct_benchmark/generation_plans/wave1_{construct_id}_v2.json"
    spec = load_construct_spec(spec_path)
    plan = load_generation_plan(plan_path, spec)
    entry = _PlanEntry(
        construct_id=spec.construct_id,
        wave=1,
        spec_path=spec_path,
        plan_path=plan_path,
        spec=spec,
        plan=plan,
        source_plan_sha256=_canonical_sha256(plan),
        spec_sha256=canonical_hash(spec.to_mapping()),
        plan_sha256=_canonical_sha256(plan),
    )
    suffix = (
        "Use a 0–100 point allocation between the options; points not assigned "
        "to Option A go to Option B. Return exactly one integer from 0 to 100."
    )
    records = [
        PromptRecord(
            prompt_id=f"calibration_{index}",
            construct_id=construct_id,
            split="calibration",
            prompt_role="calibration",
            prompt_text=(
                f"A {domain} abstract index reports results in outcome units. "
                f"Option A is the risky option: a 50% chance of {amount} outcome units "
                f"and a 50% chance of 0 outcome units. Option B is the sure option: "
                f"{amount // 2} outcome units. {suffix}"
            ),
            condition_id="neutral",
            prompt_family="realization_calibration",
            task_id="realization_risk_allocation_v2",
            parser_id="single_integer_allocation_0_to_100_v1",
            expected_output_format="single_integer_allocation_0_to_100",
            metadata={
                "generation_plan_sha256": entry.plan_sha256,
                "generation_plan_id": plan["plan_id"],
                "content_pool": "realization_calibration",
                "source_model_alias": "luna",
                "task_metadata": {
                    "outcome_valence": "neutral",
                    "stake_level": "neutral",
                    "risk_probability": "even",
                },
            },
        )
        for index, (domain, amount) in enumerate((("geometric", 12), ("pattern", 18)))
    ]

    audit = audit_downstream_inventory(records, [entry], vector_reference=None)
    assert audit["audit_version"] == "2"
    assert audit["severe_flag_count"] == 0
