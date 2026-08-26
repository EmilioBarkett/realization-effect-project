"""Focused protocol checks for the corrected Wave 1 prompt designs."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import pytest

from construct_benchmark.config import load_construct_spec
from construct_benchmark.generation import dry_run_summary, load_generation_plan


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "construct_benchmark"
CONSTRUCTS = (
    "realization_account_closure",
    "evidence_diagnosticity",
    "source_reliability",
    "persistence_continuation",
)


def _load(construct_id: str):
    spec = load_construct_spec(BASE / "constructs" / f"{construct_id}_v1.json")
    plan = load_generation_plan(
        BASE / "generation_plans" / f"wave1_{construct_id}_v1.json",
        spec,
    )
    return spec, plan


def _schedule_assignments(cell: dict, fields: list[str]) -> set[tuple[object, ...]]:
    return set(zip(*(cell["category_balance"][field] for field in fields), strict=True))


def test_wave1_plans_use_sonnet_retry_budget_and_100_40_40_pairs() -> None:
    for construct_id in CONSTRUCTS:
        spec, plan = _load(construct_id)
        assert plan["models"] == [{"alias": "sonnet", "model": "anthropic/claude-sonnet-4.6"}]
        assert plan["generation"]["max_tokens"] == 8000
        assert plan["generation"]["retries"] == 2
        assert plan["generation"]["max_items_per_request"] == 10
        cells = {cell["split"]: cell for cell in plan["cells"]}
        assert [cells[split]["count_per_model"] for split in spec.paired_splits] == [100, 40, 40]
        for split in spec.paired_splits:
            instruction = cells[split]["instructions"].lower()
            assert "duplicate" in instruction or "minimal" in instruction
            assert "within each pair" in cells[split]["instructions"].lower()


def test_realization_downstream_schedule_and_output_mapping_are_explicit() -> None:
    spec, plan = _load("realization_account_closure")
    task = spec.independent_behavior_task
    assert task["item_metadata_schema"]["required"] == [
        "outcome_valence",
        "stake_level",
        "risk_probability",
    ]
    assert "sure option" in task["prompt_template"]
    assert "risk preference second" in plan["cells"][3]["instructions"]
    fields = ["outcome_valence", "stake_level", "risk_probability"]
    expected = set(product(("gain", "loss", "neutral"), ("low", "high"), ("even", "uneven")))
    for split in ("behavior_eval", "steering_eval", "calibration"):
        cell = next(cell for cell in plan["cells"] if cell["split"] == split)
        assert _schedule_assignments(cell, fields) == expected


def test_evidence_diagnosticity_uses_likelihood_ratio_and_crossed_schedule() -> None:
    spec, plan = _load("evidence_diagnosticity")
    task = spec.independent_behavior_task
    assert "likelihood ratio" in task["prompt_template"]
    assert task["item_metadata_schema"]["properties"]["prior_probability"]["enum"] == [20, 50, 80]
    fields = ["prior_probability", "evidence_valence", "likelihood_separation"]
    expected = set(product((20, 50, 80), ("supporting", "contradicting"), ("weak", "moderate", "strong")))
    for split in ("behavior_eval", "steering_eval", "calibration"):
        cell = next(cell for cell in plan["cells"] if cell["split"] == split)
        assert _schedule_assignments(cell, fields) == expected


def test_evidence_diagnosticity_registers_strong_and_near_one_lr_bounds() -> None:
    spec, plan = _load("evidence_diagnosticity")
    conditions = {condition["condition_id"]: condition for condition in spec.contrast_conditions}
    high = conditions["high_diagnosticity"]["likelihood_ratio_constraint"]
    low = conditions["low_diagnosticity"]["likelihood_ratio_constraint"]

    assert high["minimum_absolute_log_likelihood_ratio"] == pytest.approx(math.log(10.0))
    assert high["accepted_likelihood_ratio"] == "LR >= 10 or LR <= 0.1"
    assert low["maximum_absolute_log_likelihood_ratio"] == pytest.approx(math.log(1.25))
    assert low["accepted_likelihood_ratio"] == "0.8 <= LR <= 1.25"

    checks = plan["pair_quality_checks"]
    assert checks["required_denominator"] == 100
    assert checks["same_denominator_within_pair"] is True
    assert checks["high"]["minimum_absolute_log_likelihood_ratio"] == pytest.approx(math.log(10.0))
    assert checks["low"]["maximum_absolute_log_likelihood_ratio"] == pytest.approx(math.log(1.25))
    assert any("Reject any pair" in rule for rule in plan["design_rules"])
    for split in spec.paired_splits:
        instructions = next(cell for cell in plan["cells"] if cell["split"] == split)["instructions"]
        assert "denominator of 100" in instructions or "denominator 100" in instructions
        assert "LR >=10" in instructions
        assert "0.8 and 1.25" in instructions


def test_source_reliability_crosses_track_record_without_polarity_confound() -> None:
    spec, plan = _load("source_reliability")
    task = spec.independent_behavior_task
    fields = [
        "prior_probability",
        "source_track_record",
        "testimony_valence",
        "authority_status",
        "evidence_quality",
    ]
    expected = set(
        product((20, 80), ("reliable", "unreliable"), ("supporting", "contradicting"), ("legitimate", "non_authority"), ("strong", "weak"))
    )
    assert "five-report accuracy record" in task["prompt_template"]
    assert "signed testimony update" in task["prompt_template"]
    for split in ("behavior_eval", "steering_eval", "calibration"):
        cell = next(cell for cell in plan["cells"] if cell["split"] == split)
        assert _schedule_assignments(cell, fields) == expected


def test_source_reliability_history_is_independent_and_position_balanced() -> None:
    spec, plan = _load("source_reliability")
    requirements = plan["historical_report_requirements"]
    assert requirements["report_count"] == 5
    independence_text = " ".join(
        requirements["distinctness"] + requirements["current_claim_independence"]
    ).lower()
    for forbidden_relation in ("repeat", "contradict", "paraphrase", "year/date"):
        assert forbidden_relation in independence_text
    assert "new focal entity and event" in independence_text

    schedule = plan["historical_accuracy_position_schedule"]
    assert schedule["positions"] == [1, 2, 3, 4, 5]
    assert schedule["position_index_base"] == 1
    assert schedule["full_request_size"] == 10
    assert schedule["full_request_counts_by_position"] == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}
    assert schedule["condition_mapping"]["reliable_source"] == {
        "minority_status": "inaccurate",
        "majority_status": "accurate",
    }
    assert schedule["condition_mapping"]["unreliable_source"] == {
        "minority_status": "accurate",
        "majority_status": "inaccurate",
    }
    assert "identical" in schedule["pairing_rule"].lower()
    rules = " ".join(plan["design_rules"]).lower()
    assert "fictional" in rules
    assert "non-retrievable" in rules
    assert "famous real museums" in rules
    assert "mona lisa" in plan["forbidden_terms"]
    assert "louvre" in plan["forbidden_terms"]
    paired_schema = spec.metadata["paired_item_metadata_schema"]
    assert paired_schema["required"] == ["minority_report_position"]
    assert paired_schema["properties"]["minority_report_position"]["enum"] == [1, 2, 3, 4, 5]

    cells = {cell["split"]: cell for cell in plan["cells"]}
    for split, expected_count in (
        ("direction_train", 100),
        ("direction_validation", 40),
        ("direction_heldout", 40),
    ):
        split_schedule = schedule["split_schedules"][split]
        assert split_schedule["count_per_model"] == expected_count
        counts = split_schedule["counts_by_position"]
        assert set(counts) == {"1", "2", "3", "4", "5"}
        assert sum(counts.values()) == expected_count
        assert len(set(counts.values())) == 1
        assert cells[split]["count_per_model"] == expected_count
        assert "minority_report_position" in cells[split]["instructions"]
        pair_schedule = cells[split]["paired_metadata_schedule"]
        assert pair_schedule["field"] == "minority_report_position"
        assert pair_schedule["request_size"] == 10
        assert pair_schedule["repeats_per_request"] == 2
        assert pair_schedule["seed"] == split_schedule["seed"]

    for split in ("behavior_eval", "steering_eval", "calibration"):
        instruction = cells[split]["instructions"].lower()
        assert "five distinct historical report events" in instruction
        assert "year-shift variants" in instruction
        assert "new entity and event" in instruction

    assert len(
        {
            schedule["split_schedules"][split]["seed"]
            for split in ("direction_train", "direction_validation", "direction_heldout")
        }
    ) == 3
    assert spec.independent_behavior_task["item_metadata_schema"]["required"] == [
        "prior_probability",
        "source_track_record",
        "testimony_valence",
        "authority_status",
        "evidence_quality",
    ]


def test_persistence_uses_independent_program_task_and_conditional_direction() -> None:
    spec, plan = _load("persistence_continuation")
    task = spec.independent_behavior_task
    assert task["task_id"] == "program_renewal_allocation_v1"
    assert "maintaining an existing program" in task["prompt_template"]
    assert "after a setback" not in task["prompt_template"]
    assert "largest when implementation is viable" in spec.expected_direction["behavior"]["expected_by_existing_program_value"]["high"]
    fields = ["existing_program_value", "alternative_value", "implementation_feasibility"]
    expected = set(product(("high", "low"), ("high", "low"), ("viable", "constrained")))
    for split in ("behavior_eval", "steering_eval", "calibration"):
        cell = next(cell for cell in plan["cells"] if cell["split"] == split)
        assert _schedule_assignments(cell, fields) == expected


def test_corrected_wave1_full_dry_run_is_complete_without_api_calls() -> None:
    expected_single_counts = {
        "realization_account_closure": {"behavior_eval": 12, "calibration": 12, "steering_eval": 12},
        "evidence_diagnosticity": {"behavior_eval": 18, "calibration": 18, "steering_eval": 18},
        "source_reliability": {"behavior_eval": 32, "calibration": 32, "steering_eval": 32},
        "persistence_continuation": {"behavior_eval": 8, "calibration": 8, "steering_eval": 8},
    }
    for construct_id in CONSTRUCTS:
        _, plan = _load(construct_id)
        summary = dry_run_summary(plan)
        assert summary["complete_plan"] is True
        assert summary["records_by_split"]["direction_train"] == 200
        assert summary["records_by_split"]["direction_validation"] == 80
        assert summary["records_by_split"]["direction_heldout"] == 80
        for split, count in expected_single_counts[construct_id].items():
            assert summary["records_by_split"][split] == count
