from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

from construct_benchmark.behavioral_design import (
    behavioral_design_issues,
    behavioral_record_issues,
    scheduled_rows,
    validate_behavioral_design,
)
from construct_benchmark.config import load_construct_spec
from construct_benchmark.generation import (
    expected_task_metadata_assignments,
    iter_generation_jobs,
    load_generation_plan,
)
from construct_benchmark.prompts import PromptRecord


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "construct_benchmark"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_downstream_prompts import _parser_instruction_issues  # noqa: E402

REPAIRED = {
    "realization_account_closure": (
        "realization_account_closure_v4.json",
        "wave1_realization_account_closure_behavioral_repair_v4.json",
        "wave1_realization_account_closure_behavioral_repair_v4_preflight.json",
        (32, 32, 16),
    ),
    "evidence_diagnosticity": (
        "evidence_diagnosticity_v5.json",
        "wave1_evidence_diagnosticity_behavioral_repair_v5.json",
        "wave1_evidence_diagnosticity_behavioral_repair_v5_preflight.json",
        (144, 144, 32),
    ),
    "persistence_continuation": (
        "persistence_continuation_v4.json",
        "wave1_persistence_continuation_behavioral_repair_v4.json",
        "wave1_persistence_continuation_behavioral_repair_v4_preflight.json",
        (48, 48, 48),
    ),
}


def _load_candidate(construct_id: str, *, preflight: bool = False):
    spec_name, production_name, preflight_name, _ = REPAIRED[construct_id]
    spec = load_construct_spec(CONFIG_ROOT / "constructs" / spec_name)
    plan_name = preflight_name if preflight else production_name
    plan = load_generation_plan(CONFIG_ROOT / "generation_plans" / plan_name, spec)
    return spec, plan


def _cell(plan: dict, split: str) -> dict:
    return next(cell for cell in plan["cells"] if cell["split"] == split)


REALIZATION_CURVE_TEXT = (
    "The goal is to maximize total expected portfolio return. Use complete 20-point blocks. "
    "If k complete blocks are assigned to Option A, use the first k entries of Option A "
    "and the first (5-k) entries of Option B. The allocation uses a 100-point budget in five successive 20-point blocks. "
    "For Option A, the expected portfolio returns for blocks 1 through 5 are "
    "20, 18, 16, 12, and 8 units. For Option B, they are 20, 19, 15, 14, and 13 units."
)


@pytest.mark.parametrize("construct_id", tuple(REPAIRED))
def test_wave1_repair_candidates_have_valid_factorial_designs(construct_id: str) -> None:
    spec, production = _load_candidate(construct_id)
    _, preflight = _load_candidate(construct_id, preflight=True)
    expected_behavior, expected_steering, expected_calibration = REPAIRED[construct_id][3]

    assert validate_behavioral_design(spec, production)["status"] == "pass"
    assert validate_behavioral_design(spec, preflight)["status"] == "pass"
    assert scheduled_rows(_cell(production, "behavior_eval")) == scheduled_rows(
        _cell(production, "steering_eval")
    )
    assert [
        _cell(production, split)["count_per_model"]
        for split in ("behavior_eval", "steering_eval", "calibration")
    ] == [expected_behavior, expected_steering, expected_calibration]
    assert all(
        "preflight" in _cell(preflight, split)["content_pool"]
        for split in ("behavior_eval", "steering_eval", "calibration")
    )


def test_generation_assignments_include_numeric_metadata_schedule() -> None:
    spec, plan = _load_candidate("realization_account_closure")
    del spec
    job = next(iter_generation_jobs(plan, splits={"behavior_eval"}))
    assignments = tuple(
        expected_task_metadata_assignments(job, index) for index in range(job.count)
    )
    assert assignments == scheduled_rows(job.cell)
    assert {assignment["ev_bin"] for assignment in assignments} == {
        "near_indifferent",
        "modest_risk_premium",
    }
    assert {assignment["sure_value_units"] for assignment in assignments} == {-60, -20, 20, 60}


def test_realization_rejects_endpoint_or_dominant_payoff_schedule() -> None:
    spec, plan = _load_candidate("realization_account_closure")
    bad_plan = deepcopy(plan)
    _cell(bad_plan, "behavior_eval")["metadata_schedule"]["sure_value_units"][0] = 0

    issues = behavioral_design_issues(spec, bad_plan)
    assert any("strictly between" in issue for issue in issues)
    with pytest.raises(ValueError, match="behavioral design validation failed"):
        validate_behavioral_design(spec, bad_plan)


def test_evidence_rejects_unmatched_shared_stakes() -> None:
    spec, plan = _load_candidate("evidence_diagnosticity")
    bad_plan = deepcopy(plan)
    _cell(bad_plan, "behavior_eval")["metadata_schedule"]["decision_stakes_units"][0] = 99

    issues = behavioral_design_issues(spec, bad_plan)
    assert any("decision_stakes_units does not match" in issue for issue in issues)


def test_persistence_rejects_linear_tranches() -> None:
    spec, plan = _load_candidate("persistence_continuation")
    bad_plan = deepcopy(plan)
    _cell(bad_plan, "behavior_eval")["metadata_schedule"]["established_tranche_2"][0] = 8

    issues = behavioral_design_issues(spec, bad_plan)
    assert any("strictly diminishing" in issue for issue in issues)


def test_behavioral_record_validator_rejects_probe_leakage_and_duplicate_requests() -> None:
    spec, plan = _load_candidate("realization_account_closure")
    job = next(iter_generation_jobs(plan, splits={"behavior_eval"}))
    metadata = expected_task_metadata_assignments(job, 0)
    good = PromptRecord(
        prompt_id="good_behavioral_repair_record",
        construct_id=spec.construct_id,
        split="behavior_eval",
        prompt_role="behavior",
        prompt_text=(
            "Use a fictional community setting. Option A is risky and Option B is sure. "
            f"{REALIZATION_CURVE_TEXT} Return exactly one integer from 0 to 100: "
            "the points assigned to option A."
        ),
        expected_output_format=spec.independent_behavior_task["response_format"],
        metadata={"task_metadata": metadata},
    )
    assert behavioral_record_issues(spec, plan, good) == ()

    bad = PromptRecord(
        prompt_id="bad_behavioral_repair_record",
        construct_id=spec.construct_id,
        split="behavior_eval",
        prompt_role="behavior",
        prompt_text=(
            "This account closure is provisional. Return exactly one integer from 0 to 100: "
            "the points assigned to option A. Return exactly one integer from 0 to 100: "
            "the points assigned to option A. Continue processing the scenario."
        ),
        expected_output_format=spec.independent_behavior_task["response_format"],
        metadata={"task_metadata": metadata},
    )
    issues = behavioral_record_issues(spec, plan, bad)
    assert any("leakage terms" in issue for issue in issues)
    assert any("response request count" in issue for issue in issues)
    assert any("probe-only continuation suffix" in issue for issue in issues)


def test_realization_record_requires_numeric_curve_and_rejects_mix_seeding() -> None:
    spec, plan = _load_candidate("realization_account_closure")
    job = next(iter_generation_jobs(plan, splits={"behavior_eval"}))
    metadata = expected_task_metadata_assignments(job, 0)
    missing_curve = PromptRecord(
        prompt_id="missing_realization_curve",
        construct_id=spec.construct_id,
        split="behavior_eval",
        prompt_role="behavior",
        prompt_text=(
            "Option A is risky and Option B is sure. Return exactly one integer from 0 to 100: "
            "the points assigned to option A."
        ),
        expected_output_format=spec.independent_behavior_task["response_format"],
        metadata={"task_metadata": metadata},
    )
    issues = behavioral_record_issues(spec, plan, missing_curve)
    assert any("registered total point budget" in issue for issue in issues)

    seeded = PromptRecord(
        prompt_id="seeded_realization_mix",
        construct_id=spec.construct_id,
        split="behavior_eval",
        prompt_role="behavior",
        prompt_text=(
            f"Option A is risky and Option B is sure. {REALIZATION_CURVE_TEXT} "
            "Consider a mix across the two options. Return exactly one integer from 0 to 100: "
            "the points assigned to option A."
        ),
        expected_output_format=spec.independent_behavior_task["response_format"],
        metadata={"task_metadata": metadata},
    )
    issues = behavioral_record_issues(spec, plan, seeded)
    assert any("mix" in issue for issue in issues)


def test_allocation_response_suffix_option_a_is_not_marked_truncated() -> None:
    record = PromptRecord(
        prompt_id="allocation_response_contract",
        construct_id="evidence_diagnosticity",
        split="behavior_eval",
        prompt_role="behavior",
        prompt_text=(
            "Compare two independent options. Allocate 0 to 100 points to Option A. "
            "Return exactly one integer from 0 to 100: the points assigned to Option A."
        ),
        expected_output_format="single_integer_allocation_0_to_100",
        parser_id="single_integer_allocation_0_to_100_v1",
    )
    assert _parser_instruction_issues(record) == ()
