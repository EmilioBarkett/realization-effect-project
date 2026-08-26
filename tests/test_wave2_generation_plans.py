"""Protocol checks for the four Wave 2 construct specifications and plans."""

from __future__ import annotations

import json
from pathlib import Path

from construct_benchmark.config import load_construct_spec
from construct_benchmark.generation import dry_run_summary, load_generation_plan


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "construct_benchmark"
WAVE2 = (
    "reference_frame",
    "prior_weighting",
    "authority_deference",
    "exploration_exploitation",
)


def _spec_and_plan(construct_id: str):
    spec_path = BASE / "constructs" / f"{construct_id}_v1.json"
    plan_path = BASE / "generation_plans" / f"wave2_{construct_id}_v1.json"
    spec = load_construct_spec(spec_path)
    plan = load_generation_plan(plan_path, spec)
    return spec, plan


def test_wave2_specs_and_plans_cover_frozen_constructs_without_api_calls() -> None:
    for construct_id in WAVE2:
        spec, plan = _spec_and_plan(construct_id)
        assert spec.construct_id == construct_id
        assert plan["wave"] == 2
        assert plan["models"] == [{"alias": "sonnet", "model": "anthropic/claude-sonnet-4.6"}]
        assert plan["run_modes"]["review"] == {
            "purpose": "prompt_review",
            "count_per_model_per_cell": 1,
            "partial": True,
        }
        assert plan["run_modes"]["full"]["count_per_model_per_cell"] is None
        assert plan["run_modes"]["full"]["partial"] is False
        assert plan["task_composition"]["condition_carryover"] == "state_only"
        assert plan["task_composition"]["surface_text_carryover"] == "none"
        assert plan["task_composition"]["behavior_steering_pool_separation"] is True

        summary = dry_run_summary(plan)
        assert summary["complete_plan"] is True
        assert summary["records_by_split"]["direction_train"] == 200
        assert summary["records_by_split"]["direction_validation"] == 80
        assert summary["records_by_split"]["direction_heldout"] == 80
        assert summary["records_by_split"]["behavior_eval"] >= 8
        assert summary["records_by_split"]["steering_eval"] >= 8
        assert summary["records_by_split"]["calibration"] >= 8


def test_wave2_paired_cells_have_disjoint_role_pools_and_explicit_nuisance_schedules() -> None:
    for construct_id in WAVE2:
        spec, plan = _spec_and_plan(construct_id)
        cells = {cell["split"]: cell for cell in plan["cells"]}
        assert [cells[split]["count_per_model"] for split in spec.paired_splits] == [100, 40, 40]

        paired_domains = {
            domain
            for split in spec.paired_splits
            for domain in plan["content_pools"][cells[split]["content_pool"]]["domains"]
        }
        downstream_domains = {
            domain
            for pool_id, pool in plan["content_pools"].items()
            if pool["role"] != "probe"
            for domain in pool["domains"]
        }
        assert paired_domains.isdisjoint(downstream_domains)
        assert {cells["behavior_eval"]["prompt_family"], cells["steering_eval"]["prompt_family"], cells["calibration"]["prompt_family"]}.__len__() == 3

        schedule = plan["behavior_factor_schedule"]
        fields = schedule["required_item_fields"]
        required = {
            tuple(item[field] for field in fields)
            for item in schedule["required_combinations"]
        }
        for split in ("behavior_eval", "steering_eval"):
            cell = cells[split]
            observed = set(zip(*(cell["category_balance"][field] for field in fields), strict=True))
            assert observed == required
            assert cell["factor_schedule"] == "behavior_factor_schedule"
        assert set(spec.independent_behavior_task["item_metadata_schema"]["required"]) == set(fields)


def test_wave2_registry_paths_are_ready_for_specification_status_update() -> None:
    registry = json.loads((BASE / "construct_registry_v1.json").read_text(encoding="utf-8"))
    entries = {entry["construct_id"]: entry for entry in registry["entries"]}
    for construct_id in WAVE2:
        entry = entries[construct_id]
        assert entry["wave"] == 2
        assert entry["spec_path"] == f"constructs/{construct_id}_v1.json"


def test_wave2_corrections_freeze_cross_construct_nuisances() -> None:
    reference_spec, reference_plan = _spec_and_plan("reference_frame")
    assert "50/50" not in reference_spec.independent_behavior_task["prompt_template"]
    assert "realization_status_exclusion" in reference_spec.controls
    assert "numeric_benchmark_only_pair_edit" in reference_spec.controls
    assert "number_invariant_benchmark_units" in reference_spec.controls
    assert "pending" in reference_plan["forbidden_terms"]
    assert any("account-closure" in rule for rule in reference_plan["design_rules"])
    pair_checks = reference_plan["pair_quality_checks"]
    assert pair_checks["benchmark_representation"] == "number_invariant_unit_phrase"
    assert "benchmark numeric token" in pair_checks["normalized_pair_text"]["replace_only"]
    assert "optional leading minus sign" in pair_checks["normalized_pair_text"]["replace_only"]
    assert pair_checks["normalized_pair_text"]["required_identical"] is True
    assert any("singular/plural noun" in rule for rule in pair_checks["reject_if"])
    for split in ("probe_train", "probe_validation", "probe_heldout"):
        cell = next(cell for cell in reference_plan["cells"] if cell["cell_id"] == split)
        assert "number-invariant unit phrase" in cell["instructions"]
        assert "<BENCHMARK>" in cell["instructions"]

    prior_spec, prior_plan = _spec_and_plan("prior_weighting")
    prior_task = prior_spec.independent_behavior_task
    assert prior_spec.expected_direction["behavior"]["outcome"] == "prior_anchor_distance"
    assert prior_task["primary_outcome"] == "prior_anchor_distance"
    assert prior_task["item_metadata_schema"]["properties"]["evidence_strength"]["enum"] == ["moderate"]
    assert "diagnosticity_constant_at_moderate_force" in prior_spec.controls
    assert "diagnosticity" in prior_plan["forbidden_terms"]
    assert prior_spec.probe_prompt_template.endswith("Continue processing the scenario.")
    assert prior_spec.probe_prompt_template.count("Continue processing the scenario.") == 1
    prior_pair_checks = prior_plan["pair_quality_checks"]
    assert prior_pair_checks["terminal_suffix"] == "Continue processing the scenario."
    assert prior_pair_checks["suffix_occurs_exactly_once"] is True
    assert prior_pair_checks["suffix_is_final"] is True
    assert prior_pair_checks["scenario_block_required"] is True
    assert prior_pair_checks["condition_instruction_location"] == "inside Scenario block before terminal suffix"
    assert prior_pair_checks["allowed_condition_specific_difference"] == "one weighting stance clause only"
    assert prior_pair_checks["paired_response_format"]["identical_across_conditions"] is True
    assert prior_pair_checks["paired_response_format"]["location"] == "inside Scenario block before terminal suffix"
    prior_rules = " ".join(prior_plan["design_rules"])
    assert "complete instance of the registered probe_prompt_template" in prior_rules
    assert "inside the Scenario block before the terminal suffix" in prior_rules
    assert "exact final characters" in prior_rules
    assert "absolute terminator" in prior_rules
    assert "response format/request identical" in prior_rules
    assert "only condition-specific difference may be one weighting clause" in prior_rules
    prior_rejection_criteria = " ".join(prior_plan["rejection_criteria"])
    assert "whose final characters are not exactly" in prior_rejection_criteria
    assert "suffix more than once" in prior_rejection_criteria
    assert "any character or text after the suffix" in prior_rejection_criteria
    assert "inside the Scenario block before the suffix" in prior_rejection_criteria
    assert "response request, response type, answer format" in prior_rejection_criteria
    assert "changes the prior, hypothesis, evidence valence" in prior_rejection_criteria
    assert {"prior sensitive", "case evidence sensitive", "base-rate sensitive"}.issubset(
        set(prior_plan["forbidden_terms"])
    )
    prior_probe_cells = [cell for cell in prior_plan["cells"] if cell["mode"] == "paired"]
    assert {cell["cell_id"] for cell in prior_probe_cells} == {
        "probe_train",
        "probe_validation",
        "probe_heldout",
    }
    for cell in prior_probe_cells:
        instructions = cell["instructions"]
        assert "Scenario block" in instructions
        assert "response request" in instructions
        assert "identical" in instructions
        assert "terminal suffix" in instructions
        assert "exactly once" in instructions
        assert "final characters" in instructions
        assert "after the suffix" in instructions
    for split in ("behavior_eval", "steering_eval", "calibration"):
        assert set(prior_plan["cells"][[cell["split"] for cell in prior_plan["cells"]].index(split)]["category_balance"]["evidence_strength"]) == {"moderate"}

    authority_spec, authority_plan = _spec_and_plan("authority_deference")
    authority_fields = authority_spec.independent_behavior_task["item_metadata_schema"]["required"]
    assert "source_track_record" in authority_fields
    assert "peer_consensus" in authority_fields
    assert "peer_consensus_exclusion" in authority_spec.controls
    assert "consensus" in authority_plan["forbidden_terms"]
    for split in ("behavior_eval", "steering_eval", "calibration"):
        cell = next(cell for cell in authority_plan["cells"] if cell["split"] == split)
        assert set(cell["category_balance"]["source_track_record"]) == {"neutral"}
        assert set(cell["category_balance"]["peer_consensus"]) == {"none"}

    explore_spec, explore_plan = _spec_and_plan("exploration_exploitation")
    explore_task = explore_spec.independent_behavior_task
    assert "previous setback" not in explore_spec.probe_prompt_template
    assert "information_value" in explore_task["item_metadata_schema"]["required"]
    assert "information_value_crossing" in explore_spec.controls
    assert "setback" in explore_plan["forbidden_terms"]
    assert len(explore_plan["behavior_factor_schedule"]["required_combinations"]) == 16
    for split in ("behavior_eval", "steering_eval", "calibration"):
        cell = next(cell for cell in explore_plan["cells"] if cell["split"] == split)
        assert cell["count_per_model"] == 16


def test_reference_frame_benchmark_domain_and_arithmetic_rules_cover_review_and_full() -> None:
    spec, plan = _spec_and_plan("reference_frame")

    assert "domain_plausible_benchmark_values" in spec.controls
    assert "symmetric_reference_arithmetic" in spec.controls
    assert "signed_unit_semantics" in spec.controls

    spec_policy = spec.metadata["benchmark_domain_policy"]
    assert {"crates", "liters", "boards", "items"}.issubset(
        set(spec_policy["naturally_nonnegative_units"])
    )
    assert spec_policy["signed_unit_markers"] == ["signed", "net", "change", "balance"]
    assert "both benchmark values" in spec_policy["nonnegative_rule"]
    assert "negative benchmark" in spec_policy["signed_rule"]
    assert "O-d" in spec_policy["pair_arithmetic"]
    assert "O+d" in spec_policy["pair_arithmetic"]

    pair_checks = plan["pair_quality_checks"]
    domain = pair_checks["domain_plausibility"]
    assert {"crates", "liters", "boards", "items"}.issubset(
        set(domain["naturally_nonnegative_units"])
    )
    assert domain["signed_unit_markers"] == ["signed", "net", "change", "balance"]
    assert domain["review_and_full_required"] is True
    assert "both benchmark values" in domain["nonnegative_rule"].lower()
    assert "negative value is valid only" in domain["signed_rule"]
    arithmetic = pair_checks["symmetric_arithmetic"]
    assert arithmetic["formula"] == "above_reference benchmark = O-d; below_reference benchmark = O+d"
    assert arithmetic["objective_outcome_fixed"] is True

    paired_cells = [cell for cell in plan["cells"] if cell["mode"] == "paired"]
    assert {cell["cell_id"] for cell in paired_cells} == {
        "probe_train",
        "probe_validation",
        "probe_heldout",
    }
    for cell in paired_cells:
        instructions = cell["instructions"]
        assert "crates, liters, boards, items" in instructions
        assert "both benchmarks" in instructions
        assert "negative benchmark" in instructions or "negative value" in instructions
        assert "review and full" in instructions
        assert "O-d" in instructions
        assert "O+d" in instructions

    design_rules = " ".join(plan["design_rules"])
    assert "naturally nonnegative physical counts and quantities" in design_rules
    assert "A negative benchmark is permitted only" in design_rules
    assert "above-reference member uses benchmark O-d" in design_rules
    assert "both review and full generation modes" in design_rules
    rejection = " ".join(pair_checks["reject_if"])
    assert "naturally nonnegative count or quantity uses a negative benchmark value" in rejection
    assert "negative benchmark lacks an explicit signed" in rejection
    assert "not equally distant" in rejection
