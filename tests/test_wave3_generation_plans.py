from __future__ import annotations

import itertools
import json
from pathlib import Path

from construct_benchmark.config import load_construct_spec
from construct_benchmark.generation import dry_run_summary, load_generation_plan


ROOT = Path(__file__).resolve().parents[1]
WAVE3_CONSTRUCT_IDS = (
    "ambiguity_orientation",
    "causal_interpretation",
    "consensus_conformity",
    "plan_replanning",
)


def _paths(construct_id: str) -> tuple[Path, Path]:
    return (
        ROOT / "configs/construct_benchmark/constructs" / f"{construct_id}_v1.json",
        ROOT / "configs/construct_benchmark/generation_plans" / f"wave3_{construct_id}_v1.json",
    )


def test_wave3_specs_and_plans_are_complete_and_sonnet_only() -> None:
    for construct_id in WAVE3_CONSTRUCT_IDS:
        spec_path, plan_path = _paths(construct_id)
        spec = load_construct_spec(spec_path)
        plan = load_generation_plan(plan_path, spec)
        assert plan["models"] == [{"alias": "sonnet", "model": "anthropic/claude-sonnet-4.6"}]
        assert plan["run_modes"]["review"] == {
            "purpose": "prompt_review",
            "count_per_model_per_cell": 1,
            "partial": True,
        }
        assert plan["run_modes"]["full"] == {
            "purpose": "frozen_full_inventory",
            "count_per_model_per_cell": None,
            "partial": False,
        }

        cells_by_split = {cell["split"]: cell for cell in plan["cells"]}
        assert [cells_by_split[split]["count_per_model"] for split in spec.paired_splits] == [100, 40, 40]
        assert set(cells_by_split) == set(spec.required_splits)
        paired_records = 2 * sum(cells_by_split[split]["count_per_model"] for split in spec.paired_splits)
        single_records = sum(cell["count_per_model"] for cell in plan["cells"] if cell["mode"] == "single")
        assert dry_run_summary(plan)["expected_record_count"] == paired_records + single_records

        pools = plan["content_pools"]
        assert {pool["role"] for pool in pools.values()} == {"probe", "behavior", "steering", "calibration"}
        assert len({tuple(pool["domains"]) for pool in pools.values()}) == len(pools)

        schedule = plan["behavior_factor_schedule"]
        factor_names = tuple(schedule["factors"])
        expected = {
            tuple(values)
            for values in itertools.product(*(schedule["factors"][name] for name in factor_names))
        }
        actual = {tuple(item[name] for name in factor_names) for item in schedule["required_combinations"]}
        assert actual == expected
        for split in ("behavior_eval", "steering_eval"):
            cell = cells_by_split[split]
            observed = {
                tuple(cell["category_balance"][name][index] for name in factor_names)
                for index in range(cell["count_per_model"])
            }
            assert observed == expected
            assert cell["factor_schedule"] == "behavior_factor_schedule"


def test_wave3_plan_files_have_no_stale_model_or_duplicate_pool_metadata() -> None:
    for construct_id in WAVE3_CONSTRUCT_IDS:
        _, plan_path = _paths(construct_id)
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        assert all(model["model"] == "anthropic/claude-sonnet-4.6" for model in raw["models"])
        assert len(raw["content_pools"]) == len({pool_id for pool_id in raw["content_pools"]})
        assert raw["task_composition"]["behavior_steering_pool_separation"] is True


def test_wave3_construct_separation_controls_are_explicit() -> None:
    ambiguity_spec_path, ambiguity_plan_path = _paths("ambiguity_orientation")
    ambiguity_spec = load_construct_spec(ambiguity_spec_path)
    ambiguity_plan = load_generation_plan(ambiguity_plan_path, ambiguity_spec)
    ambiguity_task = ambiguity_spec.independent_behavior_task
    assert "ambiguity_range_width" in ambiguity_task["item_metadata_schema"]["required"]
    assert "bounded numeric probability interval" in ambiguity_task["prompt_template"]
    assert "epistemic uncertainty" in ambiguity_plan["forbidden_terms"]
    assert len(ambiguity_plan["behavior_factor_schedule"]["required_combinations"]) == 16

    causal_spec_path, causal_plan_path = _paths("causal_interpretation")
    causal_spec = load_construct_spec(causal_spec_path)
    causal_plan = load_generation_plan(causal_plan_path, causal_spec)
    causal_values = set(causal_spec.independent_behavior_task["item_metadata_schema"]["properties"]["task_mode"]["enum"])
    assert causal_values == {"external_assignment", "case_selection"}
    assert {"causal", "correlational", "intervention", "association", "effect"}.issubset(
        set(causal_plan["forbidden_terms"])
    )
    causal_instructions = " ".join(cell["instructions"] for cell in causal_plan["cells"] if cell["mode"] == "paired")
    assert "assigned externally" in causal_instructions
    assert "selected by the cases" in causal_instructions

    consensus_spec_path, consensus_plan_path = _paths("consensus_conformity")
    consensus_spec = load_construct_spec(consensus_spec_path)
    consensus_plan = load_generation_plan(consensus_plan_path, consensus_spec)
    consensus_fields = consensus_spec.independent_behavior_task["item_metadata_schema"]["required"]
    assert "private_evidence_relation" in consensus_fields
    assert len(consensus_plan["behavior_factor_schedule"]["required_combinations"]) == 16
    assert "consensus" in consensus_plan["forbidden_terms"]
    consensus_instructions = " ".join(
        cell["instructions"] for cell in consensus_plan["cells"] if cell["mode"] == "paired"
    )
    assert "private evidence relation" in consensus_instructions
    consensus_rules = " ".join(consensus_plan["design_rules"])
    suffix = "Continue processing the scenario."
    assert "final characters are exactly" in consensus_rules
    assert suffix in consensus_rules
    assert "inside the Scenario block before the terminal suffix" in consensus_rules
    assert "absolute terminator" in consensus_rules
    assert "identical downstream response format and request" in consensus_rules
    assert "explicit answer, answer choice, date, number, named outcome" in consensus_rules
    rejection_criteria = " ".join(consensus_plan["rejection_criteria"])
    assert "whose final characters are not exactly" in rejection_criteria
    assert "any character or text after the suffix" in rejection_criteria
    assert "condition-specific answer or stance instruction is absent from the Scenario block" in rejection_criteria
    assert "downstream response format or request differs across conditions" in rejection_criteria
    assert "explicit answer, answer choice, date, number, named outcome" in rejection_criteria
    for cell in consensus_plan["cells"]:
        instructions = cell["instructions"]
        assert suffix in instructions
        assert "final characters" in instructions
        assert "after it" in instructions
        if cell["mode"] == "paired":
            assert "same downstream response format and request" in instructions
            assert "mirror any necessary content verbatim in both" in instructions

    replanning_spec_path, replanning_plan_path = _paths("plan_replanning")
    replanning_spec = load_construct_spec(replanning_spec_path)
    replanning_plan = load_generation_plan(replanning_plan_path, replanning_spec)
    assert replanning_spec.parsing_rules["field_order"] == [
        "revised_plan_allocation",
        "original_plan_allocation",
    ]
    assert {"setback", "sunk cost", "abandonment", "persistence", "competing goal", "distractor"}.issubset(
        set(replanning_plan["forbidden_terms"])
    )
    replanning_instructions = " ".join(
        cell["instructions"] for cell in replanning_plan["cells"] if cell["mode"] == "paired"
    )
    assert "feasible and valuable" in replanning_instructions
    assert "sentence order" in replanning_instructions


def test_plan_replanning_preserve_condition_keeps_original_means_feasible() -> None:
    spec_path, plan_path = _paths("plan_replanning")
    spec = load_construct_spec(spec_path)
    plan = load_generation_plan(plan_path, spec)

    conditions = {condition["condition_id"]: condition for condition in spec.contrast_conditions}
    assert "remain feasible and usable" in conditions["adapt_plan"]["definition"]
    assert "remain feasible and usable" in conditions["preserve_plan"]["definition"]
    assert "less suitable or efficient" in conditions["preserve_plan"]["definition"]
    assert "Both the original and alternative means remain feasible and usable" in spec.probe_prompt_template
    assert "less suitable or efficient" in spec.independent_behavior_task["prompt_template"]
    assert "original_means_feasibility_control" in spec.controls
    assert "relative_means_suitability_control" in spec.controls

    forbidden = set(plan["forbidden_terms"])
    assert {
        "not feasible",
        "infeasible",
        "impossible",
        "cannot work",
        "can't work",
        "unavailable",
        "blocked",
        "unusable",
    } <= forbidden

    pair_checks = plan["pair_quality_checks"]
    assert "remain feasible and usable" in pair_checks["feasibility_requirement"]
    assert "less suitable or efficient" in pair_checks["feasibility_requirement"]
    assert "valuable and feasible" in pair_checks["goal_requirement"]
    assert pair_checks["normalized_pair_text"] == {
        "replace_only": "the means-choice clause",
        "required_identical": True,
    }
    rejection_criteria = " ".join(plan["rejection_criteria"])
    assert "not feasible" in rejection_criteria
    assert "impossible" in rejection_criteria
    assert "cannot work" in rejection_criteria
    assert "original and alternative means do not both remain feasible and usable" in rejection_criteria

    for cell in plan["cells"]:
        if cell["mode"] != "paired":
            continue
        instructions = cell["instructions"]
        assert "both" in instructions.lower()
        assert "remain feasible and usable" in instructions
        assert "less suitable or efficient" in instructions
        assert "means-choice clause" in instructions or "vary only the means-choice clause" in instructions


def test_consensus_stance_surface_schedule_is_preregistered_and_pair_safe() -> None:
    spec_path, plan_path = _paths("consensus_conformity")
    spec = load_construct_spec(spec_path)
    plan = load_generation_plan(plan_path, spec)
    schedule = plan["stance_surface_schedule"]

    assert schedule["assignment_unit"] == "generation_job"
    assert schedule["cycle_order"] == ["weight", "guide", "basis", "priority"]
    assert schedule["cell_start_variant"] == {
        "probe_train": "weight",
        "probe_validation": "guide",
        "probe_heldout": "basis",
    }
    assert schedule["carrier"] == "Decision stance: "
    assert "lowercase first word" in schedule["capitalization_rule"]
    assert "no intervening line break" in schedule["capitalization_rule"]

    variants = {variant["variant_id"]: variant for variant in schedule["variants"]}
    assert set(variants) == set(schedule["cycle_order"])
    for variant_id in schedule["cycle_order"]:
        variant = variants[variant_id]
        assert set(variant) == {"variant_id", "follow_consensus", "independent_judgment"}
        for condition_id in ("follow_consensus", "independent_judgment"):
            clause = variant[condition_id]
            assert clause[0].islower()
            assert "\n" not in clause
            assert not clause.startswith(("Treat ", "Weigh ", "Use ", "Let ", "Give ", "Follow ", "Assess "))

    paired_cells = [cell for cell in plan["cells"] if cell["mode"] == "paired"]
    assert {cell["cell_id"] for cell in paired_cells} == set(schedule["cell_start_variant"])
    design_rules = " ".join(plan["design_rules"])
    assert "stance_surface_schedule" in design_rules
    assert "same `Decision stance: ` carrier" in design_rules
    assert "cycle the variants" in design_rules
    assert "Begin probe_train with the `weight` variant" in design_rules
    assert "never begin a sentence or line with Treat, Weigh, Use, Let, Give, Follow, or Assess" in design_rules
