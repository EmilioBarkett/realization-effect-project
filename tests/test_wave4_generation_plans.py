from __future__ import annotations

import hashlib
import json
from pathlib import Path

from construct_benchmark.config import load_construct_spec
from construct_benchmark.generation import dry_run_summary, generate_prompt_records, load_generation_plan
from construct_benchmark.prompts import validate_prompt_records


ROOT = Path(__file__).resolve().parents[1]
WAVE4_IDS = (
    "temporal_orientation",
    "epistemic_uncertainty",
    "reciprocity_obligation",
    "goal_shielding",
)


def _deterministic_response(model_id, messages, options):
    del model_id
    payload = json.loads(messages[1]["content"])
    count = int(payload["count"])
    domains = payload["assigned_content_domains"]
    nonce = hashlib.sha256(options["generation_job_id"].encode("utf-8")).hexdigest()[:10]
    if payload["generation_mode"] == "paired":
        condition_ids = [item["condition_id"] for item in payload["condition_definitions"]]
        data = {
            "pairs": [
                {
                    "pair_id": f"pair_{index}",
                    "content_domain": domains[index],
                    "prompts": [
                        {
                            "condition_id": condition_id,
                            "prompt_text": f"Scenario {nonce}_{index}_{condition_id}.",
                        }
                        for condition_id in condition_ids
                    ],
                    "notes": "deterministic test response",
                }
                for index in range(count)
            ]
        }
    else:
        schema = payload["item_metadata_schema"]
        assignments = payload.get("required_category_assignments", [])
        prompts = []
        for index in range(count):
            metadata = {}
            for field, field_schema in schema["properties"].items():
                if field_schema.get("enum"):
                    metadata[field] = field_schema["enum"][0]
                elif field_schema["type"] == "boolean":
                    metadata[field] = True
                else:
                    metadata[field] = 0
            metadata.update(assignments[index] if index < len(assignments) else {})
            prompts.append(
                {
                    "variant_id": f"variant_{index}",
                    "content_domain": domains[index],
                    "task_metadata": metadata,
                    "prompt_text": f"Independent item {nonce}_{index}.",
                    "notes": "deterministic test response",
                }
            )
        data = {"prompts": prompts}
    return {"choices": [{"message": {"content": json.dumps(data)}}]}


def test_wave4_plans_have_frozen_pair_counts_and_sonnet_only() -> None:
    for construct_id in WAVE4_IDS:
        spec = load_construct_spec(
            ROOT / f"configs/construct_benchmark/constructs/{construct_id}_v1.json"
        )
        plan = load_generation_plan(
            ROOT / f"configs/construct_benchmark/generation_plans/wave4_{construct_id}_v1.json",
            spec,
        )
        assert plan["models"] == [{"alias": "sonnet", "model": "anthropic/claude-sonnet-4.6"}]
        counts = {cell["split"]: cell["count_per_model"] for cell in plan["cells"]}
        assert counts["direction_train"] == 100
        assert counts["direction_validation"] == 40
        assert counts["direction_heldout"] == 40
        assert counts["behavior_eval"] == counts["steering_eval"] == counts["calibration"] == 8

        summary = dry_run_summary(plan)
        assert summary["expected_record_count"] == 384
        assert summary["records_by_split"] == {
            "behavior_eval": 8,
            "calibration": 8,
            "direction_heldout": 80,
            "direction_train": 200,
            "direction_validation": 80,
            "steering_eval": 8,
        }


def test_wave4_plans_emit_complete_canonical_records_without_api_calls() -> None:
    for construct_id in WAVE4_IDS:
        spec = load_construct_spec(
            ROOT / f"configs/construct_benchmark/constructs/{construct_id}_v1.json"
        )
        plan = load_generation_plan(
            ROOT / f"configs/construct_benchmark/generation_plans/wave4_{construct_id}_v1.json",
            spec,
        )
        result = generate_prompt_records(
            plan,
            spec,
            api_key="test-only",
            request_fn=_deterministic_response,
        )
        assert result.complete is True
        assert len(result.records) == 384
        validate_prompt_records(result.records, {construct_id: spec})
        assert {record.metadata["source_model_alias"] for record in result.records} == {"sonnet"}
        assert {record.split for record in result.records} == set(spec.required_splits)


def test_wave4_paired_instructions_do_not_duplicate_stale_pair_counts() -> None:
    for construct_id in ("temporal_orientation", "epistemic_uncertainty"):
        plan = json.loads(
            (ROOT / f"configs/construct_benchmark/generation_plans/wave4_{construct_id}_v1.json").read_text(
                encoding="utf-8"
            )
        )
        for cell in plan["cells"]:
            if cell["mode"] != "paired":
                continue
            instruction = cell["instructions"]
            assert "Generate 50" not in instruction
            assert "Generate 20" not in instruction
            assert "Generate 100" not in instruction
            assert "Generate 40" not in instruction


def test_wave4_audit_controls_are_frozen_and_construct_specific() -> None:
    uncertainty = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/epistemic_uncertainty_v1.json"
    )
    uncertainty_plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave4_epistemic_uncertainty_v1.json",
        uncertainty,
    )
    uncertainty_fields = uncertainty.independent_behavior_task["item_metadata_schema"]["required"]
    assert "check_count" in uncertainty_fields
    assert "check_type" in uncertainty_fields
    assert uncertainty.independent_behavior_task["item_metadata_schema"]["properties"]["check_count"]["enum"] == ["one"]
    assert uncertainty.independent_behavior_task["item_metadata_schema"]["properties"]["check_type"]["enum"] == ["binary"]
    assert "lottery" in uncertainty_plan["forbidden_terms"]

    reciprocity = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/reciprocity_obligation_v1.json"
    )
    reciprocity_plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave4_reciprocity_obligation_v1.json",
        reciprocity,
    )
    reciprocity_fields = reciprocity.independent_behavior_task["item_metadata_schema"]["required"]
    for field in ("prior_contact", "requester_status", "requester_likability", "requester_familiarity"):
        assert field in reciprocity_fields
    assert "neutral_prior_contact_control" in reciprocity.controls
    assert "status_likability_familiarity_matching" in reciprocity.controls
    assert "matched_current_requests" not in reciprocity.controls
    assert all(
        set(next(cell for cell in reciprocity_plan["cells"] if cell["split"] == split)["category_balance"]["prior_contact"])
        == {"matched_neutral"}
        for split in ("behavior_eval", "steering_eval", "calibration")
    )

    goal = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/goal_shielding_v1.json"
    )
    assert "continuing the focal task" not in goal.independent_behavior_task["prompt_template"]
    assert "focal_task_attention" == goal.independent_behavior_task["primary_outcome"]

    temporal = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/temporal_orientation_v1.json"
    )
    warning = temporal.expected_direction["behavior"]["warning"]
    assert "discount-rate mechanism" in warning
    assert "analysis_caveat" in temporal.metadata


def test_epistemic_uncertainty_pair_contract_preserves_evidence_content() -> None:
    spec = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/epistemic_uncertainty_v1.json"
    )
    plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave4_epistemic_uncertainty_v1.json",
        spec,
    )

    conditions = {condition["condition_id"]: condition for condition in spec.contrast_conditions}
    assert "identical evidence" in conditions["resolved_state"]["definition"]
    assert "decision-maker regards the focal question as settled" in conditions["resolved_state"]["definition"]
    assert "identical evidence" in conditions["unresolved_state"]["definition"]
    assert "decision-maker regards the focal question as still open" in conditions["unresolved_state"]["definition"]

    contract = spec.metadata["paired_probe_contract"]
    assert contract["resolved_status_clause"] == (
        "The decision-maker treats the focal question as settled for this decision, given the report and its stated limitations."
    )
    assert contract["unresolved_status_clause"] == (
        "The decision-maker treats the focal question as still open for this decision, given the report and its stated limitations."
    )
    assert "the evidence does not or fails to support the claim" in contract["forbidden_unresolved_semantics"]
    assert "the evidence is objectively decisive or objectively indecisive" in contract["forbidden_unresolved_semantics"]
    assert "the status follows mechanically from a threshold, cutoff, pass/fail rule, or arithmetic comparison" in contract[
        "forbidden_unresolved_semantics"
    ]

    checks = plan["pair_quality_checks"]
    assert checks["condition_edit"] == "one information-status clause only"
    assert checks["status_clause_contract"]["evidence_content_identical"] is True
    assert "decision-maker treats the focal question as settled" in checks["status_clause_contract"]["resolved"]
    assert "decision-maker treats the focal question as still open" in checks["status_clause_contract"]["unresolved"]
    assert checks["normalized_pair_text"]["replace_only"] == "the information-status clause"
    assert checks["normalized_pair_text"]["required_identical"] is True
    reject_if = " ".join(checks["reject_if"]).lower()
    assert "changes evidence valence" in reject_if
    assert "does not or fails to support" in reject_if
    assert "weak, insufficient, missing, or lower quality" in reject_if
    assert "not genuinely borderline or equivocal" in reject_if
    assert "regulatory/legal threshold" in reject_if
    assert "objectively decisive or objectively indecisive" in reject_if

    evidence_profile = checks["evidence_profile"]
    assert len(evidence_profile["required"]) >= 3
    evidence_required = " ".join(evidence_profile["required"]).lower()
    assert "different implications" in evidence_required
    assert "reasonable decision-maker" in evidence_required
    assert "hard cutoff" in evidence_required
    evidence_reject_if = " ".join(evidence_profile["reject_if"]).lower()
    assert "single conclusive measurement" in evidence_reject_if
    assert "exact count" in evidence_reject_if
    assert "arithmetic comparison" in evidence_reject_if

    plausibility = checks["plausibility_review"]
    assert "same evidence" in plausibility["pass_condition"]
    assert "either" in plausibility["pass_condition"]
    plausibility_reject_if = " ".join(plausibility["reject_if"]).lower()
    assert "invent a missing fact" in plausibility_reject_if
    assert "mechanically decides" in plausibility_reject_if

    rejection_criteria = " ".join(plan["rejection_criteria"]).lower()
    assert "decision-maker treats the focal question as settled" in rejection_criteria
    assert "decision-maker treats it as still open" in rejection_criteria
    assert "does not or fails to support" in rejection_criteria
    assert "normalized prompt texts are not identical" in rejection_criteria
    assert "plainly all one way" in rejection_criteria
    assert "regulatory/legal standard" in rejection_criteria
    assert "exact count" in rejection_criteria

    forbidden = set(plan["forbidden_terms"])
    assert {
        "does not support",
        "doesn't support",
        "fails to support",
        "weak evidence",
        "insufficient evidence",
        "missing evidence",
        "the evidence settles",
        "the report settles",
        "objectively settled",
        "objectively decisive",
        "conclusive",
        "definitive",
        "inconclusive",
        "regulatory threshold",
        "hard cutoff",
        "pass/fail gate",
        "exact count",
        "percentage",
        "ratio",
    } <= forbidden

    for cell in plan["cells"]:
        if cell["mode"] != "paired":
            continue
        instructions = cell["instructions"].lower()
        assert "borderline or equivocal" in instructions
        assert "different implications" in instructions
        assert "reasonable decision-maker" in instructions
        assert "treats the focal question as settled" in instructions
        assert "treats it as still open" in instructions or "treats the focal question as still open" in instructions
        assert "hard thresholds" in instructions or "hard cutoffs" in instructions
        assert "exact counts" in instructions
        assert "evidence" in instructions and "valence" in instructions


def test_epistemic_uncertainty_rejects_objectively_decisive_or_indecisive_probes() -> None:
    plan = json.loads(
        (
            ROOT / "configs/construct_benchmark/generation_plans/wave4_epistemic_uncertainty_v1.json"
        ).read_text(encoding="utf-8")
    )
    forbidden = set(plan["forbidden_terms"])
    for term in (
        "settles the focal question",
        "leaves the focal question open",
        "the focal question is settled",
        "the focal question remains open",
        "objectively decisive",
        "objectively indecisive",
        "decisive evidence",
        "indecisive evidence",
        "conclusive",
        "proves",
        "determines the answer",
        "inconclusive",
        "regulatory limit",
        "legal threshold",
        "hard threshold",
        "pass/fail",
        "above the limit",
        "exceeds the threshold",
        "exact cutoff",
        "arithmetic comparison",
    ):
        assert term in forbidden

    rules = " ".join(plan["design_rules"]).lower()
    assert "genuinely borderline or equivocal" in rules
    assert "different implications" in rules
    assert "credible interpretive margin" in rules
    assert "mechanically determine the answer" in rules
    assert "decision-maker's assessment" in rules


def test_goal_shielding_pair_contract_is_timing_neutral_and_minimal() -> None:
    spec = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/goal_shielding_v1.json"
    )
    plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave4_goal_shielding_v1.json",
        spec,
    )

    template = spec.probe_prompt_template.lower()
    assert "shared decision point" in template
    assert "do not infer a different order of later completion" in template

    rules = " ".join(plan["design_rules"]).lower()
    assert "currently available and feasible" in rules
    assert "timing and order neutral" in rules
    assert "present-tense decision-point clause" in rules
    assert "future response" in rules
    assert "verbatim suffix" in rules

    checks = plan["pair_quality_checks"]
    assert checks["normalized_pair_text"]["replace_only"] == "the attentional-priority clause"
    assert checks["normalized_pair_text"]["required_identical"] is True
    assert "now/later/after/before/then" in checks["timing_neutrality"]
    reject_if = " ".join(checks["reject_if"]).lower()
    assert "condition-specific now/later/after/before/then cue" in reject_if
    assert "closely length-matched minimal contrast" in reject_if
    assert "exact terminal suffix" in reject_if

    rejection_criteria = " ".join(plan["rejection_criteria"]).lower()
    assert "final characters are not exactly" in rejection_criteria
    assert "future action" in rejection_criteria
    assert "completion order" in rejection_criteria
    assert "normalized prompt texts are not identical" in rejection_criteria
    assert "closely length-matched minimal contrast" in rejection_criteria

    paired_cells = [cell for cell in plan["cells"] if cell["mode"] == "paired"]
    assert {cell["split"] for cell in paired_cells} == {
        "direction_train",
        "direction_validation",
        "direction_heldout",
    }
    for cell in paired_cells:
        instructions = cell["instructions"].lower()
        assert "available and feasible" in instructions
        assert "present-tense" in instructions
        assert "timing" in instructions
        assert "verbatim" in instructions and "suffix" in instructions
        assert "minimal" in instructions


def test_reciprocity_probe_plan_forbids_interpretations_and_freezes_shared_frame() -> None:
    plan = json.loads(
        (
            ROOT
            / "configs/construct_benchmark/generation_plans/wave4_reciprocity_obligation_v1.json"
        ).read_text(encoding="utf-8")
    )
    forbidden = set(plan["forbidden_terms"])
    assert {
        "socially expected",
        "social expectation",
        "expected in return",
        "not expected",
        "no special reason",
        "reason to respond",
        "reason to assist",
        "reason to help",
        "respond in kind",
        "obligation",
        "owe",
        "owes",
        "repay",
        "return the favor",
        "did not hand",
        "placed beside",
        "kept from",
        "offered but",
        "proximity",
    } <= forbidden

    rules = " ".join(plan["design_rules"]).lower()
    assert "prior-benefit event only" in rules
    assert "copy the current-request block verbatim" in rules
    assert "same neutral current-request/contact framing" in rules
    assert "terminal suffix" in rules
    assert "observable contact actions" in rules
    assert "completed, ordinary, observable contact" in rules
    assert "actual completed transfer" in rules
    assert "withholding" in rules

    paired_cells = [cell for cell in plan["cells"] if cell["mode"] == "paired"]
    assert {cell["split"] for cell in paired_cells} == {
        "direction_train",
        "direction_validation",
        "direction_heldout",
    }
    for cell in paired_cells:
        instruction = cell["instructions"].lower()
        assert "concrete prior-exchange" in instruction
        assert (
            "copy the complete neutral current-request block" in instruction
            or "copy the full current request" in instruction
            or ("complete current-request block" in instruction and "identical" in instruction)
        )
        assert "terminal suffix" in instruction
        assert "observable" in instruction
        assert "social norm" in instruction or "social-norm" in instruction
        assert "completed" in instruction
        assert "neutral contact" in instruction
        assert "did not hand" in instruction

    checks = plan["pair_quality_checks"]
    assert checks["condition_edit"] == "the short prior-contact span only"
    assert checks["neutral_contact_contract"]["shared_frame_identical"] is True
    assert "completed ordinary observable contact" in checks["neutral_contact_contract"]["required"]
    assert "withheld" in checks["neutral_contact_contract"]["prohibited"]
    assert checks["neutral_contact_contract"]["examples_to_reject"] == [
        "did not hand",
        "placed beside",
        "kept from",
        "offered but",
    ]
    reject_if = " ".join(checks["reject_if"]).lower()
    assert "merely proximate" in reject_if
    assert "actual completed benefit transfer" in reject_if
    assert "did not hand" in reject_if

    rejection_criteria = " ".join(plan["rejection_criteria"]).lower()
    assert "completed neutral observable contact" in rejection_criteria
    assert "places or leaves a benefit beside" in rejection_criteria
    assert "failed-offer or withholding cue" in rejection_criteria
    assert "actual completed transfer" in rejection_criteria


def test_reciprocity_probe_wrapper_is_neutral_and_complete() -> None:
    spec = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/reciprocity_obligation_v1.json"
    )
    template = spec.probe_prompt_template
    lowered = template.lower()
    assert "concrete sequence of prior contact" in lowered
    assert "current request" in lowered
    assert "who did what" in lowered
    assert "reason to respond" not in lowered
    assert "respond in kind" not in lowered
    assert "social obligation" not in lowered
    assert template.count("{scenario}") == 1
    assert template.endswith("Continue processing the scenario.")


def test_reciprocity_plan_rejects_explicit_interpretation_in_probe_text() -> None:
    spec = load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs/reciprocity_obligation_v1.json"
    )
    plan = load_generation_plan(
        ROOT / "configs/construct_benchmark/generation_plans/wave4_reciprocity_obligation_v1.json",
        spec,
    )

    def forbidden_response(model_id, messages, options):
        del model_id, messages
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "pairs": [
                                    {
                                        "pair_id": "pair_0",
                                        "content_domain": "neighborhood_projects",
                                        "prompts": [
                                            {
                                                "condition_id": condition_id,
                                                "prompt_text": (
                                                    "Scenario: The prior event was socially expected to create a return. "
                                                    "Continue processing the scenario."
                                                ),
                                            }
                                            for condition_id in (
                                                "reciprocal_exchange",
                                                "no_reciprocal_exchange",
                                            )
                                        ],
                                        "notes": "test",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }

    import pytest

    with pytest.raises(ValueError, match="forbidden term"):
        generate_prompt_records(
            plan,
            spec,
            api_key="test-only",
            request_fn=forbidden_response,
            count_per_model_override=1,
            limit_jobs=1,
        )


def test_wave4_plan_forbidden_cues_are_rejected_by_generation_validator() -> None:
    import pytest

    cases = (
        (
            "epistemic_uncertainty",
            "wave4_epistemic_uncertainty_v1.json",
            "The identical report does not support the claim. Continue processing the scenario.",
        ),
        (
            "reciprocity_obligation",
            "wave4_reciprocity_obligation_v1.json",
            "The visitor did not hand the folder to the focal person. Continue processing the scenario.",
        ),
    )
    for construct_id, plan_name, bad_prompt_text in cases:
        spec = load_construct_spec(
            ROOT / f"configs/construct_benchmark/constructs/{construct_id}_v1.json"
        )
        plan = load_generation_plan(
            ROOT / "configs/construct_benchmark/generation_plans" / plan_name,
            spec,
        )
        first_cell = next(cell for cell in plan["cells"] if cell["mode"] == "paired")
        content_domain = plan["content_pools"][first_cell["content_pool"]]["domains"][0]
        condition_ids = tuple(first_cell["condition_ids"])

        def forbidden_response(model_id, messages, options):
            del model_id, messages, options
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "pairs": [
                                        {
                                            "pair_id": "pair_0",
                                            "content_domain": content_domain,
                                            "prompts": [
                                                {
                                                    "condition_id": condition_id,
                                                    "prompt_text": bad_prompt_text,
                                                }
                                                for condition_id in condition_ids
                                            ],
                                            "notes": "test",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        with pytest.raises(ValueError, match="forbidden term"):
            generate_prompt_records(
                plan,
                spec,
                api_key="test-only",
                request_fn=forbidden_response,
                count_per_model_override=1,
                limit_jobs=1,
            )
