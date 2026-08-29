"""Strict Wave 1 output parsing and directed state-transfer metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


_INTEGER = re.compile(r"[+-]?\d+")

# Generic parsers retain their legacy field names, while registered task
# aliases make the scorer's primary-outcome lookup explicit and lossless.
_SUM_TASK_OUTCOMES = {
    "program_renewal_allocation_v1": ("existing_program_allocation", "new_program_allocation"),
    "known_unknown_lottery_allocation_v1": ("known_probability_allocation", "unknown_probability_allocation"),
    "information_seeking_commitment_v1": ("seek_information_allocation", "commit_now_allocation"),
    "help_allocation_v1": ("prior_helper_allocation", "other_requester_allocation"),
    "attention_allocation_v1": ("focal_task_attention", "distractor_attention"),
    "changed_constraint_plan_allocation_v1": ("revised_plan_allocation", "original_plan_allocation"),
    "intertemporal_allocation_v1": ("larger_later_allocation", "smaller_sooner_allocation"),
}
_PROBABILITY_TASK_OUTCOMES = {
    "testimony_weighting_v1": "signed_testimony_update",
}
_PROBABILITY_DIRECT_TASK_OUTCOMES = {
    "assigned_selected_prediction_v1": "target_outcome_probability",
    "peer_judgment_probability_v1": "claim_probability",
}
_CHOICE_TASK_OUTCOMES = {
    "sure_risky_choice_v1": ("sure_choice", "risky_choice"),
    "advice_direct_evidence_choice_v1": ("follow_specialist", "use_direct_measurement"),
    "known_new_option_choice_v1": ("known_option_choice", "new_option_choice"),
}
_ALLOCATION_TASK_OUTCOMES = {
    "goal_renewal_allocation_v2": "existing_goal_allocation",
    "realization_risk_allocation_v2": "risky_allocation",
    "diagnostic_test_allocation_v2": "high_information_test_allocation",
    "diagnostic_test_allocation_v4": "high_information_test_allocation",
    "source_evidence_allocation_v2": "source_weight_allocation",
}


@dataclass(frozen=True)
class ParsedBehavior:
    parser_id: str
    valid: bool
    values: dict[str, float]
    error: str | None = None


@dataclass(frozen=True)
class BehaviorObservation:
    item_id: str
    scale: float
    outcome: float | None
    valid: bool


@dataclass(frozen=True)
class StateTransferResult:
    positive_scale: float
    negative_scale: float
    zero_scale: float
    positive_mean: float
    negative_mean: float
    zero_sd: float
    directed_standardized_effect: float
    valid_counts: dict[str, int]
    total_counts: dict[str, int]


def _invalid(parser_id: str, message: str) -> ParsedBehavior:
    return ParsedBehavior(parser_id=parser_id, valid=False, values={}, error=message)


def _exact_integer_tokens(text: str) -> list[int] | None:
    stripped = text.strip()
    tokens = _INTEGER.findall(stripped)
    if not tokens or " ".join(tokens) != " ".join(stripped.split()):
        return None
    return [int(token) for token in tokens]


def parse_behavior_output(
    text: str,
    *,
    parser_id: str,
    item_metadata: Mapping[str, Any] | None = None,
    task_id: str | None = None,
) -> ParsedBehavior:
    """Parse one output without guessing missing task variables from prose."""

    if not isinstance(text, str):
        return _invalid(parser_id, "response is not text")
    metadata = dict(item_metadata or {})
    registered_task_id = task_id or metadata.get("task_id")
    integers = _exact_integer_tokens(text)
    if integers is None:
        return _invalid(parser_id, "response contains extra text or non-integer content")

    if parser_id == "two_integers_risk_choice_v1":
        if len(integers) != 2 or len([line for line in text.strip().splitlines() if line.strip()]) != 2:
            return _invalid(parser_id, "expected exactly two integers on separate lines")
        allocation, risk = integers
        if not 1 <= allocation <= 1000 or not 1 <= risk <= 5:
            return _invalid(parser_id, "risk-choice response is outside the registered range")
        return ParsedBehavior(parser_id, True, {"allocation_amount": float(allocation), "risk_preference": float(risk)})

    if parser_id == "single_integer_choice_1_or_2_v1":
        if len(integers) != 1 or integers[0] not in {1, 2}:
            return _invalid(parser_id, "expected one integer choice in {1, 2}")
        choice = integers[0]
        values = {"choice": float(choice)}
        correct_option = metadata.get("correct_option")
        if correct_option is not None:
            try:
                correct_option = int(correct_option)
            except (TypeError, ValueError):
                return _invalid(parser_id, "correct_option metadata is not an integer")
            if correct_option not in {1, 2}:
                return _invalid(parser_id, "correct_option metadata must be 1 or 2")
            values["correct_option"] = float(choice == correct_option)
        if registered_task_id in _CHOICE_TASK_OUTCOMES:
            first, second = _CHOICE_TASK_OUTCOMES[registered_task_id]
            values[first] = float(choice == 1)
            values[second] = float(choice == 2)
        return ParsedBehavior(parser_id, True, values)

    if parser_id == "two_integers_sum_100_v1":
        if len(integers) != 2:
            return _invalid(parser_id, "expected exactly two integers")
        continuing, alternative = integers
        if min(integers) < 0 or max(integers) > 100 or continuing + alternative != 100:
            return _invalid(parser_id, "allocations must be in [0, 100] and sum to 100")
        return ParsedBehavior(
            parser_id,
            True,
            {
                "continue_allocation": float(continuing),
                "alternative_allocation": float(alternative),
                **(
                    {
                        _SUM_TASK_OUTCOMES[registered_task_id][0]: float(continuing),
                        _SUM_TASK_OUTCOMES[registered_task_id][1]: float(alternative),
                    }
                    if registered_task_id in _SUM_TASK_OUTCOMES
                    else {}
                ),
            },
        )

    if parser_id == "single_integer_probability_v1":
        if len(integers) != 1 or not 0 <= integers[0] <= 100:
            return _invalid(parser_id, "expected one integer probability in [0, 100]")
        probability = float(integers[0])
        values = {"probability": probability, "posterior_probability": probability}
        if registered_task_id in _PROBABILITY_DIRECT_TASK_OUTCOMES:
            values[_PROBABILITY_DIRECT_TASK_OUTCOMES[registered_task_id]] = probability
        if "prior_probability" in metadata:
            prior = float(metadata["prior_probability"])
            if not 0 <= prior <= 100:
                return _invalid(parser_id, "prior_probability metadata is outside [0, 100]")
            values["absolute_posterior_update"] = abs(probability - prior)
            testimony_valence = metadata.get("testimony_valence")
            if testimony_valence == "supporting":
                signed_update = probability - prior
                values["testimony_weight"] = signed_update
            elif testimony_valence == "contradicting":
                signed_update = prior - probability
                values["testimony_weight"] = signed_update
            elif testimony_valence is not None:
                return _invalid(parser_id, "testimony_valence metadata is not registered")
            else:
                signed_update = None
            if registered_task_id in _PROBABILITY_TASK_OUTCOMES and signed_update is not None:
                values[_PROBABILITY_TASK_OUTCOMES[registered_task_id]] = signed_update
            if registered_task_id == "structured_bayesian_judgment_v1":
                values["prior_anchor_distance"] = abs(probability - prior)
                values["absolute_update"] = abs(probability - prior)
        return ParsedBehavior(parser_id, True, values)

    if parser_id == "single_integer_allocation_0_to_100_v1":
        if len(integers) != 1 or not 0 <= integers[0] <= 100:
            return _invalid(parser_id, "expected one integer allocation in [0, 100]")
        allocation = float(integers[0])
        values = {
            "allocation": allocation,
            "option_a_allocation": allocation,
        }
        if registered_task_id in _ALLOCATION_TASK_OUTCOMES:
            values[_ALLOCATION_TASK_OUTCOMES[registered_task_id]] = allocation
        return ParsedBehavior(parser_id, True, values)

    raise ValueError(f"Unsupported parser_id={parser_id!r}.")


def primary_outcome(parsed: ParsedBehavior, outcome_name: str) -> float:
    if not parsed.valid:
        raise ValueError(f"Cannot score invalid response: {parsed.error}")
    if outcome_name not in parsed.values:
        raise ValueError(
            f"Parsed response does not contain outcome={outcome_name!r}; required structured item metadata may be missing."
        )
    return float(parsed.values[outcome_name])


def orient_primary_outcome(
    construct_id: str,
    outcome: float,
    item_metadata: Mapping[str, Any],
) -> float | None:
    """Orient an outcome so positive values follow the registered state direction.

    The direction is a property of the construct *and* of the registered
    downstream cell.  For example, a reference-frame state predicts opposite
    risky-choice shifts in gain and loss frames, while a consensus state is
    oriented toward whichever peer answer was supplied.  Neutral calibration
    items return ``None`` when no directional state-transfer interpretation is
    registered.
    """

    if construct_id == "realization_account_closure":
        valence = item_metadata.get("outcome_valence")
        if valence == "gain":
            return float(outcome)
        if valence == "loss":
            return -float(outcome)
        if valence == "neutral":
            return None
        raise ValueError("realization outcome requires registered outcome_valence metadata.")
    if construct_id == "reference_frame":
        frame = item_metadata.get("payoff_frame") or item_metadata.get("surface_valence")
        if frame == "gain":
            return -float(outcome)
        if frame == "loss":
            return float(outcome)
        if frame == "neutral":
            return None
        raise ValueError("reference_frame outcome requires registered payoff_frame metadata.")
    if construct_id == "prior_weighting":
        return -float(outcome)
    if construct_id in {"authority_deference", "exploration_exploitation"}:
        return float(outcome)
    if construct_id == "ambiguity_orientation":
        # The registered primary outcome is the allocation to the known
        # probability option; accepting the interval therefore moves it down.
        return -float(outcome)
    if construct_id == "causal_interpretation":
        effect_direction = item_metadata.get("effect_direction")
        if effect_direction == "raises_outcome":
            return float(outcome)
        if effect_direction == "lowers_outcome":
            return 100.0 - float(outcome)
        raise ValueError("causal_interpretation outcome requires effect_direction metadata.")
    if construct_id == "consensus_conformity":
        peer_direction = item_metadata.get("peer_judgment_direction")
        if peer_direction == "supports_claim":
            return float(outcome)
        if peer_direction == "contradicts_claim":
            return 100.0 - float(outcome)
        raise ValueError("consensus_conformity outcome requires peer_judgment_direction metadata.")
    if construct_id in {
        "evidence_diagnosticity",
        "source_reliability",
        "persistence_continuation",
        "plan_replanning",
        "temporal_orientation",
        "epistemic_uncertainty",
        "reciprocity_obligation",
        "goal_shielding",
    }:
        if construct_id == "evidence_diagnosticity":
            high_information_option = item_metadata.get("high_information_option")
            if high_information_option == "option_b":
                return 100.0 - float(outcome)
            if high_information_option == "matched":
                return None
            # v1-v3 inventories kept the high-information test in option A;
            # retain that compatibility fallback while v4 records the option
            # identity explicitly and counterbalances it.
        return float(outcome)
    raise ValueError(f"No outcome orientation adapter for construct_id={construct_id!r}.")


def directed_mean_state_transfer(
    observations: Iterable[BehaviorObservation],
    *,
    positive_scale: float,
    negative_scale: float,
    zero_scale: float = 0.0,
    expected_sign: float = 1.0,
) -> StateTransferResult:
    """Compute the preregistered signed contrast without imputing invalid rows."""

    if expected_sign not in {-1.0, 1.0}:
        raise ValueError("expected_sign must be +1 or -1.")
    if len({positive_scale, negative_scale, zero_scale}) != 3:
        raise ValueError("positive_scale, negative_scale, and zero_scale must be distinct.")
    grouped: dict[float, list[BehaviorObservation]] = {
        positive_scale: [],
        negative_scale: [],
        zero_scale: [],
    }
    for observation in observations:
        if observation.scale in grouped:
            grouped[observation.scale].append(observation)
    valid_values: dict[float, list[float]] = {
        scale: [float(item.outcome) for item in items if item.valid and item.outcome is not None]
        for scale, items in grouped.items()
    }
    if not valid_values[positive_scale] or not valid_values[negative_scale]:
        raise ValueError("Positive and negative steering conditions require valid outcomes.")
    if len(valid_values[zero_scale]) < 2:
        raise ValueError("Zero-dose standardization requires at least two valid outcomes.")
    zero_sd = float(np.std(valid_values[zero_scale], ddof=1))
    if not np.isfinite(zero_sd) or zero_sd <= 0:
        raise ValueError("Zero-dose outcome standard deviation must be finite and greater than zero.")
    positive_mean = float(np.mean(valid_values[positive_scale]))
    negative_mean = float(np.mean(valid_values[negative_scale]))
    labels = {positive_scale: "positive", negative_scale: "negative", zero_scale: "zero"}
    return StateTransferResult(
        positive_scale=positive_scale,
        negative_scale=negative_scale,
        zero_scale=zero_scale,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        zero_sd=zero_sd,
        directed_standardized_effect=expected_sign * (positive_mean - negative_mean) / (2 * zero_sd),
        valid_counts={labels[scale]: len(values) for scale, values in valid_values.items()},
        total_counts={labels[scale]: len(grouped[scale]) for scale in grouped},
    )
