"""Strict Wave 1 output parsing and directed state-transfer metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


_INTEGER = re.compile(r"[+-]?\d+")


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
) -> ParsedBehavior:
    """Parse one output without guessing missing task variables from prose."""

    if not isinstance(text, str):
        return _invalid(parser_id, "response is not text")
    metadata = dict(item_metadata or {})
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

    if parser_id == "two_integers_sum_100_v1":
        if len(integers) != 2:
            return _invalid(parser_id, "expected exactly two integers")
        continuing, alternative = integers
        if min(integers) < 0 or max(integers) > 100 or continuing + alternative != 100:
            return _invalid(parser_id, "allocations must be in [0, 100] and sum to 100")
        return ParsedBehavior(
            parser_id,
            True,
            {"continue_allocation": float(continuing), "alternative_allocation": float(alternative)},
        )

    if parser_id == "single_integer_probability_v1":
        if len(integers) != 1 or not 0 <= integers[0] <= 100:
            return _invalid(parser_id, "expected one integer probability in [0, 100]")
        probability = float(integers[0])
        values = {"probability": probability, "posterior_probability": probability}
        if "prior_probability" in metadata:
            prior = float(metadata["prior_probability"])
            if not 0 <= prior <= 100:
                return _invalid(parser_id, "prior_probability metadata is outside [0, 100]")
            values["absolute_posterior_update"] = abs(probability - prior)
            testimony_valence = metadata.get("testimony_valence")
            if testimony_valence == "supporting":
                values["testimony_weight"] = probability - prior
            elif testimony_valence == "contradicting":
                values["testimony_weight"] = prior - probability
            elif testimony_valence is not None:
                return _invalid(parser_id, "testimony_valence metadata is not registered")
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

    Realization gain and loss items have opposite predictions; neutral items
    are controls and therefore return ``None`` for the primary directed effect.
    Other Wave 1 primary outcomes are already state-consistently oriented.
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
    if construct_id in {
        "evidence_diagnosticity",
        "source_reliability",
        "persistence_continuation",
    }:
        return float(outcome)
    raise ValueError(f"No Wave 1 outcome orientation adapter for construct_id={construct_id!r}.")


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
