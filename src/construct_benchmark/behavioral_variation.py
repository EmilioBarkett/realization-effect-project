"""Pre-run gates for model-side behavioral variation.

The benchmark standardizes steering effects by the observed zero-dose spread of
the same downstream outcome. A model-side test must therefore establish that
the target steering rows produce enough valid, non-constant zero-dose outcomes
before a full run is allowed. This module is model- and provider-independent.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .behavior import orient_primary_outcome, parse_behavior_output, primary_outcome
from .schemas import ConstructSpec


DEFAULT_GATE = {
    "minimum_zero_dose_valid": 8,
    "minimum_zero_dose_distinct": 2,
    "minimum_zero_dose_sample_sd": 1.0,
    "maximum_zero_dose_invalid": 0,
}


def behavioral_variation_gate_config(
    spec: ConstructSpec,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    """Resolve registered thresholds without changing a construct spec."""

    configured = dict((spec.metadata or {}).get("behavioral_variation_gate", {}))
    values: dict[str, float | int] = dict(DEFAULT_GATE)
    for key in DEFAULT_GATE:
        if key in configured:
            values[key] = configured[key]
    if overrides:
        for key, value in overrides.items():
            if key not in DEFAULT_GATE:
                raise ValueError(f"Unknown behavioral variation threshold={key!r}.")
            values[key] = value

    minimum_valid = values["minimum_zero_dose_valid"]
    minimum_distinct = values["minimum_zero_dose_distinct"]
    minimum_sd = values["minimum_zero_dose_sample_sd"]
    maximum_invalid = values["maximum_zero_dose_invalid"]
    if (
        not isinstance(minimum_valid, int)
        or isinstance(minimum_valid, bool)
        or minimum_valid < 1
    ):
        raise ValueError("minimum_zero_dose_valid must be a positive integer.")
    if (
        not isinstance(minimum_distinct, int)
        or isinstance(minimum_distinct, bool)
        or minimum_distinct < 2
    ):
        raise ValueError("minimum_zero_dose_distinct must be an integer >= 2.")
    if not isinstance(minimum_sd, (int, float)) or isinstance(minimum_sd, bool) or not np.isfinite(float(minimum_sd)) or float(minimum_sd) <= 0:
        raise ValueError("minimum_zero_dose_sample_sd must be finite and positive.")
    if (
        not isinstance(maximum_invalid, int)
        or isinstance(maximum_invalid, bool)
        or maximum_invalid < 0
    ):
        raise ValueError("maximum_zero_dose_invalid must be a non-negative integer.")
    values["minimum_zero_dose_sample_sd"] = float(minimum_sd)
    return values


def _is_injection_layer_row(row: Mapping[str, Any]) -> bool:
    if row.get("tracking_role") == "injection_immediate":
        return True
    try:
        tracking_layer = int(row["tracking_layer"])
        injection_layer = int(row["injection_layer"])
    except (KeyError, TypeError, ValueError):
        return False
    return tracking_layer == injection_layer


def _audit_outcome_rows(
    selected_rows: list[dict[str, Any]],
    spec: ConstructSpec,
    gate: Mapping[str, Any],
    *,
    invalid_label: str,
) -> dict[str, Any]:
    valid_outcomes: list[float] = []
    invalid_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        metadata = dict(row.get("task_metadata") or {})
        parser_id = str(row.get("parser_id") or spec.parsing_rules["parser_id"])
        task_id = str(row.get("task_id") or spec.independent_behavior_task["task_id"])
        parsed = parse_behavior_output(
            row.get("output_text", ""),
            parser_id=parser_id,
            item_metadata=metadata,
            task_id=task_id,
        )
        if not parsed.valid:
            invalid_rows.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "error": parsed.error or "invalid response",
                }
            )
            continue
        try:
            outcome = primary_outcome(
                parsed,
                str(spec.independent_behavior_task["primary_outcome"]),
            )
            directed = orient_primary_outcome(spec.construct_id, outcome, metadata)
        except (TypeError, ValueError) as exc:
            invalid_rows.append({"prompt_id": row.get("prompt_id"), "error": str(exc)})
            continue
        if directed is None:
            excluded_rows.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "reason": "outcome orientation returned None for a registered neutral control",
                }
            )
            continue
        valid_outcomes.append(float(directed))

    sample_sd = float(np.std(valid_outcomes, ddof=1)) if len(valid_outcomes) >= 2 else None
    unique_outcomes = sorted({float(value) for value in valid_outcomes})
    failures: list[str] = []
    if len(valid_outcomes) < int(gate["minimum_zero_dose_valid"]):
        failures.append(
            f"valid {invalid_label} outcomes={len(valid_outcomes)} is below "
            f"minimum={gate['minimum_zero_dose_valid']}"
        )
    if len(unique_outcomes) < int(gate["minimum_zero_dose_distinct"]):
        failures.append(
            f"distinct {invalid_label} outcomes={len(unique_outcomes)} is below "
            f"minimum={gate['minimum_zero_dose_distinct']}"
        )
    if sample_sd is None or not np.isfinite(sample_sd) or sample_sd < float(gate["minimum_zero_dose_sample_sd"]):
        failures.append(
            f"{invalid_label} sample SD={sample_sd!r} is below "
            f"minimum={gate['minimum_zero_dose_sample_sd']}"
        )
    if len(invalid_rows) > int(gate["maximum_zero_dose_invalid"]):
        failures.append(
            f"invalid {invalid_label} rows={len(invalid_rows)} exceeds "
            f"maximum={gate['maximum_zero_dose_invalid']}"
        )
    return {
        "pass": not failures,
        "selected_rows": len(selected_rows),
        "valid_rows": len(valid_outcomes),
        "invalid_rows": len(invalid_rows),
        "excluded_neutral_rows": len(excluded_rows),
        "unique_outcomes": unique_outcomes,
        "mean": float(np.mean(valid_outcomes)) if valid_outcomes else None,
        "sample_sd": sample_sd,
        "thresholds": dict(gate),
        "failures": failures,
        "invalid_row_details": invalid_rows,
        "excluded_row_details": excluded_rows,
    }


def audit_zero_dose_variation(
    raw_rows: Iterable[Mapping[str, Any]],
    spec: ConstructSpec,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit target-direction zero-dose rows and return a release decision.

    Neutral realization items are reported separately because they are
    intentionally excluded from the directed outcome. Invalid parser rows
    are never imputed and count against the registered invalid-row allowance.
    """

    gate = behavioral_variation_gate_config(spec, thresholds)
    zero_rows = [
        dict(row)
        for row in raw_rows
        if row.get("direction_kind") == "target"
        and _is_injection_layer_row(row)
        and _is_zero_dose(row.get("dose"))
    ]
    audit = _audit_outcome_rows(zero_rows, spec, gate, invalid_label="zero-dose")
    return {
        "pass": audit["pass"],
        "construct_id": spec.construct_id,
        "primary_outcome": spec.independent_behavior_task["primary_outcome"],
        "zero_dose_target_injection_rows": len(zero_rows),
        "valid_zero_dose_rows": audit["valid_rows"],
        "invalid_zero_dose_rows": audit["invalid_rows"],
        "excluded_neutral_rows": audit["excluded_neutral_rows"],
        "unique_zero_dose_outcomes": audit["unique_outcomes"],
        "zero_dose_mean": audit["mean"],
        "zero_dose_sample_sd": audit["sample_sd"],
        "thresholds": audit["thresholds"],
        "failures": audit["failures"],
        "invalid_rows": audit["invalid_row_details"],
        "excluded_rows": audit["excluded_row_details"],
    }


def audit_prompt_only_variation(
    raw_rows: Iterable[Mapping[str, Any]],
    spec: ConstructSpec,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit independent behavior prompts before any direction is applied."""

    gate = behavioral_variation_gate_config(spec, thresholds)
    behavior_rows = [
        dict(row)
        for row in raw_rows
        if row.get("construct_id") == spec.construct_id
        and row.get("split") == "behavior_eval"
        and row.get("intervention") == "none"
    ]
    audit = _audit_outcome_rows(behavior_rows, spec, gate, invalid_label="prompt-only")
    return {
        "pass": audit["pass"],
        "construct_id": spec.construct_id,
        "primary_outcome": spec.independent_behavior_task["primary_outcome"],
        "prompt_only_rows": len(behavior_rows),
        "valid_prompt_only_rows": audit["valid_rows"],
        "invalid_prompt_only_rows": audit["invalid_rows"],
        "excluded_neutral_rows": audit["excluded_neutral_rows"],
        "unique_prompt_only_outcomes": audit["unique_outcomes"],
        "prompt_only_mean": audit["mean"],
        "prompt_only_sample_sd": audit["sample_sd"],
        "thresholds": audit["thresholds"],
        "failures": audit["failures"],
        "invalid_rows": audit["invalid_row_details"],
        "excluded_rows": audit["excluded_row_details"],
    }


def _is_zero_dose(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


__all__ = [
    "audit_prompt_only_variation",
    "audit_zero_dose_variation",
    "behavioral_variation_gate_config",
]
