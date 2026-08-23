"""Split names and coverage helpers for leakage-resistant benchmark runs."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, Mapping

from .schemas import ConstructSpec

if TYPE_CHECKING:
    from .prompts import PromptRecord


SPLIT_EXECUTION_SCOPE = {
    "direction_train": "shared_activation_then_construct_direction",
    "direction_validation": "shared_activation_then_construct_validation",
    "direction_heldout": "shared_activation_then_construct_readout",
    "behavior_eval": "construct_behavior_task",
    "steering_eval": "construct_steering_task",
    "calibration": "construct_calibration_task",
}

SPLIT_PROMPT_ROLE = {
    "direction_train": "probe",
    "direction_validation": "probe",
    "direction_heldout": "probe",
    "behavior_eval": "behavior",
    "steering_eval": "steering",
    "calibration": "calibration",
}


def count_by_construct_split(records: Iterable["PromptRecord"]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for record in records:
        construct_counts = counts[record.construct_id]
        construct_counts[record.split] = construct_counts.get(record.split, 0) + 1
    return {
        construct_id: dict(sorted(split_counts.items()))
        for construct_id, split_counts in sorted(counts.items())
    }


def missing_required_splits(
    records: Iterable["PromptRecord"],
    construct_specs: Mapping[str, ConstructSpec],
) -> dict[str, list[str]]:
    counts = count_by_construct_split(records)
    return {
        construct_id: sorted(set(spec.required_splits) - set(counts.get(construct_id, {})))
        for construct_id, spec in construct_specs.items()
        if set(spec.required_splits) - set(counts.get(construct_id, {}))
    }


def validate_split_coverage(
    records: Iterable["PromptRecord"],
    construct_specs: Mapping[str, ConstructSpec],
) -> dict[str, dict[str, int]]:
    counts = count_by_construct_split(records)
    missing = missing_required_splits(records, construct_specs)
    if missing:
        details = "; ".join(f"{construct_id}: {splits}" for construct_id, splits in sorted(missing.items()))
        raise ValueError(f"Prompt inventory is missing required splits: {details}")
    return counts
