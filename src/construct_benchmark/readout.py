"""Construct-scoped train-only directions and held-out readout metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from activation_analysis.vector_analysis import PromptActivation

from .calibration import unit_direction


@dataclass(frozen=True)
class DirectionEstimate:
    construct_id: str
    positive_condition_id: str
    negative_condition_id: str
    pair_count: int
    direction: np.ndarray
    pair_differences: np.ndarray


@dataclass(frozen=True)
class PairProjectionMargin:
    pair_id: str
    positive_prompt_id: str
    negative_prompt_id: str
    positive_projection: float
    negative_projection: float
    standardized_margin: float


@dataclass(frozen=True)
class ReadoutResult:
    construct_id: str
    split: str
    pair_count: int
    mean_standardized_margin: float
    pair_accuracy: float
    margins: tuple[PairProjectionMargin, ...]


def _metadata_value(activation: PromptActivation, key: str) -> str:
    value = activation.metadata.get(key, "")
    return "" if value is None else str(value).strip()


def _paired(
    activations: Iterable[PromptActivation],
    *,
    construct_id: str,
    split: str,
) -> dict[str, dict[str, PromptActivation]]:
    pairs: dict[str, dict[str, PromptActivation]] = {}
    for activation in activations:
        if _metadata_value(activation, "construct_id") != construct_id:
            continue
        if _metadata_value(activation, "split") != split:
            continue
        pair_id = _metadata_value(activation, "pair_id")
        condition_id = _metadata_value(activation, "condition_id") or _metadata_value(
            activation, "pair_role"
        )
        if pair_id and condition_id:
            if condition_id in pairs.setdefault(pair_id, {}):
                raise ValueError(f"Duplicate condition_id={condition_id!r} in pair_id={pair_id!r}.")
            pairs[pair_id][condition_id] = activation
    return pairs


def estimate_train_direction(
    activations: Iterable[PromptActivation],
    *,
    construct_id: str,
    positive_condition_id: str,
    negative_condition_id: str,
) -> DirectionEstimate:
    """Estimate mean positive-minus-negative direction from training pairs only."""

    pairs = _paired(activations, construct_id=construct_id, split="direction_train")
    differences: list[np.ndarray] = []
    for pair_id, members in sorted(pairs.items()):
        positive = members.get(positive_condition_id)
        negative = members.get(negative_condition_id)
        if positive is None or negative is None:
            raise ValueError(f"Training pair {pair_id!r} is missing a registered condition member.")
        positive_vector = np.asarray(positive.vector, dtype=np.float32)
        negative_vector = np.asarray(negative.vector, dtype=np.float32)
        if positive_vector.shape != negative_vector.shape:
            raise ValueError(f"Training pair {pair_id!r} has inconsistent hidden sizes.")
        differences.append(positive_vector - negative_vector)
    if not differences:
        raise ValueError(f"No complete direction_train pairs found for construct_id={construct_id!r}.")
    direction = np.mean(np.stack(differences), axis=0).astype(np.float32, copy=False)
    unit_direction(direction)
    return DirectionEstimate(
        construct_id=construct_id,
        positive_condition_id=positive_condition_id,
        negative_condition_id=negative_condition_id,
        pair_count=len(differences),
        direction=direction,
        pair_differences=np.stack(differences).astype(np.float32, copy=False),
    )


def evaluate_split_readout(
    activations: Iterable[PromptActivation],
    estimate: DirectionEstimate,
    *,
    projection_scale: float,
    split: str,
) -> ReadoutResult:
    """Evaluate a frozen direction on paired prompts using a frozen scale."""

    if not np.isfinite(projection_scale) or projection_scale <= 0:
        raise ValueError("projection_scale must be finite and greater than zero.")
    direction = unit_direction(estimate.direction)
    pairs = _paired(activations, construct_id=estimate.construct_id, split=split)
    margins: list[PairProjectionMargin] = []
    for pair_id, members in sorted(pairs.items()):
        positive = members.get(estimate.positive_condition_id)
        negative = members.get(estimate.negative_condition_id)
        if positive is None or negative is None:
            raise ValueError(f"{split} pair {pair_id!r} is missing a registered condition member.")
        positive_projection = float(np.dot(np.asarray(positive.vector, dtype=np.float32), direction))
        negative_projection = float(np.dot(np.asarray(negative.vector, dtype=np.float32), direction))
        margins.append(
            PairProjectionMargin(
                pair_id=pair_id,
                positive_prompt_id=positive.prompt_id,
                negative_prompt_id=negative.prompt_id,
                positive_projection=positive_projection,
                negative_projection=negative_projection,
                standardized_margin=(positive_projection - negative_projection) / projection_scale,
            )
        )
    if not margins:
        raise ValueError(f"No complete {split} pairs found for construct_id={estimate.construct_id!r}.")
    values = np.asarray([margin.standardized_margin for margin in margins], dtype=np.float64)
    return ReadoutResult(
        construct_id=estimate.construct_id,
        split=split,
        pair_count=len(margins),
        mean_standardized_margin=float(np.mean(values)),
        pair_accuracy=float(np.mean(values > 0)),
        margins=tuple(margins),
    )


def evaluate_heldout_readout(
    activations: Iterable[PromptActivation],
    estimate: DirectionEstimate,
    *,
    projection_scale: float,
) -> ReadoutResult:
    """Evaluate the frozen direction on held-out pairs using a frozen scale."""

    return evaluate_split_readout(
        activations,
        estimate,
        projection_scale=projection_scale,
        split="direction_heldout",
    )


def evaluate_validation_readout(
    activations: Iterable[PromptActivation],
    estimate: DirectionEstimate,
    *,
    projection_scale: float,
) -> ReadoutResult:
    """Evaluate a candidate layer on validation pairs for layer selection only."""

    return evaluate_split_readout(
        activations,
        estimate,
        projection_scale=projection_scale,
        split="direction_validation",
    )
