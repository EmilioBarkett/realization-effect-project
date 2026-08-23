"""Deterministic steering conditions and control-direction primitives."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .calibration import CalibrationResult, intervention_scale, unit_direction


@dataclass(frozen=True)
class SteeringCondition:
    condition_id: str
    prompt_id: str
    direction_kind: str
    direction_index: int
    dose: float
    physical_scale: float
    intervention_timing: str
    order: int
    seed: int


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_steering_conditions(
    prompt_ids: Iterable[str],
    calibration: CalibrationResult,
    *,
    doses: Iterable[float],
    intervention_timing: str,
    seed: int,
    include_shuffled: bool = True,
    random_direction_count: int = 2,
) -> tuple[SteeringCondition, ...]:
    """Expand and deterministically randomize the registered intervention battery."""

    registered_doses = tuple(float(dose) for dose in doses)
    if not registered_doses or 0.0 not in registered_doses:
        raise ValueError("doses must include the zero-dose condition.")
    if not any(dose > 0 for dose in registered_doses) or not any(dose < 0 for dose in registered_doses):
        raise ValueError("doses must include positive and negative conditions.")
    if len(set(registered_doses)) != len(registered_doses):
        raise ValueError("doses must not contain duplicates.")
    if random_direction_count < 0:
        raise ValueError("random_direction_count must be non-negative.")

    conditions: list[SteeringCondition] = []
    for prompt_id in prompt_ids:
        prompt_seed = _stable_seed(seed, prompt_id)
        cells: list[tuple[str, int, float]] = [("target", 0, dose) for dose in registered_doses]
        nonzero_doses = [dose for dose in registered_doses if dose != 0]
        if include_shuffled:
            cells.extend(("shuffled", 0, dose) for dose in nonzero_doses)
        for direction_index in range(random_direction_count):
            cells.extend(("random", direction_index, dose) for dose in nonzero_doses)
        random.Random(prompt_seed).shuffle(cells)
        for order, (direction_kind, direction_index, dose) in enumerate(cells):
            condition_id = (
                f"{prompt_id}__{direction_kind}_{direction_index:02d}__"
                f"dose_{dose:+g}"
            )
            conditions.append(
                SteeringCondition(
                    condition_id=condition_id,
                    prompt_id=prompt_id,
                    direction_kind=direction_kind,
                    direction_index=direction_index,
                    dose=dose,
                    physical_scale=intervention_scale(calibration, dose),
                    intervention_timing=intervention_timing,
                    order=order,
                    seed=_stable_seed(prompt_seed, condition_id),
                )
            )
    return tuple(conditions)


def shuffled_label_direction(pair_differences: np.ndarray, *, seed: int) -> np.ndarray:
    """Build a label-shuffled mean direction by sign-flipping pair contrasts."""

    differences = np.asarray(pair_differences, dtype=np.float32)
    if differences.ndim != 2 or differences.shape[0] < 2:
        raise ValueError("pair_differences must have shape [pairs, hidden] with at least two pairs.")
    rng = np.random.default_rng(seed)
    negative_count = differences.shape[0] // 2
    signs = np.asarray(
        [-1.0] * negative_count + [1.0] * (differences.shape[0] - negative_count),
        dtype=np.float32,
    )
    rng.shuffle(signs)
    shuffled = np.mean(differences * signs[:, None], axis=0)
    return unit_direction(shuffled)


def random_control_direction(
    hidden_size: int,
    *,
    seed: int,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    """Create a reproducible Gaussian random unit direction.

    When a target is supplied, remove its component so the random control is
    orthogonal to the target direction.
    """

    if hidden_size < 2:
        raise ValueError("hidden_size must be at least two.")
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=hidden_size).astype(np.float32)
    if orthogonal_to is not None:
        target = unit_direction(orthogonal_to)
        if target.shape != vector.shape:
            raise ValueError("orthogonal_to hidden size does not match hidden_size.")
        vector = vector - float(np.dot(vector, target)) * target
    return unit_direction(vector)
