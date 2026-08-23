"""Training-only projection and intervention-scale calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from activation_analysis.vector_analysis import PromptActivation


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    construct_id: str
    split: str
    sample_count: int
    group_count: int
    projection_scale: float


def unit_direction(direction: np.ndarray) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"direction must be one-dimensional; received shape={vector.shape}.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction must have a finite, non-zero norm.")
    return (vector / norm).astype(np.float32, copy=False)


def _metadata_value(activation: PromptActivation, key: str) -> str:
    value = activation.metadata.get(key, "")
    return "" if value is None else str(value).strip()


def estimate_projection_scale(
    activations: Iterable[PromptActivation],
    direction: np.ndarray,
    *,
    construct_id: str,
    method: str = "neutral",
) -> CalibrationResult:
    """Estimate a scale without using positive/negative mixture separation.

    ``neutral`` uses the sample standard deviation of projections on the
    dedicated calibration split. ``within_condition`` uses direction-training
    residuals after centering each condition separately.
    """

    if method not in {"neutral", "within_condition"}:
        raise ValueError("method must be 'neutral' or 'within_condition'.")
    direction_unit = unit_direction(direction)
    split = "calibration" if method == "neutral" else "direction_train"
    selected = [
        activation
        for activation in activations
        if _metadata_value(activation, "construct_id") == construct_id
        and _metadata_value(activation, "split") == split
    ]
    if not selected:
        raise ValueError(f"No {split!r} activations found for construct_id={construct_id!r}.")

    projections = np.asarray(
        [float(np.dot(np.asarray(item.vector, dtype=np.float32), direction_unit)) for item in selected],
        dtype=np.float64,
    )
    if method == "neutral":
        if len(projections) < 2:
            raise ValueError("Neutral calibration requires at least two activation observations.")
        scale = float(np.std(projections, ddof=1))
        group_count = 1
    else:
        groups: dict[str, list[float]] = {}
        for activation, projection in zip(selected, projections, strict=True):
            condition_id = _metadata_value(activation, "condition_id") or _metadata_value(
                activation, "pair_role"
            )
            if not condition_id:
                raise ValueError("Within-condition calibration requires condition_id or pair_role metadata.")
            groups.setdefault(condition_id, []).append(float(projection))
        residual_sum_squares = sum(
            sum((value - float(np.mean(values))) ** 2 for value in values)
            for values in groups.values()
        )
        degrees_of_freedom = len(projections) - len(groups)
        if degrees_of_freedom < 1:
            raise ValueError("Within-condition calibration requires residual degrees of freedom.")
        scale = float(np.sqrt(residual_sum_squares / degrees_of_freedom))
        group_count = len(groups)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration projection scale must be finite and greater than zero.")
    return CalibrationResult(
        method=method,
        construct_id=construct_id,
        split=split,
        sample_count=len(selected),
        group_count=group_count,
        projection_scale=scale,
    )


def intervention_scale(calibration: CalibrationResult, dose: float) -> float:
    """Convert a dimensionless registered dose to a physical residual scale."""

    if not np.isfinite(dose):
        raise ValueError("dose must be finite.")
    return float(dose * calibration.projection_scale)
