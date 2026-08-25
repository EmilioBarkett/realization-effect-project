"""Deterministic bootstrap intervals for pair- and item-level estimands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .behavior import BehaviorObservation, StateTransferResult, directed_mean_state_transfer
from .readout import PairProjectionMargin


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    requested_resamples: int
    valid_resamples: int

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "requested_resamples": self.requested_resamples,
            "valid_resamples": self.valid_resamples,
        }


def _validate_bootstrap_args(resamples: int, confidence_level: float) -> None:
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    resamples: int = 2000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap the mean of complete pair/item-level values."""

    _validate_bootstrap_args(resamples, confidence_level)
    observed = np.asarray(list(values), dtype=np.float64)
    if observed.ndim != 1 or not len(observed) or not np.all(np.isfinite(observed)):
        raise ValueError("values must be a non-empty one-dimensional finite sequence.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(observed), size=(resamples, len(observed)))
    bootstrap_means = np.mean(observed[indices], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return BootstrapInterval(
        estimate=float(np.mean(observed)),
        lower=float(np.quantile(bootstrap_means, alpha)),
        upper=float(np.quantile(bootstrap_means, 1.0 - alpha)),
        confidence_level=confidence_level,
        requested_resamples=resamples,
        valid_resamples=resamples,
    )


def bootstrap_readout_margin_ci(
    margins: Iterable[PairProjectionMargin],
    *,
    resamples: int = 2000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    """Resample complete held-out pairs, not individual condition prompts."""

    return bootstrap_mean_ci(
        (margin.standardized_margin for margin in margins),
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


def _complete_item_groups(
    observations: Iterable[BehaviorObservation],
    *,
    positive_scale: float,
    negative_scale: float,
    zero_scale: float,
) -> dict[str, list[BehaviorObservation]]:
    required_scales = {positive_scale, negative_scale, zero_scale}
    groups: dict[str, list[BehaviorObservation]] = {}
    for observation in observations:
        if observation.scale in required_scales:
            groups.setdefault(observation.item_id, []).append(observation)
    complete: dict[str, list[BehaviorObservation]] = {}
    for item_id, items in groups.items():
        scales = [item.scale for item in items]
        if (
            len(items) == len(required_scales)
            and set(scales) == required_scales
            and all(item.valid and item.outcome is not None for item in items)
        ):
            complete[item_id] = items
    return complete


def bootstrap_state_transfer_ci(
    observations: Iterable[BehaviorObservation],
    *,
    positive_scale: float,
    negative_scale: float,
    zero_scale: float = 0.0,
    expected_sign: float = 1.0,
    resamples: int = 2000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> tuple[StateTransferResult, BootstrapInterval]:
    """Resample complete downstream task items for the steering estimand."""

    _validate_bootstrap_args(resamples, confidence_level)
    materialized = list(observations)
    observed = directed_mean_state_transfer(
        materialized,
        positive_scale=positive_scale,
        negative_scale=negative_scale,
        zero_scale=zero_scale,
        expected_sign=expected_sign,
    )
    groups = _complete_item_groups(
        materialized,
        positive_scale=positive_scale,
        negative_scale=negative_scale,
        zero_scale=zero_scale,
    )
    if not groups:
        raise ValueError("Bootstrap steering uncertainty requires complete item dose groups.")

    item_ids = sorted(groups)
    rng = np.random.default_rng(seed)
    bootstrap_effects: list[float] = []
    for _ in range(resamples):
        sampled_ids = rng.choice(item_ids, size=len(item_ids), replace=True)
        sampled: list[BehaviorObservation] = []
        for draw_index, item_id in enumerate(sampled_ids):
            sampled.extend(
                BehaviorObservation(
                    item_id=f"bootstrap_{draw_index}_{item_id}",
                    scale=item.scale,
                    outcome=item.outcome,
                    valid=item.valid,
                )
                for item in groups[str(item_id)]
            )
        try:
            result = directed_mean_state_transfer(
                sampled,
                positive_scale=positive_scale,
                negative_scale=negative_scale,
                zero_scale=zero_scale,
                expected_sign=expected_sign,
            )
        except ValueError:
            continue
        bootstrap_effects.append(result.directed_standardized_effect)

    if not bootstrap_effects:
        raise ValueError("No valid bootstrap resamples were available for steering uncertainty.")
    alpha = (1.0 - confidence_level) / 2.0
    interval = BootstrapInterval(
        estimate=observed.directed_standardized_effect,
        lower=float(np.quantile(bootstrap_effects, alpha)),
        upper=float(np.quantile(bootstrap_effects, 1.0 - alpha)),
        confidence_level=confidence_level,
        requested_resamples=resamples,
        valid_resamples=len(bootstrap_effects),
    )
    return observed, interval
