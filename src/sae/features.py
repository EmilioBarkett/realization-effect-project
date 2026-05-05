from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureExample:
    """Metadata for one top-activating SAE feature example."""

    feature_id: int
    activation_value: float
    prompt_id: str
    token_region: str
    token_position: int | None
    prompt_metadata: dict[str, Any]


class SAEFeatureAnalysisNotImplementedError(NotImplementedError):
    """Raised until feature-ranking utilities are implemented."""


def top_activating_examples(*_args, **_kwargs) -> list[FeatureExample]:
    """Future hook for ranking dataset examples by trained SAE feature activation.

    The project now has a local SAE training backend. This module is the next
    layer above training: loading a checkpoint, scoring activation records, and
    returning the highest-activating prompt/token examples for interpretation.
    """

    raise SAEFeatureAnalysisNotImplementedError(
        "Feature analysis is not implemented yet. Train/load an SAE checkpoint "
        "and add feature-ranking logic here."
    )
