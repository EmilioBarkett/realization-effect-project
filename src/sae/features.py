from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureExample:
    """Metadata for a future top-activating SAE feature example."""

    feature_id: int
    activation_value: float
    prompt_id: str
    token_region: str
    token_position: int | None
    prompt_metadata: dict[str, Any]


class SAEFeatureAnalysisNotImplementedError(NotImplementedError):
    """Raised until feature scoring exists."""


def top_activating_examples(*_args, **_kwargs) -> list[FeatureExample]:
    """Future hook for ranking dataset examples by SAE feature activation."""

    raise SAEFeatureAnalysisNotImplementedError(
        "Feature analysis is not implemented yet. Decide the SAE backend first."
    )
