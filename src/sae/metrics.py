from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SAEMetricPlan:
    """Names the metrics we plan to track once an SAE backend is chosen."""

    reconstruction_loss: str = "reconstruction_mse"
    sparsity: str = "mean_active_features"
    activation_density: str = "fraction_active_features"
    feature_usage: str = "feature_activation_counts"


def planned_metric_names() -> list[str]:
    plan = SAEMetricPlan()
    return [
        plan.reconstruction_loss,
        plan.sparsity,
        plan.activation_density,
        plan.feature_usage,
    ]

