"""Measured one-to-four replica policy for the single NVIDIA B300 pod.

This module does not infer memory from parameter counts.  Every admission
decision requires runtime measurements supplied by the caller, which keeps
the policy usable with different CUDA allocators and model-loading choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


B300_GPU_TYPE = "NVIDIA B300"
B300_GPU_COUNT = 1
SUPPORTED_REPLICA_STAGES = (1, 3, 4)


class GpuRolloutError(RuntimeError):
    """Raised when measured device/runtime facts violate the rollout contract."""


class GpuOutOfMemoryError(GpuRolloutError):
    """Raised by an injected stage callback when a replica launch OOMs."""


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return parsed


@dataclass(frozen=True)
class GpuMeasurement:
    """Runtime telemetry for one compatibility or replica stage."""

    replica_count: int
    gpu_type: str = B300_GPU_TYPE
    gpu_count: int = B300_GPU_COUNT
    total_vram_gb: float | None = None
    loaded_model_vram_gb: float | None = None
    peak_vram_gb: float | None = None
    projected_peak_vram_gb: float | None = None
    throughput_items_per_second: float | None = None
    runtime_seconds: float | None = None
    stable: bool = True
    oom: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.replica_count not in SUPPORTED_REPLICA_STAGES:
            raise ValueError(f"replica_count must be one of {SUPPORTED_REPLICA_STAGES}")
        if self.gpu_count != B300_GPU_COUNT:
            raise ValueError("gpu_count must be exactly 1 for the B300 rollout")
        if not isinstance(self.stable, bool) or not isinstance(self.oom, bool):
            raise ValueError("stable and oom must be booleans")
        for field_name in (
            "total_vram_gb",
            "loaded_model_vram_gb",
            "peak_vram_gb",
            "projected_peak_vram_gb",
            "throughput_items_per_second",
            "runtime_seconds",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _finite_nonnegative(value, field=field_name)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "replica_count": self.replica_count,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "total_vram_gb": self.total_vram_gb,
            "loaded_model_vram_gb": self.loaded_model_vram_gb,
            "peak_vram_gb": self.peak_vram_gb,
            "projected_peak_vram_gb": self.projected_peak_vram_gb,
            "throughput_items_per_second": self.throughput_items_per_second,
            "runtime_seconds": self.runtime_seconds,
            "stable": self.stable,
            "oom": self.oom,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ReplicaDecision:
    stage: str
    admitted: bool
    selected_replicas: int
    reason: str
    threshold_vram_gb: float | None = None
    measured_peak_vram_gb: float | None = None
    measurement: Mapping[str, Any] | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "admitted": self.admitted,
            "selected_replicas": self.selected_replicas,
            "reason": self.reason,
            "threshold_vram_gb": self.threshold_vram_gb,
            "measured_peak_vram_gb": self.measured_peak_vram_gb,
            "measurement": dict(self.measurement or {}),
        }


class B300ReplicaRollout:
    """Apply the single-pod 1 -> 3 -> 4 measured rollout."""

    def __init__(
        self,
        *,
        safety_fraction: float = 0.90,
        safety_vram_gb: float | None = None,
        target_replicas: int = 4,
    ) -> None:
        if target_replicas not in {3, 4}:
            raise ValueError("target_replicas must be 3 or 4")
        if not math.isfinite(float(safety_fraction)) or not 0 < float(safety_fraction) <= 1:
            raise ValueError("safety_fraction must be in (0, 1]")
        if safety_vram_gb is not None:
            _finite_nonnegative(safety_vram_gb, field="safety_vram_gb")
        self.safety_fraction = float(safety_fraction)
        self.safety_vram_gb = float(safety_vram_gb) if safety_vram_gb is not None else None
        self.target_replicas = int(target_replicas)
        self.decisions: list[ReplicaDecision] = []

    def _validate_measurement(
        self,
        measurement: GpuMeasurement,
        *,
        stage: str,
        expected_replica_count: int | None = None,
        require_loaded_model_vram: bool = False,
    ) -> None:
        if measurement.gpu_type != B300_GPU_TYPE:
            raise GpuRolloutError(
                f"{stage} reported GPU {measurement.gpu_type!r}; expected exact {B300_GPU_TYPE!r}"
            )
        if measurement.gpu_count != B300_GPU_COUNT:
            raise GpuRolloutError(f"{stage} reported gpu_count={measurement.gpu_count!r}; expected 1")
        if expected_replica_count is not None and measurement.replica_count != expected_replica_count:
            raise GpuRolloutError(
                f"{stage} reported replica_count={measurement.replica_count!r}; "
                f"expected {expected_replica_count}"
            )
        if require_loaded_model_vram and measurement.loaded_model_vram_gb is None:
            raise GpuRolloutError(f"{stage} did not report loaded-model VRAM")
        if measurement.oom:
            raise GpuOutOfMemoryError(f"{stage} reported out-of-memory")
        if not measurement.stable:
            raise GpuRolloutError(f"{stage} runtime measurement is not stable")

    def threshold_vram_gb(self, measurement: GpuMeasurement) -> float | None:
        candidates: list[float] = []
        if measurement.total_vram_gb is not None:
            candidates.append(float(measurement.total_vram_gb) * self.safety_fraction)
        if self.safety_vram_gb is not None:
            candidates.append(self.safety_vram_gb)
        return min(candidates) if candidates else None

    def admit_fourth(self, measurement: GpuMeasurement) -> ReplicaDecision:
        """Decide from measured three-replica telemetry whether four may run."""

        self._validate_measurement(measurement, stage="three-replica stage", expected_replica_count=3)
        threshold = self.threshold_vram_gb(measurement)
        measured_peak = measurement.peak_vram_gb
        projected_peak = measurement.projected_peak_vram_gb
        if threshold is None:
            decision = ReplicaDecision(
                stage="four",
                admitted=False,
                selected_replicas=3,
                reason="no measured VRAM ceiling was supplied",
                measurement=measurement.to_mapping(),
            )
        elif measured_peak is None or projected_peak is None:
            decision = ReplicaDecision(
                stage="four",
                admitted=False,
                selected_replicas=3,
                reason="fourth replica requires explicit measured and projected peak VRAM",
                threshold_vram_gb=threshold,
                measurement=measurement.to_mapping(),
            )
        elif measured_peak > threshold or projected_peak > threshold:
            decision = ReplicaDecision(
                stage="four",
                admitted=False,
                selected_replicas=3,
                reason="measured or projected peak VRAM exceeds safety threshold",
                threshold_vram_gb=threshold,
                measured_peak_vram_gb=max(measured_peak, projected_peak),
                measurement=measurement.to_mapping(),
            )
        else:
            decision = ReplicaDecision(
                stage="four",
                admitted=True,
                selected_replicas=4,
                reason="measured/projected peak VRAM is within safety threshold",
                threshold_vram_gb=threshold,
                measured_peak_vram_gb=max(measured_peak, projected_peak),
                measurement=measurement.to_mapping(),
            )
        self.decisions.append(decision)
        return decision

    def run_staged(
        self,
        *,
        compatibility_preflight: Callable[[], GpuMeasurement],
        run_one: Callable[[], GpuMeasurement],
        run_three: Callable[[], GpuMeasurement],
        run_four: Callable[[], GpuMeasurement],
        reduce_to_replicas: Callable[[int], None],
    ) -> dict[str, Any]:
        """Run callbacks in order and return a durable, policy-level report.

        The callbacks own process launch/termination and artifact flushing.
        On an OOM from the fourth-stage callback, the policy calls
        ``reduce_to_replicas(3)`` exactly once and returns the three-replica
        result; it never retries the fourth launch in the same rollout.
        """

        self.decisions = []
        preflight = compatibility_preflight()
        self._validate_measurement(
            preflight,
            stage="compatibility preflight",
            expected_replica_count=1,
            require_loaded_model_vram=True,
        )
        self.decisions.append(
            ReplicaDecision(
                stage="preflight",
                admitted=True,
                selected_replicas=1,
                reason="exact B300 compatibility preflight passed",
                measurement=preflight.to_mapping(),
            )
        )
        try:
            one = run_one()
            self._validate_measurement(
                one,
                stage="one-replica stage",
                expected_replica_count=1,
                require_loaded_model_vram=True,
            )
        except GpuOutOfMemoryError:
            reduce_to_replicas(0)
            return {
                "status": "failure",
                "selected_replicas": 0,
                "reason": "one-replica stage OOM; all replicas terminated",
                "decisions": [decision.to_mapping() for decision in self.decisions],
            }
        self.decisions.append(
            ReplicaDecision(stage="one", admitted=True, selected_replicas=1, reason="one replica stable", measurement=one.to_mapping())
        )
        try:
            three = run_three()
            self._validate_measurement(
                three,
                stage="three-replica stage",
                expected_replica_count=3,
                require_loaded_model_vram=True,
            )
        except GpuOutOfMemoryError:
            reduce_to_replicas(0)
            return {
                "status": "failure",
                "selected_replicas": 0,
                "reason": "three-replica stage OOM; all replicas terminated",
                "decisions": [decision.to_mapping() for decision in self.decisions],
            }
        self.decisions.append(
            ReplicaDecision(stage="three", admitted=True, selected_replicas=3, reason="three replicas stable", measurement=three.to_mapping())
        )
        if self.target_replicas == 3:
            return {
                "status": "selected",
                "selected_replicas": 3,
                "reason": "target replica count is three",
                "decisions": [decision.to_mapping() for decision in self.decisions],
            }
        decision = self.admit_fourth(three)
        if not decision.admitted:
            reduce_to_replicas(3)
            return {
                "status": "selected",
                "selected_replicas": 3,
                "reason": decision.reason,
                "decisions": [item.to_mapping() for item in self.decisions],
            }
        # Replace the admission record with the actual four-replica runtime
        # measurement below; the durable decision list should contain one
        # entry per rollout stage.
        self.decisions.pop()
        try:
            four = run_four()
            self._validate_measurement(
                four,
                stage="four-replica stage",
                expected_replica_count=4,
                require_loaded_model_vram=True,
            )
        except GpuOutOfMemoryError:
            reduce_to_replicas(3)
            oom_decision = ReplicaDecision(
                stage="four",
                admitted=False,
                selected_replicas=3,
                reason="four-replica OOM; reduced to three without retry",
                threshold_vram_gb=decision.threshold_vram_gb,
                measured_peak_vram_gb=decision.measured_peak_vram_gb,
            )
            self.decisions.append(oom_decision)
            return {
                "status": "selected",
                "selected_replicas": 3,
                "reason": oom_decision.reason,
                "decisions": [item.to_mapping() for item in self.decisions],
            }
        self.decisions.append(
            ReplicaDecision(stage="four", admitted=True, selected_replicas=4, reason="four replicas stable", measurement=four.to_mapping())
        )
        return {
            "status": "selected",
            "selected_replicas": 4,
            "reason": "four replicas stable under measured rollout",
            "decisions": [item.to_mapping() for item in self.decisions],
        }


def cuda_memory_measurement(
    *,
    replica_count: int,
    gpu_type: str = B300_GPU_TYPE,
    gpu_count: int = B300_GPU_COUNT,
    device: int = 0,
    throughput_items_per_second: float | None = None,
    runtime_seconds: float | None = None,
) -> GpuMeasurement:
    """Capture CUDA allocator/device telemetry without requiring Torch at import."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional GPU dependency
        raise GpuRolloutError("torch is required for CUDA memory measurement") from exc
    if not torch.cuda.is_available():
        raise GpuRolloutError("CUDA is not available for memory measurement")
    properties = torch.cuda.get_device_properties(device)
    total_vram_gb = float(properties.total_memory) / (1024**3)
    peak_vram_gb = float(torch.cuda.max_memory_reserved(device)) / (1024**3)
    loaded_model_vram_gb = float(torch.cuda.memory_reserved(device)) / (1024**3)
    return GpuMeasurement(
        replica_count=replica_count,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        total_vram_gb=total_vram_gb,
        loaded_model_vram_gb=loaded_model_vram_gb,
        peak_vram_gb=peak_vram_gb,
        projected_peak_vram_gb=peak_vram_gb,
        throughput_items_per_second=throughput_items_per_second,
        runtime_seconds=runtime_seconds,
        details={"device_index": device, "device_name": str(properties.name)},
    )


__all__ = [
    "B300_GPU_COUNT",
    "B300_GPU_TYPE",
    "B300ReplicaRollout",
    "GpuMeasurement",
    "GpuOutOfMemoryError",
    "GpuRolloutError",
    "ReplicaDecision",
    "SUPPORTED_REPLICA_STAGES",
    "cuda_memory_measurement",
]
