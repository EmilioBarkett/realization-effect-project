from __future__ import annotations

import pytest

from construct_benchmark.gpu_rollout import (
    B300ReplicaRollout,
    GpuMeasurement,
    GpuOutOfMemoryError,
    GpuRolloutError,
)


def _measurement(replica_count: int, *, projected: float = 70.0, stable: bool = True) -> GpuMeasurement:
    return GpuMeasurement(
        replica_count=replica_count,
        total_vram_gb=80.0,
        loaded_model_vram_gb=32.0,
        peak_vram_gb=projected,
        projected_peak_vram_gb=projected,
        stable=stable,
    )


def test_rollout_runs_one_compatibility_then_three_then_four() -> None:
    calls: list[str] = []
    rollout = B300ReplicaRollout(safety_fraction=0.95)
    result = rollout.run_staged(
        compatibility_preflight=lambda: (calls.append("preflight") or _measurement(1, projected=35.0)),
        run_one=lambda: (calls.append("one") or _measurement(1, projected=40.0)),
        run_three=lambda: (calls.append("three") or _measurement(3, projected=60.0)),
        run_four=lambda: (calls.append("four") or _measurement(4, projected=70.0)),
        reduce_to_replicas=lambda count: calls.append(f"reduce:{count}"),
    )
    assert calls == ["preflight", "one", "three", "four"]
    assert result["status"] == "selected"
    assert result["selected_replicas"] == 4
    assert [item["stage"] for item in result["decisions"]] == ["preflight", "one", "three", "four"]


def test_vram_threshold_blocks_fourth_without_launching_it() -> None:
    calls: list[str] = []
    rollout = B300ReplicaRollout(safety_fraction=0.75)
    result = rollout.run_staged(
        compatibility_preflight=lambda: (calls.append("preflight") or _measurement(1, projected=35.0)),
        run_one=lambda: (calls.append("one") or _measurement(1, projected=40.0)),
        run_three=lambda: (calls.append("three") or _measurement(3, projected=61.0)),
        run_four=lambda: (calls.append("four") or _measurement(4, projected=70.0)),
        reduce_to_replicas=lambda count: calls.append(f"reduce:{count}"),
    )
    assert result["selected_replicas"] == 3
    assert calls == ["preflight", "one", "three", "reduce:3"]
    assert "exceeds safety threshold" in result["reason"]


def test_fourth_oom_reduces_to_three_once_without_retry() -> None:
    calls: list[str] = []
    rollout = B300ReplicaRollout()

    def run_four() -> GpuMeasurement:
        calls.append("four")
        raise GpuOutOfMemoryError("simulated B300 OOM")

    result = rollout.run_staged(
        compatibility_preflight=lambda: (calls.append("preflight") or _measurement(1, projected=35.0)),
        run_one=lambda: (calls.append("one") or _measurement(1, projected=40.0)),
        run_three=lambda: (calls.append("three") or _measurement(3, projected=60.0)),
        run_four=run_four,
        reduce_to_replicas=lambda count: calls.append(f"reduce:{count}"),
    )
    assert result["selected_replicas"] == 3
    assert calls == ["preflight", "one", "three", "four", "reduce:3"]
    assert "without retry" in result["reason"]


def test_exact_gpu_and_explicit_measurements_are_required() -> None:
    rollout = B300ReplicaRollout()
    with pytest.raises(GpuRolloutError, match="exact"):
        rollout.admit_fourth(
            GpuMeasurement(replica_count=3, gpu_type="NVIDIA B200", total_vram_gb=80.0, peak_vram_gb=60.0)
        )
    blocked = rollout.admit_fourth(GpuMeasurement(replica_count=3, total_vram_gb=80.0))
    assert blocked.admitted is False
    assert "explicit measured and projected" in blocked.reason


def test_rollout_requires_stage_replica_counts_and_loaded_model_memory() -> None:
    rollout = B300ReplicaRollout()
    with pytest.raises(GpuRolloutError, match="loaded-model VRAM"):
        rollout.run_staged(
            compatibility_preflight=lambda: GpuMeasurement(replica_count=1, total_vram_gb=80.0),
            run_one=lambda: _measurement(1),
            run_three=lambda: _measurement(3),
            run_four=lambda: _measurement(4),
            reduce_to_replicas=lambda _count: None,
        )
    with pytest.raises(GpuRolloutError, match="expected 1"):
        rollout.run_staged(
            compatibility_preflight=lambda: _measurement(3),
            run_one=lambda: _measurement(1),
            run_three=lambda: _measurement(3),
            run_four=lambda: _measurement(4),
            reduce_to_replicas=lambda _count: None,
        )
