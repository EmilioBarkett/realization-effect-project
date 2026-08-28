from __future__ import annotations

import json
from pathlib import Path

import pytest

from construct_benchmark.concurrency_benchmark import (
    ConcurrencyPolicy,
    benchmark_concurrency,
    write_concurrency_report,
)


def _runner_factory(*, mismatch_at: int | None = None):
    calls: list[int] = []

    def runner(workload, worker_count):
        calls.append(worker_count)
        identities = {request_id: request_id for request_id in workload.request_ids}
        if mismatch_at == worker_count:
            identities[workload.request_ids[0]] = "different-output"
        elapsed = {1: 60.0, 3: 42.0, 4: 30.0, 5: 20.0}[worker_count]
        return {
            "elapsed_seconds": elapsed,
            "requested_requests": len(workload.request_ids),
            "valid_requests": len(workload.request_ids),
            "observations": 2 * len(workload.request_ids),
            "failures": 0,
            "retries": 0,
            "output_identities": identities,
            "hourly_rate": 12.0,
            "peak_vram_gb": 10.5,
            "gpu_utilization_pct": 82.0,
            "stable": True,
            "worker_metrics": [
                {"worker_id": index, "valid_requests": len(workload.request_ids) // worker_count, "observations": 2}
                for index in range(worker_count)
            ],
        }

    return runner, calls


def test_rollout_reaches_five_workers_and_records_telemetry(tmp_path: Path) -> None:
    runner, calls = _runner_factory()
    report = benchmark_concurrency(
        [{"request_id": f"request_{index}"} for index in range(20)],
        runner,
        policy=ConcurrencyPolicy(include_five_worker=True, max_peak_vram_gb=12.0),
    )

    assert calls == [1, 3, 4, 5]
    assert report["selection"]["selected_worker_count"] == 5
    assert [run["worker_count"] for run in report["runs"]] == [1, 3, 4, 5]
    five = report["runs"][-1]
    assert five["peak_vram_gb"] == pytest.approx(10.5)
    assert five["gpu_utilization_pct"] == pytest.approx(82.0)
    assert five["throughput"]["aggregate"]["valid_requests_per_minute"] == pytest.approx(60.0)
    assert len(five["throughput"]["workers"]) == 5
    assert all(item["status"] == "match" for item in report["output_identity_comparisons"])

    output = tmp_path / "concurrency.json"
    written = write_concurrency_report(report, output)
    assert written["frozen"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["report_sha256"] == written["report_sha256"]
    with pytest.raises(FileExistsError):
        write_concurrency_report(report, output)


def test_deterministic_output_identity_mismatch_rejects_candidate() -> None:
    runner, calls = _runner_factory(mismatch_at=3)
    report = benchmark_concurrency(
        [{"request_id": "a"}, {"request_id": "b"}],
        runner,
        policy=ConcurrencyPolicy(material_improvement=0.01),
    )

    assert calls == [1, 3]
    assert report["selection"]["selected_worker_count"] == 1
    comparison = report["output_identity_comparisons"][0]
    assert comparison["status"] == "mismatch"
    assert comparison["differing_request_ids"] == ["a"]
    assert "output_identity_mismatch" in report["selection"]["rollout"][1]["reasons"]


def test_cost_guard_blocks_a_run_without_inspecting_effects() -> None:
    def runner(workload, worker_count):
        return {
            "elapsed_seconds": 60.0,
            "requested_requests": len(workload.request_ids),
            "valid_requests": len(workload.request_ids),
            "observations": len(workload.request_ids),
        }

    report = benchmark_concurrency(
        [{"request_id": "a"}, {"request_id": "b"}],
        runner,
        policy=ConcurrencyPolicy(hourly_rate=360.0, max_cost_per_request=0.1),
    )

    assert report["selection"]["selected_worker_count"] is None
    assert "cost_per_request_threshold" in report["selection"]["rollout"][0]["reasons"]
    assert report["runs"][0]["throughput"]["aggregate"]["valid_requests_per_dollar"] == pytest.approx(1 / 3)


def test_registered_workload_size_is_enforced_when_requested() -> None:
    runner, _ = _runner_factory()
    with pytest.raises(ValueError, match="expected 100 requests, found 20"):
        benchmark_concurrency(
            [{"request_id": f"request_{index}"} for index in range(20)],
            runner,
            expected_request_count=100,
        )
