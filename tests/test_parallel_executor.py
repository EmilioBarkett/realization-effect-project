from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import construct_benchmark.parallel_executor as parallel_executor_module
from construct_benchmark.parallel_executor import (
    AdapterError,
    InvalidCheckpointError,
    ParallelExecutor,
    WorkerAContractError,
    build_shard_manifests,
    inspect_campaign,
    read_output_progress,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_inventory(path: Path, count: int = 8, *, four_constructs: bool = False) -> Path:
    constructs = (
        ("decision", "epistemic", "social", "agentic")
        if four_constructs
        else ("decision",)
    )
    rows = [
        {
            "request_id": f"request_{index:03d}",
            "observation_ids": [f"observation_{index:03d}"],
            "construct_id": constructs[index % len(constructs)],
            "stage": "fake_stage",
        }
        for index in range(count)
    ]
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return path


def _run(inventory: Path, output: Path, **kwargs: object) -> dict:
    options = {
        "campaign": "parallel-test",
        "inventory": inventory,
        "output": output,
        "fake_model": True,
        "poll_interval_seconds": 0.001,
        "idle_timeout_seconds": 1.0,
    }
    options.update(kwargs)
    return ParallelExecutor(
        **options,
    ).run()


def _state(output: Path) -> dict:
    return json.loads((output / "campaign_state.json").read_text(encoding="utf-8"))


def test_production_execution_fails_closed_without_an_explicit_adapter(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=1)
    with pytest.raises(AdapterError, match="fail-closed.*fake-model"):
        ParallelExecutor(inventory=inventory, output=tmp_path / "production")


def test_fake_adapter_requires_explicit_opt_in(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=1)
    with pytest.raises(AdapterError, match="explicit --fake-model"):
        ParallelExecutor(inventory=inventory, output=tmp_path / "implicit-fake", adapter="fake")


def test_explicit_fake_opt_in_overrides_run_config_adapter(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=1)
    run_config = {"execution": {"parallel_executor": {"adapter": "gpu"}}}
    executor = ParallelExecutor(
        inventory=inventory,
        output=tmp_path / "explicit-fake",
        run_config=run_config,
        fake_model=True,
    )
    assert executor.adapter_identity == {"kind": "fake", "name": "deterministic_fake_v1"}


def test_explicit_fake_and_real_cli_adapter_are_rejected(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=1)
    with pytest.raises(AdapterError, match="cannot be combined"):
        ParallelExecutor(
            inventory=inventory,
            output=tmp_path / "contradictory",
            adapter="gpu",
            fake_model=True,
        )


def test_generic_full_engineering_shards_preserve_nonconfirmatory_identity(tmp_path: Path) -> None:
    inventory = tmp_path / "schema-neutral.json"
    inventory.write_text(
        json.dumps(
            [
                {"request_id": "request_001", "observation_ids": ["observation_001"], "stage": "fake_stage"},
                {"request_id": "request_002", "observation_ids": ["observation_002"], "stage": "fake_stage"},
            ]
        ),
        encoding="utf-8",
    )
    run_config = {
        "execution": {
            "default_run_mode": "full",
            "run_modes": {
                "full": {
                    "purpose": "full_coverage_engineering",
                    "confirmatory": False,
                    "engineering_only": True,
                }
            },
        }
    }
    output = tmp_path / "engineering"
    report = ParallelExecutor(
        inventory=inventory,
        output=output,
        run_config=run_config,
        fake_model=True,
        run_mode="full",
    ).run()
    assert report["status"] == "success"
    assert report["confirmatory"] is False
    assert report["engineering_only"] is True
    shard = json.loads((output / "shards" / "shard_000.json").read_text(encoding="utf-8"))
    assert shard["engineering_only"] is True


def test_gpu_adapter_records_one_pod_topology_and_uses_configured_replica_count(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=1)
    output = tmp_path / "gpu-dry-run"
    report = ParallelExecutor(
        inventory=inventory,
        output=output,
        run_config=ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json",
        adapter="gpu",
        worker_count=1,
        dry_run=True,
    ).run()
    assert report["status"] == "dry_run"
    assert report["execution_topology"] == {
        "provider": "runpod",
        "pod_count": 1,
        "gpu_type": "NVIDIA B300",
        "gpu_count": 1,
        "model_replica_count": 1,
        "worker_count_semantics": "in_pod_model_replicas",
    }


def test_gpu_adapter_refuses_more_constructs_than_model_replicas(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=2, four_constructs=True)
    with pytest.raises(AdapterError, match="construct-pure shards"):
        ParallelExecutor(
            inventory=inventory,
            output=tmp_path / "gpu-mixed",
            run_config=ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json",
            adapter="gpu",
            worker_count=1,
        )


def test_gpu_four_replica_plan_has_disjoint_construct_pure_outputs(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=8, four_constructs=True)
    output = tmp_path / "gpu-four"
    report = ParallelExecutor(
        inventory=inventory,
        output=output,
        run_config=ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_repaired_v2_engineering_full_v1.json",
        adapter="gpu",
        worker_count=4,
        dry_run=True,
        run_mode="full",
    ).run()
    state = _state(output)
    workers = state["workers"]
    assert report["status"] == "dry_run"
    assert report["execution_topology"]["pod_count"] == 1
    assert report["execution_topology"]["gpu_type"] == "NVIDIA B300"
    assert len(workers) == 4
    assert len({worker["output_path"] for worker in workers}) == 4
    construct_ids = []
    for worker in workers:
        shard = json.loads(Path(worker["shard_manifest_path"]).read_text(encoding="utf-8"))
        assert len(shard["construct_ids"]) == 1
        construct_ids.append(shard["construct_ids"][0])
    assert len(set(construct_ids)) == 4


def test_fake_worker_success_and_null_outputs_are_valid(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=5)
    output = tmp_path / "success"
    report = _run(inventory, output, worker_count=2)
    assert report["status"] == "success"
    assert report["progress"]["completed_request_count"] == 5
    assert report["progress"]["completed_observation_count"] == 5
    assert all(Path(path).is_file() for path in report["output_paths"])

    shard = build_shard_manifests(
        [{"request_id": "r", "observation_ids": ["o"], "construct_ids": ["decision"]}],
        output_dir=tmp_path / "null-shards",
        worker_count=1,
        parent_inventory_sha256="inventory",
        run_config_hash="config",
        run_mode="test",
        confirmatory=False,
        stage="stage",
        campaign_identity="campaign",
    )[0]
    null_output = tmp_path / "null.jsonl"
    null_output.write_text('{"request_id":"r","observation_id":"o","output":null}\n', encoding="utf-8")
    progress = read_output_progress(null_output, shard)
    assert progress.completed_request_ids == frozenset({"r"})
    assert progress.completed_observation_ids == frozenset({"o"})


def test_worker_a_three_slots_keep_four_physical_shards_and_scale_to_four_then_five(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=12, four_constructs=True)
    output = tmp_path / "scale"
    first = _run(inventory, output, worker_count=3)
    first_state = _state(output)
    assert first["status"] == "success"
    assert first_state["worker_count"] == 3
    assert first_state["physical_shard_count"] == 4
    assert len(first_state["physical_shards"]) == 4
    assert len(first_state["worker_schedule"]) == 3
    assert len(first_state["workers"]) == 3

    second = _run(inventory, output, worker_count=4, resume=True)
    second_state = _state(output)
    assert second["status"] == "success"
    assert second_state["worker_count"] == 4
    assert len(second_state["workers"]) == 4
    third = _run(inventory, output, worker_count=5, resume=True)
    third_state = _state(output)
    assert third["status"] == "success"
    assert third_state["worker_count"] == 5
    assert len(third_state["workers"]) == 5
    rows = []
    for worker in third_state["workers"]:
        rows.extend(
            json.loads(line)
            for line in Path(worker["output_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    assert len({row["request_id"] for row in rows}) == 12
    assert len(rows) == 12


def test_construct_aware_inventory_rejection_never_silently_falls_back(tmp_path: Path) -> None:
    inventory = tmp_path / "unsupported-construct-count.json"
    inventory.write_text(
        json.dumps(
            [
                {"request_id": "a", "observation_id": "a_obs", "construct_id": "construct_a"},
                {"request_id": "b", "observation_id": "b_obs", "construct_id": "construct_b"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorkerAContractError, match="generic fallback is disabled"):
        _run(inventory, tmp_path / "unsupported", worker_count=3)


def test_crash_restarts_from_checkpoint_without_duplicate_work(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=6)
    output = tmp_path / "crash"
    report = _run(
        inventory,
        output,
        worker_count=1,
        crash_after=1,
        crash_once=True,
        max_retries=2,
    )
    assert report["status"] == "success"
    rows = [
        json.loads(line)
        for line in Path(report["output_paths"][0]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 6
    assert len({row["observation_id"] for row in rows}) == 6
    assert _state(output)["workers"][0]["retry_count"] == 1


def test_resume_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=2)
    output = tmp_path / "malformed"
    _run(inventory, output, worker_count=1)
    manifest_path = output / "workers" / "worker_000" / "worker_manifest.json"
    manifest_path.write_text("{malformed\n", encoding="utf-8")
    with pytest.raises(InvalidCheckpointError, match="malformed JSON"):
        _run(inventory, output, worker_count=1, resume=True)


def test_max_retry_terminal_failure_is_explicit(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=4)
    output = tmp_path / "retry-terminal"
    report = _run(
        inventory,
        output,
        worker_count=1,
        crash_after=1,
        max_retries=1,
    )
    assert report["status"] == "failure"
    worker = _state(output)["workers"][0]
    assert worker["status"] == "failed"
    assert worker["retry_count"] == 1
    assert worker["terminal_reason"] == "worker_exit"
    assert report["continuation"]["resume_command"]


def test_worker_launch_failure_writes_terminal_report(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=2)

    def fail_to_spawn(command: object, **kwargs: object) -> object:
        raise OSError("simulated spawn failure")

    output = tmp_path / "launch-failure"
    report = _run(
        inventory,
        output,
        worker_count=1,
        max_retries=0,
        process_factory=fail_to_spawn,
    )

    assert report["status"] == "failure"
    assert report["terminal_reason"] == "worker_terminal_failure"
    worker = _state(output)["workers"][0]
    assert worker["status"] == "failed"
    assert worker["terminal_reason"] == "worker_launch"
    assert worker["error"] == "OSError: worker could not be started"
    assert (output / "terminal_report.json").is_file()


def test_budget_cutoff_refuses_launch_and_records_spend(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=3)
    output = tmp_path / "budget"
    report = _run(
        inventory,
        output,
        worker_count=2,
        hard_ceiling_usd=0.01,
        reserve_usd=0.0,
        gpu_hourly_rate_usd=36.0,
        worker_estimate_seconds=60.0,
    )
    assert report["status"] == "failure"
    assert report["terminal_reason"] == "budget_cutoff"
    assert report["budget"]["launch_refusals"]
    assert all(worker["status"] == "budget_refused" for worker in _state(output)["workers"])


def test_idle_watchdog_recovers_then_marks_terminal(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=3)
    output = tmp_path / "idle"
    report = _run(
        inventory,
        output,
        worker_count=1,
        stall_seconds=0.004,
        idle_timeout_seconds=0.001,
        max_retries=1,
    )
    assert report["status"] == "failure"
    worker = _state(output)["workers"][0]
    assert worker["terminal_reason"] == "idle_timeout"
    assert worker["retry_count"] == 1


def test_worker_outputs_and_logs_are_separate(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=8)
    output = tmp_path / "separate"
    report = _run(inventory, output, worker_count=3)
    state = _state(output)
    workers = state["workers"]
    assert len({worker["output_path"] for worker in workers}) == 3
    assert len({worker["worker_manifest_path"] for worker in workers}) == 3
    assert len({worker["log_path"] for worker in workers}) == 3
    assert all(Path(worker["output_path"]).parent == Path(worker["log_path"]).parent for worker in workers)
    assert report["status"] == "success"


def test_terminal_report_and_monitor_cli_are_durable(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=2)
    output = tmp_path / "cli"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_parallel_benchmark.py"),
        "--campaign",
        "cli-campaign",
        "--inventory",
        str(inventory),
        "--fake-model",
        "--worker-count",
        "1",
        "--output",
        str(output),
        "--poll-interval",
        "0.001",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["status"] == "success"
    terminal = json.loads((output / "terminal_report.json").read_text(encoding="utf-8"))
    assert terminal["manifest_type"] == "parallel_campaign_terminal_report"
    assert terminal["continuation"]["resume_command"]
    monitor = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "monitor_parallel_benchmark.py"),
            "--output",
            str(output),
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(monitor.stdout)["status"] == "success"


def test_shutdown_hook_runs_after_durable_report_and_preserves_scientific_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=2)
    calls: list[dict[str, object]] = []

    def fake_shutdown(command: object, *, context: dict[str, str], timeout_seconds: float) -> dict[str, object]:
        calls.append({"command": command, "context": context, "timeout": timeout_seconds})
        assert Path(context["terminal_report"]).is_file()
        return {
            "status": "succeeded",
            "return_code": 0,
            "argv_sha256": "hook-hash",
            "executable": "stop-tool",
            "argument_count": 2,
        }

    monkeypatch.setattr(parallel_executor_module, "run_terminal_shutdown", fake_shutdown)
    report = _run(
        inventory,
        tmp_path / "hook",
        worker_count=1,
        shutdown_command=["stop-tool", "{terminal_report}"],
    )

    assert report["status"] == "success"
    assert report["shutdown_hook"]["attempted"] is True
    assert report["shutdown_hook"]["succeeded"] is True
    assert report["shutdown_hook"]["argv_sha256"] == "hook-hash"
    assert calls[0]["command"] == ["stop-tool", "{terminal_report}"]
    terminal = json.loads((tmp_path / "hook" / "terminal_report.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "success"
    assert terminal["shutdown_hook"]["succeeded"] is True


def test_monitor_marks_stale_worker_heartbeat(tmp_path: Path) -> None:
    inventory = _write_inventory(tmp_path / "inventory.json", count=1)
    output = tmp_path / "stale"
    _run(inventory, output, worker_count=1, dry_run=True)
    state = _state(output)
    state["status"] = "running"
    state["idle_timeout_seconds"] = 1.0
    (output / "campaign_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    manifest_path = output / "workers" / "worker_000" / "worker_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["heartbeat_at_epoch"] = 0.0
    manifest["heartbeat_at"] = "1970-01-01T00:00:00Z"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    snapshot = inspect_campaign(output, now=10.0)
    assert snapshot["stale_workers"] == ["worker_000"]
