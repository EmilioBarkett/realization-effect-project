from __future__ import annotations

import json
from pathlib import Path
import sys

from construct_benchmark.distributed_contracts import canonical_hash
from construct_benchmark.gpu_worker import (
    _construct_id,
    _runtime_compatibility_preflight,
    _validate_output_ownership,
    run_gpu_worker,
)
from construct_benchmark.parallel_executor import (
    _worker_payload,
    _write_worker_manifest,
    build_shard_manifests,
    load_shard_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json"


def _worker_fixture(
    tmp_path: Path,
    *,
    stage: str = "readout",
    run_config_path: Path = RUN_CONFIG,
) -> tuple[Path, Path, Path]:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            [
                {
                    "request_id": "request_001",
                    "observation_ids": ["observation_001"],
                    "construct_id": "realization_account_closure",
                    "prompt_text": "A short frozen prompt.",
                    "split": "direction_train",
                    "prompt_role": "probe",
                }
            ]
        ),
        encoding="utf-8",
    )
    shard_dir = tmp_path / "shards"
    shards = build_shard_manifests(
        [json.loads(inventory.read_text(encoding="utf-8"))[0]],
        output_dir=shard_dir,
        worker_count=1,
        parent_inventory_sha256="inventory",
        run_config_hash=canonical_hash(json.loads(run_config_path.read_text(encoding="utf-8"))),
        run_mode="test",
        confirmatory=False,
        stage=stage,
        campaign_identity="campaign",
    )
    shard_path = Path(shards[0]["manifest_path"])
    shard = load_shard_manifest(shard_path)
    worker_dir = tmp_path / "workers" / "worker_000"
    output = worker_dir / "output.jsonl"
    worker_manifest_path = worker_dir / "worker_manifest.json"
    worker_manifest = _worker_payload(
        worker_id="worker_000",
        shard=shard,
        status="planned",
        output_path=output,
        worker_manifest_path=worker_manifest_path,
        stage=stage,
        retry_count=0,
        execution_identity={
            "run_config_hash": canonical_hash(json.loads(run_config_path.read_text(encoding="utf-8"))),
            "run_mode": "test",
            "confirmatory": False,
            "stage": stage,
        },
    )
    _write_worker_manifest(worker_manifest_path, worker_manifest)
    return shard_path, worker_manifest_path, output


def test_gpu_worker_refuses_mixed_constructs_and_shared_output_layout(tmp_path: Path) -> None:
    mixed = {"construct_ids": ["construct_a", "construct_b"]}
    try:
        _construct_id(mixed)
    except Exception as exc:
        assert "construct-pure" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("mixed construct shard was accepted")

    worker_dir = tmp_path / "workers" / "worker_000"
    _validate_output_ownership(
        output=worker_dir / "output.jsonl",
        worker_manifest_path=worker_dir / "worker_manifest.json",
        worker_id="worker_000",
    )
    try:
        _validate_output_ownership(
            output=tmp_path / "shared" / "output.jsonl",
            worker_manifest_path=worker_dir / "worker_manifest.json",
            worker_id="worker_000",
        )
    except Exception as exc:
        assert "worker-owned" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("shared output layout was accepted")


def test_external_gpu_stage_requires_durable_prerequisite(tmp_path: Path) -> None:
    shard_path, worker_manifest_path, output = _worker_fixture(tmp_path)
    result = run_gpu_worker(
        shard_manifest_path=shard_path,
        worker_manifest_path=worker_manifest_path,
        output_path=output,
        stage="readout",
        run_config_path=RUN_CONFIG,
    )
    assert result == 1
    manifest = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "prerequisite_manifest" in manifest["error"]
    assert not output.exists() or output.read_text(encoding="utf-8") == ""


def test_external_gpu_stage_checkpoint_resume_does_not_duplicate_rows(tmp_path: Path) -> None:
    prerequisite = tmp_path / "residual_manifest.json"
    prerequisite.write_text("{}\n", encoding="utf-8")
    payload = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    payload.setdefault("execution", {}).setdefault("parallel_executor", {})["gpu_worker"] = {
        "stages": {
            "readout": {
                "prerequisite_manifest": str(prerequisite),
                "command": [sys.executable, "-c", "pass"],
            }
        }
    }
    run_config_path = tmp_path / "run_config.json"
    run_config_path.write_text(json.dumps(payload), encoding="utf-8")
    shard_path, worker_manifest_path, output = _worker_fixture(
        tmp_path,
        run_config_path=run_config_path,
    )
    assert (
        run_gpu_worker(
            shard_manifest_path=shard_path,
            worker_manifest_path=worker_manifest_path,
            output_path=output,
            stage="readout",
            run_config_path=run_config_path,
        )
        == 0
    )
    first_rows = output.read_text(encoding="utf-8").splitlines()
    assert len(first_rows) == 1
    assert (
        run_gpu_worker(
            shard_manifest_path=shard_path,
            worker_manifest_path=worker_manifest_path,
            output_path=output,
            stage="readout",
            run_config_path=run_config_path,
        )
        == 0
    )
    assert output.read_text(encoding="utf-8").splitlines() == first_rows


class _FakeCudaProperties:
    name = "NVIDIA B300"
    total_memory = 192 * 1024**3


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, _index: int) -> _FakeCudaProperties:
        return _FakeCudaProperties()


class _FakeTorch:
    __version__ = "fixture-torch"
    cuda = _FakeCuda()

    class version:
        cuda = "fixture-cuda"


def test_runtime_preflight_requires_one_exact_b300() -> None:
    report = _runtime_compatibility_preflight(device="cuda", torch_module=_FakeTorch())
    assert report["expected_gpu_type"] == "NVIDIA B300"
    assert report["visible_gpu_count"] == 1
    assert report["total_vram_gb"] == 192


def test_runtime_preflight_rejects_non_b300() -> None:
    class WrongCuda(_FakeCuda):
        def get_device_properties(self, _index: int) -> object:
            return type("Properties", (), {"name": "NVIDIA H100", "total_memory": 80 * 1024**3})()

    class WrongTorch(_FakeTorch):
        cuda = WrongCuda()

    try:
        _runtime_compatibility_preflight(device="cuda", torch_module=WrongTorch())
    except Exception as exc:
        assert "NVIDIA H100" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("non-B300 runtime was accepted")
