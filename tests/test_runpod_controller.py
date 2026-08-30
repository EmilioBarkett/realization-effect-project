from __future__ import annotations

import json
from pathlib import Path

import pytest

from construct_benchmark.runpod_controller import (
    B300_GPU_TYPE,
    RUNPOD_API_KEY_ENV,
    RunPodController,
    RunPodError,
    UrllibRunPodTransport,
)


def _spec() -> dict[str, object]:
    return {
        "campaign_id": "controller-test",
        "name": "controller-test-b300",
        "gpu_type_id": B300_GPU_TYPE,
        "gpu_count": 1,
        "pod_count": 1,
        "image_name": "pytorch/b300-test:latest",
        "network_volume_id": "volume-test",
        "volume_mount_path": "/workspace",
        "env": {"HF_HOME": "/workspace/huggingface"},
    }


def _pod(
    *,
    gpu_type: str = B300_GPU_TYPE,
    gpu_count: int = 1,
    pod_id: str = "pod_test_123",
) -> dict[str, object]:
    return {
        "id": pod_id,
        "name": "controller-test-b300",
        "gpu": {"id": gpu_type, "count": gpu_count, "memoryInGb": 192},
        "machine": {"gpuTypeId": gpu_type, "id": "machine_test"},
        "gpuCount": gpu_count,
        "costPerHr": 4.25,
        "imageName": "pytorch/b300-test:latest",
        "networkVolumeId": "volume-test",
        "volumeMountPath": "/workspace",
        "desiredStatus": "RUNNING",
        "status": "RUNNING",
    }


class FakeTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, object, object]] = []

    def request(self, method: str, path: str, *, query=None, body=None) -> object:
        self.calls.append((method, path, query, body))
        if not self.responses:
            raise AssertionError("unexpected transport call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_create_enforces_one_exact_b300_and_persistent_workspace(tmp_path: Path) -> None:
    transport = FakeTransport([_pod()])
    state = tmp_path / "controller.json"
    result = RunPodController(transport=transport, state_path=state).create_b300_pod(_spec())

    assert result["status"] == "created"
    assert result["gpu_type"] == B300_GPU_TYPE
    assert result["gpu_count"] == 1
    assert result["volume_mount_path"] == "/workspace"
    method, path, query, body = transport.calls[0]
    assert (method, path, query) == ("POST", "/pods", None)
    assert body["gpuTypeIds"] == [B300_GPU_TYPE]
    assert body["gpuCount"] == 1
    assert body["networkVolumeId"] == "volume-test"
    assert body["volumeMountPath"] == "/workspace"
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["pod_id"] == "pod_test_123"
    assert saved["spec"]["env_keys"] == ["HF_HOME"]
    assert "env" not in saved["spec"]


def test_controller_refuses_second_campaign_pod(tmp_path: Path) -> None:
    state = tmp_path / "controller.json"
    first = RunPodController(transport=FakeTransport([_pod()]), state_path=state)
    first.create_b300_pod(_spec())
    second_transport = FakeTransport([_pod()])
    with pytest.raises(RunPodError, match="second pod"):
        RunPodController(transport=second_transport, state_path=state).create_b300_pod(_spec())
    assert second_transport.calls == []


def test_controller_refuses_wrong_gpu_sku_and_count(tmp_path: Path) -> None:
    for index, (payload, message) in enumerate(
        ((_pod(gpu_type="NVIDIA B200"), "GPU SKU"), (_pod(gpu_count=2), "gpu_count"))
    ):
        with pytest.raises(RunPodError, match=message):
            RunPodController(
                transport=FakeTransport([payload]),
                state_path=tmp_path / f"state-{index}.json",
            ).create_b300_pod(_spec())


def test_controller_preserves_recovery_path_after_topology_failure(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    transport = FakeTransport([_pod(gpu_type="NVIDIA B200"), {}])
    controller = RunPodController(transport=transport, state_path=state)
    with pytest.raises(RunPodError, match="GPU SKU"):
        controller.create_b300_pod(_spec())

    failed = json.loads(state.read_text(encoding="utf-8"))
    assert failed["status"] == "recovery_required"
    assert failed["pod_id"] == "pod_test_123"

    stopped = controller.stop_pod()
    assert stopped["status"] == "stopped"
    assert transport.calls[-1][0:2] == ("POST", "/pods/pod_test_123/stop")


def test_controller_refuses_credential_fields_and_never_serializes_values() -> None:
    spec = _spec()
    spec[RUNPOD_API_KEY_ENV] = "new-secret"
    with pytest.raises(ValueError, match="credential-bearing") as error:
        RunPodController(transport=FakeTransport([])).create_b300_pod(spec)
    assert "new-secret" not in str(error.value)


def test_controller_refuses_runpod_config_in_pod_environment() -> None:
    spec = _spec()
    spec["env"] = {"RUNPOD_CONFIG": "/tmp/legacy-config"}
    with pytest.raises(ValueError, match="credential-bearing") as error:
        RunPodController(transport=FakeTransport([])).create_b300_pod(spec)
    assert "legacy-config" not in str(error.value)


def test_default_transport_accepts_only_new_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", "legacy-secret")
    with pytest.raises(RuntimeError, match=RUNPOD_API_KEY_ENV):
        UrllibRunPodTransport()


def test_transport_cannot_bypass_active_credential_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RUNPOD_API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=RUNPOD_API_KEY_ENV):
        UrllibRunPodTransport(api_key="explicit-secret")

    monkeypatch.setenv(RUNPOD_API_KEY_ENV, "active-secret")
    with pytest.raises(RuntimeError, match="only accepted"):
        UrllibRunPodTransport(api_key="legacy-secret")


def test_readiness_stop_and_durable_lifecycle_state(tmp_path: Path) -> None:
    state = tmp_path / "controller.json"
    transport = FakeTransport([_pod(), _pod(), {}])
    controller = RunPodController(transport=transport, state_path=state)
    controller.create_b300_pod(_spec())
    inspected = controller.inspect_pod()
    assert inspected["status"] == "ready"
    assert inspected["runtime"]["ready"] is True
    stopped = controller.stop_pod()
    assert stopped["status"] == "stopped"
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["status"] == "stopped"
    assert saved["pod_id"] == "pod_test_123"
    assert transport.calls[1][0:2] == ("GET", "/pods/pod_test_123")
    assert transport.calls[1][2] == {"includeMachine": "true", "includeNetworkVolume": "true"}
    assert transport.calls[2][0:2] == ("POST", "/pods/pod_test_123/stop")


def test_endpoint_without_runtime_status_is_ready(tmp_path: Path) -> None:
    payload = _pod()
    payload.pop("status")
    payload["publicIp"] = "203.0.113.10"
    controller = RunPodController(
        transport=FakeTransport([payload, {**payload, "publicIp": "203.0.113.10"}]),
        state_path=tmp_path / "state.json",
    )
    controller.create_b300_pod(_spec())
    inspected = controller.inspect_pod()
    assert inspected["status"] == "ready"
    assert inspected["runtime"]["ready"] is True


def test_controller_rejects_invalid_create_response_pod_id(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    controller = RunPodController(
        transport=FakeTransport([_pod(pod_id="../not-a-pod")]),
        state_path=state,
    )
    with pytest.raises(RunPodError, match="invalid pod ID"):
        controller.create_b300_pod(_spec())
    assert not state.exists()


def test_controller_rejects_invalid_availability_pod_id() -> None:
    controller = RunPodController(transport=FakeTransport([{"pods": [_pod(pod_id="bad/id")]}]))
    with pytest.raises(RunPodError, match="invalid pod ID"):
        controller.list_pods()


def test_controller_inspect_requires_response_to_match_owned_pod(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    controller = RunPodController(
        transport=FakeTransport([_pod(), _pod(pod_id="pod_other")]),
        state_path=state,
    )
    controller.create_b300_pod(_spec())
    with pytest.raises(RunPodError, match="did not match"):
        controller.inspect_pod()
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["status"] == "created"
    assert saved["pod_id"] == "pod_test_123"


@pytest.mark.parametrize("stop_response", [{"id": "pod_other"}, None])
def test_controller_does_not_mark_stop_success_for_invalid_response(
    tmp_path: Path,
    stop_response: object,
) -> None:
    state = tmp_path / "state.json"
    controller = RunPodController(
        transport=FakeTransport([_pod(), stop_response]),
        state_path=state,
    )
    controller.create_b300_pod(_spec())
    with pytest.raises(RunPodError):
        controller.stop_pod()
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["status"] == "created"
    assert saved["pod_id"] == "pod_test_123"


def test_controller_refuses_recreate_after_stop(tmp_path: Path) -> None:
    state = tmp_path / "controller.json"
    transport = FakeTransport([_pod(), {}])
    controller = RunPodController(transport=transport, state_path=state)
    controller.create_b300_pod(_spec())
    controller.stop_pod()
    replacement_transport = FakeTransport([_pod()])
    with pytest.raises(RunPodError, match="second pod"):
        RunPodController(transport=replacement_transport, state_path=state).create_b300_pod(_spec())
    assert replacement_transport.calls == []


def test_dry_run_is_network_free_and_credential_safe(tmp_path: Path) -> None:
    transport = FakeTransport([])
    result = RunPodController(transport=transport, state_path=tmp_path / "state.json").create_b300_pod(
        _spec(), dry_run=True
    )
    assert result["status"] == "dry_run"
    assert transport.calls == []
    assert "secret" not in json.dumps(result).lower()


def test_live_create_and_stop_require_durable_state(tmp_path: Path) -> None:
    with pytest.raises(RunPodError, match="durable controller state"):
        RunPodController(transport=FakeTransport([_pod()])).create_b300_pod(_spec())
    with pytest.raises(RunPodError, match="durable controller state"):
        RunPodController(transport=FakeTransport([{}])).stop_pod("pod_test_123")
