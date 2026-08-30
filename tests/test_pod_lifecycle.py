from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from construct_benchmark.pod_lifecycle import normalize_terminal_command, run_terminal_shutdown


ROOT = Path(__file__).resolve().parents[1]


def _load_stop_module():
    path = ROOT / "scripts" / "stop_benchmark_pod.py"
    spec = importlib.util.spec_from_file_location("stop_benchmark_pod_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_terminal_shutdown_is_argv_only_and_does_not_return_command_contents() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    result = run_terminal_shutdown(
        ["shutdown-tool", "--report", "{terminal_report}", "--status", "{status}"],
        context={"terminal_report": "/tmp/terminal_report.json", "status": "success"},
        runner=runner,
    )

    assert result["status"] == "succeeded"
    assert result["return_code"] == 0
    assert calls[0][0] == ["shutdown-tool", "--report", "/tmp/terminal_report.json", "--status", "success"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["stdout"] is subprocess.DEVNULL
    assert calls[0][1]["stderr"] is subprocess.DEVNULL
    assert "/tmp/terminal_report.json" not in json.dumps(result)


def test_terminal_shutdown_records_nonzero_and_timeout_without_raising() -> None:
    failed = run_terminal_shutdown(
        ["shutdown-tool"],
        context={},
        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
    )
    assert failed["status"] == "failed"
    assert failed["return_code"] == 7

    def timeout_runner(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("shutdown-tool", 1)

    timed_out = run_terminal_shutdown(["shutdown-tool"], context={}, runner=timeout_runner)
    assert timed_out["status"] == "timeout"
    assert timed_out["error"] == "shutdown command exceeded its timeout"


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -float("inf")])
def test_terminal_shutdown_rejects_non_finite_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        run_terminal_shutdown(["shutdown-tool"], context={}, timeout_seconds=timeout)


def test_terminal_shutdown_scrubs_legacy_runpod_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def runner(_argv: list[str], **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("RUNPOD_2_API_KEY", "active-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "legacy-secret")
    monkeypatch.setenv("RUNPOD_CONFIG", "/tmp/legacy-runpod-config")
    result = run_terminal_shutdown(["shutdown-tool"], context={}, runner=runner)

    assert result["status"] == "succeeded"
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["RUNPOD_2_API_KEY"] == "active-secret"
    assert "RUNPOD_API_KEY" not in child_env
    assert "RUNPOD_CONFIG" not in child_env


def test_terminal_command_rejects_empty_and_nul_arguments() -> None:
    with pytest.raises(ValueError, match="executable"):
        normalize_terminal_command([])
    with pytest.raises(ValueError, match="NUL"):
        normalize_terminal_command(["tool\x00bad"])


def test_stop_pod_uses_only_the_new_controller_credential_and_sanitizes_result(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_stop_module()
    captured: dict[str, object] = {}

    def opener(request: object, *, timeout: float) -> _Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(200)

    result = module.stop_pod("pod_abc-123", api_key="new-secret", opener=opener)
    request = captured["request"]
    assert result["status"] == "stopped"
    assert result["http_status"] == 200
    assert captured["timeout"] == 30.0
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer new-secret"
    assert "new-secret" not in json.dumps(result)

    monkeypatch.delenv(module.RUNPOD_API_KEY_ENV, raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", "legacy-secret")
    with pytest.raises(RuntimeError, match=module.RUNPOD_API_KEY_ENV):
        module.stop_pod("pod_abc-123", opener=opener)


def test_stop_pod_dry_run_is_network_free_and_cli_reports_no_secret(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_stop_module()
    monkeypatch.delenv(module.RUNPOD_POD_ID_ENV, raising=False)
    assert module.main(["--pod-id", "pod_abc-123", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run"
    assert result["request_method"] == "POST"
    assert module.RUNPOD_API_KEY_ENV in result["credential_env"]
