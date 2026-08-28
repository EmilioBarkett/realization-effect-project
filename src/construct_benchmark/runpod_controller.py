"""Local, credential-safe RunPod controller for the single-B300 topology.

The controller is intentionally local-only.  It creates, inspects, and stops
one exact GPU pod, while the pod itself receives no RunPod credential.  The
HTTP transport is injectable so all lifecycle and topology checks can be
tested without contacting RunPod.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


RUNPOD_API_KEY_ENV = "RUNPOD_2_API_KEY"
RUNPOD_BASE_URL = "https://rest.runpod.io/v1"
B300_GPU_TYPE = "NVIDIA B300"
B300_GPU_COUNT = 1
MAX_CAMPAIGN_PODS = 1
WORKSPACE_MOUNT_PATH = "/workspace"
CONTROLLER_SCHEMA_VERSION = "runpod_b300_controller_v1"
_POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SECRET_KEY_PATTERN = re.compile(r"(?:api[_-]?key|access[_-]?token|secret|password|credential)", re.I)
_PLACEHOLDER_PATTERN = re.compile(r"(?:REPLACE|REVIEW|TODO|CHANGEME)", re.I)


class RunPodError(RuntimeError):
    """Raised when the RunPod lifecycle or exact-topology contract fails."""


class RunPodTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> Any: ...


def normalize_pod_id(value: str | None) -> str:
    candidate = str(value or "").strip()
    if not candidate or not _POD_ID_PATTERN.fullmatch(candidate):
        raise ValueError("pod_id must contain only letters, digits, underscores, or hyphens")
    return candidate


def _finite_positive(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return parsed


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def sanitize_pod(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a RunPod response to lifecycle/provenance fields only.

    In particular, environment mappings, command strings, and arbitrary
    response fields are never copied into the durable controller state.
    """

    gpu = _nested_mapping(payload, "gpu")
    machine = _nested_mapping(payload, "machine")
    network_volume = _nested_mapping(payload, "networkVolume")
    gpu_type_ids = payload.get("gpuTypeIds")
    first_gpu_type_id = gpu_type_ids[0] if isinstance(gpu_type_ids, list) and gpu_type_ids else None
    gpu_type = _safe_text(
        _first_value(
            gpu.get("id"),
            machine.get("gpuTypeId"),
            payload.get("gpuTypeId"),
            first_gpu_type_id,
            gpu.get("displayName"),
        )
    )
    gpu_count = _safe_number(_first_value(gpu.get("count"), payload.get("gpuCount"), machine.get("gpuCount")))
    volume_id = _safe_text(
        _first_value(
            payload.get("networkVolumeId"),
            network_volume.get("id"),
            payload.get("volumeId"),
        )
    )
    image = _safe_text(_first_value(payload.get("imageName"), payload.get("image")))
    lifecycle = {
        "desired_status": _safe_text(payload.get("desiredStatus")),
        "actual_status": _safe_text(
            _first_value(payload.get("status"), payload.get("runtimeStatus"), payload.get("state"))
        ),
        "created_at": _safe_text(payload.get("createdAt")),
        "last_status_change": _safe_text(
            _first_value(payload.get("lastStatusChange"), payload.get("lastStatusChangeAt"))
        ),
    }
    return {
        "pod_id": _safe_text(_first_value(payload.get("id"), payload.get("podId"))),
        "name": _safe_text(payload.get("name")),
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "gpu_memory_gb": _safe_number(_first_value(gpu.get("memoryInGb"), gpu.get("memoryGb"))),
        "cost_per_hour_usd": _safe_number(
            _first_value(payload.get("costPerHr"), payload.get("costPerHour"), payload.get("adjustedCostPerHr"))
        ),
        "image": image,
        "network_volume_id": volume_id,
        "volume_mount_path": _safe_text(payload.get("volumeMountPath")),
        "volume_size_gb": _safe_number(
            _first_value(network_volume.get("size"), network_volume.get("sizeInGb"), payload.get("volumeInGb"))
        ),
        "machine_id": _safe_text(_first_value(machine.get("id"), payload.get("machineId"))),
        "data_center_id": _safe_text(
            _first_value(machine.get("dataCenterId"), payload.get("dataCenterId"))
        ),
        "lifecycle": lifecycle,
        "endpoint_present": bool(
            _first_value(payload.get("publicIp"), payload.get("publicIpAddress"), payload.get("runtime"))
        ),
    }


def _safe_spec_for_state(spec: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "schema_version",
        "campaign_id",
        "name",
        "gpu_type_id",
        "gpu_count",
        "pod_count",
        "cloud_type",
        "compute_type",
        "container_disk_gb",
        "image_name",
        "network_volume_id",
        "volume_mount_path",
        "data_center_ids",
        "ports",
    ):
        if key in spec:
            result[key] = spec[key]
    if isinstance(spec.get("env"), Mapping):
        result["env_keys"] = sorted(str(key) for key in spec["env"])
    return result


def _reject_secret_fields(value: Any, *, path: str = "spec") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text == RUNPOD_API_KEY_ENV or _SECRET_KEY_PATTERN.search(key_text):
                raise ValueError(f"{path}.{key_text} is a credential-bearing field and is not allowed")
            _reject_secret_fields(nested, path=f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_secret_fields(nested, path=f"{path}[{index}]")


class UrllibRunPodTransport:
    """Small stdlib REST transport that keeps the credential in a header only."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = RUNPOD_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        credential = api_key if api_key is not None else os.environ.get(RUNPOD_API_KEY_ENV, "")
        if not str(credential).strip():
            raise RuntimeError(f"{RUNPOD_API_KEY_ENV} is not set")
        self._api_key = str(credential)
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = _finite_positive(timeout_seconds, field="timeout_seconds")

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        normalized_path = "/" + str(path).lstrip("/")
        query_values = {str(key): value for key, value in (query or {}).items() if value not in (None, "")}
        query_string = urllib.parse.urlencode(query_values, doseq=True)
        url = f"{self.base_url}{normalized_path}"
        if query_string:
            url = f"{url}?{query_string}"
        request_body = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if body is not None:
            request_body = json.dumps(dict(body), ensure_ascii=True, sort_keys=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=request_body, headers=headers, method=str(method).upper())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = int(response.getcode())
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise RunPodError(f"RunPod returned HTTP status {int(exc.code)} for {str(method).upper()} {normalized_path}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RunPodError(f"RunPod request failed for {str(method).upper()} {normalized_path}") from None
        if not 200 <= status < 300:
            raise RunPodError(f"RunPod returned HTTP status {status} for {str(method).upper()} {normalized_path}")
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RunPodError(f"RunPod returned a non-JSON response for {str(method).upper()} {normalized_path}") from None


@dataclass
class RunPodController:
    """Enforce the one-pod/one-B300 lifecycle contract."""

    transport: RunPodTransport | None = None
    api_key: str | None = None
    state_path: Path | None = None
    expected_gpu_type: str = B300_GPU_TYPE
    max_pods: int = MAX_CAMPAIGN_PODS
    base_url: str = RUNPOD_BASE_URL
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.expected_gpu_type != B300_GPU_TYPE:
            raise ValueError(
                f"This campaign is pinned to exact GPU SKU {B300_GPU_TYPE!r}; alternate SKUs are refused."
            )
        if int(self.max_pods) != MAX_CAMPAIGN_PODS:
            raise ValueError("The B300 campaign supports exactly one pod; max_pods cannot be increased.")
        self.max_pods = int(self.max_pods)
        if self.state_path is not None:
            self.state_path = Path(self.state_path).expanduser().resolve()
        if self.transport is None:
            self.transport = UrllibRunPodTransport(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout_seconds=self.timeout_seconds,
            )

    def _call(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        assert self.transport is not None
        return self.transport.request(method, path, query=query, body=body)

    def _read_state(self) -> dict[str, Any]:
        if self.state_path is None or not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunPodError(f"controller state is unreadable: {self.state_path}") from exc
        if not isinstance(payload, dict):
            raise RunPodError("controller state must be a JSON object")
        return dict(payload)

    def _write_state(self, payload: Mapping[str, Any]) -> None:
        if self.state_path is None:
            return
        destination = self.state_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                json.dump(dict(payload), handle, indent=2, ensure_ascii=True, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    def _record_state(self, *, status: str, summary: Mapping[str, Any], spec: Mapping[str, Any] | None = None) -> None:
        previous = self._read_state()
        payload = {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "status": status,
            "pod": dict(summary),
            "pod_id": summary.get("pod_id"),
            "gpu_type": summary.get("gpu_type"),
            "gpu_count": summary.get("gpu_count"),
            "cost_per_hour_usd": summary.get("cost_per_hour_usd"),
            "image": summary.get("image"),
            "network_volume_id": summary.get("network_volume_id"),
            "volume_mount_path": summary.get("volume_mount_path"),
            "lifecycle": summary.get("lifecycle", {}),
        }
        if spec is not None:
            payload["spec"] = _safe_spec_for_state(spec)
        elif isinstance(previous.get("spec"), Mapping):
            payload["spec"] = dict(previous["spec"])
        self._write_state(payload)

    def _owned_pod_id(self) -> str | None:
        state = self._read_state()
        raw = state.get("pod_id")
        return normalize_pod_id(raw) if raw else None

    def list_pods(self, *, gpu_type_id: str = B300_GPU_TYPE, data_center_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        if gpu_type_id != B300_GPU_TYPE:
            raise ValueError(f"availability queries are pinned to exact GPU SKU {B300_GPU_TYPE!r}")
        query: dict[str, Any] = {
            "computeType": "GPU",
            "gpuTypeId": B300_GPU_TYPE,
            "includeMachine": "true",
            "includeNetworkVolume": "true",
        }
        if data_center_ids:
            query["dataCenterIds"] = list(data_center_ids)
        response = self._call("GET", "/pods", query=query)
        if isinstance(response, Mapping):
            raw_pods = response.get("pods", response.get("data", []))
        else:
            raw_pods = response
        if not isinstance(raw_pods, list):
            raise RunPodError("RunPod pod listing did not return a list")
        return [sanitize_pod(item) for item in raw_pods if isinstance(item, Mapping)]

    def query_availability(
        self,
        *,
        data_center_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        pods = self.list_pods(data_center_ids=data_center_ids)
        return {
            "status": "ok",
            "requested_gpu_type": B300_GPU_TYPE,
            "requested_gpu_count": B300_GPU_COUNT,
            "max_campaign_pods": self.max_pods,
            "matching_pods": pods,
            "matching_pod_count": len(pods),
        }

    def _validate_spec(self, spec: Mapping[str, Any], *, allow_placeholders: bool = False) -> dict[str, Any]:
        _reject_secret_fields(spec)
        normalized = dict(spec)
        if int(normalized.get("pod_count", 1)) != 1:
            raise ValueError("pod_count must be exactly 1")
        if normalized.get("gpu_type_id", B300_GPU_TYPE) != B300_GPU_TYPE:
            raise ValueError(f"gpu_type_id must be exactly {B300_GPU_TYPE!r}")
        if int(normalized.get("gpu_count", B300_GPU_COUNT)) != B300_GPU_COUNT:
            raise ValueError("gpu_count must be exactly 1")
        if str(normalized.get("compute_type", "GPU")).upper() != "GPU":
            raise ValueError("compute_type must be GPU")
        image_name = _safe_text(normalized.get("image_name"))
        if image_name is None:
            raise ValueError("image_name is required")
        volume_id = _safe_text(normalized.get("network_volume_id"))
        if volume_id is None:
            raise ValueError("network_volume_id is required for the persistent /workspace volume")
        mount_path = str(normalized.get("volume_mount_path", WORKSPACE_MOUNT_PATH)).strip()
        if mount_path != WORKSPACE_MOUNT_PATH:
            raise ValueError(f"volume_mount_path must be exactly {WORKSPACE_MOUNT_PATH!r}")
        if not allow_placeholders and (_PLACEHOLDER_PATTERN.search(image_name) or _PLACEHOLDER_PATTERN.search(volume_id)):
            raise ValueError("image_name and network_volume_id must be reviewed concrete values before create")
        name = _safe_text(normalized.get("name"))
        if name is None:
            raise ValueError("name is required")
        data_centers = normalized.get("data_center_ids", [])
        if data_centers is not None and not isinstance(data_centers, list):
            raise ValueError("data_center_ids must be a list")
        ports = normalized.get("ports", [])
        if ports is not None and not isinstance(ports, list):
            raise ValueError("ports must be a list")
        env = normalized.get("env", {})
        if env is not None and not isinstance(env, Mapping):
            raise ValueError("env must be an object")
        return normalized

    def _create_body(self, spec: Mapping[str, Any], *, allow_placeholders: bool = False) -> dict[str, Any]:
        normalized = self._validate_spec(spec, allow_placeholders=allow_placeholders)
        body: dict[str, Any] = {
            "name": str(normalized["name"]),
            "cloudType": str(normalized.get("cloud_type", "SECURE")),
            "computeType": "GPU",
            "gpuCount": B300_GPU_COUNT,
            "gpuTypeIds": [B300_GPU_TYPE],
            "imageName": str(normalized["image_name"]),
            "networkVolumeId": str(normalized["network_volume_id"]),
            "volumeMountPath": WORKSPACE_MOUNT_PATH,
            "containerDiskInGb": int(normalized.get("container_disk_gb", 100)),
            "env": dict(normalized.get("env") or {}),
        }
        for source, target in (
            ("data_center_ids", "dataCenterIds"),
            ("ports", "ports"),
            ("docker_entrypoint", "dockerEntrypoint"),
            ("docker_start_cmd", "dockerStartCmd"),
            ("min_vcpu_per_gpu", "minVCPUPerGPU"),
            ("min_ram_per_gpu", "minRAMPerGPU"),
        ):
            if source in normalized and normalized[source] not in (None, "", []):
                body[target] = normalized[source]
        return body

    def _validate_exact_b300(self, payload: Mapping[str, Any], *, expected_volume_id: str | None = None) -> dict[str, Any]:
        summary = sanitize_pod(payload)
        if summary.get("gpu_type") != B300_GPU_TYPE:
            raise RunPodError(
                f"RunPod returned GPU SKU {summary.get('gpu_type')!r}; refusing non-{B300_GPU_TYPE} pod"
            )
        if summary.get("gpu_count") != B300_GPU_COUNT:
            raise RunPodError(
                f"RunPod returned gpu_count={summary.get('gpu_count')!r}; refusing anything other than 1"
            )
        if expected_volume_id is not None and summary.get("network_volume_id") not in {None, expected_volume_id}:
            raise RunPodError("RunPod returned a different persistent volume than requested")
        if summary.get("volume_mount_path") not in {None, WORKSPACE_MOUNT_PATH}:
            raise RunPodError("RunPod returned a volume mount different from /workspace")
        return summary

    def create_b300_pod(self, spec: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
        normalized = self._validate_spec(spec, allow_placeholders=dry_run)
        if not dry_run and self.state_path is None:
            raise RunPodError("live pod creation requires a durable controller state path")
        owned = self._owned_pod_id()
        if owned:
            raise RunPodError(
                f"campaign already recorded pod {owned}; refusing a second pod; use a new state path for a new campaign"
            )
        body = self._create_body(normalized, allow_placeholders=dry_run)
        if dry_run:
            return {
                "status": "dry_run",
                "requested_gpu_type": B300_GPU_TYPE,
                "requested_gpu_count": B300_GPU_COUNT,
                "requested_pod_count": 1,
                "volume_mount_path": WORKSPACE_MOUNT_PATH,
                "persistent_volume_id": normalized["network_volume_id"],
                "image": normalized["image_name"],
                "request_fields": sorted(body),
                "credential_env": RUNPOD_API_KEY_ENV,
            }
        response = self._call("POST", "/pods", body=body)
        if not isinstance(response, Mapping):
            raise RunPodError("RunPod create response was not an object")
        summary = self._validate_exact_b300(response, expected_volume_id=str(normalized["network_volume_id"]))
        if not summary.get("pod_id"):
            raise RunPodError("RunPod create response did not include a pod ID")
        self._record_state(status="created", summary=summary, spec=normalized)
        return {"status": "created", **summary}

    def inspect_pod(self, pod_id: str | None = None) -> dict[str, Any]:
        selected = normalize_pod_id(pod_id) if pod_id is not None else self._owned_pod_id()
        if selected is None:
            raise RunPodError("no campaign pod ID is recorded")
        response = self._call("GET", f"/pods/{selected}")
        if not isinstance(response, Mapping):
            raise RunPodError("RunPod inspect response was not an object")
        summary = self._validate_exact_b300(response)
        desired = str(summary.get("lifecycle", {}).get("desired_status") or "").upper()
        actual = str(summary.get("lifecycle", {}).get("actual_status") or "").upper()
        ready = desired == "RUNNING" and actual in {"RUNNING", "READY", "ACTIVE"}
        result = {
            "status": "ready" if ready else "not_ready",
            "pod": summary,
            "runtime": {
                "ready": ready,
                "desired_status": desired or None,
                "actual_status": actual or None,
                "endpoint_present": summary.get("endpoint_present", False),
            },
        }
        self._record_state(status="ready" if ready else "inspected", summary=summary)
        return result

    def stop_pod(self, pod_id: str | None = None) -> dict[str, Any]:
        if self.state_path is None:
            raise RunPodError("stopping a pod requires durable controller state ownership")
        selected = normalize_pod_id(pod_id) if pod_id is not None else self._owned_pod_id()
        if selected is None:
            raise RunPodError("no campaign pod ID is recorded")
        owned = self._owned_pod_id()
        if owned is None:
            raise RunPodError("refusing to stop a pod without a durable campaign-owned pod ID")
        if selected != owned:
            raise RunPodError("refusing to stop a pod outside this campaign's durable controller state")
        response = self._call("POST", f"/pods/{selected}/stop")
        summary = sanitize_pod(response) if isinstance(response, Mapping) else {}
        if summary.get("pod_id") is None:
            summary["pod_id"] = selected
        self._record_state(status="stopped", summary=summary)
        return {"status": "stopped", **summary}


def load_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"controller spec is not valid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError("controller spec must be a JSON object")
    return value


__all__ = [
    "B300_GPU_COUNT",
    "B300_GPU_TYPE",
    "CONTROLLER_SCHEMA_VERSION",
    "MAX_CAMPAIGN_PODS",
    "RUNPOD_API_KEY_ENV",
    "RUNPOD_BASE_URL",
    "RunPodController",
    "RunPodError",
    "RunPodTransport",
    "UrllibRunPodTransport",
    "WORKSPACE_MOUNT_PATH",
    "load_spec",
    "normalize_pod_id",
    "sanitize_pod",
]
