"""Model-independent parallel execution, recovery, and watchdog control.

This module intentionally knows nothing about a scientific benchmark stage.  A
stage is an adapter boundary: either the built-in deterministic fake adapter,
an importable ``module:callable`` adapter, or an argv-only command template.
The executor owns campaign/shard/worker bookkeeping and never combines worker
JSONL files while workers are running.

The durable files written by one campaign are::

    campaign_state.json
    terminal_report.json
    shards/shard_NNN.json
    workers/worker_NNN/output.jsonl
    workers/worker_NNN/worker_manifest.json
    workers/worker_NNN/worker.log

Shard manifests are immutable after creation.  A worker manifest is the
worker's checkpoint and is updated atomically by the built-in/Python worker;
for a direct command adapter it is parent-owned.  Every worker has a distinct
output directory, so no two processes write one JSONL or manifest.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .pod_lifecycle import run_terminal_shutdown

try:  # Worker A's contract is optional during isolated development.
    from .distributed_contracts import (  # type: ignore
        canonical_hash as _distributed_canonical_hash,
        file_sha256 as _distributed_file_sha256,
    )
    from .sharding import (  # type: ignore
        build_shard_plan as _worker_a_build_shard_plan,
        load_inventory as _worker_a_load_inventory,
        write_shard_outputs as _worker_a_write_shard_outputs,
    )
except ImportError:  # pragma: no cover - exercised only before Worker A lands
    _distributed_canonical_hash = None
    _distributed_file_sha256 = None
    _worker_a_build_shard_plan = None
    _worker_a_load_inventory = None
    _worker_a_write_shard_outputs = None


SCHEMA_VERSION = "parallel_executor_v1"
CAMPAIGN_STATE_FILENAME = "campaign_state.json"
TERMINAL_REPORT_FILENAME = "terminal_report.json"
SHARD_MANIFEST_TYPE = "parallel_shard_manifest"
WORKER_MANIFEST_TYPE = "parallel_worker_manifest"
CAMPAIGN_STATE_TYPE = "parallel_campaign_state"
TERMINAL_REPORT_TYPE = "parallel_campaign_terminal_report"

WORKER_STATUSES = frozenset(
    {
        "planned",
        "starting",
        "running",
        "recovering",
        "complete",
        "failed",
        "budget_refused",
        "superseded",
        "dry_run",
    }
)
TERMINAL_WORKER_STATUSES = frozenset({"complete", "failed", "budget_refused", "superseded"})


class ParallelExecutorError(RuntimeError):
    """Base error for invalid execution configuration or durable state."""


class InvalidCheckpointError(ParallelExecutorError, ValueError):
    """Raised when a checkpoint cannot be trusted for resumption."""


class AdapterError(ParallelExecutorError):
    """Raised for an invalid or unusable stage adapter."""


class WorkerAContractError(ParallelExecutorError):
    """Raised when Worker A rejects an inventory or its physical shard plan."""


_WORKER_A_METADATA_FIELDS = frozenset(
    {
        "construct",
        "construct_id",
        "construct_ids",
        "construct_name",
        "pair",
        "pair_id",
        "pair_role",
        "pair_unit_id",
        "paired_id",
        "episode_id",
        "matched_episode_id",
        "unit",
        "unit_id",
        "work_unit_id",
        "batch_unit_id",
        "factor_cell",
        "factor_cell_id",
        "factor_cell_key",
        "cell",
        "cell_id",
        "generation_cell_id",
        "prompt_role",
        "request_role",
        "stage_role",
        "role",
        "condition",
        "condition_id",
        "condition_label",
        "prompt_version",
        "inventory_version",
        "version",
        "variant",
    }
)


def _inventory_is_construct_aware(requests: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether generic sharding could discard construct invariants.

    A request-only fixture is intentionally allowed to use the executor's
    schema-neutral fallback.  Any explicit construct, pair/cell/unit, role,
    or version metadata is treated as construct-aware and must pass Worker A's
    validation whenever that contract is available.
    """

    for request in requests:
        for key, value in request.items():
            normalized_key = str(key).strip().casefold().replace("-", "_")
            if normalized_key not in _WORKER_A_METADATA_FIELDS:
                continue
            if _as_string_list(value):
                return True
    return False


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 of canonical JSON data."""

    if _distributed_canonical_hash is not None:
        return _distributed_canonical_hash(value)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    if _distributed_file_sha256 is not None:
        return _distributed_file_sha256(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch_to_iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now(epoch: float | None = None) -> str:
    return _epoch_to_iso(time.time() if epoch is None else epoch) or ""


def _iso_to_epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        normalized = value.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON using a same-directory temporary file and ``os.replace``."""

    destination = Path(path)
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


def _load_json(path: str | Path, *, label: str) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InvalidCheckpointError(f"{label} does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise InvalidCheckpointError(f"{label} is malformed JSON: {source}") from exc


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    value = _load_json(path, label=label)
    if not isinstance(value, dict):
        raise InvalidCheckpointError(f"{label} must contain a JSON object: {path}")
    return dict(value)


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidCheckpointError(f"{field} must be a non-empty string.")
    return value.strip()


def _unique_strings(value: Any, *, field: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise InvalidCheckpointError(f"{field} must be a list of strings.")
    result = []
    for item in value:
        result.append(_nonempty_string(item, field=field))
    if not allow_empty and not result:
        raise InvalidCheckpointError(f"{field} must not be empty.")
    if len(set(result)) != len(result):
        raise InvalidCheckpointError(f"{field} must not contain duplicates.")
    return result


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    if isinstance(value, (tuple, list, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _request_id_from_mapping(record: Mapping[str, Any], index: int) -> str:
    for key in ("request_id", "prompt_id", "job_id", "id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"request_{index:06d}"


def normalize_inventory_requests(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize inventory rows to a stable request/observation shape.

    The executor is deliberately agnostic to prompt schemas.  It only needs a
    unique request ID, a unique set of observation IDs, and optional construct
    metadata.  Missing observation IDs mean one observation with the request
    ID, which makes simple JSON/CSV fixtures convenient.
    """

    normalized: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    seen_observations: set[str] = set()
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Inventory record {index} must be an object.")
        record = dict(raw)
        request_id = _request_id_from_mapping(record, index)
        if request_id in seen_requests:
            raise ValueError(f"Inventory contains duplicate request_id {request_id!r}.")
        raw_observations = record.get("observation_ids", record.get("expected_observation_ids"))
        if raw_observations is None:
            raw_observations = record.get("observation_id")
        observation_ids = _as_string_list(raw_observations)
        if not observation_ids:
            observation_ids = [request_id]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError(f"Inventory request {request_id!r} has duplicate observation IDs.")
        overlap = seen_observations.intersection(observation_ids)
        if overlap:
            raise ValueError(f"Inventory observation IDs are not globally unique: {sorted(overlap)!r}.")
        construct_values = record.get("construct_ids", record.get("construct_id"))
        construct_ids = _as_string_list(construct_values)
        normalized_record = dict(record)
        normalized_record["request_id"] = request_id
        normalized_record["observation_ids"] = observation_ids
        normalized_record["construct_ids"] = sorted(set(construct_ids))
        normalized.append(normalized_record)
        seen_requests.add(request_id)
        seen_observations.update(observation_ids)
    if not normalized:
        raise ValueError("Inventory must contain at least one request.")
    return normalized


def _inventory_records_from_json(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        raise ValueError("Inventory JSON must contain a list or object.")
    for key in ("requests", "records", "items", "prompts"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return candidate
    request_ids = value.get("request_ids")
    if isinstance(request_ids, list):
        return [{"request_id": item} for item in request_ids]
    if any(key in value for key in ("request_id", "prompt_id", "id")):
        return [value]
    raise ValueError("Inventory JSON has no requests/records/items/request_ids list.")


def load_inventory_requests(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON, JSONL, or CSV inventory inputs without scientific assumptions."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Inventory does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Inventory JSONL line {line_number} is malformed.") from exc
            rows.append(item)
        return normalize_inventory_requests(rows)
    if suffix == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            return normalize_inventory_requests(csv.DictReader(handle))
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Inventory is not valid JSON/CSV: {source}") from exc
    return normalize_inventory_requests(_inventory_records_from_json(payload))


def _request_observation_map(requests: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    return {str(request["request_id"]): [str(item) for item in request["observation_ids"]] for request in requests}


def _expected_observations(requests: Sequence[Mapping[str, Any]]) -> list[str]:
    return [observation for request in requests for observation in request["observation_ids"]]


def _shard_manifest_payload(
    *,
    shard_id: str,
    requests: Sequence[Mapping[str, Any]],
    parent_inventory_sha256: str,
    run_config_hash: str,
    run_mode: str,
    confirmatory: bool,
    stage: str,
    campaign_identity: str,
    worker_id: str,
    engineering_only: bool = False,
) -> dict[str, Any]:
    normalized = normalize_inventory_requests(requests) if requests else []
    request_ids = [str(request["request_id"]) for request in normalized]
    observation_ids = _expected_observations(normalized)
    construct_ids = sorted({str(value) for request in normalized for value in request.get("construct_ids", [])})
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": SHARD_MANIFEST_TYPE,
        "immutable": True,
        "shard_id": shard_id,
        "worker_id": worker_id,
        "parent_inventory_sha256": parent_inventory_sha256,
        "request_ids": request_ids,
        "expected_request_count": len(request_ids),
        "expected_observation_ids": observation_ids,
        "expected_observation_count": len(observation_ids),
        "construct_ids": construct_ids,
        "run_config_hash": run_config_hash,
        "run_mode": run_mode,
        "confirmatory": bool(confirmatory),
        "engineering_only": bool(engineering_only),
        "stage": stage,
        "campaign_identity": campaign_identity,
        "requests": normalized,
    }


def validate_shard_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the stable shard contract and return a detached mapping."""

    value = dict(manifest)
    if value.get("manifest_type") in {"benchmark_shard", "benchmark_shard_manifest"}:
        return _adapt_worker_a_shard_manifest(value, expected_identity=expected_identity)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("manifest_type") != SHARD_MANIFEST_TYPE:
        raise InvalidCheckpointError("Shard manifest has an unsupported schema or manifest_type.")
    if value.get("immutable") is not True:
        raise InvalidCheckpointError("Shard manifest must declare immutable=true.")
    for field in ("shard_id", "parent_inventory_sha256", "run_config_hash", "run_mode", "stage", "campaign_identity"):
        _nonempty_string(value.get(field), field=field)
    _nonempty_string(value.get("worker_id"), field="worker_id")
    if not isinstance(value.get("confirmatory"), bool):
        raise InvalidCheckpointError("Shard manifest confirmatory must be a boolean.")
    engineering_only = value.get("engineering_only", False)
    if not isinstance(engineering_only, bool):
        raise InvalidCheckpointError("Shard manifest engineering_only must be a boolean.")
    if value.get("run_mode") == "full" and not value["confirmatory"] and not engineering_only:
        raise InvalidCheckpointError(
            "Non-confirmatory full shard manifests must declare engineering_only=true."
        )
    if value.get("run_mode") != "full" and engineering_only:
        raise InvalidCheckpointError("engineering_only is only valid for full shard manifests.")
    request_ids = _unique_strings(value.get("request_ids"), field="request_ids")
    observation_ids = _unique_strings(value.get("expected_observation_ids"), field="expected_observation_ids")
    construct_ids = _unique_strings(value.get("construct_ids"), field="construct_ids")
    if value.get("expected_request_count") != len(request_ids):
        raise InvalidCheckpointError("Shard expected_request_count is inconsistent with request_ids.")
    if value.get("expected_observation_count") != len(observation_ids):
        raise InvalidCheckpointError("Shard expected_observation_count is inconsistent with expected_observation_ids.")
    requests = value.get("requests")
    if requests is None:
        requests = [{"request_id": request_id, "observation_ids": [request_id], "construct_ids": []} for request_id in request_ids]
        value["requests"] = requests
    if not isinstance(requests, list):
        raise InvalidCheckpointError("Shard requests must be a list.")
    normalized = normalize_inventory_requests(requests) if requests else []
    if [request["request_id"] for request in normalized] != request_ids:
        raise InvalidCheckpointError("Shard request_ids do not match requests.")
    if _expected_observations(normalized) != observation_ids:
        raise InvalidCheckpointError("Shard expected observations do not match requests.")
    request_constructs = sorted({str(item) for request in normalized for item in request.get("construct_ids", [])})
    if request_constructs != construct_ids:
        raise InvalidCheckpointError("Shard construct_ids do not match requests.")
    identity = {
        "campaign_identity": value["campaign_identity"],
        "parent_inventory_sha256": value["parent_inventory_sha256"],
        "run_config_hash": value["run_config_hash"],
        "run_mode": value["run_mode"],
        "confirmatory": value["confirmatory"],
        "engineering_only": engineering_only,
        "stage": value["stage"],
    }
    if expected_identity is not None:
        for field, expected in expected_identity.items():
            if field in identity and identity[field] != expected:
                raise InvalidCheckpointError(
                    f"Shard manifest identity mismatch for {field}: {identity[field]!r} != {expected!r}."
                )
    value["request_ids"] = request_ids
    value["expected_observation_ids"] = observation_ids
    value["construct_ids"] = construct_ids
    value["requests"] = normalized
    return value


def _adapt_worker_a_shard_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project Worker A's ``benchmark_shard`` into the executor shape.

    Worker A owns the physical sharding contract.  This adapter does not
    rewrite or replace that manifest; it only exposes the common fields the
    executor needs.  When the physical shard inventory is available, its
    records preserve request-to-observation ownership exactly.  The embedded
    ``expected_observations`` list is sufficient for a detached manifest.
    """

    raw = dict(manifest)
    if raw.get("manifest_type") not in {"benchmark_shard", "benchmark_shard_manifest"}:
        return validate_shard_manifest(raw, expected_identity=expected_identity)
    request_ids = raw.get("owned_request_ids", raw.get("request_ids"))
    request_ids = _unique_strings(request_ids, field="owned_request_ids")
    observations = raw.get("expected_observation_ids", raw.get("owned_observation_ids"))
    observations = _unique_strings(observations, field="expected_observation_ids")
    observation_to_request: dict[str, str] = {}
    expected_rows = raw.get("expected_observations")
    if isinstance(expected_rows, list):
        for item in expected_rows:
            if isinstance(item, Mapping) and item.get("observation_id") is not None and item.get("request_id") is not None:
                observation_to_request[str(item["observation_id"])] = str(item["request_id"])

    requests: list[dict[str, Any]] = []
    shard_inventory_value = raw.get("shard_inventory_path", raw.get("shard_path"))
    shard_inventory_path: Path | None = None
    if shard_inventory_value and manifest_path is not None:
        candidate = Path(str(shard_inventory_value))
        if not candidate.is_absolute():
            candidate = Path(manifest_path).resolve().parent / candidate
        shard_inventory_path = candidate.resolve()
    if shard_inventory_path is not None and shard_inventory_path.is_file() and _worker_a_load_inventory is not None:
        try:
            records = _worker_a_load_inventory(shard_inventory_path)
        except Exception as exc:
            raise InvalidCheckpointError(
                f"Worker A shard inventory cannot be loaded for {manifest_path}: {exc}"
            ) from exc
        by_id = {record.request_id: record for record in records}
        for request_id in request_ids:
            record = by_id.get(request_id)
            if record is None:
                raise InvalidCheckpointError(
                    f"Worker A shard {raw.get('shard_id')!r} is missing request {request_id!r} in its shard inventory."
                )
            requests.append(
                {
                    **dict(record.raw),
                    "request_id": record.request_id,
                    "observation_ids": list(record.observation_ids),
                    "construct_ids": [record.construct_id],
                }
            )
    else:
        by_request: dict[str, list[str]] = {request_id: [] for request_id in request_ids}
        for observation_id in observations:
            request_id = observation_to_request.get(observation_id)
            if request_id is None:
                if len(request_ids) == len(observations):
                    request_id = request_ids[observations.index(observation_id)]
                else:
                    raise InvalidCheckpointError(
                        f"Worker A shard {raw.get('shard_id')!r} has no request owner for observation {observation_id!r}."
                    )
            by_request.setdefault(request_id, []).append(observation_id)
        for request_id in request_ids:
            requests.append(
                {
                    "request_id": request_id,
                    "observation_ids": by_request.get(request_id) or [request_id],
                    "construct_ids": list(raw.get("construct_ids", [])),
                }
            )
    requests.sort(key=lambda item: str(item["request_id"]))
    identity = dict(expected_identity or {})
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": SHARD_MANIFEST_TYPE,
        "immutable": True,
        "shard_id": raw.get("shard_id"),
        "worker_id": raw.get("worker_id", raw.get("worker_slot", "worker_000")),
        "parent_inventory_sha256": raw.get("parent_inventory_sha256"),
        "request_ids": [request["request_id"] for request in requests],
        "expected_request_count": len(requests),
        "expected_observation_ids": [observation for request in requests for observation in request["observation_ids"]],
        "expected_observation_count": sum(len(request["observation_ids"]) for request in requests),
        "construct_ids": sorted({str(item) for request in requests for item in request.get("construct_ids", [])}),
        "run_config_hash": raw.get("run_config_hash", identity.get("run_config_hash", "UNSPECIFIED")),
        "run_mode": raw.get("run_mode", identity.get("run_mode", "test")),
        "confirmatory": raw.get("confirmatory", identity.get("confirmatory", False)),
        "engineering_only": raw.get("engineering_only", identity.get("engineering_only", False)),
        "stage": raw.get("stage", identity.get("stage", "benchmark")),
        "campaign_identity": raw.get("campaign_identity", identity.get("campaign_identity", "external")),
        "requests": requests,
        "external_manifest_type": raw.get("manifest_type"),
        "external_manifest": raw,
    }
    # A manifests may carry the observation list in a different order.  The
    # canonical adapter uses request order and validates the resulting sets.
    return validate_shard_manifest(canonical, expected_identity=expected_identity)


def load_shard_manifest(path: str | Path, *, expected_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path)
    raw = _load_json_object(source, label="shard manifest")
    if raw.get("manifest_type") in {"benchmark_shard", "benchmark_shard_manifest"}:
        return _adapt_worker_a_shard_manifest(raw, manifest_path=source, expected_identity=expected_identity)
    return validate_shard_manifest(raw, expected_identity=expected_identity)


def build_shard_manifests(
    requests: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    worker_count: int,
    parent_inventory_sha256: str,
    run_config_hash: str,
    run_mode: str,
    confirmatory: bool,
    stage: str,
    campaign_identity: str,
    start_index: int = 0,
    engineering_only: bool = False,
) -> list[dict[str, Any]]:
    """Create deterministic, immutable shard manifests for worker slots."""

    if not 1 <= int(worker_count) <= 5:
        raise ValueError("worker_count must be between 1 and 5.")
    if not isinstance(engineering_only, bool):
        raise ValueError("engineering_only must be a boolean.")
    if run_mode == "full" and not confirmatory and not engineering_only:
        raise ValueError(
            "full run_mode must be confirmatory unless engineering_only=true is explicitly supplied."
        )
    if run_mode != "full" and engineering_only:
        raise ValueError("engineering_only is only valid for the full run_mode.")
    normalized = normalize_inventory_requests(requests) if requests else []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # Contiguous balanced partitioning keeps neighboring records together and
    # is deterministic across processes and platforms.
    count = int(worker_count)
    base, remainder = divmod(len(normalized), count)
    shards: list[dict[str, Any]] = []
    cursor = 0
    for offset in range(count):
        size = base + (1 if offset < remainder else 0)
        part = normalized[cursor : cursor + size]
        cursor += size
        slot = start_index + offset
        worker_id = f"worker_{slot:03d}"
        shard_id = f"shard_{slot:03d}"
        payload = _shard_manifest_payload(
            shard_id=shard_id,
            worker_id=worker_id,
            requests=part,
            parent_inventory_sha256=parent_inventory_sha256,
            run_config_hash=run_config_hash,
            run_mode=run_mode,
            confirmatory=confirmatory,
            engineering_only=engineering_only,
            stage=stage,
            campaign_identity=campaign_identity,
        )
        path = destination / f"{shard_id}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to replace immutable shard manifest: {path}")
        atomic_write_json(path, payload)
        payload["manifest_path"] = str(path)
        payload["manifest_sha256"] = file_sha256(path)
        shards.append(payload)
    return shards


def _worker_payload(
    *,
    worker_id: str,
    shard: Mapping[str, Any],
    status: str,
    output_path: str | Path,
    worker_manifest_path: str | Path,
    stage: str,
    retry_count: int,
    execution_identity: Mapping[str, Any],
    now: float | None = None,
    completed_request_ids: Iterable[str] = (),
    completed_observation_ids: Iterable[str] = (),
    error: str | None = None,
    terminal_reason: str | None = None,
    pid: int | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else float(now)
    completed_requests = sorted({str(item) for item in completed_request_ids})
    completed_observations = sorted({str(item) for item in completed_observation_ids})
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": WORKER_MANIFEST_TYPE,
        "worker_id": worker_id,
        "shard_id": shard["shard_id"],
        "status": status,
        "pid": pid,
        "stage": stage,
        "heartbeat_at": _utc_now(timestamp) if status not in {"planned", "dry_run"} else None,
        "heartbeat_at_epoch": timestamp if status not in {"planned", "dry_run"} else None,
        "expected_request_ids": list(shard["request_ids"]),
        "expected_request_count": int(shard["expected_request_count"]),
        "completed_request_ids": completed_requests,
        "completed_request_count": len(completed_requests),
        "expected_observation_ids": list(shard["expected_observation_ids"]),
        "expected_observation_count": int(shard["expected_observation_count"]),
        "completed_observation_ids": completed_observations,
        "completed_observation_count": len(completed_observations),
        "retry_count": int(retry_count),
        "last_progress": {
            "at": _utc_now(timestamp) if status not in {"planned", "dry_run"} else None,
            "request_count": len(completed_requests),
            "observation_count": len(completed_observations),
        },
        "output_path": str(output_path),
        "checkpoint_path": str(worker_manifest_path),
        "error": error,
        "terminal_reason": terminal_reason,
        "execution_identity": dict(execution_identity),
    }


def validate_worker_manifest(
    manifest: Mapping[str, Any],
    *,
    shard: Mapping[str, Any],
    expected_identity: Mapping[str, Any] | None = None,
    expected_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a worker checkpoint without treating null stage outputs as errors."""

    value = dict(manifest)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("manifest_type") != WORKER_MANIFEST_TYPE:
        raise InvalidCheckpointError("Worker manifest has an unsupported schema or manifest_type.")
    for field in ("worker_id", "shard_id", "stage"):
        _nonempty_string(value.get(field), field=field)
    if value["worker_id"] != shard.get("worker_id") or value["shard_id"] != shard.get("shard_id"):
        raise InvalidCheckpointError("Worker manifest does not match its shard.")
    status = value.get("status")
    if status not in WORKER_STATUSES:
        raise InvalidCheckpointError(f"Worker manifest has unknown status {status!r}.")
    if value.get("pid") is not None and (not isinstance(value["pid"], int) or isinstance(value["pid"], bool)):
        raise InvalidCheckpointError("Worker manifest pid must be an integer or null.")
    for field in ("expected_request_ids", "completed_request_ids", "expected_observation_ids", "completed_observation_ids"):
        _unique_strings(value.get(field), field=field)
    if value["expected_request_ids"] != list(shard["request_ids"]):
        raise InvalidCheckpointError("Worker expected_request_ids do not match its immutable shard.")
    if value["expected_observation_ids"] != list(shard["expected_observation_ids"]):
        raise InvalidCheckpointError("Worker expected_observation_ids do not match its immutable shard.")
    for field, ids in (
        ("completed_request_ids", value["completed_request_ids"]),
        ("completed_observation_ids", value["completed_observation_ids"]),
    ):
        expected = set(value[field.replace("completed_", "expected_")])
        if not set(ids).issubset(expected):
            raise InvalidCheckpointError(f"{field} contains IDs outside the immutable shard.")
    if value.get("expected_request_count") != len(value["expected_request_ids"]):
        raise InvalidCheckpointError("Worker expected_request_count is inconsistent.")
    if value.get("expected_observation_count") != len(value["expected_observation_ids"]):
        raise InvalidCheckpointError("Worker expected_observation_count is inconsistent.")
    if value.get("completed_request_count") != len(value["completed_request_ids"]):
        raise InvalidCheckpointError("Worker completed_request_count is inconsistent.")
    if value.get("completed_observation_count") != len(value["completed_observation_ids"]):
        raise InvalidCheckpointError("Worker completed_observation_count is inconsistent.")
    output_path = value.get("output_path")
    if output_path is not None and not isinstance(output_path, str):
        raise InvalidCheckpointError("Worker output_path must be a string or null.")
    if expected_output_path is not None and output_path is not None:
        if Path(output_path).resolve() != Path(expected_output_path).resolve():
            raise InvalidCheckpointError("Worker output_path does not match the campaign output path.")
    identity = value.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise InvalidCheckpointError("Worker execution_identity must be an object.")
    if expected_identity is not None:
        for field, expected in expected_identity.items():
            if identity.get(field) != expected:
                raise InvalidCheckpointError(
                    f"Worker execution identity mismatch for {field}: {identity.get(field)!r} != {expected!r}."
                )
    value["completed_request_ids"] = list(value["completed_request_ids"])
    value["completed_observation_ids"] = list(value["completed_observation_ids"])
    return value


def load_worker_manifest(
    path: str | Path,
    *,
    shard: Mapping[str, Any],
    expected_identity: Mapping[str, Any] | None = None,
    expected_output_path: str | Path | None = None,
) -> dict[str, Any]:
    return validate_worker_manifest(
        _load_json_object(path, label="worker manifest"),
        shard=shard,
        expected_identity=expected_identity,
        expected_output_path=expected_output_path,
    )


@dataclass(frozen=True)
class OutputProgress:
    completed_request_ids: frozenset[str]
    completed_observation_ids: frozenset[str]
    byte_size: int
    row_count: int


def read_output_progress(path: str | Path, shard: Mapping[str, Any]) -> OutputProgress:
    """Read worker JSONL progress and validate IDs.

    The output field itself is intentionally unconstrained: ``{"output":
    null}`` is a valid completed observation and is never treated as a
    malformed output.
    """

    if path is None:
        return OutputProgress(frozenset(), frozenset(), 0, 0)
    output = Path(path)
    if not output.exists():
        return OutputProgress(frozenset(), frozenset(), 0, 0)
    expected_requests = set(shard["request_ids"])
    expected_observations = set(shard["expected_observation_ids"])
    request_observations = _request_observation_map(shard["requests"])
    seen_requests: set[str] = set()
    seen_observations: set[str] = set()
    row_count = 0
    try:
        handle = output.open("r", encoding="utf-8")
    except OSError as exc:
        raise InvalidCheckpointError(f"Cannot read worker output {output}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InvalidCheckpointError(f"Worker output {output} line {line_number} is malformed JSON.") from exc
            if not isinstance(row, Mapping):
                raise InvalidCheckpointError(f"Worker output {output} line {line_number} must be a JSON object.")
            request_value = row.get("request_id", row.get("job_id"))
            if request_value is None or not str(request_value).strip():
                raise InvalidCheckpointError(f"Worker output {output} line {line_number} has no request_id.")
            request_id = str(request_value)
            if request_id not in expected_requests:
                raise InvalidCheckpointError(f"Worker output contains unexpected request_id {request_id!r}.")
            raw_observations = row.get("observation_ids")
            if raw_observations is None:
                raw_observations = row.get("observation_id")
            observation_ids = _as_string_list(raw_observations)
            if not observation_ids:
                observation_ids = list(request_observations[request_id])
            for observation_id in observation_ids:
                if observation_id not in expected_observations:
                    raise InvalidCheckpointError(
                        f"Worker output contains unexpected observation_id {observation_id!r}."
                    )
                if observation_id in seen_observations:
                    raise InvalidCheckpointError(f"Worker output duplicates observation_id {observation_id!r}.")
                if request_id not in request_observations or observation_id not in request_observations[request_id]:
                    raise InvalidCheckpointError(
                        f"Worker output observation {observation_id!r} is assigned to request {request_id!r} incorrectly."
                    )
                seen_observations.add(observation_id)
            seen_requests.add(request_id)
            row_count += 1
    completed_requests = {
        request_id
        for request_id, observations in request_observations.items()
        if set(observations).issubset(seen_observations)
    }
    return OutputProgress(
        frozenset(completed_requests),
        frozenset(seen_observations),
        output.stat().st_size,
        row_count,
    )


def _write_jsonl_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_worker_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def _load_python_callable(spec: str) -> Callable[..., Any]:
    if spec.startswith("python:"):
        spec = spec[len("python:") :]
    if ":" not in spec:
        raise AdapterError("Python adapter must use module:callable syntax.")
    module_name, callable_name = spec.split(":", 1)
    if not module_name or not callable_name:
        raise AdapterError("Python adapter must use non-empty module:callable names.")
    try:
        value: Any = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - import errors are surfaced to the worker log
        raise AdapterError(f"Cannot import Python adapter module {module_name!r}: {exc}") from exc
    for part in callable_name.split("."):
        value = getattr(value, part, None)
    if not callable(value):
        raise AdapterError(f"Python adapter {spec!r} does not resolve to a callable.")
    return value


def _call_python_adapter(adapter: Callable[..., Any], request: Mapping[str, Any], context: Mapping[str, Any]) -> Any:
    """Invoke the documented two-argument adapter contract.

    A narrow fallback for one-argument callables is useful for small stage
    fixtures and does not change the command boundary.  We inspect the
    signature first so a TypeError raised *inside* a two-argument adapter is
    not accidentally retried with a different contract.
    """

    try:
        signature = inspect.signature(adapter)
    except (TypeError, ValueError):
        return adapter(request, context)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
    if has_varargs or len(positional) >= 2:
        return adapter(request, context)
    if len(positional) == 1:
        return adapter(request)
    return adapter()


def _result_rows(
    result: Any,
    *,
    request: Mapping[str, Any],
    remaining_observations: Sequence[str],
    worker_id: str,
    stage: str,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Normalize adapter results while allowing null output values."""

    request_id = str(request["request_id"])
    if result is None:
        values: list[Any] = [{"output": None} for _ in remaining_observations]
    elif isinstance(result, Mapping):
        result_mapping = dict(result)
        declared = _as_string_list(result_mapping.get("observation_ids", result_mapping.get("observation_id")))
        if declared:
            values = [result_mapping]
        else:
            values = [dict(result_mapping) for _ in remaining_observations]
    elif isinstance(result, (list, tuple)):
        values = list(result)
    else:
        values = [{"output": result} for _ in remaining_observations]
    rows: list[dict[str, Any]] = []
    next_observation_index = 0
    for raw in values:
        row = dict(raw) if isinstance(raw, Mapping) else {"output": raw}
        declared_ids = _as_string_list(row.get("observation_ids", row.get("observation_id")))
        if declared_ids:
            row_observations = declared_ids
        else:
            if next_observation_index >= len(remaining_observations):
                continue
            row_observations = [remaining_observations[next_observation_index]]
            next_observation_index += 1
        for observation_id in row_observations:
            if observation_id not in remaining_observations:
                raise AdapterError(
                    f"Adapter returned observation {observation_id!r} outside the remaining request scope."
                )
            normalized = dict(row)
            normalized.pop("observation_ids", None)
            normalized["request_id"] = request_id
            normalized["observation_id"] = observation_id
            normalized["worker_id"] = worker_id
            normalized["stage"] = stage
            normalized["adapter_context"] = {"fake_model": bool(context.get("fake_model", False))}
            rows.append(normalized)
    if not rows:
        rows = [
            {
                "request_id": request_id,
                "observation_id": observation_id,
                "worker_id": worker_id,
                "stage": stage,
                "output": None,
                "adapter_context": {"fake_model": bool(context.get("fake_model", False))},
            }
            for observation_id in remaining_observations
        ]
    return rows


def run_worker_process(
    *,
    shard_manifest_path: str | Path,
    worker_manifest_path: str | Path,
    output_path: str | Path,
    stage: str,
    adapter_spec: str = "fake",
    fake_model: bool = False,
    stall_seconds: float = 0.0,
    stall_after: int | None = None,
    crash_after: int | None = None,
    crash_once: bool = False,
) -> int:
    """Worker subprocess entrypoint used by the executor.

    The adapter receives ``(request, context)`` and may return a mapping,
    sequence of mappings, a scalar, or ``None``.  ``None`` is recorded as a
    valid null output for every remaining observation.
    """

    shard_path = Path(shard_manifest_path)
    manifest_path = Path(worker_manifest_path)
    output = Path(output_path)
    try:
        shard = load_shard_manifest(shard_path)
        manifest = load_worker_manifest(manifest_path, shard=shard, expected_output_path=output)
        worker_id = str(manifest["worker_id"])
        if manifest["stage"] != stage:
            raise InvalidCheckpointError("Worker stage does not match the requested stage.")
        if adapter_spec == "fake":
            def adapter(request: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
                return {
                    "output": {
                        "fake": True,
                        "request_id": request["request_id"],
                        "stage": context["stage"],
                    }
                }
        else:
            adapter = _load_python_callable(adapter_spec)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch(exist_ok=True)
        progress = read_output_progress(output, shard)
        if not set(manifest["completed_observation_ids"]).issubset(progress.completed_observation_ids):
            raise InvalidCheckpointError("Worker checkpoint claims observations absent from worker output.")
        completed_observations = set(progress.completed_observation_ids)
        completed_requests = set(progress.completed_request_ids)

        def heartbeat(status: str = "running", *, error: str | None = None) -> None:
            current = _worker_payload(
                worker_id=worker_id,
                shard=shard,
                status=status,
                output_path=output,
                worker_manifest_path=manifest_path,
                stage=stage,
                retry_count=int(manifest["retry_count"]),
                execution_identity=manifest["execution_identity"],
                now=time.time(),
                completed_request_ids=completed_requests,
                completed_observation_ids=completed_observations,
                error=error,
                terminal_reason=None,
                pid=os.getpid(),
            )
            _write_worker_manifest(manifest_path, current)

        heartbeat()
        if stall_seconds > 0 and (stall_after is None or stall_after <= 0):
            time.sleep(float(stall_seconds))
            heartbeat()
        processed_this_attempt = 0
        request_map = {str(request["request_id"]): request for request in shard["requests"]}
        for request_id in shard["request_ids"]:
            if request_id in completed_requests:
                continue
            if stall_seconds > 0 and stall_after is not None and processed_this_attempt >= int(stall_after):
                time.sleep(float(stall_seconds))
                heartbeat()
            request = request_map[request_id]
            remaining = [
                observation_id
                for observation_id in request["observation_ids"]
                if observation_id not in completed_observations
            ]
            if not remaining:
                completed_requests.add(request_id)
                continue
            heartbeat()
            context = {
                "worker_id": worker_id,
                "stage": stage,
                "shard_manifest_path": str(shard_path),
                "worker_manifest_path": str(manifest_path),
                "output_path": str(output),
                "fake_model": bool(fake_model),
                "execution_identity": manifest["execution_identity"],
            }
            result = _call_python_adapter(adapter, request, context)
            rows = _result_rows(
                result,
                request=request,
                remaining_observations=remaining,
                worker_id=worker_id,
                stage=stage,
                context=context,
            )
            for row in rows:
                observation_id = str(row["observation_id"])
                if observation_id in completed_observations:
                    continue
                _write_jsonl_row(output, row)
                completed_observations.add(observation_id)
            completed_requests.update(
                request_key
                for request_key, request_value in request_map.items()
                if set(request_value["observation_ids"]).issubset(completed_observations)
            )
            processed_this_attempt += 1
            if crash_after is not None and processed_this_attempt >= int(crash_after):
                marker = output.with_name(f".{output.name}.crash-once")
                should_crash = not crash_once or not marker.exists()
                if crash_once:
                    marker.write_text("crash injected\n", encoding="utf-8")
                heartbeat(error="injected crash" if should_crash else None)
                if should_crash:
                    return 17
            heartbeat()
        final = _worker_payload(
            worker_id=worker_id,
            shard=shard,
            status="complete",
            output_path=output,
            worker_manifest_path=manifest_path,
            stage=stage,
            retry_count=int(manifest["retry_count"]),
            execution_identity=manifest["execution_identity"],
            now=time.time(),
            completed_request_ids=completed_requests,
            completed_observation_ids=completed_observations,
        )
        _write_worker_manifest(manifest_path, final)
        return 0
    except Exception as exc:
        try:
            shard = load_shard_manifest(shard_path)
            previous = _load_json_object(manifest_path, label="worker manifest")
            retry_count = int(previous.get("retry_count", 0))
            identity = previous.get("execution_identity", {})
            progress = read_output_progress(output, shard)
            failed = _worker_payload(
                worker_id=str(previous.get("worker_id", shard.get("worker_id", "worker"))),
                shard=shard,
                status="failed",
                output_path=output,
                worker_manifest_path=manifest_path,
                stage=stage,
                retry_count=retry_count,
                execution_identity=identity if isinstance(identity, Mapping) else {},
                now=time.time(),
                completed_request_ids=progress.completed_request_ids,
                completed_observation_ids=progress.completed_observation_ids,
                error=str(exc),
                terminal_reason="worker_exception",
                pid=os.getpid(),
            )
            _write_worker_manifest(manifest_path, failed)
        except Exception:
            pass
        print(f"parallel worker failed: {exc}", file=sys.stderr)
        return 1


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> Any: ...


@dataclass
class WorkerRuntime:
    worker_id: str
    shard: dict[str, Any]
    entry: dict[str, Any]
    process: Any
    manifest_owned_by_worker: bool
    last_output_signature: tuple[int, int, int]
    last_progress_clock: float
    started_clock: float
    log_handle: Any = None


@dataclass
class BudgetGovernor:
    hard_ceiling_usd: float | None
    reserve_usd: float
    gpu_hourly_rate_usd: float
    worker_estimate_seconds: float
    clock: Callable[[], float]
    actual_spend_usd: float = 0.0
    last_update_clock: float | None = None
    launch_refusals: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.hard_ceiling_usd is not None and self.hard_ceiling_usd < 0:
            raise ValueError("hard budget ceiling must be non-negative.")
        if self.reserve_usd < 0:
            raise ValueError("budget reserve must be non-negative.")
        if self.gpu_hourly_rate_usd < 0:
            raise ValueError("GPU hourly rate must be non-negative.")
        if self.worker_estimate_seconds < 0:
            raise ValueError("worker estimate seconds must be non-negative.")
        if self.launch_refusals is None:
            self.launch_refusals = []
        if self.last_update_clock is None:
            self.last_update_clock = float(self.clock())

    @property
    def estimate_cost_per_worker_usd(self) -> float:
        return self.worker_estimate_seconds * self.gpu_hourly_rate_usd / 3600.0

    def update(self, active_workers: int) -> None:
        now = float(self.clock())
        previous = float(self.last_update_clock if self.last_update_clock is not None else now)
        elapsed = max(0.0, now - previous)
        self.actual_spend_usd += elapsed * max(0, int(active_workers)) * self.gpu_hourly_rate_usd / 3600.0
        self.last_update_clock = now

    def can_launch(self, active_workers: int) -> tuple[bool, str]:
        self.update(active_workers)
        if self.hard_ceiling_usd is None:
            return True, "no hard ceiling configured"
        projected = self.actual_spend_usd + self.estimate_cost_per_worker_usd
        if projected + self.reserve_usd > self.hard_ceiling_usd + 1e-12:
            return False, (
                f"launch estimate ${projected:.6f} plus reserve ${self.reserve_usd:.6f} "
                f"exceeds hard ceiling ${self.hard_ceiling_usd:.6f}"
            )
        return True, "within budget"

    def snapshot(self, active_workers: int) -> dict[str, Any]:
        self.update(active_workers)
        remaining = None
        projected = None
        if self.hard_ceiling_usd is not None:
            remaining = max(0.0, self.hard_ceiling_usd - self.reserve_usd - self.actual_spend_usd)
            projected = self.actual_spend_usd + max(0, int(active_workers)) * self.estimate_cost_per_worker_usd
        return {
            "hard_ceiling_usd": self.hard_ceiling_usd,
            "reserve_usd": self.reserve_usd,
            "gpu_hourly_rate_usd": self.gpu_hourly_rate_usd,
            "worker_estimate_seconds": self.worker_estimate_seconds,
            "actual_spend_usd": self.actual_spend_usd,
            "estimated_spend_usd": projected,
            "remaining_usd_after_reserve": remaining,
            "launch_refusals": list(self.launch_refusals or []),
        }


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_process(process: Any, *, wait_seconds: float = 0.2) -> None:
    try:
        process.terminate()
    except (AttributeError, OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=wait_seconds)
        return
    except (AttributeError, subprocess.TimeoutExpired, OSError):
        pass
    try:
        process.kill()
    except (AttributeError, OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=wait_seconds)
    except (AttributeError, subprocess.TimeoutExpired, OSError):
        pass


def _argv_with_tokens(command: Sequence[str], tokens: Mapping[str, str]) -> list[str]:
    """Replace fixed placeholders without invoking a shell or evaluating text."""

    result = []
    for argument in command:
        value = str(argument)
        for name, replacement in tokens.items():
            value = value.replace("{" + name + "}", replacement)
        result.append(value)
    return result


class ParallelExecutor:
    """Run one immutable-shard campaign with bounded worker recovery."""

    def __init__(
        self,
        *,
        campaign: str | Path = "campaign",
        inventory: str | Path | Sequence[Mapping[str, Any]],
        output: str | Path | None = None,
        output_dir: str | Path | None = None,
        run_config: str | Path | Mapping[str, Any] | None = None,
        stage: str = "benchmark",
        worker_count: int | None = None,
        resume: bool = False,
        dry_run: bool = False,
        fake_model: bool = False,
        adapter: str | Sequence[str] | None = None,
        worker_command: Sequence[str] | None = None,
        stagger_seconds: float = 0.0,
        stall_seconds: float = 0.0,
        stall_after: int | None = None,
        crash_after: int | None = None,
        crash_once: bool = False,
        max_retries: int = 2,
        idle_timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 0.2,
        hard_ceiling_usd: float | None = None,
        budget_usd: float | None = None,
        reserve_usd: float = 0.0,
        gpu_hourly_rate_usd: float = 0.0,
        worker_estimate_seconds: float = 3600.0,
        run_mode: str | None = None,
        confirmatory: bool | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        process_factory: Callable[..., Any] | None = None,
        shutdown_command: str | Sequence[str] | None = None,
        shutdown_timeout_seconds: float = 30.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if stagger_seconds < 0 or stall_seconds < 0 or idle_timeout_seconds < 0 or poll_interval_seconds < 0:
            raise ValueError("stagger, stall, idle timeout, and poll interval must be non-negative.")
        if output is not None and output_dir is not None and Path(output).resolve() != Path(output_dir).resolve():
            raise ValueError("output and output_dir refer to different paths.")
        self.output_dir = Path(output if output is not None else output_dir if output_dir is not None else "parallel_run").expanduser().resolve()
        self.campaign_spec = str(campaign)
        self.campaign_path = Path(campaign).expanduser().resolve() if Path(str(campaign)).is_file() else None
        self.campaign_id, self.campaign_sha256 = self._resolve_campaign_identity(campaign)
        self.inventory_path = Path(inventory).expanduser().resolve() if isinstance(inventory, (str, Path)) else None
        if self.inventory_path is not None:
            self.requests = load_inventory_requests(self.inventory_path)
            self.inventory_sha256 = file_sha256(self.inventory_path)
            self.inventory_spec = str(self.inventory_path)
        else:
            self.requests = normalize_inventory_requests(inventory)
            self.inventory_sha256 = canonical_hash(self.requests)
            self.inventory_spec = "<in-memory-inventory>"
        self.run_config_path: Path | None = None
        self.run_config_payload: dict[str, Any] = {}
        if isinstance(run_config, (str, Path)):
            self.run_config_path = Path(run_config).expanduser().resolve()
            self.run_config_payload = _load_json_object(self.run_config_path, label="run config")
            # Worker A's distributed contract defines run_config_hash as the
            # canonical JSON hash, not a file-byte hash.
            self.run_config_hash = canonical_hash(self.run_config_payload)
        elif isinstance(run_config, Mapping):
            self.run_config_payload = dict(run_config)
            self.run_config_hash = canonical_hash(self.run_config_payload)
        else:
            self.run_config_hash = "UNSPECIFIED"
        execution_config = self._parallel_execution_config()
        configured_worker_count = execution_config.get("worker_count", 1) if worker_count is None else worker_count
        if not 1 <= int(configured_worker_count) <= 5:
            raise ValueError("worker_count must be between 1 and 5.")
        self.stage = str(stage or execution_config.get("stage", "benchmark"))
        self.run_mode = str(run_mode or execution_config.get("run_mode") or self._run_mode_from_config())
        inferred_confirmatory = self._confirmatory_from_config(self.run_mode)
        self.confirmatory = inferred_confirmatory if confirmatory is None else bool(confirmatory)
        configured_mode: Mapping[str, Any] = {}
        raw_execution = self.run_config_payload.get("execution")
        if isinstance(raw_execution, Mapping):
            raw_modes = raw_execution.get("run_modes")
            if isinstance(raw_modes, Mapping) and isinstance(raw_modes.get(self.run_mode), Mapping):
                configured_mode = raw_modes[self.run_mode]
        self.engineering_only = bool(configured_mode.get("engineering_only", False))
        if self.run_mode == "full" and not self.confirmatory and not self.engineering_only:
            raise ValueError(
                "full execution is confirmatory by default; a non-confirmatory full run "
                "requires a versioned engineering_only run mode."
            )
        if self.run_mode != "full" and self.engineering_only:
            raise ValueError("engineering_only is only valid for the full run mode.")
        self.worker_count = int(configured_worker_count)
        self.resume = bool(resume)
        self.dry_run = bool(dry_run)
        self.fake_model = bool(fake_model)
        self.stagger_seconds = float(stagger_seconds)
        self.stall_seconds = float(stall_seconds)
        self.stall_after = stall_after
        self.crash_after = crash_after
        self.crash_once = bool(crash_once)
        self.max_retries = int(max_retries)
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        ceiling = hard_ceiling_usd if hard_ceiling_usd is not None else budget_usd
        if ceiling is None:
            configured_ceiling = execution_config.get("hard_ceiling_usd", execution_config.get("budget_usd"))
            ceiling = float(configured_ceiling) if configured_ceiling is not None else None
        self.reserve_usd = float(execution_config.get("reserve_usd", reserve_usd) if reserve_usd == 0.0 else reserve_usd)
        self.gpu_hourly_rate_usd = float(
            execution_config.get("gpu_hourly_rate_usd", gpu_hourly_rate_usd)
            if gpu_hourly_rate_usd == 0.0
            else gpu_hourly_rate_usd
        )
        self.worker_estimate_seconds = float(execution_config.get("worker_estimate_seconds", worker_estimate_seconds))
        self.clock = clock or time.time
        self.sleeper = sleeper or time.sleep
        self.process_factory = process_factory or subprocess.Popen
        self.adapter_kind, self.adapter_value = self._resolve_adapter(adapter, worker_command, execution_config)
        if self.adapter_kind == "gpu" and self.worker_count > 4:
            raise ValueError("GPU execution supports at most four model replicas on one B300 pod.")
        if self.adapter_kind == "gpu" and self.run_config_path is None:
            raise AdapterError("The GPU adapter requires a run-config path with exact model/revision settings.")
        if self.adapter_kind == "gpu":
            construct_ids = {
                str(value)
                for request in self.requests
                for value in request.get("construct_ids", [])
                if str(value).strip()
            }
            if not construct_ids or any(not request.get("construct_ids") for request in self.requests):
                raise AdapterError(
                    "GPU execution requires construct IDs on every request so each model process can remain construct-pure."
                )
            if len(construct_ids) > self.worker_count:
                raise AdapterError(
                    "GPU execution cannot assign more construct-pure shards than in-pod model replicas; "
                    "use a construct-pure capacity subset for the 1/3-replica rollout or use four replicas."
                )
        self.adapter_identity = self._adapter_identity()
        if self.adapter_kind == "gpu":
            self.execution_topology = {
                "provider": "runpod",
                "pod_count": 1,
                "gpu_type": "NVIDIA B300",
                "gpu_count": 1,
                "model_replica_count": self.worker_count,
                "worker_count_semantics": "in_pod_model_replicas",
            }
        else:
            self.execution_topology = {
                "provider": "local_or_external",
                "pod_count": None,
                "gpu_type": None,
                "gpu_count": None,
                "model_replica_count": self.worker_count,
                "worker_count_semantics": "adapter_worker_slots",
            }
        self.shutdown_command = self._resolve_shutdown_command(shutdown_command, execution_config)
        configured_shutdown_timeout = execution_config.get(
            "shutdown_timeout_seconds", shutdown_timeout_seconds
        )
        self.shutdown_timeout_seconds = float(configured_shutdown_timeout)
        if self.shutdown_timeout_seconds < 0:
            raise ValueError("shutdown timeout must be non-negative.")
        self.campaign_identity = canonical_hash(
            {
                "campaign_id": self.campaign_id,
                "campaign_sha256": self.campaign_sha256,
                "inventory_sha256": self.inventory_sha256,
                "run_config_hash": self.run_config_hash,
                "stage": self.stage,
                "run_mode": self.run_mode,
                "confirmatory": self.confirmatory,
                "adapter": self.adapter_identity,
            }
        )
        self.state_path = self.output_dir / CAMPAIGN_STATE_FILENAME
        self.terminal_report_path = self.output_dir / TERMINAL_REPORT_FILENAME
        self.state: dict[str, Any] = {}
        self.shards: dict[str, dict[str, Any]] = {}
        self.entries: dict[str, dict[str, Any]] = {}
        self.active: dict[str, WorkerRuntime] = {}
        self.events: list[dict[str, Any]] = []
        self.started_clock = float(self.clock())
        self.last_launch_clock: float | None = None
        self.budget = BudgetGovernor(
            hard_ceiling_usd=float(ceiling) if ceiling is not None else None,
            reserve_usd=self.reserve_usd,
            gpu_hourly_rate_usd=self.gpu_hourly_rate_usd,
            worker_estimate_seconds=self.worker_estimate_seconds,
            clock=self.clock,
        )

    def _resolve_campaign_identity(self, campaign: str | Path) -> tuple[str, str | None]:
        path = Path(str(campaign)).expanduser()
        if path.is_file():
            payload = _load_json_object(path, label="campaign")
            campaign_id = str(payload.get("campaign_id", payload.get("id", path.stem)))
            return campaign_id, file_sha256(path)
        campaign_id = str(campaign).strip()
        if not campaign_id:
            raise ValueError("campaign must be a non-empty ID or JSON path.")
        return campaign_id, None

    def _parallel_execution_config(self) -> dict[str, Any]:
        for key in ("parallel_execution", "parallel_executor", "execution"):
            value = self.run_config_payload.get(key)
            if isinstance(value, Mapping):
                candidate = value.get("parallel_executor", value) if isinstance(value, Mapping) else value
                if isinstance(candidate, Mapping):
                    return dict(candidate)
        return {}

    def _run_mode_from_config(self) -> str:
        execution = self.run_config_payload.get("execution")
        if isinstance(execution, Mapping):
            value = execution.get("default_run_mode")
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = self.run_config_payload.get("run_mode")
        return str(value).strip() if isinstance(value, str) and value.strip() else "test"

    def _confirmatory_from_config(self, mode: str) -> bool:
        execution = self.run_config_payload.get("execution")
        if isinstance(execution, Mapping):
            modes = execution.get("run_modes")
            if isinstance(modes, Mapping) and isinstance(modes.get(mode), Mapping):
                value = modes[mode].get("confirmatory")
                if isinstance(value, bool):
                    return value
        return bool(self.run_config_payload.get("confirmatory", False))

    def _resolve_adapter(
        self,
        adapter: str | Sequence[str] | None,
        worker_command: Sequence[str] | None,
        execution_config: Mapping[str, Any],
    ) -> tuple[str, str | list[str]]:
        configured = worker_command if worker_command is not None else adapter
        if self.fake_model:
            # An explicit local fake opt-in is allowed to override a real
            # adapter declared by a versioned run config.  An explicit CLI
            # adapter/command remains contradictory and is refused.
            if worker_command is not None or (
                adapter is not None
                and not (isinstance(adapter, str) and adapter.strip() in {"", "fake"})
            ):
                raise AdapterError(
                    "--fake-model cannot be combined with a real adapter or worker command."
                )
            return "fake", "fake"
        if configured is None:
            configured = execution_config.get("worker_command", execution_config.get("adapter"))
        if configured is None:
            raise AdapterError(
                "No real worker adapter is configured; production execution is fail-closed. "
                "Provide --adapter/--worker-command or explicitly opt into --fake-model."
            )
        if isinstance(configured, str):
            value = configured.strip()
            if value == "fake":
                raise AdapterError("The fake adapter requires explicit --fake-model opt-in.")
            if value == "gpu":
                return "gpu", "gpu"
            if value.startswith("python:") or value.count(":") == 1:
                return "python", value
            # A string command is parsed as argv only; shell evaluation is
            # never used.  Explicit Python adapters remain unambiguous.
            try:
                argv = shlex.split(value, posix=(os.name != "nt"))
            except ValueError as exc:
                raise AdapterError(f"Invalid worker command: {exc}") from exc
            if not argv:
                raise AdapterError("Worker command must not be empty.")
            return "command", argv
        argv = [str(item) for item in configured]
        if not argv:
            raise AdapterError("Worker command must not be empty.")
        return "command", argv

    def _resolve_shutdown_command(
        self,
        command: str | Sequence[str] | None,
        execution_config: Mapping[str, Any],
    ) -> list[str] | None:
        configured = command if command is not None else execution_config.get("shutdown_command")
        if configured is None:
            return None
        if isinstance(configured, str):
            try:
                argv = shlex.split(configured, posix=(os.name != "nt"))
            except ValueError as exc:
                raise AdapterError(f"Invalid shutdown command: {exc}") from exc
        elif isinstance(configured, Sequence) and not isinstance(configured, (bytes, bytearray)):
            argv = [str(item) for item in configured]
        else:
            raise AdapterError("Shutdown command must be an argv sequence or string.")
        if not argv:
            raise AdapterError("Shutdown command must not be empty.")
        return argv

    def _adapter_identity(self) -> dict[str, Any]:
        if self.adapter_kind == "fake":
            return {"kind": "fake", "name": "deterministic_fake_v1"}
        if self.adapter_kind == "gpu":
            return {"kind": "gpu", "name": "construct_benchmark_gpu_worker_v1"}
        if self.adapter_kind == "python":
            return {"kind": "python", "callable": str(self.adapter_value)}
        return {"kind": "command", "argv": list(self.adapter_value)}

    def _expected_identity(self, *, shard_manifest_sha256: str | None = None) -> dict[str, Any]:
        identity = {
            "campaign_identity": self.campaign_identity,
            "parent_inventory_sha256": self.inventory_sha256,
            "run_config_hash": self.run_config_hash,
            "run_mode": self.run_mode,
            "confirmatory": self.confirmatory,
            "stage": self.stage,
            "adapter": self.adapter_identity,
        }
        if shard_manifest_sha256 is not None:
            identity["shard_manifest_sha256"] = shard_manifest_sha256
        return identity

    def _try_worker_a_shards(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]], int] | None:
        """Use Worker A's physical sharder when its contract is applicable.

        Worker A's standard four-construct/three-worker plan has four physical
        construct-pure shards scheduled across three subprocess slots.  The
        executor creates one immutable per-worker bundle manifest for the
        subprocess while retaining and hashing every physical A manifest in
        ``physical_shards``.  Thus no physical work is duplicated and the
        3-worker/4-shard schedule remains inspectable and interoperable.

        The only permitted ``None`` result is a schema-neutral inventory or an
        unavailable Worker A adapter.  Once construct/pair/cell/unit/role or
        version metadata is present and Worker A is importable, every Worker A
        rejection is terminal for planning; generic contiguous sharding is
        never used as a recovery path.
        """

        if (
            self.inventory_path is None
            or self.worker_count < 3
            or _worker_a_build_shard_plan is None
            or _worker_a_load_inventory is None
            or _worker_a_write_shard_outputs is None
        ):
            return None
        if not _inventory_is_construct_aware(self.requests):
            return None
        try:
            a_records = _worker_a_load_inventory(self.inventory_path)
            construct_ids = {record.construct_id for record in a_records}
        except Exception as exc:
            raise WorkerAContractError(
                "Worker A rejected this construct-aware inventory during loading; "
                f"generic fallback is disabled: {exc}"
            ) from exc
        if len(construct_ids) not in {1, 4}:
            raise WorkerAContractError(
                "Worker A cannot plan this construct-aware inventory: "
                f"expected one or four constructs, found {sorted(construct_ids)}; "
                "generic fallback is disabled."
            )
        physical_count = 4 if self.worker_count == 3 and len(construct_ids) == 4 else self.worker_count
        try:
            plan = _worker_a_build_shard_plan(
                self.inventory_path,
                shard_count=physical_count,
                worker_count=self.worker_count,
                run_config_hash=self.run_config_hash,
                run_mode=self.run_mode,
                confirmatory=self.confirmatory,
                engineering_only=self.engineering_only,
            )
            physical_root = self.output_dir / "physical_shards"
            report = _worker_a_write_shard_outputs(
                plan,
                physical_root,
                inventory_suffix=self.inventory_path.suffix or ".jsonl",
            )
        except Exception as exc:
            raise WorkerAContractError(
                "Worker A rejected this construct-aware inventory during planning or "
                f"materialization; generic fallback is disabled: {exc}"
            ) from exc
        report_by_id = {str(item["shard_id"]): item for item in report.get("shards", [])}
        physical: list[dict[str, Any]] = []
        adapted_by_id: dict[str, dict[str, Any]] = {}
        identity = self._expected_identity()
        for shard_id in sorted(plan.manifests):
            report_item = report_by_id.get(shard_id)
            if report_item is None:
                raise WorkerAContractError(
                    f"Worker A materialization omitted physical shard {shard_id!r}; "
                    "generic fallback is disabled."
                )
            try:
                manifest_path = Path(str(report_item["manifest_path"])).resolve()
                raw = load_shard_manifest(manifest_path, expected_identity=identity)
                manifest_digest = file_sha256(manifest_path)
            except Exception as exc:
                raise WorkerAContractError(
                    "Worker A produced an invalid physical shard manifest; "
                    f"generic fallback is disabled: {exc}"
                ) from exc
            raw["manifest_path"] = str(manifest_path)
            raw["manifest_sha256"] = manifest_digest
            adapted_by_id[shard_id] = raw
            physical.append(
                {
                    "shard_id": shard_id,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": raw["manifest_sha256"],
                    "inventory_path": str(report_item.get("inventory_path", "")),
                    "request_ids": list(raw["request_ids"]),
                    "expected_observation_ids": list(raw["expected_observation_ids"]),
                    "worker_id": plan.manifests[shard_id].get("worker_slot"),
                }
            )
        aggregates: list[dict[str, Any]] = []
        schedule = {
            str(worker_id): [str(shard_id) for shard_id in shard_ids]
            for worker_id, shard_ids in plan.worker_schedule.items()
        }
        aggregate_root = self.output_dir / "shards"
        for worker_id, shard_ids in sorted(schedule.items()):
            requests = [request for shard_id in shard_ids for request in adapted_by_id[shard_id]["requests"]]
            requests.sort(key=lambda request: str(request["request_id"]))
            shard_id = f"{worker_id}_bundle"
            payload = _shard_manifest_payload(
                shard_id=shard_id,
                worker_id=worker_id,
                requests=requests,
                parent_inventory_sha256=self.inventory_sha256,
                run_config_hash=self.run_config_hash,
                run_mode=self.run_mode,
                confirmatory=self.confirmatory,
                engineering_only=self.engineering_only,
                stage=self.stage,
                campaign_identity=self.campaign_identity,
            )
            path = aggregate_root / f"{shard_id}.json"
            if path.exists():
                raise FileExistsError(f"Refusing to replace immutable worker shard manifest: {path}")
            atomic_write_json(path, payload)
            payload["manifest_path"] = str(path)
            payload["manifest_sha256"] = file_sha256(path)
            payload["physical_shard_ids"] = shard_ids
            aggregates.append(payload)
        return aggregates, physical, schedule, physical_count

    def _new_state(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        worker_a_layout = self._try_worker_a_shards()
        if worker_a_layout is None:
            shard_dir = self.output_dir / "shards"
            shard_payloads = build_shard_manifests(
                self.requests,
                output_dir=shard_dir,
                worker_count=self.worker_count,
                parent_inventory_sha256=self.inventory_sha256,
                run_config_hash=self.run_config_hash,
                run_mode=self.run_mode,
                confirmatory=self.confirmatory,
                engineering_only=self.engineering_only,
                stage=self.stage,
                campaign_identity=self.campaign_identity,
            )
            physical_shards: list[dict[str, Any]] = []
            worker_schedule = {str(item["worker_id"]): [str(item["shard_id"])] for item in shard_payloads}
            physical_shard_count = len(shard_payloads)
        else:
            shard_payloads, physical_shards, worker_schedule, physical_shard_count = worker_a_layout
        self.shards = {str(item["shard_id"]): item for item in shard_payloads}
        self.entries = {}
        for shard in shard_payloads:
            self.entries[shard["worker_id"]] = self._create_worker_entry(shard)
        self.state = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": CAMPAIGN_STATE_TYPE,
            "campaign_id": self.campaign_id,
            "campaign_identity": self.campaign_identity,
            "campaign_sha256": self.campaign_sha256,
            "inventory_path": self.inventory_spec,
            "inventory_sha256": self.inventory_sha256,
            "run_config_path": str(self.run_config_path) if self.run_config_path else None,
            "run_config_hash": self.run_config_hash,
            "stage": self.stage,
            "run_mode": self.run_mode,
            "confirmatory": self.confirmatory,
            "worker_count": self.worker_count,
            "max_retries": self.max_retries,
            "stagger_seconds": self.stagger_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "adapter": self.adapter_identity,
            "execution_topology": self.execution_topology,
            "physical_shard_count": physical_shard_count,
            "physical_shards": physical_shards,
            "worker_schedule": worker_schedule,
            "status": "planned",
            "created_at": _utc_now(self.started_clock),
            "updated_at": _utc_now(self.started_clock),
            "shards": [self._shard_entry(shard) for shard in shard_payloads],
            "workers": [],
            "output_paths": [],
            "progress": {},
            "budget": self.budget.snapshot(0),
            "events": [],
            "error": None,
            "terminal_reason": None,
            "continuation": {},
        }
        for entry in self.entries.values():
            self._write_initial_worker_manifest(entry, status="planned")
        self._refresh_state(persist=False)
        self._write_state()

    def _create_worker_entry(self, shard: Mapping[str, Any]) -> dict[str, Any]:
        worker_id = str(shard["worker_id"])
        worker_dir = self.output_dir / "workers" / worker_id
        worker_manifest_path = worker_dir / "worker_manifest.json"
        output_path = worker_dir / "output.jsonl"
        log_path = worker_dir / "worker.log"
        return {
            "worker_id": worker_id,
            "shard_id": shard["shard_id"],
            "shard_manifest_path": str(shard["manifest_path"]),
            "shard_manifest_sha256": shard["manifest_sha256"],
            "worker_manifest_path": str(worker_manifest_path),
            "checkpoint_path": str(worker_manifest_path),
            "output_path": str(output_path),
            "log_path": str(log_path),
            "status": "planned",
            "pid": None,
            "stage": self.stage,
            "retry_count": 0,
            "expected_request_count": shard["expected_request_count"],
            "completed_request_count": 0,
            "expected_observation_count": shard["expected_observation_count"],
            "completed_observation_count": 0,
            "last_progress": None,
            "heartbeat_at": None,
            "heartbeat_at_epoch": None,
            "error": None,
            "terminal_reason": None,
            "execution_identity": self._expected_identity(shard_manifest_sha256=shard["manifest_sha256"]),
        }

    def _shard_entry(self, shard: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "shard_id": shard["shard_id"],
            "worker_id": shard["worker_id"],
            "manifest_path": shard["manifest_path"],
            "manifest_sha256": shard["manifest_sha256"],
            "request_ids": list(shard["request_ids"]),
            "expected_observation_ids": list(shard["expected_observation_ids"]),
            "status": "planned",
        }

    def _write_initial_worker_manifest(self, entry: Mapping[str, Any], *, status: str) -> None:
        shard = self.shards[str(entry["shard_id"])]
        path = Path(entry["worker_manifest_path"])
        payload = _worker_payload(
            worker_id=str(entry["worker_id"]),
            shard=shard,
            status=status,
            output_path=entry["output_path"],
            worker_manifest_path=path,
            stage=self.stage,
            retry_count=int(entry.get("retry_count", 0)),
            execution_identity=entry["execution_identity"],
            now=self.started_clock,
        )
        _write_worker_manifest(path, payload)

    def _write_state(self) -> None:
        self.state["updated_at"] = _utc_now(float(self.clock()))
        atomic_write_json(self.state_path, self.state)

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            raise InvalidCheckpointError(f"Cannot resume without {self.state_path}.")
        self.state = _load_json_object(self.state_path, label="campaign state")
        if self.state.get("schema_version") != SCHEMA_VERSION or self.state.get("manifest_type") != CAMPAIGN_STATE_TYPE:
            raise InvalidCheckpointError("Campaign state has an unsupported schema or manifest_type.")
        raw_events = self.state.get("events", [])
        if not isinstance(raw_events, list):
            raise InvalidCheckpointError("Campaign state events must be a list.")
        self.events = [dict(event) for event in raw_events if isinstance(event, Mapping)]
        expected = {
            "campaign_id": self.campaign_id,
            "campaign_identity": self.campaign_identity,
            "campaign_sha256": self.campaign_sha256,
            "inventory_sha256": self.inventory_sha256,
            "run_config_hash": self.run_config_hash,
            "stage": self.stage,
            "run_mode": self.run_mode,
            "confirmatory": self.confirmatory,
        }
        for field, value in expected.items():
            if self.state.get(field) != value:
                raise InvalidCheckpointError(
                    f"Campaign identity mismatch for {field}: {self.state.get(field)!r} != {value!r}."
                )
        state_adapter = self.state.get("adapter")
        if state_adapter != self.adapter_identity:
            raise InvalidCheckpointError("Campaign adapter identity does not match the checkpoint.")
        shard_entries = self.state.get("shards")
        worker_entries = self.state.get("workers")
        if not isinstance(shard_entries, list) or not isinstance(worker_entries, list):
            raise InvalidCheckpointError("Campaign state must contain shards and workers lists.")
        self.shards = {}
        for shard_entry in shard_entries:
            if not isinstance(shard_entry, Mapping):
                raise InvalidCheckpointError("Campaign shard entry is malformed.")
            path = Path(str(shard_entry.get("manifest_path", "")))
            if not path.is_file() or file_sha256(path) != shard_entry.get("manifest_sha256"):
                raise InvalidCheckpointError(f"Shard checkpoint is missing or hash-mismatched: {path}")
            shard = load_shard_manifest(path, expected_identity={
                "campaign_identity": self.campaign_identity,
                "parent_inventory_sha256": self.inventory_sha256,
                "run_config_hash": self.run_config_hash,
                "run_mode": self.run_mode,
                "confirmatory": self.confirmatory,
                "stage": self.stage,
            })
            if shard.get("shard_id") != shard_entry.get("shard_id") or shard.get("worker_id") != shard_entry.get("worker_id"):
                raise InvalidCheckpointError("Campaign shard entry does not match its immutable shard.")
            shard["manifest_path"] = str(path)
            shard["manifest_sha256"] = str(shard_entry["manifest_sha256"])
            self.shards[shard["shard_id"]] = shard
        physical_entries = self.state.get("physical_shards", [])
        if not isinstance(physical_entries, list):
            raise InvalidCheckpointError("Campaign physical_shards must be a list.")
        for physical_entry in physical_entries:
            if not isinstance(physical_entry, Mapping):
                raise InvalidCheckpointError("Campaign physical shard entry is malformed.")
            physical_path = Path(str(physical_entry.get("manifest_path", "")))
            if not physical_path.is_file() or file_sha256(physical_path) != physical_entry.get("manifest_sha256"):
                raise InvalidCheckpointError(f"Physical shard manifest is missing or hash-mismatched: {physical_path}")
            # The adapter validates Worker A's benchmark_shard shape and its
            # exact parent/run identity while retaining the original file.
            load_shard_manifest(physical_path, expected_identity={
                "parent_inventory_sha256": self.inventory_sha256,
                "run_config_hash": self.run_config_hash,
                "run_mode": self.run_mode,
                "confirmatory": self.confirmatory,
                "stage": self.stage,
            })
        self.entries = {}
        for raw_entry in worker_entries:
            if not isinstance(raw_entry, Mapping):
                raise InvalidCheckpointError("Campaign worker entry is malformed.")
            entry = dict(raw_entry)
            worker_id = str(entry.get("worker_id", ""))
            shard_id = str(entry.get("shard_id", ""))
            if worker_id not in {str(shard["worker_id"]) for shard in self.shards.values()} and entry.get("status") != "superseded":
                raise InvalidCheckpointError(f"Worker {worker_id!r} references no shard.")
            if shard_id not in self.shards:
                raise InvalidCheckpointError(f"Worker {worker_id!r} references missing shard {shard_id!r}.")
            manifest_path = Path(str(entry.get("worker_manifest_path", "")))
            output_path = Path(str(entry.get("output_path", "")))
            shard = self.shards[shard_id]
            manifest = load_worker_manifest(
                manifest_path,
                shard=shard,
                expected_identity=self._expected_identity(shard_manifest_sha256=shard["manifest_sha256"]),
                expected_output_path=output_path,
            )
            progress = read_output_progress(output_path, shard)
            if not set(manifest["completed_observation_ids"]).issubset(progress.completed_observation_ids):
                raise InvalidCheckpointError(
                    f"Worker {worker_id!r} checkpoint claims observations absent from output."
                )
            entry.update(self._entry_from_manifest(entry, manifest, progress))
            self.entries[worker_id] = entry
        if len(self.entries) != len(worker_entries):
            raise InvalidCheckpointError("Campaign worker IDs must be unique.")
        stored_budget = self.state.get("budget")
        if isinstance(stored_budget, Mapping):
            stored_spend = stored_budget.get("actual_spend_usd")
            if isinstance(stored_spend, (int, float)) and not isinstance(stored_spend, bool):
                self.budget.actual_spend_usd = max(0.0, float(stored_spend))
        self.budget.last_update_clock = float(self.clock())
        created_epoch = _iso_to_epoch(self.state.get("created_at"))
        if created_epoch is not None:
            self.started_clock = created_epoch
        self._refresh_state(persist=False)

    def _prepare_resumable_failures(self) -> None:
        """Reopen recoverable terminal checkpoints when resume supplies budget."""

        recoverable = {
            "worker_exit",
            "exit_before_completion",
            "idle_timeout",
            "malformed_manifest",
            "malformed_output",
            "worker_exception",
            "worker_launch",
            "malformed_checkpoint",
            "campaign_terminal_worker_failure",
            "campaign_terminal_budget_failure",
            "budget_cutoff",
        }
        changed = False
        for entry in self.entries.values():
            entry_changed = False
            status = entry.get("status")
            reason = entry.get("terminal_reason")
            if status == "failed" and (
                reason in recoverable
                or (isinstance(reason, str) and reason.startswith("campaign_terminal_"))
            ) and int(entry.get("retry_count", 0)) < self.max_retries:
                entry["status"] = "recovering"
                entry["terminal_reason"] = None
                entry["error"] = None
                changed = True
                entry_changed = True
            elif status == "budget_refused" and self.budget.hard_ceiling_usd is not None:
                entry["status"] = "recovering"
                entry["terminal_reason"] = None
                entry["error"] = None
                changed = True
                entry_changed = True
            if entry_changed:
                shard = self.shards[str(entry["shard_id"])]
                progress = read_output_progress(entry["output_path"], shard)
                payload = _worker_payload(
                    worker_id=str(entry["worker_id"]),
                    shard=shard,
                    status=str(entry["status"]),
                    output_path=entry["output_path"],
                    worker_manifest_path=entry["worker_manifest_path"],
                    stage=self.stage,
                    retry_count=int(entry.get("retry_count", 0)),
                    execution_identity=entry["execution_identity"],
                    now=float(self.clock()),
                    completed_request_ids=progress.completed_request_ids,
                    completed_observation_ids=progress.completed_observation_ids,
                )
                _write_worker_manifest(Path(entry["worker_manifest_path"]), payload)
        if changed:
            self._refresh_state()

    def _entry_from_manifest(
        self,
        entry: Mapping[str, Any],
        manifest: Mapping[str, Any],
        progress: OutputProgress | None = None,
    ) -> dict[str, Any]:
        result = dict(entry)
        result.update(
            {
                "status": manifest.get("status"),
                "pid": manifest.get("pid"),
                "stage": manifest.get("stage"),
                "retry_count": manifest.get("retry_count", 0),
                "completed_request_ids": list(manifest.get("completed_request_ids", [])),
                "completed_observation_ids": list(manifest.get("completed_observation_ids", [])),
                "expected_request_count": manifest.get("expected_request_count"),
                "completed_request_count": manifest.get("completed_request_count"),
                "expected_observation_count": manifest.get("expected_observation_count"),
                "completed_observation_count": manifest.get("completed_observation_count"),
                "heartbeat_at": manifest.get("heartbeat_at"),
                "heartbeat_at_epoch": manifest.get("heartbeat_at_epoch"),
                "last_progress": manifest.get("last_progress"),
                "error": manifest.get("error"),
                "terminal_reason": manifest.get("terminal_reason"),
                "execution_identity": dict(manifest.get("execution_identity", {})),
            }
        )
        if progress is not None:
            result["observed_completed_request_ids"] = sorted(progress.completed_request_ids)
            result["observed_completed_observation_ids"] = sorted(progress.completed_observation_ids)
            result["output_size"] = progress.byte_size
            result["output_row_count"] = progress.row_count
        return result

    def _refresh_state(self, *, persist: bool = True) -> None:
        worker_state: list[dict[str, Any]] = []
        output_paths: list[str] = []
        for worker_id in sorted(self.entries):
            entry = self.entries[worker_id]
            worker_state.append(dict(entry))
            output_paths.append(str(entry["output_path"]))
        self.state["workers"] = worker_state
        self.state["output_paths"] = output_paths
        for shard_entry in self.state.get("shards", []):
            worker_id = str(shard_entry.get("worker_id"))
            if worker_id in self.entries:
                shard_entry["status"] = self.entries[worker_id].get("status")
        completed_requests: set[str] = set()
        completed_observations: set[str] = set()
        for entry in self.entries.values():
            completed_requests.update(entry.get("observed_completed_request_ids", entry.get("completed_request_ids", [])))
            completed_observations.update(entry.get("observed_completed_observation_ids", entry.get("completed_observation_ids", [])))
        expected_request_count = len(self.requests) if self.inventory_path is not None or self.requests else sum(
            int(shard.get("expected_request_count", 0)) for shard in self.shards.values()
        )
        expected_observation_count = sum(len(request["observation_ids"]) for request in self.requests)
        self.state["progress"] = {
            "expected_request_count": expected_request_count,
            "completed_request_count": len(completed_requests),
            "expected_observation_count": expected_observation_count,
            "completed_observation_count": len(completed_observations),
            "completed_request_ids": sorted(completed_requests),
            "completed_observation_ids": sorted(completed_observations),
            "last_progress": self.state.get("progress", {}).get("last_progress"),
        }
        self.state["budget"] = self.budget.snapshot(len(self.active))
        self.state["events"] = list(self.events)
        if persist:
            self._write_state()

    def _record_progress(self, worker_id: str, progress: OutputProgress, *, now: float) -> None:
        entry = self.entries[worker_id]
        previous = (
            int(entry.get("completed_request_count", 0)),
            int(entry.get("completed_observation_count", 0)),
            int(entry.get("output_size", 0)),
        )
        entry["observed_completed_request_ids"] = sorted(progress.completed_request_ids)
        entry["observed_completed_observation_ids"] = sorted(progress.completed_observation_ids)
        entry["output_size"] = progress.byte_size
        entry["output_row_count"] = progress.row_count
        current = (len(progress.completed_request_ids), len(progress.completed_observation_ids), progress.byte_size)
        if current != previous:
            entry["last_progress"] = {
                "at": _utc_now(now),
                "request_count": len(progress.completed_request_ids),
                "observation_count": len(progress.completed_observation_ids),
            }
            state_progress = self.state.setdefault("progress", {})
            state_progress["last_progress"] = dict(entry["last_progress"])

    def _ensure_fresh_output_directory(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return
        existing = list(self.output_dir.iterdir())
        if existing and not self.resume:
            raise FileExistsError(
                f"Output directory {self.output_dir} is not empty; use --resume for the same campaign or choose a new path."
            )

    def _prepare(self) -> None:
        if self.resume:
            self._load_state()
            return
        self._ensure_fresh_output_directory()
        if self.state_path.exists():
            raise FileExistsError(f"Campaign state already exists at {self.state_path}; use --resume.")
        self._new_state()

    def _worker_command(self, entry: Mapping[str, Any], shard: Mapping[str, Any]) -> list[str]:
        tokens = {
            "campaign": self.campaign_id,
            "worker_id": str(entry["worker_id"]),
            "shard_id": str(shard["shard_id"]),
            "shard_manifest": str(entry["shard_manifest_path"]),
            "worker_manifest": str(entry["worker_manifest_path"]),
            "checkpoint": str(entry["checkpoint_path"]),
            "output_path": str(entry["output_path"]),
            "log_path": str(entry["log_path"]),
            "stage": self.stage,
        }
        if self.adapter_kind == "command":
            return _argv_with_tokens(self.adapter_value, tokens)
        if self.adapter_kind == "gpu":
            command = [
                sys.executable,
                "-m",
                "construct_benchmark.gpu_worker",
                "--shard-manifest",
                str(entry["shard_manifest_path"]),
                "--worker-manifest",
                str(entry["worker_manifest_path"]),
                "--output",
                str(entry["output_path"]),
                "--stage",
                self.stage,
            ]
            if self.run_config_path is None:
                raise AdapterError("The GPU adapter requires --run-config with exact model/revision settings.")
            command.extend(["--run-config", str(self.run_config_path)])
            return command
        command = [
            sys.executable,
            "-m",
            "construct_benchmark.parallel_executor",
            "--worker-process",
            "--shard-manifest",
            str(entry["shard_manifest_path"]),
            "--worker-manifest",
            str(entry["worker_manifest_path"]),
            "--output",
            str(entry["output_path"]),
            "--stage",
            self.stage,
            "--adapter",
            "fake" if self.adapter_kind == "fake" else str(self.adapter_value),
        ]
        if self.fake_model:
            command.append("--fake-model")
        if self.stall_seconds > 0:
            command.extend(["--stall", str(self.stall_seconds)])
        if self.stall_after is not None:
            command.extend(["--stall-after", str(self.stall_after)])
        if self.crash_after is not None:
            command.extend(["--crash-after", str(self.crash_after)])
        if self.crash_once:
            command.append("--crash-once")
        return command

    def _spawn_process(self, command: Sequence[str], *, log_path: Path) -> tuple[Any, Any]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parent.parent)
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source_root if not existing_pythonpath else source_root + os.pathsep + existing_pythonpath
        try:
            try:
                process = self.process_factory(list(command), stdout=handle, stderr=subprocess.STDOUT, env=environment)
            except TypeError:
                # Test process factories commonly accept only command/env or
                # only command.  Both remain argv-only and shell-free.
                try:
                    process = self.process_factory(list(command), env=environment)
                except TypeError:
                    process = self.process_factory(list(command))
        except Exception:
            handle.close()
            raise
        return process, handle

    def _start_worker(self, worker_id: str) -> bool:
        entry = self.entries[worker_id]
        if entry.get("status") in TERMINAL_WORKER_STATUSES:
            return False
        allowed, reason = self.budget.can_launch(len(self.active))
        if not allowed:
            refusal = {
                "worker_id": worker_id,
                "at": _utc_now(float(self.clock())),
                "reason": reason,
            }
            self.budget.launch_refusals.append(refusal)
            self._update_worker_terminal(worker_id, status="budget_refused", reason="budget_cutoff", error=reason)
            self.events.append({"event": "launch_refused", **refusal})
            return False
        shard = self.shards[str(entry["shard_id"])]
        progress = read_output_progress(entry["output_path"], shard)
        manifest_path = Path(entry["worker_manifest_path"])
        if manifest_path.is_file():
            manifest = load_worker_manifest(
                manifest_path,
                shard=shard,
                expected_identity=self._expected_identity(shard_manifest_sha256=entry["shard_manifest_sha256"]),
                expected_output_path=entry["output_path"],
            )
            if not set(manifest["completed_observation_ids"]).issubset(progress.completed_observation_ids):
                raise InvalidCheckpointError(f"Worker {worker_id} checkpoint is ahead of its output.")
            completed_requests = progress.completed_request_ids
            completed_observations = progress.completed_observation_ids
            retry_count = int(entry.get("retry_count", manifest.get("retry_count", 0)))
        else:
            completed_requests = progress.completed_request_ids
            completed_observations = progress.completed_observation_ids
            retry_count = int(entry.get("retry_count", 0))
        starting = _worker_payload(
            worker_id=worker_id,
            shard=shard,
            status="starting",
            output_path=entry["output_path"],
            worker_manifest_path=manifest_path,
            stage=self.stage,
            retry_count=retry_count,
            execution_identity=self._expected_identity(shard_manifest_sha256=entry["shard_manifest_sha256"]),
            now=float(self.clock()),
            completed_request_ids=completed_requests,
            completed_observation_ids=completed_observations,
            pid=None,
        )
        # This write occurs before launch, so there is no parent/worker race.
        _write_worker_manifest(manifest_path, starting)
        command = self._worker_command(entry, shard)
        process, log_handle = self._spawn_process(command, log_path=Path(entry["log_path"]))
        pid = getattr(process, "pid", None)
        entry["pid"] = int(pid) if isinstance(pid, int) else pid
        entry["status"] = "running"
        entry["retry_count"] = retry_count
        entry["heartbeat_at"] = starting.get("heartbeat_at")
        entry["heartbeat_at_epoch"] = starting.get("heartbeat_at_epoch")
        entry["last_progress"] = starting.get("last_progress")
        entry["error"] = None
        entry["terminal_reason"] = None
        now = float(self.clock())
        runtime = WorkerRuntime(
            worker_id=worker_id,
            shard=shard,
            entry=entry,
            process=process,
            manifest_owned_by_worker=self.adapter_kind in {"fake", "python", "gpu"},
            last_output_signature=(len(progress.completed_request_ids), len(progress.completed_observation_ids), progress.byte_size),
            last_progress_clock=now,
            started_clock=now,
            log_handle=log_handle,
        )
        self.active[worker_id] = runtime
        self.last_launch_clock = now
        self.events.append({
            "event": "worker_started",
            "worker_id": worker_id,
            "pid": entry["pid"],
            "stage": self.stage,
            "at": _utc_now(now),
            "command": list(command),
        })
        self._refresh_state()
        return True

    def _update_worker_terminal(self, worker_id: str, *, status: str, reason: str, error: str | None = None) -> None:
        entry = self.entries[worker_id]
        shard = self.shards[str(entry["shard_id"])]
        progress = read_output_progress(entry["output_path"], shard)
        entry.update(
            {
                "status": status,
                "error": error,
                "terminal_reason": reason,
                "completed_request_ids": sorted(progress.completed_request_ids),
                "completed_observation_ids": sorted(progress.completed_observation_ids),
                "completed_request_count": len(progress.completed_request_ids),
                "completed_observation_count": len(progress.completed_observation_ids),
                "observed_completed_request_ids": sorted(progress.completed_request_ids),
                "observed_completed_observation_ids": sorted(progress.completed_observation_ids),
                "output_size": progress.byte_size,
                "output_row_count": progress.row_count,
                "pid": None,
            }
        )
        payload = _worker_payload(
            worker_id=worker_id,
            shard=shard,
            status=status,
            output_path=entry["output_path"],
            worker_manifest_path=entry["worker_manifest_path"],
            stage=self.stage,
            retry_count=int(entry.get("retry_count", 0)),
            execution_identity=entry["execution_identity"],
            now=float(self.clock()),
            completed_request_ids=progress.completed_request_ids,
            completed_observation_ids=progress.completed_observation_ids,
            error=error,
            terminal_reason=reason,
            pid=None,
        )
        _write_worker_manifest(Path(entry["worker_manifest_path"]), payload)

    def _finish_worker(self, worker_id: str, *, successful: bool, reason: str | None = None, error: str | None = None) -> None:
        runtime = self.active.pop(worker_id, None)
        if runtime is not None:
            if runtime.log_handle is not None:
                runtime.log_handle.flush()
                runtime.log_handle.close()
        entry = self.entries[worker_id]
        shard = self.shards[str(entry["shard_id"])]
        try:
            progress = read_output_progress(entry["output_path"], shard)
        except InvalidCheckpointError:
            if successful:
                raise
            progress = OutputProgress(frozenset(), frozenset(), 0, 0)
        if successful:
            entry.update(
                {
                    "status": "complete",
                    "error": None,
                    "terminal_reason": reason,
                    "completed_request_ids": sorted(progress.completed_request_ids),
                    "completed_observation_ids": sorted(progress.completed_observation_ids),
                    "completed_request_count": len(progress.completed_request_ids),
                    "completed_observation_count": len(progress.completed_observation_ids),
                    "observed_completed_request_ids": sorted(progress.completed_request_ids),
                    "observed_completed_observation_ids": sorted(progress.completed_observation_ids),
                    "output_size": progress.byte_size,
                    "output_row_count": progress.row_count,
                    "pid": None,
                }
            )
        else:
            self._handle_worker_failure(worker_id, reason=reason or "worker_failure", error=error, progress=progress)
            return
        payload = _worker_payload(
            worker_id=worker_id,
            shard=shard,
            status="complete",
            output_path=entry["output_path"],
            worker_manifest_path=entry["worker_manifest_path"],
            stage=self.stage,
            retry_count=int(entry.get("retry_count", 0)),
            execution_identity=entry["execution_identity"],
            now=float(self.clock()),
            completed_request_ids=progress.completed_request_ids,
            completed_observation_ids=progress.completed_observation_ids,
            error=None,
            terminal_reason=reason,
            pid=None,
        )
        _write_worker_manifest(Path(entry["worker_manifest_path"]), payload)
        self.events.append({"event": "worker_complete", "worker_id": worker_id, "at": _utc_now(float(self.clock()))})
        self._refresh_state()

    def _handle_worker_failure(
        self,
        worker_id: str,
        *,
        reason: str,
        error: str | None,
        progress: OutputProgress,
    ) -> None:
        entry = self.entries[worker_id]
        retry_count = int(entry.get("retry_count", 0))
        entry.update(
            {
                "completed_request_ids": sorted(progress.completed_request_ids),
                "completed_observation_ids": sorted(progress.completed_observation_ids),
                "completed_request_count": len(progress.completed_request_ids),
                "completed_observation_count": len(progress.completed_observation_ids),
                "observed_completed_request_ids": sorted(progress.completed_request_ids),
                "observed_completed_observation_ids": sorted(progress.completed_observation_ids),
                "output_size": progress.byte_size,
                "output_row_count": progress.row_count,
                "error": error,
                "pid": None,
            }
        )
        if retry_count < self.max_retries:
            entry["retry_count"] = retry_count + 1
            entry["status"] = "recovering"
            entry["terminal_reason"] = None
            payload = _worker_payload(
                worker_id=worker_id,
                shard=self.shards[str(entry["shard_id"])],
                status="recovering",
                output_path=entry["output_path"],
                worker_manifest_path=entry["worker_manifest_path"],
                stage=self.stage,
                retry_count=retry_count + 1,
                execution_identity=entry["execution_identity"],
                now=float(self.clock()),
                completed_request_ids=progress.completed_request_ids,
                completed_observation_ids=progress.completed_observation_ids,
                error=error,
                terminal_reason=None,
                pid=None,
            )
            _write_worker_manifest(Path(entry["worker_manifest_path"]), payload)
            self.events.append(
                {
                    "event": "worker_recovering",
                    "worker_id": worker_id,
                    "retry_count": retry_count + 1,
                    "reason": reason,
                    "at": _utc_now(float(self.clock())),
                }
            )
        else:
            entry["status"] = "failed"
            entry["terminal_reason"] = reason
            payload = _worker_payload(
                worker_id=worker_id,
                shard=self.shards[str(entry["shard_id"])],
                status="failed",
                output_path=entry["output_path"],
                worker_manifest_path=entry["worker_manifest_path"],
                stage=self.stage,
                retry_count=retry_count,
                execution_identity=entry["execution_identity"],
                now=float(self.clock()),
                completed_request_ids=progress.completed_request_ids,
                completed_observation_ids=progress.completed_observation_ids,
                error=error,
                terminal_reason=reason,
                pid=None,
            )
            _write_worker_manifest(Path(entry["worker_manifest_path"]), payload)
            self.events.append(
                {
                    "event": "worker_terminal_failure",
                    "worker_id": worker_id,
                    "retry_count": retry_count,
                    "reason": reason,
                    "at": _utc_now(float(self.clock())),
                }
            )
        self._refresh_state()

    def _observe_worker(self, runtime: WorkerRuntime) -> None:
        worker_id = runtime.worker_id
        entry = self.entries[worker_id]
        now = float(self.clock())
        return_code = runtime.process.poll()
        try:
            shard = runtime.shard
            progress = read_output_progress(entry["output_path"], shard)
        except InvalidCheckpointError as exc:
            if return_code is None:
                _terminate_process(runtime.process)
            runtime.log_handle.close() if runtime.log_handle is not None else None
            self.active.pop(worker_id, None)
            self._handle_worker_failure(worker_id, reason="malformed_output", error=str(exc), progress=OutputProgress(frozenset(), frozenset(), 0, 0))
            return
        signature = (len(progress.completed_request_ids), len(progress.completed_observation_ids), progress.byte_size)
        if signature != runtime.last_output_signature:
            runtime.last_output_signature = signature
            runtime.last_progress_clock = now
            self._record_progress(worker_id, progress, now=now)
        if runtime.manifest_owned_by_worker:
            try:
                manifest = load_worker_manifest(
                    entry["worker_manifest_path"],
                    shard=shard,
                    expected_identity=self._expected_identity(shard_manifest_sha256=entry["shard_manifest_sha256"]),
                    expected_output_path=entry["output_path"],
                )
                entry.update(self._entry_from_manifest(entry, manifest, progress))
            except InvalidCheckpointError as exc:
                if return_code is None:
                    _terminate_process(runtime.process)
                if runtime.log_handle is not None:
                    runtime.log_handle.close()
                self.active.pop(worker_id, None)
                self._handle_worker_failure(worker_id, reason="malformed_manifest", error=str(exc), progress=progress)
                return
        else:
            # Direct stage commands do not write our manifest.  The parent is
            # the sole writer for this adapter mode.
            entry["heartbeat_at"] = _utc_now(now)
            entry["heartbeat_at_epoch"] = now
            entry["status"] = "running"
            entry["completed_request_ids"] = sorted(progress.completed_request_ids)
            entry["completed_observation_ids"] = sorted(progress.completed_observation_ids)
            entry["completed_request_count"] = len(progress.completed_request_ids)
            entry["completed_observation_count"] = len(progress.completed_observation_ids)
            parent_manifest = _worker_payload(
                worker_id=worker_id,
                shard=shard,
                status="running",
                output_path=entry["output_path"],
                worker_manifest_path=entry["worker_manifest_path"],
                stage=self.stage,
                retry_count=int(entry.get("retry_count", 0)),
                execution_identity=entry["execution_identity"],
                now=now,
                completed_request_ids=progress.completed_request_ids,
                completed_observation_ids=progress.completed_observation_ids,
                pid=entry.get("pid"),
            )
            _write_worker_manifest(Path(entry["worker_manifest_path"]), parent_manifest)
        if return_code is None and self.idle_timeout_seconds > 0:
            heartbeat_epoch = entry.get("heartbeat_at_epoch")
            heartbeat_stale = heartbeat_epoch is None
            if heartbeat_epoch is not None:
                try:
                    heartbeat_stale = now - float(heartbeat_epoch) > self.idle_timeout_seconds
                except (TypeError, ValueError):
                    heartbeat_stale = True
            # With no output yet, the idle watchdog gives the clearer reason
            # for a worker that is still in model-loading/stall state.  A
            # stale heartbeat is a distinct failure once useful output has
            # already appeared; monitors still report both cases as stale.
            if heartbeat_stale and (progress.byte_size > 0 or progress.completed_observation_ids):
                _terminate_process(runtime.process)
                if runtime.log_handle is not None:
                    runtime.log_handle.close()
                self.active.pop(worker_id, None)
                self._handle_worker_failure(
                    worker_id,
                    reason="stale_heartbeat",
                    error="worker heartbeat is stale or missing",
                    progress=progress,
                )
                return
        if return_code is not None:
            if runtime.log_handle is not None:
                runtime.log_handle.close()
            self.active.pop(worker_id, None)
            if len(progress.completed_request_ids) == int(shard["expected_request_count"]):
                self._finish_worker(worker_id, successful=True, reason="process_exit_after_output")
            elif return_code == 0:
                self._finish_worker(worker_id, successful=False, reason="exit_before_completion", error="worker exited before completing its shard")
            else:
                self._finish_worker(worker_id, successful=False, reason="worker_exit", error=f"worker exited with code {return_code}")
            return
        # Idle is based on useful JSONL progress, not a heartbeat-only model
        # loading loop.  A heartbeat is still persisted and visible to the
        # monitor for stale-heartbeat diagnosis.
        if self.idle_timeout_seconds > 0 and now - runtime.last_progress_clock > self.idle_timeout_seconds:
            _terminate_process(runtime.process)
            if runtime.log_handle is not None:
                runtime.log_handle.close()
            self.active.pop(worker_id, None)
            self._handle_worker_failure(
                worker_id,
                reason="idle_timeout",
                error=f"no output progress for {now - runtime.last_progress_clock:.3f}s",
                progress=progress,
            )

    def _supersede_for_scale(self, worker_id: str) -> list[dict[str, Any]]:
        """Move only wholly-unstarted requests to new immutable shards.

        A shard containing a partially observed request is left intact; moving
        that request would create duplicate request ownership.  A wholly
        unstarted shard can be superseded without changing its immutable file,
        while its already completed output remains part of the campaign union.
        """

        entry = self.entries[worker_id]
        shard = self.shards[str(entry["shard_id"])]
        progress = read_output_progress(entry["output_path"], shard)
        if not progress.completed_observation_ids:
            pending = list(shard["requests"])
        else:
            completed_request_ids = set(progress.completed_request_ids)
            partial = any(
                request["request_id"] not in completed_request_ids
                and set(request["observation_ids"]).intersection(progress.completed_observation_ids)
                for request in shard["requests"]
            )
            if partial:
                return []
            pending = [request for request in shard["requests"] if request["request_id"] not in completed_request_ids]
        if not pending:
            return []
        if self.active.get(worker_id) is not None:
            return []
        entry["status"] = "superseded"
        entry["terminal_reason"] = "scaled_out"
        entry["error"] = None
        entry["completed_request_ids"] = sorted(progress.completed_request_ids)
        entry["completed_observation_ids"] = sorted(progress.completed_observation_ids)
        entry["completed_request_count"] = len(progress.completed_request_ids)
        entry["completed_observation_count"] = len(progress.completed_observation_ids)
        entry["observed_completed_request_ids"] = sorted(progress.completed_request_ids)
        entry["observed_completed_observation_ids"] = sorted(progress.completed_observation_ids)
        payload = _worker_payload(
            worker_id=worker_id,
            shard=shard,
            status="superseded",
            output_path=entry["output_path"],
            worker_manifest_path=entry["worker_manifest_path"],
            stage=self.stage,
            retry_count=int(entry.get("retry_count", 0)),
            execution_identity=entry["execution_identity"],
            now=float(self.clock()),
            completed_request_ids=progress.completed_request_ids,
            completed_observation_ids=progress.completed_observation_ids,
            terminal_reason="scaled_out",
            pid=None,
        )
        _write_worker_manifest(Path(entry["worker_manifest_path"]), payload)
        self.events.append({"event": "worker_superseded", "worker_id": worker_id, "at": _utc_now(float(self.clock()))})
        return pending

    def _maybe_scale(self) -> None:
        existing_slots = len(self.entries)
        if self.worker_count <= existing_slots:
            return
        additional = self.worker_count - existing_slots
        pending_pool: list[dict[str, Any]] = []
        for worker_id in sorted(self.entries):
            if len(pending_pool) >= 1_000_000:
                break
            entry = self.entries[worker_id]
            if entry.get("status") in {"complete", "failed", "budget_refused", "superseded"}:
                continue
            pending_pool.extend(self._supersede_for_scale(worker_id))
        shards_dir = self.output_dir / "shards"
        for offset in range(additional):
            if pending_pool:
                # Distribute pending work among newly requested slots without
                # changing the original shard files.
                remaining_slots = additional - offset
                size = (len(pending_pool) + remaining_slots - 1) // remaining_slots
                part = pending_pool[:size]
                pending_pool = pending_pool[size:]
            else:
                part = []
            slot = existing_slots + offset
            worker_id = f"worker_{slot:03d}"
            shard_id = f"shard_{slot:03d}"
            while worker_id in self.entries or shard_id in self.shards:
                slot += 1
                worker_id = f"worker_{slot:03d}"
                shard_id = f"shard_{slot:03d}"
            payload = _shard_manifest_payload(
                shard_id=shard_id,
                worker_id=worker_id,
                requests=part,
                parent_inventory_sha256=self.inventory_sha256,
                run_config_hash=self.run_config_hash,
                run_mode=self.run_mode,
                confirmatory=self.confirmatory,
                engineering_only=self.engineering_only,
                stage=self.stage,
                campaign_identity=self.campaign_identity,
            )
            path = shards_dir / f"{shard_id}.json"
            if path.exists():
                raise FileExistsError(f"Refusing to replace immutable shard manifest: {path}")
            atomic_write_json(path, payload)
            payload["manifest_path"] = str(path)
            payload["manifest_sha256"] = file_sha256(path)
            self.shards[shard_id] = payload
            entry = self._create_worker_entry(payload)
            self.entries[worker_id] = entry
            self.state.setdefault("shards", []).append(self._shard_entry(payload))
            self._write_initial_worker_manifest(entry, status="planned")
            self.events.append({"event": "worker_slot_added", "worker_id": worker_id, "at": _utc_now(float(self.clock()))})
        self._refresh_state()

    def _has_failed_worker(self) -> bool:
        return any(entry.get("status") in {"failed", "budget_refused"} for entry in self.entries.values())

    def _all_work_terminal(self) -> bool:
        return all(entry.get("status") in {"complete", "superseded"} for entry in self.entries.values())

    def _launch_pending(self) -> None:
        for worker_id in sorted(self.entries):
            if len(self.active) >= self.worker_count:
                break
            entry = self.entries[worker_id]
            if entry.get("status") not in {"planned", "recovering"}:
                continue
            if self.last_launch_clock is not None and self.stagger_seconds > 0:
                elapsed = float(self.clock()) - self.last_launch_clock
                if elapsed < self.stagger_seconds:
                    break
            try:
                self._start_worker(worker_id)
            except InvalidCheckpointError:
                # A corrupt or ahead-of-output checkpoint must become a
                # durable worker failure, rather than escaping before the
                # campaign can write its terminal report or run shutdown.
                try:
                    progress = read_output_progress(entry["output_path"], self.shards[str(entry["shard_id"])])
                except InvalidCheckpointError:
                    progress = OutputProgress(frozenset(), frozenset(), 0, 0)
                self._handle_worker_failure(
                    worker_id,
                    reason="malformed_checkpoint",
                    error="InvalidCheckpointError: worker could not be started",
                    progress=progress,
                )
            except (OSError, ParallelExecutorError) as exc:
                # Keep launch failures bounded and resumable while avoiding
                # command/environment details in durable manifests.
                try:
                    progress = read_output_progress(entry["output_path"], self.shards[str(entry["shard_id"])])
                except InvalidCheckpointError:
                    progress = OutputProgress(frozenset(), frozenset(), 0, 0)
                self._handle_worker_failure(
                    worker_id,
                    reason="worker_launch",
                    error=f"{type(exc).__name__}: worker could not be started",
                    progress=progress,
                )

    def _sleep_tick(self) -> None:
        duration = self.poll_interval_seconds
        if self.active and duration <= 0:
            duration = 0.001
        if not self.active and duration <= 0:
            duration = 0.0
        if duration > 0:
            self.sleeper(min(duration, 1.0))

    def _next_command(self) -> str:
        script = Path(__file__).resolve().parents[2] / "scripts" / "run_parallel_benchmark.py"
        argv = [
            sys.executable,
            str(script),
            "--campaign",
            self.campaign_spec,
            "--inventory",
            self.inventory_spec,
            "--stage",
            self.stage,
            "--worker-count",
            str(self.worker_count),
            "--max-retries",
            str(self.max_retries),
            "--idle-timeout",
            str(self.idle_timeout_seconds),
            "--output",
            str(self.output_dir),
            "--resume",
        ]
        if self.run_config_path is not None:
            argv.extend(["--run-config", str(self.run_config_path)])
        if self.fake_model:
            argv.append("--fake-model")
        elif self.adapter_kind == "gpu":
            argv.extend(["--adapter", "gpu"])
        elif self.adapter_kind == "python":
            argv.extend(["--adapter", str(self.adapter_value)])
        elif self.adapter_kind == "command":
            argv.append("--worker-command")
            argv.extend(str(item) for item in self.adapter_value)
        if self.shutdown_command is not None:
            argv.append("--shutdown-command")
            argv.extend(self.shutdown_command)
            argv.extend(["--shutdown-timeout", str(self.shutdown_timeout_seconds)])
        if self.run_mode != "test":
            argv.extend(["--run-mode", self.run_mode])
        if self.confirmatory:
            argv.append("--confirmatory")
        return shlex.join(argv)

    def _shutdown_hook_template(self, *, status: str) -> dict[str, Any]:
        if self.shutdown_command is None:
            skipped_reason = "disabled"
        elif self.dry_run:
            skipped_reason = "dry_run"
        elif status not in {"success", "failure"}:
            skipped_reason = "non_terminal_status"
        else:
            skipped_reason = None
        return {
            "configured": self.shutdown_command is not None,
            "attempted": False,
            "succeeded": False,
            "error": None,
            "skipped_reason": skipped_reason,
            "timeout_seconds": self.shutdown_timeout_seconds,
        }

    def _run_shutdown_hook(self, report: dict[str, Any]) -> dict[str, Any]:
        """Run an optional local argv hook after the first report is durable.

        The hook is deliberately not part of the worker adapter boundary.  It
        receives only fixed, non-secret tokens, never uses a shell, and sends
        both output streams to ``DEVNULL`` so a shutdown tool cannot leak
        credentials through campaign output.  A hook failure is metadata only:
        the scientific report status is never changed.
        """

        hook = dict(report["shutdown_hook"])
        if not hook["configured"] or report["status"] not in {"success", "failure"} or self.dry_run:
            return report

        hook["attempted"] = True
        hook["skipped_reason"] = None
        # Persist the attempted state before launching the command.  If the
        # process is stopped by the hook itself, the durable report still
        # records that the hook was attempted.
        report["shutdown_hook"] = hook
        try:
            atomic_write_json(self.terminal_report_path, report)
        except OSError:
            # The initial report is already durable.  Do not invoke a shutdown
            # command when its attempted state cannot be persisted.
            hook["attempted"] = False
            hook["error"] = "could not persist shutdown hook attempt"
            report["shutdown_hook"] = hook
            return report

        context = {
            "status": str(report["status"]),
            "reason": str(report.get("terminal_reason") or ""),
            "campaign": self.campaign_id,
            "output": str(self.output_dir),
            "terminal_report": str(self.terminal_report_path),
        }
        try:
            result = run_terminal_shutdown(
                self.shutdown_command,
                context=context,
                timeout_seconds=self.shutdown_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - defensive lifecycle boundary
            # Report only the exception class; never include argv or process
            # output, either of which may contain a credential.
            hook["error"] = type(exc).__name__
        else:
            hook["succeeded"] = result.get("status") == "succeeded"
            hook["status"] = result.get("status")
            hook["return_code"] = result.get("return_code")
            hook["argv_sha256"] = result.get("argv_sha256")
            hook["executable"] = result.get("executable")
            hook["argument_count"] = result.get("argument_count")
            if not hook["succeeded"]:
                hook["error"] = result.get("error") or "shutdown command did not succeed"

        report["shutdown_hook"] = hook
        try:
            atomic_write_json(self.terminal_report_path, report)
        except OSError:
            # Keep the in-memory result useful to callers while retaining the
            # already durable scientific status and pre-launch hook record.
            hook["report_update_error"] = "could not persist shutdown hook result"
            report["shutdown_hook"] = hook
        return report

    def _terminal_report(self, *, status: str, reason: str | None, error: str | None = None) -> dict[str, Any]:
        now = float(self.clock())
        self._refresh_state(persist=False)
        continuation = {
            "can_resume": status in {"failure", "dry_run"},
            "resume_command": self._next_command(),
            "next_command": self._next_command() if status == "failure" else None,
            "reason": "resume after fixing the terminal worker or budget condition" if status == "failure" else "no continuation required",
        }
        self.state.update(
            {
                "status": status,
                "terminal_reason": reason,
                "error": error,
                "ended_at": _utc_now(now),
                "duration_seconds": max(0.0, now - self.started_clock),
                "continuation": continuation,
            }
        )
        self.state["budget"] = self.budget.snapshot(len(self.active))
        self._write_state()
        report = {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": TERMINAL_REPORT_TYPE,
            "campaign_id": self.campaign_id,
            "campaign_identity": self.campaign_identity,
            "status": status,
            "terminal_reason": reason,
            "error": error,
            "started_at": self.state.get("created_at"),
            "ended_at": _utc_now(now),
            "duration_seconds": max(0.0, now - self.started_clock),
            "stage": self.stage,
            "run_mode": self.run_mode,
            "confirmatory": self.confirmatory,
            "engineering_only": self.engineering_only,
            "worker_count": self.worker_count,
            "physical_shard_count": self.state.get("physical_shard_count"),
            "adapter": self.adapter_identity,
            "progress": self.state.get("progress", {}),
            "budget": self.state.get("budget", {}),
            "workers": self.state.get("workers", []),
            "shards": self.state.get("shards", []),
            "output_paths": self.state.get("output_paths", []),
            "continuation": continuation,
            "external_calls": {"runpod": False, "model_loading": self.fake_model is False and self.adapter_kind != "fake"},
            "execution_topology": self.execution_topology,
            "shutdown_hook": self._shutdown_hook_template(status=status),
        }
        atomic_write_json(self.terminal_report_path, report)
        return self._run_shutdown_hook(report)

    def _mark_all_active_failed(self, *, reason: str) -> None:
        for worker_id, runtime in list(self.active.items()):
            _terminate_process(runtime.process)
            if runtime.log_handle is not None:
                runtime.log_handle.close()
            self.active.pop(worker_id, None)
            try:
                progress = read_output_progress(self.entries[worker_id]["output_path"], runtime.shard)
            except InvalidCheckpointError:
                progress = OutputProgress(frozenset(), frozenset(), 0, 0)
            self._handle_worker_failure(worker_id, reason=reason, error=reason, progress=progress)

    def run(self) -> dict[str, Any]:
        """Prepare/resume and execute, returning the durable terminal summary."""

        self._prepare()
        if self.resume:
            self._prepare_resumable_failures()
            self._maybe_scale()
        self.state["worker_count"] = self.worker_count
        if self.dry_run:
            for worker_id, entry in self.entries.items():
                if entry.get("status") in {"planned", "recovering"}:
                    entry["status"] = "dry_run"
                    shard = self.shards[str(entry["shard_id"])]
                    payload = _worker_payload(
                        worker_id=worker_id,
                        shard=shard,
                        status="dry_run",
                        output_path=entry["output_path"],
                        worker_manifest_path=entry["worker_manifest_path"],
                        stage=self.stage,
                        retry_count=int(entry.get("retry_count", 0)),
                        execution_identity=entry["execution_identity"],
                        now=float(self.clock()),
                    )
                    _write_worker_manifest(Path(entry["worker_manifest_path"]), payload)
            report = self._terminal_report(status="dry_run", reason="dry_run_no_workers_launched")
            return report
        if self.state.get("status") == "success" and self._all_work_terminal():
            return self._terminal_report(status="success", reason="already_complete")
        self.state["status"] = "running"
        self._refresh_state()
        while True:
            self.budget.update(len(self.active))
            for runtime in list(self.active.values()):
                self._observe_worker(runtime)
            self._refresh_state()
            if self._has_failed_worker():
                self._mark_all_active_failed(reason="campaign_terminal_worker_failure")
                return self._terminal_report(status="failure", reason="worker_terminal_failure", error="one or more workers exhausted recovery")
            if self._all_work_terminal() and not self.active:
                return self._terminal_report(status="success", reason="all_workers_complete")
            self._launch_pending()
            if self._has_failed_worker():
                if any(entry.get("status") == "failed" for entry in self.entries.values()):
                    self._mark_all_active_failed(reason="campaign_terminal_worker_failure")
                    return self._terminal_report(
                        status="failure",
                        reason="worker_terminal_failure",
                        error="one or more workers exhausted recovery",
                    )
                self._mark_all_active_failed(reason="campaign_terminal_budget_failure")
                return self._terminal_report(
                    status="failure",
                    reason="budget_cutoff",
                    error="one or more worker launches were refused",
                )
            if self._all_work_terminal() and not self.active:
                return self._terminal_report(status="success", reason="all_workers_complete")
            if not self.active and not any(entry.get("status") in {"planned", "recovering"} for entry in self.entries.values()):
                return self._terminal_report(status="failure", reason="no_recoverable_workers", error="campaign has pending work without a recoverable worker")
            self._sleep_tick()


def run_parallel_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Convenience function for tests and future stage wrappers."""

    return ParallelExecutor(**kwargs).run()


def inspect_campaign(output: str | Path, *, now: float | None = None) -> dict[str, Any]:
    """Read a campaign and all worker manifests for the monitor CLI."""

    root = Path(output).expanduser().resolve()
    state_path = root / CAMPAIGN_STATE_FILENAME
    state = _load_json_object(state_path, label="campaign state")
    if state.get("manifest_type") != CAMPAIGN_STATE_TYPE:
        raise InvalidCheckpointError("Not a parallel campaign state file.")
    current = time.time() if now is None else float(now)
    workers = []
    stale_workers = []
    for entry in state.get("workers", []):
        item = dict(entry)
        manifest_path = Path(str(item.get("worker_manifest_path", "")))
        shard_path = Path(str(item.get("shard_manifest_path", "")))
        if not manifest_path.is_file() or not shard_path.is_file():
            item["monitor_error"] = "missing manifest"
            workers.append(item)
            continue
        try:
            shard = load_shard_manifest(shard_path)
            manifest = load_worker_manifest(manifest_path, shard=shard, expected_output_path=item.get("output_path"))
            progress = read_output_progress(item.get("output_path"), shard)
            item.update(
                {
                    "status": manifest["status"],
                    "pid": manifest.get("pid"),
                    "heartbeat_at": manifest.get("heartbeat_at"),
                    "heartbeat_at_epoch": manifest.get("heartbeat_at_epoch"),
                    "completed_request_ids": sorted(progress.completed_request_ids),
                    "completed_observation_ids": sorted(progress.completed_observation_ids),
                    "completed_request_count": len(progress.completed_request_ids),
                    "completed_observation_count": len(progress.completed_observation_ids),
                    "output_size": progress.byte_size,
                }
            )
            last_output = item.get("last_progress", {}).get("at") if isinstance(item.get("last_progress"), Mapping) else None
            heartbeat_epoch = manifest.get("heartbeat_at_epoch")
            timeout = float(state.get("idle_timeout_seconds", 0) or 0)
            if timeout > 0 and manifest.get("status") in {"running", "starting"}:
                reference = float(heartbeat_epoch) if heartbeat_epoch is not None else 0.0
                if current - reference > timeout and not progress.completed_request_ids:
                    stale_workers.append(str(item.get("worker_id")))
            if last_output:
                item["last_output_progress_at"] = last_output
        except InvalidCheckpointError as exc:
            item["monitor_error"] = str(exc)
        workers.append(item)
    completed_requests = len({item for worker in workers for item in worker.get("completed_request_ids", [])})
    completed_observations = len({item for worker in workers for item in worker.get("completed_observation_ids", [])})
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "parallel_campaign_monitor_snapshot",
        "campaign_id": state.get("campaign_id"),
        "campaign_identity": state.get("campaign_identity"),
        "status": state.get("status"),
        "terminal_reason": state.get("terminal_reason"),
        "worker_count": state.get("worker_count"),
        "progress": {
            **dict(state.get("progress", {})),
            "completed_request_count_observed": completed_requests,
            "completed_observation_count_observed": completed_observations,
        },
        "stale_workers": stale_workers,
        "workers": workers,
        "budget": state.get("budget", {}),
        "terminal_report_path": str(root / TERMINAL_REPORT_FILENAME),
    }


def worker_cli_main(argv: Sequence[str] | None = None) -> int:
    parser = _worker_parser()
    args = parser.parse_args(argv)
    return run_worker_process(
        shard_manifest_path=args.shard_manifest,
        worker_manifest_path=args.worker_manifest,
        output_path=args.output,
        stage=args.stage,
        adapter_spec=args.adapter,
        fake_model=args.fake_model,
        stall_seconds=args.stall,
        stall_after=args.stall_after,
        crash_after=args.crash_after,
        crash_once=args.crash_once,
    )


def _worker_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Internal parallel benchmark worker.")
    parser.add_argument("--worker-process", action="store_true")
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--worker-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--adapter", default="fake")
    parser.add_argument("--fake-model", action="store_true")
    parser.add_argument("--stall", type=float, default=0.0)
    parser.add_argument("--stall-after", type=int, default=None)
    parser.add_argument("--crash-after", type=int, default=None)
    parser.add_argument("--crash-once", action="store_true")
    return parser


def module_main(argv: Sequence[str] | None = None) -> int:
    # This module is used only as the internal worker target.  Public campaign
    # argument parsing lives in scripts/run_parallel_benchmark.py.
    return worker_cli_main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(module_main())
