"""Model-independent concurrency benchmarking for a frozen workload.

This module measures execution capacity only.  It never reads benchmark effect
sizes and it does not choose a model, layer, sign, dose, or scientific subset.
The runner is injected so the controller can be exercised with a deterministic
fixture and can later wrap a real executor without changing the rollout rule.

The rollout is deliberately conservative:

* one worker is the required baseline;
* three workers are evaluated as the first candidate;
* four workers are evaluated only after three is a quality-passing material
  improvement over one;
* five workers are evaluated only after four is stable and a material
  improvement over three.

Selection uses valid aggregate requests per dollar, subject to operational
failure, retry, memory, cost, and deterministic-output-identity guards.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .manifests import canonical_hash


CONCURRENCY_SCHEMA_VERSION = "0.1.0"
SUPPORTED_WORKER_COUNTS = (1, 3, 4, 5)
DEFAULT_WORKER_COUNTS = (1, 3, 4)


def _jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy of a fixture value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Benchmark inputs must contain finite numeric values.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_mapping"):
        return _jsonable(value.to_mapping())
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _canonical_hash(value: Any) -> str:
    return canonical_hash(_jsonable(value))


def _as_nonnegative_int(value: Any, *, field_name: str, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return int(numeric)


def _as_positive_float(value: Any, *, field_name: str, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    return numeric


@dataclass(frozen=True)
class FrozenWorkload:
    """Canonical request records used for every worker-count measurement."""

    records: tuple[Any, ...]
    request_ids: tuple[str, ...]
    expected_output_identities: tuple[tuple[str, str], ...] = ()
    workload_id: str = ""

    @property
    def identity_sha256(self) -> str:
        return _canonical_hash(
            {
                "request_ids": self.request_ids,
                "records": self.records,
                "expected_output_identities": self.expected_output_identities,
            }
        )

    @property
    def expected_identity_map(self) -> dict[str, str]:
        return dict(self.expected_output_identities)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id or self.identity_sha256[:16],
            "request_ids": list(self.request_ids),
            "records": _jsonable(self.records),
            "expected_output_identities": {
                key: value for key, value in self.expected_output_identities
            },
            "identity_sha256": self.identity_sha256,
        }


def freeze_workload(workload: FrozenWorkload | Mapping[str, Any] | Iterable[Any]) -> FrozenWorkload:
    """Freeze request order and identity before any worker is run.

    A mapping may contain ``records`` (or ``requests``/``items``), an optional
    ``workload_id``, and optional per-record ``expected_output_identity``
    values.  For a plain iterable, request IDs are read from
    ``request_id``, ``id``, or ``record_id`` and otherwise receive deterministic
    index IDs.
    """

    if isinstance(workload, FrozenWorkload):
        return workload
    workload_id = ""
    expected: dict[str, str] = {}
    if isinstance(workload, Mapping):
        workload_id = str(workload.get("workload_id") or "").strip()
        raw_records = next(
            (workload[key] for key in ("records", "requests", "items") if key in workload),
            None,
        )
        if raw_records is None:
            raw_records = [workload]
        explicit_expected = workload.get("expected_output_identities")
        if isinstance(explicit_expected, Mapping):
            expected.update(
                (str(key), _canonical_hash(value))
                for key, value in explicit_expected.items()
            )
    else:
        raw_records = workload
    if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Iterable):
        raw_records = [raw_records]
    records = tuple(_jsonable(record) for record in raw_records)
    if not records:
        raise ValueError("The frozen concurrency workload must not be empty.")
    request_ids: list[str] = []
    for index, record in enumerate(records):
        candidate = record.get("request_id", record.get("id", record.get("record_id"))) if isinstance(record, Mapping) else None
        request_id = str(candidate).strip() if candidate is not None else f"request_{index:06d}"
        if not request_id:
            raise ValueError(f"Workload record {index} has an empty request ID.")
        if request_id in request_ids:
            raise ValueError(f"Duplicate workload request_id={request_id!r}.")
        request_ids.append(request_id)
        if isinstance(record, Mapping):
            identity = record.get("expected_output_identity")
            if identity is None:
                identity = record.get("expected_output_hash")
            if identity is not None:
                expected[request_id] = _canonical_hash(identity)
    frozen = FrozenWorkload(
        records=records,
        request_ids=tuple(request_ids),
        expected_output_identities=tuple(sorted(expected.items())),
        workload_id=workload_id,
    )
    return frozen


@dataclass(frozen=True)
class ConcurrencyMeasurement:
    """Runner/provider output accepted by :func:`benchmark_concurrency`."""

    elapsed_seconds: float | None = None
    requested_requests: int | None = None
    valid_requests: int | None = None
    observations: int | None = None
    failures: int = 0
    retries: int = 0
    worker_metrics: tuple[Mapping[str, Any], ...] = ()
    output_identities: Any = None
    peak_vram_gb: float | None = None
    gpu_utilization_pct: float | None = None
    hourly_rate: float | None = None
    stable: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConcurrencyPolicy:
    """Operational thresholds and staged worker-count rollout settings."""

    worker_counts: tuple[int, ...] = DEFAULT_WORKER_COUNTS
    include_five_worker: bool = False
    material_improvement: float = 0.10
    max_failure_rate: float = 0.05
    max_retry_rate: float = 0.10
    max_peak_vram_gb: float | None = None
    max_cost_per_request: float | None = None
    hourly_rate: float | None = None
    require_output_identity_match: bool = True

    def __post_init__(self) -> None:
        counts = tuple(int(value) for value in self.worker_counts)
        if self.include_five_worker and 5 not in counts:
            counts = (*counts, 5)
        if counts != tuple(sorted(set(counts))):
            raise ValueError("worker_counts must be sorted and unique.")
        if any(value not in SUPPORTED_WORKER_COUNTS for value in counts):
            raise ValueError(f"worker_counts must use only {SUPPORTED_WORKER_COUNTS!r}.")
        if not set(DEFAULT_WORKER_COUNTS).issubset(counts):
            raise ValueError("worker_counts must include the 1, 3, and 4 worker rollout stages.")
        object.__setattr__(self, "worker_counts", counts)
        for field_name in ("material_improvement", "max_failure_rate", "max_retry_rate"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0 or (field_name.endswith("rate") and value > 1):
                raise ValueError(f"{field_name} must be a finite non-negative threshold.")
        for field_name in ("max_peak_vram_gb", "max_cost_per_request", "hourly_rate"):
            value = getattr(self, field_name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{field_name} must be finite and greater than zero when supplied.")
        if not isinstance(self.require_output_identity_match, bool):
            raise ValueError("require_output_identity_match must be boolean.")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ConcurrencyPolicy":
        if raw is None:
            return cls()
        payload = dict(raw)
        if "min_material_improvement" in payload and "material_improvement" not in payload:
            payload["material_improvement"] = payload.pop("min_material_improvement")
        if "hourly_rate_usd" in payload and "hourly_rate" not in payload:
            payload["hourly_rate"] = payload.pop("hourly_rate_usd")
        if "max_estimated_cost_per_request" in payload and "max_cost_per_request" not in payload:
            payload["max_cost_per_request"] = payload.pop("max_estimated_cost_per_request")
        if "worker_counts" in payload:
            payload["worker_counts"] = tuple(int(value) for value in payload["worker_counts"])
        return cls(**payload)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "worker_counts": list(self.worker_counts),
            "include_five_worker": self.include_five_worker,
            "material_improvement": self.material_improvement,
            "max_failure_rate": self.max_failure_rate,
            "max_retry_rate": self.max_retry_rate,
            "max_peak_vram_gb": self.max_peak_vram_gb,
            "max_cost_per_request": self.max_cost_per_request,
            "hourly_rate": self.hourly_rate,
            "require_output_identity_match": self.require_output_identity_match,
        }


def _field(payload: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return default


def _identity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("output_identity", "identity", "output_hash", "hash", "output_text", "output"):
            if key in value:
                return value[key]
    return value


def _identity_map(value: Any, workload: FrozenWorkload) -> dict[str, str] | None:
    """Normalize output identities to request ID → digest without raw output."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        if "output_identities" in value:
            return _identity_map(value["output_identities"], workload)
        if "outputs" in value:
            return _identity_map(value["outputs"], workload)
        if "request_id" in value:
            request_id = str(value["request_id"])
            return {request_id: _canonical_hash(_identity_value(value))}
        return {str(key): _canonical_hash(_identity_value(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        result: dict[str, str] = {}
        for index, item in enumerate(value):
            if isinstance(item, Mapping) and item.get("request_id") is not None:
                request_id = str(item["request_id"])
            elif index < len(workload.request_ids):
                request_id = workload.request_ids[index]
            else:
                request_id = f"index_{index:06d}"
            if request_id in result:
                raise ValueError(f"Duplicate output identity request_id={request_id!r}.")
            result[request_id] = _canonical_hash(_identity_value(item))
        return result
    return {"__aggregate__": _canonical_hash(_identity_value(value))}


def _measurement_from_raw(
    raw: Any,
    workload: FrozenWorkload,
    *,
    elapsed_seconds: float,
) -> ConcurrencyMeasurement:
    if isinstance(raw, ConcurrencyMeasurement):
        measurement = raw
    elif isinstance(raw, Mapping):
        payload = dict(raw.get("measurement", raw)) if isinstance(raw.get("measurement", raw), Mapping) else dict(raw)
        worker_metrics = _field(payload, ("worker_metrics", "per_worker", "workers"), ())
        if isinstance(worker_metrics, Mapping):
            worker_metrics = [dict(value, worker_id=key) if isinstance(value, Mapping) else {"worker_id": key, "valid_requests": value} for key, value in worker_metrics.items()]
        if not isinstance(worker_metrics, (list, tuple)):
            worker_metrics = ()
        valid = _field(payload, ("valid_requests", "valid_request_count", "completed_valid_requests"))
        observations = _field(payload, ("observations", "observation_count", "completed_observations"))
        if isinstance(observations, (list, tuple, Mapping)):
            observations = len(observations)
        output_identities = _field(
            payload,
            ("output_identities", "output_identity_hashes", "outputs", "output_identity"),
        )
        measurement = ConcurrencyMeasurement(
            elapsed_seconds=_field(
                payload,
                ("elapsed_seconds", "elapsed", "elapsed_time_seconds", "duration_seconds"),
            ),
            requested_requests=_field(payload, ("requested_requests", "total_requests", "request_count"), len(workload.request_ids)),
            valid_requests=valid,
            observations=observations,
            failures=_field(payload, ("failures", "failed_requests", "failure_count"), 0),
            retries=_field(payload, ("retries", "retry_count"), 0),
            worker_metrics=tuple(worker_metrics),
            output_identities=output_identities,
            peak_vram_gb=_field(payload, ("peak_vram_gb", "peak_vram", "peak_vram_gib", "max_vram_gb")),
            gpu_utilization_pct=_field(payload, ("gpu_utilization_pct", "gpu_utilization_percent", "gpu_utilization")),
            hourly_rate=_field(payload, ("hourly_rate", "hourly_rate_usd", "usd_per_hour", "hourly_cost")),
            stable=_field(payload, ("stable", "is_stable")),
            metadata=_field(payload, ("metadata",), {}),
        )
    elif isinstance(raw, (list, tuple)):
        output_identities: list[Any] = []
        failures = 0
        observations = 0
        for item in raw:
            if isinstance(item, Mapping):
                valid = item.get("valid", item.get("valid_request", item.get("ok", True)))
                if not valid:
                    failures += 1
                observations += int(item.get("observations", item.get("observation_count", 1)))
                output_identities.append(item)
            else:
                observations += 1
                output_identities.append(item)
        measurement = ConcurrencyMeasurement(
            requested_requests=len(workload.request_ids),
            valid_requests=len(raw) - failures,
            observations=observations,
            failures=failures,
            output_identities=output_identities,
        )
    else:
        raise ValueError("A concurrency runner must return a measurement mapping, dataclass, or output list.")
    requested = _as_nonnegative_int(measurement.requested_requests, field_name="requested_requests", default=len(workload.request_ids))
    valid = _as_nonnegative_int(measurement.valid_requests, field_name="valid_requests", default=max(0, requested - int(measurement.failures)))
    observations = _as_nonnegative_int(measurement.observations, field_name="observations", default=valid)
    failures = _as_nonnegative_int(measurement.failures, field_name="failures", default=0)
    retries = _as_nonnegative_int(measurement.retries, field_name="retries", default=0)
    assert requested is not None and valid is not None and observations is not None and failures is not None and retries is not None
    if valid > requested:
        raise ValueError("valid_requests cannot exceed requested_requests.")
    if failures > requested:
        raise ValueError("failures cannot exceed requested_requests.")
    if requested != len(workload.request_ids):
        raise ValueError(
            "A concurrency run must report the complete frozen workload: "
            f"expected {len(workload.request_ids)} requests, found {requested}."
        )
    elapsed = measurement.elapsed_seconds if measurement.elapsed_seconds is not None else elapsed_seconds
    elapsed = _as_positive_float(elapsed, field_name="elapsed_seconds", allow_none=False)
    if elapsed is None or elapsed <= 0:
        raise ValueError("elapsed_seconds must be finite and greater than zero.")
    for field_name in ("peak_vram_gb", "gpu_utilization_pct", "hourly_rate"):
        value = getattr(measurement, field_name)
        if value is not None:
            value = _as_positive_float(value, field_name=field_name)
            if field_name != "hourly_rate" and value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
            if field_name == "hourly_rate" and value is not None and value <= 0:
                raise ValueError("hourly_rate must be greater than zero when supplied.")
            if field_name == "gpu_utilization_pct" and value is not None and value > 100:
                raise ValueError("gpu_utilization_pct must be at most 100.")
    return ConcurrencyMeasurement(
        elapsed_seconds=elapsed,
        requested_requests=requested,
        valid_requests=valid,
        observations=observations,
        failures=failures,
        retries=retries,
        worker_metrics=measurement.worker_metrics,
        output_identities=measurement.output_identities,
        peak_vram_gb=measurement.peak_vram_gb,
        gpu_utilization_pct=measurement.gpu_utilization_pct,
        hourly_rate=measurement.hourly_rate,
        stable=measurement.stable,
        metadata=dict(measurement.metadata),
    )


def _invoke_callable(function: Callable[..., Any], workload: FrozenWorkload, worker_count: int) -> Any:
    """Call common two-argument runner/provider shapes without hiding errors."""

    try:
        signature = inspect.signature(function)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) == 1:
            return function(worker_count)
        if positional and positional[0].name in {"worker_count", "workers", "n_workers"}:
            return function(worker_count, workload)
    except (TypeError, ValueError):
        pass
    return function(workload, worker_count)


def _invoke_provider(
    provider: Callable[..., Any],
    raw: Any,
    workload: FrozenWorkload,
    worker_count: int,
) -> Any:
    try:
        signature = inspect.signature(provider)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if not positional:
            return provider()
        if len(positional) == 1:
            return provider(raw)
        if len(positional) == 2:
            if positional[0].name in {"worker_count", "workers", "n_workers"}:
                return provider(worker_count, workload)
            return provider(raw, workload)
    except (TypeError, ValueError):
        pass
    return provider(raw, workload, worker_count)


def _worker_metrics(measurement: ConcurrencyMeasurement, worker_count: int) -> list[dict[str, Any]]:
    if measurement.worker_metrics:
        result = []
        for index, raw in enumerate(measurement.worker_metrics):
            payload = dict(raw)
            elapsed = float(payload.get("elapsed_seconds", measurement.elapsed_seconds or 0.0))
            if not math.isfinite(elapsed) or elapsed <= 0:
                elapsed = float(measurement.elapsed_seconds or 1.0)
            valid = int(payload.get("valid_requests", payload.get("completed_valid_requests", 0)))
            observations = int(payload.get("observations", payload.get("observation_count", 0)))
            result.append(
                {
                    "worker_id": str(payload.get("worker_id", index)),
                    "valid_requests": valid,
                    "observations": observations,
                    "elapsed_seconds": elapsed,
                    "valid_requests_per_minute": valid * 60.0 / elapsed,
                    "observations_per_minute": observations * 60.0 / elapsed,
                }
            )
        return result
    elapsed = float(measurement.elapsed_seconds or 1.0)
    return [
        {
            "worker_id": str(index),
            "valid_requests": None,
            "observations": None,
            "elapsed_seconds": elapsed,
            "valid_requests_per_minute": None,
            "observations_per_minute": None,
            "aggregate_divided_by_worker_count": True,
        }
        for index in range(worker_count)
    ]


def _run_record(
    measurement: ConcurrencyMeasurement,
    workload: FrozenWorkload,
    worker_count: int,
    *,
    policy: ConcurrencyPolicy,
    runner_error: str | None = None,
) -> dict[str, Any]:
    requested = int(measurement.requested_requests or 0)
    valid = int(measurement.valid_requests or 0)
    observations = int(measurement.observations or 0)
    failures = int(measurement.failures)
    retries = int(measurement.retries)
    elapsed = float(measurement.elapsed_seconds or 0.0)
    hourly_rate = policy.hourly_rate if policy.hourly_rate is not None else measurement.hourly_rate
    estimated_cost = None if hourly_rate is None else elapsed / 3600.0 * float(hourly_rate)
    valid_per_dollar = None if estimated_cost is None or estimated_cost <= 0 else valid / estimated_cost
    cost_per_request = None if valid <= 0 or estimated_cost is None else estimated_cost / valid
    aggregate_valid_per_minute = None if elapsed <= 0 else valid * 60.0 / elapsed
    aggregate_observations_per_minute = None if elapsed <= 0 else observations * 60.0 / elapsed
    identity_map = _identity_map(measurement.output_identities, workload)
    failure_rate = failures / requested if requested else 1.0
    retry_rate = retries / requested if requested else 1.0
    return {
        "worker_count": worker_count,
        "status": "error" if runner_error else "completed",
        "runner_error": runner_error,
        "requested_requests": requested,
        "valid_requests": valid,
        "observations": observations,
        "failures": failures,
        "retries": retries,
        "failure_rate": failure_rate,
        "retry_rate": retry_rate,
        "elapsed_seconds": elapsed,
        "valid_requests_per_minute": aggregate_valid_per_minute,
        "observations_per_minute": aggregate_observations_per_minute,
        "throughput": {
            "aggregate": {
                "valid_requests_per_minute": aggregate_valid_per_minute,
                "observations_per_minute": aggregate_observations_per_minute,
                "valid_requests_per_dollar": valid_per_dollar,
            },
            "per_worker": {
                "valid_requests_per_minute": None if aggregate_valid_per_minute is None else aggregate_valid_per_minute / worker_count,
                "observations_per_minute": None if aggregate_observations_per_minute is None else aggregate_observations_per_minute / worker_count,
            },
            "workers": _worker_metrics(measurement, worker_count),
        },
        "hourly_rate": hourly_rate,
        "hourly_rate_source": "policy" if policy.hourly_rate is not None else ("measurement" if measurement.hourly_rate is not None else "missing"),
        "estimated_cost": estimated_cost,
        "estimated_cost_per_request": cost_per_request,
        "estimated_cost_per_valid_request": cost_per_request,
        "cost_per_request": cost_per_request,
        "peak_vram_gb": measurement.peak_vram_gb,
        "gpu_utilization_pct": measurement.gpu_utilization_pct,
        "gpu_utilization": measurement.gpu_utilization_pct,
        "stable": measurement.stable,
        "output_identity": {
            "available": identity_map is not None,
            "request_count": len(identity_map or {}),
            "identities_sha256": None if identity_map is None else _canonical_hash(identity_map),
        },
        "output_identity_map": identity_map,
        "metadata": _jsonable(measurement.metadata),
    }


def _identity_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    workload: FrozenWorkload,
) -> dict[str, Any]:
    base_map = baseline.get("output_identity_map")
    candidate_map = candidate.get("output_identity_map")
    expected = workload.expected_identity_map
    if base_map is None and candidate_map is None and not expected:
        return {"status": "not_available", "match": None, "worker_count": candidate["worker_count"]}
    if base_map is None or candidate_map is None:
        return {
            "status": "unavailable_for_one_or_more_runs",
            "match": False,
            "worker_count": candidate["worker_count"],
            "missing_baseline": base_map is None,
            "missing_candidate": candidate_map is None,
        }
    keys = sorted(set(base_map) | set(candidate_map))
    differing = [key for key in keys if base_map.get(key) != candidate_map.get(key)]
    expected_differing = [key for key in sorted(expected) if base_map.get(key) != expected[key] or candidate_map.get(key) != expected[key]]
    missing = sorted(set(base_map) - set(candidate_map))
    unexpected = sorted(set(candidate_map) - set(base_map))
    match = not differing and not expected_differing
    return {
        "status": "match" if match else "mismatch",
        "match": match,
        "worker_count": candidate["worker_count"],
        "baseline_worker_count": baseline["worker_count"],
        "differing_request_ids": differing,
        "expected_differing_request_ids": expected_differing,
        "missing_request_ids": missing,
        "unexpected_request_ids": unexpected,
        "baseline_identities_sha256": baseline["output_identity"]["identities_sha256"],
        "candidate_identities_sha256": candidate["output_identity"]["identities_sha256"],
    }


def _quality_reasons(
    run: Mapping[str, Any],
    *,
    policy: ConcurrencyPolicy,
    identity_status: str,
) -> list[str]:
    reasons: list[str] = []
    if run.get("status") != "completed":
        reasons.append("runner_failed")
    if int(run.get("valid_requests", 0)) <= 0:
        reasons.append("no_valid_requests")
    if run.get("hourly_rate") is None or run.get("estimated_cost") is None:
        reasons.append("hourly_rate_missing")
    if float(run.get("failure_rate", 1.0)) > policy.max_failure_rate:
        reasons.append("failure_rate_threshold")
    if float(run.get("retry_rate", 1.0)) > policy.max_retry_rate:
        reasons.append("retry_rate_threshold")
    peak_vram = run.get("peak_vram_gb")
    if policy.max_peak_vram_gb is not None and (peak_vram is None or float(peak_vram) > policy.max_peak_vram_gb):
        reasons.append("peak_vram_threshold")
    cost_per_request = run.get("estimated_cost_per_valid_request")
    if policy.max_cost_per_request is not None and (cost_per_request is None or float(cost_per_request) > policy.max_cost_per_request):
        reasons.append("cost_per_request_threshold")
    if policy.require_output_identity_match and identity_status in {"mismatch", "unavailable_for_one_or_more_runs"}:
        reasons.append("output_identity_mismatch")
    return reasons


def _materially_better(candidate: Mapping[str, Any], reference: Mapping[str, Any], threshold: float) -> bool:
    candidate_metric = candidate.get("throughput", {}).get("aggregate", {}).get("valid_requests_per_dollar")
    reference_metric = reference.get("throughput", {}).get("aggregate", {}).get("valid_requests_per_dollar")
    if candidate_metric is None or reference_metric is None:
        return False
    candidate_metric = float(candidate_metric)
    reference_metric = float(reference_metric)
    if reference_metric <= 0:
        return candidate_metric > reference_metric
    return candidate_metric >= reference_metric * (1.0 + threshold)


def benchmark_concurrency(
    workload: FrozenWorkload | Mapping[str, Any] | Iterable[Any],
    runner: Callable[..., Any] | None = None,
    *,
    measurement_provider: Callable[..., Any] | None = None,
    policy: ConcurrencyPolicy | Mapping[str, Any] | None = None,
    expected_request_count: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run the staged concurrency benchmark and return a frozen report.

    ``runner`` is called with ``(frozen_workload, worker_count)`` and may
    return a :class:`ConcurrencyMeasurement`, a mapping, or an output list.
    ``measurement_provider`` can translate a runner result into one of those
    forms; a one-argument provider receives the raw runner result, while a
    three-argument provider receives ``(raw, workload, worker_count)``.
    """

    frozen = freeze_workload(workload)
    if expected_request_count is not None:
        expected_count = _as_nonnegative_int(
            expected_request_count,
            field_name="expected_request_count",
        )
        if expected_count is None or expected_count <= 0:
            raise ValueError("expected_request_count must be a positive integer.")
        if len(frozen.request_ids) != expected_count:
            raise ValueError(
                "The concurrency benchmark requires the registered frozen workload size: "
                f"expected {expected_count} requests, found {len(frozen.request_ids)}."
            )
    if runner is None and measurement_provider is None:
        raise ValueError("Provide a runner or measurement_provider.")
    if runner is None:
        runner = measurement_provider
        measurement_provider = None
    resolved_policy = policy if isinstance(policy, ConcurrencyPolicy) else ConcurrencyPolicy.from_mapping(policy)
    runs: dict[int, dict[str, Any]] = {}
    identity_comparisons: list[dict[str, Any]] = []
    rollout: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    def execute(worker_count: int) -> dict[str, Any]:
        start = clock()
        runner_error: str | None = None
        try:
            raw = _invoke_callable(runner, frozen, worker_count)
            if measurement_provider is not None:
                raw = _invoke_provider(measurement_provider, raw, frozen, worker_count)
            measurement = _measurement_from_raw(raw, frozen, elapsed_seconds=max(clock() - start, 1e-9))
        except Exception as exc:  # The report must retain failed rollout attempts.
            runner_error = f"{type(exc).__name__}: {exc}"
            measurement = ConcurrencyMeasurement(
                elapsed_seconds=max(clock() - start, 1e-9),
                requested_requests=len(frozen.request_ids),
                valid_requests=0,
                observations=0,
                failures=len(frozen.request_ids),
                retries=0,
                metadata={"exception_type": type(exc).__name__},
            )
        record = _run_record(measurement, frozen, worker_count, policy=resolved_policy, runner_error=runner_error)
        runs[worker_count] = record
        return record

    def skip(worker_count: int, reason: str, detail: str) -> None:
        runs[worker_count] = {"worker_count": worker_count, "status": "skipped", "skip_reason": reason, "detail": detail}
        rejections.append({"worker_count": worker_count, "reason": reason, "detail": detail})

    baseline = execute(1)
    baseline_identity = _identity_comparison(baseline, baseline, frozen)
    baseline_reasons = _quality_reasons(
        baseline,
        policy=resolved_policy,
        identity_status=str(baseline_identity["status"]),
    )
    baseline_ok = not baseline_reasons
    selected_worker_count: int | None = 1 if baseline_ok else None
    rollout.append({"worker_count": 1, "role": "baseline", "eligible": baseline_ok, "reasons": baseline_reasons})
    if not baseline_ok:
        rejections.append({"worker_count": 1, "reason": "baseline_quality", "detail": ", ".join(baseline_reasons)})
        for worker_count in resolved_policy.worker_counts:
            if worker_count != 1:
                skip(worker_count, "baseline_not_eligible", "The one-worker baseline did not pass operational guards.")
    else:
        candidate_3 = execute(3)
        comparison_3 = _identity_comparison(baseline, candidate_3, frozen)
        identity_comparisons.append(comparison_3)
        reasons_3 = _quality_reasons(candidate_3, policy=resolved_policy, identity_status=str(comparison_3["status"]))
        material_3 = _materially_better(candidate_3, baseline, resolved_policy.material_improvement)
        if not material_3:
            reasons_3.append("not_materially_better_than_baseline")
        candidate_3_ok = not reasons_3
        rollout.append({"worker_count": 3, "role": "candidate", "eligible": candidate_3_ok, "reasons": reasons_3, "comparison": comparison_3})
        if candidate_3_ok:
            selected_worker_count = 3
        else:
            rejections.append({"worker_count": 3, "reason": "candidate_rejected", "detail": ", ".join(reasons_3)})
        if 4 not in resolved_policy.worker_counts:
            pass
        elif not candidate_3_ok:
            skip(4, "three_worker_not_material", "Four workers are gated on a quality-passing three-worker material improvement.")
            if 5 in resolved_policy.worker_counts:
                skip(5, "four_worker_not_evaluated", "Five workers require an evaluated, stable, materially better four-worker run.")
        else:
            candidate_4 = execute(4)
            comparison_4 = _identity_comparison(baseline, candidate_4, frozen)
            identity_comparisons.append(comparison_4)
            reasons_4 = _quality_reasons(candidate_4, policy=resolved_policy, identity_status=str(comparison_4["status"]))
            material_4 = _materially_better(candidate_4, candidate_3, resolved_policy.material_improvement)
            if not material_4:
                reasons_4.append("not_materially_better_than_three_workers")
            candidate_4_ok = not reasons_4
            # Five workers require an explicit stability observation; missing
            # telemetry must not silently authorize the next rollout stage.
            stable_4 = candidate_4.get("stable") is True
            rollout.append({"worker_count": 4, "role": "candidate", "eligible": candidate_4_ok, "stable": bool(stable_4), "reasons": reasons_4, "comparison": comparison_4})
            if candidate_4_ok:
                selected_worker_count = 4
            else:
                rejections.append({"worker_count": 4, "reason": "candidate_rejected", "detail": ", ".join(reasons_4)})
            if 5 in resolved_policy.worker_counts:
                if not candidate_4_ok or not bool(stable_4):
                    skip(5, "four_worker_not_stable", "Five workers require a stable, quality-passing four-worker run.")
                else:
                    candidate_5 = execute(5)
                    comparison_5 = _identity_comparison(baseline, candidate_5, frozen)
                    identity_comparisons.append(comparison_5)
                    reasons_5 = _quality_reasons(candidate_5, policy=resolved_policy, identity_status=str(comparison_5["status"]))
                    if not _materially_better(candidate_5, candidate_4, resolved_policy.material_improvement):
                        reasons_5.append("not_materially_better_than_four_workers")
                    candidate_5_ok = not reasons_5
                    rollout.append({"worker_count": 5, "role": "candidate", "eligible": candidate_5_ok, "reasons": reasons_5, "comparison": comparison_5})
                    if candidate_5_ok:
                        selected_worker_count = 5
                    else:
                        rejections.append({"worker_count": 5, "reason": "candidate_rejected", "detail": ", ".join(reasons_5)})

    # A requested stage can only be absent when the policy deliberately omits
    # it.  The report remains explicit about every configured stage.
    for worker_count in resolved_policy.worker_counts:
        if worker_count not in runs:
            skip(worker_count, "rollout_gate", "The staged rollout did not permit this worker count.")
    frozen_report = {
        "schema_version": CONCURRENCY_SCHEMA_VERSION,
        "manifest_type": "concurrency_benchmark_manifest",
        "status": "frozen",
        "frozen": True,
        "confirmatory": False,
        "workload": frozen.to_mapping(),
        "policy": resolved_policy.to_mapping(),
        "runs": [runs[worker_count] for worker_count in sorted(runs)],
        "output_identity_comparisons": identity_comparisons,
        "selection": {
            "selected_worker_count": selected_worker_count,
            "metric": "valid_requests_per_dollar",
            "metric_description": "valid aggregate requests per estimated dollar; scientific effect sizes are not inspected",
            "rollout": rollout,
            "rejections": rejections,
            "explanation": (
                "Selected the highest staged worker count that passed operational guards and the registered material-improvement rule."
                if selected_worker_count is not None
                else "No worker count passed the operational guards and cost-normalized throughput rule."
            ),
        },
    }
    frozen_report["report_sha256"] = _canonical_hash({key: value for key, value in frozen_report.items() if key != "report_sha256"})
    return frozen_report


def write_concurrency_report(
    report: Mapping[str, Any],
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one frozen JSON report, refusing accidental replacement by default."""

    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing concurrency report: {output_path}")
    payload = dict(report)
    payload["status"] = "frozen"
    payload["frozen"] = True
    payload["report_sha256"] = _canonical_hash({key: value for key, value in payload.items() if key != "report_sha256"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


# Short aliases keep the controller easy to discover for callers that use
# "run" terminology, without introducing a second implementation.
run_concurrency_benchmark = benchmark_concurrency
build_concurrency_report = benchmark_concurrency


__all__ = [
    "CONCURRENCY_SCHEMA_VERSION",
    "ConcurrencyMeasurement",
    "ConcurrencyPolicy",
    "FrozenWorkload",
    "SUPPORTED_WORKER_COUNTS",
    "benchmark_concurrency",
    "build_concurrency_report",
    "freeze_workload",
    "run_concurrency_benchmark",
    "write_concurrency_report",
]
