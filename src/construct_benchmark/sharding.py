"""Deterministic, fail-closed workload sharding for benchmark inventories.

This module operates on request metadata only.  It accepts the repository's
canonical prompt CSV/JSONL shape as well as small generic request inventories
used by causal or downstream stages.  Pair units, factor cells, and explicit
unit IDs are treated as indivisible connected components; no model libraries
or construct-specific specifications are required.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .distributed_contracts import (
    DISTRIBUTED_SCHEMA_VERSION,
    UNSPECIFIED_RUN_CONFIG_HASH,
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    canonical_json,
    file_sha256,
    load_json_object,
    nonempty_text,
    provenance_version_families,
    stable_digest,
)


SHARD_MANIFEST_TYPE = "benchmark_shard"
SHARDING_ALGORITHM = "sha256-balanced-unit-components-v1"
SUPPORTED_SHARD_COUNTS = frozenset({3, 4, 5})

_SPLIT_ROLE = {
    "direction_train": "probe",
    "direction_validation": "probe",
    "direction_heldout": "probe",
    "probe_train": "probe",
    "probe_validation": "probe",
    "probe_heldout": "probe",
    "behavior_eval": "behavior",
    "steering_eval": "steering",
    "calibration": "calibration",
}
_ID_ALIASES = {
    "request_id": ("request_id", "prompt_id", "record_id", "id"),
    "construct_id": ("construct_id", "construct", "construct_name"),
    "role": ("prompt_role", "role", "request_role", "stage_role"),
    "split": ("split", "dataset_split", "prompt_split"),
    "pair_id": (
        "pair_id",
        "pair_unit_id",
        "paired_id",
        "pair",
        "episode_id",
        "matched_episode_id",
    ),
    "unit_id": ("unit_id", "unit", "work_unit_id", "batch_unit_id"),
    "factor_cell_id": (
        "factor_cell_id",
        "factor_cell",
        "cell_id",
        "cell",
        "generation_cell_id",
        "factor_cell_key",
    ),
    "stage_id": ("stage_id", "stage", "execution_stage", "phase"),
    "condition_id": ("condition_id", "condition", "pair_role", "condition_label"),
}
_OBSERVATION_ALIASES = (
    "observation_id",
    "expected_observation_id",
    "observation_ids",
    "expected_observation_ids",
    "expected_observations",
    "observations",
)


def _normalized_key(value: object) -> str:
    return str(value).strip().casefold().replace("-", "_")


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value).strip()
    return text or None


def _identifier_text(value: Any, *, field_name: str) -> str | None:
    """Normalize scalar IDs and structured cell/unit descriptors."""

    value = _parse_json_value(value, field_name=field_name)
    if isinstance(value, Mapping):
        for key in ("id", f"{field_name}_id", "pair_id", "unit_id", "cell_id"):
            if key in value and value[key] not in (None, ""):
                return _identifier_text(value[key], field_name=field_name)
        return canonical_json(value)
    if isinstance(value, (list, tuple)):
        return canonical_json(value)
    return _text_or_none(value)


def _parse_json_value(value: Any, *, field_name: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if field_name.endswith("_json") or stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            if field_name.endswith("_json"):
                raise ValueError(f"{field_name} contains invalid JSON.") from exc
    return value


def _metadata_sources(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return direct and nested mappings, preserving direct-key precedence."""

    sources: list[Mapping[str, Any]] = [row]
    for key, value in row.items():
        parsed = _parse_json_value(value, field_name=str(key))
        if isinstance(parsed, Mapping):
            sources.append(parsed)
    for key in ("metadata", "metadata_json", "request", "prompt", "provenance", "execution"):
        value = row.get(key)
        parsed = _parse_json_value(value, field_name=key)
        if isinstance(parsed, Mapping) and parsed not in sources:
            sources.append(parsed)
    return sources


def _find_value(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    normalized_aliases = {_normalized_key(alias) for alias in aliases}
    for source in _metadata_sources(row):
        for key, value in source.items():
            if _normalized_key(key) in normalized_aliases and value not in (None, ""):
                return _parse_json_value(value, field_name=str(key))
    return None


def _extract_nested_id(value: Any, *, field_name: str) -> list[str]:
    """Extract one or more IDs from a scalar, list, or observation object."""

    value = _parse_json_value(value, field_name=field_name)
    if value is None or value == "":
        return []
    if isinstance(value, Mapping):
        for key in (
            "observation_id",
            "expected_observation_id",
            "record_id",
            "id",
            "request_id",
        ):
            if key in value and value[key] not in (None, ""):
                return _extract_nested_id(value[key], field_name=field_name)
        return []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_extract_nested_id(item, field_name=field_name))
        return result
    text = _text_or_none(value)
    return [text] if text is not None else []


def _iter_mappings(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _iter_mappings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_mappings(item)


def _version_families(row: Mapping[str, Any]) -> frozenset[str]:
    """Find v1/v2 markers in prompt/inventory provenance fields only."""

    return provenance_version_families(row)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Inventory does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON.") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object.")
                rows.append(value)
        return rows
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON.") from exc
        if isinstance(payload, Mapping):
            for key in ("records", "requests", "items", "rows"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"{path} must contain a JSON array of objects.")
        return [dict(row) for row in payload]

    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header.")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path} has duplicate CSV header names.")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{line_number} has more fields than its CSV header.")
            if any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line_number} has a missing CSV field.")
            rows.append(dict(row))
    return rows


@dataclass(frozen=True)
class InventoryRecord:
    """Schema-neutral identity extracted from one inventory request."""

    request_id: str
    construct_id: str
    prompt_role: str
    split: str
    pair_id: str | None
    pair_role: str | None
    unit_id: str | None
    factor_cell_id: str | None
    stage_id: str
    observation_ids: tuple[str, ...]
    version_family: str | None
    raw: Mapping[str, Any] = field(repr=False)

    @property
    def scope(self) -> tuple[str, str]:
        return self.construct_id, self.stage_id

    @property
    def pair_key(self) -> tuple[str, str, str] | None:
        if self.pair_id is None:
            return None
        return self.construct_id, self.stage_id, self.pair_id

    @property
    def factor_key(self) -> tuple[str, str, str] | None:
        if self.factor_cell_id is None:
            return None
        return self.construct_id, self.stage_id, self.factor_cell_id

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, row_number: int | None = None) -> "InventoryRecord":
        if not isinstance(row, Mapping):
            location = f" on row {row_number}" if row_number is not None else ""
            raise ValueError(f"Inventory record{location} must be an object.")
        raw = dict(row)
        request_id = _text_or_none(_find_value(raw, _ID_ALIASES["request_id"]))
        construct_id = _text_or_none(_find_value(raw, _ID_ALIASES["construct_id"]))
        if not request_id:
            raise ValueError(f"Inventory row {row_number or '?'} is missing request_id/prompt_id.")
        if not construct_id:
            raise ValueError(f"Inventory request {request_id!r} is missing construct_id.")

        split = _text_or_none(_find_value(raw, _ID_ALIASES["split"])) or "unspecified"
        explicit_role = _text_or_none(_find_value(raw, _ID_ALIASES["role"]))
        inferred_role = _SPLIT_ROLE.get(split)
        if explicit_role and inferred_role and explicit_role != inferred_role:
            raise ValueError(
                f"Inventory request {request_id!r} has incompatible prompt role={explicit_role!r} "
                f"for split={split!r}; expected {inferred_role!r}."
            )
        prompt_role = explicit_role or inferred_role or "unknown"
        stage_id = _text_or_none(_find_value(raw, _ID_ALIASES["stage_id"])) or split
        pair_id = _identifier_text(_find_value(raw, _ID_ALIASES["pair_id"]), field_name="pair_id")
        pair_role = _identifier_text(_find_value(raw, _ID_ALIASES["condition_id"]), field_name="condition_id")
        explicit_unit = _identifier_text(_find_value(raw, _ID_ALIASES["unit_id"]), field_name="unit_id")
        factor_cell_id = _identifier_text(
            _find_value(raw, _ID_ALIASES["factor_cell_id"]), field_name="factor_cell_id"
        )

        observation_ids: list[str] = []
        for alias in _OBSERVATION_ALIASES:
            value = _find_value(raw, (alias,))
            if value not in (None, ""):
                observation_ids.extend(_extract_nested_id(value, field_name=alias))
                break
        if not observation_ids:
            count_value = _find_value(raw, ("expected_observation_count", "observation_count"))
            if count_value not in (None, ""):
                try:
                    count = int(count_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Inventory request {request_id!r} has an invalid observation count."
                    ) from exc
                if count < 1:
                    raise ValueError(f"Inventory request {request_id!r} must expect at least one observation.")
                observation_ids = [
                    f"{request_id}__observation_{index:03d}" for index in range(1, count + 1)
                ]
            else:
                # One request with one output is the least surprising generic
                # default.  Multi-observation stages must declare their IDs.
                observation_ids = [request_id]
        if not observation_ids or any(not str(value).strip() for value in observation_ids):
            raise ValueError(f"Inventory request {request_id!r} has empty observation IDs.")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError(f"Inventory request {request_id!r} declares duplicate observation IDs.")

        families = _version_families(raw)
        if len(families) > 1:
            raise ValueError(
                f"Inventory request {request_id!r} contains mixed version markers: {sorted(families)}."
            )
        return cls(
            request_id=request_id,
            construct_id=construct_id,
            prompt_role=prompt_role,
            split=split,
            pair_id=pair_id,
            pair_role=pair_role,
            unit_id=explicit_unit,
            factor_cell_id=factor_cell_id,
            stage_id=stage_id,
            observation_ids=tuple(str(value).strip() for value in observation_ids),
            version_family=next(iter(families), None),
            raw=raw,
        )


def normalize_inventory_records(
    records: Iterable[InventoryRecord | Mapping[str, Any]],
) -> list[InventoryRecord]:
    """Normalize mappings while retaining their original output rows."""

    materialized = list(records)
    if not materialized:
        raise ValueError("Prompt/request inventory must contain at least one record.")
    normalized: list[InventoryRecord] = []
    for index, record in enumerate(materialized, start=1):
        if isinstance(record, InventoryRecord):
            normalized.append(record)
        else:
            normalized.append(InventoryRecord.from_mapping(record, row_number=index))
    return normalized


def load_inventory(path: str | Path) -> list[InventoryRecord]:
    """Load and normalize a frozen CSV/JSONL (or JSON-array) inventory."""

    file_path = Path(path)
    return normalize_inventory_records(_load_rows(file_path))


def _coerce_expected_ids(value: Iterable[str] | str | Path | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if isinstance(payload, Mapping):
                for key in ("request_ids", "expected_request_ids", "observation_ids", "expected_observation_ids", "ids"):
                    if key in payload:
                        payload = payload[key]
                        break
            if not isinstance(payload, (list, tuple, set)):
                raise ValueError(f"Expected ID file must contain a list of strings: {path}")
            value = payload
        else:
            value = [item for item in str(value).split(",") if item.strip()]
    result = {str(item).strip() for item in value if str(item).strip()}
    if not result:
        raise ValueError("Expected ID set must not be empty.")
    return result


def validate_inventory(
    records: Iterable[InventoryRecord | Mapping[str, Any]],
    *,
    expected_request_ids: Iterable[str] | str | Path | None = None,
    expected_construct_ids: Iterable[str] | str | Path | None = None,
    expected_observation_ids: Iterable[str] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate global identity, pair/cell integrity, and version boundaries."""

    materialized = normalize_inventory_records(records)
    request_ids = [record.request_id for record in materialized]
    duplicate_requests = sorted(request_id for request_id, count in Counter(request_ids).items() if count > 1)
    if duplicate_requests:
        raise ValueError(f"Inventory contains duplicate request IDs: {duplicate_requests[:5]}.")

    observation_owner: dict[str, str] = {}
    for record in materialized:
        for observation_id in record.observation_ids:
            previous = observation_owner.get(observation_id)
            if previous is not None:
                raise ValueError(
                    f"Inventory contains duplicate observation ID {observation_id!r} "
                    f"(requests {previous!r} and {record.request_id!r})."
                )
            observation_owner[observation_id] = record.request_id

    expected_requests = _coerce_expected_ids(expected_request_ids)
    if expected_requests is not None:
        actual_requests = set(request_ids)
        unknown = sorted(actual_requests - expected_requests)
        missing = sorted(expected_requests - actual_requests)
        if unknown or missing:
            raise ValueError(
                f"Inventory request ID set differs from the frozen expectation; "
                f"unknown={unknown[:5]}, missing={missing[:5]}."
            )
    expected_constructs = _coerce_expected_ids(expected_construct_ids)
    actual_constructs = {record.construct_id for record in materialized}
    if expected_constructs is not None:
        unknown = sorted(actual_constructs - expected_constructs)
        missing = sorted(expected_constructs - actual_constructs)
        if unknown or missing:
            raise ValueError(
                f"Inventory construct ID set differs from the frozen expectation; "
                f"unknown={unknown[:5]}, missing={missing[:5]}."
            )
    expected_observations = _coerce_expected_ids(expected_observation_ids)
    actual_observations = set(observation_owner)
    if expected_observations is not None:
        unknown = sorted(actual_observations - expected_observations)
        missing = sorted(expected_observations - actual_observations)
        if unknown or missing:
            raise ValueError(
                f"Inventory observation ID set differs from the frozen expectation; "
                f"unknown={unknown[:5]}, missing={missing[:5]}."
            )

    version_families = {record.version_family for record in materialized if record.version_family}
    if len(version_families) > 1:
        raise ValueError(
            f"Inventory mixes incompatible prompt/request versions: {sorted(version_families)}. "
            "Do not mix v1 and v2 artifacts."
        )

    pair_groups: dict[tuple[str, str, str], list[InventoryRecord]] = defaultdict(list)
    factor_groups: dict[tuple[str, str, str], list[InventoryRecord]] = defaultdict(list)
    explicit_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_constructs: dict[str, set[str]] = defaultdict(set)
    unit_constructs: dict[str, set[str]] = defaultdict(set)
    for record in materialized:
        if record.pair_key is not None:
            pair_groups[record.pair_key].append(record)
            pair_constructs[record.pair_id or ""].add(record.construct_id)
        if record.factor_key is not None:
            factor_groups[record.factor_key].append(record)
        if record.unit_id is not None:
            explicit_units[(record.construct_id, record.stage_id)].add(record.unit_id)
            unit_constructs[record.unit_id].add(record.construct_id)

    pooled_pairs = sorted(pair_id for pair_id, constructs in pair_constructs.items() if len(constructs) > 1)
    pooled_units = sorted(unit_id for unit_id, constructs in unit_constructs.items() if len(constructs) > 1)
    if pooled_pairs or pooled_units:
        raise ValueError(
            "Construct pooling is not allowed: shared pair/unit IDs span constructs; "
            f"pairs={pooled_pairs[:5]}, units={pooled_units[:5]}."
        )

    for key, group in sorted(pair_groups.items()):
        construct_id, stage_id, pair_id = key
        if len(group) != 2:
            raise ValueError(
                f"Incomplete pair unit {construct_id}/{stage_id}/{pair_id}: "
                f"expected exactly 2 requests, found {len(group)}."
            )
        roles = {record.prompt_role for record in group}
        if len(roles) != 1:
            raise ValueError(
                f"Pair unit {construct_id}/{stage_id}/{pair_id} has incompatible prompt roles: "
                f"{sorted(roles)}."
            )
        pair_roles = [record.pair_role for record in group if record.pair_role is not None]
        if pair_roles and len(pair_roles) != len(group):
            raise ValueError(
                f"Pair unit {construct_id}/{stage_id}/{pair_id} has an incomplete pair-role declaration."
            )
        if pair_roles and len(set(pair_roles)) != len(pair_roles):
            raise ValueError(
                f"Pair unit {construct_id}/{stage_id}/{pair_id} has duplicate pair roles: {pair_roles}."
            )

    for key, group in sorted(factor_groups.items()):
        construct_id, stage_id, cell_id = key
        roles = {record.prompt_role for record in group}
        if len(roles) != 1:
            raise ValueError(
                f"Factor cell {construct_id}/{stage_id}/{cell_id} has incompatible prompt roles: "
                f"{sorted(roles)}."
            )

    return {
        "record_count": len(materialized),
        "request_ids": sorted(request_ids),
        "observation_ids": sorted(actual_observations),
        "construct_ids": sorted(actual_constructs),
        "version_families": sorted(version_families),
        "pair_count": len(pair_groups),
        "factor_cell_count": len(factor_groups),
        "counts_by_construct": dict(sorted(Counter(record.construct_id for record in materialized).items())),
        "counts_by_role": dict(sorted(Counter(record.prompt_role for record in materialized).items())),
        "counts_by_stage": dict(sorted(Counter(record.stage_id for record in materialized).items())),
    }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        parent = self.parent[index]
        if parent != index:
            self.parent[index] = self.find(parent)
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class _Unit:
    records: tuple[InventoryRecord, ...]
    key: str

    @property
    def construct_id(self) -> str:
        return self.records[0].construct_id

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(record.request_id for record in self.records))


def _unit_components(records: Sequence[InventoryRecord], *, seed: int) -> list[_Unit]:
    union_find = _UnionFind(len(records))
    groups: dict[tuple[str, str, str, str], int] = {}
    for index, record in enumerate(records):
        links: list[tuple[str, str, str, str]] = []
        if record.pair_id is not None:
            links.append(("pair", record.construct_id, record.stage_id, record.pair_id))
        if record.factor_cell_id is not None:
            links.append(("factor", record.construct_id, record.stage_id, record.factor_cell_id))
        if record.unit_id is not None:
            links.append(("unit", record.construct_id, record.stage_id, record.unit_id))
        for link in links:
            previous = groups.get(link)
            if previous is None:
                groups[link] = index
            else:
                union_find.union(index, previous)

    by_root: dict[int, list[InventoryRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        by_root[union_find.find(index)].append(record)
    units: list[_Unit] = []
    for group in by_root.values():
        ordered = tuple(sorted(group, key=lambda record: record.request_id))
        construct_ids = {record.construct_id for record in ordered}
        if len(construct_ids) != 1:
            raise ValueError(
                "Construct pooling is not allowed: one indivisible unit contains multiple constructs."
            )
        request_key = "|".join(record.request_id for record in ordered)
        key = f"unit_{stable_digest(seed, ordered[0].construct_id, request_key)[:20]}"
        units.append(_Unit(records=ordered, key=key))
    return sorted(units, key=lambda unit: (unit.construct_id, unit.key, unit.request_ids))


def _partition_units(units: Sequence[_Unit], count: int, *, seed: int) -> list[list[_Unit]]:
    if len(units) < count:
        raise ValueError(
            f"Cannot create {count} non-empty shards from {len(units)} indivisible units; "
            "a pair or factor cell would have to be split."
        )
    bins: list[list[_Unit]] = [[] for _ in range(count)]
    loads = [0] * count
    observation_loads = [0] * count
    ordered = sorted(
        units,
        key=lambda unit: (
            -len(unit.records),
            -sum(len(record.observation_ids) for record in unit.records),
            stable_digest(seed, "unit-order", unit.construct_id, unit.key),
            unit.request_ids,
        ),
    )
    for unit in ordered:
        target = min(
            range(count),
            key=lambda index: (
                loads[index],
                observation_loads[index],
                index,
            ),
        )
        bins[target].append(unit)
        loads[target] += len(unit.records)
        observation_loads[target] += sum(len(record.observation_ids) for record in unit.records)
    return [sorted(bucket, key=lambda unit: unit.request_ids) for bucket in bins]


def _counts_for_records(records: Sequence[InventoryRecord]) -> dict[str, Any]:
    by_role = Counter(record.prompt_role for record in records)
    by_construct = Counter(record.construct_id for record in records)
    by_stage = Counter(record.stage_id for record in records)
    by_pair: Counter[str] = Counter()
    by_cell: Counter[str] = Counter()
    pair_units: dict[str, list[InventoryRecord]] = defaultdict(list)
    for record in records:
        if record.pair_id is not None:
            pair_key = f"{record.stage_id}::{record.pair_id}"
            by_pair[pair_key] += 1
            pair_units[pair_key].append(record)
        if record.factor_cell_id is not None:
            by_cell[f"{record.stage_id}::{record.factor_cell_id}"] += 1
    return {
        "requests": len(records),
        "observations": sum(len(record.observation_ids) for record in records),
        "by_role": dict(sorted(by_role.items())),
        "by_construct": dict(sorted(by_construct.items())),
        "by_stage": dict(sorted(by_stage.items())),
        "by_pair": dict(sorted(by_pair.items())),
        "by_factor_cell": dict(sorted(by_cell.items())),
        "pair_units": [
            {
                "pair_key": pair_key,
                "request_ids": sorted(record.request_id for record in pair_records),
            }
            for pair_key, pair_records in sorted(pair_units.items())
        ],
    }


@dataclass
class ShardPlan:
    """In-memory deterministic plan plus the manifest projections."""

    inventory_path: str
    parent_inventory_sha256: str
    records: tuple[InventoryRecord, ...]
    records_by_shard: dict[str, tuple[InventoryRecord, ...]]
    manifests: dict[str, dict[str, Any]]
    assignment: dict[str, str]
    worker_count: int
    worker_schedule: dict[str, tuple[str, ...]]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": DISTRIBUTED_SCHEMA_VERSION,
            "manifest_type": "benchmark_shard_plan",
            "parent_inventory_path": self.inventory_path,
            "parent_inventory_sha256": self.parent_inventory_sha256,
            "shard_count": len(self.manifests),
            "worker_count": self.worker_count,
            "worker_schedule": {
                worker_id: list(shard_ids)
                for worker_id, shard_ids in sorted(self.worker_schedule.items())
            },
            "shards": [self.manifests[shard_id] for shard_id in sorted(self.manifests)],
            "assignment": dict(sorted(self.assignment.items())),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_mapping()[key]


def build_shard_plan(
    inventory: str | Path | Iterable[InventoryRecord | Mapping[str, Any]],
    *,
    shard_count: int | None = None,
    num_shards: int | None = None,
    worker_count: int | None = None,
    num_workers: int | None = None,
    seed: int = 0,
    run_config_hash: str | None = None,
    run_mode: str = "test",
    confirmatory: bool = False,
    engineering_only: bool = False,
    expected_request_ids: Iterable[str] | str | Path | None = None,
    expected_construct_ids: Iterable[str] | str | Path | None = None,
    expected_observation_ids: Iterable[str] | str | Path | None = None,
    split_construct_id: str | None = None,
) -> ShardPlan:
    """Build a reproducible plan without writing any files.

    For the standard four-construct wave, three workers receive four physical
    construct-pure shards through a deterministic three-slot schedule, four
    workers receive one construct each, and five workers split one
    deterministically selected construct into two balanced, disjoint unit
    sets.  A one-construct fixture may be split into any of the supported
    physical shard counts for deterministic testing.
    """

    requested_shard_count = shard_count if shard_count is not None else num_shards
    requested_worker_count = worker_count if worker_count is not None else num_workers
    if requested_shard_count is not None:
        if isinstance(requested_shard_count, bool) or requested_shard_count not in SUPPORTED_SHARD_COUNTS:
            raise ValueError(f"shard_count must be one of {sorted(SUPPORTED_SHARD_COUNTS)}.")
    if requested_worker_count is not None:
        if isinstance(requested_worker_count, bool) or requested_worker_count not in SUPPORTED_SHARD_COUNTS:
            raise ValueError(f"worker_count must be one of {sorted(SUPPORTED_SHARD_COUNTS)}.")
    if requested_shard_count is not None and requested_worker_count is not None:
        if requested_worker_count > requested_shard_count:
            raise ValueError("worker_count cannot exceed the physical shard_count.")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer.") from exc

    source_path: str
    if isinstance(inventory, (str, Path)):
        path = Path(inventory)
        records = tuple(load_inventory(path))
        source_path = str(path.resolve())
        parent_hash = file_sha256(path)
    else:
        records = tuple(normalize_inventory_records(inventory))
        source_path = "<in-memory>"
        parent_hash = canonical_hash(
            [dict(record.raw) for record in sorted(records, key=lambda record: record.request_id)]
        )
    validate_inventory(
        records,
        expected_request_ids=expected_request_ids,
        expected_construct_ids=expected_construct_ids,
        expected_observation_ids=expected_observation_ids,
    )
    if not isinstance(run_mode, str) or not run_mode.strip():
        raise ValueError("run_mode must be a non-empty string.")
    run_mode = run_mode.strip()
    if not isinstance(confirmatory, bool):
        raise ValueError("confirmatory must be a boolean.")
    if not isinstance(engineering_only, bool):
        raise ValueError("engineering_only must be a boolean.")
    if run_mode == "test" and confirmatory:
        raise ValueError("test run_mode cannot be confirmatory.")
    if run_mode == "full" and not confirmatory and not engineering_only:
        raise ValueError(
            "full run_mode must be confirmatory unless engineering_only=true is explicitly supplied."
        )
    if run_mode != "full" and engineering_only:
        raise ValueError("engineering_only is only valid for the full run_mode.")
    if split_construct_id is not None:
        split_construct_id = nonempty_text(split_construct_id, field_name="split_construct_id")

    if run_config_hash is None:
        run_config_hash = UNSPECIFIED_RUN_CONFIG_HASH
    run_config_hash = nonempty_text(run_config_hash, field_name="run_config_hash")

    components = _unit_components(records, seed=seed)
    by_construct: dict[str, list[_Unit]] = defaultdict(list)
    for component in components:
        by_construct[component.construct_id].append(component)
    construct_ids = sorted(by_construct)
    construct_count = len(construct_ids)
    if requested_worker_count is None:
        worker_count = requested_shard_count if requested_shard_count is not None else construct_count
    else:
        worker_count = requested_worker_count
    if worker_count not in SUPPORTED_SHARD_COUNTS:
        raise ValueError(f"worker_count must be one of {sorted(SUPPORTED_SHARD_COUNTS)}.")
    if requested_shard_count is None:
        shard_count = max(construct_count, worker_count)
    else:
        shard_count = requested_shard_count
        # The common four-construct/three-worker rollout is four physical
        # shards scheduled onto three subprocess slots.  Accepting the
        # shorthand (shard_count=3, worker_count=3) keeps the CLI ergonomic,
        # while the emitted manifest records the actual four-shard layout.
        if (
            requested_worker_count is not None
            and
            construct_count == 4
            and shard_count == 3
            and worker_count == 3
        ):
            shard_count = 4
    if shard_count not in SUPPORTED_SHARD_COUNTS:
        raise ValueError(f"physical shard_count must be one of {sorted(SUPPORTED_SHARD_COUNTS)}.")
    if worker_count > shard_count:
        raise ValueError("worker_count cannot exceed the physical shard_count.")
    if construct_count > shard_count:
        raise ValueError(
            f"Cannot assign {construct_count} constructs to {shard_count} shards without construct pooling."
        )

    extra = shard_count - construct_count
    split_ids: dict[str, int] = {}
    if extra == 0:
        pass
    elif construct_count == 1:
        split_ids[construct_ids[0]] = shard_count
    elif shard_count == 5 and extra == 1:
        if split_construct_id is None:
            split_construct_id = max(
                construct_ids,
                key=lambda construct_id: (
                    sum(len(unit.records) for unit in by_construct[construct_id]),
                    sum(len(unit.records) for unit in by_construct[construct_id]),
                    stable_digest(seed, "split-construct", construct_id),
                ),
            )
        if split_construct_id not in by_construct:
            raise ValueError(
                f"split_construct_id={split_construct_id!r} is not present in the inventory."
            )
        split_ids[split_construct_id] = 2
    else:
        raise ValueError(
            f"Unsupported layout for {construct_count} constructs and {shard_count} shards; "
            "use one shard per construct or the five-worker one-construct split layout."
        )

    bucket_specs: list[tuple[str, str, list[_Unit]]] = []
    for construct_id in construct_ids:
        units = sorted(by_construct[construct_id], key=lambda unit: unit.request_ids)
        bucket_count = split_ids.get(construct_id, 1)
        buckets = _partition_units(units, bucket_count, seed=seed) if bucket_count > 1 else [units]
        for part_index, bucket in enumerate(buckets, start=1):
            suffix = f"_part_{part_index:02d}" if bucket_count > 1 else ""
            bucket_specs.append((construct_id, suffix, bucket))

    if len(bucket_specs) != shard_count:
        raise AssertionError("Internal shard layout did not produce the requested shard count.")

    records_by_shard: dict[str, tuple[InventoryRecord, ...]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    assignment: dict[str, str] = {}
    for shard_index, (construct_id, suffix, units) in enumerate(bucket_specs, start=1):
        shard_id = f"shard_{shard_index:03d}{suffix}"
        shard_records = tuple(
            sorted(
                (record for unit in units for record in unit.records),
                key=lambda record: record.request_id,
            )
        )
        if not shard_records:
            raise ValueError(f"Shard {shard_id} would be empty.")
        for record in shard_records:
            if record.request_id in assignment:
                raise ValueError(f"Request {record.request_id!r} was assigned to multiple shards.")
            assignment[record.request_id] = shard_id
        counts = _counts_for_records(shard_records)
        observation_records = [
            {
                "observation_id": observation_id,
                "request_id": record.request_id,
                "construct_id": record.construct_id,
                "stage_id": record.stage_id,
            }
            for record in shard_records
            for observation_id in record.observation_ids
        ]
        observation_records.sort(key=lambda item: item["observation_id"])
        version_families = sorted({record.version_family for record in shard_records if record.version_family})
        manifest = {
            "schema_version": DISTRIBUTED_SCHEMA_VERSION,
            "manifest_type": SHARD_MANIFEST_TYPE,
            "immutable": True,
            "shard_id": shard_id,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "worker_count": worker_count,
            "worker_slot": None,
            "algorithm": SHARDING_ALGORITHM,
            "seed": seed,
            "parent_inventory_path": source_path,
            "parent_path": source_path,
            "parent_inventory_sha256": parent_hash,
            "parent_inventory_hash": parent_hash,
            "run_config_hash": run_config_hash,
            "run_mode": run_mode,
            "confirmatory": confirmatory,
            "engineering_only": engineering_only,
            "construct_ids": [construct_id],
            "construct_id": construct_id,
            "owned_request_ids": [record.request_id for record in shard_records],
            "request_ids": [record.request_id for record in shard_records],
            "owned_observation_ids": sorted(
                observation_id for record in shard_records for observation_id in record.observation_ids
            ),
            "expected_observation_ids": sorted(
                observation_id for record in shard_records for observation_id in record.observation_ids
            ),
            "expected_observations": observation_records,
            "expected_observation_count": len(observation_records),
            "counts": counts,
            "counts_by_role": counts["by_role"],
            "counts_by_pair": counts["by_pair"],
            "counts_by_factor_cell": counts["by_factor_cell"],
            "counts_by_stage": counts["by_stage"],
            "unit_count": len(units),
            "unit_ids": [unit.key for unit in sorted(units, key=lambda unit: unit.key)],
            "owned_unit_ids": [unit.key for unit in sorted(units, key=lambda unit: unit.key)],
            "owned_pair_unit_ids": sorted(counts["by_pair"]),
            "owned_factor_cell_ids": sorted(counts["by_factor_cell"]),
            "version_families": version_families,
            "parent_request_count": len(records),
            "parent_observation_count": sum(len(record.observation_ids) for record in records),
            "owned_request_count": len(shard_records),
            "owned_observation_count": len(observation_records),
            "expected_request_count": len(shard_records),
        }
        records_by_shard[shard_id] = shard_records
        manifests[shard_id] = manifest

    if set(assignment) != {record.request_id for record in records}:
        raise ValueError("Sharding assignment omitted one or more inventory requests.")
    worker_schedule: dict[str, list[str]] = {
        f"worker_{index:03d}": [] for index in range(1, worker_count + 1)
    }
    for shard_index, shard_id in enumerate(sorted(manifests), start=1):
        worker_id = f"worker_{((shard_index - 1) % worker_count) + 1:03d}"
        worker_schedule[worker_id].append(shard_id)
    final_schedule = {
        slot: list(shard_ids) for slot, shard_ids in sorted(worker_schedule.items())
    }
    for shard_id in sorted(manifests):
        worker_id = next(
            slot for slot, shard_ids in worker_schedule.items() if shard_id in shard_ids
        )
        manifests[shard_id]["worker_slot"] = worker_id
        manifests[shard_id]["worker_schedule"] = final_schedule
    return ShardPlan(
        inventory_path=source_path,
        parent_inventory_sha256=parent_hash,
        records=records,
        records_by_shard=records_by_shard,
        manifests=manifests,
        assignment=assignment,
        worker_count=worker_count,
        worker_schedule={slot: tuple(shard_ids) for slot, shard_ids in worker_schedule.items()},
    )


def plan_shards(*args: Any, **kwargs: Any) -> ShardPlan:
    """Alias for :func:`build_shard_plan` used by callers that only plan."""

    return build_shard_plan(*args, **kwargs)


def assign_shards(
    inventory: str | Path | Iterable[InventoryRecord | Mapping[str, Any]],
    *,
    shard_count: int | None = None,
    num_shards: int | None = None,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, str]:
    """Return only the deterministic request-to-shard assignment."""

    plan = build_shard_plan(
        inventory,
        shard_count=shard_count,
        num_shards=num_shards,
        seed=seed,
        **kwargs,
    )
    return dict(sorted(plan.assignment.items()))


def _write_inventory_records(records: Sequence[InventoryRecord], path: Path, *, suffix: str) -> None:
    rows = [dict(record.raw) for record in records]
    if suffix.casefold() in {".jsonl", ".ndjson"}:
        atomic_write_text(
            path,
            "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
            label="shard inventory",
        )
        return
    if suffix.casefold() == ".json":
        atomic_write_text(
            path,
            json.dumps(rows, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            label="shard inventory",
        )
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        raise ValueError(f"Cannot write empty-column shard inventory: {path}")
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = {
            key: canonical_json(value) if isinstance(value, (Mapping, list, tuple)) else value
            for key, value in row.items()
        }
        writer.writerow(serialized)
    atomic_write_text(path, buffer.getvalue(), label="shard inventory")


def write_shard_outputs(
    plan: ShardPlan,
    output_dir: str | Path,
    *,
    inventory_suffix: str = ".jsonl",
) -> dict[str, Any]:
    """Write shard inventories and immutable manifests after full validation."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = inventory_suffix if inventory_suffix.startswith(".") else f".{inventory_suffix}"
    targets: list[Path] = []
    for shard_id in sorted(plan.manifests):
        targets.extend(
            [output_root / f"{shard_id}{suffix}", output_root / f"{shard_id}.manifest.json"]
        )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing shard artifacts: {existing[:5]}")

    written: list[dict[str, Any]] = []
    for shard_id in sorted(plan.manifests):
        shard_path = output_root / f"{shard_id}{suffix}"
        manifest_path = output_root / f"{shard_id}.manifest.json"
        records = plan.records_by_shard[shard_id]
        _write_inventory_records(records, shard_path, suffix=suffix)
        manifest = dict(plan.manifests[shard_id])
        manifest["shard_inventory_path"] = str(shard_path.resolve())
        manifest["shard_path"] = str(shard_path.resolve())
        manifest["shard_inventory_sha256"] = file_sha256(shard_path)
        manifest["manifest_path"] = str(manifest_path.resolve())
        atomic_write_json(manifest_path, manifest, label="shard manifest")
        manifest["manifest_sha256"] = file_sha256(manifest_path)
        written.append(
            {
                "shard_id": shard_id,
                "inventory_path": str(shard_path),
                "manifest_path": str(manifest_path),
                "request_count": len(records),
                "observation_count": sum(len(record.observation_ids) for record in records),
            }
        )
    return {
        "schema_version": DISTRIBUTED_SCHEMA_VERSION,
        "manifest_type": "benchmark_shard_plan",
        "algorithm": SHARDING_ALGORITHM,
        "seed": plan.manifests[sorted(plan.manifests)[0]]["seed"],
        "parent_inventory_path": plan.inventory_path,
        "parent_inventory_sha256": plan.parent_inventory_sha256,
        "shard_count": len(plan.manifests),
        "worker_count": plan.worker_count,
        "worker_schedule": {
            worker_id: list(shard_ids)
            for worker_id, shard_ids in sorted(plan.worker_schedule.items())
        },
        "shards": written,
        "assignment": dict(sorted(plan.assignment.items())),
    }


def _load_manifest(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return load_json_object(value, label="shard manifest")


def validate_shard_manifests(
    manifests: Iterable[Mapping[str, Any] | str | Path],
    *,
    expected_request_ids: Iterable[str] | str | Path | None = None,
    expected_construct_ids: Iterable[str] | str | Path | None = None,
    expected_observation_ids: Iterable[str] | str | Path | None = None,
    expected_shard_count: int | None = None,
    expected_worker_count: int | None = None,
) -> dict[str, Any]:
    """Validate a complete set of shard manifests without reading workers."""

    loaded = [_load_manifest(value) for value in manifests]
    if not loaded:
        raise ValueError("At least one shard manifest is required.")
    if expected_shard_count is not None and len(loaded) != expected_shard_count:
        raise ValueError(
            f"Expected {expected_shard_count} shard manifests, found {len(loaded)}."
        )
    shard_ids: list[str] = []
    all_requests: list[str] = []
    all_observations: list[str] = []
    all_constructs: set[str] = set()
    parent_hashes: set[str] = set()
    modes: set[str] = set()
    confirmatory_states: set[bool] = set()
    engineering_states: set[bool] = set()
    versions: set[str] = set()
    worker_counts: set[int] = set()
    pair_locations: dict[str, str] = {}
    factor_locations: dict[str, str] = {}
    for manifest in loaded:
        required = (
            "schema_version",
            "manifest_type",
            "shard_id",
            "shard_count",
            "worker_count",
            "parent_inventory_sha256",
            "owned_request_ids",
            "expected_observation_ids",
            "construct_ids",
            "algorithm",
            "seed",
            "run_config_hash",
            "run_mode",
            "confirmatory",
        )
        missing = [field_name for field_name in required if field_name not in manifest]
        if missing:
            raise ValueError(f"Shard manifest is missing required fields: {missing}.")
        if manifest["manifest_type"] != SHARD_MANIFEST_TYPE:
            raise ValueError(f"Unexpected shard manifest_type={manifest['manifest_type']!r}.")
        if manifest["schema_version"] != DISTRIBUTED_SCHEMA_VERSION:
            raise ValueError(f"Unsupported shard schema_version={manifest['schema_version']!r}.")
        shard_id = nonempty_text(manifest["shard_id"], field_name="shard_id")
        shard_ids.append(shard_id)
        requests = manifest["owned_request_ids"]
        observations = manifest["expected_observation_ids"]
        if not isinstance(requests, list) or not all(isinstance(item, str) and item for item in requests):
            raise ValueError(f"Shard {shard_id} owned_request_ids must be a list of non-empty strings.")
        if not isinstance(observations, list) or not all(isinstance(item, str) and item for item in observations):
            raise ValueError(f"Shard {shard_id} expected_observation_ids must be a list of non-empty strings.")
        if len(requests) != len(set(requests)):
            raise ValueError(f"Shard {shard_id} contains duplicate owned request IDs.")
        if len(observations) != len(set(observations)):
            raise ValueError(f"Shard {shard_id} contains duplicate observation IDs.")
        counts = manifest.get("counts", {})
        if not isinstance(counts, Mapping):
            raise ValueError(f"Shard {shard_id} counts must be an object.")
        pair_counts = manifest.get("counts_by_pair", counts.get("by_pair", {}))
        if not isinstance(pair_counts, Mapping):
            raise ValueError(f"Shard {shard_id} counts_by_pair must be an object.")
        for pair_key, pair_count in pair_counts.items():
            if pair_key in pair_locations:
                raise ValueError(
                    f"Pair unit {pair_key!r} is split across shards "
                    f"{pair_locations[pair_key]!r} and {shard_id!r}."
                )
            try:
                if int(pair_count) != 2:
                    raise ValueError(
                        f"Incomplete pair unit {pair_key!r} in shard {shard_id}: expected 2 requests."
                    )
            except (TypeError, ValueError) as exc:
                if isinstance(exc, ValueError) and str(exc).startswith("Incomplete pair unit"):
                    raise
                raise ValueError(f"Shard {shard_id} has invalid pair count for {pair_key!r}.") from exc
            pair_locations[str(pair_key)] = shard_id
        factor_counts = manifest.get("counts_by_factor_cell", counts.get("by_factor_cell", {}))
        if not isinstance(factor_counts, Mapping):
            raise ValueError(f"Shard {shard_id} counts_by_factor_cell must be an object.")
        for factor_key in factor_counts:
            if factor_key in factor_locations:
                raise ValueError(
                    f"Factor cell {factor_key!r} is split across shards "
                    f"{factor_locations[factor_key]!r} and {shard_id!r}."
                )
            factor_locations[str(factor_key)] = shard_id
        all_requests.extend(requests)
        all_observations.extend(observations)
        constructs = manifest["construct_ids"]
        if not isinstance(constructs, list) or not constructs:
            raise ValueError(f"Shard {shard_id} construct_ids must be a non-empty list.")
        if len(constructs) != 1:
            raise ValueError(
                f"Shard {shard_id} pools constructs {constructs}; one shard may own only one construct."
            )
        all_constructs.update(str(item) for item in constructs)
        parent_hashes.add(nonempty_text(manifest["parent_inventory_sha256"], field_name="parent_inventory_sha256"))
        modes.add(nonempty_text(manifest["run_mode"], field_name="run_mode"))
        if not isinstance(manifest["confirmatory"], bool):
            raise ValueError(f"Shard {shard_id} confirmatory must be a boolean.")
        confirmatory_states.add(manifest["confirmatory"])
        engineering_only = manifest.get("engineering_only", False)
        if not isinstance(engineering_only, bool):
            raise ValueError(f"Shard {shard_id} engineering_only must be a boolean.")
        if manifest["run_mode"] == "full" and not manifest["confirmatory"] and not engineering_only:
            raise ValueError(
                f"Shard {shard_id} is a non-confirmatory full shard without engineering_only=true."
            )
        if manifest["run_mode"] != "full" and engineering_only:
            raise ValueError(f"Shard {shard_id} engineering_only is only valid for full mode.")
        engineering_states.add(engineering_only)
        versions.update(str(value) for value in manifest.get("version_families", []))
        try:
            worker_counts.add(int(manifest["worker_count"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Shard {shard_id} worker_count must be an integer.") from exc

    if len(set(shard_ids)) != len(shard_ids):
        raise ValueError(f"Shard manifests contain duplicate shard IDs: {sorted(set(shard_ids))}.")
    if len(parent_hashes) != 1:
        raise ValueError(f"Shard manifests disagree on parent inventory hash: {sorted(parent_hashes)}.")
    if len(modes) != 1 or len(confirmatory_states) != 1 or len(engineering_states) != 1:
        raise ValueError("Shard manifests disagree on run mode or confirmatory state.")
    if len(worker_counts) != 1:
        raise ValueError("Shard manifests disagree on worker_count.")
    declared_worker_count = next(iter(worker_counts))
    if expected_worker_count is not None and declared_worker_count != expected_worker_count:
        raise ValueError(
            f"Expected worker_count={expected_worker_count}, found {declared_worker_count}."
        )
    if len(versions) > 1:
        raise ValueError(f"Shard manifests mix v1/v2 prompt versions: {sorted(versions)}.")
    duplicate_requests = sorted(item for item, count in Counter(all_requests).items() if count > 1)
    duplicate_observations = sorted(item for item, count in Counter(all_observations).items() if count > 1)
    if duplicate_requests:
        raise ValueError(f"Shard manifests duplicate request IDs across shards: {duplicate_requests[:5]}.")
    if duplicate_observations:
        raise ValueError(
            f"Shard manifests duplicate observation IDs across shards: {duplicate_observations[:5]}.")

    expected_requests = _coerce_expected_ids(expected_request_ids)
    if expected_requests is not None:
        actual = set(all_requests)
        if actual - expected_requests or expected_requests - actual:
            raise ValueError(
                f"Shard request ID coverage differs from the parent; "
                f"unknown={sorted(actual - expected_requests)[:5]}, "
                f"missing={sorted(expected_requests - actual)[:5]}."
            )
    expected_observations_set = _coerce_expected_ids(expected_observation_ids)
    if expected_observations_set is not None:
        actual = set(all_observations)
        if actual - expected_observations_set or expected_observations_set - actual:
            raise ValueError(
                f"Shard observation ID coverage differs from the parent; "
                f"unknown={sorted(actual - expected_observations_set)[:5]}, "
                f"missing={sorted(expected_observations_set - actual)[:5]}."
            )
    expected_constructs_set = _coerce_expected_ids(expected_construct_ids)
    if expected_constructs_set is not None and all_constructs != expected_constructs_set:
        raise ValueError(
            f"Shard construct coverage differs from the parent; "
            f"unknown={sorted(all_constructs - expected_constructs_set)}, "
            f"missing={sorted(expected_constructs_set - all_constructs)}."
        )
    declared_counts = {int(manifest["shard_count"]) for manifest in loaded}
    if len(declared_counts) != 1 or next(iter(declared_counts)) != len(loaded):
        raise ValueError("Shard manifests do not declare one consistent shard count.")
    return {
        "schema_version": DISTRIBUTED_SCHEMA_VERSION,
        "manifest_type": "benchmark_shard_set",
        "shard_ids": sorted(shard_ids),
        "shard_count": len(loaded),
        "worker_count": declared_worker_count,
        "parent_inventory_sha256": next(iter(parent_hashes)),
        "run_mode": next(iter(modes)),
        "confirmatory": next(iter(confirmatory_states)),
        "engineering_only": next(iter(engineering_states)),
        "construct_ids": sorted(all_constructs),
        "request_ids": sorted(all_requests),
        "observation_ids": sorted(all_observations),
        "pair_units": sorted(pair_locations),
        "factor_cells": sorted(factor_locations),
    }


def shard_inventory(
    inventory: str | Path | Iterable[InventoryRecord | Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    inventory_suffix: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Plan a frozen inventory and optionally materialize immutable shards."""

    plan = build_shard_plan(inventory, **kwargs)
    if output_dir is None:
        return plan.to_mapping()
    if inventory_suffix is None:
        inventory_suffix = Path(inventory).suffix if isinstance(inventory, (str, Path)) else ".jsonl"
    return write_shard_outputs(plan, output_dir, inventory_suffix=inventory_suffix)


__all__ = [
    "InventoryRecord",
    "SHARD_MANIFEST_TYPE",
    "SHARDING_ALGORITHM",
    "SUPPORTED_SHARD_COUNTS",
    "ShardPlan",
    "assign_shards",
    "build_shard_plan",
    "load_inventory",
    "normalize_inventory_records",
    "plan_shards",
    "shard_inventory",
    "validate_inventory",
    "validate_shard_manifests",
    "write_shard_outputs",
]
