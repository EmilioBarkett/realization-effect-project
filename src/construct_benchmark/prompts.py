"""Canonical prompt inventory format shared by all constructs."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import PROMPT_ROLES, ConstructSpec
from .splits import SPLIT_PROMPT_ROLE


PROMPT_FIELDS = (
    "prompt_id",
    "construct_id",
    "split",
    "prompt_role",
    "pair_id",
    "pair_role",
    "condition_id",
    "prompt_family",
    "task_id",
    "prompt_text",
    "expected_output_format",
    "parser_id",
    "metadata_json",
)
_RESERVED_FIELDS = set(PROMPT_FIELDS)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    value = _text(value)
    return value or None


def _infer_prompt_role(split: str) -> str:
    return SPLIT_PROMPT_ROLE.get(split, "probe")


def _normalized_prompt_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass(frozen=True)
class PromptRecord:
    """One prompt in the combined multi-construct inventory."""

    prompt_id: str
    construct_id: str
    split: str
    prompt_role: str
    prompt_text: str
    condition_id: str | None = None
    pair_id: str | None = None
    pair_role: str | None = None
    prompt_family: str | None = None
    task_id: str | None = None
    expected_output_format: str | None = None
    parser_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PromptRecord":
        payload = dict(row)
        prompt_id = _text(payload.get("prompt_id"))
        construct_id = _text(payload.get("construct_id"))
        split = _text(payload.get("split"))
        prompt_text = _text(payload.get("prompt_text"))
        prompt_role = _text(payload.get("prompt_role")) or _infer_prompt_role(split)
        condition_id = _optional_text(payload.get("condition_id") or payload.get("condition"))
        pair_id = _optional_text(payload.get("pair_id"))
        pair_role = _optional_text(payload.get("pair_role"))
        if pair_role is None and condition_id is not None and pair_id is not None:
            pair_role = condition_id

        metadata: dict[str, Any] = {}
        raw_metadata = payload.get("metadata_json", payload.get("metadata"))
        if isinstance(raw_metadata, Mapping):
            metadata.update(dict(raw_metadata))
        elif isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                parsed_metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Prompt {prompt_id or '<unknown>'} has invalid metadata_json.") from exc
            if not isinstance(parsed_metadata, dict):
                raise ValueError(f"Prompt {prompt_id or '<unknown>'} metadata_json must contain an object.")
            metadata.update(parsed_metadata)
        for key, value in payload.items():
            if key not in _RESERVED_FIELDS and key not in metadata and value not in (None, ""):
                metadata[key] = value

        return cls(
            prompt_id=prompt_id,
            construct_id=construct_id,
            split=split,
            prompt_role=prompt_role,
            prompt_text=prompt_text,
            condition_id=condition_id,
            pair_id=pair_id,
            pair_role=pair_role,
            prompt_family=_optional_text(payload.get("prompt_family")),
            task_id=_optional_text(payload.get("task_id")),
            expected_output_format=_optional_text(payload.get("expected_output_format")),
            parser_id=_optional_text(payload.get("parser_id")),
            metadata=metadata,
        )

    def to_mapping(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "prompt_id": self.prompt_id,
            "construct_id": self.construct_id,
            "split": self.split,
            "prompt_role": self.prompt_role,
            "pair_id": self.pair_id or "",
            "pair_role": self.pair_role or "",
            "condition_id": self.condition_id or "",
            "prompt_family": self.prompt_family or "",
            "task_id": self.task_id or "",
            "prompt_text": self.prompt_text,
            "expected_output_format": self.expected_output_format or "",
            "parser_id": self.parser_id or "",
            "metadata_json": json.dumps(self.metadata, sort_keys=True, ensure_ascii=True),
        }
        for key, value in self.metadata.items():
            if key in _RESERVED_FIELDS:
                continue
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(value, sort_keys=True, ensure_ascii=True)
            else:
                row[key] = value
        return row


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    prompt_path = Path(path)
    if prompt_path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{prompt_path}:{line_number} is not valid JSON.") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{prompt_path}:{line_number} must contain a JSON object.")
                rows.append(row)
    else:
        rows = []
        with prompt_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{prompt_path} is missing a CSV header.")
            rows.extend(reader)
    records = [PromptRecord.from_mapping(row) for row in rows]
    if not records:
        raise ValueError(f"{prompt_path} contains no prompt records.")
    return records


def write_prompt_records(records: Iterable[PromptRecord], path: str | Path) -> int:
    prompt_path = Path(path)
    materialized = list(records)
    if not materialized:
        raise ValueError("Cannot write an empty prompt inventory.")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    if prompt_path.suffix.lower() == ".jsonl":
        with prompt_path.open("w", encoding="utf-8") as handle:
            for record in materialized:
                handle.write(json.dumps(record.to_mapping(), ensure_ascii=True) + "\n")
        return len(materialized)

    rows = [record.to_mapping() for record in materialized]
    extra_fields = sorted({key for row in rows for key in row if key not in PROMPT_FIELDS})
    fieldnames = [*PROMPT_FIELDS, *extra_fields]
    with prompt_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(materialized)


def combine_prompt_files(paths: Iterable[str | Path], output_path: str | Path) -> int:
    records: list[PromptRecord] = []
    for path in paths:
        records.extend(load_prompt_records(path))
    _validate_global_prompt_ids(records)
    return write_prompt_records(records, output_path)


def _validate_global_prompt_ids(records: Iterable[PromptRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if not record.prompt_id:
            raise ValueError("Every prompt must have a prompt_id.")
        if record.prompt_id in seen:
            raise ValueError(f"Duplicate prompt_id across the combined inventory: {record.prompt_id}")
        seen.add(record.prompt_id)


def validate_prompt_records(
    records: Iterable[PromptRecord],
    construct_specs: Mapping[str, ConstructSpec],
) -> dict[str, Any]:
    """Validate IDs, pairs, construct namespaces, and split coverage."""

    materialized = list(records)
    if not materialized:
        raise ValueError("Prompt inventory must contain at least one record.")
    _validate_global_prompt_ids(materialized)
    if not construct_specs:
        raise ValueError("At least one construct specification is required.")

    by_construct: dict[str, list[PromptRecord]] = {construct_id: [] for construct_id in construct_specs}
    pair_groups: dict[tuple[str, str, str], list[PromptRecord]] = {}
    counts_by_split: dict[str, dict[str, int]] = {construct_id: {} for construct_id in construct_specs}
    text_by_construct: dict[str, dict[str, str]] = {construct_id: {} for construct_id in construct_specs}
    family_roles_by_construct: dict[str, dict[str, str]] = {
        construct_id: {} for construct_id in construct_specs
    }

    for record in materialized:
        spec = construct_specs.get(record.construct_id)
        if spec is None:
            raise ValueError(f"Prompt {record.prompt_id} references unknown construct_id={record.construct_id!r}.")
        if not record.split or record.split not in spec.required_splits:
            raise ValueError(
                f"Prompt {record.prompt_id} has split={record.split!r}, which is not required "
                f"by {record.construct_id}."
            )
        if record.prompt_role not in PROMPT_ROLES:
            raise ValueError(f"Prompt {record.prompt_id} has unsupported prompt_role={record.prompt_role!r}.")
        expected_role = SPLIT_PROMPT_ROLE.get(record.split)
        if expected_role is None:
            raise ValueError(f"Prompt {record.prompt_id} has unsupported split={record.split!r}.")
        if record.prompt_role != expected_role:
            raise ValueError(
                f"Prompt {record.prompt_id} has role={record.prompt_role!r}, but split "
                f"{record.split!r} requires role={expected_role!r}."
            )
        if not record.prompt_text:
            raise ValueError(f"Prompt {record.prompt_id} has empty prompt_text.")
        if not record.prompt_family:
            raise ValueError(f"Prompt {record.prompt_id} requires prompt_family metadata.")
        normalized_text = _normalized_prompt_text(record.prompt_text)
        previous_prompt_id = text_by_construct[record.construct_id].get(normalized_text)
        if previous_prompt_id is not None:
            raise ValueError(
                f"Construct {record.construct_id} reuses normalized prompt text across records: "
                f"{previous_prompt_id!r} and {record.prompt_id!r}."
            )
        text_by_construct[record.construct_id][normalized_text] = record.prompt_id
        previous_role = family_roles_by_construct[record.construct_id].get(record.prompt_family)
        if previous_role is not None and previous_role != record.prompt_role:
            raise ValueError(
                f"Construct {record.construct_id} reuses prompt_family={record.prompt_family!r} "
                f"across roles {previous_role!r} and {record.prompt_role!r}."
            )
        family_roles_by_construct[record.construct_id][record.prompt_family] = record.prompt_role
        if record.prompt_role in {"behavior", "steering", "calibration"}:
            missing_fields = [
                field_name
                for field_name, value in (
                    ("task_id", record.task_id),
                    ("parser_id", record.parser_id),
                    ("expected_output_format", record.expected_output_format),
                )
                if not value
            ]
            if missing_fields:
                raise ValueError(
                    f"Prompt {record.prompt_id} is missing role-specific fields: {missing_fields}."
                )
        if record.split in spec.paired_splits:
            if not record.pair_id or not record.pair_role or not record.condition_id:
                raise ValueError(
                    f"Paired split prompt {record.prompt_id} requires pair_id, pair_role, and condition_id."
                )
            if record.condition_id not in spec.condition_ids:
                raise ValueError(f"Prompt {record.prompt_id} uses unknown condition_id={record.condition_id!r}.")
            if record.pair_role != record.condition_id:
                raise ValueError(f"Prompt {record.prompt_id} pair_role must equal condition_id.")
            pair_key = (record.construct_id, record.split, record.pair_id)
            pair_groups.setdefault(pair_key, []).append(record)
        elif record.condition_id not in {None, "", "neutral", *spec.condition_ids}:
            raise ValueError(f"Prompt {record.prompt_id} uses unknown condition_id={record.condition_id!r}.")

        by_construct[record.construct_id].append(record)
        counts_by_split[record.construct_id][record.split] = (
            counts_by_split[record.construct_id].get(record.split, 0) + 1
        )

    for construct_id, spec in construct_specs.items():
        if not by_construct[construct_id]:
            raise ValueError(f"No prompt records supplied for construct {construct_id}.")
        missing_splits = set(spec.required_splits) - set(counts_by_split[construct_id])
        if missing_splits:
            raise ValueError(f"Construct {construct_id} is missing required splits: {sorted(missing_splits)}")

    for pair_key, pair_records in pair_groups.items():
        construct_id, split, pair_id = pair_key
        spec = construct_specs[construct_id]
        if len(pair_records) != 2:
            raise ValueError(f"Pair {construct_id}/{split}/{pair_id} must contain exactly two prompts.")
        condition_ids = {record.condition_id for record in pair_records}
        if condition_ids != set(spec.condition_ids):
            raise ValueError(
                f"Pair {construct_id}/{split}/{pair_id} must contain both conditions {spec.condition_ids}."
            )

    return {
        "total_prompts": len(materialized),
        "construct_ids": sorted(construct_specs),
        "counts_by_construct": {construct_id: len(rows) for construct_id, rows in by_construct.items()},
        "counts_by_construct_split": {
            construct_id: dict(sorted(counts.items()))
            for construct_id, counts in counts_by_split.items()
        },
        "pair_count": len(pair_groups),
        "prompt_families_by_construct_role": {
            construct_id: {
                role: sorted(
                    family
                    for family, family_role in family_roles.items()
                    if family_role == role
                )
                for role in sorted(PROMPT_ROLES)
            }
            for construct_id, family_roles in family_roles_by_construct.items()
        },
    }
