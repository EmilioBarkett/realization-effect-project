"""Versioned construct-bank registry and spec cross-validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .schemas import ConstructSpec, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_REGISTRY_STATUSES = frozenset({"specified", "planned"})
_EXPECTED_FAMILIES = frozenset({"decision", "epistemic", "social", "agentic"})
_EXPECTED_WAVES = frozenset({1, 2, 3, 4})


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _identifier(value: Any, *, field_name: str) -> str:
    identifier = _text(value, field_name=field_name)
    if not _ID_PATTERN.fullmatch(identifier):
        raise ValueError(f"{field_name}={identifier!r} is not a valid lowercase identifier.")
    return identifier


def _validate_declared_topology(entries: tuple["ConstructRegistryEntry", ...]) -> None:
    if len(entries) != 16:
        raise ValueError("The v1 construct registry must declare exactly 16 entries in a 4x4 topology.")
    if {entry.wave for entry in entries} != _EXPECTED_WAVES:
        raise ValueError("The v1 construct registry must declare exactly waves 1, 2, 3, and 4.")
    if {entry.family for entry in entries} != _EXPECTED_FAMILIES:
        raise ValueError(
            "The v1 construct registry must use exactly the decision, epistemic, social, and agentic families."
        )
    for wave in sorted(_EXPECTED_WAVES):
        wave_entries = [entry for entry in entries if entry.wave == wave]
        if len(wave_entries) != 4 or {entry.family for entry in wave_entries} != _EXPECTED_FAMILIES:
            raise ValueError(f"Wave {wave} must contain exactly one construct from each registry family.")


@dataclass(frozen=True)
class ConstructRegistryEntry:
    construct_id: str
    family: str
    wave: int
    status: str
    spec_path: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, index: int) -> "ConstructRegistryEntry":
        payload = dict(data)
        construct_id = _identifier(payload.get("construct_id"), field_name=f"entries[{index}].construct_id")
        family = _identifier(payload.get("family"), field_name=f"entries[{index}].family")
        wave = payload.get("wave")
        if not isinstance(wave, int) or wave < 1:
            raise ValueError(f"entries[{index}].wave must be a positive integer.")
        status = _text(payload.get("status"), field_name=f"entries[{index}].status")
        if status not in _REGISTRY_STATUSES:
            raise ValueError(
                f"entries[{index}].status must be one of {sorted(_REGISTRY_STATUSES)}."
            )
        spec_path = _text(payload.get("spec_path"), field_name=f"entries[{index}].spec_path")
        return cls(
            construct_id=construct_id,
            family=family,
            wave=wave,
            status=status,
            spec_path=spec_path,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "construct_id": self.construct_id,
            "family": self.family,
            "wave": self.wave,
            "status": self.status,
            "spec_path": self.spec_path,
        }


@dataclass(frozen=True)
class ConstructRegistry:
    registry_id: str
    version: str
    entries: tuple[ConstructRegistryEntry, ...]
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ConstructRegistry":
        payload = dict(data)
        schema_version = _text(payload.get("schema_version", SCHEMA_VERSION), field_name="schema_version")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version={schema_version!r}; supported versions are "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}."
            )
        registry_id = _identifier(payload.get("registry_id"), field_name="registry_id")
        version = _text(payload.get("version"), field_name="version")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("entries must be a non-empty list.")
        entries = tuple(
            ConstructRegistryEntry.from_mapping(entry, index=index)
            for index, entry in enumerate(raw_entries)
            if isinstance(entry, Mapping)
        )
        if len(entries) != len(raw_entries):
            raise ValueError("Every registry entry must be an object.")
        construct_ids = [entry.construct_id for entry in entries]
        if len(set(construct_ids)) != len(construct_ids):
            raise ValueError("Registry construct_id values must be unique.")
        _validate_declared_topology(entries)
        return cls(
            registry_id=registry_id,
            version=version,
            entries=entries,
            schema_version=schema_version,
        )

    @property
    def entries_by_id(self) -> dict[str, ConstructRegistryEntry]:
        return {entry.construct_id: entry for entry in self.entries}

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "version": self.version,
            "entries": [entry.to_mapping() for entry in self.entries],
        }


def load_construct_registry(path: str | Path) -> ConstructRegistry:
    registry_path = Path(path)
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{registry_path} is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{registry_path} must contain a JSON object.")
    return ConstructRegistry.from_mapping(data)


def validate_registry_against_specs(
    registry: ConstructRegistry,
    construct_specs: Mapping[str, ConstructSpec],
) -> dict[str, Any]:
    """Verify that loaded specs agree with their frozen registry entries."""

    entries = registry.entries_by_id
    loaded_ids = set(construct_specs)
    unknown_specs = loaded_ids - set(entries)
    if unknown_specs:
        raise ValueError(f"Loaded construct specs are absent from the registry: {sorted(unknown_specs)}")

    specified_ids = {entry.construct_id for entry in registry.entries if entry.status == "specified"}
    missing_specs = specified_ids - loaded_ids
    if missing_specs:
        raise ValueError(f"Registry entries marked specified have no loaded spec: {sorted(missing_specs)}")

    for construct_id, spec in construct_specs.items():
        entry = entries[construct_id]
        if entry.status != "specified":
            raise ValueError(
                f"Construct {construct_id} is loaded but registry status is {entry.status!r}."
            )
        if spec.family != entry.family:
            raise ValueError(
                f"Construct {construct_id} family mismatch: registry={entry.family!r}, spec={spec.family!r}."
            )

    return {
        "registry_id": registry.registry_id,
        "registry_version": registry.version,
        "construct_count": len(registry.entries),
        "specified_count": len(specified_ids),
        "planned_count": len(registry.entries) - len(specified_ids),
        "construct_ids_by_wave": {
            str(wave): [
                entry.construct_id
                for entry in registry.entries
                if entry.wave == wave
            ]
            for wave in sorted({entry.wave for entry in registry.entries})
        },
        "loaded_spec_ids": sorted(loaded_ids),
    }
