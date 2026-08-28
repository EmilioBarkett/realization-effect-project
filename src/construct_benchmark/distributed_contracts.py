"""Small, dependency-light contracts shared by distributed benchmark stages.

The distributed workers intentionally exchange plain JSON and CSV/JSONL
artifacts.  This module keeps the pieces that must have exactly one
implementation: canonical hashing, strict JSON loading, stable ranking, and
atomic no-overwrite writes.  It does not know anything about a model or a
particular construct schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DISTRIBUTED_SCHEMA_VERSION = "1.0.0"
UNSPECIFIED_RUN_CONFIG_HASH = "UNSPECIFIED"
_VERSION_TOKEN_RE = re.compile(r"(?<![a-z0-9])v([12])(?=$|[^a-z0-9])", re.IGNORECASE)
_VERSION_TOKEN_RE_ALT = re.compile(r"(?:^|[_-])v([12])(?:$|[_-])", re.IGNORECASE)

# Task/parser/output-format versions describe the consumer of a prompt, not
# the prompt inventory family.  Only fields that identify prompt, inventory,
# generation, construct, or artifact provenance may establish v1/v2 family.
_VERSION_PROVENANCE_KEYS = frozenset(
    {
        "prompt_id",
        "request_id",
        "generation_plan_id",
        "generation_batch_id",
        "construct",
        "construct_id",
        "inventory",
        "inventory_id",
        "artifact",
        "artifact_id",
        "version",
        "prompt_version",
        "inventory_version",
        "artifact_version",
        "construct_version",
        "prompt_variant",
        "prompt_variant_id",
        "inventory_variant",
        "inventory_variant_id",
        "artifact_variant",
        "artifact_variant_id",
        "construct_variant",
        "construct_variant_id",
    }
)


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for contract hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value with cross-process stable serialization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""

    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(seed: int | str, *parts: object) -> str:
    """Return a deterministic digest for ranking IDs.

    Length-prefixing each component avoids accidental ambiguity such as
    ``("ab", "c")`` versus ``("a", "bc")`` and avoids Python's process-
    randomized ``hash()`` implementation.
    """

    encoded_parts = [str(seed), *(str(part) for part in parts)]
    payload = b"".join(
        len(part.encode("utf-8")).to_bytes(8, "big")
        + part.encode("utf-8")
        for part in encoded_parts
    )
    return hashlib.sha256(payload).hexdigest()


def provenance_version_families(value: Any) -> frozenset[str]:
    """Return v1/v2 markers from prompt/inventory provenance fields only.

    Benchmark prompts can legitimately use a v2 prompt inventory with a v1
    task or parser implementation.  Scanning every ``*_id`` field therefore
    produces false contamination failures.  This traversal intentionally
    ignores task IDs, parser IDs, expected output-format IDs, and prompt text.
    """

    families: set[str] = set()

    def visit(node: Any, key: str = "") -> None:
        if isinstance(node, Mapping):
            for nested_key, nested_value in node.items():
                visit(nested_value, str(nested_key))
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item, key)
            return
        normalized = key.strip().casefold().replace("-", "_")
        if isinstance(node, str) and normalized.endswith("_json"):
            try:
                parsed = json.loads(node)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (Mapping, list, tuple)):
                visit(parsed, key)
                return
        if normalized not in _VERSION_PROVENANCE_KEYS:
            return
        matches = [*_VERSION_TOKEN_RE.findall(str(node)), *_VERSION_TOKEN_RE_ALT.findall(str(node))]
        families.update(f"v{match.lower()}" for match in matches)

    visit(value)
    return frozenset(families)


def load_json_object(path: str | Path, *, label: str = "JSON file") -> dict[str, Any]:
    """Load a JSON object and turn parse/type failures into clear errors."""

    file_path = Path(path)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {file_path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {file_path}")
    return value


def atomic_write_text(path: str | Path, text: str, *, label: str = "output") -> None:
    """Atomically create a file, refusing to replace an existing artifact.

    A temporary file is created in the destination directory and linked into
    place.  ``link`` is used instead of ``replace`` so a concurrent writer
    cannot silently overwrite an immutable artifact after the existence check.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing {label}: {destination}")

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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite existing {label}: {destination}") from exc
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def atomic_write_json(path: str | Path, value: Mapping[str, Any], *, label: str = "manifest") -> None:
    """Serialize and atomically create a JSON object."""

    atomic_write_text(
        path,
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        label=label,
    )


def nonempty_text(value: Any, *, field_name: str) -> str:
    """Normalize a required identifier-like string."""

    if value is None:
        raise ValueError(f"{field_name} must be a non-empty string.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def bool_value(value: Any, *, field_name: str) -> bool:
    """Parse a strict boolean from JSON/CSV-friendly values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field_name} must be a boolean.")


__all__ = [
    "DISTRIBUTED_SCHEMA_VERSION",
    "UNSPECIFIED_RUN_CONFIG_HASH",
    "atomic_write_json",
    "atomic_write_text",
    "bool_value",
    "canonical_hash",
    "canonical_json",
    "file_sha256",
    "load_json_object",
    "nonempty_text",
    "provenance_version_families",
    "stable_digest",
]
