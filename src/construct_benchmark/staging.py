"""Portable, hash-verified staging bundles for model-side benchmark runs.

Staging is deliberately separate from model execution.  It copies only files
explicitly named by the caller into a self-contained directory and refuses to
replace a file or manifest with different content.  This makes it safe to
prepare a RunPod volume before premium compute starts while preserving the
source paths and hashes needed for later provenance checks.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .distributed_contracts import atomic_write_text


STAGING_SCHEMA_VERSION = "0.1.0"
STAGING_MANIFEST_NAME = "staging_manifest.json"


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Staging input must be a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically without leaving a partial manifest."""

    serialized = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, serialized, label="staging manifest")


def _safe_label(label: str) -> str:
    normalized = label.strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("Staging labels must be non-empty and path-safe.")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in normalized):
        raise ValueError(f"Staging label is not path-safe: {label!r}")
    return normalized


def _copy_if_identical(source: Path, destination: Path) -> None:
    """Copy one input, refusing to overwrite a different staged artifact."""

    source_hash = file_sha256(source)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or file_sha256(destination) != source_hash:
            raise ValueError(f"Refusing to overwrite a different staged file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.staging.tmp")
    try:
        shutil.copy2(source, temporary)
        if file_sha256(temporary) != source_hash:
            raise IOError(f"Staged copy hash changed while copying: {source}")
        try:
            # A hard link makes the final creation atomic and refuses a
            # concurrent writer's destination instead of replacing it.
            destination.hardlink_to(temporary)
        except FileExistsError as exc:
            raise ValueError(f"Refusing to overwrite a staged file created concurrently: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stage_bundle(
    output_dir: str | Path,
    files: Mapping[str, str | Path],
    *,
    bundle_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or verify an immutable portable staging bundle.

    ``files`` maps safe logical labels (for example ``inventory`` or
    ``construct_source_reliability``) to source files.  The resulting bundle
    stores each file under ``inputs/<label><suffix>``.  Repeating the call is
    idempotent when all hashes match; a changed input or manifest is rejected.
    """

    root = Path(output_dir).expanduser().resolve()
    if not files:
        raise ValueError("A staging bundle requires at least one input file.")
    normalized: dict[str, Path] = {}
    for raw_label, raw_path in files.items():
        label = _safe_label(str(raw_label))
        if label in normalized:
            raise ValueError(f"Duplicate staging label: {label}")
        source = Path(raw_path).expanduser().resolve()
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Staging input does not exist as a regular file: {source}")
        normalized[label] = source

    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for label, source in sorted(normalized.items()):
        staged_name = f"{label}{source.suffix}"
        staged = root / "inputs" / staged_name
        digest = file_sha256(source)
        _copy_if_identical(source, staged)
        entries.append(
            {
                "label": label,
                "source_path": str(source),
                "staged_path": str(staged.relative_to(root)),
                "sha256": digest,
                "size_bytes": source.stat().st_size,
            }
        )

    manifest_path = root / STAGING_MANIFEST_NAME
    candidate: dict[str, Any] = {
        "schema_version": STAGING_SCHEMA_VERSION,
        "manifest_type": "benchmark_staging_bundle",
        "bundle_id": bundle_id or root.name,
        "status": "staged",
        "created_at": _utc_now(),
        "input_count": len(entries),
        "inputs": entries,
        "metadata": dict(metadata or {}),
        "policy": {
            "external_calls": False,
            "model_weights_loaded": False,
            "credentials_copied": False,
            "overwrite_policy": "refuse_different_content",
        },
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Existing staging manifest is unreadable: {manifest_path}") from exc
        comparable_existing = dict(existing)
        comparable_candidate = dict(candidate)
        comparable_existing.pop("created_at", None)
        comparable_candidate.pop("created_at", None)
        if comparable_existing != comparable_candidate:
            raise ValueError(f"Refusing to overwrite a different staging manifest: {manifest_path}")
        candidate = existing
    else:
        _atomic_write_json(manifest_path, candidate)
    return candidate


def validate_staging_bundle(path: str | Path) -> dict[str, Any]:
    """Validate a staging manifest and every staged input hash."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / STAGING_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing staging manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid staging manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("manifest_type") != "benchmark_staging_bundle":
        raise ValueError(f"Not a benchmark staging bundle: {manifest_path}")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != manifest.get("input_count"):
        raise ValueError("Staging manifest input count does not match its entries.")
    labels: set[str] = set()
    for entry in inputs:
        if not isinstance(entry, dict):
            raise ValueError("Staging manifest contains a malformed input entry.")
        label = _safe_label(str(entry.get("label", "")))
        if label in labels:
            raise ValueError(f"Duplicate staged label: {label}")
        labels.add(label)
        relative = Path(str(entry.get("staged_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe staged path: {relative}")
        target = root / relative
        if not target.is_file() or file_sha256(target) != entry.get("sha256"):
            raise ValueError(f"Staged input hash mismatch: {target}")
    return manifest


__all__ = ["STAGING_MANIFEST_NAME", "STAGING_SCHEMA_VERSION", "file_sha256", "stage_bundle", "validate_staging_bundle"]
