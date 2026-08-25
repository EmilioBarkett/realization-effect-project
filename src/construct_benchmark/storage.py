"""Portable run storage, checksums, and S3-compatible archive helpers.

The benchmark deliberately separates three concerns:

* a workspace where a local machine or RunPod can execute a run;
* a durable archive addressed by an S3 URI; and
* public release artifacts, which are curated separately from raw outputs.

This module only implements the first two. It never stores credentials in a
run manifest and does not require an S3 client during ordinary development.
The AWS CLI is invoked only by an explicit archive/finalize command.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .manifests import canonical_hash, file_sha256
from .schemas import RunConfig


CHECKSUMS_FILENAME = "checksums.sha256"
STORAGE_MANIFEST_FILENAME = "storage_manifest.json"
ARCHIVE_RECEIPT_FILENAME = "archive_receipt.json"
_NON_ARTIFACT_FILENAMES = frozenset(
    {CHECKSUMS_FILENAME, STORAGE_MANIFEST_FILENAME, ARCHIVE_RECEIPT_FILENAME, "run_status.json"}
)


@dataclass(frozen=True)
class StorageLayout:
    """Resolved physical paths for one run."""

    workspace_root: Path
    output_root: Path
    run_root: Path

    @property
    def raw_root(self) -> Path:
        return self.run_root / "raw"

    @property
    def config_root(self) -> Path:
        return self.run_root / "config_snapshot"

    @property
    def prompts_root(self) -> Path:
        return self.run_root / "prompts"

    @property
    def activations_root(self) -> Path:
        return self.run_root / "activations"

    @property
    def plan_path(self) -> Path:
        return self.run_root / "run_plan.json"

    @property
    def storage_manifest_path(self) -> Path:
        return self.run_root / STORAGE_MANIFEST_FILENAME

    @property
    def checksums_path(self) -> Path:
        return self.run_root / CHECKSUMS_FILENAME

    @property
    def archive_receipt_path(self) -> Path:
        return self.run_root / ARCHIVE_RECEIPT_FILENAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_storage_layout(run_config: RunConfig, workspace_root: str | Path | None = None) -> StorageLayout:
    """Resolve a run's workspace and output paths without changing the config.

    Relative ``output_root`` values are anchored at ``workspace_root``. When
    no explicit workspace is supplied, the configured environment variable is
    used, falling back to the current working directory.
    """

    workspace_value = workspace_root
    if workspace_value is None:
        workspace_value = os.environ.get(str(run_config.storage["workspace_root_env"]), ".")
    workspace = Path(workspace_value).expanduser()
    if not workspace.is_absolute():
        workspace = Path.cwd() / workspace
    workspace = workspace.resolve()

    output = Path(run_config.output_root).expanduser()
    if not output.is_absolute():
        output = workspace / output
    output = output.resolve()
    return StorageLayout(workspace_root=workspace, output_root=output, run_root=output / run_config.run_id)


def layout_for_existing_run(
    run_config: RunConfig,
    run_root: str | Path,
    workspace_root: str | Path | None = None,
) -> StorageLayout:
    """Build a layout for a run directory that was already prepared."""

    resolved_run_root = Path(run_root).expanduser().resolve()
    if workspace_root is None:
        workspace = resolved_run_root.parent
    else:
        workspace = Path(workspace_root).expanduser().resolve()
    return StorageLayout(
        workspace_root=workspace,
        output_root=resolved_run_root.parent,
        run_root=resolved_run_root,
    )


def prepare_run_directories(run_config: RunConfig, layout: StorageLayout) -> None:
    """Create only the deterministic directories needed by a run."""

    directories = [
        layout.run_root,
        layout.raw_root,
        layout.config_root,
        layout.prompts_root,
        layout.activations_root,
        layout.run_root / "constructs",
    ]
    for construct_id in run_config.construct_ids:
        construct_root = layout.run_root / "constructs" / construct_id
        directories.extend(
            construct_root / name
            for name in (
                "direction",
                "readout",
                "calibration",
                "behavior_baseline",
                "behavior_steered",
                "steering",
            )
        )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_archive_uri(uri: str) -> str:
    """Validate a credential-free S3 URI used as an archive prefix."""

    normalized = uri.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("archive URI must be a credential-free s3://bucket[/prefix] URI.")
    if parsed.query or parsed.fragment:
        raise ValueError("archive URI must not contain a query string or fragment.")
    return normalized


def resolve_archive_uri(run_config: RunConfig, archive_uri: str | None = None) -> str | None:
    """Resolve an archive URI from a CLI override or the configured env var."""

    value = archive_uri
    if value is None:
        value = os.environ.get(str(run_config.storage["archive_uri_env"]))
    if value is None or not value.strip():
        return None
    return validate_archive_uri(value)


def archive_destination(archive_uri: str, run_id: str) -> str:
    """Return the immutable per-run destination under an archive prefix."""

    return f"{validate_archive_uri(archive_uri)}/{run_id}"


def build_storage_manifest(
    run_config: RunConfig,
    layout: StorageLayout,
    *,
    status: str,
    plan_path: str | Path | None = None,
    archive_uri: str | None = None,
    archive_status: str = "not_synced",
    archived_at: str | None = None,
) -> dict[str, Any]:
    """Build a provenance manifest without reading any secret values."""

    resolved_archive = validate_archive_uri(archive_uri) if archive_uri else None
    plan = Path(plan_path) if plan_path is not None else layout.plan_path
    manifest: dict[str, Any] = {
        "schema_version": run_config.schema_version,
        "manifest_type": "benchmark_storage",
        "run_id": run_config.run_id,
        "status": status,
        "updated_at": _utc_now(),
        "workspace": {
            "root": str(layout.workspace_root),
            "output_root": str(layout.output_root),
            "run_root": str(layout.run_root),
        },
        "artifact_layout": {
            "raw": str(layout.raw_root),
            "prompts": str(layout.prompts_root),
            "activations": str(layout.activations_root),
            "config_snapshot": str(layout.config_root),
            "constructs": str(layout.run_root / "constructs"),
        },
        "archive": {
            "uri_env": run_config.storage["archive_uri_env"],
            "sync_endpoint_env": run_config.storage["sync_endpoint_env"],
            "sync_tool": run_config.storage["sync_tool"],
            "sync_on_finalize": run_config.storage["sync_on_finalize"],
            "keep_local_copy": run_config.storage["keep_local_copy"],
            "status": archive_status,
        },
        "provenance": {
            "run_config_sha256": canonical_hash(run_config.to_mapping()),
            "model_id": run_config.model["model_id"],
            "model_revision": run_config.model.get("revision"),
            "construct_ids": list(run_config.construct_ids),
        },
    }
    if resolved_archive is not None:
        manifest["archive"]["uri"] = resolved_archive
        manifest["archive"]["destination"] = archive_destination(resolved_archive, run_config.run_id)
    if archived_at is not None:
        manifest["archive"]["archived_at"] = archived_at
    if plan.is_file():
        manifest["provenance"]["run_plan_sha256"] = file_sha256(plan)
    if layout.checksums_path.is_file():
        manifest["provenance"]["checksums_sha256"] = file_sha256(layout.checksums_path)
    return manifest


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_storage_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _artifact_paths(run_root: Path) -> list[Path]:
    paths: list[Path] = []
    for candidate in sorted(run_root.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        if candidate.name in _NON_ARTIFACT_FILENAMES:
            continue
        paths.append(candidate)
    return paths


def write_checksums(run_root: str | Path) -> Path:
    """Write deterministic SHA-256 checksums for scientific run artifacts."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root}")
    checksum_path = root / CHECKSUMS_FILENAME
    lines = []
    for path in _artifact_paths(root):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{file_sha256(path)}  {relative}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return checksum_path


def verify_checksums(run_root: str | Path) -> dict[str, Any]:
    """Verify a run's checksum file and return a compact audit summary."""

    root = Path(run_root).resolve()
    checksum_path = root / CHECKSUMS_FILENAME
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing checksum file: {checksum_path}")
    checked = 0
    failures: list[str] = []
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid checksum line {line_number} in {checksum_path}.") from exc
        target = root / relative
        checked += 1
        if not target.is_file() or file_sha256(target) != expected:
            failures.append(relative)
    return {"run_root": str(root), "checked_files": checked, "failures": failures, "valid": not failures}


def build_sync_command(
    run_config: RunConfig,
    layout: StorageLayout,
    archive_uri: str,
    *,
    dry_run: bool = False,
    endpoint_url: str | None = None,
) -> list[str]:
    """Build an AWS CLI sync command without embedding credentials."""

    destination = archive_destination(archive_uri, run_config.run_id)
    command = [
        str(run_config.storage["sync_tool"]),
        "s3",
        "sync",
        str(layout.run_root),
        destination,
        "--only-show-errors",
        "--no-progress",
    ]
    if endpoint_url:
        command.extend(["--endpoint-url", endpoint_url])
    if dry_run:
        command.append("--dryrun")
    return command


def run_sync_command(command: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    """Run an archive command, or return it without executing in dry-run mode."""

    if dry_run:
        return {"ran": False, "returncode": None, "command": command}
    if shutil.which(command[0]) is None:
        raise FileNotFoundError(
            f"Archive tool {command[0]!r} is not installed or not on PATH; "
            "install/configure the AWS CLI before syncing."
        )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no command output"
        raise RuntimeError(f"Archive command failed with exit code {completed.returncode}: {detail}")
    return {"ran": True, "returncode": completed.returncode, "command": command}


def write_archive_receipt(
    layout: StorageLayout,
    *,
    archive_uri: str,
    command: list[str],
    completed_at: str,
) -> None:
    """Record successful archive completion without recording command output."""

    write_json(
        layout.archive_receipt_path,
        {
            "schema_version": "0.1.0",
            "manifest_type": "benchmark_archive_receipt",
            "run_id": layout.run_root.name,
            "destination": archive_destination(archive_uri, layout.run_root.name),
            "command": command,
            "completed_at": completed_at,
            "checksums_sha256": file_sha256(layout.checksums_path)
            if layout.checksums_path.is_file()
            else None,
        },
    )
