#!/usr/bin/env python3
"""Validate the canonical repository-relative model preflight artifact index.

The index is intentionally narrower than an execution configuration.  It
binds each wave/model pair to the exact inventory, selection, gate, and run
configuration files that an executor may use.  A pending or invalid reference
always blocks readiness; this command never contacts an API or a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = _ROOT / (
    "configs/construct_benchmark/preflight_campaigns/"
    "waves1_4_preflight_artifact_index_v1.json"
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN_ROOT_MARKERS = ("superseded", "failed", "failure", "invalidated")
_BLOCKED_JSON_STATUSES = frozenset({"failed", "superseded", "invalidated"})
_ARTIFACT_NAMES = ("inventory", "selection", "gate", "run_config")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _issue(code: str, message: str, *, scope: str | None = None) -> dict[str, str]:
    result = {"code": code, "message": message}
    if scope is not None:
        result["scope"] = scope
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.strip()))


def _path_parts(value: str) -> tuple[str, ...]:
    # Treat Windows separators as separators too, even when validation runs on
    # POSIX.  This prevents a path written on one platform from bypassing the
    # traversal check on another.
    return tuple(PurePosixPath(value.replace("\\", "/")).parts)


def _forbidden_path_token(value: str) -> str | None:
    lowered_parts = tuple(part.casefold() for part in _path_parts(value))
    for part in lowered_parts:
        for marker in _FORBIDDEN_ROOT_MARKERS:
            if marker in part:
                return marker
    return None


def _safe_path(
    value: Any,
    *,
    repo_root: Path,
    base: Path,
    label: str,
    issues: list[dict[str, str]],
    require_repo_relative: bool = True,
) -> Path | None:
    """Resolve a path and record a fail-closed issue when it is unsafe."""

    if not isinstance(value, str) or not value.strip():
        issues.append(_issue("missing_path", f"{label} must be a non-empty path."))
        return None
    raw = value.strip()
    if "\x00" in raw:
        issues.append(_issue("unsafe_path", f"{label} contains a NUL byte."))
        return None
    if require_repo_relative:
        windows_path = PureWindowsPath(raw)
        if Path(raw).is_absolute() or windows_path.is_absolute() or bool(windows_path.drive):
            issues.append(_issue("absolute_path", f"{label} must be repository-relative: {raw!r}."))
            return None
        if ".." in _path_parts(raw):
            issues.append(_issue("path_traversal", f"{label} contains path traversal: {raw!r}."))
            return None
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw):
            issues.append(_issue("external_path", f"{label} must not be a URI: {raw!r}."))
            return None

    marker = _forbidden_path_token(raw)
    if marker is not None:
        issues.append(
            _issue(
                "forbidden_root",
                f"{label} references a {marker!r} artifact root: {raw!r}.",
            )
        )
        return None

    candidate = (base / raw).resolve(strict=False)
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        issues.append(
            _issue(
                "external_path",
                f"{label} resolves outside the repository root: {raw!r}.",
            )
        )
        return None

    # A marker file in an ancestor makes an otherwise innocuous-looking path
    # part of an explicitly failed/superseded root.  Check existing ancestors
    # only; absent pending paths are handled by their pending status.
    ancestor = candidate if candidate.is_dir() else candidate.parent
    while True:
        for marker_name in ("SUPERSEDED.json", "FAILED.json", "FAILURE.json", "INVALIDATED.json"):
            if (ancestor / marker_name).is_file():
                issues.append(
                    _issue(
                        "forbidden_root",
                        f"{label} is under a root marked {marker_name}: {candidate}.",
                    )
                )
                return None
        if ancestor == repo_root or repo_root not in ancestor.parents:
            break
        ancestor = ancestor.parent
    return candidate


def _load_json(path: Path, *, label: str, issues: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(_issue("invalid_json", f"{label} is not readable JSON: {exc}."))
        return None
    if not isinstance(payload, dict):
        issues.append(_issue("invalid_json", f"{label} must contain a JSON object."))
        return None
    status = payload.get("status")
    if isinstance(status, str) and any(marker in status.casefold() for marker in _BLOCKED_JSON_STATUSES):
        issues.append(_issue("blocked_status", f"{label} has blocked status {status!r}."))
    if payload.get("do_not_merge") is True or payload.get("superseded") is True:
        issues.append(_issue("blocked_status", f"{label} is explicitly marked superseded/do_not_merge."))
    return payload


def _validate_declared_hash(
    artifact: Mapping[str, Any],
    *,
    path: Path,
    label: str,
    issues: list[dict[str, str]],
) -> bool:
    declared = artifact.get("sha256")
    if not _is_sha256(declared):
        issues.append(_issue("missing_hash", f"{label}.sha256 must be a 64-character SHA-256 digest."))
        return False
    actual = file_sha256(path)
    if actual.casefold() != str(declared).casefold():
        issues.append(
            _issue(
                "hash_mismatch",
                f"{label} SHA-256 mismatch: declared {declared}, actual {actual}.",
            )
        )
        return False
    return True


def _artifact(
    entry: Mapping[str, Any],
    name: str,
    *,
    repo_root: Path,
    entry_scope: str,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    artifacts = entry.get("artifacts")
    raw_artifact = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    label = f"{entry_scope}/{name}"
    if not isinstance(raw_artifact, Mapping):
        issue = _issue("missing_artifact", f"{label} is missing an artifact object.", scope=entry_scope)
        issues.append(issue)
        return {"status": "fail", "issues": [issue]}

    status = raw_artifact.get("status")
    if status not in {"ready", "pending"}:
        issue = _issue(
            "invalid_status",
            f"{label}.status must be 'ready' or 'pending', got {status!r}.",
            scope=entry_scope,
        )
        issues.append(issue)
        return {"status": "fail", "issues": [issue]}

    local_issues: list[dict[str, str]] = []
    path_value = raw_artifact.get("path")
    path = None
    if path_value is not None:
        path = _safe_path(
            path_value,
            repo_root=repo_root,
            base=repo_root,
            label=f"{label}.path",
            issues=local_issues,
        )
    if status == "pending":
        pending = _issue(
            "pending",
            str(raw_artifact.get("reason") or f"{label} is explicitly pending."),
            scope=entry_scope,
        )
        local_issues.append(pending)
        if raw_artifact.get("sha256") is not None and not _is_sha256(raw_artifact.get("sha256")):
            local_issues.append(
                _issue("invalid_hash", f"{label}.sha256 is not a valid SHA-256 digest.", scope=entry_scope)
            )
        issues.extend(local_issues)
        return {
            "status": "pending" if not any(item["code"] != "pending" for item in local_issues) else "fail",
            "path": str(path) if path is not None else None,
            "sha256": raw_artifact.get("sha256"),
            "payload": None,
            "issues": local_issues,
        }

    if path is None:
        local_issues.append(_issue("missing_path", f"{label}.path is required for a ready artifact."))
    elif not path.is_file():
        local_issues.append(_issue("missing_file", f"{label} does not exist: {path}."))
    else:
        _validate_declared_hash(raw_artifact, path=path, label=label, issues=local_issues)

    payload = None
    if path is not None and path.is_file() and path.suffix.casefold() == ".json":
        payload = _load_json(path, label=label, issues=local_issues)
    issues.extend(local_issues)
    return {
        "status": "pass" if not local_issues else "fail",
        "path": str(path) if path is not None else None,
        "sha256": raw_artifact.get("sha256"),
        "payload": payload,
        "issues": local_issues,
    }


def _validate_inventory_manifest(
    entry: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    repo_root: Path,
    entry_scope: str,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    inventory_result = inventory.get("result")
    if not isinstance(inventory_result, Mapping) or inventory_result.get("status") != "pass":
        return {"status": "fail"}
    raw_inventory = entry.get("artifacts", {}).get("inventory", {})
    manifest_spec = raw_inventory.get("manifest") if isinstance(raw_inventory, Mapping) else None
    label = f"{entry_scope}/inventory.manifest"
    if not isinstance(manifest_spec, Mapping):
        issue = _issue("missing_manifest", f"{label} is required for a ready inventory.", scope=entry_scope)
        issues.append(issue)
        return {"status": "fail", "issues": [issue]}

    local_issues: list[dict[str, str]] = []
    manifest_status = manifest_spec.get("status")
    if manifest_status != "ready":
        local_issues.append(
            _issue("invalid_status", f"{label}.status must be 'ready' when the inventory is ready.", scope=entry_scope)
        )
    manifest_path = _safe_path(
        manifest_spec.get("path"),
        repo_root=repo_root,
        base=repo_root,
        label=f"{label}.path",
        issues=local_issues,
    )
    manifest_payload = None
    if manifest_path is None:
        pass
    elif not manifest_path.is_file():
        local_issues.append(_issue("missing_file", f"{label} does not exist: {manifest_path}.", scope=entry_scope))
    else:
        _validate_declared_hash(manifest_spec, path=manifest_path, label=label, issues=local_issues)
        manifest_payload = _load_json(manifest_path, label=label, issues=local_issues)

    inventory_path = Path(str(inventory_result.get("path"))) if inventory_result.get("path") else None
    if manifest_payload is not None and inventory_path is not None:
        output_value = manifest_payload.get("output_path", manifest_payload.get("combined_path"))
        if output_value is None:
            local_issues.append(_issue("missing_cross_reference", f"{label} has no output_path/combined_path."))
        else:
            output_path = _safe_path(
                output_value,
                repo_root=repo_root,
                base=manifest_path.parent if manifest_path is not None else repo_root,
                label=f"{label}.output_path",
                issues=local_issues,
            )
            if output_path is not None and output_path != inventory_path:
                local_issues.append(
                    _issue(
                        "cross_reference_mismatch",
                        f"{label}.output_path does not point to the indexed inventory.",
                        scope=entry_scope,
                    )
                )
        inventory_hash = str(raw_inventory.get("sha256") or "").casefold()
        for hash_key in ("output_sha256", "combined_sha256", "combined_csv_sha256"):
            declared = manifest_payload.get(hash_key)
            if declared is not None and str(declared).casefold() != inventory_hash:
                local_issues.append(
                    _issue(
                        "cross_reference_hash_mismatch",
                        f"{label}.{hash_key} does not match the indexed inventory SHA-256.",
                        scope=entry_scope,
                    )
                )
        if manifest_payload.get("wave") is not None and manifest_payload.get("wave") != entry.get("wave"):
            local_issues.append(
                _issue("cross_reference_mismatch", f"{label}.wave does not match the indexed wave.", scope=entry_scope)
            )
        declared_constructs = manifest_payload.get("construct_ids")
        expected_constructs = entry.get("construct_ids")
        if isinstance(declared_constructs, list) and isinstance(expected_constructs, list):
            if set(declared_constructs) != set(expected_constructs):
                local_issues.append(
                    _issue(
                        "cross_reference_mismatch",
                        f"{label}.construct_ids do not match the indexed construct set.",
                        scope=entry_scope,
                    )
                )

    issues.extend(local_issues)
    return {
        "status": "pass" if not local_issues else "fail",
        "path": str(manifest_path) if manifest_path is not None else None,
        "payload": manifest_payload,
        "issues": local_issues,
    }


def _validate_selection_cross_references(
    entry: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    repo_root: Path,
    entry_scope: str,
    issues: list[dict[str, str]],
) -> None:
    selection = artifacts.get("selection")
    inventory = artifacts.get("inventory")
    gate = artifacts.get("gate")
    if not isinstance(selection, Mapping) or selection.get("status") != "pass":
        return
    payload = selection.get("payload")
    if not isinstance(payload, Mapping):
        return
    raw_selection = entry.get("artifacts", {}).get("selection", {})
    expected_selection_hash = raw_selection.get("selection_sha256") if isinstance(raw_selection, Mapping) else None
    if expected_selection_hash is not None:
        declared_selection_hash = payload.get("selection_sha256")
        if not _is_sha256(expected_selection_hash) or not _is_sha256(declared_selection_hash):
            issues.append(
                _issue(
                    "invalid_selection_hash",
                    f"{entry_scope}/selection selection_sha256 must be a valid SHA-256 digest.",
                    scope=entry_scope,
                )
            )
        elif str(expected_selection_hash).casefold() != str(declared_selection_hash).casefold():
            issues.append(
                _issue(
                    "cross_reference_hash_mismatch",
                    f"{entry_scope}/selection selection_sha256 does not match the indexed value.",
                    scope=entry_scope,
                )
            )
    if not isinstance(inventory, Mapping) or inventory.get("status") != "pass":
        issues.append(
            _issue(
                "selection_inventory_pending",
                f"{entry_scope}/selection cannot be ready while its indexed inventory is not ready.",
                scope=entry_scope,
            )
        )
        return
    source_inventory = payload.get("source_inventory")
    if not isinstance(source_inventory, str) or not source_inventory.strip():
        issues.append(_issue("missing_cross_reference", f"{entry_scope}/selection has no source_inventory.", scope=entry_scope))
    else:
        source_path = _safe_path(
            source_inventory,
            repo_root=repo_root,
            base=repo_root,
            label=f"{entry_scope}/selection.source_inventory",
            issues=issues,
        )
        if source_path is not None and source_path != Path(str(inventory.get("path"))):
            issues.append(
                _issue(
                    "cross_reference_mismatch",
                    f"{entry_scope}/selection.source_inventory does not match the indexed inventory.",
                    scope=entry_scope,
                )
            )
    source_hash = payload.get("source_inventory_sha256")
    inventory_hash = inventory.get("sha256")
    if not _is_sha256(source_hash):
        issues.append(
            _issue("missing_cross_reference", f"{entry_scope}/selection lacks source_inventory_sha256.", scope=entry_scope)
        )
    elif str(source_hash).casefold() != str(inventory_hash).casefold():
        issues.append(
            _issue(
                "cross_reference_hash_mismatch",
                f"{entry_scope}/selection.source_inventory_sha256 does not match inventory SHA-256.",
                scope=entry_scope,
            )
        )

    model = payload.get("model")
    expected_model = entry.get("model")
    if isinstance(model, Mapping) and isinstance(expected_model, Mapping):
        if model.get("model_id") != expected_model.get("model_id") or model.get("revision") != expected_model.get("revision"):
            issues.append(
                _issue("cross_reference_mismatch", f"{entry_scope}/selection model does not match the index.", scope=entry_scope)
            )
    if isinstance(payload.get("construct_ids"), list) and isinstance(entry.get("construct_ids"), list):
        if set(payload["construct_ids"]) != set(entry["construct_ids"]):
            issues.append(
                _issue("cross_reference_mismatch", f"{entry_scope}/selection construct_ids do not match the index.", scope=entry_scope)
            )
    if isinstance(gate, Mapping) and gate.get("status") == "pass":
        gate_payload = gate.get("payload")
        if payload.get("gate_config_sha256") is not None:
            if str(payload["gate_config_sha256"]).casefold() != str(gate.get("sha256")).casefold():
                issues.append(
                    _issue("cross_reference_hash_mismatch", f"{entry_scope}/selection gate hash does not match the indexed gate.", scope=entry_scope)
                )
        if payload.get("gate_id") is not None and isinstance(gate_payload, Mapping):
            if payload.get("gate_id") != gate_payload.get("gate_id"):
                issues.append(
                    _issue("cross_reference_mismatch", f"{entry_scope}/selection gate_id does not match the indexed gate.", scope=entry_scope)
                )


def _validate_gate_cross_references(
    entry: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    entry_scope: str,
    issues: list[dict[str, str]],
) -> None:
    if gate.get("status") != "pass":
        return
    payload = gate.get("payload")
    raw_gate = entry.get("artifacts", {}).get("gate", {})
    if not isinstance(payload, Mapping):
        return
    expected_id = raw_gate.get("id") if isinstance(raw_gate, Mapping) else None
    if expected_id is not None and payload.get("gate_id") != expected_id:
        issues.append(_issue("cross_reference_mismatch", f"{entry_scope}/gate_id does not match the index.", scope=entry_scope))
    expected_model = entry.get("model")
    models = payload.get("models")
    if isinstance(models, list) and isinstance(expected_model, Mapping):
        matching = [
            model
            for model in models
            if isinstance(model, Mapping) and model.get("model_id") == expected_model.get("model_id")
        ]
        if not matching:
            issues.append(_issue("cross_reference_mismatch", f"{entry_scope}/gate does not list the indexed model.", scope=entry_scope))
        elif any(model.get("revision") not in (None, expected_model.get("revision")) for model in matching):
            issues.append(_issue("cross_reference_mismatch", f"{entry_scope}/gate model revision does not match the index.", scope=entry_scope))
    if isinstance(payload.get("construct_ids"), list) and isinstance(entry.get("construct_ids"), list):
        if set(payload["construct_ids"]) != set(entry["construct_ids"]):
            issues.append(_issue("cross_reference_mismatch", f"{entry_scope}/gate construct_ids do not match the index.", scope=entry_scope))


def _validate_run_config_cross_references(
    entry: Mapping[str, Any],
    run_config: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    entry_scope: str,
    issues: list[dict[str, str]],
) -> None:
    if run_config.get("status") != "pass":
        return
    payload = run_config.get("payload")
    raw_run_config = entry.get("artifacts", {}).get("run_config", {})
    if not isinstance(payload, Mapping):
        return
    expected_id = raw_run_config.get("id") if isinstance(raw_run_config, Mapping) else None
    if expected_id is not None and payload.get("run_id") != expected_id:
        issues.append(_issue("cross_reference_mismatch", f"{entry_scope}/run_config run_id does not match the index.", scope=entry_scope))
    expected_model = entry.get("model")
    model = payload.get("model")
    if isinstance(model, Mapping) and isinstance(expected_model, Mapping):
        if model.get("model_id") != expected_model.get("model_id") or model.get("revision") != expected_model.get("revision"):
            issues.append(_issue("cross_reference_mismatch", f"{entry_scope}/run_config model does not match the index.", scope=entry_scope))
    if isinstance(payload.get("construct_ids"), list) and isinstance(entry.get("construct_ids"), list):
        if set(payload["construct_ids"]) != set(entry["construct_ids"]):
            issues.append(_issue("cross_reference_mismatch", f"{entry_scope}/run_config construct_ids do not match the index.", scope=entry_scope))
    preflight = payload.get("preflight")
    if isinstance(preflight, Mapping) and preflight.get("preflight_only") is not True:
        issues.append(_issue("not_preflight", f"{entry_scope}/run_config is not marked preflight_only.", scope=entry_scope))
    run_kind = payload.get("run_kind")
    if run_kind is not None and run_kind != "preflight_only":
        issues.append(_issue("not_preflight", f"{entry_scope}/run_config has run_kind={run_kind!r}.", scope=entry_scope))
    gate_payload = gate.get("payload") if isinstance(gate, Mapping) else None
    if isinstance(preflight, Mapping) and isinstance(gate_payload, Mapping):
        if preflight.get("gate_id") is not None and preflight.get("gate_id") != gate_payload.get("gate_id"):
            issues.append(_issue("cross_reference_mismatch", f"{entry_scope}/run_config gate_id does not match the indexed gate.", scope=entry_scope))


def _entry_report(
    entry: Mapping[str, Any],
    *,
    repo_root: Path,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    wave = entry.get("wave")
    model = entry.get("model")
    alias = model.get("alias") if isinstance(model, Mapping) else "?"
    entry_scope = f"wave{wave}/{alias}"
    local_start = len(issues)
    artifacts: dict[str, dict[str, Any]] = {}
    for name in _ARTIFACT_NAMES:
        artifacts[name] = _artifact(entry, name, repo_root=repo_root, entry_scope=entry_scope, issues=issues)
    issue_start = len(issues)
    _validate_inventory_manifest(entry, artifacts["inventory"], repo_root=repo_root, entry_scope=entry_scope, issues=issues)
    if len(issues) != issue_start and artifacts["inventory"]["status"] == "pass":
        artifacts["inventory"]["status"] = "fail"
    issue_start = len(issues)
    _validate_selection_cross_references(entry, artifacts, repo_root=repo_root, entry_scope=entry_scope, issues=issues)
    if len(issues) != issue_start and artifacts["selection"]["status"] == "pass":
        artifacts["selection"]["status"] = "fail"
    issue_start = len(issues)
    _validate_gate_cross_references(entry, artifacts["gate"], entry_scope=entry_scope, issues=issues)
    if len(issues) != issue_start and artifacts["gate"]["status"] == "pass":
        artifacts["gate"]["status"] = "fail"
    issue_start = len(issues)
    _validate_run_config_cross_references(
        entry,
        artifacts["run_config"],
        artifacts["gate"],
        entry_scope=entry_scope,
        issues=issues,
    )
    if len(issues) != issue_start and artifacts["run_config"]["status"] == "pass":
        artifacts["run_config"]["status"] = "fail"
    entry_issues = issues[local_start:]
    status = "ready" if not entry_issues else "blocked"
    return {
        "wave": wave,
        "model_alias": alias,
        "model_id": model.get("model_id") if isinstance(model, Mapping) else None,
        "artifacts": {name: result.get("status", "fail") for name, result in artifacts.items()},
        "status": status,
        "issues": entry_issues,
    }


def validate_index(index_path: str | Path = DEFAULT_INDEX, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Validate an index and return a JSON-serializable readiness report."""

    index_path = Path(index_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else _ROOT
    issues: list[dict[str, str]] = []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "schema_version": "0.1.0",
            "index_path": str(index_path),
            "repo_root": str(root),
            "ready": False,
            "entries": [],
            "issues": [_issue("invalid_index", f"Cannot read index: {exc}.")],
        }
    if not isinstance(payload, dict):
        issues.append(_issue("invalid_index", "The artifact index must contain a JSON object."))
        payload = {}
    if payload.get("index_type") != "model_preflight_artifact_index":
        issues.append(_issue("invalid_index", "index_type must be model_preflight_artifact_index."))
    if payload.get("execution_allowed") is not False:
        issues.append(_issue("invalid_index", "execution_allowed must be false; the index is not an execution configuration."))
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        issues.append(_issue("invalid_index", "The artifact index must contain a non-empty entries list."))
        entries = []

    reports: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            issue = _issue("invalid_entry", f"entries[{index}] must be an object.")
            issues.append(issue)
            reports.append({"wave": None, "model_alias": "?", "artifacts": {}, "status": "blocked", "issues": [issue]})
            continue
        model = entry.get("model")
        key = (entry.get("wave"), model.get("alias") if isinstance(model, Mapping) else None)
        if key in seen:
            issues.append(_issue("duplicate_entry", f"Duplicate wave/model entry: {key!r}."))
        seen.add(key)
        if entry.get("wave") not in {1, 2, 3, 4}:
            issues.append(_issue("invalid_entry", f"Entry wave must be one of 1-4, got {entry.get('wave')!r}."))
        if not isinstance(model, Mapping) or not all(
            isinstance(model.get(field), str) and model.get(field).strip()
            for field in ("alias", "model_id", "revision")
        ):
            issues.append(_issue("invalid_entry", f"Entry {key!r} has incomplete model identity."))
        if not isinstance(entry.get("construct_ids"), list) or not entry.get("construct_ids"):
            issues.append(_issue("invalid_entry", f"Entry {key!r} has no construct_ids."))
        reports.append(_entry_report(entry, repo_root=root, issues=issues))

    # A ready entry has no local issues; all pending/invalid rows remain
    # blocked.  Top-level issues are also execution blockers.
    ready = not issues and bool(reports) and all(report["status"] == "ready" for report in reports)
    return {
        "schema_version": payload.get("schema_version", "0.1.0"),
        "index_path": str(index_path),
        "repo_root": str(root),
        "index_id": payload.get("index_id"),
        "ready": ready,
        "entries": reports,
        "issues": issues,
        "summary": {
            "entry_count": len(reports),
            "ready_entries": sum(report["status"] == "ready" for report in reports),
            "blocked_entries": sum(report["status"] != "ready" for report in reports),
            "issue_count": len(issues),
        },
    }


def _short_status(value: str) -> str:
    return {"pass": "PASS", "ready": "PASS", "pending": "PENDING", "fail": "FAIL", "blocked": "BLOCKED"}.get(value, value.upper())


def print_matrix(report: Mapping[str, Any]) -> None:
    """Print the concise human-readable readiness matrix."""

    print("Wave  Model     Inventory  Selection  Gate  RunConfig  Status")
    print("----  --------  ---------  ---------  ----  ---------  -------")
    for entry in report.get("entries", []):
        artifacts = entry.get("artifacts", {})
        print(
            f"{str(entry.get('wave', '?')):>4}  "
            f"{str(entry.get('model_alias', '?')):<8}  "
            f"{_short_status(str(artifacts.get('inventory', 'FAIL'))):<9}  "
            f"{_short_status(str(artifacts.get('selection', 'FAIL'))):<9}  "
            f"{_short_status(str(artifacts.get('gate', 'FAIL'))):<4}  "
            f"{_short_status(str(artifacts.get('run_config', 'FAIL'))):<9}  "
            f"{entry.get('status', 'blocked').upper()}"
        )
    summary = report.get("summary", {})
    print(
        f"\nReadiness: {'READY' if report.get('ready') else 'BLOCKED'} "
        f"({summary.get('ready_entries', 0)}/{summary.get('entry_count', 0)} entries ready; "
        f"{summary.get('issue_count', len(report.get('issues', [])))} blockers)."
    )
    if report.get("issues"):
        print("\nBlockers:")
        for issue in report["issues"]:
            scope = f" [{issue['scope']}]" if issue.get("scope") else ""
            print(f"- {issue['code']}{scope}: {issue['message']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the canonical Wave 1-4 preflight artifact index.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--repo-root", type=Path, default=_ROOT)
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print the complete JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_index(args.index, repo_root=args.repo_root)
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_matrix(report)
    return 0 if report.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_INDEX", "file_sha256", "main", "print_matrix", "validate_index"]
