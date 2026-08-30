#!/usr/bin/env python3
"""Prepare a hash-bound, train-only B/R preflight manifest.

This command is deliberately a preparation boundary.  It validates the
registered Wave 1--4 input audit and freezes the identity of a future model
side run, but it does not load a model, create directions, or copy weights or
activation tensors.  ``full`` and confirmatory execution are refused here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "train_only_br_preflight_v1"
AUDIT_TYPE = "registered_waves1_4_model_input_audit"
MODEL_ALIASES = ("qwen", "mistral")
WAVES = (1, 2, 3, 4)
CONSTRUCT_COUNT = 4
ROWS_PER_CONSTRUCT_WAVE = 200
PAIRS_PER_CONSTRUCT_WAVE = 100
ROWS_PER_WAVE = ROWS_PER_CONSTRUCT_WAVE * CONSTRUCT_COUNT
PAIRS_PER_WAVE = PAIRS_PER_CONSTRUCT_WAVE * CONSTRUCT_COUNT
ROWS_PER_MODEL = ROWS_PER_WAVE * len(WAVES)
PAIRS_PER_MODEL = PAIRS_PER_WAVE * len(WAVES)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class PreflightValidationError(ValueError):
    """Raised when the registered train-only preparation contract is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.strip()) is None:
        raise PreflightValidationError(f"{label} must be a 64-character SHA-256 digest")
    return value.strip().lower()


def _require_nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreflightValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightValidationError(f"audit is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreflightValidationError("audit must contain a JSON object")
    return payload


def load_audit(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load one audit and return it with the hash of its exact JSON bytes."""

    audit_path = Path(path).expanduser().resolve()
    if not audit_path.is_file() or audit_path.is_symlink():
        raise PreflightValidationError(f"audit must be a regular file: {audit_path}")
    return _load_json(audit_path), file_sha256(audit_path)


def _validate_entry_inventory(entry: Mapping[str, Any], *, label: str) -> tuple[str, tuple[str, ...]]:
    inventory = entry.get("inventory")
    if not isinstance(inventory, Mapping):
        raise PreflightValidationError(f"{label}.inventory is missing")
    if inventory.get("direction_train_record_count") != ROWS_PER_WAVE:
        raise PreflightValidationError(f"{label} must contain {ROWS_PER_WAVE} direction-train rows")
    if inventory.get("direction_train_pair_count") != PAIRS_PER_WAVE:
        raise PreflightValidationError(f"{label} must contain {PAIRS_PER_WAVE} direction-train pairs")
    _require_digest(inventory.get("sha256"), label=f"{label}.inventory.sha256")
    _require_digest(inventory.get("manifest_sha256"), label=f"{label}.inventory.manifest_sha256")
    _require_digest(
        inventory.get("direction_train_content_sha256"),
        label=f"{label}.inventory.direction_train_content_sha256",
    )
    selection_filter = inventory.get("selection_filter")
    expected_filter = {
        "downstream_task_id_required": "empty",
        "pair_conditions_from_construct_spec": True,
        "prompt_role": "probe",
        "split": "direction_train",
    }
    if selection_filter != expected_filter:
        raise PreflightValidationError(f"{label}.inventory.selection_filter is not the frozen train-only filter")

    pair_counts = inventory.get("direction_train_pair_counts_by_construct")
    row_counts = inventory.get("vector_split_counts_by_construct")
    if not isinstance(pair_counts, Mapping) or not isinstance(row_counts, Mapping):
        raise PreflightValidationError(f"{label}.inventory lacks per-construct train counts")
    construct_ids = tuple(sorted(str(key) for key in pair_counts))
    if len(construct_ids) != CONSTRUCT_COUNT or any(
        pair_counts.get(construct_id) != PAIRS_PER_CONSTRUCT_WAVE for construct_id in construct_ids
    ):
        raise PreflightValidationError(
            f"{label}.inventory must contain {PAIRS_PER_CONSTRUCT_WAVE} pairs for each of four constructs"
        )
    if set(row_counts) != set(construct_ids):
        raise PreflightValidationError(f"{label}.inventory construct row-count scope does not match pair scope")
    for construct_id in construct_ids:
        counts = row_counts.get(construct_id)
        if not isinstance(counts, Mapping) or counts.get("direction_train") != ROWS_PER_CONSTRUCT_WAVE:
            raise PreflightValidationError(
                f"{label}.inventory must contain {ROWS_PER_CONSTRUCT_WAVE} rows for {construct_id}"
            )
    raw_specs = entry.get("construct_specs")
    if not isinstance(raw_specs, list):
        raise PreflightValidationError(f"{label}.construct_specs must be a list")
    declared_constructs: list[str] = []
    for index, spec in enumerate(raw_specs):
        if not isinstance(spec, Mapping):
            raise PreflightValidationError(f"{label}.construct_specs[{index}] is malformed")
        declared_constructs.append(
            _require_nonempty(spec.get("construct_id"), label=f"{label}.construct_specs[{index}].construct_id")
        )
    return construct_ids, tuple(sorted(declared_constructs))


def validate_audit(audit: Mapping[str, Any], model_alias: str) -> dict[str, Any]:
    """Validate the registered four-wave audit for one model alias."""

    if model_alias not in MODEL_ALIASES:
        raise PreflightValidationError(f"model_alias must be one of {MODEL_ALIASES}")
    if audit.get("audit_type") != AUDIT_TYPE or audit.get("audit_revision") != 2:
        raise PreflightValidationError("audit type or revision is not the registered Wave 1--4 audit")
    canonical_index = audit.get("canonical_index")
    if not isinstance(canonical_index, Mapping):
        raise PreflightValidationError("audit.canonical_index is missing")
    if canonical_index.get("status") != "ready" or canonical_index.get("execution_allowed") is not False:
        raise PreflightValidationError("canonical input index must be ready with execution_allowed=false")
    if canonical_index.get("entry_count") != len(MODEL_ALIASES) * len(WAVES):
        raise PreflightValidationError("canonical input index must contain exactly eight entries")
    if canonical_index.get("artifact_ready_count") != len(MODEL_ALIASES) * len(WAVES) * 4:
        raise PreflightValidationError("canonical input index must contain 32 ready artifacts")
    canonical_index_hash = _require_digest(canonical_index.get("sha256"), label="audit.canonical_index.sha256")

    entries = audit.get("entries")
    if not isinstance(entries, list) or len(entries) != len(MODEL_ALIASES) * len(WAVES):
        raise PreflightValidationError("audit.entries must contain exactly eight model/wave entries")
    matching: list[Mapping[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    constructs: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise PreflightValidationError(f"audit.entries[{index}] is malformed")
        label = f"audit.entries[{index}]"
        alias = _require_nonempty(raw_entry.get("model_alias"), label=f"{label}.model_alias")
        if alias not in MODEL_ALIASES:
            raise PreflightValidationError(f"{label}.model_alias is not registered")
        try:
            wave = int(raw_entry.get("wave"))
        except (TypeError, ValueError) as exc:
            raise PreflightValidationError(f"{label}.wave is not an integer") from exc
        if wave not in WAVES or (alias, wave) in seen:
            raise PreflightValidationError(f"duplicate or out-of-range model/wave entry: {alias}/{wave}")
        seen.add((alias, wave))
        _require_nonempty(raw_entry.get("model_id"), label=f"{label}.model_id")
        _require_nonempty(raw_entry.get("revision"), label=f"{label}.revision")
        entry_constructs, declared_constructs = _validate_entry_inventory(raw_entry, label=label)
        if set(entry_constructs) != set(declared_constructs) or len(declared_constructs) != CONSTRUCT_COUNT:
            raise PreflightValidationError(f"{label}.construct_specs does not match inventory construct scope")
        if alias == model_alias:
            matching.append(raw_entry)
            constructs.update(entry_constructs)

    if seen != {(alias, wave) for alias in MODEL_ALIASES for wave in WAVES}:
        raise PreflightValidationError("audit does not cover each registered model alias and wave")
    if len(matching) != len(WAVES):
        raise PreflightValidationError(f"audit does not contain all four waves for {model_alias}")
    if len(constructs) != len(WAVES) * CONSTRUCT_COUNT:
        raise PreflightValidationError("audit must cover exactly 16 construct IDs across the four waves")

    aggregate = audit.get("aggregate_by_model")
    model_aggregate = aggregate.get(model_alias) if isinstance(aggregate, Mapping) else None
    if not isinstance(model_aggregate, Mapping):
        raise PreflightValidationError(f"audit.aggregate_by_model.{model_alias} is missing")
    if (
        model_aggregate.get("direction_train_rows") != ROWS_PER_MODEL
        or model_aggregate.get("direction_train_pairs") != PAIRS_PER_MODEL
        or sorted(model_aggregate.get("waves", [])) != list(WAVES)
    ):
        raise PreflightValidationError(
            f"audit aggregate for {model_alias} must contain {ROWS_PER_MODEL} rows and {PAIRS_PER_MODEL} pairs"
        )

    model_ids = {_require_nonempty(entry.get("model_id"), label=f"{model_alias}.model_id") for entry in matching}
    revisions = {_require_nonempty(entry.get("revision"), label=f"{model_alias}.revision") for entry in matching}
    if len(model_ids) != 1 or len(revisions) != 1:
        raise PreflightValidationError(f"{model_alias} model identity changes across waves")
    input_hashes = {
        f"wave{int(entry['wave'])}": {
            "inventory_sha256": entry["inventory"]["sha256"].lower(),
            "manifest_sha256": entry["inventory"]["manifest_sha256"].lower(),
            "direction_train_content_sha256": entry["inventory"]["direction_train_content_sha256"].lower(),
        }
        for entry in sorted(matching, key=lambda item: int(item["wave"]))
    }
    return {
        "model_alias": model_alias,
        "model_id": next(iter(model_ids)),
        "revision": next(iter(revisions)),
        "waves": list(WAVES),
        "construct_ids": sorted(constructs),
        "direction_train_rows": ROWS_PER_MODEL,
        "direction_train_pairs": PAIRS_PER_MODEL,
        "canonical_index_sha256": canonical_index_hash,
        "input_hashes": input_hashes,
    }


def select_train_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select and validate the frozen train-only probe rows from canonical records."""

    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PreflightValidationError(f"row {index} is not a mapping")
        task_id = row.get("task_id")
        if (
            row.get("split") == "direction_train"
            and row.get("prompt_role") == "probe"
            and (task_id is None or str(task_id).strip() == "")
        ):
            selected.append(dict(row))
    if not selected:
        raise PreflightValidationError("no direction_train/probe rows with empty task_id were selected")

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    prompt_ids: set[str] = set()
    for row in selected:
        try:
            wave = int(row.get("wave"))
        except (TypeError, ValueError) as exc:
            raise PreflightValidationError("selected row has a non-integer wave") from exc
        construct_id = _require_nonempty(row.get("construct_id"), label="selected row.construct_id")
        prompt_id = _require_nonempty(row.get("prompt_id"), label="selected row.prompt_id")
        pair_id = _require_nonempty(row.get("pair_id"), label="selected row.pair_id")
        if prompt_id in prompt_ids:
            raise PreflightValidationError(f"duplicate selected prompt_id: {prompt_id}")
        prompt_ids.add(prompt_id)
        row["wave"] = wave
        row["construct_id"] = construct_id
        row["pair_id"] = pair_id
        groups[(wave, construct_id)].append(row)

    if len(selected) != ROWS_PER_MODEL:
        raise PreflightValidationError(f"selected rows={len(selected)}; expected {ROWS_PER_MODEL}")
    for (wave, construct_id), group in sorted(groups.items()):
        if wave not in WAVES:
            raise PreflightValidationError(f"selected row has out-of-range wave {wave}")
        if len(group) != ROWS_PER_CONSTRUCT_WAVE:
            raise PreflightValidationError(
                f"wave {wave} construct {construct_id} has {len(group)} rows; expected {ROWS_PER_CONSTRUCT_WAVE}"
            )
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            pairs[str(row["pair_id"])].append(row)
        if len(pairs) != PAIRS_PER_CONSTRUCT_WAVE or any(len(pair) != 2 for pair in pairs.values()):
            raise PreflightValidationError(
                f"wave {wave} construct {construct_id} must contain {PAIRS_PER_CONSTRUCT_WAVE} two-row pairs"
            )

    expected_groups = len(WAVES) * CONSTRUCT_COUNT
    group_counts_by_wave: dict[int, int] = defaultdict(int)
    constructs_by_wave: dict[int, set[str]] = defaultdict(set)
    for wave, construct_id in groups:
        group_counts_by_wave[wave] += 1
        constructs_by_wave[wave].add(construct_id)
    if len(groups) != expected_groups or any(
        group_counts_by_wave.get(wave, 0) != CONSTRUCT_COUNT for wave in WAVES
    ) or len(set().union(*constructs_by_wave.values())) != len(WAVES) * CONSTRUCT_COUNT:
        raise PreflightValidationError("selected rows do not cover exactly four constructs in each wave")
    return sorted(
        selected,
        key=lambda row: (
            int(row["wave"]),
            str(row["construct_id"]),
            str(row["pair_id"]),
            str(row.get("pair_role", "")),
            str(row["prompt_id"]),
        ),
    )


def build_manifest(
    audit: Mapping[str, Any],
    *,
    model_alias: str,
    audit_sha256: str,
    repo_sha: str,
    run_mode: str = "test",
    audit_path: str | Path | None = None,
    selected_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a planned, identity-bound manifest without creating B/R artifacts."""

    if run_mode != "test":
        raise PreflightValidationError("train-only B/R preflight refuses full or confirmatory mode")
    summary = validate_audit(audit, model_alias)
    audit_hash = _require_digest(audit_sha256, label="audit_sha256")
    if not isinstance(repo_sha, str) or _GIT_SHA_RE.fullmatch(repo_sha.strip()) is None:
        raise PreflightValidationError("repo_sha must be a 40-character git SHA")
    selected_count = None
    if selected_rows is not None:
        selected_count = len(select_train_rows(selected_rows))
        if selected_count != ROWS_PER_MODEL:
            raise PreflightValidationError("selected row count does not match the registered model total")
    identity = {
        "model_alias": model_alias,
        "model_id": summary["model_id"],
        "revision": summary["revision"],
        "audit_sha256": audit_hash,
        "canonical_index_sha256": summary["canonical_index_sha256"],
        "repo_sha": repo_sha.strip().lower(),
        "run_mode": run_mode,
        "input_hashes": summary["input_hashes"],
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "train_only_br_preflight",
        "status": "PLANNED",
        "created_at": _now(),
        "updated_at": _now(),
        "run_mode": "test",
        "preflight_only": True,
        "confirmatory": False,
        "semantic_runner": "not_executed",
        "model": {
            "alias": summary["model_alias"],
            "model_id": summary["model_id"],
            "revision": summary["revision"],
        },
        "repository": {"sha": repo_sha.strip().lower()},
        "audit": {
            "path": str(Path(audit_path).expanduser().resolve()) if audit_path is not None else None,
            "sha256": audit_hash,
            "canonical_index_sha256": summary["canonical_index_sha256"],
        },
        "input_hashes": summary["input_hashes"],
        "counts": {
            "waves": len(WAVES),
            "constructs": len(summary["construct_ids"]),
            "direction_train_rows": ROWS_PER_MODEL,
            "direction_train_pairs": PAIRS_PER_MODEL,
            "selected_rows": selected_count if selected_count is not None else ROWS_PER_MODEL,
        },
        "construct_ids": summary["construct_ids"],
        "artifacts": [],
        "resume_identity": _sha256_bytes(_canonical(identity).encode("utf-8")),
        "resume_count": 0,
        "policy": {
            "model_weights_loaded": False,
            "raw_activations_copied": False,
            "directions_created": False,
            "synthetic_directions": False,
            "full_execution_allowed": False,
        },
    }
    return manifest


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor_open = False
            handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            pass
        else:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if descriptor_open:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def write_manifest(output_root: str | Path, manifest: Mapping[str, Any], *, resume: bool = False) -> dict[str, Any]:
    """Atomically write one planned manifest, allowing only identity-bound resume."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    candidate = dict(manifest)
    if path.exists():
        if not resume:
            raise PreflightValidationError(f"refusing to overwrite existing manifest: {path}")
        extra_paths = sorted(item.name for item in root.iterdir() if item.name != path.name)
        if extra_paths:
            raise PreflightValidationError(
                "refusing to resume a preflight root containing non-manifest artifacts: " + ", ".join(extra_paths)
            )
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreflightValidationError(f"existing manifest is unreadable: {exc}") from exc
        if not isinstance(existing, Mapping) or existing.get("resume_identity") != candidate.get("resume_identity"):
            raise PreflightValidationError("resume identity does not match the existing manifest")
        candidate["created_at"] = existing.get("created_at", candidate.get("created_at"))
        candidate["resume_count"] = int(existing.get("resume_count", 0)) + 1
        candidate["previous_manifest_sha256"] = file_sha256(path)
        candidate["updated_at"] = _now()
    elif any(root.iterdir()):
        raise PreflightValidationError(f"refusing to write into a non-empty output root without a manifest: {root}")
    _atomic_write_json(path, candidate)
    return candidate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--model-alias", choices=MODEL_ALIASES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-sha", required=True, help="Exact repository SHA bound to this preparation.")
    parser.add_argument("--run-mode", choices=("test", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        audit, audit_hash = load_audit(args.audit)
        manifest = build_manifest(
            audit,
            model_alias=args.model_alias,
            audit_sha256=audit_hash,
            repo_sha=args.repo_sha,
            run_mode=args.run_mode,
            audit_path=args.audit,
        )
        written = write_manifest(args.output_root, manifest, resume=args.resume)
    except PreflightValidationError as exc:
        print(f"run_train_only_br_preflight: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: written[key] for key in ("status", "run_mode", "resume_identity")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
