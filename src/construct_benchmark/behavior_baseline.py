"""Prompt-only behavior execution helpers and manifest validation.

The prompt-only baseline is deliberately separate from steering outputs.  It
answers whether the independent downstream task is usable at all before any
activation direction or control vector is applied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import validate_run_constructs
from .manifests import canonical_hash, file_sha256
from .prompts import PromptRecord, validate_prompt_records
from .run_modes import resolve_run_mode
from .schemas import ConstructSpec, RunConfig


BEHAVIOR_SPLITS = frozenset({"behavior_eval", "steering_eval", "calibration"})
BASELINE_MANIFEST_TYPE = "construct_behavior_output"


def _stable_rank(seed: int, *parts: str) -> str:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_behavior_records(
    records: Iterable[PromptRecord],
    *,
    run_config: RunConfig,
    construct_specs: Mapping[str, ConstructSpec],
    split: str = "behavior_eval",
    mode: str | None = None,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Select a deterministic independent-task subset for baseline execution."""

    if split not in BEHAVIOR_SPLITS:
        raise ValueError(f"split must be one of: {sorted(BEHAVIOR_SPLITS)}")
    validate_run_constructs(run_config, dict(construct_specs))
    materialized = list(records)
    if not materialized:
        raise ValueError("Prompt inventory must contain at least one record.")
    validate_prompt_records(materialized, construct_specs)
    mode_id, mode_config = resolve_run_mode(run_config, mode)
    candidates = [record for record in materialized if record.split == split]
    if not candidates:
        raise ValueError(f"Prompt inventory has no {split!r} records.")

    selection = dict(mode_config["prompt_selection"])
    if selection["strategy"] == "all":
        selected_ids = {record.prompt_id for record in candidates}
    else:
        limit = int(selection["max_items_per_single_split"])
        selected_ids = set()
        for construct_id in run_config.construct_ids:
            construct_candidates = [
                record for record in candidates if record.construct_id == construct_id
            ]
            ranked = sorted(
                construct_candidates,
                key=lambda record: _stable_rank(
                    run_config.seed,
                    construct_id,
                    split,
                    record.prompt_id,
                ),
            )
            selected_ids.update(record.prompt_id for record in ranked[:limit])

    selected = [record for record in candidates if record.prompt_id in selected_ids]
    if not selected:
        raise ValueError("Independent-task selection produced no records.")
    selected_specs = {
        construct_id: construct_specs[construct_id] for construct_id in run_config.construct_ids
    }
    validate_prompt_records(selected, selected_specs, require_all_splits=False)
    selection_manifest = {
        "schema_version": run_config.schema_version,
        "manifest_type": "behavior_prompt_selection",
        "run_id": run_config.run_id,
        "mode": mode_id,
        "purpose": mode_config["purpose"],
        "confirmatory": bool(mode_config["confirmatory"]),
        "split": split,
        "prompt_selection": selection,
        "source_prompt_count": len(materialized),
        "source_split_count": len(candidates),
        "selected_prompt_count": len(selected),
        "selected_counts_by_construct": {
            construct_id: sum(record.construct_id == construct_id for record in selected)
            for construct_id in run_config.construct_ids
        },
        "selected_prompt_ids": [record.prompt_id for record in selected],
    }
    selection_manifest["selection_sha256"] = canonical_hash(selection_manifest)
    return selected, selection_manifest


def output_manifest_path(raw_output: str | Path) -> Path:
    return Path(raw_output).with_suffix(Path(raw_output).suffix + ".manifest.json")


def build_behavior_output_manifest(
    *,
    run_config: RunConfig,
    construct_specs: Mapping[str, ConstructSpec],
    output: Path,
    prompt_inventory_sha256: str,
    selection_manifest: Mapping[str, Any],
    prompt_format: str,
    system_prompt_sha256: str,
    max_new_tokens: int,
    min_new_tokens: int,
    max_length: int,
    dtype: str,
    device: str,
    device_map: str | None,
    block_path: str | None,
) -> dict[str, Any]:
    """Build the manifest used by the prompt-only runner and scorer."""

    prompt_ids = [str(value) for value in selection_manifest["selected_prompt_ids"]]
    return {
        "schema_version": run_config.schema_version,
        "manifest_type": BASELINE_MANIFEST_TYPE,
        "run_id": run_config.run_id,
        "output": str(output),
        "construct_ids": list(run_config.construct_ids),
        "construct_spec_hashes": {
            construct_id: canonical_hash(construct_specs[construct_id].to_mapping())
            for construct_id in run_config.construct_ids
        },
        "split": str(selection_manifest["split"]),
        "intervention": "none",
        "prompt_inventory_sha256": prompt_inventory_sha256,
        "run_config_hash": canonical_hash(run_config.to_mapping()),
        "model": dict(run_config.model),
        "prompt_format": prompt_format,
        "system_prompt_sha256": system_prompt_sha256,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "max_length": max_length,
        "dtype": dtype,
        "device": device,
        "device_map": device_map,
        "block_path": block_path,
        "selection": dict(selection_manifest),
        "expected_prompt_ids": prompt_ids,
        "expected_record_ids": [f"{prompt_id}__prompt_only" for prompt_id in prompt_ids],
        "expected_record_count": len(prompt_ids),
        "completed_record_count": 0,
        "complete": False,
    }


def _raw_rows(raw_output: Path) -> list[dict[str, Any]]:
    if not raw_output.is_file():
        raise ValueError(f"Raw behavior output does not exist: {raw_output}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_output.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on behavior output line {line_number}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Behavior output line {line_number} is not a JSON object.")
        rows.append(row)
    return rows


def read_behavior_output(raw_output: str | Path) -> list[dict[str, Any]]:
    """Read JSONL output for callers that also need manifest validation."""

    return _raw_rows(Path(raw_output))


def validate_behavior_output_manifest(
    raw_output: str | Path,
    raw_rows: list[dict[str, Any]],
    *,
    run_config: RunConfig,
    construct_specs: Mapping[str, ConstructSpec],
    allow_incomplete_diagnostic: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Validate identity, provenance, row count, and completion status."""

    raw_path = Path(raw_output)
    manifest_path = output_manifest_path(raw_path)
    if not manifest_path.is_file():
        raise ValueError(f"Cannot score {raw_path} without {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path} is not valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object.")

    required = (
        "manifest_type",
        "run_id",
        "construct_ids",
        "construct_spec_hashes",
        "split",
        "intervention",
        "prompt_inventory_sha256",
        "run_config_hash",
        "model",
        "expected_prompt_ids",
        "expected_record_ids",
        "expected_record_count",
        "completed_record_count",
        "complete",
    )
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(f"{manifest_path} is missing required fields: {missing}.")
    if manifest["manifest_type"] != BASELINE_MANIFEST_TYPE:
        raise ValueError(f"{manifest_path} is not a prompt-only behavior output manifest.")
    if manifest["run_id"] != run_config.run_id:
        raise ValueError(f"{manifest_path} run_id differs from the requested run config.")
    if manifest["construct_ids"] != list(run_config.construct_ids):
        raise ValueError(f"{manifest_path} construct_ids differ from the requested run config.")
    expected_hashes = {
        construct_id: canonical_hash(construct_specs[construct_id].to_mapping())
        for construct_id in run_config.construct_ids
    }
    if manifest["construct_spec_hashes"] != expected_hashes:
        raise ValueError(f"{manifest_path} construct specification hashes differ.")
    if manifest["run_config_hash"] != canonical_hash(run_config.to_mapping()):
        raise ValueError(f"{manifest_path} run_config_hash differs from the requested run config.")
    if manifest["intervention"] != "none":
        raise ValueError(f"{manifest_path} is not marked as prompt-only intervention=none.")

    prompt_ids = manifest["expected_prompt_ids"]
    record_ids = manifest["expected_record_ids"]
    expected_count = manifest["expected_record_count"]
    if not isinstance(prompt_ids, list) or not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError(f"{manifest_path} expected_prompt_ids are invalid.")
    expected_record_ids = [f"{prompt_id}__prompt_only" for prompt_id in prompt_ids]
    if record_ids != expected_record_ids or expected_count != len(expected_record_ids):
        raise ValueError(f"{manifest_path} expected record identities are inconsistent.")
    complete = manifest["complete"]
    if not isinstance(complete, bool):
        raise ValueError(f"{manifest_path} complete must be a boolean.")
    if not complete and not allow_incomplete_diagnostic:
        raise ValueError(
            f"Refusing to score incomplete prompt-only output {raw_path}; "
            "pass --allow-incomplete-diagnostic for diagnostics only."
        )
    if complete:
        if manifest["completed_record_count"] != expected_count:
            raise ValueError(f"{manifest_path} completed_record_count is inconsistent.")
        if len(raw_rows) != expected_count:
            raise ValueError(
                f"Raw behavior output has {len(raw_rows)} rows but the completed manifest expects "
                f"{expected_count}."
            )
    elif len(raw_rows) > expected_count:
        raise ValueError(f"Raw behavior output has more rows than {manifest_path} permits.")

    raw_hash = manifest.get("raw_generations_sha256")
    if complete and not raw_hash:
        raise ValueError(f"{manifest_path} is complete but has no raw_generations_sha256.")
    if raw_hash and file_sha256(raw_path) != raw_hash:
        raise ValueError(f"{raw_path} does not match raw_generations_sha256.")

    expected_spec_hashes = manifest["construct_spec_hashes"]
    expected_model = manifest["model"]
    seen: set[str] = set()
    for row in raw_rows:
        record_id = row.get("record_id")
        prompt_id = row.get("prompt_id")
        if record_id not in expected_record_ids or prompt_id not in prompt_ids:
            raise ValueError(f"Raw behavior output contains an unexpected identity: {record_id!r}.")
        if record_id in seen:
            raise ValueError(f"Raw behavior output contains duplicate record_id={record_id!r}.")
        if record_id != f"{prompt_id}__prompt_only":
            raise ValueError(f"Raw behavior output record {record_id!r} has an invalid prompt-only identity.")
        seen.add(record_id)
        construct_id = row.get("construct_id")
        if construct_id not in expected_spec_hashes:
            raise ValueError(f"Raw behavior output record {record_id!r} has an unknown construct.")
        for field, expected in (
            ("split", manifest["split"]),
            ("intervention", "none"),
            ("prompt_inventory_sha256", manifest["prompt_inventory_sha256"]),
            ("run_config_hash", manifest["run_config_hash"]),
            ("construct_spec_hash", expected_spec_hashes[construct_id]),
        ):
            if row.get(field) != expected:
                raise ValueError(f"Raw behavior output record {record_id!r} has incompatible {field}.")
        if row.get("model") != expected_model:
            raise ValueError(f"Raw behavior output record {record_id!r} has different model metadata.")

    if complete and seen != set(expected_record_ids):
        missing_ids = sorted(set(expected_record_ids) - seen)
        raise ValueError(f"Raw behavior output is missing expected records: {missing_ids[:3]}.")
    return manifest, complete


def score_behavior_rows(
    raw_rows: Iterable[Mapping[str, Any]],
    construct_specs: Mapping[str, ConstructSpec],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse prompt-only outputs and summarize task usability per construct."""

    parsed_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = {construct_id: [] for construct_id in construct_specs}
    valid_parser: dict[str, int] = {construct_id: 0 for construct_id in construct_specs}
    valid_primary: dict[str, int] = {construct_id: 0 for construct_id in construct_specs}
    total: dict[str, int] = {construct_id: 0 for construct_id in construct_specs}
    from .behavior import orient_primary_outcome, parse_behavior_output, primary_outcome

    for raw in raw_rows:
        construct_id = str(raw.get("construct_id"))
        if construct_id not in construct_specs:
            raise ValueError(f"Raw behavior output has unknown construct_id={construct_id!r}.")
        spec = construct_specs[construct_id]
        total[construct_id] += 1
        metadata = dict(raw.get("task_metadata") or {})
        parser_id = str(raw.get("parser_id") or spec.parsing_rules["parser_id"])
        task_id = str(raw.get("task_id") or spec.independent_behavior_task["task_id"])
        parsed = parse_behavior_output(
            raw.get("output_text", ""),
            parser_id=parser_id,
            item_metadata=metadata,
            task_id=task_id,
        )
        outcome = None
        directed = None
        error = parsed.error
        if parsed.valid:
            valid_parser[construct_id] += 1
            try:
                outcome = primary_outcome(parsed, str(spec.independent_behavior_task["primary_outcome"]))
                directed = orient_primary_outcome(construct_id, outcome, metadata)
                if directed is not None:
                    valid_primary[construct_id] += 1
                    grouped[construct_id].append(float(directed))
                else:
                    error = "outcome orientation returned None"
            except (TypeError, ValueError) as exc:
                error = str(exc)
        parsed_rows.append(
            {
                "record_id": raw.get("record_id", ""),
                "prompt_id": raw.get("prompt_id", ""),
                "construct_id": construct_id,
                "parser_valid": parsed.valid,
                "primary_valid": directed is not None and error is None,
                "outcome": outcome,
                "directed_outcome": directed,
                "error": error or "",
                "task_metadata_json": json.dumps(metadata, sort_keys=True),
            }
        )

    constructs: dict[str, Any] = {}
    for construct_id in construct_specs:
        values = np.asarray(grouped[construct_id], dtype=np.float64)
        constructs[construct_id] = {
            "total_rows": total[construct_id],
            "valid_parser_rows": valid_parser[construct_id],
            "valid_primary_rows": valid_primary[construct_id],
            "invalid_rows": total[construct_id] - valid_parser[construct_id],
            "compliance_rate": (
                valid_parser[construct_id] / total[construct_id] if total[construct_id] else None
            ),
            "primary_valid_rate": (
                valid_primary[construct_id] / total[construct_id] if total[construct_id] else None
            ),
            "outcome_mean": float(values.mean()) if values.size else None,
            "outcome_sample_sd": float(values.std(ddof=1)) if values.size >= 2 else None,
            "outcome_min": float(values.min()) if values.size else None,
            "outcome_max": float(values.max()) if values.size else None,
            "unique_outcome_count": int(np.unique(values).size) if values.size else 0,
        }
    return parsed_rows, {"constructs": constructs, "total_rows": len(parsed_rows)}


__all__ = [
    "BASELINE_MANIFEST_TYPE",
    "BEHAVIOR_SPLITS",
    "build_behavior_output_manifest",
    "output_manifest_path",
    "read_behavior_output",
    "score_behavior_rows",
    "select_behavior_records",
    "validate_behavior_output_manifest",
]
