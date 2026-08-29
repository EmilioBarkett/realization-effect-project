"""Model-side behavioral and accessibility preflight gates.

The preflight is intentionally a small, outcome-independent release gate.  A
selection manifest is frozen from the prompt inventory before any model output
is inspected, and a later validator checks a manifest-backed behavior baseline,
collateral task, and steering output for one model at a time.

This is not a substitute for the full benchmark.  It answers the narrower
question that must be settled first: can this model produce usable answers on
8--16 registered items per construct, including under the registered steering
controls?
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .behavior import orient_primary_outcome, parse_behavior_output, primary_outcome
from .behavior_baseline import score_behavior_rows
from .manifests import canonical_hash, file_sha256
from .prompts import PromptRecord
from .schemas import ConstructSpec, SCHEMA_VERSION


PREFLIGHT_MANIFEST_TYPE = "model_behavior_accessibility_preflight_selection"
PREFLIGHT_REPORT_TYPE = "model_behavior_accessibility_preflight_report"
PREFLIGHT_ID = "model_behavior_accessibility_v2"
PREFLIGHT_SPLITS = ("behavior_eval", "steering_eval", "collateral_eval")
DEFAULT_MINIMUM_ITEMS = 8
DEFAULT_TARGET_ITEMS = 16
DEFAULT_MAXIMUM_ITEMS = 16
DEFAULT_STEERING_DOSES = {
    "target": (-1.0, 0.0, 1.0),
    "shuffled": (0.0,),
    "random": (0.0,),
}
_DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_relative_path(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> str:
    """Serialize repository-contained paths without machine-specific prefixes.

    Paths outside the repository keep their caller-provided spelling so that
    temporary or externally staged inventories remain usable in tests.  A
    path that resolves under ``repository_root`` is always emitted with POSIX
    separators relative to that root, which makes frozen manifests portable
    and safe for repository-relative artifact validation.
    """

    raw_path = Path(path)
    root = Path(repository_root or _DEFAULT_REPOSITORY_ROOT).resolve()
    resolved = raw_path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return raw_path.as_posix() if not raw_path.is_absolute() else str(raw_path)


def load_preflight_gate_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the versioned model-side preflight contract."""

    config = _read_json_object(path)
    gate_id = config.get("gate_id")
    if not isinstance(gate_id, str) or not gate_id.strip():
        raise ValueError("Preflight gate config requires a non-empty gate_id.")
    if config.get("confirmatory") is not False:
        raise ValueError("The model-side preflight gate must be non-confirmatory.")
    if config.get("selection_informed_by_outcomes") is not False:
        raise ValueError("The model-side preflight gate must be outcome-independent.")
    construct_ids = config.get("construct_ids")
    if (
        not isinstance(construct_ids, list)
        or not construct_ids
        or any(not isinstance(value, str) or not value.strip() for value in construct_ids)
        or len(set(construct_ids)) != len(construct_ids)
    ):
        raise ValueError("Preflight gate config construct_ids must be a unique non-empty list.")
    bounds = config.get("item_bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("Preflight gate config requires item_bounds.")
    _validate_bounds(int(bounds["minimum"]), int(bounds["target"]), int(bounds["maximum"]))
    thresholds = config.get("release_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("Preflight gate config requires release_thresholds.")
    required_thresholds = (
        "behavior_minimum_valid_rate",
        "behavior_maximum_invalid_items",
        "behavior_minimum_distinct_outcomes",
        "behavior_minimum_sample_sd",
        "collateral_minimum_valid_rate",
        "collateral_minimum_correctness_rate",
        "steering_minimum_valid_rate",
    )
    missing = [key for key in required_thresholds if key not in thresholds]
    if missing:
        raise ValueError(f"Preflight gate config is missing thresholds: {missing}.")
    steering_kinds = thresholds.get("steering_required_direction_kinds")
    steering_doses = thresholds.get("steering_required_target_doses")
    if not isinstance(steering_kinds, list) or not steering_kinds:
        raise ValueError("release_thresholds.steering_required_direction_kinds must be non-empty.")
    if not isinstance(steering_doses, list) or not steering_doses:
        raise ValueError("release_thresholds.steering_required_target_doses must be non-empty.")
    if "steering_intervention_timing" not in thresholds:
        raise ValueError("release_thresholds.steering_intervention_timing is required.")
    config["construct_ids"] = [str(value) for value in construct_ids]
    config["item_bounds"] = {
        "minimum": int(bounds["minimum"]),
        "target": int(bounds["target"]),
        "maximum": int(bounds["maximum"]),
    }
    config["release_thresholds"] = dict(thresholds)
    selection = config.get("selection")
    if selection is not None:
        if not isinstance(selection, Mapping):
            raise ValueError("Preflight selection must be an object when provided.")
        selection = dict(selection)
        if selection.get("selection_informed_by_outcomes") is not False:
            raise ValueError("Preflight selection must explicitly be outcome-independent.")
        if selection.get("position_balance_required") is True:
            raw_splits = selection.get(
                "position_balanced_splits",
                ["behavior_eval", "steering_eval"],
            )
            if (
                not isinstance(raw_splits, list)
                or not raw_splits
                or any(str(split) not in PREFLIGHT_SPLITS for split in raw_splits)
                or len(set(str(split) for split in raw_splits)) != len(raw_splits)
            ):
                raise ValueError(
                    "Position-balanced preflight selection must name unique registered splits."
                )
            levels = selection.get("position_levels", [1, 2])
            if (
                not isinstance(levels, list)
                or len(levels) < 2
                or len(set(levels)) != len(levels)
            ):
                raise ValueError("Position-balanced preflight selection requires distinct position_levels.")
            fields_by_split = selection.get("position_fields_by_split")
            fields = selection.get("position_fields")
            if not isinstance(fields_by_split, Mapping) and not isinstance(fields, Mapping):
                raise ValueError(
                    "Position-balanced preflight selection requires position field mappings."
                )
            for construct_id in construct_ids:
                for split in raw_splits:
                    field = None
                    field_registered = False
                    if isinstance(fields_by_split, Mapping):
                        construct_fields = fields_by_split.get(construct_id)
                        if isinstance(construct_fields, Mapping):
                            if split in construct_fields:
                                field_registered = True
                                field = construct_fields.get(split)
                    if field is None and isinstance(fields, Mapping):
                        construct_field = fields.get(construct_id)
                        if isinstance(construct_field, Mapping):
                            if split in construct_field:
                                field_registered = True
                                field = construct_field.get(split)
                        elif split in {"behavior_eval", "steering_eval"}:
                            field_registered = construct_id in fields
                            field = construct_field
                    if field_registered and field is None:
                        continue
                    if not isinstance(field, str) or not field.strip():
                        raise ValueError(
                            f"Position-balanced preflight selection has no field for {construct_id} {split}."
                        )
        config["selection"] = selection
    return config


def _stable_rank(seed: int, *parts: str) -> str:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_model(value: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(value)
    model_id = model.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("The preflight model must include a non-empty model_id.")
    if "revision" not in model or not isinstance(model.get("revision"), str) or not str(model.get("revision")).strip():
        raise ValueError("The preflight model must include a non-empty frozen revision.")
    if "fake" in model_id.lower() or model_id.lower().startswith(("test/", "dummy/")):
        raise ValueError("The model-side preflight requires a real model identifier, not a fixture model.")
    return model


def _validate_bounds(minimum_items: int, target_items: int, maximum_items: int) -> None:
    values = (minimum_items, target_items, maximum_items)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("Preflight item bounds must be integers.")
    if not 1 <= minimum_items <= target_items <= maximum_items:
        raise ValueError("Require 1 <= minimum_items <= target_items <= maximum_items.")


def _position_balance_spec(
    selection: Mapping[str, Any],
    *,
    construct_id: str,
    split: str,
) -> tuple[str, tuple[Any, ...]] | None:
    """Resolve the registered position field and levels for one split.

    The ordinary preflight gate predates position-aware repaired-v3 inputs, so
    this is opt-in through the gate's selection contract.  Repaired-v3 gates
    provide a per-split mapping because collateral answers use
    ``correct_option`` while behavior/steering use the construct's
    ``target_option`` or ``primary_position`` field.
    """

    if selection.get("position_balance_required") is not True:
        return None
    raw_splits = selection.get("position_balanced_splits", ("behavior_eval", "steering_eval"))
    if not isinstance(raw_splits, list) or split not in {str(value) for value in raw_splits}:
        return None

    field: Any = None
    field_registered = False
    fields_by_split = selection.get("position_fields_by_split")
    if isinstance(fields_by_split, Mapping):
        construct_fields = fields_by_split.get(construct_id)
        if isinstance(construct_fields, Mapping):
            if split in construct_fields:
                field_registered = True
                field = construct_fields.get(split)
    if field is None:
        fields = selection.get("position_fields")
        if isinstance(fields, Mapping):
            construct_field = fields.get(construct_id)
            if isinstance(construct_field, Mapping):
                if split in construct_field:
                    field_registered = True
                    field = construct_field.get(split)
            elif construct_field is not None:
                field_registered = True
                field = construct_field
            elif construct_id in fields and split in {"behavior_eval", "steering_eval"}:
                field_registered = True
    if field_registered and field is None:
        # Probability forecasts have no positional option to balance.  An
        # explicit null in the gate mapping keeps those splits on the normal
        # deterministic selector without allowing a missing mapping to
        # silently bypass a required position contract.
        return None
    if not isinstance(field, str) or not field.strip():
        raise ValueError(
            f"Position-balanced preflight selection has no field for {construct_id} {split}."
        )

    raw_levels = selection.get("position_levels", [1, 2])
    if (
        not isinstance(raw_levels, list)
        or len(raw_levels) < 2
        or len(set(raw_levels)) != len(raw_levels)
    ):
        raise ValueError("Position-balanced preflight selection requires distinct position_levels.")
    return str(field), tuple(raw_levels)


def _record_task_metadata(record: PromptRecord) -> Mapping[str, Any]:
    raw = record.metadata.get("task_metadata")
    return raw if isinstance(raw, Mapping) else record.metadata


def _select_position_balanced_candidates(
    candidates: list[PromptRecord],
    *,
    seed: int,
    frozen_model: Mapping[str, Any],
    construct_id: str,
    split: str,
    target_count: int,
    minimum_items: int,
    position_spec: tuple[str, tuple[Any, ...]],
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Select equal-sized stable-hash strata for a registered position field."""

    field, levels = position_spec
    by_level: dict[Any, list[PromptRecord]] = {level: [] for level in levels}
    for record in candidates:
        value = _record_task_metadata(record).get(field)
        if isinstance(value, bool) or value not in by_level:
            raise ValueError(
                f"{construct_id} {split} prompt {record.prompt_id} has {field}={value!r}; "
                f"expected one of {list(levels)!r}."
            )
        by_level[value].append(record)

    level_count = len(levels)
    if target_count % level_count:
        raise ValueError(
            f"Position-balanced preflight target for {construct_id} {split} is not divisible "
            f"by {level_count}: {target_count}."
        )
    per_level = min(target_count // level_count, *(len(by_level[level]) for level in levels))
    selected_count = per_level * level_count
    if selected_count < minimum_items:
        counts = {str(level): len(by_level[level]) for level in levels}
        raise ValueError(
            f"{construct_id} {split} has insufficient balanced position coverage for the "
            f"preflight minimum: field={field!r}, candidates_by_position={counts}."
        )

    selected: list[PromptRecord] = []
    for level in levels:
        ranked = sorted(
            by_level[level],
            key=lambda record: _stable_rank(
                seed,
                str(frozen_model["model_id"]),
                str(frozen_model["revision"]),
                construct_id,
                split,
                field,
                str(level),
                record.prompt_id,
            ),
        )
        selected.extend(ranked[:per_level])
    selected.sort(key=lambda record: record.prompt_id)
    return selected, {
        "field": field,
        "levels": list(levels),
        "counts": {str(level): per_level for level in levels},
        "selected_count": selected_count,
    }


def prepare_selection_manifest(
    records: Iterable[PromptRecord],
    *,
    source_inventory: str | Path,
    model: Mapping[str, Any],
    construct_ids: Iterable[str],
    seed: int = 1729,
    minimum_items: int = DEFAULT_MINIMUM_ITEMS,
    target_items: int = DEFAULT_TARGET_ITEMS,
    maximum_items: int = DEFAULT_MAXIMUM_ITEMS,
    gate_config: Mapping[str, Any] | None = None,
    gate_config_sha256: str | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze an outcome-independent preflight subset for one model."""

    if gate_config is not None:
        raw_bounds = gate_config.get("item_bounds")
        if not isinstance(raw_bounds, Mapping):
            raise ValueError("gate_config.item_bounds must be an object.")
        configured_bounds = (
            int(raw_bounds["minimum"]),
            int(raw_bounds["target"]),
            int(raw_bounds["maximum"]),
        )
        if (minimum_items, target_items, maximum_items) != configured_bounds:
            raise ValueError(
                "Explicit preflight item bounds do not match the frozen gate config."
            )
        gate_id = str(gate_config["gate_id"])
        configured_thresholds = dict(gate_config.get("release_thresholds", {}))
    else:
        gate_id = PREFLIGHT_ID
        configured_thresholds = {}
    _validate_bounds(minimum_items, target_items, maximum_items)
    frozen_model = _as_model(model)
    construct_ids = [str(value) for value in construct_ids]
    if not construct_ids or len(set(construct_ids)) != len(construct_ids):
        raise ValueError("construct_ids must be a non-empty list without duplicates.")
    materialized = list(records)
    if not materialized:
        raise ValueError("The prompt inventory is empty.")

    selection_contract = (
        dict(gate_config.get("selection", {}))
        if isinstance(gate_config, Mapping) and isinstance(gate_config.get("selection"), Mapping)
        else {}
    )
    selected: dict[str, dict[str, Any]] = {}
    for construct_id in construct_ids:
        selected[construct_id] = {}
        for split in PREFLIGHT_SPLITS:
            candidates = [
                record
                for record in materialized
                if record.construct_id == construct_id and record.split == split
            ]
            if len(candidates) < minimum_items:
                raise ValueError(
                    f"{construct_id} has only {len(candidates)} {split} items; "
                    f"the preflight requires at least {minimum_items}."
                )
            count = min(target_items, maximum_items, len(candidates))
            position_spec = _position_balance_spec(
                selection_contract,
                construct_id=construct_id,
                split=split,
            )
            if position_spec is None:
                ranked = sorted(
                    candidates,
                    key=lambda record: _stable_rank(
                        seed,
                        frozen_model["model_id"],
                        str(frozen_model["revision"]),
                        construct_id,
                        split,
                        record.prompt_id,
                    ),
                )[:count]
                ranked = sorted(ranked, key=lambda record: record.prompt_id)
                position_balance = None
            else:
                ranked, position_balance = _select_position_balanced_candidates(
                    candidates,
                    seed=seed,
                    frozen_model=frozen_model,
                    construct_id=construct_id,
                    split=split,
                    target_count=count,
                    minimum_items=minimum_items,
                    position_spec=position_spec,
                )
            prompt_ids = [record.prompt_id for record in ranked]
            split_selection: dict[str, Any] = {
                "prompt_ids": prompt_ids,
                "item_count": len(prompt_ids),
                "source_item_count": len(candidates),
                "minimum_items": minimum_items,
                "target_items": target_items,
                "maximum_items": maximum_items,
            }
            if position_balance is not None:
                split_selection["position_balance"] = position_balance
            selected[construct_id][split] = split_selection

    position_balanced_splits = selection_contract.get("position_balanced_splits", [])
    if not isinstance(position_balanced_splits, list):
        position_balanced_splits = []
    selection_summary = {
        "strategy": str(selection_contract.get("strategy", "stable_hash")),
        "position_balance_required": selection_contract.get("position_balance_required") is True,
        "position_balanced_splits": [str(value) for value in position_balanced_splits],
    }
    if selection_summary["position_balance_required"]:
        selection_summary["position_levels"] = list(selection_contract.get("position_levels", [1, 2]))

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": PREFLIGHT_MANIFEST_TYPE,
        "preflight_id": PREFLIGHT_ID,
        "gate_id": gate_id,
        "gate_config_sha256": gate_config_sha256,
        "purpose": "release_gate_before_any_further_large_model_execution",
        "confirmatory": False,
        "selection_informed_by_outcomes": False,
        "source_inventory": repository_relative_path(
            source_inventory,
            repository_root=repository_root,
        ),
        "source_inventory_sha256": file_sha256(source_inventory),
        "model": frozen_model,
        "construct_ids": construct_ids,
        "seed": int(seed),
        "item_bounds": {
            "minimum": minimum_items,
            "target": target_items,
            "maximum": maximum_items,
        },
        "selection_contract": selection_summary,
        "selected": selected,
        "steering_requirements": {
            "required_direction_kinds": list(
                configured_thresholds.get(
                    "steering_required_direction_kinds", DEFAULT_STEERING_DOSES
                )
            ),
            "required_doses_by_direction_kind": {
                kind: list(doses)
                for kind, doses in (
                    {
                        **DEFAULT_STEERING_DOSES,
                        "target": tuple(
                            configured_thresholds.get(
                                "steering_required_target_doses",
                                DEFAULT_STEERING_DOSES["target"],
                            )
                        ),
                    }
                ).items()
                if kind
                in set(
                    configured_thresholds.get(
                        "steering_required_direction_kinds", DEFAULT_STEERING_DOSES
                    )
                )
            },
        },
        "selection_rule": (
            "For each model, construct, split, and registered position stratum, rank prompt IDs "
            "by a frozen SHA-256 schedule, retain equal counts from every stratum, and inspect no "
            "outcomes during selection."
            if selection_summary["position_balance_required"]
            else "For each model, construct, and split, rank registered prompt IDs by a frozen "
            "SHA-256 schedule and retain the first target_items; inspect no outcomes during selection."
        ),
    }
    manifest["selection_sha256"] = canonical_hash(manifest)
    return manifest


def _read_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing model output: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_number} is not valid JSON.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_number} must be a JSON object.")
        rows.append(row)
    return rows


def _output_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _validate_model_output_manifest(
    path: str | Path,
    *,
    expected_manifest_type: str,
    expected_model: Mapping[str, Any],
    expected_inventory_sha256: str | None = None,
    required_prompt_format: str | None = None,
    require_constrained_numeric_generation: bool = False,
    require_manifest_record_count: bool = False,
    require_thinking_disabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = Path(path)
    rows = _read_jsonl(output)
    manifest_path = _output_manifest_path(output)
    manifest = _read_json_object(manifest_path)
    if manifest.get("manifest_type") != expected_manifest_type:
        raise ValueError(
            f"{manifest_path} has manifest_type={manifest.get('manifest_type')!r}; "
            f"expected {expected_manifest_type!r}."
        )
    if manifest.get("complete") is not True:
        raise ValueError(f"{output} is not backed by a complete output manifest.")
    if manifest.get("model") != dict(expected_model):
        raise ValueError(f"{output} model metadata differs from the frozen preflight model.")
    if expected_inventory_sha256 is not None and manifest.get("prompt_inventory_sha256") != expected_inventory_sha256:
        raise ValueError(f"{output} does not use the frozen preflight prompt inventory.")
    if required_prompt_format is not None and manifest.get("prompt_format") != required_prompt_format:
        raise ValueError(
            f"{output} prompt_format={manifest.get('prompt_format')!r}; "
            f"expected {required_prompt_format!r}."
        )
    if require_constrained_numeric_generation and manifest.get("constrained_numeric_generation") is not True:
        raise ValueError(f"{output} was not generated with the shared constrained numeric response channel.")
    if require_thinking_disabled and manifest.get("enable_thinking") is not False:
        raise ValueError(
            f"{output} does not declare explicit no-thinking generation under the v2 parser contract."
        )
    expected_count = manifest.get("expected_record_count")
    completed_count = manifest.get("completed_record_count")
    if require_manifest_record_count:
        if not isinstance(expected_count, int) or expected_count < 1:
            raise ValueError(f"{manifest_path} is missing a positive expected_record_count.")
        if completed_count != expected_count or len(rows) != expected_count:
            raise ValueError(
                f"{output} has {len(rows)} rows, but its complete manifest expects "
                f"{expected_count} completed rows."
            )
    elif expected_count is not None and (
        not isinstance(expected_count, int)
        or expected_count < 1
        or completed_count != expected_count
        or len(rows) != expected_count
    ):
        raise ValueError(f"{output} has an inconsistent completed record count.")
    expected_record_ids = manifest.get("expected_record_ids")
    if require_manifest_record_count:
        if (
            not isinstance(expected_record_ids, list)
            or len(expected_record_ids) != len(rows)
            or len(set(expected_record_ids)) != len(expected_record_ids)
            or {str(row.get("record_id")) for row in rows} != {str(value) for value in expected_record_ids}
        ):
            raise ValueError(f"{output} record identities do not match its complete manifest.")
    raw_hash = manifest.get("raw_generations_sha256", manifest.get("raw_output_sha256"))
    if not raw_hash:
        raise ValueError(f"{manifest_path} is complete but has no raw-output checksum.")
    if file_sha256(output) != raw_hash:
        raise ValueError(f"{output} does not match its manifest checksum.")
    return rows, manifest


def _selected_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    construct_id: str,
    split: str,
    prompt_ids: Iterable[str],
    expected_model: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    wanted = {str(value) for value in prompt_ids}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        if row.get("construct_id") != construct_id or row.get("split") != split:
            continue
        prompt_id = row.get("prompt_id")
        if prompt_id in wanted:
            if row.get("model") != dict(expected_model):
                raise ValueError(f"{construct_id} {split} row has different model metadata.")
            grouped[str(prompt_id)].append(row)
    missing = sorted(wanted - set(grouped))
    duplicate = sorted(prompt_id for prompt_id, values in grouped.items() if len(values) != 1)
    if duplicate:
        raise ValueError(
            f"{construct_id} {split} has duplicate or repeated preflight prompt IDs: {duplicate[:3]}"
        )
    selected = [grouped[prompt_id][0] for prompt_id in sorted(wanted) if prompt_id in grouped]
    return selected, missing


def _behavior_stats(
    rows: list[dict[str, Any]],
    spec: ConstructSpec,
    *,
    collateral: bool = False,
) -> dict[str, Any]:
    parsed_rows, scored = score_behavior_rows(rows, {spec.construct_id: spec})
    entry = dict(scored["constructs"][spec.construct_id])
    entry["selected_item_count"] = len(rows)
    entry["valid_primary_rate"] = (
        entry["valid_primary_rows"] / len(rows) if rows else 0.0
    )
    entry["invalid_or_unscorable_items"] = len(rows) - entry["valid_primary_rows"]
    entry["sample_sd"] = entry.get("outcome_sample_sd")
    entry["distinct_outcomes"] = entry.get("unique_outcome_count", 0)
    entry["mean_correctness"] = (
        entry.get("outcome_mean") if collateral else None
    )
    valid_outcomes = [
        float(row["outcome"])
        for row in parsed_rows
        if row.get("primary_valid") and row.get("outcome") is not None
    ]
    outcome_frequency: dict[str, int] = {}
    for value in valid_outcomes:
        key = str(int(value)) if value.is_integer() else str(value)
        outcome_frequency[key] = outcome_frequency.get(key, 0) + 1
    entry["outcome_frequency"] = dict(sorted(outcome_frequency.items()))
    if valid_outcomes:
        floor = min(valid_outcomes)
        ceiling = max(valid_outcomes)
        entry["floor_outcome"] = floor
        entry["ceiling_outcome"] = ceiling
        entry["floor_share"] = valid_outcomes.count(floor) / len(valid_outcomes)
        entry["ceiling_share"] = valid_outcomes.count(ceiling) / len(valid_outcomes)
    else:
        entry["floor_outcome"] = None
        entry["ceiling_outcome"] = None
        entry["floor_share"] = None
        entry["ceiling_share"] = None
    return entry


def _gate_behavior(
    stats: Mapping[str, Any],
    *,
    minimum_items: int,
    maximum_items: int,
    minimum_valid_rate: float,
    maximum_invalid_items: int,
    minimum_distinct_outcomes: int,
    minimum_sample_sd: float,
    maximum_ceiling_share: float | None = None,
    maximum_floor_share: float | None = None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    count = int(stats["selected_item_count"])
    if not minimum_items <= count <= maximum_items:
        failures.append(f"item_count={count} outside [{minimum_items}, {maximum_items}]")
    if float(stats["valid_primary_rate"]) < minimum_valid_rate:
        failures.append(
            f"valid_primary_rate={stats['valid_primary_rate']:.4f} < {minimum_valid_rate:.4f}"
        )
    if int(stats["invalid_or_unscorable_items"]) > maximum_invalid_items:
        failures.append(
            f"invalid_or_unscorable_items={stats['invalid_or_unscorable_items']} > {maximum_invalid_items}"
        )
    if int(stats["distinct_outcomes"]) < minimum_distinct_outcomes:
        failures.append(
            f"distinct_outcomes={stats['distinct_outcomes']} < {minimum_distinct_outcomes}"
        )
    sample_sd = stats.get("sample_sd")
    if sample_sd is None or not math.isfinite(float(sample_sd)) or float(sample_sd) < minimum_sample_sd:
        failures.append(f"sample_sd={sample_sd!r} < {minimum_sample_sd:.4f}")
    ceiling_share = stats.get("ceiling_share")
    if (
        maximum_ceiling_share is not None
        and ceiling_share is not None
        and float(ceiling_share) > maximum_ceiling_share
    ):
        failures.append(
            f"ceiling_share={float(ceiling_share):.4f} > {maximum_ceiling_share:.4f}"
        )
    floor_share = stats.get("floor_share")
    if (
        maximum_floor_share is not None
        and floor_share is not None
        and float(floor_share) > maximum_floor_share
    ):
        failures.append(
            f"floor_share={float(floor_share):.4f} > {maximum_floor_share:.4f}"
        )
    return not failures, failures


def _steering_preflight(
    rows: list[dict[str, Any]],
    spec: ConstructSpec,
    *,
    selected_prompt_ids: list[str],
    expected_model: Mapping[str, Any],
    minimum_valid_rate: float,
    required_doses_by_direction_kind: Mapping[str, Iterable[float]],
    require_correct_injection_sign: bool = False,
    minimum_mean_abs_injection_shift: float = 0.0,
    minimum_target_dose_response_span: float = 0.0,
    expected_intervention_timing: str = "prefill_only",
) -> dict[str, Any]:
    selected = set(selected_prompt_ids)
    relevant = [
        row
        for row in rows
        if row.get("construct_id") == spec.construct_id and row.get("prompt_id") in selected
    ]
    observed_prompts = {str(row.get("prompt_id")) for row in relevant}
    missing_prompts = sorted(selected - observed_prompts)
    required_pairs = {
        (kind, float(dose))
        for kind, doses in required_doses_by_direction_kind.items()
        for dose in doses
    }
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    invalid_examples: list[dict[str, Any]] = []
    injection_rows = 0
    timing_values: set[str] = set()
    for row in relevant:
        if row.get("model") != dict(expected_model):
            raise ValueError(f"{spec.construct_id} steering row has different model metadata.")
        kind = str(row.get("direction_kind"))
        try:
            dose = float(row["dose"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{spec.construct_id} steering row has an invalid dose.") from exc
        grouped[(kind, dose)].append(row)
        if row.get("injection_applied") is True:
            injection_rows += 1
        if row.get("intervention_timing") is not None:
            timing_values.add(str(row["intervention_timing"]))

    group_reports: dict[str, Any] = {}
    group_failures: list[str] = []
    for kind, dose in sorted(required_pairs):
        group_rows = grouped.get((kind, dose), [])
        valid = 0
        for row in group_rows:
            parsed = parse_behavior_output(
                row.get("output_text", ""),
                parser_id=str(row.get("parser_id") or spec.parsing_rules["parser_id"]),
                item_metadata=dict(row.get("task_metadata") or {}),
                task_id=str(row.get("task_id") or spec.independent_behavior_task["task_id"]),
            )
            if parsed.valid:
                try:
                    outcome = primary_outcome(parsed, str(spec.independent_behavior_task["primary_outcome"]))
                    if orient_primary_outcome(
                        spec.construct_id, outcome, dict(row.get("task_metadata") or {})
                    ) is not None:
                        valid += 1
                except (TypeError, ValueError) as exc:
                    if len(invalid_examples) < 5:
                        invalid_examples.append({"record_id": row.get("record_id"), "reason": str(exc)})
            elif len(invalid_examples) < 5:
                invalid_examples.append(
                    {"record_id": row.get("record_id"), "reason": parsed.error or "invalid parser output"}
                )
        rate = valid / len(group_rows) if group_rows else 0.0
        key = f"{kind}:{dose:g}"
        group_reports[key] = {
            "rows": len(group_rows),
            "valid_primary_rows": valid,
            "primary_valid_rate": rate,
            "selected_prompt_count": len({row.get("prompt_id") for row in group_rows}),
        }
        if not group_rows:
            group_failures.append(f"missing required steering group {key}")
        elif rate < minimum_valid_rate:
            group_failures.append(
                f"{key} primary_valid_rate={rate:.4f} < {minimum_valid_rate:.4f}"
            )
        if group_rows and {str(row.get("prompt_id")) for row in group_rows} != selected:
            group_failures.append(
                f"{key} does not contain every selected steering prompt"
            )

    target_intervention_rows = sum(
        len(grouped.get(("target", dose), []))
        for dose in required_doses_by_direction_kind.get("target", ())
        if float(dose) != 0.0
    )
    if injection_rows <= 0 or target_intervention_rows <= 0:
        group_failures.append("no registered nonzero target intervention row was observed")
    if timing_values != {expected_intervention_timing}:
        group_failures.append(
            f"intervention_timing={sorted(timing_values)!r}, "
            f"expected [{expected_intervention_timing!r}]"
        )
    if missing_prompts:
        group_failures.append(f"missing selected steering prompts: {missing_prompts[:3]}")

    # Movement is evaluated only on the injection-layer row.  Tracking rows at
    # later layers live in a different representation space and must not be
    # counted as repeated evidence for the injection manipulation check.
    movement_rows: list[dict[str, Any]] = []
    movement_failures: list[str] = []
    movement_by_dose: dict[float, list[float]] = defaultdict(list)
    for row in relevant:
        if row.get("injection_applied") is not True:
            continue
        try:
            dose = float(row["dose"])
        except (KeyError, TypeError, ValueError):
            continue
        if dose == 0.0 or str(row.get("direction_kind")) != "target":
            continue
        tracking_layer = row.get("tracking_layer")
        injection_layer = row.get("injection_layer")
        if (
            tracking_layer is not None
            and injection_layer is not None
            and int(tracking_layer) != int(injection_layer)
        ):
            continue
        observed_raw = row.get("observed_shift")
        if observed_raw is None and row.get("pre_projection") is not None and row.get("post_projection") is not None:
            observed_raw = float(row["post_projection"]) - float(row["pre_projection"])
        expected_raw = row.get("expected_shift", row.get("physical_scale"))
        try:
            observed = float(observed_raw)
            expected = float(expected_raw)
        except (TypeError, ValueError):
            if (
                require_correct_injection_sign
                or minimum_mean_abs_injection_shift > 0.0
                or minimum_target_dose_response_span > 0.0
            ):
                movement_failures.append(
                    f"{row.get('record_id')}: missing numeric observed_shift/expected_shift"
                )
            continue
        if not math.isfinite(observed) or not math.isfinite(expected) or expected == 0.0:
            movement_failures.append(f"{row.get('record_id')}: non-finite or zero expected injection shift")
            continue
        movement_rows.append(row)
        movement_by_dose[dose].append(observed)
        if require_correct_injection_sign and observed * expected <= 0.0:
            movement_failures.append(
                f"{row.get('record_id')}: observed injection shift has the wrong sign"
            )

    movement_means = {
        f"{dose:g}": float(sum(values) / len(values))
        for dose, values in sorted(movement_by_dose.items())
        if values
    }
    movement_abs_means = {
        f"{dose:g}": float(sum(abs(value) for value in values) / len(values))
        for dose, values in sorted(movement_by_dose.items())
        if values
    }
    for dose, values in sorted(movement_by_dose.items()):
        abs_mean = sum(abs(value) for value in values) / len(values)
        if abs_mean < minimum_mean_abs_injection_shift:
            movement_failures.append(
                f"target:{dose:g} mean_abs_injection_shift={abs_mean:.6g} "
                f"< {minimum_mean_abs_injection_shift:.6g}"
            )
    positive = movement_by_dose.get(1.0, [])
    negative = movement_by_dose.get(-1.0, [])
    dose_response_span = None
    if positive and negative:
        dose_response_span = abs(
            sum(positive) / len(positive) - sum(negative) / len(negative)
        )
        if dose_response_span < minimum_target_dose_response_span:
            movement_failures.append(
                f"target dose-response span={dose_response_span:.6g} "
                f"< {minimum_target_dose_response_span:.6g}"
            )
    elif minimum_target_dose_response_span > 0.0:
        movement_failures.append("target -1 and +1 dose groups are required for dose-response span")
    if require_correct_injection_sign and not movement_rows:
        movement_failures.append("no numeric nonzero target injection movement rows were observed")
    if movement_failures:
        group_failures.extend(movement_failures)

    return {
        "selected_item_count": len(selected_prompt_ids),
        "observed_selected_item_count": len(observed_prompts),
        "missing_prompt_count": len(missing_prompts),
        "missing_prompt_examples": missing_prompts[:5],
        "required_direction_kinds": sorted(required_doses_by_direction_kind),
        "required_doses_by_direction_kind": {
            kind: [float(dose) for dose in doses]
            for kind, doses in required_doses_by_direction_kind.items()
        },
        "groups": group_reports,
        "injection_applied_rows": injection_rows,
        "intervention_timings": sorted(timing_values),
        "expected_intervention_timing": expected_intervention_timing,
        "invalid_examples": invalid_examples,
        "injection_movement": {
            "numeric_nonzero_target_rows": len(movement_rows),
            "mean_observed_shift_by_dose": movement_means,
            "mean_abs_observed_shift_by_dose": movement_abs_means,
            "dose_response_span_abs_plus_minus": dose_response_span,
            "require_correct_sign": require_correct_injection_sign,
            "minimum_mean_abs_shift": minimum_mean_abs_injection_shift,
            "minimum_dose_response_span": minimum_target_dose_response_span,
            "failures": movement_failures,
        },
        "pass": not group_failures,
        "failures": group_failures,
    }


def validate_preflight(
    *,
    selection_manifest: Mapping[str, Any],
    construct_specs: Mapping[str, ConstructSpec],
    behavior_output: str | Path,
    collateral_output: str | Path,
    steering_outputs: Mapping[str, str | Path],
    thresholds: Mapping[str, Any] | None = None,
    gate_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one model's frozen preflight outputs without inference."""

    if selection_manifest.get("manifest_type") != PREFLIGHT_MANIFEST_TYPE:
        raise ValueError("Unexpected preflight selection manifest type.")
    if selection_manifest.get("selection_informed_by_outcomes") is not False:
        raise ValueError("Preflight selection must explicitly be outcome-independent.")
    gate_config = dict(gate_config or {})
    if gate_config:
        if selection_manifest.get("gate_id") != gate_config.get("gate_id"):
            raise ValueError("Preflight selection does not use the requested gate config.")
        if selection_manifest.get("preflight_id") != PREFLIGHT_ID:
            raise ValueError("Preflight selection is not a v2 selection manifest.")
        configured_hash = gate_config.get("gate_config_sha256")
        if configured_hash is not None and selection_manifest.get("gate_config_sha256") != configured_hash:
            raise ValueError("Preflight selection was created from a different gate-config file.")
        release = dict(gate_config.get("prompt_release", {}))
        inventory_name = str(selection_manifest.get("source_inventory", "")).casefold()
        missing_tokens = [
            str(value).casefold()
            for value in release.get("required_path_tokens", [])
            if str(value).casefold() not in inventory_name
        ]
        forbidden_tokens = [
            str(value).casefold()
            for value in release.get("forbidden_path_tokens", [])
            if str(value).casefold() in inventory_name
        ]
        if missing_tokens or forbidden_tokens:
            raise ValueError(
                "Preflight selection source inventory is not the approved v2 release: "
                f"missing={missing_tokens}, forbidden={forbidden_tokens}"
            )
        configured_construct_ids = [str(value) for value in gate_config.get("construct_ids", [])]
        if configured_construct_ids != [str(value) for value in selection_manifest.get("construct_ids", [])]:
            raise ValueError("Preflight selection construct IDs differ from the v2 gate config.")
    selection_hash = selection_manifest.get("selection_sha256")
    if not isinstance(selection_hash, str) or not selection_hash:
        raise ValueError("Preflight selection is missing selection_sha256.")
    selection_without_hash = {
        key: value for key, value in selection_manifest.items() if key != "selection_sha256"
    }
    if canonical_hash(selection_without_hash) != selection_hash:
        raise ValueError("Preflight selection manifest has an invalid selection_sha256.")
    model = _as_model(selection_manifest.get("model", {}))
    construct_ids = [str(value) for value in selection_manifest.get("construct_ids", [])]
    if set(construct_ids) != set(construct_specs):
        raise ValueError("Preflight construct IDs and loaded construct specifications differ.")
    bounds = dict(selection_manifest.get("item_bounds", {}))
    minimum_items = int(bounds.get("minimum", DEFAULT_MINIMUM_ITEMS))
    maximum_items = int(bounds.get("maximum", DEFAULT_MAXIMUM_ITEMS))
    _validate_bounds(minimum_items, int(bounds.get("target", DEFAULT_TARGET_ITEMS)), maximum_items)
    if gate_config:
        configured_bounds = dict(gate_config.get("item_bounds", {}))
        expected_bounds = {
            "minimum": int(configured_bounds["minimum"]),
            "target": int(configured_bounds["target"]),
            "maximum": int(configured_bounds["maximum"]),
        }
        if bounds != expected_bounds:
            raise ValueError("Preflight selection item bounds differ from the v2 gate config.")
        thresholds = {
            **dict(gate_config.get("execution_contract", {})),
            **dict(gate_config.get("release_thresholds", {})),
        }
    else:
        thresholds = dict(thresholds or {})
    behavior_min_valid = float(thresholds.get("behavior_minimum_valid_rate", 1.0))
    behavior_max_invalid = int(thresholds.get("behavior_maximum_invalid_items", 0))
    behavior_min_distinct = int(thresholds.get("behavior_minimum_distinct_outcomes", 3))
    behavior_min_sd = float(thresholds.get("behavior_minimum_sample_sd", 2.0))
    behavior_max_ceiling = thresholds.get("behavior_maximum_ceiling_share")
    behavior_max_floor = thresholds.get("behavior_maximum_floor_share")
    behavior_max_ceiling = None if behavior_max_ceiling is None else float(behavior_max_ceiling)
    behavior_max_floor = None if behavior_max_floor is None else float(behavior_max_floor)
    collateral_min_valid = float(thresholds.get("collateral_minimum_valid_rate", 0.95))
    collateral_min_correct = float(thresholds.get("collateral_minimum_correctness_rate", 0.75))
    steering_min_valid = float(thresholds.get("steering_minimum_valid_rate", 0.95))
    required_prompt_format = thresholds.get(
        "required_prompt_format", thresholds.get("prompt_format")
    )
    require_constrained = bool(
        thresholds.get(
            "require_constrained_numeric_generation",
            thresholds.get("constrained_numeric_generation", False),
        )
    )
    require_thinking_disabled = bool(thresholds.get("disable_thinking_when_supported", False))
    require_injection_sign = bool(thresholds.get("steering_require_correct_injection_sign", False))
    minimum_abs_shift = float(thresholds.get("steering_minimum_mean_abs_injection_shift", 0.0))
    minimum_dose_span = float(thresholds.get("steering_minimum_target_dose_response_span", 0.0))
    inventory_hash = selection_manifest.get("source_inventory_sha256")

    behavior_rows, behavior_manifest = _validate_model_output_manifest(
        behavior_output,
        expected_manifest_type="construct_behavior_output",
        expected_model=model,
        expected_inventory_sha256=inventory_hash if gate_config else None,
        required_prompt_format=required_prompt_format,
        require_constrained_numeric_generation=require_constrained,
        require_manifest_record_count=bool(gate_config),
        require_thinking_disabled=require_thinking_disabled,
    )
    collateral_rows, collateral_manifest = _validate_model_output_manifest(
        collateral_output,
        expected_manifest_type="construct_behavior_output",
        expected_model=model,
        expected_inventory_sha256=inventory_hash if gate_config else None,
        required_prompt_format=required_prompt_format,
        require_constrained_numeric_generation=require_constrained,
        require_manifest_record_count=bool(gate_config),
        require_thinking_disabled=require_thinking_disabled,
    )
    steering_rows_by_construct: dict[str, list[dict[str, Any]]] = {}
    steering_manifests: dict[str, dict[str, Any]] = {}
    for construct_id in construct_ids:
        path = steering_outputs.get(construct_id)
        if path is None:
            raise ValueError(f"No steering output supplied for {construct_id}.")
        steering_rows_by_construct[construct_id], steering_manifests[construct_id] = _validate_model_output_manifest(
            path,
            expected_manifest_type="construct_steering_output",
            expected_model=model,
            expected_inventory_sha256=inventory_hash if gate_config else None,
            required_prompt_format=required_prompt_format,
            require_constrained_numeric_generation=require_constrained,
            require_manifest_record_count=bool(gate_config),
            require_thinking_disabled=require_thinking_disabled,
        )

    constructs: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    selected = dict(selection_manifest["selected"])
    for construct_id in construct_ids:
        spec = construct_specs[construct_id]
        selected_behavior = list(selected[construct_id]["behavior_eval"]["prompt_ids"])
        selected_collateral = list(selected[construct_id]["collateral_eval"]["prompt_ids"])
        selected_steering = list(selected[construct_id]["steering_eval"]["prompt_ids"])
        behavior_selected_rows, behavior_missing = _selected_rows(
            behavior_rows,
            construct_id=construct_id,
            split="behavior_eval",
            prompt_ids=selected_behavior,
            expected_model=model,
        )
        collateral_selected_rows, collateral_missing = _selected_rows(
            collateral_rows,
            construct_id=construct_id,
            split="collateral_eval",
            prompt_ids=selected_collateral,
            expected_model=model,
        )
        behavior_stats = _behavior_stats(behavior_selected_rows, spec)
        behavior_pass, behavior_failures = _gate_behavior(
            behavior_stats,
            minimum_items=minimum_items,
            maximum_items=maximum_items,
            minimum_valid_rate=behavior_min_valid,
            maximum_invalid_items=behavior_max_invalid,
            minimum_distinct_outcomes=behavior_min_distinct,
            minimum_sample_sd=behavior_min_sd,
            maximum_ceiling_share=behavior_max_ceiling,
            maximum_floor_share=behavior_max_floor,
        )
        if behavior_missing:
            behavior_pass = False
            behavior_failures.append(f"missing selected behavior prompts: {behavior_missing[:3]}")

        collateral_stats = _behavior_stats(collateral_selected_rows, spec, collateral=True)
        collateral_correctness = collateral_stats.get("mean_correctness")
        collateral_failures: list[str] = []
        if collateral_missing:
            collateral_failures.append(f"missing selected collateral prompts: {collateral_missing[:3]}")
        if not minimum_items <= len(collateral_selected_rows) <= maximum_items:
            collateral_failures.append(f"item_count={len(collateral_selected_rows)} outside [{minimum_items}, {maximum_items}]")
        if float(collateral_stats["valid_primary_rate"]) < collateral_min_valid:
            collateral_failures.append(
                f"valid_primary_rate={collateral_stats['valid_primary_rate']:.4f} < {collateral_min_valid:.4f}"
            )
        if collateral_correctness is None or float(collateral_correctness) < collateral_min_correct:
            collateral_failures.append(
                f"correctness_rate={collateral_correctness!r} < {collateral_min_correct:.4f}"
            )
        collateral_pass = not collateral_failures

        steering = _steering_preflight(
            steering_rows_by_construct[construct_id],
            spec,
            selected_prompt_ids=selected_steering,
            expected_model=model,
            minimum_valid_rate=steering_min_valid,
            required_doses_by_direction_kind=selection_manifest.get(
                "steering_requirements", {}
            ).get("required_doses_by_direction_kind", DEFAULT_STEERING_DOSES),
            require_correct_injection_sign=require_injection_sign,
            minimum_mean_abs_injection_shift=minimum_abs_shift,
            minimum_target_dose_response_span=minimum_dose_span,
            expected_intervention_timing=str(
                thresholds.get("steering_intervention_timing", "prefill_only")
            ),
        )
        constructs[construct_id] = {
            "behavior": {"pass": behavior_pass, "stats": behavior_stats, "failures": behavior_failures},
            "collateral": {
                "pass": collateral_pass,
                "stats": collateral_stats,
                "correctness_rate": collateral_correctness,
                "failures": collateral_failures,
            },
            "accessibility": steering,
            "pass": behavior_pass and collateral_pass and bool(steering["pass"]),
        }
        for stage, passed, stage_failures in (
            ("behavior", behavior_pass, behavior_failures),
            ("collateral", collateral_pass, collateral_failures),
            ("accessibility", bool(steering["pass"]), list(steering["failures"])),
        ):
            if not passed:
                failures.append(
                    {
                        "construct_id": construct_id,
                        "stage": stage,
                        "failures": list(stage_failures),
                    }
                )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": PREFLIGHT_REPORT_TYPE,
        "preflight_id": selection_manifest.get("preflight_id", PREFLIGHT_ID),
        "confirmatory": False,
        "status": "complete",
        "release_decision": "pass_preflight" if not failures else "hold_large_execution",
        "model": model,
        "selection_manifest": {
            "path": str(selection_manifest.get("output", "")),
            "selection_sha256": selection_manifest.get("selection_sha256"),
            "source_inventory_sha256": selection_manifest.get("source_inventory_sha256"),
        },
        "source_outputs": {
            "behavior": {"path": str(behavior_output), "manifest": behavior_manifest},
            "collateral": {"path": str(collateral_output), "manifest": collateral_manifest},
            "steering": {
                construct_id: {"path": str(steering_outputs[construct_id]), "manifest": steering_manifests[construct_id]}
                for construct_id in construct_ids
            },
        },
        "thresholds": {
            "item_bounds": {"minimum": minimum_items, "maximum": maximum_items},
            "behavior_minimum_valid_rate": behavior_min_valid,
            "behavior_maximum_invalid_items": behavior_max_invalid,
            "behavior_minimum_distinct_outcomes": behavior_min_distinct,
            "behavior_minimum_sample_sd": behavior_min_sd,
            "behavior_maximum_ceiling_share": behavior_max_ceiling,
            "behavior_maximum_floor_share": behavior_max_floor,
            "collateral_minimum_valid_rate": collateral_min_valid,
            "collateral_minimum_correctness_rate": collateral_min_correct,
            "steering_minimum_valid_rate": steering_min_valid,
            "required_prompt_format": required_prompt_format,
            "require_constrained_numeric_generation": require_constrained,
            "disable_thinking_when_supported": require_thinking_disabled,
            "steering_require_correct_injection_sign": require_injection_sign,
            "steering_minimum_mean_abs_injection_shift": minimum_abs_shift,
            "steering_minimum_target_dose_response_span": minimum_dose_span,
        },
        "constructs": constructs,
        "failures": failures,
        "interpretation": (
            "A pass releases only the model/construct pair for subsequent larger execution. "
            "It does not establish decodability, causal interchange, or steerability."
        ),
    }
    report["report_sha256"] = canonical_hash(report)
    return report


__all__ = [
    "DEFAULT_MAXIMUM_ITEMS",
    "DEFAULT_MINIMUM_ITEMS",
    "DEFAULT_TARGET_ITEMS",
    "PREFLIGHT_MANIFEST_TYPE",
    "PREFLIGHT_REPORT_TYPE",
    "PREFLIGHT_SPLITS",
    "prepare_selection_manifest",
    "repository_relative_path",
    "validate_preflight",
]
