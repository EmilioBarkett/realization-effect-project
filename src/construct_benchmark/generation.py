"""Generic construct-aware synthetic prompt generation.

This module owns the benchmark-facing contract. It uses the OpenAI Responses
transport for the active Luna generation workflow and does not reuse the
realization-specific response fields or validation rules from the legacy
activation prompt generator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from activation_analysis.openai_prompt_generation import call_openai_responses

from .prompts import PromptRecord, validate_prompt_records, write_prompt_records
from .schemas import ConstructSpec, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from .splits import SPLIT_PROMPT_ROLE


RequestFn = Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]]
JobCompletionFn = Callable[["ConstructGenerationJob", tuple[PromptRecord, ...], dict[str, Any]], None]
JobStartFn = Callable[["ConstructGenerationJob"], None]
JobAttemptFn = Callable[["ConstructGenerationJob", int, Mapping[str, Any], str | None], None]
JobRecordsValidatorFn = Callable[["ConstructGenerationJob", tuple[PromptRecord, ...]], None]
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULT_ESTIMATED_INPUT_TOKENS_PER_REQUEST = 1400
DEFAULT_ESTIMATED_OUTPUT_TOKENS_PER_RECORD = 300
MAX_GENERATED_PROMPT_CHARS = 2000
MAX_GENERATED_NOTES_CHARS = 2000
PROBE_WRAPPER_NORMALIZATION_VERSION = "2"

# Final downstream inputs must not contain control-plane instructions or the
# probe-only continuation tail.  These checks are shared by generation and
# release auditing so a prompt cannot pass one path and fail the other.
DOWNSTREAM_PROBE_ONLY_SUFFIX = "continue processing the scenario."
DOWNSTREAM_GENERATION_ONLY_PATTERNS = (
    re.compile(r"\buse (?:new|distinct|different) (?:domains?|actors?|entities?|scenarios?|item identifiers?)\b"),
    re.compile(r"\b(?:registered|preregistered) (?:task|response|prompt|category|schedule)\b"),
    re.compile(r"\b(?:content pool|prompt family|construct id|generation mode|generation plan)\b"),
    re.compile(r"\bdo not mention (?:or reuse )?an earlier (?:scenario|prompt|item)\b"),
)
DOWNSTREAM_GENERIC_SETUP_MARKERS = (
    re.compile(r"\bfor a separate (?:decision|choice)\b"),
    re.compile(r"\bfor the separate allocation problem\b"),
)
DOWNSTREAM_RESPONSE_MARKER = re.compile(
    r"\b(?:return|respond|report|output|provide|reply|answer)(?:\s+(?:with|as))?\s+exactly\b"
)
SINGLE_INTEGER_CHOICE_PATTERN = re.compile(
    r"\b(?:1\s*(?:or|/)\s*2|one\s+or\s+two|1\s+(?:for|if)\b.{0,180}\b2\s+(?:for|if)\b)"
)


def downstream_prompt_text_issues(
    prompt_text: str,
    *,
    expected_output_format: str | None = None,
) -> tuple[str, ...]:
    """Return lexical violations of the final downstream prompt contract.

    The checks are deliberately conservative.  They detect composition
    artifacts such as a copied generation instruction, a duplicated generic
    task setup, or a probe continuation suffix; they do not claim to prove
    semantic independence from the probe.
    """

    del expected_output_format  # reserved for future format-specific checks
    folded = str(prompt_text).casefold()
    issues: list[str] = []
    if DOWNSTREAM_PROBE_ONLY_SUFFIX in folded:
        issues.append("downstream prompt contains the probe-only continuation suffix")
    for pattern in DOWNSTREAM_GENERATION_ONLY_PATTERNS:
        if pattern.search(folded):
            issues.append(f"downstream prompt contains generation-only directive: {pattern.pattern}")
    if any(len(pattern.findall(folded)) > 1 for pattern in DOWNSTREAM_GENERIC_SETUP_MARKERS):
        issues.append("downstream prompt repeats the generic task setup")
    response_count = len(DOWNSTREAM_RESPONSE_MARKER.findall(folded))
    if response_count > 1:
        issues.append("downstream prompt contains multiple response contracts")
    if response_count == 1:
        response_tail = folded[folded.rfind("exactly") :]
        if re.search(r"\b(?:continue processing|generate|rewrite|use new|do not mention)\b", response_tail):
            issues.append("downstream prompt contains an instruction after its response contract")
    return tuple(dict.fromkeys(issues))


def _default_generation_run_modes() -> dict[str, dict[str, Any]]:
    return {
        "review": {
            "purpose": "prompt_review",
            "count_per_model_per_cell": 1,
            "partial": True,
        },
        "full": {
            "purpose": "frozen_full_inventory",
            "count_per_model_per_cell": None,
            "partial": False,
        },
    }


def resolve_generation_mode(plan: Mapping[str, Any], mode: str) -> tuple[str, dict[str, Any]]:
    """Return a validated generation mode from a loaded plan."""

    modes = plan.get("run_modes", _default_generation_run_modes())
    if not isinstance(modes, Mapping):
        raise ValueError("run_modes must be an object.")
    mode_id = _identifier(mode, field_name="generation mode")
    if mode_id not in modes:
        raise ValueError(f"Unknown generation mode={mode_id!r}; available modes are {sorted(modes)}.")
    return mode_id, dict(modes[mode_id])


def _validate_generation_run_modes(value: Any) -> dict[str, dict[str, Any]]:
    raw_modes = _default_generation_run_modes() if value is None else _mapping(value, field_name="run_modes")
    required_modes = {"review", "full"}
    missing_modes = required_modes - set(raw_modes)
    if missing_modes:
        raise ValueError(f"run_modes is missing required mode(s): {sorted(missing_modes)}")
    validated: dict[str, dict[str, Any]] = {}
    for raw_mode_id, raw_mode in raw_modes.items():
        mode_id = _identifier(raw_mode_id, field_name="run_modes key")
        mode = _mapping(raw_mode, field_name=f"run_modes.{mode_id}")
        _text(mode.get("purpose"), field_name=f"run_modes.{mode_id}.purpose")
        count = mode.get("count_per_model_per_cell")
        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 1):
            raise ValueError(
                f"run_modes.{mode_id}.count_per_model_per_cell must be a positive integer or null."
            )
        partial = mode.get("partial")
        if not isinstance(partial, bool):
            raise ValueError(f"run_modes.{mode_id}.partial must be a boolean.")
        if mode_id == "review" and not partial:
            raise ValueError("run_modes.review.partial must be true.")
        if mode_id == "full" and (partial or count is not None):
            raise ValueError(
                "run_modes.full must have partial=false and count_per_model_per_cell=null."
            )
        mode["count_per_model_per_cell"] = count
        mode["partial"] = partial
        validated[mode_id] = mode
    return validated


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "none"


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _identifier(value: Any, *, field_name: str) -> str:
    identifier = _text(value, field_name=field_name)
    if not _ID_PATTERN.fullmatch(identifier):
        raise ValueError(f"{field_name}={identifier!r} is not a valid lowercase identifier.")
    return identifier


def _plan_hash(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    return dict(value)


def _string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field_name} must be a non-empty list of strings.")
    return [item.strip() for item in value]


def _task_for_prompt_role(spec: ConstructSpec, prompt_role: str) -> Mapping[str, Any]:
    if prompt_role == "collateral":
        if spec.collateral_behavior_task is None:
            raise ValueError(
                f"Construct {spec.construct_id!r} does not define collateral_behavior_task."
            )
        return spec.collateral_behavior_task
    return spec.independent_behavior_task


def _category_balance(
    value: Any,
    *,
    field_name: str,
    count: int,
    spec: ConstructSpec,
    prompt_role: str,
) -> dict[str, list[Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object mapping metadata fields to schedules.")
    task = _task_for_prompt_role(spec, prompt_role)
    properties = dict(task["item_metadata_schema"]["properties"])
    validated: dict[str, list[Any]] = {}
    for field_name_key, raw_schedule in value.items():
        field = _identifier(field_name_key, field_name=f"{field_name}.field")
        if field not in properties:
            raise ValueError(f"{field_name} references unknown task metadata field={field!r}.")
        property_schema = properties[field]
        enum = property_schema.get("enum")
        if isinstance(enum, list) and enum:
            allowed_values = enum
        elif property_schema.get("type") == "boolean":
            allowed_values = [True, False]
        else:
            raise ValueError(f"{field_name}.{field} must reference a registered categorical field.")
        if not isinstance(raw_schedule, list) or len(raw_schedule) != count:
            raise ValueError(f"{field_name}.{field} must contain exactly {count} scheduled values.")
        if any(item not in allowed_values for item in raw_schedule):
            raise ValueError(f"{field_name}.{field} contains values outside the registered categories.")
        validated[field] = list(raw_schedule)
    return validated


def _paired_metadata_schema(spec: ConstructSpec) -> dict[str, Any] | None:
    """Return an optional schema for metadata attached to paired prompts.

    Most constructs only need metadata on downstream single prompts.  A
    construct may opt into paired metadata through its opaque ``metadata``
    section; keeping this opt-in preserves the existing paired response
    contract for the other constructs.
    """

    raw_metadata = dict(spec.metadata or {})
    raw_schema = raw_metadata.get("paired_item_metadata_schema")
    if raw_schema is None:
        return None
    schema = _mapping(raw_schema, field_name="metadata.paired_item_metadata_schema")
    properties = _mapping(
        schema.get("properties"),
        field_name="metadata.paired_item_metadata_schema.properties",
    )
    required = _string_list(
        schema.get("required"),
        field_name="metadata.paired_item_metadata_schema.required",
    )
    if not required:
        raise ValueError("metadata.paired_item_metadata_schema.required must not be empty.")
    if set(required) != set(properties):
        raise ValueError(
            "metadata.paired_item_metadata_schema must require every declared property exactly once."
        )
    for property_name, raw_property in properties.items():
        _identifier(property_name, field_name="paired metadata property")
        property_schema = _mapping(
            raw_property,
            field_name=f"metadata.paired_item_metadata_schema.properties.{property_name}",
        )
        property_type = _text(
            property_schema.get("type"),
            field_name=f"metadata.paired_item_metadata_schema.properties.{property_name}.type",
        )
        if property_type not in {"string", "integer", "number", "boolean"}:
            raise ValueError(f"Unsupported paired metadata type={property_type!r}.")
        enum = property_schema.get("enum")
        if enum is not None and (not isinstance(enum, list) or not enum):
            raise ValueError(
                f"metadata.paired_item_metadata_schema.properties.{property_name}.enum must be a non-empty list."
            )
    schema["required"] = list(required)
    schema["properties"] = properties
    return schema


def _paired_metadata_schedule(value: Any, *, field_name: str, count: int, spec: ConstructSpec) -> dict[str, Any] | None:
    """Validate a deterministic per-pair metadata schedule on a paired cell."""

    if value is None:
        return None
    schedule = _mapping(value, field_name=field_name)
    metadata_schema = _paired_metadata_schema(spec)
    if metadata_schema is None:
        raise ValueError(f"{field_name} requires metadata.paired_item_metadata_schema.")
    field = _identifier(schedule.get("field"), field_name=f"{field_name}.field")
    properties = dict(metadata_schema["properties"])
    if field not in properties:
        raise ValueError(f"{field_name}.field={field!r} is not a paired metadata property.")
    positions = schedule.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError(f"{field_name}.positions must be a non-empty list.")
    property_schema = properties[field]
    enum = property_schema.get("enum")
    if isinstance(enum, list) and any(position not in enum for position in positions):
        raise ValueError(f"{field_name}.positions contains values outside the metadata enum.")
    repeats = schedule.get("repeats_per_request")
    request_size = schedule.get("request_size")
    seed = schedule.get("seed")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"{field_name}.repeats_per_request must be a positive integer.")
    if not isinstance(request_size, int) or isinstance(request_size, bool) or request_size < 1:
        raise ValueError(f"{field_name}.request_size must be a positive integer.")
    if request_size != len(positions) * repeats:
        raise ValueError(
            f"{field_name}.request_size must equal len(positions) * repeats_per_request."
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{field_name}.seed must be an integer.")
    review_policy = _text(schedule.get("review_policy"), field_name=f"{field_name}.review_policy")
    if review_policy != "first_scheduled_value":
        raise ValueError(f"{field_name}.review_policy must be 'first_scheduled_value'.")
    if count >= request_size and count % request_size:
        raise ValueError(
            f"{field_name} requires count_per_model={count} to be divisible by request_size={request_size}."
        )
    schedule["field"] = field
    schedule["positions"] = list(positions)
    schedule["repeats_per_request"] = repeats
    schedule["request_size"] = request_size
    schedule["seed"] = seed
    schedule["review_policy"] = review_policy
    return schedule


def _paired_metadata_assignment(job: ConstructGenerationJob, index: int) -> dict[str, Any]:
    schedule = job.cell.get("paired_metadata_schedule")
    if not isinstance(schedule, Mapping):
        return {}
    positions = list(schedule["positions"])
    schedule_request_size = int(schedule["request_size"])

    # Review requests intentionally contain one pair and retain the first
    # value from the registered deterministic schedule.  Full requests may be
    # larger than the plan's historical ten-pair block (for example, a
    # forty-pair request).  Build a balanced block for the actual request so
    # every registered position occurs equally often within that request.
    if job.count == 1:
        repeats = int(schedule["repeats_per_request"])
        global_index = int(job.item_offset) + index
        block_index, block_offset = divmod(global_index, schedule_request_size)
        block = positions * repeats
        random.Random(int(schedule["seed"]) + block_index).shuffle(block)
        return {str(schedule["field"]): block[block_offset]}

    if job.count % len(positions) == 0:
        repeats = job.count // len(positions)
        block = positions * repeats
        # request_index is stable for a chunked cell and keeps retries and
        # resumed jobs deterministic while allowing arbitrary full request
        # sizes that are multiples of the number of registered positions.
        random.Random(int(schedule["seed"]) + max(int(job.request_index) - 1, 0)).shuffle(block)
        return {str(schedule["field"]): block[index]}

    # Non-full fixtures smaller than the registered request size are retained
    # for backwards-compatible unit tests; full jobs are rejected by
    # _validate_paired_job_schedule before this fallback can be used.
    global_index = int(job.item_offset) + index
    block_index, block_offset = divmod(global_index, schedule_request_size)
    block = positions * int(schedule["repeats_per_request"])
    random.Random(int(schedule["seed"]) + block_index).shuffle(block)
    return {str(schedule["field"]): block[block_offset]}


def _expected_category_assignments(job: ConstructGenerationJob, index: int) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for field, schedule in dict(job.cell.get("category_balance", {})).items():
        if not schedule:
            continue
        assignments[str(field)] = schedule[(job.item_offset + index) % len(schedule)]
    assignments.update(_paired_metadata_assignment(job, index))
    return assignments


@dataclass(frozen=True)
class ConstructGenerationJob:
    plan_id: str
    construct_id: str
    model_alias: str
    model_id: str
    cell: dict[str, Any]
    count: int
    seed: int
    temperature: float
    model_index: int = 0
    item_offset: int = 0
    request_index: int = 1
    request_total: int = 1

    @property
    def cell_id(self) -> str:
        return str(self.cell["cell_id"])

    @property
    def split(self) -> str:
        return str(self.cell["split"])

    @property
    def prompt_role(self) -> str:
        return str(self.cell["prompt_role"])

    @property
    def mode(self) -> str:
        return str(self.cell["mode"])

    @property
    def prompt_family(self) -> str:
        return str(self.cell["prompt_family"])

    @property
    def content_pool(self) -> str:
        return str(self.cell["content_pool"])

    @property
    def job_id(self) -> str:
        parts = [self.plan_id, self.construct_id, self.model_alias, self.cell_id]
        if self.request_total > 1:
            parts.append(f"part_{self.request_index:03d}")
        return "__".join(_slug(part) for part in parts)


@dataclass(frozen=True)
class GenerationResult:
    records: tuple[PromptRecord, ...]
    jobs: tuple[ConstructGenerationJob, ...]
    request_count: int
    complete: bool
    request_metadata: tuple[dict[str, Any], ...] = ()

    def summary(self) -> dict[str, Any]:
        by_split: dict[str, int] = {}
        by_model: dict[str, int] = {}
        by_condition: dict[str, int] = {}
        for record in self.records:
            by_split[record.split] = by_split.get(record.split, 0) + 1
            model_alias = str(record.metadata.get("source_model_alias", ""))
            by_model[model_alias] = by_model.get(model_alias, 0) + 1
            if record.condition_id:
                by_condition[record.condition_id] = by_condition.get(record.condition_id, 0) + 1
        input_tokens = sum(int(item.get("input_tokens", 0) or 0) for item in self.request_metadata)
        output_tokens = sum(int(item.get("output_tokens", 0) or 0) for item in self.request_metadata)
        actual_cost_usd = sum(float(item.get("actual_cost_usd", 0.0) or 0.0) for item in self.request_metadata)
        return {
            "complete": self.complete,
            "construct_ids": sorted({record.construct_id for record in self.records}),
            "job_count": len(self.jobs),
            "request_count": self.request_count,
            "record_count": len(self.records),
            "records_by_split": dict(sorted(by_split.items())),
            "records_by_model": dict(sorted(by_model.items())),
            "records_by_condition": dict(sorted(by_condition.items())),
            "actual_input_tokens": input_tokens,
            "actual_output_tokens": output_tokens,
            "actual_total_tokens": input_tokens + output_tokens,
            "actual_cost_usd": actual_cost_usd,
            "attempt_count": len(self.request_metadata),
            "rejected_attempt_count": sum(
                item.get("semantic_attempt_status") == "rejected"
                for item in self.request_metadata
            ),
        }


def _apply_generation_plan_overrides(
    plan: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a small downstream-only overlay without changing the base plan."""

    if not isinstance(overrides, Mapping):
        raise ValueError("generation-plan overrides must be an object.")
    allowed = {
        "plan_id",
        "calibration_factor_schedule",
        "downstream_pool_separation",
        "cells",
        "content_pools",
        "append_cells",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unsupported generation-plan override fields: {sorted(unknown)}")
    effective = copy.deepcopy(dict(plan))
    for field in (
        "plan_id",
        "calibration_factor_schedule",
        "downstream_pool_separation",
    ):
        if field in overrides:
            effective[field] = copy.deepcopy(overrides[field])
    raw_cell_overrides = overrides.get("cells", {})
    if not isinstance(raw_cell_overrides, Mapping):
        raise ValueError("generation-plan override cells must be an object keyed by cell_id.")
    if raw_cell_overrides:
        cells = effective.get("cells")
        if not isinstance(cells, list):
            raise ValueError("generation-plan override cells require a base cells list.")
        cells_by_id = {
            str(cell.get("cell_id")): cell
            for cell in cells
            if isinstance(cell, Mapping) and cell.get("cell_id")
        }
        for cell_id, raw_override in raw_cell_overrides.items():
            if str(cell_id) not in cells_by_id:
                raise ValueError(f"generation-plan override references unknown cell_id={cell_id!r}.")
            if not isinstance(raw_override, Mapping):
                raise ValueError(f"generation-plan override for {cell_id!r} must be an object.")
            cells_by_id[str(cell_id)].update(copy.deepcopy(dict(raw_override)))
    if "content_pools" in overrides:
        raw_pool_overrides = overrides["content_pools"]
        if not isinstance(raw_pool_overrides, Mapping):
            raise ValueError("generation-plan override content_pools must be an object.")
        pools = effective.setdefault("content_pools", {})
        if not isinstance(pools, dict):
            raise ValueError("generation-plan override content_pools requires a base object.")
        pools.update(copy.deepcopy(dict(raw_pool_overrides)))
    if "append_cells" in overrides:
        appended = overrides["append_cells"]
        if not isinstance(appended, list) or any(not isinstance(cell, Mapping) for cell in appended):
            raise ValueError("generation-plan override append_cells must be a list of objects.")
        cells = effective.setdefault("cells", [])
        if not isinstance(cells, list):
            raise ValueError("generation-plan override append_cells requires a base cells list.")
        cells.extend(copy.deepcopy(appended))
    return effective


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Merge a versioned generation-plan overlay without mutating its base."""

    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = copy.deepcopy(base)
        for key, value in overlay.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(overlay)


def _load_inherited_plan_payload(path: Path, *, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Generation-plan inheritance cycle: {cycle}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    base_ref = payload.pop("base_plan_path", None)
    overlay = payload.pop("overrides", None)
    if base_ref is None:
        if overlay is not None:
            raise ValueError(f"{path}.overrides requires base_plan_path.")
        return payload
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError(f"{path}.base_plan_path must be a non-empty string.")
    base_path = (path.parent / base_ref).resolve()
    base = _load_inherited_plan_payload(base_path, stack=(*stack, path))
    effective = _deep_merge(base, payload)
    if overlay is not None:
        effective = _apply_generation_plan_overrides(effective, overlay)
    return effective


def load_generation_plan(
    path: str | Path,
    spec: ConstructSpec,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate one construct-specific generation plan.

    A plan may be a small, versioned overlay with ``base_plan_path`` and
    ``overrides``.  The base plan remains immutable; the returned mapping is
    the effective plan whose hash is used for generated-record provenance.
    """

    plan_path = Path(path).resolve()
    plan = _load_inherited_plan_payload(plan_path)
    if overrides is not None:
        plan = _apply_generation_plan_overrides(plan, overrides)
    schema_version = plan.get("schema_version", SCHEMA_VERSION)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported generation-plan schema_version={schema_version!r}.")
    plan_id = _identifier(plan.get("plan_id"), field_name="plan_id")
    construct_id = _identifier(plan.get("construct_id"), field_name="construct_id")
    if construct_id != spec.construct_id:
        raise ValueError(
            f"Generation plan construct_id={construct_id!r} does not match spec {spec.construct_id!r}."
        )
    models = plan.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list.")
    model_aliases: set[str] = set()
    for index, model in enumerate(models):
        model_payload = _mapping(model, field_name=f"models[{index}]")
        alias = _identifier(model_payload.get("alias"), field_name=f"models[{index}].alias")
        _text(model_payload.get("model"), field_name=f"models[{index}].model")
        if alias in model_aliases:
            raise ValueError(f"Duplicate model alias: {alias}")
        model_aliases.add(alias)
    generation = _mapping(plan.get("generation", {}), field_name="generation")
    generation_modes = _validate_generation_run_modes(plan.get("run_modes"))
    if not isinstance(generation.get("seed", 0), int):
        raise ValueError("generation.seed must be an integer.")
    if float(generation.get("temperature", 0.7)) < 0:
        raise ValueError("generation.temperature must be non-negative.")
    retries = generation.get("retries", 0)
    if not isinstance(retries, int) or retries < 0:
        raise ValueError("generation.retries must be a non-negative integer.")
    for estimate_key in (
        "estimated_input_tokens_per_request",
        "estimated_output_tokens_per_record",
    ):
        estimate = generation.get(estimate_key)
        if estimate is not None and (not isinstance(estimate, int) or estimate < 1):
            raise ValueError(f"generation.{estimate_key} must be a positive integer when provided.")

    task_composition = _mapping(plan.get("task_composition"), field_name="task_composition")
    required_composition = {
        "prompt_order": "probe_then_downstream",
        "probe_context": "paired_probe_only",
        "downstream_context": "independent_behavior_task_only",
        "condition_carryover": "state_only",
        "surface_text_carryover": "none",
        "behavior_steering_pool_separation": True,
    }
    for key, expected in required_composition.items():
        if task_composition.get(key) != expected:
            raise ValueError(
                f"task_composition.{key} must be {expected!r}; "
                f"received {task_composition.get(key)!r}."
            )

    raw_pools = _mapping(plan.get("content_pools"), field_name="content_pools")
    if not raw_pools:
        raise ValueError("content_pools must not be empty.")
    pools: dict[str, dict[str, Any]] = {}
    for pool_id, raw_pool in raw_pools.items():
        pool_key = _identifier(pool_id, field_name="content_pools key")
        pool = _mapping(raw_pool, field_name=f"content_pools.{pool_key}")
        pool_role = _text(pool.get("role"), field_name=f"content_pools.{pool_key}.role")
        if pool_role not in {"probe", "behavior", "steering", "calibration", "collateral"}:
            raise ValueError(f"content_pools.{pool_key}.role is unsupported: {pool_role!r}")
        _string_list(pool.get("domains"), field_name=f"content_pools.{pool_key}.domains")
        pools[pool_key] = pool

    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("cells must be a non-empty list.")
    cells: list[dict[str, Any]] = []
    cell_ids: set[str] = set()
    covered_splits: set[str] = set()
    for index, raw_cell in enumerate(raw_cells):
        cell = _mapping(raw_cell, field_name=f"cells[{index}]")
        cell_id = _identifier(cell.get("cell_id"), field_name=f"cells[{index}].cell_id")
        if cell_id in cell_ids:
            raise ValueError(f"Duplicate cell_id: {cell_id}")
        cell_ids.add(cell_id)
        split = _text(cell.get("split"), field_name=f"cells[{index}].split")
        if split not in spec.required_splits:
            raise ValueError(f"cells[{index}].split={split!r} is not required by the construct spec.")
        prompt_role = _text(cell.get("prompt_role"), field_name=f"cells[{index}].prompt_role")
        expected_role = SPLIT_PROMPT_ROLE.get(split)
        if prompt_role != expected_role:
            raise ValueError(
                f"cells[{index}] role={prompt_role!r} does not match split {split!r} role {expected_role!r}."
            )
        mode = _text(cell.get("mode"), field_name=f"cells[{index}].mode")
        expected_mode = "paired" if split in spec.paired_splits else "single"
        if mode != expected_mode:
            raise ValueError(f"cells[{index}].mode must be {expected_mode!r} for split {split!r}.")
        prompt_family = _identifier(cell.get("prompt_family"), field_name=f"cells[{index}].prompt_family")
        pool_id = _identifier(cell.get("content_pool"), field_name=f"cells[{index}].content_pool")
        if pool_id not in pools:
            raise ValueError(f"cells[{index}] references unknown content_pool={pool_id!r}.")
        if pools[pool_id]["role"] != prompt_role:
            raise ValueError(f"content_pool={pool_id!r} role does not match prompt_role={prompt_role!r}.")
        count = cell.get("count_per_model", plan.get("default_count_per_cell_per_model", 1))
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"cells[{index}].count_per_model must be a positive integer.")
        cell["category_balance"] = _category_balance(
            cell.get("category_balance"),
            field_name=f"cells[{index}].category_balance",
            count=count,
            spec=spec,
            prompt_role=prompt_role,
        )
        if mode == "paired":
            cell["paired_metadata_schedule"] = _paired_metadata_schedule(
                cell.get("paired_metadata_schedule"),
                field_name=f"cells[{index}].paired_metadata_schedule",
                count=count,
                spec=spec,
            )
            condition_ids = cell.get("condition_ids")
            if not isinstance(condition_ids, list) or set(condition_ids) != set(spec.condition_ids):
                raise ValueError(
                    f"cells[{index}].condition_ids must exactly match {list(spec.condition_ids)}."
                )
        else:
            if cell.get("paired_metadata_schedule") is not None:
                raise ValueError(
                    f"cells[{index}].paired_metadata_schedule is only valid for paired cells."
                )
            condition_id = str(cell.get("condition_id", "neutral"))
            if condition_id not in {"neutral", *spec.condition_ids}:
                raise ValueError(f"cells[{index}].condition_id is not a construct condition or neutral.")
        covered_splits.add(split)
        cell["cell_id"] = cell_id
        cell["prompt_family"] = prompt_family
        cell["content_pool"] = pool_id
        cell["count_per_model"] = count
        cells.append(cell)
    missing_splits = set(spec.required_splits) - covered_splits
    if missing_splits:
        raise ValueError(f"Generation plan is missing required splits: {sorted(missing_splits)}")

    validated = dict(plan)
    validated["schema_version"] = schema_version
    validated["plan_id"] = plan_id
    validated["construct_id"] = construct_id
    validated["run_modes"] = generation_modes
    validated["cells"] = cells
    validated["content_pools"] = pools
    return validated


def iter_generation_jobs(
    plan: Mapping[str, Any],
    *,
    model_aliases: set[str] | None = None,
    count_per_model_override: int | None = None,
    limit_jobs: int | None = None,
    splits: set[str] | None = None,
) -> Iterable[ConstructGenerationJob]:
    generation = dict(plan.get("generation", {}))
    base_seed = int(generation.get("seed", 0))
    temperature = float(generation.get("temperature", 0.7))
    if count_per_model_override is not None and count_per_model_override < 1:
        raise ValueError("count_per_model_override must be positive.")
    max_items_per_request = int(generation.get("max_items_per_request", 0) or 0)
    request_seed_stride = max_items_per_request + 1 if max_items_per_request > 0 else 1
    emitted = 0
    for model_index, model in enumerate(plan["models"]):
        alias = str(model["alias"])
        if model_aliases is not None and alias not in model_aliases:
            continue
        for cell_index, cell in enumerate(plan["cells"]):
            if splits is not None and str(cell["split"]) not in splits:
                continue
            count = count_per_model_override if count_per_model_override is not None else int(cell["count_per_model"])
            job_index = model_index * len(plan["cells"]) + cell_index
            yield ConstructGenerationJob(
                plan_id=str(plan["plan_id"]),
                construct_id=str(plan["construct_id"]),
                model_alias=alias,
                model_id=str(model["model"]),
                cell=dict(cell),
                count=count,
                seed=base_seed + job_index * request_seed_stride,
                temperature=temperature,
                model_index=model_index,
            )
            emitted += 1
            if limit_jobs is not None and emitted >= limit_jobs:
                return


def _planned_model_aliases(plan: Mapping[str, Any]) -> set[str]:
    return {str(model["alias"]) for model in plan["models"]}


def _assigned_content_domains(plan: Mapping[str, Any], job: ConstructGenerationJob) -> list[str]:
    domains = [str(domain) for domain in plan["content_pools"][job.content_pool]["domains"]]
    return [domains[(job.model_index + job.item_offset + index) % len(domains)] for index in range(job.count)]


def _token_estimate(plan: Mapping[str, Any]) -> tuple[int, int]:
    generation = dict(plan.get("generation", {}))
    return (
        int(generation.get("estimated_input_tokens_per_request", DEFAULT_ESTIMATED_INPUT_TOKENS_PER_REQUEST)),
        int(generation.get("estimated_output_tokens_per_record", DEFAULT_ESTIMATED_OUTPUT_TOKENS_PER_RECORD)),
    )


def _estimated_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    input_usd_per_million_tokens: float | None,
    output_usd_per_million_tokens: float | None,
) -> float | None:
    if input_usd_per_million_tokens is None or output_usd_per_million_tokens is None:
        return None
    return (
        input_tokens / 1_000_000 * input_usd_per_million_tokens
        + output_tokens / 1_000_000 * output_usd_per_million_tokens
    )


def dry_run_summary(
    plan: Mapping[str, Any],
    *,
    model_aliases: set[str] | None = None,
    count_per_model_override: int | None = None,
    splits: set[str] | None = None,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
) -> dict[str, Any]:
    jobs = list(
        iter_generation_jobs(
            plan,
            model_aliases=model_aliases,
            count_per_model_override=count_per_model_override,
            splits=splits,
        )
    )
    max_per_request = int(plan.get("generation", {}).get("max_items_per_request", 0) or 0)
    records_by_split: dict[str, int] = {}
    records_by_model: dict[str, int] = {}
    condition_counts: dict[str, int] = {}
    request_count = 0
    expected_records = 0
    for job in jobs:
        request_count += math.ceil(job.count / max_per_request) if max_per_request else 1
        multiplier = 2 if job.mode == "paired" else 1
        count = job.count * multiplier
        expected_records += count
        records_by_split[job.split] = records_by_split.get(job.split, 0) + count
        records_by_model[job.model_alias] = records_by_model.get(job.model_alias, 0) + count
        if job.mode == "paired":
            for condition_id in job.cell["condition_ids"]:
                condition_counts[condition_id] = condition_counts.get(condition_id, 0) + job.count
        else:
            condition_id = str(job.cell.get("condition_id", "neutral"))
            condition_counts[condition_id] = condition_counts.get(condition_id, 0) + job.count
    input_tokens_per_request, output_tokens_per_record = _token_estimate(plan)
    estimated_input_tokens = request_count * input_tokens_per_request
    estimated_output_tokens = expected_records * output_tokens_per_record
    planned_model_aliases = _planned_model_aliases(plan)
    selected_model_aliases = planned_model_aliases if model_aliases is None else set(model_aliases)
    return {
        "plan_id": plan["plan_id"],
        "construct_id": plan["construct_id"],
        "complete_plan": (
            splits is None
            and count_per_model_override is None
            and selected_model_aliases == planned_model_aliases
        ),
        "selected_splits": sorted(splits) if splits is not None else None,
        "planned_model_aliases": sorted(planned_model_aliases),
        "selected_model_aliases": sorted(selected_model_aliases),
        "count_per_model_override": count_per_model_override,
        "job_count": len(jobs),
        "request_count": request_count,
        "expected_record_count": expected_records,
        "records_by_split": dict(sorted(records_by_split.items())),
        "records_by_model": dict(sorted(records_by_model.items())),
        "records_by_condition": dict(sorted(condition_counts.items())),
        "content_pools": sorted(plan["content_pools"]),
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
        "estimated_cost_usd": _estimated_cost(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            input_usd_per_million_tokens=input_usd_per_million_tokens,
            output_usd_per_million_tokens=output_usd_per_million_tokens,
        ),
        "token_estimate_assumptions": {
            "input_tokens_per_request": input_tokens_per_request,
            "output_tokens_per_record": output_tokens_per_record,
            "pricing_configured": input_usd_per_million_tokens is not None
            and output_usd_per_million_tokens is not None,
        },
    }


def response_schema_for_job(job: ConstructGenerationJob, spec: ConstructSpec) -> dict[str, Any]:
    if job.mode == "paired":
        paired_metadata_schema = _paired_metadata_schema(spec)
        paired_prompt_required = ["condition_id", "prompt_text"]
        paired_prompt_properties: dict[str, Any] = {
            "condition_id": {"type": "string"},
            "prompt_text": {
                "type": "string",
                "maxLength": MAX_GENERATED_PROMPT_CHARS,
            },
        }
        if paired_metadata_schema is not None:
            paired_prompt_required.append("task_metadata")
            paired_prompt_properties["task_metadata"] = {
                "type": "object",
                "additionalProperties": False,
                "required": list(paired_metadata_schema["required"]),
                "properties": dict(paired_metadata_schema["properties"]),
            }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"{_slug(job.construct_id)}_paired_prompt_batch",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pairs"],
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["pair_id", "content_domain", "prompts", "notes"],
                                "properties": {
                                    "pair_id": {"type": "string", "maxLength": 120},
                                    "content_domain": {"type": "string"},
                                    "prompts": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": paired_prompt_required,
                                            "properties": paired_prompt_properties,
                                        },
                                    },
                                    "notes": {"type": "string", "maxLength": MAX_GENERATED_NOTES_CHARS},
                                },
                            },
                        }
                    },
                },
            },
        }
    if job.mode == "single":
        task = _task_for_prompt_role(spec, job.prompt_role)
        item_metadata_schema = dict(task["item_metadata_schema"])
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"{_slug(job.construct_id)}_single_prompt_batch",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["prompts"],
                    "properties": {
                        "prompts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "variant_id",
                                    "content_domain",
                                    "task_metadata",
                                    "prompt_text",
                                    "notes",
                                ],
                                "properties": {
                                    "variant_id": {"type": "string", "maxLength": 120},
                                    "content_domain": {"type": "string"},
                                    "task_metadata": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": list(item_metadata_schema["required"]),
                                        "properties": dict(item_metadata_schema["properties"]),
                                    },
                                    "prompt_text": {
                                        "type": "string",
                                        "maxLength": MAX_GENERATED_PROMPT_CHARS,
                                    },
                                    "notes": {"type": "string", "maxLength": MAX_GENERATED_NOTES_CHARS},
                                },
                            },
                        }
                    },
                },
            },
        }
    raise ValueError(f"Unsupported generation mode: {job.mode!r}")


def build_generation_messages(spec: ConstructSpec, plan: Mapping[str, Any], job: ConstructGenerationJob) -> list[dict[str, str]]:
    condition_descriptions = [
        {
            "condition_id": condition["condition_id"],
            "label": condition["label"],
            "definition": condition["definition"],
        }
        for condition in spec.contrast_conditions
    ]
    cell = job.cell
    task = _task_for_prompt_role(spec, job.prompt_role)
    assigned_domains = _assigned_content_domains(plan, job)
    scheduled_pair_requirements = [
        {
            "pair_index": index + 1,
            "content_domain": assigned_domains[index],
            "task_metadata": _expected_category_assignments(job, index),
        }
        for index in range(job.count)
    ] if job.mode == "paired" else []
    system = (
        "You generate controlled, theory-relevant prompts for a multi-construct "
        "representation benchmark. Return only JSON matching the requested schema. "
        "Do not include markdown or commentary. Do not put condition IDs, construct "
        "names, or benchmark labels in prompt_text unless forbidden terms explicitly "
        "allow a lexical control. Preserve independence between probe and downstream "
        "content pools. In paired mode, make each pair a minimal contrast: preserve the "
        "same actor, entities, quantities, chronology, prose structure, and processing "
        "instruction, changing only the theory-relevant state needed to distinguish the "
        "two registered conditions. Treat the numbered schedule in the user payload as "
        "authoritative: copy each pair's assigned metadata exactly by pair index, do not "
        "reorder, rotate, or infer schedule values. "
        "The forbidden_terms list is an output ban, not vocabulary to echo: "
        "never copy any forbidden term into generated content. For probe prompts, "
        "apply that ban to the Scenario body; terms that occur only in the registered "
        "wrapper are allowed there. Use concrete facts and plain descriptions instead "
        "of naming the target construct or its conditions."
    )
    if job.prompt_role != "probe":
        system += (
            " For downstream single prompts, prompt_text is the final end-user model input, not a description "
            "of the generation task. Instantiate exactly one concrete task from the registered task specification. "
            "Do not paste or repeat the registered task template, cell instructions, design rules, content-pool "
            "names, generation instructions, or benchmark language into prompt_text. Include exactly one concrete "
            "task setup and exactly one response contract as the final instruction. Never include the probe-only "
            "sentence 'Continue processing the scenario.' and never refer to an earlier, prior, or probe scenario."
        )
        if isinstance(plan.get("downstream_pool_separation"), Mapping):
            system += (
                " The downstream pool contract below is authoritative: every generated prompt must contain "
                "at least one required anchor for its assigned content pool, must contain none of that pool's "
                "forbidden anchors, and should use the required anchor word literally in the final prompt."
            )
    user_payload = {
        "task": "Generate prompts for one registered behavioral construct.",
        "construct_id": spec.construct_id,
        "construct_family": spec.family,
        "construct_description": spec.description,
        "condition_definitions": condition_descriptions,
        "prompt_role": job.prompt_role,
        "split": job.split,
        "prompt_family": job.prompt_family,
        "content_pool": job.content_pool,
        "content_pool_domains": plan["content_pools"][job.content_pool]["domains"],
        "assigned_content_domains": assigned_domains,
        "scheduled_pair_requirements": scheduled_pair_requirements,
        "count": job.count,
        "generation_mode": job.mode,
        "probe_template": spec.probe_prompt_template if job.prompt_role == "probe" else None,
        "independent_task": task if job.prompt_role != "probe" else None,
        "registered_downstream_task_template": (
            task["prompt_template"] if job.prompt_role != "probe" else None
        ),
        "item_metadata_schema": task["item_metadata_schema"] if job.prompt_role != "probe" else None,
        "paired_item_metadata_schema": _paired_metadata_schema(spec) if job.mode == "paired" else None,
        "cell_instructions": cell.get("instructions", ""),
        "category_balance": dict(cell.get("category_balance", {})),
        "downstream_pool_separation": (
            plan.get("downstream_pool_separation") if job.prompt_role != "probe" else None
        ),
        "required_downstream_prompt_anchors": (
            list(
                plan.get("downstream_pool_separation", {})
                .get("required_prompt_anchors", {})
                .get(job.content_pool, [])
            )
            if job.prompt_role != "probe"
            else []
        ),
        "forbidden_downstream_prompt_anchors": (
            list(
                plan.get("downstream_pool_separation", {})
                .get("forbidden_prompt_anchors", {})
                .get(job.content_pool, [])
            )
            if job.prompt_role != "probe"
            else []
        ),
        "required_category_assignments": [
            _expected_category_assignments(job, index) for index in range(job.count)
        ],
        "design_rules": plan.get("design_rules", []),
        "task_composition": plan["task_composition"],
        "forbidden_terms": list(plan.get("forbidden_terms", [])) + list(cell.get("forbidden_terms", [])),
        "output_requirement": (
            "For paired mode return exactly count pairs, and each pair must contain exactly one prompt "
            "for each condition_id and the assigned content_domain at the corresponding index. The two "
            "prompt_text values in a pair must be self-contained, closely length-matched minimal contrasts; "
            "When paired_item_metadata_schema is present, include the required task_metadata object on "
            "each condition prompt, use the assigned minority_report_position exactly, and use the same "
            "metadata value in both members of the pair. "
            "do not change names, numbers, setting, stakes, evidence valence, or response instructions unless "
            "that field is itself the registered construct manipulation. Keep every prompt_text at or below "
            f"{MAX_GENERATED_PROMPT_CHARS} characters and notes at or below {MAX_GENERATED_NOTES_CHARS} characters. For single "
            "mode return exactly count prompts with the assigned content_domain at the corresponding index and "
            "task_metadata that exactly follows item_metadata_schema. "
            "Every prompt_text must be a complete model input, not a summary. For every downstream single prompt, "
            "keep prompt_text at or below 1900 characters so the response instruction has safe truncation headroom. "
            "The registered_downstream_task_template is normative as a specification: instantiate its task setup "
            "and include one complete response-format instruction in the final prompt. Do not paste the template "
            "or any generation-only instruction verbatim, and do not append text after the response contract."
        ),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, indent=2, sort_keys=True)},
    ]


def _extract_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Model response did not include choices.")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping):
        raise ValueError("Model response choice did not include a message.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    raise ValueError("Model response message did not include string content.")


def _parse_json_response(response: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(_extract_content(response))
    except json.JSONDecodeError as exc:
        raise ValueError("Model response content was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object.")
    return parsed


def _validate_object_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    context: str,
) -> None:
    keys = set(value)
    unexpected = keys - allowed
    missing = required - keys
    if unexpected:
        raise ValueError(f"{context} contains unexpected field(s): {sorted(unexpected)}.")
    if missing:
        raise ValueError(f"{context} is missing required field(s): {sorted(missing)}.")


def normalize_probe_prompt_wrapper(
    prompt_text: str,
    *,
    probe_prompt_template: str,
) -> tuple[str, bool]:
    """Apply a bounded repair to a registered probe wrapper.

    Prompt generation models occasionally put the scenario on the same line as
    ``Scenario:`` or put the registered continuation tail on the same line as
    the final sentence, or duplicate the wrapper marker immediately before the
    scenario. They can also reproduce a superseded wrapper while preserving the
    current scenario body. The wrapper is part of the benchmark contract, so
    these deviations are canonicalized deterministically rather than silently
    accepted. This helper is intentionally narrow: it requires the registered
    suffix, only removes an adjacent duplicate marker, and never edits the
    scenario body.
    """

    if "{scenario}" not in probe_prompt_template:
        return prompt_text.strip(), False
    prefix, suffix = probe_prompt_template.split("{scenario}", maxsplit=1)
    candidate = prompt_text.strip()
    normalized = candidate

    prefix_anchor = prefix.rstrip()
    if not normalized.startswith(prefix) and normalized.startswith(prefix_anchor):
        remainder = normalized[len(prefix_anchor) :].lstrip()
        normalized = prefix + remainder

    suffix_anchor = suffix.strip()
    if not normalized.endswith(suffix) and suffix_anchor and normalized.endswith(suffix_anchor):
        body = normalized[: -len(suffix_anchor)].rstrip()
        normalized = body + suffix

    scenario_marker = "Scenario:"
    first_marker = normalized.find(scenario_marker)
    second_marker = normalized.find(scenario_marker, first_marker + len(scenario_marker))
    if (
        first_marker >= 0
        and second_marker >= 0
        and normalized[first_marker + len(scenario_marker) : second_marker].strip() == ""
    ):
        normalized = normalized[:second_marker] + normalized[second_marker + len(scenario_marker) :].lstrip()

    if not normalized.startswith(prefix) or not normalized.endswith(suffix):
        if (
            normalized.count(scenario_marker) == 1
            and suffix_anchor
            and normalized.endswith(suffix_anchor)
        ):
            marker_index = normalized.index(scenario_marker) + len(scenario_marker)
            body_end = len(normalized) - len(suffix_anchor)
            scenario_body = normalized[marker_index:body_end].strip()
            if scenario_body:
                normalized = prefix + scenario_body + suffix

    return normalized, normalized != candidate


def _require_registered_probe_wrapper(
    prompt_text: str,
    *,
    probe_prompt_template: str,
    job: ConstructGenerationJob,
) -> None:
    """Reject probe text that remains outside the registered wrapper contract."""

    if "{scenario}" not in probe_prompt_template:
        return
    # Generic fixture generators may intentionally emit a bare placeholder
    # prompt.  The production inventory audit remains the authority for rows
    # that do not contain a scenario wrapper at all; when a response does claim
    # to use the scenario form, enforce the registered wrapper here.
    if "Scenario:" not in prompt_text:
        return
    prefix, suffix = probe_prompt_template.split("{scenario}", maxsplit=1)
    issues: list[str] = []
    if not prompt_text.startswith(prefix):
        issues.append("missing registered wrapper prefix")
    if not prompt_text.endswith(suffix):
        issues.append("missing registered wrapper suffix")
    if prompt_text.count("Scenario:") != 1:
        issues.append("scenario marker count is not exactly one")
    if issues:
        raise ValueError(
            f"{job.job_id} prompt does not satisfy the registered probe wrapper: {'; '.join(issues)}."
        )


def _validate_text(
    prompt_text: Any,
    *,
    plan: Mapping[str, Any],
    job: ConstructGenerationJob,
    probe_prompt_template: str | None = None,
) -> str:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError(f"{job.job_id} returned an empty or non-string prompt_text.")
    text = prompt_text.strip()
    if len(text) > MAX_GENERATED_PROMPT_CHARS:
        raise ValueError(
            f"{job.job_id} prompt_text exceeds {MAX_GENERATED_PROMPT_CHARS} characters."
        )
    forbidden_terms = [
        str(term).strip().lower()
        for term in list(plan.get("forbidden_terms", [])) + list(job.cell.get("forbidden_terms", []))
        if str(term).strip()
    ]
    text_forbidden_check = text
    if job.prompt_role == "probe" and isinstance(probe_prompt_template, str):
        normalized, _ = normalize_probe_prompt_wrapper(
            text,
            probe_prompt_template=probe_prompt_template,
        )
        prefix, suffix = probe_prompt_template.split("{scenario}", maxsplit=1)
        if normalized.startswith(prefix) and normalized.endswith(suffix):
            suffix_length = len(suffix)
            text_forbidden_check = normalized[len(prefix) : len(normalized) - suffix_length]
    lowered = text_forbidden_check.lower()
    forbidden_hits = [term for term in forbidden_terms if re.search(rf"\b{re.escape(term)}\b", lowered)]
    if forbidden_hits:
        raise ValueError(f"{job.job_id} prompt contains forbidden term(s): {sorted(set(forbidden_hits))}.")
    if job.prompt_role != "probe":
        composition_issues = downstream_prompt_text_issues(
            text,
            expected_output_format=str(job.cell.get("expected_output_format", "")),
        )
        if composition_issues:
            raise ValueError(
                f"{job.job_id} downstream prompt composition failed: {'; '.join(composition_issues)}."
            )
    return text


def _registered_response_instruction(task: Mapping[str, Any], expected_format: str) -> str:
    """Extract the registered answer-format sentence for downstream completion.

    Prompt generation models occasionally return a well-formed scenario but omit
    the answer request.  The task template is the source of truth for completing
    that mechanical suffix; this keeps the repair bounded to output formatting
    and avoids inventing task-specific mappings in the runtime.
    """

    template = str(task.get("prompt_template", ""))
    candidates = re.findall(r"(?i)(?:return|report|allocate|provide|enter|reply|answer)[^.?!]*[.?!]", template)
    for candidate in candidates:
        folded = candidate.casefold()
        if expected_format == "single_integer_1_or_2":
            if "1" in folded and "2" in folded and ("return" in folded or "report" in folded):
                return candidate.strip()
        elif expected_format == "single_integer_0_to_100":
            if "0 to 100" in folded and "integer" in folded and ("return" in folded or "report" in folded):
                return candidate.strip()
        elif expected_format == "single_integer_allocation_0_to_100":
            if (
                "0 to 100" in folded
                and "integer" in folded
                and "option a" in folded
                and ("return" in folded or "report" in folded)
            ):
                return candidate.strip()
        elif expected_format == "two_integers_sum_100":
            if "two integers" in folded and "separate line" in folded and "100" in folded:
                return candidate.strip()
        elif expected_format == "two_integers_on_separate_lines":
            if "two integers" in folded and "separate line" in folded:
                return candidate.strip()
    fallbacks = {
        "single_integer_1_or_2": "Return exactly one integer: 1 or 2.",
        "single_integer_0_to_100": "Report the requested probability from 0 to 100 as one integer.",
        "single_integer_allocation_0_to_100": "Return exactly one integer from 0 to 100: the points assigned to option A.",
        "two_integers_sum_100": "Return exactly two integers on separate lines; the two integers must sum to 100.",
        "two_integers_on_separate_lines": "Return exactly two integers on separate lines.",
    }
    try:
        return fallbacks[expected_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported downstream response format={expected_format!r}.") from exc


def _response_instruction_is_present(prompt_text: str, expected_format: str) -> bool:
    """Check whether a generated downstream prompt already has its parser request."""

    folded = prompt_text.casefold()
    if not re.search(r"\b(?:return|report|allocate|provide|enter|reply|answer)\b", folded):
        return False
    if expected_format == "single_integer_1_or_2":
        return bool(
            re.search(r"\binteger\b", folded)
            and SINGLE_INTEGER_CHOICE_PATTERN.search(folded)
        )
    if expected_format == "single_integer_0_to_100":
        return bool(
            re.search(r"\binteger\b", folded)
            and re.search(r"\b0\s*(?:to|-|through)\s*100\b|\bbetween\s+0\s+and\s+100\b", folded)
        )
    if expected_format == "single_integer_allocation_0_to_100":
        return bool(
            re.search(r"\binteger\b", folded)
            and re.search(r"\b0\s*(?:to|-|through)\s*100\b|\bbetween\s+0\s+and\s+100\b", folded)
            and re.search(r"\boption\s+a\b", folded)
        )
    if expected_format == "two_integers_sum_100":
        return bool(
            re.search(r"\btwo\s+integers?\b", folded)
            and re.search(r"\bseparate\s+lines?\b", folded)
            and re.search(r"\b100\b", folded)
        )
    if expected_format == "two_integers_on_separate_lines":
        return bool(
            re.search(r"\btwo\s+integers?\b", folded)
            and re.search(r"\bseparate\s+lines?\b", folded)
        )
    raise ValueError(f"Unsupported downstream response format={expected_format!r}.")


def _registered_neutral_calibration_instruction(plan: Mapping[str, Any]) -> str | None:
    """Return a mechanical neutral-payoff suffix when a plan registers one.

    Calibration contracts sometimes contain a fixed nuisance-only payoff that
    must be present in every item.  The model may produce the substantive item
    while omitting one of those fixed clauses, so the runtime may append this
    bounded, plan-derived sentence.  It is intentionally limited to the
    registered payoff fields and does not invent construct-specific content.
    """

    contract = plan.get("calibration_factor_schedule")
    if not isinstance(contract, Mapping):
        return None
    payoff = contract.get("neutral_payoff")
    if not isinstance(payoff, Mapping):
        return None
    sure_units = payoff.get("sure_outcome_units")
    risky_high_units = payoff.get("risky_high_outcome_units")
    risky_low_units = payoff.get("risky_low_outcome_units")
    probability = str(payoff.get("probability", "")).strip().casefold()
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (sure_units, risky_high_units, risky_low_units)
    ):
        return None
    probability_phrase = {
        "even": "probability one-half",
        "one-half": "probability one-half",
        "1/2": "probability one-half",
        "50%": "probability 50%",
    }.get(probability)
    if probability_phrase is None:
        return None
    return (
        "For this neutral calibration item, the sure option produces exactly "
        f"{sure_units} neutral outcome units with certainty. The risky option produces exactly "
        f"{risky_high_units} neutral outcome units with {probability_phrase} and "
        f"{risky_low_units} neutral outcome units otherwise. These abstract outcome units have no external meaning."
    )


def _calibration_instruction_is_present(prompt_text: str, instruction: str) -> bool:
    """Check whether a generated item already contains the registered suffix."""

    return instruction.casefold() in prompt_text.casefold()


def _complete_downstream_prompt_text(
    prompt_text: str,
    *,
    task: Mapping[str, Any],
    expected_format: str,
    prompt_role: str,
    plan: Mapping[str, Any] | None = None,
) -> tuple[str, bool]:
    """Append only missing mechanical instructions to a downstream item."""

    completed = prompt_text.rstrip()
    if prompt_role == "calibration" and plan is not None:
        calibration_instruction = _registered_neutral_calibration_instruction(plan)
        if (
            calibration_instruction is not None
            and not _calibration_instruction_is_present(completed, calibration_instruction)
        ):
            # Put the canonical payoff first so validators that inspect the
            # first sure/risky-option section cannot be shadowed by an
            # incomplete model-generated paraphrase later in the item.
            completed = f"{calibration_instruction}\n\n{completed}"
    response_instruction_completed = False
    if prompt_role != "probe" and not _response_instruction_is_present(completed, expected_format):
        instruction = _registered_response_instruction(task, expected_format)
        completed = f"{completed}\n\n{instruction}"
        response_instruction_completed = True
    if len(completed) > MAX_GENERATED_PROMPT_CHARS:
        raise ValueError("Completed downstream prompt exceeds the registered prompt length limit.")
    return completed, response_instruction_completed


def _validate_metadata_schema(
    value: Any,
    *,
    schema: Mapping[str, Any],
    job: ConstructGenerationJob,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{job.job_id} {context} must be an object.")
    metadata = dict(value)
    properties = dict(schema["properties"])
    _validate_object_keys(
        metadata,
        allowed=set(properties),
        required=set(schema["required"]),
        context=f"{job.job_id} {context}",
    )
    for field_name, field_schema in properties.items():
        field_value = metadata[field_name]
        field_type = field_schema["type"]
        valid_type = {
            "string": isinstance(field_value, str),
            "integer": isinstance(field_value, int) and not isinstance(field_value, bool),
            "number": isinstance(field_value, (int, float)) and not isinstance(field_value, bool),
            "boolean": isinstance(field_value, bool),
        }[field_type]
        if not valid_type:
            raise ValueError(f"{job.job_id} {context}.{field_name} must have type={field_type}.")
        if "enum" in field_schema and field_value not in field_schema["enum"]:
            raise ValueError(f"{job.job_id} {context}.{field_name} is outside its registered enum.")
        if "minimum" in field_schema and field_value < field_schema["minimum"]:
            raise ValueError(f"{job.job_id} {context}.{field_name} is below its registered minimum.")
        if "maximum" in field_schema and field_value > field_schema["maximum"]:
            raise ValueError(f"{job.job_id} {context}.{field_name} is above its registered maximum.")
    return metadata


def _validate_task_metadata(value: Any, *, spec: ConstructSpec, job: ConstructGenerationJob) -> dict[str, Any]:
    task = _task_for_prompt_role(spec, job.prompt_role)
    schema = dict(task["item_metadata_schema"])
    return _validate_metadata_schema(
        value,
        schema=schema,
        job=job,
        context="task_metadata",
    )


def _validate_paired_task_metadata(value: Any, *, spec: ConstructSpec, job: ConstructGenerationJob) -> dict[str, Any]:
    schema = _paired_metadata_schema(spec)
    if schema is None:
        if value is not None:
            raise ValueError(f"{job.job_id} paired task_metadata is not registered for this construct.")
        return {}
    return _validate_metadata_schema(
        value,
        schema=schema,
        job=job,
        context="paired task_metadata",
    )


def _parse_response(
    response: Mapping[str, Any],
    *,
    spec: ConstructSpec,
    plan: Mapping[str, Any],
    job: ConstructGenerationJob,
) -> list[dict[str, Any]]:
    data = _parse_json_response(response)
    expected_domains = _assigned_content_domains(plan, job)
    if job.mode == "paired":
        pairs = data.get("pairs")
        _validate_object_keys(
            data,
            allowed={"pairs"},
            required={"pairs"},
            context=f"{job.job_id} response",
        )
        if not isinstance(pairs, list) or len(pairs) != job.count:
            raise ValueError(f"{job.job_id} must return exactly {job.count} pairs.")
        expected_conditions = set(job.cell["condition_ids"])
        paired_metadata_schema = _paired_metadata_schema(spec)
        seen_pair_ids: set[str] = set()
        parsed: list[dict[str, Any]] = []
        for pair in pairs:
            if not isinstance(pair, Mapping):
                raise ValueError(f"{job.job_id} contains a non-object pair.")
            _validate_object_keys(
                pair,
                allowed={"pair_id", "content_domain", "prompts", "notes"},
                required={"pair_id", "content_domain", "prompts", "notes"},
                context=f"{job.job_id} pair",
            )
            raw_pair_id = pair.get("pair_id")
            if not isinstance(raw_pair_id, str) or not raw_pair_id.strip():
                raise ValueError(f"{job.job_id} returned a pair without pair_id.")
            pair_id = _slug(raw_pair_id)
            if pair_id in seen_pair_ids:
                raise ValueError(f"{job.job_id} returned duplicate pair_id={pair_id!r}.")
            seen_pair_ids.add(pair_id)
            content_domain = pair.get("content_domain")
            if content_domain != expected_domains[len(parsed)]:
                raise ValueError(
                    f"{job.job_id} pair {pair_id} returned content_domain={content_domain!r}; "
                    f"expected {expected_domains[len(parsed)]!r}."
                )
            if not isinstance(pair["notes"], str):
                raise ValueError(f"{job.job_id} pair {pair_id} is missing string notes.")
            if len(pair["notes"]) > MAX_GENERATED_NOTES_CHARS:
                raise ValueError(
                    f"{job.job_id} pair {pair_id} notes exceed {MAX_GENERATED_NOTES_CHARS} characters."
                )
            prompts = pair.get("prompts")
            if not isinstance(prompts, list) or len(prompts) != len(expected_conditions):
                raise ValueError(f"{job.job_id} pair {pair_id} has the wrong number of condition prompts.")
            seen_conditions: set[str] = set()
            prompt_rows: list[dict[str, Any]] = []
            for prompt in prompts:
                if not isinstance(prompt, Mapping):
                    raise ValueError(f"{job.job_id} pair {pair_id} contains a non-object prompt.")
                allowed_prompt_keys = {"condition_id", "prompt_text"}
                required_prompt_keys = {"condition_id", "prompt_text"}
                if paired_metadata_schema is not None:
                    allowed_prompt_keys.add("task_metadata")
                    required_prompt_keys.add("task_metadata")
                _validate_object_keys(
                    prompt,
                    allowed=allowed_prompt_keys,
                    required=required_prompt_keys,
                    context=f"{job.job_id} pair {pair_id} prompt",
                )
                condition_id = _slug(prompt.get("condition_id"))
                if condition_id not in expected_conditions or condition_id in seen_conditions:
                    raise ValueError(f"{job.job_id} pair {pair_id} has invalid or duplicate condition_id.")
                seen_conditions.add(condition_id)
                prompt_row: dict[str, Any] = {
                    "condition_id": condition_id,
                    "prompt_text": _validate_text(
                        prompt.get("prompt_text"),
                        plan=plan,
                        job=job,
                        probe_prompt_template=spec.probe_prompt_template,
                    ),
                }
                if paired_metadata_schema is not None:
                    prompt_row["task_metadata"] = _validate_paired_task_metadata(
                        prompt.get("task_metadata"),
                        spec=spec,
                        job=job,
                    )
                prompt_rows.append(prompt_row)
            if seen_conditions != expected_conditions:
                raise ValueError(f"{job.job_id} pair {pair_id} is missing a condition prompt.")
            parsed.append(
                {
                    "pair_id": pair_id,
                    "content_domain": content_domain,
                    "prompts": prompt_rows,
                    "notes": pair["notes"],
                }
            )
        return parsed

    prompts = data.get("prompts")
    _validate_object_keys(
        data,
        allowed={"prompts"},
        required={"prompts"},
        context=f"{job.job_id} response",
    )
    if not isinstance(prompts, list) or len(prompts) != job.count:
        raise ValueError(f"{job.job_id} must return exactly {job.count} prompts.")
    seen_variants: set[str] = set()
    parsed = []
    for prompt in prompts:
        if not isinstance(prompt, Mapping):
            raise ValueError(f"{job.job_id} contains a non-object prompt.")
        _validate_object_keys(
            prompt,
            allowed={"variant_id", "content_domain", "task_metadata", "prompt_text", "notes"},
            required={"variant_id", "content_domain", "task_metadata", "prompt_text", "notes"},
            context=f"{job.job_id} prompt",
        )
        raw_variant_id = prompt.get("variant_id")
        if not isinstance(raw_variant_id, str) or not raw_variant_id.strip():
            raise ValueError(f"{job.job_id} returned a prompt without variant_id.")
        variant_id = _slug(raw_variant_id)
        if variant_id in seen_variants:
            raise ValueError(f"{job.job_id} returned duplicate variant_id={variant_id!r}.")
        seen_variants.add(variant_id)
        content_domain = prompt.get("content_domain")
        if content_domain != expected_domains[len(parsed)]:
            raise ValueError(
                f"{job.job_id} prompt {variant_id} returned content_domain={content_domain!r}; "
                f"expected {expected_domains[len(parsed)]!r}."
            )
        if not isinstance(prompt["notes"], str):
            raise ValueError(f"{job.job_id} prompt {variant_id} is missing string notes.")
        if len(prompt["notes"]) > MAX_GENERATED_NOTES_CHARS:
            raise ValueError(
                f"{job.job_id} prompt {variant_id} notes exceed {MAX_GENERATED_NOTES_CHARS} characters."
            )
        parsed.append(
            {
                "variant_id": variant_id,
                "content_domain": content_domain,
                "task_metadata": _validate_task_metadata(prompt["task_metadata"], spec=spec, job=job),
                "prompt_text": _validate_text(
                    prompt.get("prompt_text"),
                    plan=plan,
                    job=job,
                    probe_prompt_template=spec.probe_prompt_template,
                ),
                "notes": prompt["notes"],
            }
        )
    return parsed


def _record_metadata(
    spec: ConstructSpec,
    plan: Mapping[str, Any],
    job: ConstructGenerationJob,
    *,
    notes: str,
    variant_id: str,
    content_domain: str,
    generation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    transport = dict(generation_metadata or {})
    provider = str(transport.get("provider", "openai")).strip().lower() or "openai"
    metadata = {
        "source": f"{provider}_generated",
        "generation_provider": provider,
        "source_model": job.model_id,
        "source_model_alias": job.model_alias,
        "generation_plan_id": plan["plan_id"],
        "generation_plan_sha256": _plan_hash(plan),
        "generation_job_id": job.job_id,
        "generation_batch_id": job.job_id,
        "generation_cell_id": job.cell_id,
        "generation_seed": job.seed,
        "generation_temperature": job.temperature,
        "content_pool": job.content_pool,
        "content_domain": content_domain,
        "family": spec.family,
        "wave": plan.get("wave"),
        "variant_id": variant_id,
        "notes": notes,
        "expected_readout_positive_condition": spec.positive_condition_id,
        "expected_readout_negative_condition": spec.negative_condition_id,
    }
    for key in (
        "response_id",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "actual_cost_usd",
        "reasoning_effort",
        "seed_supported",
    ):
        if key in transport:
            metadata[f"generation_{key}"] = transport[key]
    return metadata


def _records_from_response(
    response: Mapping[str, Any],
    *,
    spec: ConstructSpec,
    plan: Mapping[str, Any],
    job: ConstructGenerationJob,
) -> list[PromptRecord]:
    parsed = _parse_response(response, spec=spec, plan=plan, job=job)
    generation_metadata = response.get("_generation_metadata")
    if generation_metadata is not None and not isinstance(generation_metadata, Mapping):
        raise ValueError(f"{job.job_id} response _generation_metadata must be an object.")
    task = _task_for_prompt_role(spec, job.prompt_role)
    records: list[PromptRecord] = []
    if job.mode == "paired":
        paired_metadata_schema = _paired_metadata_schema(spec)
        for pair_index, pair in enumerate(parsed):
            pair_id = f"{job.job_id}__{pair['pair_id']}"
            for prompt in pair["prompts"]:
                condition_id = prompt["condition_id"]
                task_metadata = dict(prompt.get("task_metadata", {}))
                expected_categories = _expected_category_assignments(job, pair_index)
                for field, expected_value in expected_categories.items():
                    actual_value = task_metadata.get(field)
                    if actual_value != expected_value:
                        raise ValueError(
                            f"{job.job_id} pair {pair['pair_id']} condition {condition_id} has "
                            f"{field}={actual_value!r}; expected the pre-registered category "
                            f"{expected_value!r}."
                        )
                metadata = _record_metadata(
                    spec,
                    plan,
                    job,
                    notes=str(pair["notes"]),
                    variant_id=condition_id,
                    content_domain=pair["content_domain"],
                    generation_metadata=generation_metadata,
                )
                prompt_text, wrapper_normalized = normalize_probe_prompt_wrapper(
                    prompt["prompt_text"],
                    probe_prompt_template=spec.probe_prompt_template,
                ) if job.prompt_role == "probe" else (prompt["prompt_text"], False)
                if job.prompt_role == "probe":
                    _require_registered_probe_wrapper(
                        prompt_text,
                        probe_prompt_template=spec.probe_prompt_template,
                        job=job,
                    )
                if wrapper_normalized:
                    metadata["probe_wrapper_normalization_version"] = PROBE_WRAPPER_NORMALIZATION_VERSION
                    metadata["probe_wrapper_normalization_applied"] = True
                if paired_metadata_schema is not None:
                    metadata["task_metadata"] = task_metadata
                    metadata.update(task_metadata)
                records.append(
                    PromptRecord(
                        prompt_id=f"{pair_id}__{condition_id}",
                        construct_id=spec.construct_id,
                        split=job.split,
                        prompt_role=job.prompt_role,
                        prompt_text=prompt_text,
                        condition_id=condition_id,
                        pair_id=pair_id,
                        pair_role=condition_id,
                        prompt_family=job.prompt_family,
                        metadata=metadata,
                    )
                )
        return records

    task_id = str(job.cell.get("task_id", task["task_id"]))
    parser_id = str(job.cell.get("parser_id", spec.parsing_rules["parser_id"]))
    expected_format = str(job.cell.get("expected_output_format", task["response_format"]))
    condition_id = str(job.cell.get("condition_id", "neutral"))
    for prompt in parsed:
        prompt_index = len(records)
        calibration_instruction = (
            _registered_neutral_calibration_instruction(plan)
            if job.prompt_role == "calibration"
            else None
        )
        calibration_instruction_completed = (
            calibration_instruction is not None
            and not _calibration_instruction_is_present(prompt["prompt_text"], calibration_instruction)
        )
        prompt_text, response_instruction_completed = _complete_downstream_prompt_text(
            prompt["prompt_text"],
            task=task,
            expected_format=expected_format,
            prompt_role=job.prompt_role,
            plan=plan,
        )
        expected_categories = _expected_category_assignments(job, prompt_index)
        for field, expected_value in expected_categories.items():
            actual_value = prompt["task_metadata"].get(field)
            if actual_value != expected_value:
                raise ValueError(
                    f"{job.job_id} prompt {prompt['variant_id']} has {field}={actual_value!r}; "
                    f"expected the pre-registered category {expected_value!r}."
                )
        metadata = _record_metadata(
            spec,
            plan,
            job,
            notes=prompt["notes"],
            variant_id=prompt["variant_id"],
            content_domain=prompt["content_domain"],
            generation_metadata=generation_metadata,
        )
        if job.prompt_role == "probe":
            prompt_text, wrapper_normalized = normalize_probe_prompt_wrapper(
                prompt_text,
                probe_prompt_template=spec.probe_prompt_template,
            )
            _require_registered_probe_wrapper(
                prompt_text,
                probe_prompt_template=spec.probe_prompt_template,
                job=job,
            )
        else:
            wrapper_normalized = False
        if wrapper_normalized:
            metadata["probe_wrapper_normalization_version"] = PROBE_WRAPPER_NORMALIZATION_VERSION
            metadata["probe_wrapper_normalization_applied"] = True
        metadata["task_metadata"] = dict(prompt["task_metadata"])
        metadata.update(prompt["task_metadata"])
        metadata["response_instruction_completion"] = (
            "appended_registered_task_instruction"
            if response_instruction_completed
            else "model_supplied"
        )
        metadata["calibration_instruction_completion"] = (
            "appended_registered_neutral_payoff"
            if calibration_instruction_completed
            else "model_supplied_or_not_required"
        )
        records.append(
            PromptRecord(
                prompt_id=f"{job.job_id}__{prompt['variant_id']}",
                construct_id=spec.construct_id,
                split=job.split,
                prompt_role=job.prompt_role,
                prompt_text=prompt_text,
                condition_id=condition_id,
                prompt_family=job.prompt_family,
                task_id=task_id,
                expected_output_format=expected_format,
                parser_id=parser_id,
                metadata=metadata,
            )
        )
    return records


def _chunk_jobs(job: ConstructGenerationJob, max_items_per_request: int) -> Iterable[ConstructGenerationJob]:
    if max_items_per_request <= 0 or job.count <= max_items_per_request:
        yield job
        return
    total = math.ceil(job.count / max_items_per_request)
    for index in range(total):
        start = index * max_items_per_request
        yield ConstructGenerationJob(
            plan_id=job.plan_id,
            construct_id=job.construct_id,
            model_alias=job.model_alias,
            model_id=job.model_id,
            cell=job.cell,
            count=min(max_items_per_request, job.count - start),
            seed=job.seed + index,
            temperature=job.temperature,
            model_index=job.model_index,
            item_offset=job.item_offset + start,
            request_index=index + 1,
            request_total=total,
        )


def _validate_paired_job_schedule(job: ConstructGenerationJob) -> None:
    """Require equal registered-position counts in each full paired request."""

    schedule = job.cell.get("paired_metadata_schedule")
    if not isinstance(schedule, Mapping):
        return
    request_size = int(schedule["request_size"])
    positions = list(schedule["positions"])
    logical_count = int(job.cell.get("count_per_model", 0))
    if logical_count >= request_size and job.count != 1 and job.count % len(positions):
        raise ValueError(
            f"{job.job_id} must be generated in review-sized one-pair requests or "
            f"full balanced requests whose size is a positive multiple of {len(positions)} "
            f"(received {job.count})."
        )
    if job.count == 1 or logical_count < request_size:
        return
    expected = [
        _paired_metadata_assignment(job, index)[str(schedule["field"])]
        for index in range(job.count)
    ]
    repeats = job.count // len(positions)
    expected_counts = {position: repeats for position in positions}
    observed_counts = {position: expected.count(position) for position in positions}
    if observed_counts != expected_counts:
        raise ValueError(
            f"{job.job_id} schedule does not contain exactly the registered balanced positions: "
            f"observed={observed_counts}, expected={expected_counts}."
        )


def _request_with_retries(
    request_fn: RequestFn,
    job: ConstructGenerationJob,
    messages: list[dict[str, str]],
    options: dict[str, Any],
) -> dict[str, Any]:
    retries = int(options.get("retries", 0))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return request_fn(job.model_id, messages, options)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
    assert last_error is not None
    raise last_error


def _append_semantic_retry_reason(
    messages: list[dict[str, str]],
    *,
    attempt: int,
    rejection_reason: str,
) -> list[dict[str, str]]:
    """Add an exact, bounded corrective instruction to a semantic retry."""

    if not messages:
        raise ValueError("Cannot construct a semantic retry without generation messages.")
    retry_messages = [dict(message) for message in messages]
    last = retry_messages[-1]
    content = last.get("content")
    if not isinstance(content, str):
        raise ValueError("Cannot construct a semantic retry without a textual user message.")
    last["content"] = (
        content
        + "\n\nCORRECTIVE RETRY "
        + str(attempt)
        + ": The previous response was rejected by the downstream contract. "
        + "Fix only the rejection below and return the complete requested JSON. "
        + "Exact rejection reason: "
        + rejection_reason
    )
    return retry_messages


def iter_generation_request_jobs(
    plan: Mapping[str, Any],
    *,
    model_aliases: set[str] | None = None,
    count_per_model_override: int | None = None,
    splits: set[str] | None = None,
) -> Iterable[ConstructGenerationJob]:
    """Yield the stable request-level jobs used by ``generate_prompt_records``.

    ``iter_generation_jobs`` describes logical cells.  A plan may split one
    logical cell into several API requests through ``max_items_per_request``;
    recovery checkpoints need the latter request-level identity, including its
    deterministic ``part_N`` suffix.
    """

    max_per_request = int(plan.get("generation", {}).get("max_items_per_request", 0) or 0)
    for parent_job in iter_generation_jobs(
        plan,
        model_aliases=model_aliases,
        count_per_model_override=count_per_model_override,
        splits=splits,
    ):
        yield from _chunk_jobs(parent_job, max_per_request)


def _incomplete_response_reason(response: Mapping[str, Any]) -> str | None:
    """Return a provider-reported incomplete reason, if one is available."""

    metadata = response.get("_generation_metadata")
    if not isinstance(metadata, Mapping):
        return None
    reason = metadata.get("incomplete_reason") or metadata.get("incomplete_details")
    if isinstance(reason, Mapping):
        reason = reason.get("reason") or reason.get("code")
    status = metadata.get("status") or metadata.get("finish_reason")
    incomplete = metadata.get("incomplete")
    if incomplete is True:
        return str(reason or status or "provider marked response incomplete")
    if isinstance(reason, str) and reason.strip():
        normalized = reason.strip().lower()
        if normalized in {"max_output_tokens", "length", "incomplete", "cancelled", "failed"}:
            return reason.strip()
    if isinstance(status, str) and status.strip().lower() in {"incomplete", "cancelled", "failed"}:
        return str(reason or status).strip()
    return None


def generate_prompt_records(
    plan: Mapping[str, Any],
    spec: ConstructSpec,
    *,
    api_key: str,
    request_fn: RequestFn = call_openai_responses,
    workers: int = 1,
    model_aliases: set[str] | None = None,
    count_per_model_override: int | None = None,
    limit_jobs: int | None = None,
    splits: set[str] | None = None,
    transport_options: Mapping[str, Any] | None = None,
    completed_job_records: Mapping[str, tuple[PromptRecord, ...]] | None = None,
    completed_job_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    on_job_complete: JobCompletionFn | None = None,
    before_job_request: JobStartFn | None = None,
    semantic_retry_limit: int = 0,
    semantic_attempt_history: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    on_job_attempt: JobAttemptFn | None = None,
    job_records_validator: JobRecordsValidatorFn | None = None,
) -> GenerationResult:
    """Generate and validate a complete or explicitly limited inventory."""

    if isinstance(semantic_retry_limit, bool) or semantic_retry_limit < 0:
        raise ValueError("semantic_retry_limit must be a non-negative integer.")
    if isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer.")

    jobs = tuple(
        iter_generation_jobs(
            plan,
            model_aliases=model_aliases,
            count_per_model_override=count_per_model_override,
            limit_jobs=limit_jobs,
            splits=splits,
        )
    )
    selected_model_aliases = _planned_model_aliases(plan) if model_aliases is None else set(model_aliases)
    complete = (
        limit_jobs is None
        and splits is None
        and count_per_model_override is None
        and selected_model_aliases == _planned_model_aliases(plan)
    )
    generation = dict(plan.get("generation", {}))
    max_per_request = int(generation.get("max_items_per_request", 0) or 0)
    records: list[PromptRecord] = []
    request_metadata: list[dict[str, Any]] = []
    cached_records = dict(completed_job_records or {})
    cached_metadata = dict(completed_job_metadata or {})
    prior_attempts = {
        str(job_id): [dict(item) for item in history]
        for job_id, history in dict(semantic_attempt_history or {}).items()
    }
    request_jobs = tuple(
        job
        for parent_job in jobs
        for job in _chunk_jobs(parent_job, max_per_request)
    )
    cached_results: dict[str, tuple[tuple[PromptRecord, ...], tuple[dict[str, Any], ...]]] = {}
    pending_jobs: list[ConstructGenerationJob] = []
    for job in request_jobs:
        _validate_paired_job_schedule(job)
        if job.job_id in cached_records:
            recovered = tuple(cached_records[job.job_id])
            validate_prompt_records(
                recovered,
                {spec.construct_id: spec},
                require_all_splits=False,
            )
            metadata = cached_metadata.get(job.job_id)
            cached_results[job.job_id] = (
                recovered,
                (dict(metadata),) if metadata is not None else (),
            )
        else:
            pending_jobs.append(job)

    def generate_request(
        job: ConstructGenerationJob,
    ) -> tuple[ConstructGenerationJob, tuple[PromptRecord, ...], tuple[dict[str, Any], ...]]:
        """Generate one request-level job.

        The transport and callbacks are deliberately scoped to one stable job.
        Results are gathered in request-plan order below, while API calls can
        complete out of order in parallel.
        """

        if before_job_request is not None:
            before_job_request(job)
        options = {
            **generation,
            **dict(transport_options or {}),
            "api_key": api_key,
            "seed": job.seed,
            "temperature": job.temperature,
            "generation_job_id": job.job_id,
            "response_schema": response_schema_for_job(job, spec),
        }
        messages = build_generation_messages(spec, plan, job)
        history = [dict(item) for item in prior_attempts.get(job.job_id, [])]
        if history and history[-1].get("status") == "rejected":
            if len(history) > semantic_retry_limit:
                reason = str(history[-1].get("rejection_reason") or "unknown semantic rejection")
                raise ValueError(
                    f"{job.job_id} exhausted semantic retry limit={semantic_retry_limit}; "
                    f"last rejection: {reason}"
                )
            rejection_reason = str(history[-1].get("rejection_reason") or "unknown semantic rejection")
        else:
            rejection_reason = None
        accepted = False
        attempt_metadata: list[dict[str, Any]] = []
        for semantic_attempt_index in range(len(history), semantic_retry_limit + 1):
            attempt_number = semantic_attempt_index + 1
            attempt_messages = messages
            if rejection_reason is not None:
                attempt_messages = _append_semantic_retry_reason(
                    messages,
                    attempt=attempt_number,
                    rejection_reason=rejection_reason,
                )
            response = _request_with_retries(request_fn, job, attempt_messages, options)
            raw_metadata = response.get("_generation_metadata")
            normalized_metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            normalized_metadata["semantic_attempt"] = attempt_number
            attempt_metadata.append(normalized_metadata)
            try:
                incomplete_reason = _incomplete_response_reason(response)
                if incomplete_reason is not None:
                    raise ValueError(
                        f"{job.job_id} response incomplete before prompt parsing "
                        f"(reason={incomplete_reason}). Increase max_output_tokens or regenerate the job."
                    )
                job_records = tuple(_records_from_response(response, spec=spec, plan=plan, job=job))
                # Validate each request before exposing it to a recovery callback.
                # A malformed response therefore cannot become a durable checkpoint.
                validate_prompt_records(
                    job_records,
                    {spec.construct_id: spec},
                    require_all_splits=False,
                )
                if job_records_validator is not None:
                    job_records_validator(job, job_records)
            except (ValueError, json.JSONDecodeError) as exc:
                rejection_reason = f"{type(exc).__name__}: {exc}"
                normalized_metadata["semantic_attempt_status"] = "rejected"
                normalized_metadata["semantic_rejection_reason"] = rejection_reason
                history.append({
                    "attempt": attempt_number,
                    "status": "rejected",
                    "rejection_reason": rejection_reason,
                    "response_metadata": dict(normalized_metadata),
                })
                if on_job_attempt is not None:
                    on_job_attempt(job, attempt_number, normalized_metadata, rejection_reason)
                if semantic_attempt_index >= semantic_retry_limit:
                    raise ValueError(
                        f"{job.job_id} failed after {attempt_number} semantic attempt(s); "
                        f"last rejection: {rejection_reason}"
                    ) from exc
                continue
            normalized_metadata["semantic_attempt_status"] = "accepted"
            history.append({
                "attempt": attempt_number,
                "status": "accepted",
                "rejection_reason": None,
                "response_metadata": dict(normalized_metadata),
            })
            if on_job_attempt is not None:
                on_job_attempt(job, attempt_number, normalized_metadata, None)
            if on_job_complete is not None:
                on_job_complete(job, job_records, normalized_metadata)
            accepted = True
            return job, job_records, tuple(attempt_metadata)
        if not accepted:
            raise ValueError(f"{job.job_id} did not produce an accepted response.")
        raise AssertionError(f"Unreachable generation state for {job.job_id}.")

    generated_results: dict[str, tuple[tuple[PromptRecord, ...], tuple[dict[str, Any], ...]]] = {}
    if pending_jobs:
        if workers == 1 or len(pending_jobs) == 1:
            generated = [generate_request(job) for job in pending_jobs]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending_jobs))) as executor:
                futures = [executor.submit(generate_request, job) for job in pending_jobs]
                # Resolve in submission order for deterministic failure reporting;
                # the requests themselves still execute concurrently.
                generated = [future.result() for future in futures]
        generated_results = {
            job.job_id: (job_records, metadata)
            for job, job_records, metadata in generated
        }

    # Keep the canonical inventory and usage metadata deterministic even when
    # requests finish in a different order on the wire.
    for job in request_jobs:
        job_result = cached_results.get(job.job_id) or generated_results.get(job.job_id)
        if job_result is None:
            raise ValueError(f"No result was produced for request job {job.job_id}.")
        job_records, metadata = job_result
        records.extend(job_records)
        request_metadata.extend(metadata)
    request_count = len(request_jobs)
    validate_prompt_records(
        records,
        {spec.construct_id: spec},
        require_all_splits=complete,
    )
    prompt_ids = [record.prompt_id for record in records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Generated prompt IDs are not globally unique.")
    return GenerationResult(tuple(records), jobs, request_count, complete, tuple(request_metadata))


def write_generation_result(result: GenerationResult, path: str | Path) -> int:
    """Write only the canonical records; callers must handle partial status."""

    return write_prompt_records(result.records, path)
