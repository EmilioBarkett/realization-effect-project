"""Generic construct-aware synthetic prompt generation.

This module owns the benchmark-facing contract. It reuses the OpenRouter
transport but does not reuse the realization-specific response fields or
validation rules from the legacy activation prompt generator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from activation_analysis.openrouter_prompt_generation import call_openrouter_chat_completion

from .prompts import PromptRecord, validate_prompt_records, write_prompt_records
from .schemas import ConstructSpec, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from .splits import SPLIT_PROMPT_ROLE


RequestFn = Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]]
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULT_ESTIMATED_INPUT_TOKENS_PER_REQUEST = 1400
DEFAULT_ESTIMATED_OUTPUT_TOKENS_PER_RECORD = 300


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


def _category_balance(value: Any, *, field_name: str, count: int, spec: ConstructSpec) -> dict[str, list[Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object mapping metadata fields to schedules.")
    properties = dict(spec.independent_behavior_task["item_metadata_schema"]["properties"])
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


def _expected_category_assignments(job: ConstructGenerationJob, index: int) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for field, schedule in dict(job.cell.get("category_balance", {})).items():
        if not schedule:
            continue
        assignments[str(field)] = schedule[(job.item_offset + index) % len(schedule)]
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
        return {
            "complete": self.complete,
            "construct_ids": sorted({record.construct_id for record in self.records}),
            "job_count": len(self.jobs),
            "request_count": self.request_count,
            "record_count": len(self.records),
            "records_by_split": dict(sorted(by_split.items())),
            "records_by_model": dict(sorted(by_model.items())),
            "records_by_condition": dict(sorted(by_condition.items())),
        }


def load_generation_plan(path: str | Path, spec: ConstructSpec) -> dict[str, Any]:
    """Load and validate one construct-specific generation plan."""

    plan_path = Path(path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{plan_path} is not valid JSON.") from exc
    if not isinstance(plan, dict):
        raise ValueError(f"{plan_path} must contain a JSON object.")
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
        if pool_role not in {"probe", "behavior", "steering", "calibration"}:
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
        )
        if mode == "paired":
            condition_ids = cell.get("condition_ids")
            if not isinstance(condition_ids, list) or set(condition_ids) != set(spec.condition_ids):
                raise ValueError(
                    f"cells[{index}].condition_ids must exactly match {list(spec.condition_ids)}."
                )
        else:
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
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
) -> dict[str, Any]:
    jobs = list(
        iter_generation_jobs(
            plan,
            model_aliases=model_aliases,
            count_per_model_override=count_per_model_override,
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
        "complete_plan": count_per_model_override is None and selected_model_aliases == planned_model_aliases,
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
                                    "pair_id": {"type": "string"},
                                    "content_domain": {"type": "string"},
                                    "prompts": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": ["condition_id", "prompt_text"],
                                            "properties": {
                                                "condition_id": {"type": "string"},
                                                "prompt_text": {"type": "string"},
                                            },
                                        },
                                    },
                                    "notes": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            },
        }
    if job.mode == "single":
        item_metadata_schema = dict(spec.independent_behavior_task["item_metadata_schema"])
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
                                    "variant_id": {"type": "string"},
                                    "content_domain": {"type": "string"},
                                    "task_metadata": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": list(item_metadata_schema["required"]),
                                        "properties": dict(item_metadata_schema["properties"]),
                                    },
                                    "prompt_text": {"type": "string"},
                                    "notes": {"type": "string"},
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
    task = spec.independent_behavior_task
    assigned_domains = _assigned_content_domains(plan, job)
    system = (
        "You generate controlled, theory-relevant prompts for a multi-construct "
        "representation benchmark. Return only JSON matching the requested schema. "
        "Do not include markdown or commentary. Do not put condition IDs, construct "
        "names, or benchmark labels in prompt_text unless forbidden terms explicitly "
        "allow a lexical control. Preserve independence between probe and downstream "
        "content pools."
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
        "count": job.count,
        "generation_mode": job.mode,
        "probe_template": spec.probe_prompt_template if job.prompt_role == "probe" else None,
        "independent_task": task if job.prompt_role != "probe" else None,
        "item_metadata_schema": task["item_metadata_schema"] if job.prompt_role != "probe" else None,
        "cell_instructions": cell.get("instructions", ""),
        "category_balance": dict(cell.get("category_balance", {})),
        "required_category_assignments": [
            _expected_category_assignments(job, index) for index in range(job.count)
        ],
        "design_rules": plan.get("design_rules", []),
        "task_composition": plan["task_composition"],
        "forbidden_terms": list(plan.get("forbidden_terms", [])) + list(cell.get("forbidden_terms", [])),
        "output_requirement": (
            "For paired mode return exactly count pairs, and each pair must contain exactly one prompt "
            "for each condition_id and the assigned content_domain at the corresponding index. For single "
            "mode return exactly count prompts with the assigned content_domain at the corresponding index and "
            "task_metadata that exactly follows item_metadata_schema. "
            "Every prompt_text must be a complete model input, not a summary."
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


def _validate_text(prompt_text: Any, *, plan: Mapping[str, Any], job: ConstructGenerationJob) -> str:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError(f"{job.job_id} returned an empty or non-string prompt_text.")
    text = prompt_text.strip()
    forbidden_terms = [
        str(term).strip().lower()
        for term in list(plan.get("forbidden_terms", [])) + list(job.cell.get("forbidden_terms", []))
        if str(term).strip()
    ]
    lowered = text.lower()
    forbidden_hits = [term for term in forbidden_terms if re.search(rf"\b{re.escape(term)}\b", lowered)]
    if forbidden_hits:
        raise ValueError(f"{job.job_id} prompt contains forbidden term(s): {sorted(set(forbidden_hits))}.")
    return text


def _validate_task_metadata(value: Any, *, spec: ConstructSpec, job: ConstructGenerationJob) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{job.job_id} task_metadata must be an object.")
    metadata = dict(value)
    schema = dict(spec.independent_behavior_task["item_metadata_schema"])
    properties = dict(schema["properties"])
    _validate_object_keys(
        metadata,
        allowed=set(properties),
        required=set(schema["required"]),
        context=f"{job.job_id} task_metadata",
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
            raise ValueError(f"{job.job_id} task_metadata.{field_name} must have type={field_type}.")
        if "enum" in field_schema and field_value not in field_schema["enum"]:
            raise ValueError(f"{job.job_id} task_metadata.{field_name} is outside its registered enum.")
        if "minimum" in field_schema and field_value < field_schema["minimum"]:
            raise ValueError(f"{job.job_id} task_metadata.{field_name} is below its registered minimum.")
        if "maximum" in field_schema and field_value > field_schema["maximum"]:
            raise ValueError(f"{job.job_id} task_metadata.{field_name} is above its registered maximum.")
    return metadata


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
            prompts = pair.get("prompts")
            if not isinstance(prompts, list) or len(prompts) != len(expected_conditions):
                raise ValueError(f"{job.job_id} pair {pair_id} has the wrong number of condition prompts.")
            seen_conditions: set[str] = set()
            prompt_rows: list[dict[str, str]] = []
            for prompt in prompts:
                if not isinstance(prompt, Mapping):
                    raise ValueError(f"{job.job_id} pair {pair_id} contains a non-object prompt.")
                _validate_object_keys(
                    prompt,
                    allowed={"condition_id", "prompt_text"},
                    required={"condition_id", "prompt_text"},
                    context=f"{job.job_id} pair {pair_id} prompt",
                )
                condition_id = _slug(prompt.get("condition_id"))
                if condition_id not in expected_conditions or condition_id in seen_conditions:
                    raise ValueError(f"{job.job_id} pair {pair_id} has invalid or duplicate condition_id.")
                seen_conditions.add(condition_id)
                prompt_rows.append(
                    {
                        "condition_id": condition_id,
                        "prompt_text": _validate_text(prompt.get("prompt_text"), plan=plan, job=job),
                    }
                )
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
        parsed.append(
            {
                "variant_id": variant_id,
                "content_domain": content_domain,
                "task_metadata": _validate_task_metadata(prompt["task_metadata"], spec=spec, job=job),
                "prompt_text": _validate_text(prompt.get("prompt_text"), plan=plan, job=job),
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
) -> dict[str, Any]:
    return {
        "source": "openrouter_generated",
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


def _records_from_response(
    response: Mapping[str, Any],
    *,
    spec: ConstructSpec,
    plan: Mapping[str, Any],
    job: ConstructGenerationJob,
) -> list[PromptRecord]:
    parsed = _parse_response(response, spec=spec, plan=plan, job=job)
    task = spec.independent_behavior_task
    records: list[PromptRecord] = []
    if job.mode == "paired":
        for pair in parsed:
            pair_id = f"{job.job_id}__{pair['pair_id']}"
            for prompt in pair["prompts"]:
                condition_id = prompt["condition_id"]
                records.append(
                    PromptRecord(
                        prompt_id=f"{pair_id}__{condition_id}",
                        construct_id=spec.construct_id,
                        split=job.split,
                        prompt_role=job.prompt_role,
                        prompt_text=prompt["prompt_text"],
                        condition_id=condition_id,
                        pair_id=pair_id,
                        pair_role=condition_id,
                        prompt_family=job.prompt_family,
                        metadata=_record_metadata(
                            spec,
                            plan,
                            job,
                            notes=str(pair["notes"]),
                            variant_id=condition_id,
                            content_domain=pair["content_domain"],
                        ),
                    )
                )
        return records

    task_id = str(job.cell.get("task_id", task["task_id"]))
    parser_id = str(job.cell.get("parser_id", spec.parsing_rules["parser_id"]))
    expected_format = str(job.cell.get("expected_output_format", task["response_format"]))
    condition_id = str(job.cell.get("condition_id", "neutral"))
    for prompt in parsed:
        prompt_index = len(records)
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
        )
        metadata["task_metadata"] = dict(prompt["task_metadata"])
        metadata.update(prompt["task_metadata"])
        records.append(
            PromptRecord(
                prompt_id=f"{job.job_id}__{prompt['variant_id']}",
                construct_id=spec.construct_id,
                split=job.split,
                prompt_role=job.prompt_role,
                prompt_text=prompt["prompt_text"],
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


def generate_prompt_records(
    plan: Mapping[str, Any],
    spec: ConstructSpec,
    *,
    api_key: str,
    request_fn: RequestFn = call_openrouter_chat_completion,
    model_aliases: set[str] | None = None,
    count_per_model_override: int | None = None,
    limit_jobs: int | None = None,
) -> GenerationResult:
    """Generate and validate a complete or explicitly limited inventory."""

    jobs = tuple(
        iter_generation_jobs(
            plan,
            model_aliases=model_aliases,
            count_per_model_override=count_per_model_override,
            limit_jobs=limit_jobs,
        )
    )
    selected_model_aliases = _planned_model_aliases(plan) if model_aliases is None else set(model_aliases)
    complete = (
        limit_jobs is None
        and count_per_model_override is None
        and selected_model_aliases == _planned_model_aliases(plan)
    )
    generation = dict(plan.get("generation", {}))
    max_per_request = int(generation.get("max_items_per_request", 0) or 0)
    records: list[PromptRecord] = []
    request_count = 0
    for parent_job in jobs:
        for job in _chunk_jobs(parent_job, max_per_request):
            request_count += 1
            options = {
                **generation,
                "api_key": api_key,
                "seed": job.seed,
                "temperature": job.temperature,
                "generation_job_id": job.job_id,
                "response_schema": response_schema_for_job(job, spec),
            }
            messages = build_generation_messages(spec, plan, job)
            response = _request_with_retries(request_fn, job, messages, options)
            records.extend(_records_from_response(response, spec=spec, plan=plan, job=job))
    if complete:
        validate_prompt_records(records, {spec.construct_id: spec})
    prompt_ids = [record.prompt_id for record in records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Generated prompt IDs are not globally unique.")
    return GenerationResult(tuple(records), jobs, request_count, complete)


def write_generation_result(result: GenerationResult, path: str | Path) -> int:
    """Write only the canonical records; callers must handle partial status."""

    return write_prompt_records(result.records, path)
