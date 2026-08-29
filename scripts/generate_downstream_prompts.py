#!/usr/bin/env python3
"""Generate and freeze the independent downstream prompt inventory.

The vector inventory has a deliberately separate orchestrator.  This module
does the same for ``behavior_eval``, ``steering_eval``, ``calibration``, and
the role-specific ``collateral_eval`` control:
plans remain the authority for counts and categorical schedules, the generic
construct generator owns response/schema validation, and this entry point owns
request-level checkpoints, spending guards, cross-pool audits, and immutable
manifests.

The command never asks the model to choose the design.  Every item receives
its categories from the registered plan before an API request is made.  A
review run is partial and non-confirmatory; a full run is only marked frozen
after its audit has passed and an approved review manifest has been supplied.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openai_prompt_generation import call_openai_responses  # noqa: E402
from construct_benchmark.behavioral_design import (  # noqa: E402
    behavioral_record_issues,
    registered_task_for_role,
    validate_behavioral_design,
)
from construct_benchmark.config import load_construct_spec  # noqa: E402
from construct_benchmark.generation import (  # noqa: E402
    RequestFn,
    SINGLE_INTEGER_CHOICE_PATTERN,
    downstream_prompt_text_issues,
    dry_run_summary,
    expected_task_metadata_assignments,
    generate_prompt_records,
    iter_generation_request_jobs,
    load_generation_plan,
)
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402
from construct_benchmark.prompts import (  # noqa: E402
    PromptRecord,
    load_prompt_records,
    validate_prompt_records,
)
from construct_benchmark.registry import load_construct_registry  # noqa: E402
try:  # direct CLI execution has ``scripts/`` on sys.path; tests use a package import
    from scripts.generate_all_vector_prompts import (  # type: ignore
        GenerationPaused,
        NewJobLimit,
        RuntimeBudget,
        RuntimeBudgetExceeded,
        _atomic_write_json,
        _atomic_write_records,
        _request_with_runtime_budget,
        _request_with_runtime_options,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the direct CLI path
    from generate_all_vector_prompts import (
        GenerationPaused,
        NewJobLimit,
        RuntimeBudget,
        RuntimeBudgetExceeded,
        _atomic_write_json,
        _atomic_write_records,
        _request_with_runtime_budget,
        _request_with_runtime_options,
    )


DOWNSTREAM_SPLITS = frozenset({"behavior_eval", "steering_eval", "calibration"})
COLLATERAL_SPLITS = frozenset({"collateral_eval"})
ALL_DOWNSTREAM_SPLITS = DOWNSTREAM_SPLITS | COLLATERAL_SPLITS
DOWNSTREAM_ROLES = frozenset({"behavior", "steering", "calibration", "collateral"})
VECTOR_SPLITS = frozenset({"direction_train", "direction_validation", "direction_heldout"})
DEFAULT_REGISTRY = _ROOT / "configs/construct_benchmark/construct_registry_v1.json"
DEFAULT_OUTPUT_DIR = _ROOT / "results/benchmark/downstream_prompts_v1"
DEFAULT_VECTOR_REFERENCE = _ROOT / "results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv"
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_OUTPUT_TOKENS = 30_000
MIN_FULL_BATCH_SIZE = 20
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_COST_SAFETY_MULTIPLIER = 1.25
DOWNSTREAM_SEMANTIC_RETRY_LIMIT = 5
DOWNSTREAM_MANIFEST_VERSION = "1"
DOWNSTREAM_AUDIT_VERSION = "2"
QUALITY_GATE_VERSION = "1"
DOWNSTREAM_SAFE_PROMPT_MAX_CHARS = 1_900

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "has", "have", "he",
    "her", "his", "how", "if", "in", "into", "is", "it", "its", "may", "more", "must", "no", "not",
    "of", "on", "one", "or", "our", "out", "over", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "two", "use", "was", "were", "what", "when", "where", "which", "who", "with", "would", "you",
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "none"


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _source_plan_for_provenance(plan_path: Path, spec: Any) -> dict[str, Any]:
    """Load the reviewed parent plan behind a versioned generation overlay.

    Preflight plans inherit a production plan and replace only counts, pools,
    and seeds.  Their effective hash must remain distinct, while the source
    hash must identify the reviewed parent plan so the quality gate can pin
    both execution pools to the same reviewed contract.
    """

    payload = _load_json_object(plan_path, label="generation plan")
    base_ref = payload.get("base_plan_path")
    if (
        str(payload.get("plan_id", "")).endswith("_preflight")
        and isinstance(base_ref, str)
        and base_ref.strip()
    ):
        base_path = (plan_path.parent / base_ref).resolve()
        return _source_plan_for_provenance(base_path, spec)
    return load_generation_plan(plan_path, spec)


def _with_runtime_identity(
    plan: Mapping[str, Any],
    *,
    model: str,
    batch_size: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Apply the runtime identity fields included in plan provenance hashes."""

    effective = copy.deepcopy(dict(plan))
    generation = effective.setdefault("generation", {})
    for model_payload in effective.get("models", []):
        model_payload["alias"] = "luna"
        model_payload["model"] = model
    generation["max_items_per_request"] = int(batch_size)
    generation["max_tokens"] = int(max_output_tokens)
    generation["max_output_tokens"] = int(max_output_tokens)
    return effective


def _effective_entries(
    registry_path: str | Path,
    *,
    waves: Iterable[int | str] | None,
    construct_ids: Iterable[str] | None,
    batch_size: int,
    max_output_tokens: int,
    model: str | None = None,
) -> tuple[Any, ...]:
    """Discover registry plans and apply only explicit runtime batching caps."""

    registry_path = Path(registry_path).resolve()
    registry = load_construct_registry(registry_path)
    raw_registry = _load_json_object(registry_path, label="construct registry")
    raw_entries = {
        str(item["construct_id"]): item
        for item in raw_registry.get("entries", [])
        if isinstance(item, Mapping) and isinstance(item.get("construct_id"), str)
    }
    values = [str(value).strip().lower() for value in waves] if waves is not None else ["all"]
    if not values or values == ["all"]:
        selected_waves = {entry.wave for entry in registry.entries}
    else:
        if "all" in values:
            raise ValueError("--waves accepts 'all' or wave numbers, not both.")
        try:
            selected_waves = {int(value) for value in values}
        except ValueError as exc:
            raise ValueError("waves must contain only 1, 2, 3, 4, or all.") from exc
        if not selected_waves.issubset({entry.wave for entry in registry.entries}):
            raise ValueError(f"waves must be drawn from {sorted({entry.wave for entry in registry.entries})}.")
    selected = [entry for entry in registry.entries if entry.wave in selected_waves]
    requested = {str(value).strip() for value in construct_ids or () if str(value).strip()}
    if requested:
        known = {entry.construct_id for entry in selected}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown or out-of-wave construct IDs: {sorted(unknown)}.")
        selected = [entry for entry in selected if entry.construct_id in requested]
    if not selected:
        raise ValueError("No constructs matched the requested waves/construct IDs.")
    result: list[Any] = []
    for registry_entry in selected:
        raw_entry = raw_entries.get(registry_entry.construct_id, {})
        reference = next(
            (
                raw_entry.get(key)
                for key in ("downstream_plan_path", "plan_path", "generation_plan_path", "generation_plan")
                if raw_entry.get(key)
            ),
            f"generation_plans/wave{registry_entry.wave}_{registry_entry.construct_id}_v1.json",
        )
        spec_path = (registry_path.parent / registry_entry.spec_path).resolve()
        plan_path = (registry_path.parent / str(reference)).resolve()
        spec = load_construct_spec(spec_path)
        source_plan = _source_plan_for_provenance(plan_path, spec)
        base_plan = load_generation_plan(plan_path, spec)
        downstream_overlay_path = raw_entry.get("downstream_plan_path")
        if downstream_overlay_path:
            overlay_path = (registry_path.parent / str(downstream_overlay_path)).resolve()
            overlay = _load_json_object(overlay_path, label="downstream plan overlay")
            if overlay.get("base_plan_id") != base_plan["plan_id"]:
                raise ValueError(
                    f"{overlay_path} base_plan_id does not match {base_plan['plan_id']!r}."
                )
            plan = load_generation_plan(
                plan_path,
                spec,
                overrides=overlay.get("overrides"),
            )
        else:
            plan = base_plan
        if int(plan.get("wave")) != registry_entry.wave:
            raise ValueError(f"{plan_path} wave does not match registry entry {registry_entry.construct_id!r}.")
        if str(plan.get("construct_id")) != registry_entry.construct_id:
            raise ValueError(f"{plan_path} construct_id does not match registry entry.")
        if set(spec.paired_splits) != VECTOR_SPLITS:
            raise ValueError(
                f"{registry_entry.construct_id} must retain exactly the three paired vector splits; "
                f"received {list(spec.paired_splits)}."
            )
        # The downstream workflow is Luna-only.  The checked-in generation
        # plans retain their historical source model for scientific provenance,
        # but every effective job, prompt metadata field, and manifest must
        # identify the actual Luna model used for this workflow.
        effective_model = str(model or DEFAULT_MODEL)
        effective = _with_runtime_identity(
            plan,
            model=effective_model,
            batch_size=batch_size,
            max_output_tokens=max_output_tokens,
        )
        validate_behavioral_design(spec, effective)
        result.append(
            replace(
                _PlanEntry(
                    construct_id=registry_entry.construct_id,
                    wave=registry_entry.wave,
                    spec_path=spec_path,
                    plan_path=plan_path,
                    spec=spec,
                    plan=effective,
                    source_plan_sha256=_canonical_sha256(source_plan),
                    spec_sha256=canonical_hash(spec.to_mapping()),
                    plan_sha256=_canonical_sha256(effective),
                ),
            )
        )
    return tuple(result)


from dataclasses import dataclass  # noqa: E402  (kept near local plan type)


@dataclass(frozen=True)
class _PlanEntry:
    construct_id: str
    wave: int
    spec_path: Path
    plan_path: Path
    spec: Any
    plan: dict[str, Any]
    source_plan_sha256: str
    spec_sha256: str
    plan_sha256: str


def _splits_for_entry(entry: _PlanEntry) -> frozenset[str]:
    """Return the downstream splits registered by this plan generation family."""

    design = dict(entry.spec.metadata or {}).get("behavioral_design", {})
    if str(design.get("repair_family", "")) == "waves2_4_repaired_v3":
        return ALL_DOWNSTREAM_SPLITS
    return DOWNSTREAM_SPLITS


def _mode_count(mode: str) -> int | None:
    if mode == "review":
        return 1
    if mode == "full":
        return None
    raise ValueError("mode must be review or full")


def _expected_summary(
    entry: _PlanEntry,
    *,
    mode: str,
    input_price: float | None,
    output_price: float | None,
    selected_splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    return dry_run_summary(
        entry.plan,
        count_per_model_override=_mode_count(mode),
        splits=set(selected_splits) if selected_splits is not None else set(_splits_for_entry(entry)),
        input_usd_per_million_tokens=input_price,
        output_usd_per_million_tokens=output_price,
    )


def _allocate_request_workers(
    pending: Iterable[Any],
    workers: int,
) -> dict[str, int]:
    """Allocate one global request budget across pending constructs.

    The downstream orchestrator has a construct-level pool and
    ``generate_prompt_records`` has a request-level pool.  Keeping this
    allocation in one place prevents nested pools from multiplying API
    concurrency.  Extra capacity is assigned in stable pending-order, so a
    single-construct recovery run receives the full worker budget while a
    multi-construct run remains globally bounded.
    """

    if isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    pending_entries = tuple(pending)
    if not pending_entries:
        return {}
    allocation = {str(entry.construct_id): 1 for entry in pending_entries}
    for index in range(max(0, workers - len(pending_entries))):
        construct_id = str(pending_entries[index % len(pending_entries)].construct_id)
        allocation[construct_id] += 1
    return allocation


def _cells_by_split(entry: _PlanEntry) -> dict[str, dict[str, Any]]:
    allowed_splits = _splits_for_entry(entry)
    return {
        str(cell["split"]): dict(cell)
        for cell in entry.plan["cells"]
        if str(cell["split"]) in allowed_splits
    }


def _role_task_contract(
    entry: _PlanEntry,
    cell: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str]:
    """Resolve task, parser, and output format for one registered role."""

    role = str(cell.get("prompt_role"))
    task = registered_task_for_role(entry.spec, role)
    design = dict(entry.spec.metadata or {}).get("behavioral_design", {})
    if role == "collateral":
        parser_id = str(
            design.get("collateral_parser_id")
            or cell.get("parser_id")
            or ""
        )
    else:
        parser_id = str(entry.spec.parsing_rules["parser_id"])
    response_format = str(task["response_format"])
    return task, parser_id, response_format


def _schedule_combinations(cell: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    raw: dict[str, Any] = {}
    for schedule_name in ("category_balance", "metadata_schedule"):
        schedule = cell.get(schedule_name, {})
        if not isinstance(schedule, Mapping):
            raise ValueError(f"Cell {cell.get('cell_id')} {schedule_name} must be an object.")
        overlap = set(raw) & set(schedule)
        if overlap:
            raise ValueError(f"Cell {cell.get('cell_id')} schedules field(s) twice: {sorted(overlap)}")
        raw.update({str(field): values for field, values in schedule.items()})
    if not isinstance(raw, Mapping) or not raw:
        return ()
    fields = list(raw)
    schedules = [list(raw[field]) for field in fields]
    if any(len(values) != len(schedules[0]) for values in schedules):
        raise ValueError(f"Cell {cell.get('cell_id')} has unequal category schedule lengths.")
    return tuple(tuple(values[index] for values in schedules) for index in range(len(schedules[0])))


def _parser_instruction_issues(record: PromptRecord) -> tuple[str, ...]:
    """Return missing pieces of the registered response instruction.

    A model can return valid JSON while still producing a downstream item that
    omits the request the scorer relies on.  This small contract check is kept
    independent of the model response parser so it also audits frozen CSVs.
    """

    text = record.prompt_text.casefold()
    stripped_text = record.prompt_text.rstrip()
    output_format = str(record.expected_output_format or "")
    issues: list[str] = []
    issues.extend(
        downstream_prompt_text_issues(
            record.prompt_text,
            expected_output_format=output_format,
        )
    )
    if len(record.prompt_text) > DOWNSTREAM_SAFE_PROMPT_MAX_CHARS:
        issues.append(
            f"prompt exceeds safe {DOWNSTREAM_SAFE_PROMPT_MAX_CHARS}-character headroom"
        )
    if not re.search(r"[.!?][\"')\]]*$", stripped_text):
        issues.append("prompt does not end with terminal punctuation")
    incomplete_phrase = re.search(
        r"\b(?:the|a|an|to|of|for|with|and|or|that|which|because|claim|from|than|as|on|in|at|by)\s*[.!?]?$",
        stripped_text.casefold(),
    )
    allocation_response_tail = output_format == "single_integer_allocation_0_to_100" and re.search(
        r"\b(?:points|allocation)\b.{0,160}\boption\s+a\s*[.!?]?$",
        stripped_text.casefold(),
    )
    if incomplete_phrase and not allocation_response_tail:
        issues.append("prompt appears truncated at an incomplete phrase")
    if not re.search(r"\b(return|report|output|provide|enter|reply|answer)\b", text):
        issues.append("missing response directive")
    if output_format == "two_integers_on_separate_lines":
        if not re.search(r"\btwo\s+integers?\b", text):
            issues.append("missing two-integer instruction")
        if not re.search(r"\bseparate\s+lines?\b", text):
            issues.append("missing separate-lines instruction")
        if record.parser_id == "two_integers_risk_choice_v1":
            if not re.search(r"(?:\bfirst\b.{0,180}\b(?:risky|risk)\s+option\b|\ballocate\b.{0,120}\b(?:risky|risk)\s+option\b)", text):
                issues.append("first integer is not mapped to risky-option allocation")
            if not re.search(r"\b(?:remainder|remaining)\b.{0,100}\bsure\s+option\b", text):
                issues.append("risky allocation does not specify remainder for sure option")
    elif output_format == "two_integers_sum_100":
        if not re.search(r"\btwo\s+integers?\b", text):
            issues.append("missing two-integer instruction")
        if not re.search(r"\bseparate\s+lines?\b", text):
            issues.append("missing separate-lines instruction")
        if not re.search(r"\b100\b", text):
            issues.append("missing sum-to-100 instruction")
    elif output_format == "single_integer_0_to_100":
        if not re.search(r"\binteger\b", text):
            issues.append("missing integer instruction")
        if not re.search(r"\b0\s*(?:to|-|through)\s*100\b|\bbetween\s+0\s+and\s+100\b", text):
            issues.append("missing 0-to-100 range instruction")
    elif output_format == "single_integer_allocation_0_to_100":
        if not re.search(r"\binteger\b", text):
            issues.append("missing integer instruction")
        if not re.search(r"\b0\s*(?:to|-|through)\s*100\b|\bbetween\s+0\s+and\s+100\b", text):
            issues.append("missing 0-to-100 range instruction")
        if not re.search(r"\boption\s+a\b", text):
            issues.append("missing option-A allocation mapping")
    elif output_format == "single_integer_1_or_2":
        if not re.search(r"\binteger\b", text):
            issues.append("missing integer instruction")
        if not SINGLE_INTEGER_CHOICE_PATTERN.search(text):
            issues.append("missing 1-or-2 instruction")
    else:
        issues.append(f"unsupported expected output format {output_format!r}")
    return tuple(issues)


def _calibration_contract(entry: _PlanEntry) -> dict[str, Any]:
    raw = entry.plan.get("calibration_factor_schedule")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{entry.construct_id} is missing calibration_factor_schedule.")
    nuisance_fields = raw.get("nuisance_fields", [])
    neutral_fields = raw.get("neutral_fields", {})
    forbidden_terms = raw.get("forbidden_terms", [])
    if not isinstance(nuisance_fields, list) or not all(isinstance(field, str) for field in nuisance_fields):
        raise ValueError(f"{entry.construct_id} calibration nuisance_fields must be a list of strings.")
    if not isinstance(neutral_fields, Mapping) or not all(isinstance(field, str) for field in neutral_fields):
        raise ValueError(f"{entry.construct_id} calibration neutral_fields must be an object.")
    if not isinstance(forbidden_terms, list) or not all(isinstance(term, str) and term.strip() for term in forbidden_terms):
        raise ValueError(f"{entry.construct_id} calibration forbidden_terms must be a non-empty-term list.")
    required_format = raw.get("required_response_format")
    if not isinstance(required_format, str) or not required_format:
        raise ValueError(f"{entry.construct_id} calibration required_response_format is missing.")
    return dict(raw)


def _calibration_cue_hits(entry: _PlanEntry, text: str) -> tuple[str, ...]:
    contract = _calibration_contract(entry)
    folded = text.casefold()
    hits = [
        str(term)
        for term in contract.get("forbidden_terms", [])
        if re.search(r"(?<![a-z0-9])" + re.escape(str(term).casefold()) + r"(?![a-z0-9])", folded)
    ]
    payoff = contract.get("neutral_payoff")
    if isinstance(payoff, Mapping):
        for term in payoff.get("forbidden_terms", []):
            if re.search(r"(?<![a-z0-9])" + re.escape(str(term).casefold()) + r"(?![a-z0-9])", folded):
                hits.append(str(term))
    return tuple(hits)


def _calibration_text_issues(entry: _PlanEntry, record: PromptRecord) -> tuple[str, ...]:
    """Check semantic requirements that cannot be represented by metadata alone."""

    if record.prompt_role != "calibration":
        return ()
    contract = _calibration_contract(entry)
    payoff = contract.get("neutral_payoff")
    if not isinstance(payoff, Mapping):
        return ()
    text = record.prompt_text.casefold()
    issues: list[str] = []
    sure_match = re.search(r"\bsure\s+option\b(?P<section>.{0,260})", text)
    sure_section = sure_match.group("section") if sure_match else ""
    if not re.search(r"\b10\b|\bten\b", sure_section) or not re.search(r"\b(?:certain|certainty)\b", sure_section):
        issues.append("neutral payoff is missing 10 outcome units with certainty for the sure option")
    risky_match = re.search(r"\brisky\s+option\b(?P<section>.{0,420})", text)
    risky_section = risky_match.group("section") if risky_match else ""
    if not re.search(r"\b20\b|\btwenty\b", risky_section):
        issues.append("neutral payoff is missing 20 outcome units for the risky option")
    if not re.search(r"\b0\b|\bzero\b", risky_section):
        issues.append("neutral payoff is missing 0 outcome units for the risky option")
    if not re.search(r"1\s*/\s*2|50\s*%|one[- ]half|equally\s+likely|equal(?:ly)?\s+probab", risky_section):
        issues.append("neutral payoff is missing equal-probability wording for the risky option")
    required_terms = payoff.get("required_terms", [])
    missing_terms = [
        str(term)
        for term in required_terms
        if not re.search(r"(?<![a-z0-9])" + re.escape(str(term).casefold()) + r"(?![a-z0-9])", text)
    ]
    if missing_terms:
        issues.append(f"neutral payoff is missing required terms: {missing_terms}")
    return tuple(issues)


def _downstream_pool_separation(entry: _PlanEntry) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return validated prompt-anchor rules for a plan's downstream pools."""

    raw = entry.plan.get("downstream_pool_separation")
    if not isinstance(raw, Mapping):
        return {}
    required = raw.get("required_prompt_anchors", {})
    forbidden = raw.get("forbidden_prompt_anchors", {})
    if not isinstance(required, Mapping) or not isinstance(forbidden, Mapping):
        raise ValueError(f"{entry.construct_id} downstream_pool_separation anchor maps must be objects.")
    pools = {str(cell["content_pool"]) for cell in _cells_by_split(entry).values()}
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for pool in pools:
        required_terms = required.get(pool)
        forbidden_terms = forbidden.get(pool)
        if (
            not isinstance(required_terms, list)
            or not required_terms
            or not all(isinstance(term, str) and term.strip() for term in required_terms)
        ):
            raise ValueError(f"{entry.construct_id} downstream pool {pool!r} lacks required prompt anchors.")
        if (
            not isinstance(forbidden_terms, list)
            or not all(isinstance(term, str) and term.strip() for term in forbidden_terms)
        ):
            raise ValueError(f"{entry.construct_id} downstream pool {pool!r} has invalid forbidden prompt anchors.")
        required_set = {str(term).casefold() for term in required_terms}
        forbidden_set = {str(term).casefold() for term in forbidden_terms}
        if required_set & forbidden_set:
            raise ValueError(f"{entry.construct_id} downstream pool {pool!r} both requires and forbids an anchor.")
        result[pool] = {
            "required": tuple(str(term) for term in required_terms),
            "forbidden": tuple(str(term) for term in forbidden_terms),
        }
    return result


def _downstream_pool_text_issues(entry: _PlanEntry, record: PromptRecord) -> tuple[str, ...]:
    rules = _downstream_pool_separation(entry)
    if not rules:
        return ()
    pool = str(record.metadata.get("content_pool", ""))
    rule = rules.get(pool)
    if rule is None:
        return (f"content pool {pool!r} has no downstream separation rules",)
    folded = record.prompt_text.casefold()

    def contains(term: str) -> bool:
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(term.casefold()) + r"(?![a-z0-9])", folded))

    issues: list[str] = []
    if not any(contains(term) for term in rule["required"]):
        issues.append(f"prompt lacks a required semantic anchor for pool {pool!r}")
    forbidden_hits = [term for term in rule["forbidden"] if contains(term)]
    if forbidden_hits:
        issues.append(f"prompt uses anchors reserved for another pool: {sorted(forbidden_hits)}")
    return tuple(issues)


def _validate_downstream_job_records(entry: _PlanEntry, job: Any, records: Iterable[PromptRecord]) -> None:
    """Fail a request before checkpointing when a downstream item is invalid."""

    materialized = tuple(records)
    validate_prompt_records(materialized, {entry.construct_id: entry.spec}, require_all_splits=False)
    cells = _cells_by_split(entry)
    cell = cells.get(str(job.split))
    if cell is None:
        raise ValueError(f"{job.job_id} is not a registered downstream cell.")
    for index, record in enumerate(materialized):
        if record.split != str(job.split) or record.prompt_role != str(cell["prompt_role"]):
            raise ValueError(f"{record.prompt_id} has a role or split inconsistent with its request job.")
        if record.prompt_family != str(cell["prompt_family"]):
            raise ValueError(f"{record.prompt_id} has an unexpected prompt family.")
        if record.metadata.get("content_pool") != cell["content_pool"]:
            raise ValueError(f"{record.prompt_id} has an unexpected content_pool.")
        if record.condition_id not in (None, "", "neutral"):
            raise ValueError(f"Downstream prompt {record.prompt_id} must have neutral condition_id.")
        expected_task, expected_parser, expected_format = _role_task_contract(entry, cell)
        if record.task_id != str(expected_task["task_id"]):
            raise ValueError(f"{record.prompt_id} has an incompatible task_id for its prompt role.")
        if record.parser_id != str(cell.get("parser_id", expected_parser)) or record.parser_id != expected_parser:
            raise ValueError(f"{record.prompt_id} has an incompatible parser_id.")
        if record.expected_output_format != str(cell.get("expected_output_format", expected_format)) or record.expected_output_format != expected_format:
            raise ValueError(f"{record.prompt_id} has an incompatible expected_output_format.")
        parser_issues = _parser_instruction_issues(record)
        if parser_issues:
            raise ValueError(f"{record.prompt_id} has an incomplete parser instruction: {'; '.join(parser_issues)}.")
        if record.prompt_role == "calibration":
            cue_hits = _calibration_cue_hits(entry, record.prompt_text)
            if cue_hits:
                raise ValueError(f"{record.prompt_id} contains calibration target-cue terms: {sorted(set(cue_hits))}.")
            text_issues = _calibration_text_issues(entry, record)
            if text_issues:
                raise ValueError(f"{record.prompt_id} violates its calibration text contract: {'; '.join(text_issues)}.")
        pool_issues = _downstream_pool_text_issues(entry, record)
        if pool_issues:
            raise ValueError(f"{record.prompt_id} violates downstream pool separation: {'; '.join(pool_issues)}.")
        behavioral_issues = behavioral_record_issues(entry.spec, entry.plan, record)
        if behavioral_issues:
            raise ValueError(
                f"{record.prompt_id} violates its registered behavioral repair contract: "
                + "; ".join(behavioral_issues)
                + "."
            )
        metadata = record.metadata.get("task_metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else record.metadata
        expected_assignments = expected_task_metadata_assignments(job, index)
        for field, expected_value in expected_assignments.items():
            if metadata.get(field) != expected_value:
                raise ValueError(
                    f"{record.prompt_id} has {field}={metadata.get(field)!r}; expected "
                    f"registered schedule value {expected_value!r}."
            )


_COLLATERAL_LABEL_EXCEPTION_MARKER = "COLLATERAL_FACT_ID_LABEL_EXCEPTION_V1"


def _request_fn_with_collateral_label_exception(
    request_fn: RequestFn,
    entry: _PlanEntry,
) -> RequestFn:
    """Resolve the generic no-label rule for an explicit opaque-card contract.

    The shared generation core quite properly bans benchmark labels by
    default.  Versioned collateral repairs are the one registered exception:
    an opaque ``Fact ID`` is required so the exact fact card can be audited.
    Keep this exception in the downstream adapter rather than weakening the
    shared probe-generation policy.
    """

    task = entry.spec.collateral_behavior_task
    requires_opaque_label = bool(
        task is not None
        and str(task.get("fact_bank_version", "v1")) != "v1"
        and task.get("label_contract_version")
    )
    if not requires_opaque_label:
        return request_fn

    def wrapped(model_id: str, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        adjusted = [dict(message) for message in messages]
        exception = (
            f" {_COLLATERAL_LABEL_EXCEPTION_MARKER}: This registered collateral task explicitly "
            "requires one opaque neutral factual-card label in the exact form 'Fact ID: <fact_id>'. "
            "This is an allowed task-specific exception to the generic benchmark-label ban; the "
            "identifier is not a construct name or condition label. Include the assigned label exactly "
            "once and do not replace it with a construct-specific name."
        )
        for index, message in enumerate(adjusted):
            if message.get("role") == "system":
                content = str(message.get("content", ""))
                if _COLLATERAL_LABEL_EXCEPTION_MARKER not in content:
                    adjusted[index] = {**message, "content": content + exception}
                break
        return request_fn(model_id, adjusted, options)

    return wrapped


def _validate_calibration_plan(entry: _PlanEntry) -> None:
    """Validate a plan-level neutral calibration contract before generation."""

    contract = _calibration_contract(entry)
    cells = _cells_by_split(entry)
    required_splits = _splits_for_entry(entry)
    for split in required_splits:
        cell = cells.get(split)
        if cell is None:
            raise ValueError(f"{entry.construct_id} is missing the {split} downstream cell.")
        task, expected_parser, expected_format = _role_task_contract(entry, cell)
        if str(cell.get("task_id")) != str(task["task_id"]):
            raise ValueError(f"{entry.construct_id} {split} task_id does not match its role task.")
        if str(cell.get("parser_id")) != expected_parser:
            raise ValueError(f"{entry.construct_id} {split} parser_id does not match its role parser.")
        if str(cell.get("expected_output_format")) != expected_format:
            raise ValueError(f"{entry.construct_id} {split} response format does not match its role task.")
    behavior_schedule_id = str(cells["behavior_eval"].get("factor_schedule", "behavior_factor_schedule"))
    steering_schedule_id = str(cells["steering_eval"].get("factor_schedule", "behavior_factor_schedule"))
    calibration_schedule_id = str(cells["calibration"].get("factor_schedule", ""))
    if behavior_schedule_id != steering_schedule_id:
        raise ValueError(f"{entry.construct_id} behavior_eval and steering_eval use different factor schedules.")
    if calibration_schedule_id != "calibration_factor_schedule":
        raise ValueError(
            f"{entry.construct_id} calibration must use the separate calibration_factor_schedule, "
            f"not {calibration_schedule_id!r}."
        )
    if str(contract.get("required_response_format")) != str(cells["calibration"].get("expected_output_format")):
        raise ValueError(f"{entry.construct_id} calibration response format does not match its contract.")
    calibration_categories = {
        **dict(cells["calibration"].get("category_balance", {})),
        **dict(cells["calibration"].get("metadata_schedule", {})),
    }
    for field, expected_value in dict(contract.get("neutral_fields", {})).items():
        values = calibration_categories.get(field)
        if not isinstance(values, list) or not values or any(value != expected_value for value in values):
            raise ValueError(
                f"{entry.construct_id} calibration neutral field {field!r} must be fixed at {expected_value!r}."
            )
    for field in contract.get("nuisance_fields", []):
        if field not in calibration_categories:
            raise ValueError(f"{entry.construct_id} calibration is missing nuisance field {field!r}.")
    # These two numeric contracts are deliberately exact: a midpoint source
    # record and LR=1 are neutral controls, not mixtures of target conditions.
    if "likelihood_ratio" in contract:
        likelihood_ratio = contract["likelihood_ratio"]
        if not isinstance(likelihood_ratio, Mapping) or not likelihood_ratio.get("exact"):
            raise ValueError(f"{entry.construct_id} calibration likelihood-ratio contract must be exact.")
        if likelihood_ratio.get("p_observation_given_hypothesis") != likelihood_ratio.get("p_observation_given_alternatives"):
            raise ValueError(f"{entry.construct_id} calibration likelihood ratio must equal 1.")
    if "source_record" in contract:
        source_record = contract["source_record"]
        if (
            not isinstance(source_record, Mapping)
            or not source_record.get("exact")
            or source_record.get("confirmed_reports") != 3
            or source_record.get("total_reports") != 5
        ):
            raise ValueError(f"{entry.construct_id} calibration source record must be an exact 3-of-5 midpoint.")
    if "neutral_payoff" in contract:
        payoff = contract["neutral_payoff"]
        required = {
            "sure_outcome_units",
            "risky_high_outcome_units",
            "risky_low_outcome_units",
            "probability",
            "required_terms",
        }
        if not isinstance(payoff, Mapping) or not required.issubset(payoff):
            raise ValueError(f"{entry.construct_id} calibration neutral_payoff contract is incomplete.")
        if (
            payoff.get("sure_outcome_units") != 10
            or payoff.get("risky_high_outcome_units") != 20
            or payoff.get("risky_low_outcome_units") != 0
            or payoff.get("probability") != "even"
            or not isinstance(payoff.get("required_terms"), list)
            or not payoff.get("required_terms")
        ):
            raise ValueError(f"{entry.construct_id} calibration neutral_payoff must be exact 10 vs 20-or-0 at even probability.")
    _downstream_pool_separation(entry)


def _validate_downstream_records(
    entry: _PlanEntry,
    records: Iterable[PromptRecord],
    *,
    mode: str,
    input_price: float | None = None,
    output_price: float | None = None,
    selected_splits: Iterable[str] | None = None,
    allowed_plan_hashes_by_split: Mapping[str, Iterable[str]] | None = None,
    allowed_plan_ids_by_split: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    materialized = tuple(records)
    if not materialized:
        raise ValueError(f"Generated downstream output for {entry.construct_id} is empty.")
    validate_prompt_records(materialized, {entry.construct_id: entry.spec}, require_all_splits=False)
    required_splits = frozenset(selected_splits) if selected_splits is not None else _splits_for_entry(entry)
    registered_splits = _splits_for_entry(entry)
    if not required_splits or not required_splits.issubset(registered_splits):
        raise ValueError(
            f"{entry.construct_id} selected downstream splits {sorted(required_splits)} "
            f"are not a subset of {sorted(registered_splits)}."
        )
    allowed_hashes = {
        str(split): {str(value) for value in values}
        for split, values in (allowed_plan_hashes_by_split or {}).items()
    }
    allowed_ids = {
        str(split): {str(value) for value in values}
        for split, values in (allowed_plan_ids_by_split or {}).items()
    }
    if any(not values for values in allowed_hashes.values()):
        raise ValueError("allowed_plan_hashes_by_split cannot contain an empty allowance.")
    if any(not values for values in allowed_ids.values()):
        raise ValueError("allowed_plan_ids_by_split cannot contain an empty allowance.")
    unknown_hash_splits = set(allowed_hashes) - set(registered_splits)
    unknown_id_splits = set(allowed_ids) - set(registered_splits)
    if unknown_hash_splits or unknown_id_splits:
        raise ValueError(
            f"Plan provenance allowances contain unknown splits: "
            f"hashes={sorted(unknown_hash_splits)}, ids={sorted(unknown_id_splits)}."
        )
    if any(record.split not in required_splits for record in materialized):
        raise ValueError(f"{entry.construct_id} downstream output contains vector/non-downstream records.")
    expected = _expected_summary(
        entry,
        mode=mode,
        input_price=input_price,
        output_price=output_price,
        selected_splits=required_splits,
    )
    split_counts: dict[str, int] = {}
    cells = _cells_by_split(entry)
    if not required_splits.issubset(set(cells)):
        raise ValueError(f"{entry.construct_id} plan does not expose the selected downstream cells.")
    _validate_calibration_plan(entry)
    observed_schedules: dict[str, tuple[tuple[Any, ...], ...]] = {}
    seen_ids: set[str] = set()
    for record in materialized:
        if record.prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt_id in {entry.construct_id}: {record.prompt_id}")
        seen_ids.add(record.prompt_id)
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        cell = cells[record.split]
        if record.prompt_role != str(cell["prompt_role"]):
            raise ValueError(f"{record.prompt_id} has a role inconsistent with its registered cell.")
        if record.prompt_family != str(cell["prompt_family"]):
            raise ValueError(f"{record.prompt_id} has a prompt family inconsistent with its registered cell.")
        if record.metadata.get("content_pool") != cell["content_pool"]:
            raise ValueError(f"{record.prompt_id} has an unexpected content_pool.")
        if record.condition_id not in (None, "", "neutral"):
            raise ValueError(f"Downstream prompt {record.prompt_id} must have neutral condition_id.")
        expected_task, expected_parser, expected_format = _role_task_contract(entry, cell)
        if record.task_id != str(cell.get("task_id", expected_task["task_id"])) or record.task_id != str(expected_task["task_id"]):
            raise ValueError(f"{record.prompt_id} has an incompatible task_id.")
        if record.parser_id != str(cell.get("parser_id", expected_parser)) or record.parser_id != expected_parser:
            raise ValueError(f"{record.prompt_id} has an incompatible parser_id.")
        if (
            record.expected_output_format != str(cell.get("expected_output_format", expected_format))
            or record.expected_output_format != expected_format
        ):
            raise ValueError(f"{record.prompt_id} has an incompatible expected_output_format.")
        parser_issues = _parser_instruction_issues(record)
        if parser_issues:
            raise ValueError(f"{record.prompt_id} has an incomplete parser instruction: {'; '.join(parser_issues)}.")
        if record.prompt_role == "calibration":
            cue_hits = _calibration_cue_hits(entry, record.prompt_text)
            if cue_hits:
                raise ValueError(
                    f"{record.prompt_id} contains calibration target-cue terms: {sorted(set(cue_hits))}."
                )
            text_issues = _calibration_text_issues(entry, record)
            if text_issues:
                raise ValueError(
                    f"{record.prompt_id} violates its calibration text contract: {'; '.join(text_issues)}."
                )
        pool_issues = _downstream_pool_text_issues(entry, record)
        if pool_issues:
            raise ValueError(f"{record.prompt_id} violates downstream pool separation: {'; '.join(pool_issues)}.")
        behavioral_issues = behavioral_record_issues(entry.spec, entry.plan, record)
        if behavioral_issues:
            raise ValueError(
                f"{record.prompt_id} violates its registered behavioral repair contract: "
                + "; ".join(behavioral_issues)
                + "."
            )
        categories = dict(cell.get("category_balance", {}))
        metadata_schedule = dict(cell.get("metadata_schedule", {}))
        metadata = record.metadata.get("task_metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else record.metadata
        missing = [field for field in (*categories, *metadata_schedule) if field not in metadata]
        if missing:
            raise ValueError(f"{record.prompt_id} is missing scheduled task metadata: {missing}.")
        # The generic parser checks the request-local assignment.  Re-check
        # the final CSV against its exact global schedule to catch accidental
        # concatenation or hand-edits after a checkpoint.
        index = split_counts[record.split] - 1
        expected_assignments = {
            **{field: schedule[index] for field, schedule in categories.items()},
            **{field: schedule[index] for field, schedule in metadata_schedule.items()},
        }
        for field, expected_value in expected_assignments.items():
            if metadata[field] != expected_value:
                raise ValueError(
                    f"{record.prompt_id} has {field}={metadata[field]!r}; expected registered "
                    f"schedule value {expected_value!r} at index {index}."
                )
    for split in required_splits:
        observed_schedules[split] = _schedule_combinations(cells[split])
        if split_counts.get(split, 0) != expected["records_by_split"].get(split, 0):
            raise ValueError(
                f"{entry.construct_id} {split} count={split_counts.get(split, 0)}; "
                f"expected {expected['records_by_split'].get(split, 0)}."
            )
    if (
        "behavior_eval" in observed_schedules
        and "steering_eval" in observed_schedules
        and observed_schedules["behavior_eval"] != observed_schedules["steering_eval"]
    ):
        raise ValueError(f"{entry.construct_id} behavior_eval and steering_eval do not share an identical factor schedule.")
    for record in materialized:
        permitted_hashes = allowed_hashes.get(record.split, {entry.plan_sha256})
        observed_hash = record.metadata.get("generation_plan_sha256")
        if observed_hash not in permitted_hashes:
            raise ValueError(
                f"{record.prompt_id} has plan hash {observed_hash!r}; expected one of "
                f"{sorted(permitted_hashes)} for split {record.split}."
            )
        permitted_ids = allowed_ids.get(record.split, {str(entry.plan["plan_id"])})
        observed_id = record.metadata.get("generation_plan_id")
        if observed_id not in permitted_ids:
            raise ValueError(
                f"{record.prompt_id} has plan ID {observed_id!r}; expected one of "
                f"{sorted(permitted_ids)} for split {record.split}."
            )
    expected_aliases = {str(model["alias"]) for model in entry.plan["models"]}
    aliases = {str(record.metadata.get("source_model_alias")) for record in materialized}
    if aliases != expected_aliases:
        raise ValueError(f"{entry.construct_id} downstream output has model aliases {sorted(aliases)}; expected {sorted(expected_aliases)}.")
    pools = [str(cells[split]["content_pool"]) for split in sorted(required_splits)]
    if len(set(pools)) != len(required_splits):
        raise ValueError(
            f"{entry.construct_id} must use one distinct downstream content pool per role."
        )
    families = [str(cells[split]["prompt_family"]) for split in sorted(required_splits)]
    if len(set(families)) != len(required_splits):
        raise ValueError(
            f"{entry.construct_id} must use one distinct downstream prompt family per role."
        )
    _downstream_pool_separation(entry)
    # Every downstream pool must be domain-disjoint from the probe pools and
    # from the other downstream pools.  This is a plan-level guard; text-level
    # overlap is handled by the audit below.
    all_pool_domains: dict[str, set[str]] = {
        str(pool_id): {str(domain) for domain in pool["domains"]}
        for pool_id, pool in entry.plan["content_pools"].items()
    }
    for left in pools:
        for right in all_pool_domains:
            if left != right and all_pool_domains[left] & all_pool_domains[right]:
                raise ValueError(f"{entry.construct_id} content pools {left} and {right} share domains.")
    return {
        "record_count": len(materialized),
        "split_counts": dict(sorted(split_counts.items())),
        "expected_record_count": expected["expected_record_count"],
        "expected_split_counts": dict(expected["records_by_split"]),
        "request_count": expected["request_count"],
        "estimated_input_tokens": expected["estimated_input_tokens"],
        "estimated_output_tokens": expected["estimated_output_tokens"],
        "estimated_total_tokens": expected["estimated_total_tokens"],
        "estimated_cost_usd": expected["estimated_cost_usd"],
        "schedule_combinations": {
            split: [list(item) for item in combinations]
            for split, combinations in sorted(observed_schedules.items())
        },
    }


def _normalise_text(text: str) -> str:
    return " ".join(_WORD_PATTERN.findall(text.casefold()))


def _content_tokens(text: str, *, template_tokens: set[str] = frozenset()) -> frozenset[str]:
    normalized = _normalise_text(text)
    return frozenset(
        token
        for token in normalized.split()
        if len(token) > 2 and token not in _STOPWORDS and token not in template_tokens
    )


def _ngrams(tokens: Iterable[str], n: int = 4) -> frozenset[tuple[str, ...]]:
    values = list(tokens)
    return frozenset(tuple(values[index : index + n]) for index in range(max(0, len(values) - n + 1)))


def _jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _prompt_signature(record: PromptRecord, template_tokens: set[str]) -> dict[str, Any]:
    normalized = _normalise_text(record.prompt_text)
    content_sequence = tuple(
        token
        for token in normalized.split()
        if token not in template_tokens and token not in _STOPWORDS
    )
    tokens = frozenset(content_sequence)
    return {
        "normalized": normalized,
        "tokens": tokens,
        "content_normalized": " ".join(content_sequence),
        "ngrams": _ngrams(content_sequence),
    }


def _template_tokens(entry: _PlanEntry, prompt_role: str | None = None) -> set[str]:
    task = registered_task_for_role(entry.spec, prompt_role) if prompt_role else entry.spec.independent_behavior_task
    task_template = str(task.get("prompt_template", ""))
    return set(_WORD_PATTERN.findall(task_template.casefold()))


def audit_downstream_inventory(
    records: Iterable[PromptRecord],
    entries: Iterable[_PlanEntry],
    *,
    vector_reference: Path | None = DEFAULT_VECTOR_REFERENCE,
    char_threshold: float = 0.93,
    token_threshold: float = 0.82,
    ngram_threshold: float = 0.55,
) -> dict[str, Any]:
    """Audit duplicates, near-overlap, pool separation, and probe leakage."""

    materialized = tuple(records)
    entries_by_id = {entry.construct_id: entry for entry in entries}
    validate_prompt_records(materialized, {entry.construct_id: entry.spec for entry in entries}, require_all_splits=False)
    flags: list[dict[str, Any]] = []
    signatures: dict[str, dict[str, Any]] = {}
    normalized_index: dict[str, str] = {}
    for record in materialized:
        entry = entries_by_id[record.construct_id]
        parser_issues = _parser_instruction_issues(record)
        if parser_issues:
            flags.append({
                "severity": "severe",
                "flag_type": "parser_instruction_incomplete",
                "candidate_prompt_id": record.prompt_id,
                "candidate_role": record.prompt_role,
                "issues": list(parser_issues),
            })
        if record.prompt_role == "calibration":
            try:
                calibration_contract = _calibration_contract(entry)
                cue_hits = _calibration_cue_hits(entry, record.prompt_text)
                calibration_text_issues = _calibration_text_issues(entry, record)
            except ValueError as exc:
                calibration_contract = None
                cue_hits = ()
                calibration_text_issues = ()
                flags.append({
                    "severity": "severe",
                    "flag_type": "calibration_schedule_missing",
                    "candidate_prompt_id": record.prompt_id,
                    "candidate_role": record.prompt_role,
                    "message": str(exc),
                })
            if cue_hits:
                flags.append({
                    "severity": "severe",
                    "flag_type": "calibration_target_cue",
                    "candidate_prompt_id": record.prompt_id,
                    "candidate_role": record.prompt_role,
                    "terms": sorted(set(cue_hits)),
                })
            if calibration_text_issues:
                flags.append({
                    "severity": "severe",
                    "flag_type": "calibration_text_contract",
                    "candidate_prompt_id": record.prompt_id,
                    "candidate_role": record.prompt_role,
                    "issues": list(calibration_text_issues),
                })
            if calibration_contract is not None:
                task_metadata = record.metadata.get("task_metadata")
                task_metadata = dict(task_metadata) if isinstance(task_metadata, Mapping) else record.metadata
                for field, expected_value in dict(calibration_contract.get("neutral_fields", {})).items():
                    if task_metadata.get(field) != expected_value:
                        flags.append({
                            "severity": "severe",
                            "flag_type": "calibration_non_neutral_metadata",
                            "candidate_prompt_id": record.prompt_id,
                            "candidate_role": record.prompt_role,
                            "field": field,
                            "observed": task_metadata.get(field),
                            "expected": expected_value,
                        })
        pool_text_issues = _downstream_pool_text_issues(entry, record)
        if pool_text_issues:
            flags.append({
                "severity": "severe",
                "flag_type": "downstream_pool_semantic_separation",
                "candidate_prompt_id": record.prompt_id,
                "candidate_role": record.prompt_role,
                "content_pool": record.metadata.get("content_pool"),
                "issues": list(pool_text_issues),
            })
        behavioral_issues = behavioral_record_issues(entry.spec, entry.plan, record)
        if behavioral_issues:
            flags.append({
                "severity": "severe",
                "flag_type": "behavioral_contract_violation",
                "candidate_prompt_id": record.prompt_id,
                "candidate_role": record.prompt_role,
                "issues": list(behavioral_issues),
            })
        signature = _prompt_signature(record, _template_tokens(entry, record.prompt_role))
        signatures[record.prompt_id] = signature
        previous = normalized_index.get(signature["normalized"])
        if previous is not None and previous != record.prompt_id:
            flags.append({
                "severity": "severe",
                "flag_type": "exact_normalized_duplicate",
                "candidate_prompt_id": record.prompt_id,
                "reference_prompt_id": previous,
                "candidate_role": record.prompt_role,
                "reference_role": "downstream",
                "char_similarity": 1.0,
                "token_jaccard": 1.0,
                "ngram_jaccard": 1.0,
            })
        else:
            normalized_index[signature["normalized"]] = record.prompt_id

    # Compare only records that can actually leak into one another.  Same-role
    # pairs in different constructs are also compared, while the current
    # record's own pair is impossible for downstream singles.
    record_list = list(materialized)
    for index, candidate in enumerate(record_list):
        left = signatures[candidate.prompt_id]
        for reference in record_list[index + 1 :]:
            if candidate.prompt_id == reference.prompt_id:
                continue
            right = signatures[reference.prompt_id]
            token_score = _jaccard(left["tokens"], right["tokens"])
            ngram_score = _jaccard(left["ngrams"], right["ngrams"])
            if token_score < token_threshold and ngram_score < ngram_threshold:
                continue
            # Compare substantive content rather than the repeated task and
            # response scaffolding. Exact full-text duplicates were already
            # handled above and remain severe regardless of this adjustment.
            char_score = SequenceMatcher(
                None,
                left["content_normalized"],
                right["content_normalized"],
            ).ratio()
            if char_score < char_threshold and token_score < token_threshold and ngram_score < ngram_threshold:
                continue
            same_construct = candidate.construct_id == reference.construct_id
            same_pool = candidate.prompt_role == reference.prompt_role and candidate.prompt_family == reference.prompt_family
            # Shared pool membership is useful context for review but is not
            # itself evidence of a duplicate: calibration items necessarily
            # share a registered neutral task scaffold. Escalate only when the
            # substantive content is genuinely near-identical.
            severity = "severe" if char_score >= 0.97 or token_score >= 0.92 else "warning"
            flags.append({
                "severity": severity,
                "flag_type": "downstream_near_overlap",
                "candidate_prompt_id": candidate.prompt_id,
                "reference_prompt_id": reference.prompt_id,
                "candidate_role": candidate.prompt_role,
                "reference_role": reference.prompt_role,
                "candidate_construct_id": candidate.construct_id,
                "reference_construct_id": reference.construct_id,
                "same_construct": same_construct,
                "same_pool": same_pool,
                "char_similarity": round(char_score, 6),
                "token_jaccard": round(token_score, 6),
                "ngram_jaccard": round(ngram_score, 6),
            })

    vector_count = 0
    if vector_reference is not None:
        if not vector_reference.exists():
            raise ValueError(f"Vector reference inventory does not exist: {vector_reference}")
        vector_records = load_prompt_records(vector_reference)
        vector_count = len(vector_records)
        # Use the construct's downstream task template to remove generic task
        # boilerplate from the candidate signature, and a probe template token
        # set for the vector side.  Exact normalized text is always severe.
        vector_signatures = {
            record.prompt_id: _prompt_signature(record, set())
            for record in vector_records
        }
        vector_exact: dict[str, str] = {}
        for vector in vector_records:
            vector_exact.setdefault(vector_signatures[vector.prompt_id]["normalized"], vector.prompt_id)
        for candidate in materialized:
            left = signatures[candidate.prompt_id]
            previous = vector_exact.get(left["normalized"])
            if previous is not None:
                flags.append({
                    "severity": "severe",
                    "flag_type": "probe_exact_normalized_leakage",
                    "candidate_prompt_id": candidate.prompt_id,
                    "reference_prompt_id": previous,
                    "candidate_role": candidate.prompt_role,
                    "reference_role": "probe",
                    "char_similarity": 1.0,
                    "token_jaccard": 1.0,
                    "ngram_jaccard": 1.0,
                })
            # Compare on a compact token gate first.  This avoids an expensive
            # all-pairs SequenceMatcher over the 5,760-record vector bank.
            best: tuple[float, float, float, str] | None = None
            for vector in vector_records:
                right = vector_signatures[vector.prompt_id]
                token_score = _jaccard(left["tokens"], right["tokens"])
                ngram_score = _jaccard(left["ngrams"], right["ngrams"])
                if token_score < token_threshold and ngram_score < ngram_threshold:
                    continue
                char_score = SequenceMatcher(None, left["normalized"], right["normalized"]).ratio()
                candidate_score = max(char_score, token_score, ngram_score)
                if best is None or candidate_score > max(best[:3]):
                    best = (char_score, token_score, ngram_score, vector.prompt_id)
            if best is not None:
                char_score, token_score, ngram_score, vector_id = best
                flags.append({
                    "severity": "severe" if char_score >= 0.97 or token_score >= 0.92 else "warning",
                    "flag_type": "probe_near_overlap",
                    "candidate_prompt_id": candidate.prompt_id,
                    "reference_prompt_id": vector_id,
                    "candidate_role": candidate.prompt_role,
                    "reference_role": "probe",
                    "candidate_construct_id": candidate.construct_id,
                    "char_similarity": round(char_score, 6),
                    "token_jaccard": round(token_score, 6),
                    "ngram_jaccard": round(ngram_score, 6),
                })
    flags.sort(key=lambda row: (
        0 if row.get("severity") == "severe" else 1,
        str(row.get("flag_type", "")),
        str(row.get("candidate_prompt_id", "")),
        str(row.get("reference_prompt_id", "")),
    ))
    return {
        "audit_version": DOWNSTREAM_AUDIT_VERSION,
        "record_count": len(materialized),
        "vector_reference_path": str(vector_reference) if vector_reference is not None else None,
        "vector_reference_record_count": vector_count,
        "flag_count": len(flags),
        "severe_flag_count": sum(flag.get("severity") == "severe" for flag in flags),
        "warning_flag_count": sum(flag.get("severity") == "warning" for flag in flags),
        "passed": not any(flag.get("severity") == "severe" for flag in flags),
        "flags": flags,
    }


def _checkpoint_checksum(payload: Mapping[str, Any]) -> str:
    return _canonical_sha256({key: value for key, value in payload.items() if key != "checksum_sha256"})


def _checkpoint_identity(entry: _PlanEntry, run_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DOWNSTREAM_MANIFEST_VERSION,
        "checkpoint_type": "downstream_prompt_generation_job_checkpoint",
        "construct_id": entry.construct_id,
        "source_plan_sha256": entry.source_plan_sha256,
        "effective_plan_sha256": entry.plan_sha256,
        "run_identity": dict(run_identity),
    }


def _write_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    attempts: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": DOWNSTREAM_MANIFEST_VERSION,
        "checkpoint_type": "downstream_prompt_generation_job_checkpoint",
        "identity": dict(identity),
        "jobs": [dict(jobs[job_id]) for job_id in sorted(jobs)],
        "attempts": {
            str(job_id): [dict(attempt) for attempt in history]
            for job_id, history in sorted((attempts or {}).items())
        },
    }
    payload["checksum_sha256"] = _checkpoint_checksum(payload)
    _atomic_write_json(payload, path)


def _checkpoint_payload(job_id: str, records: Iterable[PromptRecord], metadata: Mapping[str, Any]) -> dict[str, Any]:
    raw_cost = metadata.get("actual_cost_usd")
    if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
        raise ValueError(f"Checkpoint job {job_id} is missing numeric actual_cost_usd provenance.")
    cost = float(raw_cost)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError(f"Checkpoint job {job_id} has invalid actual_cost_usd provenance.")
    return {
        "job_id": job_id,
        "records": [record.to_mapping() for record in records],
        "response_metadata": dict(metadata),
        "actual_cost_usd": cost,
    }


def _load_checkpoint(
    path: Path,
    *,
    entry: _PlanEntry,
    identity: Mapping[str, Any],
    mode: str,
    selected_splits: Iterable[str] | None = None,
) -> tuple[dict[str, tuple[PromptRecord, ...]], dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _load_json_object(path, label="downstream prompt checkpoint")
    checksum = payload.get("checksum_sha256")
    if not isinstance(checksum, str) or checksum != _checkpoint_checksum(payload):
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    if payload.get("checkpoint_type") != "downstream_prompt_generation_job_checkpoint":
        raise ValueError(f"Unsupported checkpoint type in {path}.")
    if payload.get("identity") != dict(identity):
        raise ValueError(f"Checkpoint identity is stale for {entry.construct_id}; use a new output directory.")
    expected_jobs = {job.job_id for job in iter_generation_request_jobs(
        entry.plan,
        count_per_model_override=_mode_count(mode),
        splits=set(selected_splits) if selected_splits is not None else set(_splits_for_entry(entry)),
    )}
    records_by_job: dict[str, tuple[PromptRecord, ...]] = {}
    metadata_by_job: dict[str, dict[str, Any]] = {}
    seen_prompt_ids: set[str] = set()
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError(f"Checkpoint {path} must contain a jobs list.")
    for raw_job in raw_jobs:
        if not isinstance(raw_job, Mapping):
            raise ValueError(f"Checkpoint {path} contains a malformed job.")
        job_id = raw_job.get("job_id")
        if not isinstance(job_id, str) or job_id not in expected_jobs or job_id in records_by_job:
            raise ValueError(f"Checkpoint {path} contains an unknown or duplicate job: {job_id!r}.")
        raw_records = raw_job.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"Checkpoint job {job_id} has no records.")
        records = tuple(PromptRecord.from_mapping(row) for row in raw_records if isinstance(row, Mapping))
        if len(records) != len(raw_records):
            raise ValueError(f"Checkpoint job {job_id} contains malformed records.")
        validate_prompt_records(records, {entry.construct_id: entry.spec}, require_all_splits=False)
        for record in records:
            if record.metadata.get("generation_job_id") != job_id:
                raise ValueError(f"Checkpoint job {job_id} contains a record from another job.")
            if record.prompt_id in seen_prompt_ids:
                raise ValueError(f"Checkpoint {path} duplicates prompt_id={record.prompt_id!r}.")
            seen_prompt_ids.add(record.prompt_id)
        raw_metadata = raw_job.get("response_metadata")
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"Checkpoint job {job_id} is missing response_metadata.")
        raw_cost = raw_metadata.get("actual_cost_usd", raw_job.get("actual_cost_usd"))
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise ValueError(f"Checkpoint job {job_id} is missing numeric cost provenance.")
        metadata = dict(raw_metadata)
        metadata["actual_cost_usd"] = float(raw_cost)
        records_by_job[job_id] = records
        metadata_by_job[job_id] = metadata
    raw_attempts = payload.get("attempts", {})
    if not isinstance(raw_attempts, Mapping):
        raise ValueError(f"Checkpoint {path} attempts must be an object.")
    attempts_by_job: dict[str, list[dict[str, Any]]] = {}
    for raw_job_id, raw_history in raw_attempts.items():
        job_id = str(raw_job_id)
        if job_id not in expected_jobs:
            raise ValueError(f"Checkpoint {path} contains attempts for an unknown job: {job_id!r}.")
        if not isinstance(raw_history, list):
            raise ValueError(f"Checkpoint {path} attempts for {job_id} must be a list.")
        history: list[dict[str, Any]] = []
        for attempt in raw_history:
            if not isinstance(attempt, Mapping):
                raise ValueError(f"Checkpoint {path} contains a malformed attempt for {job_id}.")
            status = attempt.get("status")
            attempt_number = attempt.get("attempt")
            if status not in {"accepted", "rejected"} or not isinstance(attempt_number, int) or attempt_number < 1:
                raise ValueError(f"Checkpoint {path} contains an invalid attempt record for {job_id}.")
            if status == "rejected" and not isinstance(attempt.get("rejection_reason"), str):
                raise ValueError(f"Checkpoint {path} rejected attempt {job_id} lacks an exact rejection reason.")
            history.append(dict(attempt))
        if history:
            attempts_by_job[job_id] = history
    # Checkpoints written before attempt-level logging was introduced still
    # contain accepted response metadata.  Reconstruct those accepted attempts
    # explicitly so a resume cannot silently lose their usage/provenance.  A
    # historical rejected attempt that was never logged remains unattributed
    # and is surfaced by the run-level accounting instead of being invented.
    for job_id, metadata in metadata_by_job.items():
        if job_id in attempts_by_job:
            continue
        reconstructed_metadata = dict(metadata)
        reconstructed_metadata.setdefault("semantic_attempt", 1)
        reconstructed_metadata.setdefault("semantic_attempt_status", "accepted")
        reconstructed_metadata["attempt_provenance"] = "reconstructed_from_checkpoint_metadata"
        attempts_by_job[job_id] = [{
            "attempt": 1,
            "status": "accepted",
            "rejection_reason": None,
            "response_metadata": reconstructed_metadata,
        }]
    return records_by_job, metadata_by_job, {
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_job_count": len(records_by_job),
        "checkpoint_actual_cost_usd": (
            sum(
                float(attempt.get("response_metadata", {}).get("actual_cost_usd", 0.0) or 0.0)
                for history in attempts_by_job.values()
                for attempt in history
            )
            if attempts_by_job
            else sum(float(metadata.get("actual_cost_usd", 0.0)) for metadata in metadata_by_job.values())
        ),
        "attempts_by_job": attempts_by_job,
        "checkpoint_attempt_count": sum(len(history) for history in attempts_by_job.values()),
        "checkpoint_rejected_attempt_count": sum(
            sum(attempt.get("status") == "rejected" for attempt in history)
            for history in attempts_by_job.values()
        ),
    }


def _reconstruct_spend(records: Iterable[PromptRecord]) -> float:
    costs: dict[str, float] = {}
    for record in records:
        job_id = str(record.metadata.get("generation_job_id") or "")
        raw = record.metadata.get("generation_actual_cost_usd")
        if not job_id or isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise ValueError(f"Cannot reconstruct generation cost from prompt {record.prompt_id}.")
        cost = float(raw)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"Prompt {record.prompt_id} has invalid generation cost provenance.")
        if job_id in costs and abs(costs[job_id] - cost) > 1e-12:
            raise ValueError(f"Prompt {record.prompt_id} has inconsistent cost for generation job {job_id}.")
        costs[job_id] = cost
    return sum(costs.values())


def _attempt_usage(job_attempts: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    """Summarize durable request-attempt metadata without double-counting jobs."""

    attempt_count = 0
    rejected_attempt_count = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    actual_cost_usd = 0.0
    for history in job_attempts.values():
        for attempt in history:
            if not isinstance(attempt, Mapping):
                continue
            attempt_count += 1
            rejected_attempt_count += int(attempt.get("status") == "rejected")
            metadata = attempt.get("response_metadata")
            if not isinstance(metadata, Mapping):
                continue
            input_tokens += int(metadata.get("input_tokens", 0) or 0)
            output_tokens += int(metadata.get("output_tokens", 0) or 0)
            total_tokens += int(metadata.get("total_tokens", 0) or 0)
            actual_cost_usd += float(metadata.get("actual_cost_usd", 0.0) or 0.0)
    return {
        "attempt_count": attempt_count,
        "rejected_attempt_count": rejected_attempt_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "actual_cost_usd": actual_cost_usd,
    }


def _attempt_history_map(
    raw_histories: Any,
    *,
    label: str,
) -> dict[str, list[dict[str, Any]]]:
    """Validate and normalize a persisted job-attempt map."""

    if raw_histories is None:
        return {}
    if not isinstance(raw_histories, Mapping):
        raise ValueError(f"{label} must be an object mapping job IDs to attempt histories.")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for raw_job_id, raw_history in raw_histories.items():
        job_id = str(raw_job_id)
        if not job_id.strip():
            raise ValueError(f"{label} contains an empty job ID.")
        if not isinstance(raw_history, list):
            raise ValueError(f"{label} history for {job_id} must be a list.")
        history: list[dict[str, Any]] = []
        seen_attempts: set[int] = set()
        for raw_attempt in raw_history:
            if not isinstance(raw_attempt, Mapping):
                raise ValueError(f"{label} contains a malformed attempt for {job_id}.")
            attempt_number = raw_attempt.get("attempt")
            status = raw_attempt.get("status")
            if isinstance(attempt_number, bool) or not isinstance(attempt_number, int) or attempt_number < 1:
                raise ValueError(f"{label} contains an invalid attempt number for {job_id}.")
            if attempt_number in seen_attempts or status not in {"accepted", "rejected"}:
                raise ValueError(f"{label} contains a duplicate or invalid attempt for {job_id}.")
            if status == "rejected" and not isinstance(raw_attempt.get("rejection_reason"), str):
                raise ValueError(f"{label} rejected attempt {job_id} lacks an exact rejection reason.")
            response_metadata = raw_attempt.get("response_metadata", {})
            if not isinstance(response_metadata, Mapping):
                raise ValueError(f"{label} attempt {job_id} has malformed response metadata.")
            item = dict(raw_attempt)
            item["response_metadata"] = dict(response_metadata)
            history.append(item)
            seen_attempts.add(attempt_number)
        normalized[job_id] = sorted(history, key=lambda item: int(item["attempt"]))
    return normalized


def _merge_attempt_histories(
    *history_maps: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Merge histories from run state/checkpoints, rejecting conflicting provenance."""

    merged: dict[str, dict[int, dict[str, Any]]] = {}
    for history_map in history_maps:
        for raw_job_id, raw_history in history_map.items():
            job_id = str(raw_job_id)
            attempts = merged.setdefault(job_id, {})
            for raw_attempt in raw_history:
                attempt = dict(raw_attempt)
                attempt_number = int(attempt["attempt"])
                existing = attempts.get(attempt_number)
                if existing is not None and existing != attempt:
                    existing_metadata = existing.get("response_metadata", {})
                    attempt_metadata = attempt.get("response_metadata", {})
                    if isinstance(existing_metadata, Mapping) and existing_metadata.get("attempt_provenance") == "reconstructed_from_checkpoint_metadata":
                        attempts[attempt_number] = attempt
                        continue
                    if isinstance(attempt_metadata, Mapping) and attempt_metadata.get("attempt_provenance") == "reconstructed_from_checkpoint_metadata":
                        continue
                    raise ValueError(
                        f"Conflicting persisted provenance for job {job_id} attempt {attempt_number}."
                    )
                attempts[attempt_number] = attempt
    return {
        job_id: [attempts[number] for number in sorted(attempts)]
        for job_id, attempts in merged.items()
    }


def _attempt_keys(job_attempts: Mapping[str, Iterable[Mapping[str, Any]]]) -> set[tuple[str, int]]:
    return {
        (str(job_id), int(attempt["attempt"]))
        for job_id, history in job_attempts.items()
        for attempt in history
    }


def _attempt_delta(
    job_attempts: Mapping[str, Iterable[Mapping[str, Any]]],
    baseline_keys: set[tuple[str, int]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(job_id): [
            dict(attempt)
            for attempt in history
            if (str(job_id), int(attempt["attempt"])) not in baseline_keys
        ]
        for job_id, history in job_attempts.items()
        if any((str(job_id), int(attempt["attempt"])) not in baseline_keys for attempt in history)
    }


def _load_review_manifest_for_gate(
    manifest_path: Path,
    manifest_hash: str,
    *,
    label: str,
) -> dict[str, Any]:
    if len(manifest_hash) != 64 or file_sha256(manifest_path) != manifest_hash:
        raise ValueError(f"{label} review manifest hash mismatch.")
    manifest = _load_json_object(manifest_path, label=f"{label} downstream review manifest")
    if (
        manifest.get("manifest_type") != "downstream_prompt_generation"
        or manifest.get("status") != "complete_review"
        or manifest.get("run_mode") != "review"
        or manifest.get("partial") is not True
        or manifest.get("dry_run") is not False
        or manifest.get("frozen") is not False
        or int(manifest.get("audit", {}).get("severe_flag_count", 1)) != 0
    ):
        raise ValueError(
            f"{label} requires a complete, audited, non-dry review manifest with no severe flags."
        )
    if (
        manifest.get("provider") != DEFAULT_PROVIDER
        or manifest.get("requested_model") != DEFAULT_MODEL
        or manifest.get("reasoning_effort") != DEFAULT_REASONING_EFFORT
    ):
        raise ValueError(f"{label} review manifest is not an OpenAI GPT-5.6 Luna xhigh artifact.")
    serialized = json.dumps(manifest, sort_keys=True).lower()
    if "sonnet" in serialized or "claude" in serialized:
        raise ValueError(f"{label} review manifest contains non-Luna model provenance.")
    return manifest


def _resolve_gate_manifest_path(gate_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must include a review_manifest_path.")
    manifest_path = Path(value)
    if not manifest_path.is_absolute():
        manifest_path = (gate_path.parent / manifest_path).resolve()
    return manifest_path


def _review_manifest_item_for_entry(
    manifest: Mapping[str, Any],
    entry: _PlanEntry,
    *,
    label: str,
    reviewed_plan_sha256: str | None = None,
) -> dict[str, Any]:
    review_constructs = {
        str(item.get("construct_id")): dict(item)
        for item in manifest.get("constructs", [])
        if isinstance(item, Mapping) and item.get("construct_id")
    }
    item = review_constructs.get(entry.construct_id)
    if item is None:
        raise ValueError(f"{label} review manifest is missing {entry.construct_id}.")
    expected_plan_sha256 = reviewed_plan_sha256 or entry.plan_sha256
    for field_name, expected in (
        ("source_plan_sha256", entry.source_plan_sha256),
        ("plan_sha256", expected_plan_sha256),
        ("spec_sha256", entry.spec_sha256),
    ):
        if item.get(field_name) != expected:
            raise ValueError(
                f"{label} review manifest has a stale {field_name} for {entry.construct_id}."
            )
    output = item.get("output_path")
    output_hash = item.get("output_sha256")
    if (
        not isinstance(output, str)
        or not isinstance(output_hash, str)
        or len(output_hash) != 64
        or file_sha256(Path(output)) != output_hash
    ):
        raise ValueError(f"{label} review output hash mismatch for {entry.construct_id}.")
    return {
        "construct_id": entry.construct_id,
        "source_plan_sha256": item["source_plan_sha256"],
        "plan_sha256": item["plan_sha256"],
        "reviewed_plan_sha256": item["plan_sha256"],
        "spec_sha256": item["spec_sha256"],
        "output_path": output,
        "output_sha256": output_hash,
    }


def _validate_quality_gate(path: Path, entries: Iterable[_PlanEntry]) -> dict[str, Any]:
    gate = _load_json_object(path, label="downstream quality gate")
    if gate.get("quality_gate_version") != QUALITY_GATE_VERSION or gate.get("approved") is not True or gate.get("status") != "approved":
        raise ValueError("Quality gate must set quality_gate_version='1', approved=true, status='approved'.")
    reviewer = gate.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Quality gate must identify a reviewer.")
    selected_entries = tuple(entries)
    expected_ids = {entry.construct_id for entry in selected_entries}
    components = gate.get("components")
    if components is None:
        manifest_value = gate.get("review_manifest_path")
        manifest_hash = gate.get("review_manifest_sha256")
        if not isinstance(manifest_hash, str):
            raise ValueError("Quality gate must include review_manifest_sha256.")
        manifest_path = _resolve_gate_manifest_path(path, manifest_value, label="Quality gate")
        manifest = _load_review_manifest_for_gate(manifest_path, manifest_hash, label="Quality gate")
        provenance = {
            entry.construct_id: _review_manifest_item_for_entry(
                manifest,
                entry,
                label="Quality gate",
            )
            for entry in selected_entries
        }
        return {
            "quality_gate_path": str(path.resolve()),
            "review_manifest_path": str(manifest_path),
            "review_manifest_sha256": manifest_hash,
            "reviewer": reviewer.strip(),
            "approved": True,
            "composite": False,
            "construct_provenance": provenance,
        }
    if not isinstance(components, list) or not components:
        raise ValueError("Composite quality gate components must be a non-empty list.")
    provenance: dict[str, Any] = {}
    component_records: list[dict[str, Any]] = []
    for index, raw_component in enumerate(components):
        if not isinstance(raw_component, Mapping):
            raise ValueError(f"Composite quality gate component {index} must be an object.")
        label = str(raw_component.get("label") or f"component_{index + 1}")
        manifest_hash = raw_component.get("review_manifest_sha256")
        if not isinstance(manifest_hash, str):
            raise ValueError(f"{label} must include review_manifest_sha256.")
        manifest_path = _resolve_gate_manifest_path(
            path,
            raw_component.get("review_manifest_path"),
            label=label,
        )
        manifest = _load_review_manifest_for_gate(manifest_path, manifest_hash, label=label)
        raw_construct_ids = raw_component.get("construct_ids")
        if not isinstance(raw_construct_ids, list) or not raw_construct_ids or any(
            not isinstance(construct_id, str) or not construct_id.strip()
            for construct_id in raw_construct_ids
        ):
            raise ValueError(f"{label} must list one or more construct_ids explicitly.")
        component_ids = [str(construct_id).strip() for construct_id in raw_construct_ids]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError(f"{label} contains duplicate construct_ids.")
        for construct_id in component_ids:
            if construct_id not in expected_ids:
                raise ValueError(f"{label} selects an unrequested construct {construct_id!r}.")
            if construct_id in provenance:
                raise ValueError(f"Composite quality gate selects {construct_id} more than once.")
            entry = next(entry for entry in selected_entries if entry.construct_id == construct_id)
            reviewed_plan_sha256 = raw_component.get("reviewed_plan_sha256")
            if reviewed_plan_sha256 is not None:
                if (
                    not isinstance(reviewed_plan_sha256, str)
                    or len(reviewed_plan_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in reviewed_plan_sha256.lower())
                ):
                    raise ValueError(f"{label} reviewed_plan_sha256 must be a lowercase SHA-256 string.")
            if raw_component.get("allow_runtime_plan_override") is True and reviewed_plan_sha256 is None:
                raise ValueError(
                    f"{label} runtime plan override requires an explicit reviewed_plan_sha256."
                )
            provenance[construct_id] = _review_manifest_item_for_entry(
                manifest,
                entry,
                label=label,
                reviewed_plan_sha256=reviewed_plan_sha256,
            )
        component_records.append({
            "label": label,
            "review_manifest_path": str(manifest_path),
            "review_manifest_sha256": manifest_hash,
            "construct_ids": component_ids,
            **({
                "allow_runtime_plan_override": True,
                "reviewed_plan_sha256": reviewed_plan_sha256,
            } if raw_component.get("allow_runtime_plan_override") is True else {}),
        })
    missing = expected_ids - set(provenance)
    if missing:
        raise ValueError(f"Composite quality gate is missing constructs: {sorted(missing)}.")
    return {
        "quality_gate_path": str(path.resolve()),
        "reviewer": reviewer.strip(),
        "approved": True,
        "composite": True,
        "components": component_records,
        "construct_provenance": provenance,
    }


def _construct_manifest(
    entry: _PlanEntry,
    output_path: Path,
    details: Mapping[str, Any],
    *,
    mode: str,
    dry_run: bool,
    provider: str,
    model: str | None,
    reasoning_effort: str | None,
    runtime_settings: Mapping[str, Any],
    selected_splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    scope_splits = frozenset(selected_splits) if selected_splits is not None else ALL_DOWNSTREAM_SPLITS
    return {
        "construct_id": entry.construct_id,
        "wave": entry.wave,
        "spec_path": str(entry.spec_path),
        "plan_path": str(entry.plan_path),
        "spec_sha256": entry.spec_sha256,
        "source_plan_sha256": entry.source_plan_sha256,
        "plan_sha256": entry.plan_sha256,
        "models": [dict(model_item) for model_item in entry.plan["models"]],
        "provider": provider,
        "requested_model": model,
        "reasoning_effort": reasoning_effort,
        "runtime_settings": dict(runtime_settings),
        "run_mode": mode,
        "partial": mode == "review",
        "confirmatory": False,
        "scope": "downstream",
        "scope_splits": sorted(scope_splits),
        "scope_partial": mode == "review" or scope_splits != ALL_DOWNSTREAM_SPLITS,
        "dry_run": dry_run,
        "record_count": int(details.get("record_count", 0)),
        "split_counts": dict(details.get("split_counts", {})),
        "expected_record_count": int(details.get("expected_record_count", 0)),
        "expected_split_counts": dict(details.get("expected_split_counts", {})),
        "request_count": int(details.get("request_count", 0)),
        "estimated_input_tokens": int(details.get("estimated_input_tokens", 0)),
        "estimated_output_tokens": int(details.get("estimated_output_tokens", 0)),
        "estimated_total_tokens": int(details.get("estimated_total_tokens", 0)),
        "estimated_cost_usd": details.get("estimated_cost_usd"),
        "actual_input_tokens": int(details.get("actual_input_tokens", 0) or 0),
        "actual_output_tokens": int(details.get("actual_output_tokens", 0) or 0),
        "actual_total_tokens": int(details.get("actual_total_tokens", 0) or 0),
        "actual_cost_usd": float(details.get("actual_cost_usd", 0.0) or 0.0),
        "attempted_cost_usd": float(details.get("attempted_cost_usd", details.get("actual_cost_usd", 0.0)) or 0.0),
        "accepted_record_cost_usd": float(details.get("accepted_record_cost_usd", 0.0) or 0.0),
        "attempt_count": int(details.get("attempt_count", 0) or 0),
        "rejected_attempt_count": int(details.get("rejected_attempt_count", 0) or 0),
        "job_attempts": details.get("job_attempts", {}),
        "output_path": str(output_path),
        "output_sha256": None if dry_run else details.get("output_sha256"),
        "checkpoint_path": details.get("checkpoint_path"),
        "checkpoint_sha256": details.get("checkpoint_sha256"),
        "checkpoint_job_count": int(details.get("checkpoint_job_count", 0) or 0),
    }


def orchestrate_downstream_prompts(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    waves: Iterable[int | str] | None = None,
    construct_ids: Iterable[str] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    mode: str = "full",
    selected_splits: Iterable[str] | None = None,
    workers: int = 4,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    resume: bool = False,
    dry_run: bool = False,
    api_key: str | None = None,
    request_fn: RequestFn | None = None,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_new_jobs: int | None = None,
    max_estimated_cost_usd: float | None = None,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
    cost_safety_multiplier: float = DEFAULT_COST_SAFETY_MULTIPLIER,
    vector_reference: str | Path | None = DEFAULT_VECTOR_REFERENCE,
    quality_gate_file: str | Path | None = None,
) -> dict[str, Any]:
    if workers < 1 or batch_size < 1 or max_output_tokens < 1:
        raise ValueError("workers, batch_size, and max_output_tokens must be positive.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a finite positive number.")
    if mode not in {"review", "full"}:
        raise ValueError("mode must be review or full.")
    if mode == "full" and batch_size < MIN_FULL_BATCH_SIZE:
        raise ValueError(
            f"full generation requires batch_size >= {MIN_FULL_BATCH_SIZE}; "
            "the final scheduled remainder of a cell may be smaller."
        )
    active_splits = None
    if selected_splits is not None:
        active_splits = frozenset(str(split).strip() for split in selected_splits if str(split).strip())
        if not active_splits or not active_splits.issubset(ALL_DOWNSTREAM_SPLITS):
            raise ValueError(
                f"selected_splits must be a non-empty subset of {sorted(ALL_DOWNSTREAM_SPLITS)}."
            )

    def entry_splits(entry: _PlanEntry) -> frozenset[str]:
        return active_splits if active_splits is not None else _splits_for_entry(entry)
    provider = str(provider).strip().lower()
    if provider != "openai":
        raise ValueError("The downstream prompt workflow is Luna-only: provider must be 'openai'.")
    model = str(model or DEFAULT_MODEL).strip()
    if model != DEFAULT_MODEL:
        raise ValueError(f"The downstream prompt workflow is Luna-only: model must be {DEFAULT_MODEL!r}.")
    reasoning_effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    if reasoning_effort != DEFAULT_REASONING_EFFORT:
        raise ValueError(
            f"The downstream prompt workflow requires reasoning_effort={DEFAULT_REASONING_EFFORT!r}."
        )
    if cost_safety_multiplier < 1:
        raise ValueError("cost_safety_multiplier must be at least 1.0.")
    if not dry_run and max_estimated_cost_usd is None:
        raise ValueError("Non-dry generation requires --max-estimated-cost-usd.")
    if not dry_run and (input_usd_per_million_tokens is None or output_usd_per_million_tokens is None):
        raise ValueError("Non-dry generation requires explicit input/output token prices.")
    if max_new_jobs is not None and (max_new_jobs < 0 or mode != "full"):
        raise ValueError("max_new_jobs is supported only as a non-negative full-mode invocation limit.")
    entries = _effective_entries(
        registry_path,
        waves=waves,
        construct_ids=construct_ids,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        model=model,
    )
    if not dry_run:
        # Validate every selected construct's calibration contract before the
        # first request.  Waves 2-4 remain planning artifacts until their
        # neutral schedules are specified; fail closed rather than generating
        # an inventory that cannot be calibrated scientifically.
        for entry in entries:
            _validate_calibration_plan(entry)
    vector_path = None if vector_reference is None else Path(vector_reference).resolve()
    output_root = Path(output_dir).resolve()
    output_paths = {entry.construct_id: output_root / f"{entry.construct_id}.csv" for entry in entries}
    checkpoint_paths = {entry.construct_id: output_root / "checkpoints" / f"{entry.construct_id}.json" for entry in entries}
    combined_path = output_root / "combined.csv"
    manifest_path = output_root / ("review_manifest.json" if mode == "review" else "final_inventory_manifest.json")
    run_state_path = output_root / "downstream_prompt_run_state.json"
    audit_summary_path = output_root / "audit_summary.json"
    audit_flags_path = output_root / "audit_flags.csv"
    run_identity = {
        "schema_version": DOWNSTREAM_MANIFEST_VERSION,
        "registry_path": str(Path(registry_path).resolve()),
        "construct_ids": [entry.construct_id for entry in entries],
        "source_plan_sha256": {entry.construct_id: entry.source_plan_sha256 for entry in entries},
        "effective_plan_sha256": {entry.construct_id: entry.plan_sha256 for entry in entries},
        "mode": mode,
        "provider": provider,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "scope": "downstream",
        "scope_splits": sorted(active_splits) if active_splits is not None else None,
        "runtime_settings": {
            "batch_size": batch_size,
            "max_output_tokens": max_output_tokens,
            "timeout_seconds": float(timeout_seconds),
            "input_usd_per_million_tokens": input_usd_per_million_tokens,
            "output_usd_per_million_tokens": output_usd_per_million_tokens,
            "cost_safety_multiplier": cost_safety_multiplier,
        },
    }
    runtime_settings = dict(run_identity["runtime_settings"])
    expected_summaries = {
        entry.construct_id: _expected_summary(
            entry,
            mode=mode,
            input_price=input_usd_per_million_tokens,
            output_price=output_usd_per_million_tokens,
            selected_splits=entry_splits(entry),
        )
        for entry in entries
    }
    estimated_costs = [summary["estimated_cost_usd"] for summary in expected_summaries.values()]
    estimated_cost = sum(estimated_costs) if all(cost is not None for cost in estimated_costs) else None
    budget_estimate = None if estimated_cost is None else estimated_cost * cost_safety_multiplier
    if not dry_run:
        if budget_estimate is None:
            raise ValueError("Cannot enforce the spending cap without explicit token prices.")
        if budget_estimate > float(max_estimated_cost_usd) + 1e-12:
            raise ValueError(
                f"Estimated generation cost ${budget_estimate:.4f} including safety margin exceeds cap "
                f"${float(max_estimated_cost_usd):.4f}."
            )
    quality_gate = None
    if mode == "full" and not dry_run:
        if quality_gate_file is None:
            raise ValueError("Full non-dry generation requires --quality-gate-file from an approved review manifest.")
        quality_gate = _validate_quality_gate(Path(quality_gate_file).resolve(), entries)

    if dry_run:
        construct_manifests = []
        split_counts: dict[str, int] = {}
        record_count = 0
        request_count = 0
        input_tokens = output_tokens = 0
        for entry in entries:
            summary = expected_summaries[entry.construct_id]
            details = {
                "record_count": summary["expected_record_count"],
                "split_counts": summary["records_by_split"],
                "expected_record_count": summary["expected_record_count"],
                "expected_split_counts": summary["records_by_split"],
                "request_count": summary["request_count"],
                "estimated_input_tokens": summary["estimated_input_tokens"],
                "estimated_output_tokens": summary["estimated_output_tokens"],
                "estimated_total_tokens": summary["estimated_total_tokens"],
                "estimated_cost_usd": summary["estimated_cost_usd"],
            }
            construct_manifests.append(_construct_manifest(
                entry, output_paths[entry.construct_id], details, mode=mode, dry_run=True,
                provider=provider, model=model, reasoning_effort=reasoning_effort, runtime_settings=runtime_settings,
                selected_splits=entry_splits(entry),
            ))
            for split, count in summary["records_by_split"].items():
                split_counts[split] = split_counts.get(split, 0) + int(count)
            record_count += int(summary["expected_record_count"])
            request_count += int(summary["request_count"])
            input_tokens += int(summary["estimated_input_tokens"])
            output_tokens += int(summary["estimated_output_tokens"])
        return {
            "schema_version": DOWNSTREAM_MANIFEST_VERSION,
            "manifest_type": "downstream_prompt_generation",
            "status": "dry_run",
            "registry_path": str(Path(registry_path).resolve()),
            "waves": sorted({entry.wave for entry in entries}),
            "construct_ids": [entry.construct_id for entry in entries],
            "scope": "downstream",
            "scope_splits": sorted(active_splits) if active_splits is not None else None,
            "scope_partial": mode == "review" or (active_splits is not None and active_splits != ALL_DOWNSTREAM_SPLITS),
            "run_mode": mode,
            "partial": mode == "review",
            "frozen": False,
            "confirmatory": False,
            "dry_run": True,
            "workers": workers,
            "batch_size": batch_size,
            "provider": provider,
            "requested_model": model,
            "reasoning_effort": reasoning_effort,
            "runtime_settings": runtime_settings,
            "estimated_cost_usd_preflight": estimated_cost,
            "budget_estimate_usd": budget_estimate,
            "max_estimated_cost_usd": max_estimated_cost_usd,
            "constructs": construct_manifests,
            "counts": {
                "split_counts": dict(sorted(split_counts.items())),
                "record_count": record_count,
                "request_count": request_count,
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "estimated_total_tokens": input_tokens + output_tokens,
                "estimated_cost_usd": estimated_cost,
                "budget_estimate_usd": budget_estimate,
            },
            "combined_path": str(combined_path),
            "manifest_path": str(manifest_path),
        }

    if not api_key:
        raise ValueError("An API key is required for non-dry-run generation.")
    if request_fn is None:
        request_fn = call_openai_responses
    output_root.mkdir(parents=True, exist_ok=True)
    if resume and run_state_path.exists():
        prior_state = _load_json_object(run_state_path, label="downstream prompt run state")
        if prior_state.get("run_identity") != run_identity:
            raise ValueError("Cannot resume downstream generation with a different plan/model/runtime identity.")
    elif not resume:
        existing = [path for path in (*output_paths.values(), *checkpoint_paths.values(), combined_path, manifest_path, run_state_path) if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing downstream outputs without --resume: " + ", ".join(map(str, existing)))
        prior_state = None
    else:
        prior_state = None

    checkpoint_identities = {entry.construct_id: _checkpoint_identity(entry, run_identity) for entry in entries}
    records_by_construct: dict[str, tuple[PromptRecord, ...]] = {}
    details_by_construct: dict[str, dict[str, Any]] = {}
    completed_by_construct: dict[str, dict[str, tuple[PromptRecord, ...]]] = {}
    completed_metadata_by_construct: dict[str, dict[str, dict[str, Any]]] = {}
    checkpoint_details_by_construct: dict[str, dict[str, Any]] = {}
    attempts_by_construct: dict[str, dict[str, list[dict[str, Any]]]] = {}
    persisted_attempts = _attempt_history_map(
        (prior_state or {}).get("job_attempts"),
        label="downstream prompt run state job_attempts",
    )
    legacy_attempts = _attempt_history_map(
        (prior_state or {}).get("legacy_attempts"),
        label="downstream prompt run state legacy_attempts",
    )
    for entry in entries:
        checkpoint_path = checkpoint_paths[entry.construct_id]
        checkpoint_attempts: dict[str, list[dict[str, Any]]] = {}
        if resume and checkpoint_path.exists():
            recovered, recovered_metadata, checkpoint_details = _load_checkpoint(
                checkpoint_path,
                entry=entry,
                identity=checkpoint_identities[entry.construct_id],
                mode=mode,
                selected_splits=entry_splits(entry),
            )
            completed_by_construct[entry.construct_id] = recovered
            completed_metadata_by_construct[entry.construct_id] = recovered_metadata
            checkpoint_details_by_construct[entry.construct_id] = checkpoint_details
            checkpoint_attempts = _attempt_history_map(
                checkpoint_details.get("attempts_by_job", {}),
                label=f"checkpoint {entry.construct_id} attempts",
            )
            # Materialize reconstructed accepted-attempt provenance immediately
            # so a later interruption cannot discard it again.
            checkpoint_jobs = {
                job_id: _checkpoint_payload(job_id, job_records, recovered_metadata[job_id])
                for job_id, job_records in recovered.items()
            }
            _write_checkpoint(
                checkpoint_path,
                identity=checkpoint_identities[entry.construct_id],
                jobs=checkpoint_jobs,
                attempts=checkpoint_attempts,
            )
            checkpoint_details["checkpoint_sha256"] = file_sha256(checkpoint_path)
            checkpoint_details["checkpoint_actual_cost_usd"] = _attempt_usage(checkpoint_attempts)["actual_cost_usd"]
            checkpoint_details["checkpoint_attempt_count"] = _attempt_usage(checkpoint_attempts)["attempt_count"]
            checkpoint_details["checkpoint_rejected_attempt_count"] = _attempt_usage(checkpoint_attempts)["rejected_attempt_count"]
        expected_job_ids = {
            job.job_id
            for job in iter_generation_request_jobs(
                entry.plan,
                count_per_model_override=_mode_count(mode),
                splits=set(entry_splits(entry)),
            )
        }
        prior_entry_attempts = {
            job_id: persisted_attempts[job_id]
            for job_id in expected_job_ids
            if job_id in persisted_attempts
        }
        merged_entry_attempts = _merge_attempt_histories(
            prior_entry_attempts,
            checkpoint_attempts,
        )
        if merged_entry_attempts:
            attempts_by_construct[entry.construct_id] = merged_entry_attempts
        output_path = output_paths[entry.construct_id]
        if output_path.exists():
            if not resume:
                raise FileExistsError(f"Output already exists: {output_path}")
            records = tuple(load_prompt_records(output_path))
            details = _validate_downstream_records(
                entry,
                records,
                mode=mode,
                input_price=input_usd_per_million_tokens,
                output_price=output_usd_per_million_tokens,
                selected_splits=entry_splits(entry),
            )
            details["actual_cost_usd"] = _reconstruct_spend(records)
            details["actual_request_count"] = len({record.metadata.get("generation_job_id") for record in records})
            details["resumed"] = True
            details["output_sha256"] = file_sha256(output_path)
            records_by_construct[entry.construct_id] = records
            details_by_construct[entry.construct_id] = details
            continue

    initial_attempts = _merge_attempt_histories(
        persisted_attempts,
        {
            job_id: history
            for construct_history in attempts_by_construct.values()
            for job_id, history in construct_history.items()
        },
    )
    initial_ledger_attempts = _merge_attempt_histories(initial_attempts, legacy_attempts)
    initial_attempt_usage = _attempt_usage(initial_ledger_attempts)
    prior_budget_state = (
        dict(prior_state.get("budget_state", {}))
        if prior_state and isinstance(prior_state.get("budget_state"), Mapping)
        else {}
    )
    prior_spend = 0.0
    for key in ("actual_spent_usd", "outstanding_reserved_usd"):
        prior_spend += float(prior_budget_state.get(key, 0.0) or 0.0)
    prior_attempt_count = max(
        int(
            (prior_state or {}).get(
                "cumulative_attempt_count",
                prior_budget_state.get("reservation_count", 0),
            )
            or 0
        ),
        int(initial_attempt_usage["attempt_count"]),
    )
    prior_rejected_attempt_count = max(
        int((prior_state or {}).get("cumulative_rejected_attempt_count", 0) or 0),
        int(initial_attempt_usage["rejected_attempt_count"]),
    )
    prior_failed_request_count = max(
        int(
            (prior_state or {}).get(
                "cumulative_failed_request_count",
                prior_budget_state.get("failed_request_count", 0),
            )
            or 0
        ),
        0,
    )
    prior_input_tokens = max(
        int((prior_state or {}).get("cumulative_actual_input_tokens", 0) or 0),
        int(initial_attempt_usage["input_tokens"]),
    )
    prior_output_tokens = max(
        int((prior_state or {}).get("cumulative_actual_output_tokens", 0) or 0),
        int(initial_attempt_usage["output_tokens"]),
    )
    prior_total_tokens = max(
        int((prior_state or {}).get("cumulative_actual_total_tokens", 0) or 0),
        int(initial_attempt_usage["total_tokens"]),
    )
    prior_known_attempt_count = int(initial_attempt_usage["attempt_count"])
    prior_unattributed_attempt_count = max(0, prior_attempt_count - prior_known_attempt_count)
    prior_unattributed_spend_usd = max(
        0.0,
        prior_spend - float(initial_attempt_usage["actual_cost_usd"]),
    )
    baseline_attempt_keys = _attempt_keys(initial_attempts)
    initial_spend = max(
        prior_spend,
        sum(float(details.get("actual_cost_usd", 0.0) or 0.0) for details in details_by_construct.values()),
        sum(float(details.get("checkpoint_actual_cost_usd", 0.0) or 0.0) for details in checkpoint_details_by_construct.values()),
    )
    if initial_spend > float(max_estimated_cost_usd) + 1e-12:
        raise RuntimeBudgetExceeded("Resumed downstream generation has already reached its spending cap.")
    state_lock = threading.Lock()
    run_state: dict[str, Any] = {
        "schema_version": DOWNSTREAM_MANIFEST_VERSION,
        "status": "running",
        "run_identity": run_identity,
        "quality_gate": quality_gate,
        "max_estimated_cost_usd": max_estimated_cost_usd,
        "estimated_cost_usd": estimated_cost,
        "budget_estimate_usd": budget_estimate,
        "completed_construct_ids": sorted(records_by_construct),
        "cumulative_actual_spent_usd": initial_spend,
        "cumulative_attempt_count": prior_attempt_count,
        "cumulative_rejected_attempt_count": prior_rejected_attempt_count,
        "cumulative_failed_request_count": prior_failed_request_count,
        "cumulative_actual_input_tokens": prior_input_tokens,
        "cumulative_actual_output_tokens": prior_output_tokens,
        "cumulative_actual_total_tokens": prior_total_tokens,
        "prior_unattributed_attempt_count": prior_unattributed_attempt_count,
        "prior_unattributed_spend_usd": prior_unattributed_spend_usd,
        "job_attempts": {
            str(job_id): [dict(attempt) for attempt in history]
            for job_id, history in initial_attempts.items()
        },
        "legacy_attempts": {
            str(job_id): [dict(attempt) for attempt in history]
            for job_id, history in legacy_attempts.items()
        },
    }
    if prior_state:
        run_state.update({key: value for key, value in prior_state.items() if key not in {"status", "error"}})
        run_state["job_attempts"] = {
            str(job_id): [dict(attempt) for attempt in history]
            for job_id, history in initial_attempts.items()
        }
        run_state["legacy_attempts"] = {
            str(job_id): [dict(attempt) for attempt in history]
            for job_id, history in legacy_attempts.items()
        }
        run_state["cumulative_actual_spent_usd"] = initial_spend
        run_state["cumulative_attempt_count"] = prior_attempt_count
        run_state["cumulative_rejected_attempt_count"] = prior_rejected_attempt_count
        run_state["cumulative_failed_request_count"] = prior_failed_request_count
        run_state["cumulative_actual_input_tokens"] = prior_input_tokens
        run_state["cumulative_actual_output_tokens"] = prior_output_tokens
        run_state["cumulative_actual_total_tokens"] = prior_total_tokens
        run_state["prior_unattributed_attempt_count"] = prior_unattributed_attempt_count
        run_state["prior_unattributed_spend_usd"] = prior_unattributed_spend_usd
    _atomic_write_json(run_state, run_state_path)
    runtime_budget = RuntimeBudget(
        max_budget_usd=float(max_estimated_cost_usd),
        input_usd_per_million_tokens=float(input_usd_per_million_tokens),
        output_usd_per_million_tokens=float(output_usd_per_million_tokens),
        initial_spent_usd=initial_spend,
        on_change=lambda snapshot: _persist_budget(run_state, run_state_path, state_lock, snapshot),
    )

    def all_attempts() -> dict[str, list[dict[str, Any]]]:
        return _merge_attempt_histories(
            {
                job_id: history
                for construct_history in attempts_by_construct.values()
                for job_id, history in construct_history.items()
            }
        )

    def update_cumulative_accounting(snapshot: Mapping[str, Any]) -> None:
        delta_usage = _attempt_usage(_attempt_delta(all_attempts(), baseline_attempt_keys))
        cumulative_recorded_attempt_cost = (
            float(initial_attempt_usage["actual_cost_usd"])
            + float(delta_usage["actual_cost_usd"])
        )
        cumulative_attempt_count = prior_attempt_count + int(snapshot["reservation_count"])
        run_state["cumulative_actual_spent_usd"] = float(snapshot["actual_spent_usd"])
        run_state["cumulative_attempt_count"] = cumulative_attempt_count
        run_state["cumulative_rejected_attempt_count"] = (
            prior_rejected_attempt_count + int(delta_usage["rejected_attempt_count"])
        )
        run_state["cumulative_failed_request_count"] = (
            prior_failed_request_count + int(snapshot["failed_request_count"])
        )
        run_state["cumulative_actual_input_tokens"] = (
            prior_input_tokens + int(delta_usage["input_tokens"])
        )
        run_state["cumulative_actual_output_tokens"] = (
            prior_output_tokens + int(delta_usage["output_tokens"])
        )
        run_state["cumulative_actual_total_tokens"] = (
            prior_total_tokens + int(delta_usage["total_tokens"])
        )
        run_state["cumulative_recorded_attempt_cost_usd"] = cumulative_recorded_attempt_cost
        run_state["unattributed_attempt_count"] = max(
            0,
            cumulative_attempt_count
            - int(initial_attempt_usage["attempt_count"])
            - int(delta_usage["attempt_count"]),
        )
        run_state["unattributed_spend_usd"] = max(
            0.0,
            float(snapshot["actual_spent_usd"]) - cumulative_recorded_attempt_cost,
        )

    new_job_limit = NewJobLimit(max_new_jobs)
    runtime_request_fn = _request_with_runtime_budget(
        _request_with_runtime_options(
            request_fn,
            provider=provider,
            requested_model=model,
            reasoning_effort=reasoning_effort,
            input_usd_per_million_tokens=float(input_usd_per_million_tokens),
            output_usd_per_million_tokens=float(output_usd_per_million_tokens),
        ),
        runtime_budget,
    )
    pending = [entry for entry in entries if entry.construct_id not in records_by_construct]
    # ``workers`` is a global request-concurrency budget.  Allocate at least
    # one request worker to every pending construct, then distribute any
    # remaining capacity deterministically.  This avoids accidentally
    # multiplying concurrency when the outer construct pool and the inner
    # request pool are both active, while still allowing a single-construct
    # recovery run to issue four substantial requests in parallel.
    request_workers_by_construct = _allocate_request_workers(pending, workers)

    def generate_one(entry: _PlanEntry) -> tuple[str, tuple[PromptRecord, ...], dict[str, Any]]:
        checkpoint_jobs = {
            job_id: _checkpoint_payload(job_id, job_records, completed_metadata_by_construct[entry.construct_id][job_id])
            for job_id, job_records in completed_by_construct.get(entry.construct_id, {}).items()
        }
        checkpoint_path = checkpoint_paths[entry.construct_id]
        identity = checkpoint_identities[entry.construct_id]
        construct_attempts = attempts_by_construct.setdefault(entry.construct_id, {})
        checkpoint_lock = threading.Lock()

        def on_job_attempt(
            job: Any,
            attempt_number: int,
            metadata: Mapping[str, Any],
            rejection_reason: str | None,
        ) -> None:
            with checkpoint_lock:
                history = construct_attempts.setdefault(job.job_id, [])
                history.append({
                    "attempt": attempt_number,
                    "status": "rejected" if rejection_reason is not None else "accepted",
                    "rejection_reason": rejection_reason,
                    "response_metadata": dict(metadata),
                })
                _write_checkpoint(
                    checkpoint_path,
                    identity=identity,
                    jobs=checkpoint_jobs,
                    attempts=construct_attempts,
                )
                with state_lock:
                    state_attempts = run_state.setdefault("job_attempts", {})
                    state_attempts[job.job_id] = [dict(item) for item in history]
                    budget_snapshot = runtime_budget.snapshot()
                    run_state["budget_state"] = budget_snapshot
                    update_cumulative_accounting(budget_snapshot)
                    _atomic_write_json(run_state, run_state_path)

        def on_job_complete(job: Any, job_records: tuple[PromptRecord, ...], metadata: dict[str, Any]) -> None:
            with checkpoint_lock:
                checkpoint_jobs[job.job_id] = _checkpoint_payload(job.job_id, job_records, metadata)
                _write_checkpoint(
                    checkpoint_path,
                    identity=identity,
                    jobs=checkpoint_jobs,
                    attempts=construct_attempts,
                )
                if max_new_jobs is not None:
                    new_job_limit.complete(job.job_id, metadata)

        result = generate_prompt_records(
            entry.plan,
            entry.spec,
            api_key=api_key,
            request_fn=_request_fn_with_collateral_label_exception(runtime_request_fn, entry),
            workers=request_workers_by_construct[entry.construct_id],
            count_per_model_override=_mode_count(mode),
            splits=set(entry_splits(entry)),
            completed_job_records=completed_by_construct.get(entry.construct_id),
            completed_job_metadata=completed_metadata_by_construct.get(entry.construct_id),
            on_job_complete=on_job_complete,
            before_job_request=new_job_limit.reserve if max_new_jobs is not None else None,
            # Keep a bounded deterministic recovery budget for a batch that is
            # rejected by the local contract validator.  The normal path is
            # one accepted request; the sixth total attempt is reserved for
            # recovery after a validator implementation fix and never changes
            # the registered categories or task design.
            semantic_retry_limit=DOWNSTREAM_SEMANTIC_RETRY_LIMIT,
            semantic_attempt_history=construct_attempts,
            on_job_attempt=on_job_attempt,
            transport_options={"timeout": float(timeout_seconds)},
            job_records_validator=lambda job, job_records: _validate_downstream_job_records(
                entry,
                job,
                job_records,
            ),
        )
        records = tuple(result.records)
        details = _validate_downstream_records(
            entry,
            records,
            mode=mode,
            input_price=input_usd_per_million_tokens,
            output_price=output_usd_per_million_tokens,
            selected_splits=entry_splits(entry),
        )
        summary = result.summary()
        details.update({
            "actual_input_tokens": summary.get("actual_input_tokens", 0),
            "actual_output_tokens": summary.get("actual_output_tokens", 0),
            "actual_total_tokens": summary.get("actual_total_tokens", 0),
            "actual_cost_usd": summary.get("actual_cost_usd", 0.0),
            "attempt_count": sum(len(history) for history in construct_attempts.values()),
            "rejected_attempt_count": sum(
                sum(item.get("status") == "rejected" for item in history)
                for history in construct_attempts.values()
            ),
            "job_attempts": {job_id: [dict(item) for item in history] for job_id, history in construct_attempts.items()},
            "resumed": bool(completed_by_construct.get(entry.construct_id)),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path.exists() else None,
            "checkpoint_job_count": len(checkpoint_jobs),
        })
        return entry.construct_id, records, details

    try:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as executor:
            futures = {executor.submit(generate_one, entry): entry for entry in pending}
            for future in as_completed(futures):
                construct_id, records, details = future.result()
                records_by_construct[construct_id] = records
                details_by_construct[construct_id] = details
                with state_lock:
                    run_state["completed_construct_ids"] = sorted(records_by_construct)
                    budget_snapshot = runtime_budget.snapshot()
                    run_state["budget_state"] = budget_snapshot
                    update_cumulative_accounting(budget_snapshot)
                    _atomic_write_json(run_state, run_state_path)
    except GenerationPaused as exc:
        budget_snapshot = runtime_budget.snapshot()
        with state_lock:
            run_state["status"] = "paused"
            run_state["pause_reason"] = "max_new_jobs"
            run_state["pause_message"] = str(exc)
            run_state["budget_state"] = budget_snapshot
            update_cumulative_accounting(budget_snapshot)
            _atomic_write_json(run_state, run_state_path)
        return {
            "schema_version": DOWNSTREAM_MANIFEST_VERSION,
            "manifest_type": "downstream_prompt_generation",
            "status": "paused",
            "partial": True,
            "frozen": False,
            "dry_run": False,
            "run_mode": mode,
            "construct_ids": [entry.construct_id for entry in entries],
            "progress": {
                "completed_construct_ids": sorted(records_by_construct),
                "budget_state": runtime_budget.snapshot(),
                "new_job_limit": new_job_limit.snapshot(),
            },
            "run_state_path": str(run_state_path),
            "manifest_path": str(manifest_path),
        }
    except Exception as exc:
        budget_snapshot = runtime_budget.snapshot()
        with state_lock:
            run_state["status"] = "failed"
            run_state["error"] = f"{type(exc).__name__}: {exc}"
            run_state["completed_construct_ids"] = sorted(records_by_construct)
            run_state["budget_state"] = budget_snapshot
            update_cumulative_accounting(budget_snapshot)
            _atomic_write_json(run_state, run_state_path)
        raise

    # A complete generation is materialized and audited atomically at the
    # combined-inventory level.  No final manifest is written as frozen if the
    # severe audit gate fails.
    combined_records: list[PromptRecord] = []
    construct_manifests: list[dict[str, Any]] = []
    all_specs = {entry.construct_id: entry.spec for entry in entries}
    for entry in entries:
        records = records_by_construct[entry.construct_id]
        details = details_by_construct[entry.construct_id]
        entry_attempts = attempts_by_construct.get(entry.construct_id, {})
        entry_attempt_usage = _attempt_usage(entry_attempts)
        details.update({
            "actual_input_tokens": entry_attempt_usage["input_tokens"],
            "actual_output_tokens": entry_attempt_usage["output_tokens"],
            "actual_total_tokens": entry_attempt_usage["total_tokens"],
            "actual_cost_usd": entry_attempt_usage["actual_cost_usd"],
            "attempted_cost_usd": entry_attempt_usage["actual_cost_usd"],
            "accepted_record_cost_usd": _reconstruct_spend(records),
            "attempt_count": entry_attempt_usage["attempt_count"],
            "rejected_attempt_count": entry_attempt_usage["rejected_attempt_count"],
            "job_attempts": {
                job_id: [dict(attempt) for attempt in history]
                for job_id, history in entry_attempts.items()
            },
            "checkpoint_path": str(checkpoint_paths[entry.construct_id])
            if checkpoint_paths[entry.construct_id].exists()
            else details.get("checkpoint_path"),
            "checkpoint_sha256": file_sha256(checkpoint_paths[entry.construct_id])
            if checkpoint_paths[entry.construct_id].exists()
            else details.get("checkpoint_sha256"),
            "checkpoint_job_count": len(completed_by_construct.get(entry.construct_id, {})),
        })
        output_path = output_paths[entry.construct_id]
        _atomic_write_records(records, output_path)
        details["output_sha256"] = file_sha256(output_path)
        combined_records.extend(records)
        construct_manifests.append(_construct_manifest(
            entry, output_path, details, mode=mode, dry_run=False,
            provider=provider, model=model, reasoning_effort=reasoning_effort, runtime_settings=runtime_settings,
            selected_splits=entry_splits(entry),
        ))
    validate_prompt_records(combined_records, all_specs, require_all_splits=False)
    _atomic_write_records(combined_records, combined_path)
    audit = audit_downstream_inventory(combined_records, entries, vector_reference=vector_path)
    _atomic_write_json({key: value for key, value in audit.items() if key != "flags"}, audit_summary_path)
    flag_fields = sorted({key for flag in audit["flags"] for key in flag})
    with audit_flags_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flag_fields or ["severity", "flag_type"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit["flags"])
    if not audit["passed"]:
        with state_lock:
            run_state["status"] = "audit_failed"
            run_state["audit_summary_path"] = str(audit_summary_path)
            run_state["audit_flags_path"] = str(audit_flags_path)
            run_state["budget_state"] = runtime_budget.snapshot()
            _atomic_write_json(run_state, run_state_path)
        manifest = {
            "schema_version": DOWNSTREAM_MANIFEST_VERSION,
            "manifest_type": "downstream_prompt_generation",
            "status": "audit_failed",
            "run_mode": mode,
            "partial": mode == "review",
            "frozen": False,
            "dry_run": False,
            "construct_ids": [entry.construct_id for entry in entries],
            "scope_splits": sorted(active_splits) if active_splits is not None else None,
            "scope_partial": mode == "review" or (active_splits is not None and active_splits != ALL_DOWNSTREAM_SPLITS),
            "constructs": construct_manifests,
            "audit": {key: value for key, value in audit.items() if key != "flags"},
            "audit_summary_path": str(audit_summary_path),
            "audit_flags_path": str(audit_flags_path),
            "combined_path": str(combined_path),
            "combined_sha256": file_sha256(combined_path),
            "run_state_path": str(run_state_path),
        }
        _atomic_write_json(manifest, output_root / "downstream_prompt_manifest.json")
        raise ValueError(f"Downstream prompt audit failed with {audit['severe_flag_count']} severe flag(s).")

    budget_snapshot = runtime_budget.snapshot()
    final_attempts = all_attempts()
    new_attempts = _attempt_delta(final_attempts, baseline_attempt_keys)
    new_attempt_usage = _attempt_usage(new_attempts)
    final_ledger_attempts = _merge_attempt_histories(initial_ledger_attempts, new_attempts)
    final_attempt_usage = _attempt_usage(final_ledger_attempts)
    # A rejected semantic attempt is part of cumulative API spend but has no
    # materialized prompt records.  Reconstruct this separate accepted-record
    # total from the canonical rows rather than from request summaries, which
    # intentionally include rejected attempts.
    materialized_record_cost = _reconstruct_spend(combined_records)
    new_attempt_count = int(budget_snapshot["reservation_count"])
    new_rejected_attempt_count = int(new_attempt_usage["rejected_attempt_count"])
    new_failed_request_count = int(budget_snapshot["failed_request_count"])
    cumulative_attempt_count = prior_attempt_count + new_attempt_count
    cumulative_rejected_attempt_count = prior_rejected_attempt_count + new_rejected_attempt_count
    cumulative_failed_request_count = prior_failed_request_count + new_failed_request_count
    cumulative_input_tokens = prior_input_tokens + int(new_attempt_usage["input_tokens"])
    cumulative_output_tokens = prior_output_tokens + int(new_attempt_usage["output_tokens"])
    cumulative_total_tokens = prior_total_tokens + int(new_attempt_usage["total_tokens"])
    cumulative_recorded_attempt_cost = (
        float(initial_attempt_usage["actual_cost_usd"])
        + float(new_attempt_usage["actual_cost_usd"])
    )
    unattributed_attempt_count = max(
        0,
        cumulative_attempt_count - int(final_attempt_usage["attempt_count"]),
    )
    unattributed_spend_usd = max(
        0.0,
        float(budget_snapshot["actual_spent_usd"]) - cumulative_recorded_attempt_cost,
    )
    manifest = {
        "schema_version": DOWNSTREAM_MANIFEST_VERSION,
        "manifest_type": "downstream_prompt_generation",
        "status": "frozen" if mode == "full" else "complete_review",
        "registry_path": str(Path(registry_path).resolve()),
        "waves": sorted({entry.wave for entry in entries}),
        "construct_ids": [entry.construct_id for entry in entries],
        "scope": "downstream",
        "scope_splits": sorted(active_splits) if active_splits is not None else None,
        "scope_partial": mode == "review" or (active_splits is not None and active_splits != ALL_DOWNSTREAM_SPLITS),
        "run_mode": mode,
        "partial": mode == "review",
        "frozen": mode == "full",
        "confirmatory": False,
        "dry_run": False,
        "workers": workers,
        "batch_size": batch_size,
        "provider": provider,
        "requested_model": model,
        "reasoning_effort": reasoning_effort,
        "runtime_settings": runtime_settings,
        "max_estimated_cost_usd": max_estimated_cost_usd,
        "estimated_cost_usd_preflight": estimated_cost,
        "budget_estimate_usd": budget_estimate,
        "runtime_budget": budget_snapshot,
        "accounting": {
            "prior_actual_spent_usd": initial_spend,
            "new_actual_spent_usd": max(0.0, budget_snapshot["actual_spent_usd"] - initial_spend),
            "cumulative_actual_spent_usd": budget_snapshot["actual_spent_usd"],
            "materialized_record_cost_usd": materialized_record_cost,
            "prior_attempt_count": prior_attempt_count,
            "new_attempt_count": new_attempt_count,
            "cumulative_attempt_count": cumulative_attempt_count,
            "prior_rejected_attempt_count": prior_rejected_attempt_count,
            "new_rejected_attempt_count": new_rejected_attempt_count,
            "cumulative_rejected_attempt_count": cumulative_rejected_attempt_count,
            "prior_failed_request_count": prior_failed_request_count,
            "new_failed_request_count": new_failed_request_count,
            "cumulative_failed_request_count": cumulative_failed_request_count,
            "cumulative_actual_input_tokens": cumulative_input_tokens,
            "cumulative_actual_output_tokens": cumulative_output_tokens,
            "cumulative_actual_total_tokens": cumulative_total_tokens,
            "unattributed_attempt_count": unattributed_attempt_count,
            "unattributed_spend_usd": unattributed_spend_usd,
        },
        "quality_gate": quality_gate,
        "constructs": construct_manifests,
        "audit": {key: value for key, value in audit.items() if key != "flags"},
        "audit_summary_path": str(audit_summary_path),
        "audit_flags_path": str(audit_flags_path),
        "counts": {
            "split_counts": dict(sorted({
                split: sum(1 for record in combined_records if record.split == split)
                for split in sorted({record.split for record in combined_records})
            }.items())),
            "record_count": len(combined_records),
            "request_count": sum(item["request_count"] for item in construct_manifests),
            "estimated_input_tokens": sum(item["estimated_input_tokens"] for item in construct_manifests),
            "estimated_output_tokens": sum(item["estimated_output_tokens"] for item in construct_manifests),
            "estimated_total_tokens": sum(item["estimated_total_tokens"] for item in construct_manifests),
            "estimated_cost_usd": estimated_cost,
            "actual_input_tokens": cumulative_input_tokens,
            "actual_output_tokens": cumulative_output_tokens,
            "actual_total_tokens": cumulative_total_tokens,
            "actual_cost_usd": budget_snapshot["actual_spent_usd"],
            "materialized_record_cost_usd": materialized_record_cost,
            "prior_actual_cost_usd": initial_spend,
            "new_actual_cost_usd": max(0.0, budget_snapshot["actual_spent_usd"] - initial_spend),
            "attempt_count": cumulative_attempt_count,
            "prior_attempt_count": prior_attempt_count,
            "new_attempt_count": new_attempt_count,
            "rejected_attempt_count": cumulative_rejected_attempt_count,
            "prior_rejected_attempt_count": prior_rejected_attempt_count,
            "new_rejected_attempt_count": new_rejected_attempt_count,
            "failed_request_count": cumulative_failed_request_count,
        },
        "combined_path": str(combined_path),
        "combined_sha256": file_sha256(combined_path),
        "manifest_path": str(manifest_path),
    }
    _atomic_write_json(manifest, manifest_path)
    with state_lock:
        run_state["status"] = manifest["status"]
        run_state["manifest_path"] = str(manifest_path)
        run_state["audit_summary_path"] = str(audit_summary_path)
        run_state["audit_flags_path"] = str(audit_flags_path)
        run_state["budget_state"] = budget_snapshot
        run_state["cumulative_actual_spent_usd"] = budget_snapshot["actual_spent_usd"]
        run_state["cumulative_attempt_count"] = cumulative_attempt_count
        run_state["cumulative_rejected_attempt_count"] = cumulative_rejected_attempt_count
        run_state["cumulative_failed_request_count"] = cumulative_failed_request_count
        run_state["cumulative_actual_input_tokens"] = cumulative_input_tokens
        run_state["cumulative_actual_output_tokens"] = cumulative_output_tokens
        run_state["cumulative_actual_total_tokens"] = cumulative_total_tokens
        run_state["cumulative_recorded_attempt_cost_usd"] = cumulative_recorded_attempt_cost
        run_state["unattributed_attempt_count"] = unattributed_attempt_count
        run_state["unattributed_spend_usd"] = unattributed_spend_usd
        _atomic_write_json(run_state, run_state_path)
    return manifest


def _persist_budget(run_state: dict[str, Any], path: Path, lock: threading.Lock, snapshot: Mapping[str, Any]) -> None:
    with lock:
        run_state["budget_state"] = dict(snapshot)
        _atomic_write_json(run_state, path)


def _parse_waves(values: list[str]) -> list[int | str]:
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate downstream behavior, steering, and calibration prompts.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--waves", nargs="+", default=["all"])
    parser.add_argument("--constructs", nargs="+", default=None)
    parser.add_argument("--mode", choices=("review", "full"), default="full")
    parser.add_argument(
        "--selected-splits",
        nargs="+",
        choices=sorted(ALL_DOWNSTREAM_SPLITS),
        default=None,
        help="Generate only these registered downstream partitions; use for versioned component repairs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--provider",
        choices=("openai",),
        default=DEFAULT_PROVIDER,
        help="The downstream workflow is intentionally restricted to OpenAI GPT-5.6 Luna.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--max-new-jobs", type=int, default=None)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=None)
    parser.add_argument("--input-price-usd-per-million", type=float, default=None)
    parser.add_argument("--output-price-usd-per-million", type=float, default=None)
    parser.add_argument("--cost-safety-multiplier", type=float, default=DEFAULT_COST_SAFETY_MULTIPLIER)
    parser.add_argument("--vector-reference", type=Path, default=DEFAULT_VECTOR_REFERENCE)
    parser.add_argument("--skip-vector-reference", action="store_true")
    parser.add_argument("--quality-gate-file", type=Path, default=None)
    args = parser.parse_args()
    api_env = args.api_key_env or "OPENAI_API_KEY"
    manifest = orchestrate_downstream_prompts(
        registry_path=args.registry,
        waves=_parse_waves(args.waves),
        construct_ids=args.constructs,
        output_dir=args.output_dir,
        mode=args.mode,
        selected_splits=args.selected_splits,
        workers=args.workers,
        batch_size=args.batch_size,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        resume=args.resume,
        dry_run=args.dry_run,
        api_key=None if args.dry_run else os.environ.get(api_env),
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_new_jobs=args.max_new_jobs,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
        input_usd_per_million_tokens=args.input_price_usd_per_million,
        output_usd_per_million_tokens=args.output_price_usd_per_million,
        cost_safety_multiplier=args.cost_safety_multiplier,
        vector_reference=None if args.skip_vector_reference else args.vector_reference,
        quality_gate_file=args.quality_gate_file,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
