#!/usr/bin/env python3
"""Generate the frozen train/validation/held-out prompt vectors for registry waves.

This entrypoint is deliberately narrower than ``generate_construct_prompts``:
it discovers construct plans from the versioned registry, emits only the three
paired direction splits, and writes a combined inventory only after every
selected construct has completed successfully.  The public orchestration
function accepts an injected request function so deterministic tests can run
without network access.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import call_openrouter_chat_completion  # noqa: E402
from construct_benchmark.config import load_construct_spec  # noqa: E402
from construct_benchmark.generation import (  # noqa: E402
    RequestFn,
    dry_run_summary,
    generate_prompt_records,
    iter_generation_request_jobs,
    load_generation_plan,
    resolve_generation_mode,
    write_generation_result,
)
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402
from construct_benchmark.prompts import PromptRecord, load_prompt_records, validate_prompt_records  # noqa: E402
from construct_benchmark.registry import ConstructRegistry, load_construct_registry  # noqa: E402


VECTOR_SPLITS = frozenset({"direction_train", "direction_validation", "direction_heldout"})
DEFAULT_REGISTRY = _ROOT / "configs/construct_benchmark/construct_registry_v1.json"
DEFAULT_OUTPUT_DIR = _ROOT / "results/benchmark/vector_prompts_v1/prompts"
# The active benchmark workflow uses the OpenAI Responses API and Luna.  The
# legacy OpenRouter transport remains available only when an explicit
# ``--provider openrouter`` override is supplied for historical reproduction.
DEFAULT_PROVIDER = "openai"
DEFAULT_LUNA_MODEL = "gpt-5.6-luna"
# These are the published Luna prices used for preflight accounting.  They are
# intentionally explicit here instead of silently inheriting a provider's
# current price table.  Callers can override them for a different model.
LUNA_INPUT_USD_PER_MILLION_TOKENS = 0.20
LUNA_OUTPUT_USD_PER_MILLION_TOKENS = 1.20
COST_SAFETY_MULTIPLIER = 1.25
QUALITY_GATE_VERSION = "1"


class RuntimeBudgetExceeded(RuntimeError):
    """Raised before a request when its worst-case reservation would exceed the cap."""


class GenerationPaused(RuntimeError):
    """Raised internally when an invocation-level staged limit is reached."""


class NewJobLimit:
    """Thread-safe cap on newly completed request checkpoints for one invocation."""

    def __init__(self, max_new_jobs: int | None) -> None:
        if max_new_jobs is not None and max_new_jobs < 0:
            raise ValueError("max_new_jobs must be non-negative.")
        self.max_new_jobs = max_new_jobs
        self._completed = 0
        self._reserved = 0
        self._spent_usd = 0.0
        self._lock = threading.Lock()

    def reserve(self, job: Any) -> None:
        """Reserve one new job before its API request is submitted."""

        del job
        if self.max_new_jobs is None:
            return
        with self._lock:
            if self._completed + self._reserved >= self.max_new_jobs:
                raise GenerationPaused(
                    f"Invocation max_new_jobs={self.max_new_jobs} reached; "
                    "resume without (or with a larger) invocation limit to continue."
                )
            self._reserved += 1

    def complete(self, job_id: str, response_metadata: Mapping[str, Any]) -> None:
        """Record a job only after its durable checkpoint has been written."""

        del job_id
        if self.max_new_jobs is None:
            return
        raw_cost = response_metadata.get("actual_cost_usd", 0.0)
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise ValueError("New-job limit requires numeric actual_cost_usd provenance.")
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("New-job limit received invalid actual_cost_usd provenance.")
        with self._lock:
            if self._reserved <= 0:
                raise RuntimeError("New-job limit completed a job without a reservation.")
            self._reserved -= 1
            self._completed += 1
            self._spent_usd += cost

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_new_jobs": self.max_new_jobs,
                "new_jobs_completed": self._completed,
                "new_jobs_reserved": self._reserved,
                "new_spend_usd": self._spent_usd,
            }


@dataclass
class _BudgetReservation:
    amount_usd: float
    released: bool = False


class RuntimeBudget:
    """Thread-safe request-level spending guard.

    The preflight estimate is useful for planning but cannot protect a live
    concurrent run from an unusually long response.  This guard reserves the
    worst-case cost of each request before it is sent, then settles that
    reservation against the transport's measured cost.  Missing transport
    cost metadata is charged at the reservation amount so an unpriced response
    can never make the guard under-count spending.
    """

    def __init__(
        self,
        *,
        max_budget_usd: float,
        input_usd_per_million_tokens: float,
        output_usd_per_million_tokens: float,
        initial_spent_usd: float = 0.0,
        on_change: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not math.isfinite(max_budget_usd) or max_budget_usd < 0:
            raise ValueError("max_budget_usd must be a finite non-negative number.")
        if not math.isfinite(input_usd_per_million_tokens) or input_usd_per_million_tokens < 0:
            raise ValueError("input_usd_per_million_tokens must be finite and non-negative.")
        if not math.isfinite(output_usd_per_million_tokens) or output_usd_per_million_tokens < 0:
            raise ValueError("output_usd_per_million_tokens must be finite and non-negative.")
        if not math.isfinite(initial_spent_usd) or initial_spent_usd < 0:
            raise ValueError("initial_spent_usd must be finite and non-negative.")
        self.max_budget_usd = float(max_budget_usd)
        self.input_usd_per_million_tokens = float(input_usd_per_million_tokens)
        self.output_usd_per_million_tokens = float(output_usd_per_million_tokens)
        self._actual_spent_usd = float(initial_spent_usd)
        self._outstanding_reserved_usd = 0.0
        self._completed_request_count = 0
        self._failed_request_count = 0
        self._reservation_count = 0
        self._unpriced_response_count = 0
        self._lock = threading.Lock()
        self._on_change = on_change

    @staticmethod
    def _finite_number(value: Any, *, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeBudgetExceeded(f"Runtime budget requires a finite numeric {field_name}.")
        if float(value) < 0:
            raise RuntimeBudgetExceeded(f"Runtime budget requires a non-negative {field_name}.")
        return float(value)

    def request_reservation_cost(self, options: Mapping[str, Any]) -> float:
        """Compute a request's conservative maximum cost from runtime options."""

        max_output_value = options.get("max_output_tokens", options.get("max_tokens"))
        estimated_input_value = options.get("estimated_input_tokens_per_request")
        max_output_tokens = self._finite_number(max_output_value, field_name="max output token limit")
        estimated_input_tokens = self._finite_number(
            estimated_input_value,
            field_name="estimated input token count",
        )
        return (
            estimated_input_tokens * self.input_usd_per_million_tokens
            + max_output_tokens * self.output_usd_per_million_tokens
        ) / 1_000_000.0

    def _notify(self) -> None:
        callback = self._on_change
        if callback is not None:
            callback(self.snapshot())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            actual = self._actual_spent_usd
            outstanding = self._outstanding_reserved_usd
            return {
                "max_budget_usd": self.max_budget_usd,
                "actual_spent_usd": actual,
                "outstanding_reserved_usd": outstanding,
                "reserved_plus_spent_usd": actual + outstanding,
                "remaining_budget_usd": self.max_budget_usd - actual - outstanding,
                "completed_request_count": self._completed_request_count,
                "failed_request_count": self._failed_request_count,
                "reservation_count": self._reservation_count,
                "unpriced_response_count": self._unpriced_response_count,
            }

    def reserve(self, options: Mapping[str, Any]) -> _BudgetReservation:
        amount_usd = self.request_reservation_cost(options)
        with self._lock:
            committed_plus_reserved = self._actual_spent_usd + self._outstanding_reserved_usd
            if committed_plus_reserved + amount_usd > self.max_budget_usd + 1e-12:
                raise RuntimeBudgetExceeded(
                    "Runtime budget would be exceeded before request: "
                    f"spent=${self._actual_spent_usd:.6f}, "
                    f"outstanding=${self._outstanding_reserved_usd:.6f}, "
                    f"reservation=${amount_usd:.6f}, "
                    f"cap=${self.max_budget_usd:.6f}."
                )
            self._outstanding_reserved_usd += amount_usd
            self._reservation_count += 1
            reservation = _BudgetReservation(amount_usd=amount_usd)
        self._notify()
        return reservation

    @staticmethod
    def _response_cost(response: Mapping[str, Any]) -> float | None:
        metadata = response.get("_generation_metadata")
        if not isinstance(metadata, Mapping):
            return None
        raw_cost = metadata.get("actual_cost_usd")
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            return None
        value = float(raw_cost)
        if not math.isfinite(value) or value < 0:
            return None
        return value

    def settle(self, reservation: _BudgetReservation, response: Mapping[str, Any]) -> dict[str, Any]:
        """Release a reservation and record measured (or conservative fallback) cost."""

        if reservation.released:
            raise RuntimeError("Runtime budget reservation was settled more than once.")
        actual_cost = self._response_cost(response)
        unpriced = actual_cost is None
        if actual_cost is None:
            actual_cost = reservation.amount_usd
        with self._lock:
            self._outstanding_reserved_usd -= reservation.amount_usd
            if self._outstanding_reserved_usd < 0 and self._outstanding_reserved_usd > -1e-12:
                self._outstanding_reserved_usd = 0.0
            self._actual_spent_usd += actual_cost
            self._completed_request_count += 1
            if unpriced:
                self._unpriced_response_count += 1
            reservation.released = True
        self._notify()

        # Preserve a reconstructable cost in canonical CSV metadata even when
        # a test/custom transport omits _generation_metadata.  This charged
        # fallback is intentionally conservative.
        if unpriced:
            normalized = dict(response)
            metadata = response.get("_generation_metadata")
            normalized_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            normalized_metadata["actual_cost_usd"] = actual_cost
            normalized_metadata["runtime_budget_cost_fallback"] = True
            normalized["_generation_metadata"] = normalized_metadata
            return normalized
        return dict(response)

    def release_on_error(self, reservation: _BudgetReservation) -> None:
        """Release a reservation when the transport raises before a response."""

        if reservation.released:
            return
        with self._lock:
            self._outstanding_reserved_usd -= reservation.amount_usd
            if self._outstanding_reserved_usd < 0 and self._outstanding_reserved_usd > -1e-12:
                self._outstanding_reserved_usd = 0.0
            self._failed_request_count += 1
            reservation.released = True
        self._notify()


@dataclass(frozen=True)
class VectorPlanEntry:
    """One registry entry plus its validated scientific artifacts."""

    construct_id: str
    wave: int
    spec_path: Path
    plan_path: Path
    spec: Any
    plan: dict[str, Any]
    spec_sha256: str
    plan_sha256: str
    models: tuple[dict[str, str], ...]
    # The hash of the checked-in plan.  ``plan_sha256`` may describe an
    # explicit CLI model override while this remains the scientific source
    # artifact used by the quality gate.
    source_plan_sha256: str | None = None


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _raw_registry_entries(registry_path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{registry_path} is not valid JSON.") from exc
    raw_entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(raw_entries, list):
        raise ValueError(f"{registry_path} must contain an entries list.")
    result: dict[str, dict[str, Any]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping) or not isinstance(raw_entry.get("construct_id"), str):
            raise ValueError(f"{registry_path} contains a malformed registry entry.")
        construct_id = str(raw_entry["construct_id"])
        if construct_id in result:
            raise ValueError(f"{registry_path} contains duplicate construct_id={construct_id!r}.")
        result[construct_id] = dict(raw_entry)
    return result


def _selected_wave_set(waves: Iterable[int | str] | None, registry: ConstructRegistry) -> set[int]:
    if waves is None:
        return {entry.wave for entry in registry.entries}
    values = [str(value).strip().lower() for value in waves]
    if not values or "all" in values:
        if len(values) > 1:
            raise ValueError("--waves accepts 'all' or one or more wave numbers, not both.")
        return {entry.wave for entry in registry.entries}
    try:
        selected = {int(value) for value in values}
    except ValueError as exc:
        raise ValueError("waves must contain only 1, 2, 3, 4, or all.") from exc
    known = {entry.wave for entry in registry.entries}
    if not selected or not selected.issubset(known):
        raise ValueError(f"waves must be drawn from {sorted(known)}.")
    return selected


def _plan_reference(raw_entry: Mapping[str, Any], *, registry_path: Path, wave: int, construct_id: str) -> Path:
    """Resolve an explicit registry plan path, with the frozen v1 convention as fallback."""

    reference = None
    for key in ("plan_path", "generation_plan_path", "generation_plan"):
        candidate = raw_entry.get(key)
        if isinstance(candidate, str) and candidate.strip():
            reference = candidate.strip()
            break
    if reference is None:
        reference = f"generation_plans/wave{wave}_{construct_id}_v1.json"
    return (registry_path.parent / reference).resolve()


def discover_vector_plans(
    registry_path: str | Path = DEFAULT_REGISTRY,
    waves: Iterable[int | str] | None = None,
) -> tuple[VectorPlanEntry, ...]:
    """Discover and validate the exact spec/plan pair for selected registry waves."""

    registry_path = Path(registry_path).resolve()
    registry = load_construct_registry(registry_path)
    raw_entries = _raw_registry_entries(registry_path)
    selected_waves = _selected_wave_set(waves, registry)
    selected_entries = [entry for entry in registry.entries if entry.wave in selected_waves]
    if not selected_entries:
        raise ValueError("No registry constructs matched the selected waves.")

    discovered: list[VectorPlanEntry] = []
    for entry in selected_entries:
        raw_entry = raw_entries[entry.construct_id]
        spec_path = (registry_path.parent / entry.spec_path).resolve()
        plan_path = _plan_reference(
            raw_entry,
            registry_path=registry_path,
            wave=entry.wave,
            construct_id=entry.construct_id,
        )
        if not spec_path.is_file():
            raise FileNotFoundError(f"Registry spec for {entry.construct_id} is missing: {spec_path}")
        if not plan_path.is_file():
            raise FileNotFoundError(f"Generation plan for {entry.construct_id} is missing: {plan_path}")

        raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        declared_spec = raw_plan.get("construct_spec_path") if isinstance(raw_plan, dict) else None
        if not isinstance(declared_spec, str) or not declared_spec.strip():
            raise ValueError(f"{plan_path} must declare construct_spec_path.")
        declared_spec_path = (plan_path.parent / declared_spec).resolve()
        if declared_spec_path != spec_path:
            raise ValueError(
                f"{plan_path} points to {declared_spec_path}, but the registry points to {spec_path}."
            )
        if raw_plan.get("construct_id") != entry.construct_id:
            raise ValueError(f"{plan_path} construct_id does not match registry entry {entry.construct_id!r}.")
        if raw_plan.get("wave") != entry.wave:
            raise ValueError(f"{plan_path} wave does not match registry wave {entry.wave}.")

        spec = load_construct_spec(spec_path)
        plan = load_generation_plan(plan_path, spec)
        if set(spec.paired_splits) != VECTOR_SPLITS:
            raise ValueError(
                f"{entry.construct_id} paired_splits must be exactly {sorted(VECTOR_SPLITS)}; "
                f"received {list(spec.paired_splits)}."
            )
        models = tuple({"alias": str(model["alias"]), "model": str(model["model"])} for model in plan["models"])
        discovered.append(
            VectorPlanEntry(
                construct_id=entry.construct_id,
                wave=entry.wave,
                spec_path=spec_path,
                plan_path=plan_path,
                spec=spec,
                plan=plan,
                spec_sha256=canonical_hash(spec.to_mapping()),
                plan_sha256=_canonical_sha256(plan),
                models=models,
                source_plan_sha256=_canonical_sha256(plan),
            )
        )
    return tuple(discovered)


def _effective_plan_entries(
    entries: Iterable[VectorPlanEntry],
    *,
    model: str | None = None,
    model_alias: str | None = None,
    max_items_per_request: int | None = None,
    max_output_tokens: int | None = None,
) -> tuple[VectorPlanEntry, ...]:
    """Apply an explicit model override without mutating checked-in plans.

    The generated records carry a plan hash.  When the CLI selects a model
    different from the plan's model, validating against the original hash
    would make a successful run look stale.  Keep the source hash separately
    and hash the effective, in-memory plan used for that run.
    """

    if (
        model is None
        and model_alias is None
        and max_items_per_request is None
        and max_output_tokens is None
    ):
        return tuple(entries)
    if max_items_per_request is not None and max_items_per_request < 1:
        raise ValueError("max_items_per_request must be a positive integer.")
    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer.")
    effective: list[VectorPlanEntry] = []
    for entry in entries:
        plan = copy.deepcopy(entry.plan)
        raw_models = plan.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError(f"{entry.plan_path} has no models to override.")
        for raw_model in raw_models:
            if model is not None:
                raw_model["model"] = model
            if model_alias is not None:
                raw_model["alias"] = model_alias
        generation = plan.setdefault("generation", {})
        if max_items_per_request is not None:
            generation["max_items_per_request"] = max_items_per_request
        if max_output_tokens is not None:
            # Keep the override under the provider-neutral name consumed by
            # the OpenAI Responses adapter.  It intentionally remains in the
            # effective plan so hashes/run identities cannot mix token caps.
            generation["max_output_tokens"] = max_output_tokens
        models = tuple({"alias": str(item["alias"]), "model": str(item["model"])} for item in raw_models)
        aliases = [item["alias"] for item in models]
        if len(aliases) != len(set(aliases)):
            raise ValueError("--model-alias would create duplicate model aliases in a generation plan.")
        effective.append(
            replace(
                entry,
                plan=plan,
                plan_sha256=_canonical_sha256(plan),
                models=models,
                source_plan_sha256=entry.source_plan_sha256 or entry.plan_sha256,
            )
        )
    return tuple(effective)


def _default_prices_for_model(
    *,
    provider: str,
    model: str | None,
) -> tuple[float | None, float | None]:
    """Return explicit preflight prices for known models.

    Unknown models intentionally return ``None`` so a full run cannot proceed
    on an unpriced estimate.  OpenRouter plan models retain the historical
    no-price behavior unless the caller supplies prices explicitly.
    """

    normalized = (model or "").strip().lower()
    if provider == "openai" and normalized in {DEFAULT_LUNA_MODEL, f"openai/{DEFAULT_LUNA_MODEL}"}:
        return LUNA_INPUT_USD_PER_MILLION_TOKENS, LUNA_OUTPUT_USD_PER_MILLION_TOKENS
    return None, None


def _resolve_prices(
    *,
    provider: str,
    model: str | None,
    input_usd_per_million_tokens: float | None,
    output_usd_per_million_tokens: float | None,
) -> tuple[float | None, float | None]:
    default_input, default_output = _default_prices_for_model(provider=provider, model=model)
    resolved_input = default_input if input_usd_per_million_tokens is None else input_usd_per_million_tokens
    resolved_output = default_output if output_usd_per_million_tokens is None else output_usd_per_million_tokens
    if resolved_input is not None and resolved_input < 0:
        raise ValueError("input_usd_per_million_tokens must be non-negative.")
    if resolved_output is not None and resolved_output < 0:
        raise ValueError("output_usd_per_million_tokens must be non-negative.")
    return resolved_input, resolved_output


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


def _resolve_artifact_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (relative_to.parent / path).resolve()


def validate_quality_gate(
    quality_gate_file: str | Path,
    *,
    entries: Iterable[VectorPlanEntry],
) -> dict[str, Any]:
    """Validate an explicit human approval artifact for a full run.

    The gate references a completed, non-dry review manifest and its SHA-256.
    The review manifest must in turn hash every selected construct output and
    the combined CSV.  This makes approval deterministic and prevents a stale
    pilot from silently authorizing a changed plan.
    """

    gate_path = Path(quality_gate_file).resolve()
    gate = _load_json_object(gate_path, label="quality gate file")
    if str(gate.get("quality_gate_version", "")) != QUALITY_GATE_VERSION:
        raise ValueError(
            f"Quality gate {gate_path} must declare quality_gate_version={QUALITY_GATE_VERSION!r}."
        )
    if gate.get("approved") is not True or str(gate.get("status", "")).lower() != "approved":
        raise ValueError("Quality gate must explicitly set approved=true and status='approved'.")
    reviewer = gate.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Quality gate must identify a non-empty reviewer.")
    manifest_value = gate.get("review_manifest_path")
    manifest_hash = gate.get("review_manifest_sha256")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("Quality gate must reference review_manifest_path.")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError("Quality gate must include a 64-character review_manifest_sha256.")
    manifest_path = _resolve_artifact_path(manifest_value, relative_to=gate_path)
    try:
        actual_manifest_hash = file_sha256(manifest_path)
    except OSError as exc:
        raise ValueError(f"Quality gate review manifest is not readable: {manifest_path}") from exc
    if actual_manifest_hash != manifest_hash:
        raise ValueError(
            f"Quality gate review manifest hash mismatch: expected {manifest_hash}, got {actual_manifest_hash}."
        )
    manifest = _load_json_object(manifest_path, label="review manifest")
    if manifest.get("manifest_type") != "vector_prompt_generation":
        raise ValueError("Quality gate review manifest has the wrong manifest_type.")
    if manifest.get("dry_run") is not False:
        raise ValueError("Quality gate requires a real (non-dry-run) review manifest.")
    if manifest.get("run_mode") != "review" or manifest.get("partial") is not True:
        raise ValueError("Quality gate requires a partial review-mode manifest.")
    if manifest.get("confirmatory") is not False:
        raise ValueError("Quality gate review manifest must be non-confirmatory.")

    selected_entries = tuple(entries)
    selected_ids = {entry.construct_id for entry in selected_entries}
    manifest_ids = set(manifest.get("construct_ids", []))
    if not selected_ids.issubset(manifest_ids):
        missing = sorted(selected_ids - manifest_ids)
        raise ValueError(f"Quality gate review manifest is missing selected constructs: {missing}.")
    manifest_constructs = {
        str(item.get("construct_id")): item
        for item in manifest.get("constructs", [])
        if isinstance(item, Mapping)
    }
    for entry in selected_entries:
        item = manifest_constructs.get(entry.construct_id)
        if item is None:
            raise ValueError(f"Quality gate review manifest has no entry for {entry.construct_id}.")
        expected_source_hash = entry.source_plan_sha256 or entry.plan_sha256
        review_source_hash = item.get("source_plan_sha256") or item.get("plan_sha256")
        if review_source_hash != expected_source_hash:
            raise ValueError(
                f"Quality gate review plan hash for {entry.construct_id} does not match the checked-in plan."
            )
        output_value = item.get("output_path")
        output_hash = item.get("output_sha256")
        if not isinstance(output_value, str) or not isinstance(output_hash, str) or len(output_hash) != 64:
            raise ValueError(f"Quality gate review output metadata is incomplete for {entry.construct_id}.")
        output_path = _resolve_artifact_path(output_value, relative_to=manifest_path)
        try:
            actual_output_hash = file_sha256(output_path)
        except OSError as exc:
            raise ValueError(f"Quality gate review output is not readable for {entry.construct_id}.") from exc
        if actual_output_hash != output_hash:
            raise ValueError(f"Quality gate review output hash mismatch for {entry.construct_id}.")
        try:
            review_records = tuple(load_prompt_records(output_path))
            review_models = item.get("models")
            review_entry = entry
            if isinstance(review_models, list) and review_models:
                review_entry = replace(
                    entry,
                    plan_sha256=str(item.get("plan_sha256") or expected_source_hash),
                    models=tuple(
                        {"alias": str(model["alias"]), "model": str(model["model"])}
                        for model in review_models
                        if isinstance(model, Mapping) and "alias" in model and "model" in model
                    ),
                )
            _validate_vector_records(
                review_entry,
                review_records,
                mode="review",
                context="Quality gate review output",
            )
        except (OSError, ValueError) as exc:
            raise ValueError(f"Quality gate review output is not valid for {entry.construct_id}: {exc}") from exc
    combined_value = manifest.get("combined_path")
    combined_hash = manifest.get("combined_sha256")
    if not isinstance(combined_value, str) or not isinstance(combined_hash, str) or len(combined_hash) != 64:
        raise ValueError("Quality gate review manifest has incomplete combined-output metadata.")
    combined_path = _resolve_artifact_path(combined_value, relative_to=manifest_path)
    try:
        actual_combined_hash = file_sha256(combined_path)
    except OSError as exc:
        raise ValueError(f"Quality gate review combined output is not readable: {combined_path}") from exc
    if actual_combined_hash != combined_hash:
        raise ValueError("Quality gate review combined-output hash mismatch.")
    return {
        "quality_gate_path": str(gate_path),
        "review_manifest_path": str(manifest_path),
        "review_manifest_sha256": actual_manifest_hash,
        "reviewer": reviewer.strip(),
        "approved": True,
    }


def _mode_settings(
    entry: VectorPlanEntry,
    mode: str,
    *,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
) -> tuple[dict[str, Any], int | None, dict[str, Any]]:
    _, mode_config = resolve_generation_mode(entry.plan, mode)
    count_override = mode_config.get("count_per_model_per_cell") if mode == "review" else None
    summary = dry_run_summary(
        entry.plan,
        count_per_model_override=count_override,
        splits=set(VECTOR_SPLITS),
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        output_usd_per_million_tokens=output_usd_per_million_tokens,
    )
    return dict(mode_config), count_override, summary


def _vector_counts(records: Iterable[PromptRecord]) -> tuple[dict[str, int], int, int]:
    materialized = list(records)
    split_counts: dict[str, int] = {}
    pair_keys: set[tuple[str, str]] = set()
    for record in materialized:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        if record.pair_id:
            pair_keys.add((record.split, record.pair_id))
    return dict(sorted(split_counts.items())), len(pair_keys), len(materialized)


def _validate_vector_records(
    entry: VectorPlanEntry,
    records: Iterable[PromptRecord],
    *,
    mode: str,
    context: str,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
) -> dict[str, Any]:
    materialized = tuple(records)
    validate_prompt_records(
        materialized,
        {entry.construct_id: entry.spec},
        require_all_splits=False,
    )
    if not materialized:
        raise ValueError(f"{context} for {entry.construct_id} is empty.")
    if {record.split for record in materialized} != VECTOR_SPLITS:
        raise ValueError(f"{context} for {entry.construct_id} contains non-vector or missing splits.")
    mode_config, _, expected = _mode_settings(
        entry,
        mode,
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        output_usd_per_million_tokens=output_usd_per_million_tokens,
    )
    split_counts, pair_count, record_count = _vector_counts(materialized)
    if record_count != expected["expected_record_count"]:
        raise ValueError(
            f"{context} for {entry.construct_id} has {record_count} records; "
            f"expected {expected['expected_record_count']} for mode={mode}."
        )
    if pair_count != record_count // 2:
        raise ValueError(f"{context} for {entry.construct_id} has an invalid pair count={pair_count}.")
    if split_counts != expected["records_by_split"]:
        raise ValueError(
            f"{context} for {entry.construct_id} has split counts {split_counts}; "
            f"expected {expected['records_by_split']}."
        )

    plan_hashes = {record.metadata.get("generation_plan_sha256") for record in materialized}
    if plan_hashes != {entry.plan_sha256}:
        raise ValueError(f"{context} for {entry.construct_id} has a stale or missing generation plan hash.")
    plan_ids = {record.metadata.get("generation_plan_id") for record in materialized}
    if plan_ids != {entry.plan["plan_id"]}:
        raise ValueError(f"{context} for {entry.construct_id} has a stale or missing generation plan ID.")
    source_aliases = {record.metadata.get("source_model_alias") for record in materialized}
    expected_aliases = {model["alias"] for model in entry.models}
    if source_aliases != expected_aliases:
        raise ValueError(
            f"{context} for {entry.construct_id} has model aliases {sorted(source_aliases)}; "
            f"expected {sorted(expected_aliases)}."
        )
    return {
        "split_counts": split_counts,
        "pair_count": pair_count,
        "record_count": record_count,
        "expected_split_counts": expected["records_by_split"],
        "expected_pair_count": expected["expected_record_count"] // 2,
        "expected_record_count": expected["expected_record_count"],
        "request_count": expected["request_count"],
        "estimated_input_tokens": expected["estimated_input_tokens"],
        "estimated_output_tokens": expected["estimated_output_tokens"],
        "estimated_total_tokens": expected["estimated_total_tokens"],
        "estimated_cost_usd": expected["estimated_cost_usd"],
        "mode_purpose": mode_config["purpose"],
    }


def _existing_output(
    entry: VectorPlanEntry,
    output_path: Path,
    *,
    mode: str,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
) -> tuple[tuple[PromptRecord, ...], dict[str, Any]]:
    records = tuple(load_prompt_records(output_path))
    details = _validate_vector_records(
        entry,
        records,
        mode=mode,
        context="Existing output",
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        output_usd_per_million_tokens=output_usd_per_million_tokens,
    )
    response_costs: dict[str, float] = {}
    for record in records:
        metadata = record.metadata
        response_key = (
            metadata.get("generation_job_id")
            or metadata.get("generation_batch_id")
            or metadata.get("generation_response_id")
        )
        raw_cost = metadata.get("generation_actual_cost_usd")
        if not isinstance(response_key, str) or not response_key.strip():
            raise ValueError(
                f"Existing output for {entry.construct_id} cannot reconstruct unique response costs: "
                f"prompt {record.prompt_id} has no generation job/batch/response ID."
            )
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float, str)):
            raise ValueError(
                f"Existing output for {entry.construct_id} cannot reconstruct unique response costs: "
                f"prompt {record.prompt_id} has no generation_actual_cost_usd."
            )
        try:
            response_cost = float(raw_cost)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Existing output for {entry.construct_id} has a non-numeric generation_actual_cost_usd."
            ) from exc
        if not math.isfinite(response_cost) or response_cost < 0:
            raise ValueError(
                f"Existing output for {entry.construct_id} has an invalid generation_actual_cost_usd."
            )
        response_key = str(response_key)
        previous_cost = response_costs.get(response_key)
        if previous_cost is not None and abs(previous_cost - response_cost) > 1e-12:
            raise ValueError(
                f"Existing output for {entry.construct_id} has inconsistent costs for response {response_key!r}."
            )
        response_costs[response_key] = response_cost
    details["actual_cost_usd"] = sum(response_costs.values())
    details["actual_request_count"] = len(response_costs)
    details["resumed"] = True
    details["output_sha256"] = file_sha256(output_path)
    return records, details


def _generate_one(
    entry: VectorPlanEntry,
    output_path: Path,
    *,
    mode: str,
    api_key: str,
    request_fn: RequestFn,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
    completed_job_records: Mapping[str, tuple[PromptRecord, ...]] | None = None,
    completed_job_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    before_job_request: Callable[[Any], None] | None = None,
    on_job_checkpoint: Callable[[str, Mapping[str, Any]], None] | None = None,
    write_output: bool = True,
) -> tuple[tuple[PromptRecord, ...], dict[str, Any]]:
    _, count_override, _ = _mode_settings(entry, mode)
    checkpoint_jobs: dict[str, dict[str, Any]] = {}
    if completed_job_records:
        if checkpoint_path is None or checkpoint_identity is None:
            raise ValueError("Recovered jobs require a checkpoint path and identity.")
        for job_id, records in completed_job_records.items():
            metadata = dict((completed_job_metadata or {}).get(job_id, {}))
            checkpoint_jobs[job_id] = _checkpoint_job_payload(job_id, records, metadata)

    def checkpoint_job(
        job: Any,
        records: tuple[PromptRecord, ...],
        response_metadata: dict[str, Any],
    ) -> None:
        if checkpoint_path is None or checkpoint_identity is None:
            return
        checkpoint_jobs[job.job_id] = _checkpoint_job_payload(job.job_id, records, response_metadata)
        _write_job_checkpoint(
            checkpoint_path,
            identity=checkpoint_identity,
            jobs=checkpoint_jobs,
        )
        if on_job_checkpoint is not None:
            on_job_checkpoint(job.job_id, response_metadata)

    result = generate_prompt_records(
        entry.plan,
        entry.spec,
        api_key=api_key,
        request_fn=request_fn,
        count_per_model_override=count_override,
        splits=set(VECTOR_SPLITS),
        completed_job_records=completed_job_records,
        completed_job_metadata=completed_job_metadata,
        on_job_complete=checkpoint_job if checkpoint_path is not None else None,
        before_job_request=before_job_request,
    )
    records = tuple(result.records)
    details = _validate_vector_records(
        entry,
        records,
        mode=mode,
        context="Generated output",
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        output_usd_per_million_tokens=output_usd_per_million_tokens,
    )
    result_summary = result.summary()
    for key in (
        "actual_input_tokens",
        "actual_output_tokens",
        "actual_total_tokens",
        "actual_cost_usd",
    ):
        details[key] = result_summary.get(key, 0)
    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_generation_result(result, output_path)
        details["output_sha256"] = file_sha256(output_path)
    else:
        details["output_sha256"] = None
    details["resumed"] = bool(completed_job_records)
    if checkpoint_path is not None:
        details["checkpoint_path"] = str(checkpoint_path)
        details["checkpoint_sha256"] = file_sha256(checkpoint_path)
        details["checkpoint_job_count"] = len(checkpoint_jobs)
    return records, details


def _request_with_runtime_options(
    request_fn: RequestFn,
    *,
    provider: str,
    requested_model: str | None,
    reasoning_effort: str | None,
    input_usd_per_million_tokens: float | None,
    output_usd_per_million_tokens: float | None,
) -> RequestFn:
    """Attach provider/runtime selections without changing generic generation."""

    def request(model_id: str, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        runtime_options = dict(options)
        runtime_options["provider"] = provider
        if requested_model is not None:
            runtime_options["model_override"] = requested_model
            runtime_options["openai_model"] = requested_model
        if reasoning_effort is not None:
            runtime_options["reasoning_effort"] = reasoning_effort
            runtime_options["reasoning"] = {"effort": reasoning_effort}
        if input_usd_per_million_tokens is not None:
            runtime_options["input_usd_per_million_tokens"] = input_usd_per_million_tokens
        if output_usd_per_million_tokens is not None:
            runtime_options["output_usd_per_million_tokens"] = output_usd_per_million_tokens
        effective_model = requested_model or model_id
        return request_fn(effective_model, messages, runtime_options)

    return request


def _request_with_runtime_budget(
    request_fn: RequestFn,
    budget: RuntimeBudget,
) -> RequestFn:
    """Reserve the worst-case request cost before invoking a transport."""

    def request(model_id: str, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        reservation: _BudgetReservation | None = None
        try:
            reservation = budget.reserve(options)
            response = request_fn(model_id, messages, options)
            if not isinstance(response, Mapping):
                raise ValueError("Request function must return a mapping response.")
            return budget.settle(reservation, response)
        except Exception:
            if reservation is not None:
                budget.release_on_error(reservation)
            raise

    return request


def _default_request_fn_for_provider(provider: str) -> RequestFn:
    """Resolve the provider transport lazily so dry-runs remain dependency-free."""

    if provider == "openrouter":
        return call_openrouter_chat_completion
    if provider == "openai":
        # The OpenAI transport is optional in the base environment.  Keep the
        # import lazy and support the two module locations used during the
        # provider-adapter migration.
        try:
            from activation_analysis.openai_prompt_generation import call_openai_responses
        except ImportError:
            try:
                from activation_analysis.openrouter_prompt_generation import call_openai_responses
            except ImportError as exc:
                raise ValueError(
                    "provider=openai requires call_openai_responses in the active activation_analysis transport."
                ) from exc
        return call_openai_responses
    raise ValueError("provider must be openrouter or openai.")


def _construct_manifest(
    entry: VectorPlanEntry,
    output_path: Path,
    details: Mapping[str, Any],
    *,
    mode: str,
    mode_config: Mapping[str, Any],
    dry_run: bool,
    provider: str = DEFAULT_PROVIDER,
    requested_model: str | None = None,
    reasoning_effort: str | None = None,
    runtime_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "construct_id": entry.construct_id,
        "wave": entry.wave,
        "spec_path": str(entry.spec_path),
        "plan_path": str(entry.plan_path),
        "spec_hash": entry.spec_sha256,
        "source_plan_sha256": entry.source_plan_sha256 or entry.plan_sha256,
        "spec_sha256": entry.spec_sha256,
        "plan_sha256": entry.plan_sha256,
        "models": [dict(model) for model in entry.models],
        "provider": provider,
        "requested_model": requested_model,
        "reasoning_effort": reasoning_effort,
        "runtime_settings": dict(runtime_settings or {}),
        "run_mode": mode,
        "run_mode_purpose": mode_config["purpose"],
        "partial": bool(mode_config["partial"]),
        "confirmatory": False,
        "scope": "vector",
        "scope_partial": True,
        "dry_run": dry_run,
        "resumed": bool(details.get("resumed", False)),
        "split_counts": dict(details["split_counts"]),
        "pair_count": int(details["pair_count"]),
        "record_count": int(details["record_count"]),
        "expected_split_counts": dict(details["expected_split_counts"]),
        "expected_pair_count": int(details["expected_pair_count"]),
        "expected_record_count": int(details["expected_record_count"]),
        "request_count": int(details["request_count"]),
        "estimated_input_tokens": int(details["estimated_input_tokens"]),
        "estimated_output_tokens": int(details["estimated_output_tokens"]),
        "estimated_total_tokens": int(details["estimated_total_tokens"]),
        "estimated_cost_usd": details.get("estimated_cost_usd"),
        "actual_input_tokens": int(details.get("actual_input_tokens", 0) or 0),
        "actual_output_tokens": int(details.get("actual_output_tokens", 0) or 0),
        "actual_total_tokens": int(details.get("actual_total_tokens", 0) or 0),
        "actual_cost_usd": float(details.get("actual_cost_usd", 0.0) or 0.0),
        "output_path": str(output_path),
        "output_sha256": None if dry_run else details.get("output_sha256"),
        "checkpoint_path": details.get("checkpoint_path"),
        "checkpoint_sha256": details.get("checkpoint_sha256"),
        "checkpoint_job_count": int(details.get("checkpoint_job_count", 0) or 0),
    }


def _atomic_write_records(records: Iterable[PromptRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=path.suffix or ".csv",
            prefix=f".{path.stem}.",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        from construct_benchmark.prompts import write_prompt_records

        write_prompt_records(tuple(records), temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f".{path.stem}.",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _checkpoint_checksum(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "checksum_sha256"}
    return _canonical_sha256(unsigned)


def _checkpoint_identity(
    entry: VectorPlanEntry,
    *,
    run_identity: Mapping[str, Any],
    runtime_settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "run_identity": dict(run_identity),
        "construct_id": entry.construct_id,
        "source_plan_sha256": entry.source_plan_sha256 or entry.plan_sha256,
        "effective_plan_sha256": entry.plan_sha256,
        "provider": run_identity.get("provider"),
        "model": run_identity.get("model"),
        "model_alias": run_identity.get("model_alias"),
        "reasoning_effort": run_identity.get("reasoning_effort"),
        "runtime_settings": dict(runtime_settings),
    }


def _checkpoint_job_payload(
    job_id: str,
    records: Iterable[PromptRecord],
    response_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(response_metadata)
    raw_cost = metadata.get("actual_cost_usd")
    if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
        raise ValueError(f"Checkpoint job {job_id} is missing numeric actual_cost_usd provenance.")
    cost = float(raw_cost)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError(f"Checkpoint job {job_id} has invalid actual_cost_usd provenance.")
    return {
        "job_id": job_id,
        "records": [record.to_mapping() for record in records],
        "response_metadata": metadata,
        "actual_cost_usd": cost,
    }


def _write_job_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "schema_version": "1",
        "checkpoint_type": "vector_prompt_generation_job_checkpoint",
        "identity": dict(identity),
        "jobs": [dict(jobs[job_id]) for job_id in sorted(jobs)],
    }
    payload["checksum_sha256"] = _checkpoint_checksum(payload)
    _atomic_write_json(payload, path)


def _load_job_checkpoint(
    path: Path,
    *,
    entry: VectorPlanEntry,
    expected_identity: Mapping[str, Any],
) -> tuple[dict[str, tuple[PromptRecord, ...]], dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _load_json_object(path, label="vector prompt job checkpoint")
    expected_checksum = payload.get("checksum_sha256")
    if not isinstance(expected_checksum, str) or len(expected_checksum) != 64:
        raise ValueError(f"Checkpoint {path} is missing checksum_sha256.")
    actual_checksum = _checkpoint_checksum(payload)
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Checkpoint checksum mismatch for {path}: expected {expected_checksum}, got {actual_checksum}."
        )
    if payload.get("checkpoint_type") != "vector_prompt_generation_job_checkpoint":
        raise ValueError(f"Checkpoint {path} has an unsupported checkpoint_type.")
    if payload.get("identity") != dict(expected_identity):
        raise ValueError(
            f"Checkpoint identity is stale for {entry.construct_id}; start a fresh output directory "
            "or resume with the original provider/model/plan/runtime settings."
        )
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError(f"Checkpoint {path} must contain a jobs list.")

    expected_jobs = {
        job.job_id
        for job in iter_generation_request_jobs(entry.plan, splits=set(VECTOR_SPLITS))
    }
    records_by_job: dict[str, tuple[PromptRecord, ...]] = {}
    metadata_by_job: dict[str, dict[str, Any]] = {}
    seen_prompt_ids: set[str] = set()
    for raw_job in raw_jobs:
        if not isinstance(raw_job, Mapping):
            raise ValueError(f"Checkpoint {path} contains a malformed job entry.")
        job_id = raw_job.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip() or job_id not in expected_jobs:
            raise ValueError(f"Checkpoint {path} contains an unknown or invalid job identity: {job_id!r}.")
        if job_id in records_by_job:
            raise ValueError(f"Checkpoint {path} contains duplicate job identity: {job_id!r}.")
        raw_records = raw_job.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"Checkpoint {path} job {job_id} has no records.")
        try:
            records = tuple(PromptRecord.from_mapping(row) for row in raw_records if isinstance(row, Mapping))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint {path} job {job_id} has malformed records.") from exc
        if len(records) != len(raw_records):
            raise ValueError(f"Checkpoint {path} job {job_id} has malformed records.")
        validate_prompt_records(records, {entry.construct_id: entry.spec}, require_all_splits=False)
        for record in records:
            if record.metadata.get("generation_job_id") != job_id:
                raise ValueError(f"Checkpoint {path} job {job_id} contains a record from another job.")
            if record.prompt_id in seen_prompt_ids:
                raise ValueError(f"Checkpoint {path} duplicates prompt_id={record.prompt_id!r}.")
            seen_prompt_ids.add(record.prompt_id)
        raw_metadata = raw_job.get("response_metadata")
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"Checkpoint {path} job {job_id} is missing response_metadata.")
        metadata = dict(raw_metadata)
        raw_cost = metadata.get("actual_cost_usd", raw_job.get("actual_cost_usd"))
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise ValueError(f"Checkpoint {path} job {job_id} is missing numeric cost provenance.")
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(f"Checkpoint {path} job {job_id} has invalid cost provenance.")
        metadata["actual_cost_usd"] = cost
        records_by_job[job_id] = records
        metadata_by_job[job_id] = metadata
    return records_by_job, metadata_by_job, {
        "checkpoint_path": str(path),
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_job_count": len(records_by_job),
        "checkpoint_actual_cost_usd": sum(
            float(metadata.get("actual_cost_usd", 0.0) or 0.0)
            for metadata in metadata_by_job.values()
        ),
    }


def _prior_budget_spend(run_state: Mapping[str, Any] | None) -> float:
    """Recover conservative prior spend for a resumable run-state artifact."""

    if not run_state:
        return 0.0
    budget_state = run_state.get("budget_state")
    status = str(run_state.get("status", ""))
    if not isinstance(budget_state, Mapping):
        if status in {"running", "failed"}:
            raise ValueError(
                "Cannot resume an interrupted vector generation: its runtime budget state is missing, "
                "so prior request spending is not reconstructable."
            )
        return 0.0
    values: list[float] = []
    for field_name in ("actual_spent_usd", "outstanding_reserved_usd"):
        raw_value = budget_state.get(field_name, 0.0)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"Run-state budget field {field_name} is not numeric.")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Run-state budget field {field_name} is invalid.")
        values.append(value)
    # Any in-flight reservation at interruption is charged conservatively on
    # resume because the provider may have accepted the request already.
    return sum(values)


def _progress_summary(
    entries: Iterable[VectorPlanEntry],
    *,
    mode: str,
    records_by_construct: Mapping[str, tuple[PromptRecord, ...]],
    details_by_construct: Mapping[str, Mapping[str, Any]],
    checkpoint_details_by_construct: Mapping[str, Mapping[str, Any]],
    runtime_budget: RuntimeBudget,
    new_job_limit: NewJobLimit | None,
) -> dict[str, Any]:
    """Summarize durable request-level progress without creating final outputs."""

    planned_job_count = 0
    completed_job_count = 0
    checkpoint_job_counts: dict[str, int] = {}
    construct_progress: list[dict[str, Any]] = []
    for entry in entries:
        _, _, expected = _mode_settings(entry, mode)
        planned_count = int(expected["request_count"])
        planned_job_count += planned_count
        if entry.construct_id in records_by_construct:
            completed_count = planned_count
            source = "construct_output"
        else:
            checkpoint_details = checkpoint_details_by_construct.get(entry.construct_id, {})
            completed_count = int(checkpoint_details.get("checkpoint_job_count", 0) or 0)
            source = "job_checkpoint" if completed_count else "not_started"
        completed_job_count += completed_count
        checkpoint_job_counts[entry.construct_id] = completed_count
        construct_progress.append(
            {
                "construct_id": entry.construct_id,
                "planned_job_count": planned_count,
                "completed_job_count": completed_count,
                "remaining_job_count": max(planned_count - completed_count, 0),
                "source": source,
            }
        )
    budget_snapshot = runtime_budget.snapshot()
    limit_snapshot = new_job_limit.snapshot() if new_job_limit is not None else {
        "max_new_jobs": None,
        "new_jobs_completed": 0,
        "new_jobs_reserved": 0,
        "new_spend_usd": 0.0,
    }
    return {
        "planned_job_count": planned_job_count,
        "completed_job_count": completed_job_count,
        "remaining_job_count": max(planned_job_count - completed_job_count, 0),
        "checkpoint_job_counts": checkpoint_job_counts,
        "constructs": construct_progress,
        "new_jobs_completed_this_invocation": limit_snapshot["new_jobs_completed"],
        "new_spend_usd_this_invocation": limit_snapshot["new_spend_usd"],
        "budget_state": budget_snapshot,
        "actual_spent_usd": budget_snapshot["actual_spent_usd"],
        "remaining_budget_usd": budget_snapshot["remaining_budget_usd"],
    }


def orchestrate_vector_prompts(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    waves: Iterable[int | str] | None = None,
    construct_ids: Iterable[str] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    mode: str = "full",
    workers: int = 4,
    resume: bool = False,
    dry_run: bool = False,
    api_key: str | None = None,
    request_fn: RequestFn | None = None,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    model_alias: str | None = None,
    reasoning_effort: str | None = None,
    max_items_per_request: int | None = None,
    max_output_tokens: int | None = None,
    max_new_jobs: int | None = None,
    max_estimated_cost_usd: float | None = None,
    input_usd_per_million_tokens: float | None = None,
    output_usd_per_million_tokens: float | None = None,
    cost_safety_multiplier: float = COST_SAFETY_MULTIPLIER,
    quality_gate_file: str | Path | None = None,
) -> dict[str, Any]:
    """Generate or dry-run vector prompts for selected registry waves."""

    if workers < 1:
        raise ValueError("workers must be a positive integer.")
    if mode not in {"review", "full"}:
        raise ValueError("mode must be review or full.")
    provider = str(provider).strip().lower()
    if provider not in {"openrouter", "openai"}:
        raise ValueError("provider must be openrouter or openai.")
    if cost_safety_multiplier < 1:
        raise ValueError("cost_safety_multiplier must be at least 1.0.")
    if max_estimated_cost_usd is not None and max_estimated_cost_usd < 0:
        raise ValueError("max_estimated_cost_usd must be non-negative.")
    if max_items_per_request is not None and max_items_per_request < 1:
        raise ValueError("max_items_per_request must be a positive integer.")
    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError("max_output_tokens must be a positive integer.")
    if max_new_jobs is not None and max_new_jobs < 0:
        raise ValueError("max_new_jobs must be non-negative.")
    if max_new_jobs is not None and mode != "full":
        raise ValueError("max_new_jobs is supported only for full-mode generation.")
    source_entries = discover_vector_plans(registry_path, waves)
    if construct_ids is not None:
        requested_constructs = {str(value).strip() for value in construct_ids if str(value).strip()}
        known_constructs = {entry.construct_id for entry in source_entries}
        unknown_constructs = requested_constructs - known_constructs
        if unknown_constructs:
            raise ValueError(
                f"Unknown or out-of-wave construct IDs: {sorted(unknown_constructs)}; "
                f"available IDs are {sorted(known_constructs)}."
            )
        source_entries = tuple(entry for entry in source_entries if entry.construct_id in requested_constructs)
        if not source_entries:
            raise ValueError("construct_ids must select at least one construct.")
    quality_gate = None
    if mode == "full" and not dry_run:
        if quality_gate_file is None:
            raise ValueError(
                "Full non-dry generation requires --quality-gate-file referencing an approved review manifest."
            )
        quality_gate = validate_quality_gate(quality_gate_file, entries=source_entries)
        if max_estimated_cost_usd is None:
            raise ValueError("Full non-dry generation requires --max-estimated-cost-usd.")
    if model is None and provider == "openai":
        model = DEFAULT_LUNA_MODEL
    resolved_input_price, resolved_output_price = _resolve_prices(
        provider=provider,
        model=model,
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        output_usd_per_million_tokens=output_usd_per_million_tokens,
    )
    entries = _effective_plan_entries(
        source_entries,
        model=model,
        model_alias=model_alias,
        max_items_per_request=max_items_per_request,
        max_output_tokens=max_output_tokens,
    )
    output_root = Path(output_dir).resolve()
    output_paths = {entry.construct_id: output_root / f"{entry.construct_id}.csv" for entry in entries}
    combined_path = output_root / "combined.csv"
    manifest_path = output_root / "vector_prompt_manifest.json"
    run_state_path = output_root / "vector_prompt_run_state.json"
    checkpoint_root = output_root / "checkpoints"
    checkpoint_paths = {
        entry.construct_id: checkpoint_root / f"{entry.construct_id}.json" for entry in entries
    }

    # A provider/model override is part of the run identity.  This prevents a
    # resume from silently mixing records produced by Sonnet and Luna.
    run_identity = {
        "schema_version": "2",
        "registry_path": str(Path(registry_path).resolve()),
        "construct_ids": [entry.construct_id for entry in entries],
        "plan_hashes": {entry.construct_id: entry.plan_sha256 for entry in entries},
        "source_plan_hashes": {
            entry.construct_id: entry.source_plan_sha256 or entry.plan_sha256 for entry in entries
        },
        "mode": mode,
        "provider": provider,
        "model": model,
        "model_alias": model_alias,
        "reasoning_effort": reasoning_effort,
        "runtime_settings": {
            "max_items_per_request": max_items_per_request,
            "max_output_tokens": max_output_tokens,
            "input_usd_per_million_tokens": resolved_input_price,
            "output_usd_per_million_tokens": resolved_output_price,
            "cost_safety_multiplier": cost_safety_multiplier,
        },
    }
    runtime_settings = dict(run_identity["runtime_settings"])

    preflight_summaries = []
    for entry in entries:
        _, _, expected = _mode_settings(
            entry,
            mode,
            input_usd_per_million_tokens=resolved_input_price,
            output_usd_per_million_tokens=resolved_output_price,
        )
        preflight_summaries.append(expected)
    estimated_costs = [item["estimated_cost_usd"] for item in preflight_summaries]
    estimated_cost = sum(estimated_costs) if all(cost is not None for cost in estimated_costs) else None
    budget_estimate = None if estimated_cost is None else estimated_cost * cost_safety_multiplier
    if not dry_run:
        if max_estimated_cost_usd is None:
            raise ValueError(
                "Non-dry generation requires --max-estimated-cost-usd as an explicit spending cap."
            )
        if budget_estimate is None:
            raise ValueError(
                "Cannot enforce the spending cap because model pricing is unknown; "
                "provide --input-price-usd-per-million and --output-price-usd-per-million."
            )
        if budget_estimate > max_estimated_cost_usd:
            raise ValueError(
                f"Estimated generation cost ${budget_estimate:.4f} (including "
                f"{cost_safety_multiplier:.2f}x safety margin) exceeds the cap "
                f"${max_estimated_cost_usd:.4f}."
            )

    checkpoint_identities = {
        entry.construct_id: _checkpoint_identity(
            entry,
            run_identity=run_identity,
            runtime_settings=runtime_settings,
        )
        for entry in entries
    }

    if dry_run:
        construct_manifests: list[dict[str, Any]] = []
        total_split_counts: dict[str, int] = {}
        total_pairs = 0
        total_records = 0
        for entry in entries:
            mode_config, _, expected = _mode_settings(
                entry,
                mode,
                input_usd_per_million_tokens=resolved_input_price,
                output_usd_per_million_tokens=resolved_output_price,
            )
            details = {
                "split_counts": expected["records_by_split"],
                "pair_count": expected["expected_record_count"] // 2,
                "record_count": expected["expected_record_count"],
                "expected_split_counts": expected["records_by_split"],
                "expected_pair_count": expected["expected_record_count"] // 2,
                "expected_record_count": expected["expected_record_count"],
                "request_count": expected["request_count"],
                "estimated_input_tokens": expected["estimated_input_tokens"],
                "estimated_output_tokens": expected["estimated_output_tokens"],
                "estimated_total_tokens": expected["estimated_total_tokens"],
                "estimated_cost_usd": expected["estimated_cost_usd"],
                "resumed": False,
            }
            construct_manifests.append(
                _construct_manifest(
                    entry,
                    output_paths[entry.construct_id],
                    details,
                    mode=mode,
                    mode_config=mode_config,
                    dry_run=True,
                    provider=provider,
                    requested_model=model,
                    reasoning_effort=reasoning_effort,
                    runtime_settings=runtime_settings,
                )
            )
            for split, count in expected["records_by_split"].items():
                total_split_counts[split] = total_split_counts.get(split, 0) + count
            total_pairs += details["pair_count"]
            total_records += details["record_count"]
        return {
            "schema_version": "0.1.0",
            "manifest_type": "vector_prompt_generation",
            "registry_path": str(Path(registry_path).resolve()),
            "waves": sorted({entry.wave for entry in entries}),
            "construct_ids": [entry.construct_id for entry in entries],
            "scope": "vector",
            "scope_partial": True,
            "run_mode": mode,
            "partial": mode == "review",
            "confirmatory": False,
            "dry_run": True,
            "workers": workers,
            "provider": provider,
            "requested_model": model,
            "reasoning_effort": reasoning_effort,
            "runtime_settings": runtime_settings,
            "input_usd_per_million_tokens": resolved_input_price,
            "output_usd_per_million_tokens": resolved_output_price,
            "cost_safety_multiplier": cost_safety_multiplier,
            "max_estimated_cost_usd": max_estimated_cost_usd,
            "max_new_jobs": max_new_jobs,
            "budget_estimate_usd": budget_estimate,
            "constructs": construct_manifests,
        "counts": {
            "split_counts": dict(sorted(total_split_counts.items())),
            "pair_count": total_pairs,
            "record_count": total_records,
            "request_count": sum(item["request_count"] for item in construct_manifests),
            "estimated_input_tokens": sum(item["estimated_input_tokens"] for item in construct_manifests),
            "estimated_output_tokens": sum(item["estimated_output_tokens"] for item in construct_manifests),
            "estimated_total_tokens": sum(item["estimated_total_tokens"] for item in construct_manifests),
            "estimated_cost_usd": (
                sum(item["estimated_cost_usd"] for item in construct_manifests)
                if all(item["estimated_cost_usd"] is not None for item in construct_manifests)
                else None
            ),
            "budget_estimate_usd": budget_estimate,
        },
            "combined_path": str(combined_path),
            "combined_sha256": None,
            "manifest_path": str(manifest_path),
        }

    if not api_key:
        raise ValueError("An API key is required for non-dry-run generation.")
    if request_fn is None:
        request_fn = _default_request_fn_for_provider(provider)
    output_root.mkdir(parents=True, exist_ok=True)
    prior_run_state: dict[str, Any] | None = None
    if resume and run_state_path.exists():
        prior_run_state = _load_json_object(run_state_path, label="vector prompt run state")
        if prior_run_state.get("run_identity") != run_identity:
            raise ValueError(
                "Cannot resume vector generation with a different provider/model/reasoning or plan selection."
            )
    elif resume and (
        model is not None
        or provider != DEFAULT_PROVIDER
        or reasoning_effort is not None
        or max_items_per_request is not None
        or max_output_tokens is not None
    ):
        # Legacy output directories predate run-state metadata.  Do not allow
        # an explicit runtime override to mix with those records.
        if any(path.exists() for path in (*output_paths.values(), *checkpoint_paths.values())):
            raise ValueError(
                "Cannot resume legacy outputs with an explicit provider/model override; "
                "use a fresh output directory or create matching run-state metadata."
            )
    if not resume:
        existing = [
            path
            for path in (
                *output_paths.values(),
                *checkpoint_paths.values(),
                combined_path,
                manifest_path,
                run_state_path,
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing vector outputs without --resume: "
                + ", ".join(str(path) for path in existing)
            )
    run_state: dict[str, Any] = dict(prior_run_state or {})
    run_state.update({
        "schema_version": "1",
        "status": "running",
        "run_identity": run_identity,
        "quality_gate": quality_gate,
        "max_estimated_cost_usd": max_estimated_cost_usd,
        "estimated_cost_usd": estimated_cost,
        "budget_estimate_usd": budget_estimate,
        "completed_construct_ids": list((prior_run_state or {}).get("completed_construct_ids", [])),
        # This is invocation bookkeeping only.  It is deliberately excluded
        # from run_identity so a later resume may omit or change the limit.
        "invocation_max_new_jobs": max_new_jobs,
        "new_jobs_completed_this_invocation": 0,
        "new_spend_usd_this_invocation": 0.0,
    })
    run_state.pop("error", None)
    _atomic_write_json(run_state, run_state_path)

    records_by_construct: dict[str, tuple[PromptRecord, ...]] = {}
    details_by_construct: dict[str, dict[str, Any]] = {}
    checkpoint_records_by_construct: dict[str, dict[str, tuple[PromptRecord, ...]]] = {}
    checkpoint_metadata_by_construct: dict[str, dict[str, dict[str, Any]]] = {}
    checkpoint_details_by_construct: dict[str, dict[str, Any]] = {}
    if mode == "full" and resume:
        for entry in entries:
            checkpoint_path = checkpoint_paths[entry.construct_id]
            if not checkpoint_path.exists():
                continue
            checkpoint_records, checkpoint_metadata, checkpoint_details = _load_job_checkpoint(
                checkpoint_path,
                entry=entry,
                expected_identity=checkpoint_identities[entry.construct_id],
            )
            checkpoint_records_by_construct[entry.construct_id] = checkpoint_records
            checkpoint_metadata_by_construct[entry.construct_id] = checkpoint_metadata
            checkpoint_details_by_construct[entry.construct_id] = checkpoint_details
    pending: list[VectorPlanEntry] = []
    for entry in entries:
        output_path = output_paths[entry.construct_id]
        if output_path.exists():
            if not resume:
                raise FileExistsError(f"Output already exists: {output_path}")
            records, details = _existing_output(
                entry,
                output_path,
                mode=mode,
                input_usd_per_million_tokens=resolved_input_price,
                output_usd_per_million_tokens=resolved_output_price,
            )
            checkpoint_details = checkpoint_details_by_construct.get(entry.construct_id)
            if checkpoint_details is not None:
                checkpoint_records = checkpoint_records_by_construct[entry.construct_id]
                checkpoint_prompt_ids = {
                    record.prompt_id
                    for records in checkpoint_records.values()
                    for record in records
                }
                output_prompt_ids = {record.prompt_id for record in records}
                if not checkpoint_prompt_ids.issubset(output_prompt_ids):
                    raise ValueError(
                        f"Existing output for {entry.construct_id} is missing records from its job checkpoint."
                    )
                details.update(checkpoint_details)
                details["checkpoint_job_count"] = checkpoint_details["checkpoint_job_count"]
            records_by_construct[entry.construct_id] = records
            details_by_construct[entry.construct_id] = details
        else:
            pending.append(entry)

    csv_spend = sum(float(details.get("actual_cost_usd", 0.0) or 0.0) for details in details_by_construct.values())
    checkpoint_spend = sum(
        float(details.get("checkpoint_actual_cost_usd", 0.0) or 0.0)
        for details in checkpoint_details_by_construct.values()
    )
    prior_spend = _prior_budget_spend(prior_run_state)
    initial_spend = max(csv_spend, checkpoint_spend, prior_spend)
    if initial_spend > max_estimated_cost_usd + 1e-12:
        raise RuntimeBudgetExceeded(
            "Resumed vector generation has already reached or exceeded its spending cap: "
            f"reconstructed=${initial_spend:.6f}, cap=${max_estimated_cost_usd:.6f}."
        )

    state_lock = threading.Lock()

    def persist_budget_state(snapshot: dict[str, Any]) -> None:
        with state_lock:
            run_state["budget_state"] = snapshot
            _atomic_write_json(run_state, run_state_path)

    runtime_budget = RuntimeBudget(
        max_budget_usd=max_estimated_cost_usd,
        input_usd_per_million_tokens=resolved_input_price,
        output_usd_per_million_tokens=resolved_output_price,
        initial_spent_usd=initial_spend,
        on_change=persist_budget_state,
    )
    new_job_limit = NewJobLimit(max_new_jobs)
    run_state["reconstructed_csv_spend_usd"] = csv_spend
    run_state["reconstructed_checkpoint_spend_usd"] = checkpoint_spend
    run_state["prior_run_state_spend_usd"] = prior_spend
    run_state["budget_state"] = runtime_budget.snapshot()
    run_state["completed_construct_ids"] = sorted(records_by_construct)
    _atomic_write_json(run_state, run_state_path)

    runtime_request_fn = _request_with_runtime_budget(
        _request_with_runtime_options(
            request_fn,
            provider=provider,
            requested_model=model,
            reasoning_effort=reasoning_effort,
            input_usd_per_million_tokens=resolved_input_price,
            output_usd_per_million_tokens=resolved_output_price,
        ),
        runtime_budget,
    )

    def generate(entry: VectorPlanEntry) -> tuple[str, tuple[PromptRecord, ...], dict[str, Any]]:
        records, details = _generate_one(
            entry,
            output_paths[entry.construct_id],
            mode=mode,
            api_key=api_key,
            request_fn=runtime_request_fn,
            input_usd_per_million_tokens=resolved_input_price,
            output_usd_per_million_tokens=resolved_output_price,
            checkpoint_path=checkpoint_paths[entry.construct_id] if mode == "full" else None,
            checkpoint_identity=checkpoint_identities[entry.construct_id] if mode == "full" else None,
            completed_job_records=checkpoint_records_by_construct.get(entry.construct_id),
            completed_job_metadata=checkpoint_metadata_by_construct.get(entry.construct_id),
            before_job_request=new_job_limit.reserve if max_new_jobs is not None else None,
            on_job_checkpoint=new_job_limit.complete if max_new_jobs is not None else None,
            write_output=False,
        )
        return entry.construct_id, records, details

    pause_requested: GenerationPaused | None = None
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as executor:
        futures = {executor.submit(generate, entry): entry for entry in pending}
        try:
            for future in as_completed(futures):
                construct_id, records, details = future.result()
                records_by_construct[construct_id] = records
                details_by_construct[construct_id] = details
                completed_ids = sorted(records_by_construct)
                with state_lock:
                    run_state["completed_construct_ids"] = completed_ids
                    run_state["budget_state"] = runtime_budget.snapshot()
                    _atomic_write_json(run_state, run_state_path)
        except GenerationPaused as exc:
            pause_requested = exc
        except Exception as exc:
            with state_lock:
                run_state["status"] = "failed"
                run_state["error"] = f"{type(exc).__name__}: {exc}"
                run_state["completed_construct_ids"] = sorted(records_by_construct)
                run_state["budget_state"] = runtime_budget.snapshot()
                _atomic_write_json(run_state, run_state_path)
            raise

    if pause_requested is not None:
        # Other construct workers may have checkpointed before the first pause
        # signal reached the coordinator.  Reload every checkpoint so the
        # persisted progress summary reflects exactly what is durable.
        if mode == "full":
            for entry in entries:
                checkpoint_path = checkpoint_paths[entry.construct_id]
                if not checkpoint_path.exists():
                    continue
                checkpoint_records, checkpoint_metadata, checkpoint_details = _load_job_checkpoint(
                    checkpoint_path,
                    entry=entry,
                    expected_identity=checkpoint_identities[entry.construct_id],
                )
                checkpoint_records_by_construct[entry.construct_id] = checkpoint_records
                checkpoint_metadata_by_construct[entry.construct_id] = checkpoint_metadata
                checkpoint_details_by_construct[entry.construct_id] = checkpoint_details
        progress = _progress_summary(
            entries,
            mode=mode,
            records_by_construct=records_by_construct,
            details_by_construct=details_by_construct,
            checkpoint_details_by_construct=checkpoint_details_by_construct,
            runtime_budget=runtime_budget,
            new_job_limit=new_job_limit,
        )
        with state_lock:
            run_state["status"] = "paused"
            run_state["pause_reason"] = "max_new_jobs"
            run_state["pause_message"] = str(pause_requested)
            run_state["completed_construct_ids"] = sorted(records_by_construct)
            run_state["checkpoint_job_counts"] = progress["checkpoint_job_counts"]
            run_state["new_jobs_completed_this_invocation"] = progress[
                "new_jobs_completed_this_invocation"
            ]
            run_state["new_spend_usd_this_invocation"] = progress["new_spend_usd_this_invocation"]
            run_state["progress"] = progress
            run_state["budget_state"] = runtime_budget.snapshot()
            _atomic_write_json(run_state, run_state_path)
        return {
            "schema_version": "0.1.0",
            "manifest_type": "vector_prompt_generation",
            "status": "paused",
            "paused": True,
            "pause_reason": "max_new_jobs",
            "pause_message": str(pause_requested),
            "dry_run": False,
            "run_mode": mode,
            "construct_ids": [entry.construct_id for entry in entries],
            "provider": provider,
            "requested_model": model,
            "reasoning_effort": reasoning_effort,
            "runtime_settings": runtime_settings,
            "max_new_jobs": max_new_jobs,
            "progress": progress,
            "run_state_path": str(run_state_path),
            "combined_path": str(combined_path),
            "manifest_path": str(manifest_path),
        }

    combined_records: list[PromptRecord] = []
    construct_manifests: list[dict[str, Any]] = []
    all_specs = {entry.construct_id: entry.spec for entry in entries}
    for entry in entries:
        records = records_by_construct[entry.construct_id]
        validate_prompt_records(records, {entry.construct_id: entry.spec}, require_all_splits=False)
        output_path = output_paths[entry.construct_id]
        if not output_path.exists():
            _atomic_write_records(records, output_path)
            details_by_construct[entry.construct_id]["output_sha256"] = file_sha256(output_path)
        combined_records.extend(records)
        mode_config, _, _ = _mode_settings(
            entry,
            mode,
            input_usd_per_million_tokens=resolved_input_price,
            output_usd_per_million_tokens=resolved_output_price,
        )
        construct_manifests.append(
            _construct_manifest(
                entry,
                output_paths[entry.construct_id],
                details_by_construct[entry.construct_id],
                mode=mode,
                mode_config=mode_config,
                dry_run=False,
                provider=provider,
                requested_model=model,
                reasoning_effort=reasoning_effort,
                runtime_settings=runtime_settings,
            )
        )
    validate_prompt_records(combined_records, all_specs, require_all_splits=False)
    _atomic_write_records(combined_records, combined_path)
    manifest = {
        "schema_version": "0.1.0",
        "manifest_type": "vector_prompt_generation",
        "registry_path": str(Path(registry_path).resolve()),
        "waves": sorted({entry.wave for entry in entries}),
        "construct_ids": [entry.construct_id for entry in entries],
        "scope": "vector",
        "scope_partial": True,
        "run_mode": mode,
        "partial": mode == "review",
        "confirmatory": False,
        "dry_run": False,
        "workers": workers,
        "provider": provider,
        "requested_model": model,
        "reasoning_effort": reasoning_effort,
        "runtime_settings": runtime_settings,
        "input_usd_per_million_tokens": resolved_input_price,
        "output_usd_per_million_tokens": resolved_output_price,
        "cost_safety_multiplier": cost_safety_multiplier,
        "max_estimated_cost_usd": max_estimated_cost_usd,
        "estimated_cost_usd_preflight": estimated_cost,
        "budget_estimate_usd": budget_estimate,
        "runtime_budget": runtime_budget.snapshot(),
        "quality_gate": quality_gate,
        "constructs": construct_manifests,
        "counts": {
            "split_counts": dict(sorted(_vector_counts(combined_records)[0].items())),
            "pair_count": _vector_counts(combined_records)[1],
            "record_count": len(combined_records),
            "request_count": sum(item["request_count"] for item in construct_manifests),
            "estimated_input_tokens": sum(item["estimated_input_tokens"] for item in construct_manifests),
            "estimated_output_tokens": sum(item["estimated_output_tokens"] for item in construct_manifests),
            "estimated_total_tokens": sum(item["estimated_total_tokens"] for item in construct_manifests),
            "estimated_cost_usd": (
                sum(item["estimated_cost_usd"] for item in construct_manifests)
                if all(item["estimated_cost_usd"] is not None for item in construct_manifests)
                else None
            ),
            "actual_input_tokens": sum(item["actual_input_tokens"] for item in construct_manifests),
            "actual_output_tokens": sum(item["actual_output_tokens"] for item in construct_manifests),
            "actual_total_tokens": sum(item["actual_total_tokens"] for item in construct_manifests),
            "actual_cost_usd": sum(item["actual_cost_usd"] for item in construct_manifests),
        },
        "combined_path": str(combined_path),
        "combined_sha256": file_sha256(combined_path),
        "manifest_path": str(manifest_path),
    }
    _atomic_write_json(manifest, manifest_path)
    with state_lock:
        run_state["status"] = "complete"
        run_state["completed_construct_ids"] = sorted(records_by_construct)
        run_state["manifest_path"] = str(manifest_path)
        run_state["budget_state"] = runtime_budget.snapshot()
        _atomic_write_json(run_state, run_state_path)
    return manifest


def _parse_waves(values: list[str]) -> list[int | str]:
    if values == ["all"]:
        return values
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate registry-scoped vector prompt inventories.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--waves", nargs="+", default=["all"], help="all or one or more wave numbers (1-4).")
    parser.add_argument("--constructs", nargs="+", default=None, help="Optional construct-ID subset within waves.")
    parser.add_argument("--mode", choices=("review", "full"), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Plan counts without API calls or file outputs.")
    parser.add_argument("--provider", choices=("openrouter", "openai"), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=None, help="Runtime model override (OpenAI defaults to gpt-5.6-luna).")
    parser.add_argument("--model-alias", default=None, help="Optional alias for an overridden runtime model.")
    parser.add_argument("--reasoning-effort", default=None, help="Provider reasoning setting, e.g. xhigh.")
    parser.add_argument(
        "--max-items-per-request",
        type=int,
        default=None,
        help="Override the effective plan's request chunk size (included in run identity).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override the provider output-token cap (included in run identity).",
    )
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=None,
        help=(
            "Pause a full run after this many newly checkpointed jobs in this invocation; "
            "the limit is resumable invocation bookkeeping and is not part of run identity."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="API-key environment variable (defaults to OPENAI_API_KEY for openai, otherwise OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--max-estimated-cost-usd",
        "--max-budget-usd",
        dest="max_estimated_cost_usd",
        type=float,
        default=None,
        help="Required non-dry spending cap; includes the configured safety margin.",
    )
    parser.add_argument("--input-price-usd-per-million", type=float, default=None)
    parser.add_argument("--output-price-usd-per-million", type=float, default=None)
    parser.add_argument("--cost-safety-multiplier", type=float, default=COST_SAFETY_MULTIPLIER)
    parser.add_argument(
        "--quality-gate-file",
        type=Path,
        default=None,
        help="Approved review-quality artifact required for non-dry full generation.",
    )
    args = parser.parse_args()
    default_api_key_env = "OPENAI_API_KEY" if args.provider == "openai" else "OPENROUTER_API_KEY"
    api_key_env = args.api_key_env or default_api_key_env
    api_key = None if args.dry_run else os.environ.get(api_key_env)
    manifest = orchestrate_vector_prompts(
        registry_path=args.registry,
        waves=_parse_waves(args.waves),
        construct_ids=args.constructs,
        output_dir=args.output_dir,
        mode=args.mode,
        workers=args.workers,
        resume=args.resume,
        dry_run=args.dry_run,
        api_key=api_key,
        provider=args.provider,
        model=args.model,
        model_alias=args.model_alias,
        reasoning_effort=args.reasoning_effort,
        max_items_per_request=args.max_items_per_request,
        max_output_tokens=args.max_output_tokens,
        max_new_jobs=args.max_new_jobs,
        max_estimated_cost_usd=args.max_estimated_cost_usd,
        input_usd_per_million_tokens=args.input_price_usd_per_million,
        output_usd_per_million_tokens=args.output_price_usd_per_million,
        cost_safety_multiplier=args.cost_safety_multiplier,
        quality_gate_file=args.quality_gate_file,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
