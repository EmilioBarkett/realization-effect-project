"""Deterministic selection of prompt inventories for model-side run modes.

The complete generated inventory is the scientific source of truth. A test
run is a derived, hashable subset used only to exercise the model/runtime
path. It preserves complete paired probes and at least one item in every
required split so that a short run cannot silently become an unlabelled
prefix of the confirmatory inventory.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from .prompts import PromptRecord, validate_prompt_records
from .schemas import ConstructSpec, RunConfig


def resolve_run_mode(run_config: RunConfig, mode: str | None) -> tuple[str, dict[str, Any]]:
    """Resolve an explicit run mode, defaulting only when the caller omits it."""

    mode_id = str(mode or run_config.execution["default_run_mode"])
    modes = run_config.execution["run_modes"]
    if mode_id not in modes:
        raise ValueError(
            f"Unknown run mode={mode_id!r}; available modes are {sorted(modes)}."
        )
    return mode_id, dict(modes[mode_id])


def _stable_rank(seed: int, *parts: str) -> str:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _counts_by_construct_split(records: Iterable[PromptRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        construct_counts = counts.setdefault(record.construct_id, {})
        construct_counts[record.split] = construct_counts.get(record.split, 0) + 1
    return {
        construct_id: dict(sorted(split_counts.items()))
        for construct_id, split_counts in sorted(counts.items())
    }


def select_prompt_records(
    records: Iterable[PromptRecord],
    *,
    run_config: RunConfig,
    construct_specs: Mapping[str, ConstructSpec],
    mode: str | None = None,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Select and validate a complete or bounded inventory for a run mode.

    ``full`` preserves every record. ``test`` selects complete paired probe
    groups and a deterministic subset of independent-task items for each
    configured construct and required split. The selected records retain the
    source inventory order so downstream batching remains reproducible.
    """

    materialized = list(records)
    if not materialized:
        raise ValueError("Prompt inventory must contain at least one record.")
    mode_id, mode_config = resolve_run_mode(run_config, mode)
    validate_prompt_records(materialized, construct_specs)

    selection = dict(mode_config["prompt_selection"])
    strategy = str(selection["strategy"])
    selected_ids: set[str]
    if strategy == "all":
        selected_ids = {record.prompt_id for record in materialized}
    elif strategy == "balanced_pair_preserving":
        max_pairs = int(selection["max_pairs_per_paired_split"])
        max_items = int(selection["max_items_per_single_split"])
        selected_ids = set()
        seed = int(run_config.seed)
        for construct_id in run_config.construct_ids:
            spec = construct_specs[construct_id]
            for split in spec.required_splits:
                split_records = [
                    record
                    for record in materialized
                    if record.construct_id == construct_id and record.split == split
                ]
                if split in spec.paired_splits:
                    pair_groups: dict[str, list[PromptRecord]] = {}
                    for record in split_records:
                        if not record.pair_id:
                            raise ValueError(
                                f"Test selection requires pair_id for paired prompt {record.prompt_id}."
                            )
                        pair_groups.setdefault(record.pair_id, []).append(record)
                    ranked_pairs = sorted(
                        pair_groups,
                        key=lambda pair_id: _stable_rank(seed, construct_id, split, pair_id),
                    )[:max_pairs]
                    for pair_id in ranked_pairs:
                        selected_ids.update(record.prompt_id for record in pair_groups[pair_id])
                else:
                    ranked_records = sorted(
                        split_records,
                        key=lambda record: _stable_rank(seed, construct_id, split, record.prompt_id),
                    )[:max_items]
                    selected_ids.update(record.prompt_id for record in ranked_records)
    else:  # Defensive guard for configs created outside RunConfig validation.
        raise ValueError(f"Unsupported prompt selection strategy={strategy!r}.")

    selected = [record for record in materialized if record.prompt_id in selected_ids]
    validate_prompt_records(selected, construct_specs)
    manifest = {
        "schema_version": run_config.schema_version,
        "manifest_type": "benchmark_run_mode_selection",
        "run_id": run_config.run_id,
        "mode": mode_id,
        "purpose": mode_config["purpose"],
        "confirmatory": bool(mode_config["confirmatory"]),
        "max_runtime_minutes": mode_config["max_runtime_minutes"],
        "prompt_selection": selection,
        "source_prompt_count": len(materialized),
        "selected_prompt_count": len(selected),
        "complete_inventory": mode_id == "full" and len(selected) == len(materialized),
        "construct_ids": list(run_config.construct_ids),
        "source_counts_by_construct_split": _counts_by_construct_split(materialized),
        "selected_counts_by_construct_split": _counts_by_construct_split(selected),
        "selected_prompt_ids": [record.prompt_id for record in selected],
    }
    return selected, manifest


__all__ = ["resolve_run_mode", "select_prompt_records"]
