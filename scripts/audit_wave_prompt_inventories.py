#!/usr/bin/env python3
"""Audit a frozen wave inventory before model execution.

This is a release audit, not a prompt generator.  It combines the existing
schema-aware downstream audit with a small set of invariants that are easy to
miss when a language model completes a prompt: downstream rows must not carry
the probe-only continuation suffix, and the task family must be reviewed for
near-transfer designs before it is called confirmatory.

The command is intentionally read-only.  It never rewrites an inventory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.generation import (  # noqa: E402
    DOWNSTREAM_PROBE_ONLY_SUFFIX,
    downstream_prompt_text_issues,
)
from construct_benchmark.prompts import PromptRecord, load_prompt_records  # noqa: E402
from construct_benchmark.registry import load_construct_registry  # noqa: E402


VECTOR_SPLITS = frozenset({"direction_train", "direction_validation", "direction_heldout"})
DOWNSTREAM_ROLES = frozenset({"behavior", "steering", "calibration"})
WORD_RE = re.compile(r"[a-z0-9]+")

# These are release blockers for the currently frozen Wave 2--4 v1 plans.  A
# task can be mechanically valid and still fail the benchmark's independence
# requirement when the downstream item asks the same surface question as the
# probe.  Keep these keyed by the task family/construct rather than by prompt
# wording so a future repaired plan can pass without weakening the audit.
_TASK_INDEPENDENCE_BLOCKERS = {
    "reference_frame": {
        "task_family": "reference_dependent_risk_choice",
        "message": (
            "The downstream task is another sure-versus-risky choice while the construct probe changes "
            "relative reference position; release a distal task or explicitly pre-register a transfer "
            "interpretation before confirmatory use."
        ),
    },
    "prior_weighting": {
        "task_family": "prior_evidence_probability_judgment",
        "message": (
            "The downstream task repeats prior-plus-case posterior judgment. This is near transfer, not "
            "an independent behavioral test of prior weighting; release a distal task before confirmation."
        ),
    },
    "authority_deference": {
        "task_family": "authority_evidence_conflict_choice",
        "message": (
            "The downstream task directly repeats the probe's specialist-versus-direct-measurement choice; "
            "release an independent task before confirmation."
        ),
    },
    "exploration_exploitation": {
        "task_family": "two_option_bandit_choice",
        "message": (
            "The downstream task directly repeats the probe's known-versus-new option choice; release an "
            "independent task before confirmation."
        ),
    },
}

# Prompt-level cues that make the transfer target explicit in a frozen item.
# These are intentionally narrow: the plan-level blockers above handle the
# deeper same-task problem, while these catch direct condition text in a
# purportedly independent steering item.
_DIRECT_STEERING_CUE_RULES = {
    "reference_frame": (
        "comparison point",
        "reference point",
        "benchmark",
        "relative to",
    ),
    "temporal_orientation": (
        "near-term consequence",
        "longer-term consequence",
        "long-term consequence",
        "prioritize the near",
        "primary weight to the longer",
    ),
    "goal_shielding": (
        "receives attentional priority",
        "given attentional priority",
        "prioritize the focal",
        "focal objective receives",
    ),
}


def _load_downstream_module() -> Any:
    """Load the CLI module without executing its command-line entry point."""

    module_name = "construct_benchmark_downstream_audit"
    module_path = _ROOT / "scripts" / "generate_downstream_prompts.py"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not load downstream audit module: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def _normalise(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def _pair_summary(records: Iterable[PromptRecord]) -> dict[str, Any]:
    vector = [record for record in records if record.split in VECTOR_SPLITS]
    groups: dict[tuple[str, str, str], list[PromptRecord]] = defaultdict(list)
    for record in vector:
        groups[(record.construct_id, record.split, str(record.pair_id))].append(record)
    malformed = [key for key, values in groups.items() if len(values) != 2]
    duplicate_text_count = len(vector) - len({_normalise(record.prompt_text) for record in vector})
    cross_split: dict[str, set[str]] = defaultdict(set)
    for record in vector:
        cross_split[_normalise(record.prompt_text)].add(record.split)
    cross_split_duplicates = sum(1 for splits in cross_split.values() if len(splits) > 1)
    return {
        "record_count": len(vector),
        "pair_group_count": len(groups),
        "malformed_pair_count": len(malformed),
        "malformed_pair_examples": [list(key) for key in malformed[:10]],
        "normalized_duplicate_count": duplicate_text_count,
        "cross_split_normalized_duplicate_count": cross_split_duplicates,
        "split_counts": dict(sorted(Counter(record.split for record in vector).items())),
    }


def _downstream_invariant_flags(records: Iterable[PromptRecord]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for record in records:
        if record.prompt_role not in DOWNSTREAM_ROLES:
            continue
        folded = record.prompt_text.casefold()
        composition_issues = downstream_prompt_text_issues(record.prompt_text)
        for issue in composition_issues:
            flags.append(
                {
                    "severity": "severe",
                    "flag_type": (
                        "probe_suffix_in_downstream_prompt"
                        if DOWNSTREAM_PROBE_ONLY_SUFFIX in folded
                        else "downstream_prompt_composition"
                    ),
                    "prompt_id": record.prompt_id,
                    "construct_id": record.construct_id,
                    "prompt_role": record.prompt_role,
                    "message": issue,
                }
            )
        # A downstream task may legitimately say that its own item is prior
        # contact, prior evidence, or a prior helper.  Do not treat those
        # ordinary task facts as a probe-leakage warning.  Explicit generation
        # instructions such as "do not mention an earlier scenario" are
        # already caught by downstream_prompt_text_issues above.
    return flags


def _independence_flags(
    records: Iterable[PromptRecord],
    entries: Iterable[Any],
) -> list[dict[str, Any]]:
    """Report known same-task and direct-cue blockers for frozen inventories."""

    flags: list[dict[str, Any]] = []
    materialized = tuple(records)
    for entry in entries:
        rule = _TASK_INDEPENDENCE_BLOCKERS.get(entry.construct_id)
        if rule and entry.spec.independent_behavior_task.get("task_family") == rule["task_family"]:
            flags.append(
                {
                    "severity": "severe",
                    "flag_type": "downstream_task_independence_blocker",
                    "construct_id": entry.construct_id,
                    "task_id": entry.spec.independent_behavior_task.get("task_id"),
                    "task_family": rule["task_family"],
                    "message": rule["message"],
                }
            )

    for record in materialized:
        if record.prompt_role != "steering":
            continue
        terms = _DIRECT_STEERING_CUE_RULES.get(record.construct_id, ())
        hits = tuple(term for term in terms if term in record.prompt_text.casefold())
        if hits:
            flags.append(
                {
                    "severity": "severe",
                    "flag_type": "direct_construct_cue_in_steering_prompt",
                    "prompt_id": record.prompt_id,
                    "construct_id": record.construct_id,
                    "prompt_role": record.prompt_role,
                    "terms": list(hits),
                    "message": (
                        "Steering prompt states the construct-relevant target state directly; this can "
                        "confound a zero-dose downstream transfer comparison."
                    ),
                }
            )
    return flags


def _probe_wrapper_flags(records: Iterable[PromptRecord], specs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Require every probe row to use its registered wrapper exactly."""

    flags: list[dict[str, Any]] = []
    for record in records:
        if record.split not in VECTOR_SPLITS:
            continue
        spec = specs.get(record.construct_id)
        template = getattr(spec, "probe_prompt_template", None)
        if not isinstance(template, str) or "{scenario}" not in template:
            continue
        prefix, suffix = template.split("{scenario}", maxsplit=1)
        text = record.prompt_text
        issues: list[str] = []
        if not text.startswith(prefix):
            issues.append("missing registered wrapper prefix")
        if not text.endswith(suffix):
            issues.append("missing registered wrapper suffix")
        if text.count("Scenario:") != 1:
            issues.append("scenario marker count is not exactly one")
        if issues:
            flags.append(
                {
                    "severity": "severe",
                    "flag_type": "probe_wrapper_violation",
                    "prompt_id": record.prompt_id,
                    "construct_id": record.construct_id,
                    "split": record.split,
                    "issues": issues,
                }
            )
    return flags


def _scientific_warnings(entries: Iterable[Any]) -> list[dict[str, Any]]:
    task_family_warnings = {
        "reference_dependent_risk_choice": (
            "The downstream task is a sure-versus-risky choice; review whether reference-frame state "
            "transfer is being distinguished from a generic risk-choice response."
        ),
        "prior_evidence_probability_judgment": (
            "The downstream task repeats numerical prior/evidence updating; treat it as near-transfer "
            "unless the independent-task rationale is explicitly defended."
        ),
        "ambiguity_sensitive_allocation": (
            "The downstream task directly repeats bounded-probability ambiguity; review its independence "
            "from the probe before confirmatory use."
        ),
        "causal_prediction": (
            "The downstream task still asks for an outcome probability; verify that the causal-versus-selected "
            "representation is not being read out by the same wording as the probe."
        ),
    }
    flags: list[dict[str, Any]] = []
    for entry in entries:
        task = entry.spec.independent_behavior_task
        message = task_family_warnings.get(str(task.get("task_family")))
        if message:
            flags.append(
                {
                    "severity": "warning",
                    "flag_type": "near_transfer_task_family",
                    "construct_id": entry.construct_id,
                    "task_id": task.get("task_id"),
                    "task_family": task.get("task_family"),
                    "message": message,
                }
            )
    return flags


def audit_wave_inventory(
    inventory_path: str | Path,
    *,
    registry_path: str | Path,
    wave: int,
) -> dict[str, Any]:
    """Return a JSON-serializable audit for one combined wave inventory."""

    inventory = Path(inventory_path).resolve()
    registry_path = Path(registry_path).resolve()
    if not inventory.exists():
        raise ValueError(f"Inventory does not exist: {inventory}")
    registry = load_construct_registry(registry_path)
    registry_entries = [entry for entry in registry.entries if entry.wave == wave]
    if len(registry_entries) != 4:
        raise ValueError(f"Wave {wave} must have exactly four registry entries.")
    records = tuple(load_prompt_records(inventory))
    ids = {entry.construct_id for entry in registry_entries}
    observed_ids = {record.construct_id for record in records}
    if observed_ids != ids:
        raise ValueError(f"Inventory construct IDs {sorted(observed_ids)} do not match wave {wave}: {sorted(ids)}")

    downstream_module = _load_downstream_module()
    entries = downstream_module._effective_entries(
        registry_path,
        waves=[wave],
        construct_ids=None,
        batch_size=20,
        max_output_tokens=30_000,
        model="gpt-5.6-luna",
    )
    downstream_records = tuple(record for record in records if record.prompt_role in DOWNSTREAM_ROLES)
    specs = {entry.construct_id: entry.spec for entry in entries}
    per_construct: dict[str, Any] = {}
    hard_flags: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for entry in entries:
        construct_records = tuple(record for record in records if record.construct_id == entry.construct_id)
        expected = downstream_module.dry_run_summary(entry.plan, model_aliases={"luna"})
        observed_counts = Counter(record.split for record in construct_records)
        expected_counts = Counter(expected["records_by_split"])
        if observed_counts != expected_counts:
            hard_flags.append(
                {
                    "severity": "severe",
                    "flag_type": "count_mismatch",
                    "construct_id": entry.construct_id,
                    "observed": dict(sorted(observed_counts.items())),
                    "expected": dict(sorted(expected_counts.items())),
                }
            )
        per_construct[entry.construct_id] = {
            "observed_counts": dict(sorted(observed_counts.items())),
            "expected_counts": dict(sorted(expected_counts.items())),
            "task_id": entry.spec.independent_behavior_task["task_id"],
            "task_family": entry.spec.independent_behavior_task.get("task_family"),
        }

    downstream_audit = downstream_module.audit_downstream_inventory(
        downstream_records,
        entries,
        vector_reference=None,
    )
    for flag in downstream_audit["flags"]:
        (hard_flags if flag.get("severity") == "severe" else warnings).append(dict(flag))
    invariant_flags = _downstream_invariant_flags(records)
    wrapper_flags = _probe_wrapper_flags(records, specs)
    hard_flags.extend(flag for flag in invariant_flags if flag["severity"] == "severe")
    hard_flags.extend(wrapper_flags)
    warnings.extend(flag for flag in invariant_flags if flag["severity"] == "warning")
    independence_flags = _independence_flags(records, entries)
    hard_flags.extend(flag for flag in independence_flags if flag["severity"] == "severe")
    warnings.extend(flag for flag in independence_flags if flag["severity"] == "warning")
    warnings.extend(_scientific_warnings(entries))
    pair_summary = _pair_summary(records)
    if pair_summary["malformed_pair_count"] or pair_summary["normalized_duplicate_count"]:
        hard_flags.append(
            {
                "severity": "severe",
                "flag_type": "vector_pair_structure",
                "message": "Vector pair structure contains malformed pairs or normalized duplicates.",
                **pair_summary,
            }
        )
    return {
        "audit_version": "2",
        "inventory_path": str(inventory),
        "registry_path": str(registry_path),
        "wave": wave,
        "record_count": len(records),
        "vector": pair_summary,
        "downstream": {
            key: downstream_audit[key]
            for key in ("record_count", "flag_count", "severe_flag_count", "warning_flag_count", "passed")
        },
        "per_construct": per_construct,
        "severe_flag_count": len(hard_flags),
        "warning_count": len(warnings),
        "passed": not hard_flags,
        "severe_flags": hard_flags,
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit one combined wave prompt inventory.")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--wave", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-severe", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = audit_wave_inventory(args.inventory, registry_path=args.registry, wave=args.wave)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.fail_on_severe and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
