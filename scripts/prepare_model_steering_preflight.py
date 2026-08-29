#!/usr/bin/env python3
"""Derive a tiny construct-pure steering plan from a frozen v2 selection.

The full steering plan remains immutable.  This adapter chooses one registered
condition for every selected prompt and required direction/dose group, then
rewrites only the provenance fields needed to bind the derived plan to the
preflight inventory.  It performs no model inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return value


def _selected_prompt_ids(selection: Mapping[str, Any], construct_id: str) -> list[str]:
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Selection manifest is missing selected construct entries.")
    construct = selected.get(construct_id)
    if not isinstance(construct, Mapping):
        raise ValueError(f"Selection manifest has no entry for {construct_id}.")
    steering = construct.get("steering_eval")
    if not isinstance(steering, Mapping):
        raise ValueError(f"Selection manifest has no steering_eval entry for {construct_id}.")
    prompt_ids = steering.get("prompt_ids")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or len(set(prompt_ids)) != len(prompt_ids)
    ):
        raise ValueError(f"Selection manifest has invalid steering prompt IDs for {construct_id}.")
    return [str(value) for value in prompt_ids]


def derive_preflight_plan(
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    prompt_inventory: Path,
) -> dict[str, Any]:
    """Return a derived preflight plan with exact required groups."""

    if selection.get("manifest_type") != "model_behavior_accessibility_preflight_selection":
        raise ValueError("Unexpected model-side preflight selection manifest type.")
    if selection.get("preflight_id") != "model_behavior_accessibility_v2":
        raise ValueError("The steering preflight requires a v2 selection manifest.")
    if plan.get("model") != selection.get("model"):
        raise ValueError("Steering plan and preflight selection model metadata differ.")
    expected_inventory_hash = selection.get("source_inventory_sha256")
    actual_inventory_hash = file_sha256(prompt_inventory)
    if expected_inventory_hash != actual_inventory_hash:
        raise ValueError("Prompt inventory does not match the frozen preflight selection.")

    construct_id = str(plan.get("construct_id"))
    prompt_ids = set(_selected_prompt_ids(selection, construct_id))
    requirements = selection.get("steering_requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("Selection manifest is missing steering requirements.")
    doses_by_kind = requirements.get("required_doses_by_direction_kind")
    if not isinstance(doses_by_kind, Mapping) or not doses_by_kind:
        raise ValueError("Selection manifest has no required steering dose groups.")
    required_pairs = {
        (str(kind), float(dose))
        for kind, doses in doses_by_kind.items()
        for dose in doses
    }

    source_conditions = list(plan.get("conditions", []))
    if not source_conditions or any(not isinstance(value, Mapping) for value in source_conditions):
        raise ValueError("Steering plan must contain a non-empty conditions list.")
    candidates: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for raw in source_conditions:
        condition = dict(raw)
        key = (
            str(condition.get("prompt_id")),
            str(condition.get("direction_kind")),
            float(condition.get("dose")),
        )
        candidates.setdefault(key, []).append(condition)

    selected_conditions: list[dict[str, Any]] = []
    missing: list[str] = []
    for prompt_id in sorted(prompt_ids):
        for kind, dose in sorted(required_pairs):
            choices = candidates.get((prompt_id, kind, dose), [])
            if not choices:
                missing.append(f"{prompt_id}:{kind}:{dose:g}")
                continue
            # Full plans may register multiple random controls.  The lowest
            # direction index is a frozen, outcome-independent choice.
            chosen = min(choices, key=lambda value: int(value.get("direction_index", 0)))
            selected_conditions.append(chosen)
    if missing:
        raise ValueError(f"Steering plan is missing required preflight groups: {missing[:5]}")

    derived = copy.deepcopy(dict(plan))
    derived["conditions"] = selected_conditions
    derived["source_condition_count"] = len(source_conditions)
    derived["selected_condition_count"] = len(selected_conditions)
    derived["execution_scope"] = "model_behavior_accessibility_preflight_v2"
    derived["confirmatory"] = False
    derived["preflight_selection_sha256"] = selection.get("selection_sha256")
    derived["preflight_prompt_inventory_sha256"] = actual_inventory_hash
    derived["preflight_selection_rule"] = (
        "For each selected steering prompt and required kind/dose pair, retain the registered "
        "lowest direction_index condition without inspecting model outputs."
    )
    provenance = dict(derived.get("provenance", {}))
    provenance["source_prompt_inventory_sha256"] = provenance.get("prompt_inventory_sha256")
    provenance["prompt_inventory_sha256"] = actual_inventory_hash
    provenance["preflight_selection_sha256"] = selection.get("selection_sha256")
    derived["provenance"] = provenance
    return derived


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steering-plan", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--prompt-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    derived = derive_preflight_plan(
        _read_object(args.steering_plan),
        _read_object(args.selection_manifest),
        prompt_inventory=args.prompt_inventory,
    )
    derived["derived_plan_sha256"] = canonical_hash(
        {key: value for key, value in derived.items() if key != "derived_plan_sha256"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "construct_id": derived.get("construct_id"),
                "source_condition_count": derived["source_condition_count"],
                "selected_condition_count": derived["selected_condition_count"],
                "prompt_inventory_sha256": derived["preflight_prompt_inventory_sha256"],
                "derived_plan_sha256": derived["derived_plan_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
