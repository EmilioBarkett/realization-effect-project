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
from construct_benchmark.prompts import load_prompt_records  # noqa: E402


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
    inventory_records = load_prompt_records(prompt_inventory)
    inventory_prompt_ids = sorted(
        {
            record.prompt_id
            for record in inventory_records
            if record.construct_id == construct_id and record.split == "steering_eval"
        }
    )
    if not prompt_ids.issubset(set(inventory_prompt_ids)):
        missing_inventory_ids = sorted(prompt_ids - set(inventory_prompt_ids))
        raise ValueError(
            f"Preflight selection contains steering IDs absent from the prompt inventory: "
            f"{missing_inventory_ids[:5]}"
        )
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
    source_prompt_ids = sorted({str(value.get("prompt_id")) for value in source_conditions})
    source_prompt_id_set = set(source_prompt_ids)
    prompt_rebind: dict[str, str] = {}
    if not prompt_ids.issubset(source_prompt_id_set):
        if len(source_prompt_ids) != len(inventory_prompt_ids) or source_prompt_id_set.intersection(
            inventory_prompt_ids
        ):
            raise ValueError(
                "Steering plan prompt IDs do not cover the frozen inventory and cannot be "
                "rebound deterministically."
            )
        # A direction plan can be reused for a new, independent downstream
        # inventory when the construct/split cardinality is unchanged.  Bind
        # IDs by their canonical lexical schedule, never by model outputs.
        prompt_rebind = {
            inventory_id: source_id
            for source_id, inventory_id in zip(
                source_prompt_ids, inventory_prompt_ids, strict=True
            )
        }
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
    synthesized_zero_dose_kinds: set[str] = set()
    missing: list[str] = []
    for prompt_id in sorted(prompt_ids):
        source_prompt_id = prompt_rebind.get(prompt_id, prompt_id)
        for kind, dose in sorted(required_pairs):
            choices = candidates.get((source_prompt_id, kind, dose), [])
            if not choices and dose == 0.0 and kind in {"shuffled", "random"}:
                # The frozen full plans register nonzero control doses but omit
                # the zero-dose rows required by the v2 model-side gate.  A
                # zero-dose control is still outcome-independent: reuse the
                # lowest-index registered control vector for this prompt and
                # set its physical intervention to exactly zero.  Preserve the
                # source identity so this repair cannot be mistaken for an
                # original confirmatory condition.
                nonzero_choices = [
                    candidate
                    for (candidate_prompt, candidate_kind, candidate_dose), values in candidates.items()
                    if candidate_prompt == source_prompt_id
                    and candidate_kind == kind
                    and candidate_dose != 0.0
                    for candidate in values
                ]
                if nonzero_choices:
                    source = min(
                        nonzero_choices,
                        key=lambda value: (
                            int(value.get("direction_index", 0)),
                            abs(float(value.get("dose", 0.0))),
                            str(value.get("condition_id", "")),
                        ),
                    )
                    synthesized = dict(source)
                    synthesized["source_condition_id"] = source.get("condition_id")
                    synthesized["source_prompt_id"] = source_prompt_id
                    synthesized["prompt_id"] = prompt_id
                    synthesized["condition_id"] = (
                        f"{prompt_id}__preflight_{kind}_zero_dose"
                    )
                    synthesized["dose"] = 0.0
                    synthesized["physical_scale"] = 0.0
                    synthesized["preflight_synthesized_zero_dose"] = True
                    selected_conditions.append(synthesized)
                    synthesized_zero_dose_kinds.add(kind)
                    continue
            if not choices:
                missing.append(f"{prompt_id}:{kind}:{dose:g}")
                continue
            # Full plans may register multiple random controls.  The lowest
            # direction index is a frozen, outcome-independent choice.
            chosen = min(choices, key=lambda value: int(value.get("direction_index", 0)))
            if source_prompt_id != prompt_id:
                rebound = dict(chosen)
                rebound["source_condition_id"] = chosen.get("condition_id")
                rebound["source_prompt_id"] = source_prompt_id
                rebound["prompt_id"] = prompt_id
                rebound["condition_id"] = (
                    f"{prompt_id}__preflight_{kind}_{int(rebound.get('direction_index', 0)):02d}"
                    f"__dose_{dose:g}"
                )
                chosen = rebound
            selected_conditions.append(chosen)
    if missing:
        raise ValueError(f"Steering plan is missing required preflight groups: {missing[:5]}")

    derived = copy.deepcopy(dict(plan))
    derived["conditions"] = selected_conditions
    derived["source_condition_count"] = len(source_conditions)
    derived["selected_condition_count"] = len(selected_conditions)
    derived["preflight_rebound_prompt_count"] = len(prompt_rebind)
    if prompt_rebind:
        derived["preflight_prompt_rebind_rule"] = (
            "The source direction plan used a prior inventory with the same construct/split "
            "cardinality. Source prompt IDs were rebound to the frozen inventory by sorted "
            "prompt-ID order without inspecting outputs."
        )
    derived["preflight_synthesized_zero_dose_kinds"] = sorted(synthesized_zero_dose_kinds)
    if synthesized_zero_dose_kinds:
        derived["preflight_zero_dose_control_rule"] = (
            "For missing zero-dose shuffled/random controls only, retain the lowest-index "
            "registered nonzero control vector and set dose and physical_scale to zero; "
            "these rows are diagnostic preflight repairs and are not source-plan conditions."
        )
    derived["execution_scope"] = "model_behavior_accessibility_preflight_v2"
    derived["confirmatory"] = False
    derived["preflight_selection_sha256"] = selection.get("selection_sha256")
    derived["preflight_prompt_inventory_sha256"] = actual_inventory_hash
    derived["preflight_selection_rule"] = (
        "For each selected steering prompt and required kind/dose pair, retain the registered "
        "lowest direction_index condition without inspecting model outputs; synthesize only "
        "missing zero-dose shuffled/random controls from registered nonzero controls; when a "
        "same-cardinality source plan uses prior prompt IDs, rebind by sorted ID order."
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
