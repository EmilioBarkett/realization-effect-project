#!/usr/bin/env python3
"""Prepare immutable Wave 1 expansion specifications and generation plans.

This script never edits an existing v2 specification, plan, or inventory.  It
creates v3 construct specs with an explicitly unrelated collateral task,
three downstream-only supplement overlays for the underpowered constructs, and
one collateral overlay per Wave 1 construct.  API generation and final
freezing are separate steps so reviewed prompts can be audited before release.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "configs/construct_benchmark/constructs"
PLAN_DIR = ROOT / "configs/construct_benchmark/generation_plans"
OUT_DIR = ROOT / "configs/construct_benchmark/release"

WAVE1 = {
    "realization_account_closure": {
        "spec": "realization_account_closure_v2.json",
        "collateral_domains": ["mineral identification", "electrical materials", "geology", "weather instruments"],
        "required_anchors": ["mineral", "material", "rock", "instrument"],
        "forbidden_anchors": ["cargo", "reserve", "telescope", "microgrant", "account", "settled"],
    },
    "evidence_diagnosticity": {
        "spec": "evidence_diagnosticity_v2.json",
        "collateral_domains": ["bird biology", "plant biology", "animal anatomy", "ecology"],
        "required_anchors": ["species", "organism", "plant", "animal"],
        "forbidden_anchors": ["test", "hypothesis", "evidence", "diagnostic", "posterior", "probability"],
    },
    "source_reliability": {
        "spec": "source_reliability_v2.json",
        "collateral_domains": ["astronomy", "planetary science", "weather science", "geography"],
        "required_anchors": ["planet", "star", "weather", "map"],
        "forbidden_anchors": ["source", "report", "measurement", "reliable", "authority", "testimony"],
    },
    "persistence_continuation": {
        "spec": "persistence_continuation_v2.json",
        "collateral_domains": ["grammar", "vocabulary", "arithmetic", "geometry"],
        "required_anchors": ["sentence", "word", "calculation", "shape"],
        "forbidden_anchors": ["goal", "plan", "continue", "abandon", "setback", "effort"],
    },
}

SUPPLEMENT_IDS = {
    "realization_account_closure": "realization_account_closure_supplement_v1",
    "evidence_diagnosticity": "evidence_diagnosticity_supplement_v1",
    "source_reliability": "source_reliability_supplement_v1",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing release file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collateral_task(construct_id: str) -> dict[str, Any]:
    task_id = f"collateral_factual_choice_{construct_id}_v1"
    return {
        "task_id": task_id,
        "task_family": "unrelated_factual_two_option_choice",
        "prompt_template": (
            "Answer the unrelated factual question below. Choose the option that is factually correct. "
            "Return exactly one integer: 1 for option 1 or 2 for option 2. Use only the information in this item."
        ),
        "primary_outcome": "correct_option",
        "secondary_outcomes": ["choice_accuracy", "compliance"],
        "response_format": "single_integer_1_or_2",
        "item_metadata_schema": {
            "required": ["correct_option", "difficulty", "option_order", "domain_family"],
            "properties": {
                "correct_option": {"type": "integer", "enum": [1, 2]},
                "difficulty": {"type": "string", "enum": ["low", "high"]},
                "option_order": {"type": "string", "enum": ["option_1_first", "option_2_first"]},
                "domain_family": {"type": "string", "enum": ["domain_a", "domain_b", "domain_c", "domain_d"]},
            },
        },
        "independence_requirements": [
            "Use a concrete factual question unrelated to the construct probe and primary downstream task.",
            "Do not ask the model to classify, describe, or reason about the target construct.",
            "Make exactly one option factually correct and keep the alternative plausible but incorrect.",
            "Use the registered metadata schedule exactly; option order must not change which integer is correct.",
        ],
    }


def _balanced_schedule(values: list[Any], count: int = 32) -> list[Any]:
    if count % len(values):
        raise ValueError("Schedule count must be divisible by the number of values.")
    return values * (count // len(values))


def _collateral_cell(construct_id: str, pool_id: str, domains: list[str]) -> dict[str, Any]:
    return {
        "cell_id": "collateral_eval",
        "split": "collateral_eval",
        "prompt_role": "collateral",
        "mode": "single",
        "prompt_family": f"{construct_id}_collateral_eval_v1",
        "content_pool": pool_id,
        "condition_id": "neutral",
        "task_id": f"collateral_factual_choice_{construct_id}_v1",
        "parser_id": "single_integer_choice_1_or_2_v1",
        "expected_output_format": "single_integer_1_or_2",
        "count_per_model": 32,
        "category_balance": {
            "correct_option": _balanced_schedule([1, 2]),
            "difficulty": _balanced_schedule(["low", "high"]),
            "option_order": _balanced_schedule(["option_1_first", "option_2_first"]),
            "domain_family": _balanced_schedule(["domain_a", "domain_b", "domain_c", "domain_d"]),
        },
        "instructions": (
            "Generate one independent factual two-option question. The topic must be unrelated to the "
            "construct and its primary downstream task. Use the assigned domain, make the registered "
            "correct_option true, vary difficulty and option order as scheduled, and include exactly one "
            "response contract at the end. Do not mention a probe, construct, behavior, or benchmark."
        ),
        "metadata": {"collateral_domains": domains},
    }


def _make_v3_spec(construct_id: str, base: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(base)
    spec["version"] = "v3"
    required = list(spec.get("required_splits", []))
    if "collateral_eval" not in required:
        required.append("collateral_eval")
    spec["required_splits"] = required
    spec["collateral_behavior_task"] = _collateral_task(construct_id)
    metadata = dict(spec.get("metadata") or {})
    metadata["wave1_release_v3"] = {
        "purpose": "independent collateral behavior control",
        "source_spec_version": "v2",
        "collateral_task_id": spec["collateral_behavior_task"]["task_id"],
    }
    spec["metadata"] = metadata
    return spec


def _make_supplement_plan(construct_id: str, base: dict[str, Any]) -> dict[str, Any]:
    plan_id = SUPPLEMENT_IDS[construct_id]
    pools = {
        f"{construct_id}_supplement_behavior": {
            "role": "behavior",
            "domains": [f"{construct_id}_supplement_behavior_{index}" for index in range(1, 5)],
        },
        f"{construct_id}_supplement_steering": {
            "role": "steering",
            "domains": [f"{construct_id}_supplement_steering_{index}" for index in range(1, 5)],
        },
        f"{construct_id}_supplement_calibration": {
            "role": "calibration",
            "domains": [f"{construct_id}_supplement_calibration_{index}" for index in range(1, 5)],
        },
    }
    cells = {}
    for split, role in (("behavior_eval", "behavior"), ("steering_eval", "steering"), ("calibration", "calibration")):
        cells[split] = {
            "prompt_family": f"{construct_id}_supplement_{split}_v1",
            "content_pool": f"{construct_id}_supplement_{role}",
            "count_per_model": 16,
            "instructions": (
                "Generate a fresh independent downstream item using a domain not present in the original "
                "Wave 1 inventory. Preserve the registered category schedule and task response contract; "
                "do not mention the probe, construct, or prompt-generation process."
            ),
        }
    return {
        "schema_version": base.get("schema_version", "0.1.0"),
        "plan_id": plan_id,
        "construct_id": construct_id,
        "construct_spec_path": f"../constructs/{WAVE1[construct_id]['spec']}",
        "base_plan_path": f"wave1_{construct_id}_v2.json",
        "overrides": {
            "plan_id": plan_id,
            "content_pools": pools,
            "cells": cells,
        },
    }


def _make_collateral_plan(construct_id: str, base: dict[str, Any]) -> dict[str, Any]:
    info = WAVE1[construct_id]
    pool_id = f"{construct_id}_collateral"
    separation = copy.deepcopy(base.get("downstream_pool_separation", {}))
    separation.setdefault("required_prompt_anchors", {})[pool_id] = info["required_anchors"]
    separation.setdefault("forbidden_prompt_anchors", {})[pool_id] = info["forbidden_anchors"]
    return {
        "schema_version": base.get("schema_version", "0.1.0"),
        "plan_id": f"{construct_id}_collateral_v1",
        "construct_id": construct_id,
        "construct_spec_path": f"../constructs/{construct_id}_v3.json",
        "base_plan_path": f"wave1_{construct_id}_v2.json",
        "overrides": {
            "plan_id": f"{construct_id}_collateral_v1",
            "content_pools": {
                pool_id: {"role": "collateral", "domains": info["collateral_domains"]},
            },
            "downstream_pool_separation": separation,
            "append_cells": [_collateral_cell(construct_id, pool_id, info["collateral_domains"])],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "manifest_type": "wave1_release_v3_preparation",
        "source": "immutable v2 specs/plans",
        "supplement_constructs": sorted(SUPPLEMENT_IDS),
        "constructs": {},
    }
    for construct_id, info in WAVE1.items():
        base_spec_path = SPEC_DIR / info["spec"]
        base_spec = _read_json(base_spec_path)
        v3_spec = _make_v3_spec(construct_id, base_spec)
        v3_spec_path = SPEC_DIR / f"{construct_id}_v3.json"
        _write_new(v3_spec_path, v3_spec)

        base_plan_path = PLAN_DIR / f"wave1_{construct_id}_v2.json"
        base_plan = _read_json(base_plan_path)
        collateral_plan = _make_collateral_plan(construct_id, base_plan)
        collateral_plan_path = PLAN_DIR / f"wave1_{construct_id}_collateral_v1.json"
        _write_new(collateral_plan_path, collateral_plan)

        entry = {
            "source_spec": str(base_spec_path.relative_to(ROOT)),
            "source_spec_sha256": _sha256(base_spec_path),
            "v3_spec": str(v3_spec_path.relative_to(ROOT)),
            "v3_spec_sha256": _sha256(v3_spec_path),
            "collateral_plan": str(collateral_plan_path.relative_to(ROOT)),
            "collateral_plan_sha256": _sha256(collateral_plan_path),
        }
        if construct_id in SUPPLEMENT_IDS:
            supplement_plan = _make_supplement_plan(construct_id, base_plan)
            supplement_path = PLAN_DIR / f"wave1_{construct_id}_supplement_v1.json"
            _write_new(supplement_path, supplement_plan)
            entry["supplement_plan"] = str(supplement_path.relative_to(ROOT))
            entry["supplement_plan_sha256"] = _sha256(supplement_path)
        manifest["constructs"][construct_id] = entry
    manifest_path = output_dir / "wave1_release_v3_preparation.json"
    _write_new(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "constructs": sorted(WAVE1)}, indent=2))


if __name__ == "__main__":
    main()
