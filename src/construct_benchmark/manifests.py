"""Run plans and provenance manifests for shared multi-construct execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import validate_analysis_spec, validate_run_constructs
from .prompts import PromptRecord, load_prompt_records, validate_prompt_records
from .run_modes import resolve_run_mode
from .schemas import AnalysisSpec, ConstructSpec, RunConfig, SUPPORTED_SCHEMA_VERSIONS
from .splits import SPLIT_EXECUTION_SCOPE


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _construct_entry(spec: ConstructSpec, root: Path) -> dict[str, Any]:
    construct_root = root / "constructs" / spec.construct_id
    return {
        "construct_id": spec.construct_id,
        "version": spec.version,
        "family": spec.family,
        "spec_hash": canonical_hash(spec.to_mapping()),
        "condition_ids": list(spec.condition_ids),
        "positive_condition_id": spec.positive_condition_id,
        "negative_condition_id": spec.negative_condition_id,
        "required_splits": list(spec.required_splits),
        "paired_splits": list(spec.paired_splits),
        "behavior_task_id": spec.independent_behavior_task["task_id"],
        "parser_id": spec.parsing_rules["parser_id"],
        "output_layout": {
            "root": str(construct_root),
            "direction": str(construct_root / "direction"),
            "readout": str(construct_root / "readout"),
            "calibration": str(construct_root / "calibration"),
            "behavior_baseline": str(construct_root / "behavior_baseline"),
            "zero_dose_steering": str(construct_root / "zero_dose_steering"),
            "behavior_steered": str(construct_root / "behavior_steered"),
            "steering": str(construct_root / "steering"),
        },
    }


def build_run_plan(
    run_config: RunConfig,
    construct_specs: Mapping[str, ConstructSpec],
    analysis_spec: AnalysisSpec,
    *,
    prompt_inventory_path: str | Path | None = None,
    prompt_records: Iterable[PromptRecord] | None = None,
    output_root: str | Path | None = None,
    run_mode: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic execution plan for one or many constructs.

    The plan has one shared prompt/activation stage and construct-scoped
    downstream jobs. It is intentionally a manifest, not a model runner: a
    later executor can schedule shared work once and fan out construct jobs
    without changing the scientific split rules.
    """

    validate_run_constructs(run_config, dict(construct_specs))
    validate_analysis_spec(run_config, analysis_spec)
    resolved_run_mode, run_mode_config = resolve_run_mode(run_config, run_mode)
    if prompt_records is not None:
        validate_prompt_records(prompt_records, construct_specs)
    if prompt_inventory_path is not None and Path(prompt_inventory_path).exists() and prompt_records is None:
        records = load_prompt_records(prompt_inventory_path)
        validate_prompt_records(records, construct_specs)

    root = Path(output_root if output_root is not None else run_config.output_root) / run_config.run_id
    ordered_specs = [construct_specs[construct_id] for construct_id in run_config.construct_ids]
    construct_entries = [_construct_entry(spec, root) for spec in ordered_specs]

    combined_prompt_path = Path(prompt_inventory_path) if prompt_inventory_path else root / "prompts" / "combined.csv"
    shared_execution = {
        "run_mode": resolved_run_mode,
        "run_mode_purpose": run_mode_config["purpose"],
        "confirmatory": bool(run_mode_config["confirmatory"]),
        "construct_ids": list(run_config.construct_ids),
        "prompt_inventory": str(combined_prompt_path),
        "activation_output": str(root / "activations"),
        "model": dict(run_config.model),
        "activation": dict(run_config.activation),
        "batching_policy": "batch all selected construct prompts together; retain construct_id in every row",
    }

    execution_graph: list[dict[str, Any]] = [
        {
            "stage_id": "validate_shared_prompt_inventory",
            "scope": "shared",
            "construct_ids": list(run_config.construct_ids),
            "input": str(combined_prompt_path),
        },
        {
            "stage_id": "log_shared_activations",
            "scope": "shared",
            "construct_ids": list(run_config.construct_ids),
            "input": str(combined_prompt_path),
            "output": str(root / "activations"),
            "activation_site": run_config.activation["activation_site"],
            "layers": list(run_config.activation["layers"]),
        },
    ]
    for entry in construct_entries:
        construct_id = entry["construct_id"]
        execution_graph.extend(
            [
                {
                    "stage_id": f"build_direction:{construct_id}",
                    "scope": "construct",
                    "construct_id": construct_id,
                    "input": str(root / "activations"),
                    "output": entry["output_layout"]["direction"],
                    "group_key": ["construct_id", "pair_id"],
                    "training_split": "direction_train",
                    "validation_split": "direction_validation",
                },
                {
                    "stage_id": f"evaluate_readout:{construct_id}",
                    "scope": "construct",
                    "construct_id": construct_id,
                    "input": entry["output_layout"]["direction"],
                    "output": entry["output_layout"]["readout"],
                    "heldout_split": "direction_heldout",
                    "estimand": analysis_spec.primary_readout.get("estimand"),
                },
                {
                    "stage_id": f"evaluate_behavior_baseline:{construct_id}",
                    "scope": "construct",
                    "construct_id": construct_id,
                    "input": str(combined_prompt_path),
                    "output": entry["output_layout"]["behavior_baseline"],
                    "behavior_task_id": entry["behavior_task_id"],
                    "behavior_split": "behavior_eval",
                    "parser_id": entry["parser_id"],
                    "intervention": "none",
                    "prompt_only_baseline": True,
                    "scope_rule": SPLIT_EXECUTION_SCOPE["behavior_eval"],
                },
                {
                    "stage_id": f"evaluate_zero_dose_behavior:{construct_id}",
                    "scope": "construct",
                    "construct_id": construct_id,
                    "input": str(combined_prompt_path),
                    "output": entry["output_layout"]["zero_dose_steering"],
                    "behavior_task_id": entry["behavior_task_id"],
                    "behavior_split": "steering_eval",
                    "parser_id": entry["parser_id"],
                    "intervention": "target_zero_dose_only",
                    "prompt_only_baseline": False,
                    "purpose": "pre_full_run_behavioral_variation_gate",
                    "scope_rule": SPLIT_EXECUTION_SCOPE["steering_eval"],
                },
                {
                    "stage_id": f"calibrate_and_steer:{construct_id}",
                    "scope": "construct",
                    "construct_id": construct_id,
                    "input": entry["output_layout"]["direction"],
                    "output": entry["output_layout"]["steering"],
                    "calibration_output": entry["output_layout"]["calibration"],
                    "scales": list(run_config.steering["scales"]),
                    "intervention_timing": run_config.steering["intervention_timing"],
                    "calibration_method": run_config.steering["calibration"],
                    "controls": list(analysis_spec.primary_steering.get("controls", [])),
                    "random_direction_count": run_config.steering["random_direction_count"],
                    "scope_rule": SPLIT_EXECUTION_SCOPE["steering_eval"],
                },
                {
                    "stage_id": f"evaluate_behavior_steered:{construct_id}",
                    "scope": "construct",
                    "construct_id": construct_id,
                    "input": entry["output_layout"]["steering"],
                    "output": entry["output_layout"]["behavior_steered"],
                    "behavior_task_id": entry["behavior_task_id"],
                    "behavior_split": "steering_eval",
                    "parser_id": entry["parser_id"],
                    "intervention": "steered",
                    "comparison": "positive_vs_negative_standardized_by_zero_dose_on_same_items",
                    "scope_rule": SPLIT_EXECUTION_SCOPE["steering_eval"],
                },
            ]
        )

    prompt_hash = None
    if prompt_inventory_path is not None and Path(prompt_inventory_path).exists():
        prompt_hash = file_sha256(prompt_inventory_path)
    manifest = {
        "schema_version": run_config.schema_version,
        "manifest_type": "multi_construct_run_plan",
        "run_id": run_config.run_id,
        "run_mode": {
            "mode": resolved_run_mode,
            "purpose": run_mode_config["purpose"],
            "confirmatory": bool(run_mode_config["confirmatory"]),
            "max_runtime_minutes": run_mode_config["max_runtime_minutes"],
            "prompt_selection": dict(run_mode_config["prompt_selection"]),
        },
        "construct_count": len(construct_entries),
        "constructs": construct_entries,
        "shared_execution": shared_execution,
        "analysis": analysis_spec.to_mapping(),
        "execution_graph": execution_graph,
        "provenance": {
            "run_config_hash": canonical_hash(run_config.to_mapping()),
            "analysis_spec_hash": canonical_hash(analysis_spec.to_mapping()),
            "construct_spec_hashes": {
                entry["construct_id"]: entry["spec_hash"] for entry in construct_entries
            },
            "prompt_inventory_sha256": prompt_hash,
        },
        "run_config": run_config.to_mapping(),
    }
    return manifest


def write_run_plan(path: str | Path, manifest: Mapping[str, Any]) -> None:
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_run_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{plan_path} is not valid JSON.") from exc
    if not isinstance(data, dict) or data.get("manifest_type") != "multi_construct_run_plan":
        raise ValueError(f"{plan_path} is not a multi_construct_run_plan manifest.")
    if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version={data.get('schema_version')!r}; supported versions are "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    return data
