"""Load and cross-validate benchmark configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import AnalysisSpec, ConstructSpec, RunConfig


def load_json(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return data


def load_construct_spec(path: str | Path) -> ConstructSpec:
    return ConstructSpec.from_mapping(load_json(path))


def load_construct_specs(paths: Iterable[str | Path]) -> dict[str, ConstructSpec]:
    specs: dict[str, ConstructSpec] = {}
    for path in paths:
        spec = load_construct_spec(path)
        if spec.construct_id in specs:
            raise ValueError(f"Duplicate construct_id: {spec.construct_id}")
        specs[spec.construct_id] = spec
    if not specs:
        raise ValueError("At least one construct specification is required.")
    return specs


def load_run_config(path: str | Path) -> RunConfig:
    return RunConfig.from_mapping(load_json(path))


def load_analysis_spec(path: str | Path) -> AnalysisSpec:
    return AnalysisSpec.from_mapping(load_json(path))


def validate_run_constructs(run_config: RunConfig, construct_specs: dict[str, ConstructSpec]) -> None:
    configured = set(run_config.construct_ids)
    available = set(construct_specs)
    missing = configured - available
    if missing:
        raise ValueError(f"Run config references missing construct specs: {sorted(missing)}")
    extra = available - configured
    if extra:
        raise ValueError(
            "Construct specs not listed in run config: "
            f"{sorted(extra)}. Pass only the specs used by this run."
        )


def validate_analysis_spec(run_config: RunConfig, analysis_spec: AnalysisSpec) -> None:
    if run_config.analysis_spec_id != analysis_spec.analysis_id:
        raise ValueError(
            "run_config.analysis_spec_id does not match analysis_spec.analysis_id: "
            f"{run_config.analysis_spec_id!r} != {analysis_spec.analysis_id!r}"
        )
