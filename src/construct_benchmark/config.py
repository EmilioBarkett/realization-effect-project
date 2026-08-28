"""Load and cross-validate benchmark configuration files."""

from __future__ import annotations

import json
import copy
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


def _deep_merge(base: Any, overlay: Any) -> Any:
    """Merge a versioned config overlay without mutating its base mapping."""

    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = copy.deepcopy(base)
        for key, value in overlay.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    return copy.deepcopy(overlay)


def _load_inherited_spec_payload(path: Path, *, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Construct-spec inheritance cycle: {cycle}")
    payload = load_json(path)
    base_ref = payload.pop("base_spec_path", None)
    if base_ref is None:
        return payload
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError(f"{path}.base_spec_path must be a non-empty string.")
    base_path = (path.parent / base_ref).resolve()
    base = _load_inherited_spec_payload(base_path, stack=(*stack, path))
    return _deep_merge(base, payload)


def load_construct_spec(path: str | Path) -> ConstructSpec:
    """Load a full spec or an explicit, versioned overlay over a base spec."""

    return ConstructSpec.from_mapping(_load_inherited_spec_payload(Path(path)))


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
