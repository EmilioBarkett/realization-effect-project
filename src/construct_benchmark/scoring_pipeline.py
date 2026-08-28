"""One-command, model-independent orchestration of B/R/C/S scoring.

The four namespaces are intentionally independent:

``B``
    Prompt-only behavioral validity and the existing zero-dose baseline gate.
``R``
    Train-only representation/readout and calibration summaries.
``C``
    Matched-episode causal-interchange validation.
``S``
    Independent-task steering, manipulation checks, and uncertainty.

The default adapters call the repository's existing pure scorers and CLI
helpers.  The orchestration layer only validates provenance and assembles a
report; it never changes a registered sign, layer, scale, prompt subset, or
scientific estimand.  Custom adapters can be registered for a local run
format, which keeps no-Torch tests and future model runners decoupled from the
reporting contract.
"""

from __future__ import annotations

import json
import inspect
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .behavior_baseline import (
    read_behavior_output,
    score_behavior_rows,
    validate_behavior_output_manifest,
)
from .behavioral_variation import audit_prompt_only_variation
from .campaign import confirmatory_execution_report
from .config import load_construct_specs, load_run_config
from .manifests import canonical_hash


SCORING_PIPELINE_SCHEMA_VERSION = "0.1.0"
STAGE_CODES = ("B", "R", "C", "S")
_STAGE_ALIASES = {
    "b": "B",
    "behavior": "B",
    "behavior_baseline": "B",
    "behavioral_validity": "B",
    "r": "R",
    "readout": "R",
    "representation": "R",
    "representation_profile": "R",
    "c": "C",
    "causal": "C",
    "causal_interchange": "C",
    "residual_interchange": "C",
    "s": "S",
    "steering": "S",
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_mapping"):
        return _jsonable(value.to_mapping())
    raise TypeError(f"Cannot serialize {type(value).__name__} in a scoring report.")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL output does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} has invalid JSON on line {line_number}.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} must be a JSON object.")
        rows.append(value)
    return rows


def normalize_stage_code(value: str) -> str:
    """Normalize a public stage name to one of ``B``, ``R``, ``C``, ``S``."""

    normalized = str(value).strip()
    if normalized in STAGE_CODES:
        return normalized
    code = _STAGE_ALIASES.get(normalized.lower())
    if code is None:
        raise ValueError(f"Unknown scoring stage {value!r}; expected one of {STAGE_CODES!r}.")
    return code


def _resolve_path(value: Any, *, base_dir: Path | None) -> Any:
    if isinstance(value, str) and value.strip():
        path = Path(value)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path.resolve()
    if isinstance(value, list):
        return [_resolve_path(item, base_dir=base_dir) for item in value]
    return value


@dataclass(frozen=True)
class StageInput:
    """A stage path plus auxiliary paths/options consumed by an adapter."""

    code: str
    path: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", normalize_stage_code(self.code))
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(self, "options", dict(self.options))

    @classmethod
    def from_value(
        cls,
        code: str,
        value: Any,
        *,
        base_dir: Path | None = None,
    ) -> "StageInput":
        normalized = normalize_stage_code(code)
        if isinstance(value, (str, Path)):
            return cls(normalized, _resolve_path(value, base_dir=base_dir))
        if value is None:
            return cls(normalized, None, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"Stage {normalized} input must be a path or object.")
        raw = dict(value)
        path_value = next(
            (raw.pop(key) for key in ("path", "summary", "raw_generations", "raw_output", "output") if key in raw),
            None,
        )
        options = {key: _resolve_path(item, base_dir=base_dir) for key, item in raw.items()}
        if path_value is not None:
            path_value = _resolve_path(path_value, base_dir=base_dir)
        return cls(normalized, Path(path_value) if path_value is not None else None, options)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": None if self.path is None else str(self.path),
            "options": _jsonable(self.options),
        }

    def option_path(self, *names: str) -> Path | None:
        for name in names:
            value = self.options.get(name)
            if value is None:
                continue
            return Path(value).resolve()
        return self.path


@dataclass(frozen=True)
class ScoringContext:
    """Immutable context shared by all stage adapters."""

    mode: str
    config_path: Path | None = None
    run_config_path: Path | None = None
    construct_spec_paths: tuple[Path, ...] = ()
    campaign_path: Path | None = None

    @property
    def allow_incomplete_diagnostic(self) -> bool:
        return self.mode == "diagnostic"


@dataclass(frozen=True)
class StageResult:
    """Adapter result with status/provenance separated from its summary."""

    status: str = "passed"
    complete: bool = True
    confirmatory: bool = False
    summary: Mapping[str, Any] = field(default_factory=dict)
    exclusions: Any = field(default_factory=list)
    uncertainty: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    preflight: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "complete": self.complete,
            "confirmatory": self.confirmatory,
            "summary": _jsonable(self.summary),
            "exclusions": _jsonable(self.exclusions),
            "uncertainty": _jsonable(self.uncertainty),
            "provenance": _jsonable(self.provenance),
            "preflight": _jsonable(self.preflight),
        }


StageValidator = Callable[[StageInput, ScoringContext], StageResult | Mapping[str, Any]]


@dataclass(frozen=True)
class StageAdapter:
    """Named callable registered for one scoring namespace."""

    code: str
    name: str
    validator: StageValidator

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", normalize_stage_code(self.code))


class StageAdapterRegistry:
    """Small dependency-injection registry for B/R/C/S validators."""

    def __init__(self, adapters: Iterable[StageAdapter] | Mapping[str, StageValidator] | None = None) -> None:
        self._adapters: dict[str, StageAdapter] = {}
        if isinstance(adapters, Mapping):
            for code, validator in adapters.items():
                self.register(code, validator, name=f"custom_{normalize_stage_code(code)}")
        else:
            for adapter in adapters or ():
                self.register(adapter)

    def register(
        self,
        adapter: StageAdapter | str,
        validator: StageValidator | None = None,
        *,
        name: str | None = None,
    ) -> None:
        if isinstance(adapter, StageAdapter):
            resolved = adapter
        else:
            if validator is None:
                raise ValueError("A validator callable is required when registering by stage code.")
            resolved = StageAdapter(normalize_stage_code(adapter), name or str(adapter), validator)
        self._adapters[resolved.code] = resolved

    def get(self, code: str) -> StageAdapter:
        normalized = normalize_stage_code(code)
        try:
            return self._adapters[normalized]
        except KeyError as exc:
            raise ValueError(f"No adapter is registered for stage {normalized}.") from exc

    def copy(self) -> "StageAdapterRegistry":
        return StageAdapterRegistry(self._adapters.values())

    @classmethod
    def default(cls) -> "StageAdapterRegistry":
        return cls(
            (
                StageAdapter("B", "behavior_baseline", _validate_behavior_stage),
                StageAdapter("R", "readout_representation", _validate_readout_stage),
                StageAdapter("C", "causal_residual_interchange", _validate_causal_stage),
                StageAdapter("S", "steering", _validate_steering_stage),
            )
        )


def default_stage_adapters() -> StageAdapterRegistry:
    """Return a fresh registry using current repository implementations."""

    return StageAdapterRegistry.default()


def _summary_path(stage: StageInput) -> Path | None:
    explicit = stage.options.get("summary_path") or stage.options.get("summary")
    if explicit:
        return Path(explicit).resolve()
    path = stage.path
    if path is None:
        return None
    if path.is_dir():
        candidate = path / "summary.json"
        return candidate if candidate.is_file() else None
    if path.name == "summary.json" or path.suffix.lower() == ".json":
        return path
    sibling = path.with_name("summary.json")
    return sibling if sibling.is_file() else None


def _summary_complete(summary: Mapping[str, Any]) -> bool:
    for key in ("manifest_complete", "complete"):
        if key in summary:
            return bool(summary[key])
    provenance = summary.get("provenance")
    if isinstance(provenance, Mapping):
        execution = provenance.get("execution")
        if isinstance(execution, Mapping) and "complete" in execution:
            return bool(execution["complete"])
    return True


def _summary_result(
    summary: Mapping[str, Any],
    *,
    stage: StageInput,
    context: ScoringContext,
    complete: bool | None = None,
    confirmatory: bool | None = None,
    status: str | None = None,
    exclusions: Any = None,
    uncertainty: Any = None,
) -> StageResult:
    payload = dict(summary)
    resolved_complete = _summary_complete(payload) if complete is None else bool(complete)
    resolved_confirmatory = bool(payload.get("confirmatory", False)) if confirmatory is None else bool(confirmatory)
    resolved_status = status or ("passed" if payload.get("pass", True) is not False else "failed")
    resolved_exclusions = payload.get("exclusions", exclusions if exclusions is not None else [])
    resolved_uncertainty = payload.get("uncertainty", uncertainty)
    return StageResult(
        status=resolved_status,
        complete=resolved_complete,
        confirmatory=resolved_confirmatory and resolved_complete and context.mode == "complete",
        summary=payload,
        exclusions=resolved_exclusions,
        uncertainty=resolved_uncertainty,
        provenance={"input": stage.to_mapping()},
        preflight={"status": "pass", "input_exists": stage.path is None or stage.path.exists()},
    )


def _load_summary_stage(stage: StageInput, context: ScoringContext) -> StageResult | None:
    path = _summary_path(stage)
    if path is None:
        return None
    summary = _load_object(path, label=f"{stage.code} summary")
    return _summary_result(summary, stage=stage, context=context)


def _construct_paths(stage: StageInput, context: ScoringContext) -> list[Path]:
    value = stage.options.get("construct_specs")
    if value is None:
        value = stage.options.get("construct_spec")
    if value is None:
        return list(context.construct_spec_paths)
    if isinstance(value, (str, Path)):
        return [Path(value).resolve()]
    return [Path(item).resolve() for item in value]


def _invoke_stage_validator(validator: StageValidator, stage: StageInput, context: ScoringContext) -> Any:
    """Call the documented two-argument adapter or a compact one-argument fixture."""

    try:
        parameters = [
            parameter
            for parameter in inspect.signature(validator).parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(parameters) <= 1:
            return validator(stage)  # type: ignore[call-arg]
    except (TypeError, ValueError):
        pass
    return validator(stage, context)


def _behavior_exclusions(parsed_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    exclusions = []
    for row in parsed_rows:
        if row.get("parser_valid") and row.get("primary_valid"):
            continue
        exclusions.append(
            {
                "record_id": row.get("record_id"),
                "prompt_id": row.get("prompt_id"),
                "construct_id": row.get("construct_id"),
                "reason": row.get("error") or "invalid_or_unparseable",
            }
        )
    return exclusions


def _validate_behavior_stage(stage: StageInput, context: ScoringContext) -> StageResult:
    summary_result = _load_summary_stage(stage, context)
    explicit_raw = stage.options.get("raw_generations") or stage.options.get("raw_output")
    if summary_result is not None and explicit_raw is None and (
        stage.path is None or stage.path.suffix.lower() != ".jsonl"
    ):
        return summary_result
    raw_path = Path(explicit_raw).resolve() if explicit_raw is not None else stage.path
    if raw_path is None:
        raise ValueError("B stage requires a summary.json or raw_generations JSONL path.")
    rows = read_behavior_output(raw_path)
    spec_paths = _construct_paths(stage, context)
    run_config_value = stage.options.get("run_config") or context.run_config_path
    if not spec_paths or run_config_value is None:
        raise ValueError("B raw scoring requires run_config and construct_specs.")
    run_config = load_run_config(run_config_value)
    construct_specs = load_construct_specs(spec_paths)
    manifest, complete = validate_behavior_output_manifest(
        raw_path,
        rows,
        run_config=run_config,
        construct_specs=construct_specs,
        allow_incomplete_diagnostic=context.allow_incomplete_diagnostic,
    )
    parsed_rows, behavior_summary = score_behavior_rows(rows, construct_specs)
    variation = {
        construct_id: audit_prompt_only_variation(rows, spec)
        for construct_id, spec in construct_specs.items()
    }
    exclusions = _behavior_exclusions(parsed_rows)
    summary = {
        "manifest_type": "behavior_baseline_score",
        "behavior": behavior_summary,
        "variation_gate": variation,
        "manifest_complete": complete,
        "raw_record_count": len(rows),
        "invalid_or_unparseable_exclusions": exclusions,
        "uncertainty": {
            "status": "not_available",
            "reason": "The existing prompt-only baseline scorer reports validity and variation, not an inferential interval.",
        },
    }
    return StageResult(
        status=(
            "passed"
            if complete and all(item["pass"] for item in variation.values())
            else ("diagnostic_incomplete" if not complete else "failed")
        ),
        complete=complete,
        confirmatory=bool(manifest.get("confirmatory", False)) and complete and context.mode == "complete",
        summary=summary,
        exclusions=exclusions,
        uncertainty=summary["uncertainty"],
        provenance={
            "raw_generations": str(raw_path),
            "output_manifest": str(raw_path.with_suffix(raw_path.suffix + ".manifest.json")),
            "run_id": manifest.get("run_id"),
            "prompt_inventory_sha256": manifest.get("prompt_inventory_sha256"),
            "run_config_hash": manifest.get("run_config_hash"),
        },
        preflight={"status": "pass", "manifest_validated": True},
    )


def _validate_readout_summary(summary: Mapping[str, Any]) -> None:
    readout = summary.get("readout")
    if not isinstance(readout, Mapping):
        raise ValueError("R summary is missing the readout namespace.")
    direction = summary.get("direction")
    if isinstance(direction, Mapping) and direction.get("source_split") not in {None, "direction_train"}:
        raise ValueError("R direction must be sourced from direction_train.")
    calibration = summary.get("calibration")
    if isinstance(calibration, Mapping) and calibration.get("projection_scale") is not None:
        if float(calibration["projection_scale"]) <= 0:
            raise ValueError("R calibration projection_scale must be positive.")
    layer_selection = summary.get("layer_selection")
    if isinstance(layer_selection, Mapping):
        rule = layer_selection.get("rule")
        selection_split = layer_selection.get("selection_split")
        if rule == "validation_max_margin" and selection_split not in {None, "direction_validation"}:
            raise ValueError("R validation layer selection must use direction_validation.")


def _run_readout_cli(stage: StageInput) -> Path:
    activation_run = stage.options.get("activation_run")
    construct_spec = stage.options.get("construct_spec")
    output_dir = stage.options.get("output_dir")
    if not activation_run or not construct_spec or not output_dir:
        raise ValueError("R stage requires an existing summary or activation_run, construct_spec, and output_dir.")
    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        layers = stage.options.get("layers")
        layer = stage.options.get("layer")
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts" / "analyze_construct_readout.py"),
            "--activation-run",
            str(Path(activation_run).resolve()),
            "--construct-spec",
            str(Path(construct_spec).resolve()),
            "--output-dir",
            str(output_dir),
        ]
        if layer is not None:
            command.extend(("--layer", str(layer)))
        elif isinstance(layers, Sequence) and not isinstance(layers, str):
            command.extend(("--layers", ",".join(str(item) for item in layers)))
        elif layers is not None:
            command.extend(("--layers", str(layers)))
        else:
            raise ValueError("R CLI adapter requires layer or layers.")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or completed.stdout.strip() or "R scoring CLI failed.")
    return summary_path


def _validate_readout_stage(stage: StageInput, context: ScoringContext) -> StageResult:
    summary_result = _load_summary_stage(stage, context)
    if summary_result is None:
        summary_path = _run_readout_cli(stage)
        summary_result = _summary_result(
            _load_object(summary_path, label="R summary"),
            stage=StageInput("R", summary_path),
            context=context,
        )
    _validate_readout_summary(summary_result.summary)
    return StageResult(
        status=summary_result.status,
        complete=summary_result.complete,
        confirmatory=summary_result.confirmatory,
        summary=summary_result.summary,
        exclusions=summary_result.exclusions,
        uncertainty=summary_result.summary.get("readout", {}).get("uncertainty", summary_result.uncertainty)
        if isinstance(summary_result.summary.get("readout"), Mapping)
        else summary_result.uncertainty,
        provenance=summary_result.provenance,
        preflight={"status": "pass", "train_only_direction_and_frozen_summary": True},
    )


def _causal_summary(raw_path: Path, *, allow_incomplete: bool) -> dict[str, Any]:
    try:
        from scripts.score_residual_interchange import build_summary
    except ModuleNotFoundError:  # pragma: no cover - direct script fallback
        from score_residual_interchange import build_summary  # type: ignore
    return build_summary(raw_path, allow_incomplete_diagnostic=allow_incomplete)


def _validate_causal_stage(stage: StageInput, context: ScoringContext) -> StageResult:
    summary_result = _load_summary_stage(stage, context)
    raw_path = stage.option_path("raw_output")
    if summary_result is not None and (raw_path is None or raw_path.suffix.lower() != ".jsonl"):
        return summary_result
    if raw_path is None:
        raise ValueError("C stage requires a summary.json or residual-interchange JSONL path.")
    summary = _causal_summary(raw_path, allow_incomplete=context.allow_incomplete_diagnostic)
    expected = int(summary.get("expected_observation_count", summary.get("observation_count", 0)))
    observed = int(summary.get("observation_count", 0))
    exclusions = [] if expected <= observed else [{"reason": "incomplete_output", "missing_observation_count": expected - observed}]
    manifest_complete = bool(summary.get("complete", False))
    return StageResult(
        status="passed" if manifest_complete else "diagnostic_incomplete",
        complete=manifest_complete,
        confirmatory=bool(summary.get("confirmatory", False)) and manifest_complete and context.mode == "complete",
        summary=summary,
        exclusions=exclusions,
        uncertainty=summary.get("uncertainty", {"status": "not_available"}),
        provenance={"raw_output": str(raw_path), "manifest": str(raw_path.with_suffix(raw_path.suffix + ".manifest.json"))},
        preflight={"status": "pass", "manifest_validated": True},
    )


def _run_steering_cli(stage: StageInput, context: ScoringContext) -> Path:
    raw_path = stage.option_path("raw_generations", "raw_output")
    construct_spec = stage.options.get("construct_spec")
    if construct_spec is None:
        specs = _construct_paths(stage, context)
        if len(specs) == 1:
            construct_spec = specs[0]
    output_dir = stage.options.get("output_dir")
    if raw_path is None or construct_spec is None:
        raise ValueError("S stage requires raw_generations, construct_spec, and an output directory or summary.")
    output_dir = Path(output_dir).resolve() if output_dir is not None else raw_path.with_name(raw_path.stem + "_scored")
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts" / "score_construct_steering.py"),
            "--raw-generations",
            str(raw_path),
            "--construct-spec",
            str(Path(construct_spec).resolve()),
            "--output-dir",
            str(output_dir),
        ]
        if context.allow_incomplete_diagnostic:
            command.append("--allow-incomplete-diagnostic")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or completed.stdout.strip() or "S scoring CLI failed.")
    return summary_path


def _validate_steering_stage(stage: StageInput, context: ScoringContext) -> StageResult:
    summary_result = _load_summary_stage(stage, context)
    raw_path = stage.option_path("raw_generations", "raw_output")
    if summary_result is None or (raw_path is not None and raw_path.suffix.lower() == ".jsonl"):
        summary_path = _run_steering_cli(stage, context)
        summary_result = _summary_result(
            _load_object(summary_path, label="S summary"),
            stage=StageInput("S", summary_path),
            context=context,
        )
    summary = dict(summary_result.summary)
    if "target_direction_effect" not in summary and "steering" in summary:
        steering = summary["steering"]
        if isinstance(steering, Mapping) and "target_direction_effect" not in steering:
            raise ValueError("S summary is missing the directed state-transfer result.")
    uncertainty = summary.get("uncertainty")
    if uncertainty is None and isinstance(summary.get("steering"), Mapping):
        uncertainty = summary["steering"].get("uncertainty")
    exclusions = summary.get("exclusions", [])
    manipulation = summary.get("manipulation_checks")
    if isinstance(manipulation, Mapping) and manipulation.get("missing_or_unscorable_records", 0):
        existing_exclusions = list(exclusions) if isinstance(exclusions, list) else []
        exclusions = [
            *existing_exclusions,
            {
                "reason": "missing_or_unscorable_manipulation_records",
                "count": manipulation["missing_or_unscorable_records"],
            },
        ]
    return StageResult(
        status=summary_result.status,
        complete=summary_result.complete,
        confirmatory=summary_result.confirmatory,
        summary=summary,
        exclusions=exclusions,
        uncertainty=uncertainty,
        provenance=summary_result.provenance,
        preflight={"status": "pass", "manifest_or_summary_validated": True},
    )


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"complete", "full"}:
        return "complete"
    if normalized in {"diagnostic", "incomplete", "partial"}:
        return "diagnostic"
    raise ValueError("mode must be complete or diagnostic.")


def _normalize_stage_inputs(
    stages: Mapping[str, Any] | None,
    *,
    base_dir: Path | None = None,
) -> dict[str, StageInput]:
    result: dict[str, StageInput] = {}
    for raw_code, value in (stages or {}).items():
        stage = StageInput.from_value(str(raw_code), value, base_dir=base_dir)
        if stage.code in result:
            raise ValueError(f"Duplicate stage input for {stage.code}.")
        result[stage.code] = stage
    return result


def evaluate_expansion_gates(
    campaign_path: str | Path | None = None,
    *,
    campaign_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate existing Wave 1 measurement/precision gates without mutation."""

    if campaign_report is None and campaign_path is None:
        return {
            "evaluated": False,
            "status": "not_evaluated",
            "wave1_measurement_gate": {"status": "not_evaluated"},
            "precision_simulation": {"status": "not_evaluated"},
            "expansion_decision": "hold",
            "reasons": ["No confirmatory campaign was supplied."],
        }
    if campaign_report is None:
        try:
            campaign_report = confirmatory_execution_report(Path(campaign_path), mode="full")
        except (OSError, TypeError, ValueError) as exc:
            return {
                "evaluated": False,
                "status": "error",
                "campaign_path": str(Path(campaign_path).resolve()),
                "wave1_measurement_gate": {"status": "error", "detail": str(exc)},
                "precision_simulation": {"status": "error", "detail": str(exc)},
                "expansion_decision": "hold",
                "reasons": [f"Campaign validator could not evaluate the gates: {exc}"],
            }
    prerequisites = campaign_report.get("prerequisites", [])
    by_name = {
        str(item.get("name", item.get("id"))): item
        for item in prerequisites
        if isinstance(item, Mapping) and (item.get("name") or item.get("id"))
    }

    def gate(name: str) -> dict[str, Any]:
        item = by_name.get(name)
        if item is None:
            return {"status": "unknown", "detail": "The campaign does not declare this prerequisite."}
        raw_status = str(item.get("status", "unknown"))
        normalized = "pass" if raw_status in {"pass", "satisfied"} else raw_status
        return {"status": normalized, "detail": item.get("detail", "")}

    measurement = gate("wave1_measurement_gate")
    precision = gate("precision_simulation")
    reasons = []
    for name, item in (("wave1_measurement_gate", measurement), ("precision_simulation", precision)):
        if item["status"] != "pass":
            reasons.append(f"{name} is {item['status']}: {item.get('detail', '')}")
    blocking = campaign_report.get("blocking_checks", [])
    for item in blocking:
        if isinstance(item, Mapping):
            detail = f"{item.get('name')}: {item.get('detail')}"
            if detail not in reasons:
                reasons.append(detail)
    can_expand = bool(
        measurement["status"] == "pass"
        and precision["status"] == "pass"
        and campaign_report.get("ready") is True
        and campaign_report.get("confirmatory") is True
    )
    return {
        "evaluated": True,
        "status": "pass" if can_expand else "blocked",
        "campaign_path": None if campaign_path is None else str(Path(campaign_path).resolve()),
        "campaign_ready": bool(campaign_report.get("ready", False)),
        "campaign_confirmatory": bool(campaign_report.get("confirmatory", False)),
        "wave1_measurement_gate": measurement,
        "precision_simulation": precision,
        "expansion_decision": "expand" if can_expand else "hold",
        "reasons": reasons or ["Both named gates passed and the campaign validator reported confirmatory readiness."],
        "campaign_report": dict(campaign_report),
    }


def build_scoring_report(
    stages: Mapping[str, Any] | None = None,
    *,
    mode: str = "complete",
    adapters: StageAdapterRegistry | Mapping[str, StageValidator] | None = None,
    campaign_path: str | Path | None = None,
    campaign_report: Mapping[str, Any] | None = None,
    run_config_path: str | Path | None = None,
    construct_spec_paths: Iterable[str | Path] = (),
    config_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate available B/R/C/S inputs and return a frozen pipeline report."""

    resolved_mode = _normalize_mode(mode)
    config_resolved = None if config_path is None else Path(config_path).resolve()
    registry = adapters.copy() if isinstance(adapters, StageAdapterRegistry) else StageAdapterRegistry.default()
    if isinstance(adapters, Mapping):
        registry = StageAdapterRegistry.default()
        for code, validator in adapters.items():
            registry.register(code, validator, name=f"custom_{normalize_stage_code(code)}")
    stage_inputs = _normalize_stage_inputs(stages, base_dir=config_resolved.parent if config_resolved else None)
    run_config_resolved = None if run_config_path is None else Path(run_config_path).resolve()
    spec_paths = tuple(Path(path).resolve() for path in construct_spec_paths)
    context = ScoringContext(
        mode=resolved_mode,
        config_path=config_resolved,
        run_config_path=run_config_resolved,
        construct_spec_paths=spec_paths,
        campaign_path=None if campaign_path is None else Path(campaign_path).resolve(),
    )

    stage_reports: dict[str, dict[str, Any]] = {}
    for code in STAGE_CODES:
        stage = stage_inputs.get(code)
        if stage is None or stage.path is None and not stage.options:
            stage_reports[code] = {
                "stage": code,
                "available": False,
                "status": "not_available",
                "complete": False,
                "confirmatory": False,
                "summary": {},
                "exclusions": [],
                "uncertainty": {"status": "not_available"},
                "preflight": {"status": "not_available"},
            }
            continue
        adapter = registry.get(code)
        try:
            raw_result = _invoke_stage_validator(adapter.validator, stage, context)
            if isinstance(raw_result, StageResult):
                result = raw_result
            elif isinstance(raw_result, Mapping):
                generic_keys = {"status", "complete", "confirmatory", "summary", "exclusions", "uncertainty", "provenance", "preflight"}
                summary = raw_result.get("summary") if isinstance(raw_result.get("summary"), Mapping) else {
                    key: value for key, value in raw_result.items() if key not in generic_keys
                }
                result = StageResult(
                    status=str(raw_result.get("status", "passed")),
                    complete=bool(raw_result.get("complete", True)),
                    confirmatory=bool(raw_result.get("confirmatory", False)),
                    summary=summary,
                    exclusions=raw_result.get("exclusions", []),
                    uncertainty=raw_result.get("uncertainty", summary.get("uncertainty")),
                    provenance=raw_result.get("provenance", {}),
                    preflight=raw_result.get("preflight", {}),
                )
            stage_report = {
                "stage": code,
                "adapter": adapter.name,
                "available": True,
                **result.to_mapping(),
            }
            if resolved_mode == "complete" and not result.complete and stage_report["status"] not in {"error", "failed"}:
                stage_report["status"] = "incomplete"
            if resolved_mode == "diagnostic" and not result.complete and stage_report["status"] not in {"error", "failed"}:
                stage_report["status"] = "diagnostic_incomplete"
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            stage_report = {
                "stage": code,
                "adapter": adapter.name,
                "available": True,
                "status": "error",
                "complete": False,
                "confirmatory": False,
                "summary": {},
                "exclusions": [],
                "uncertainty": {"status": "not_available"},
                "provenance": {"input": stage.to_mapping()},
                "preflight": {"status": "error"},
                "error": f"{type(exc).__name__}: {exc}",
            }
        stage_reports[code] = _jsonable(stage_report)

    gates = evaluate_expansion_gates(campaign_path, campaign_report=campaign_report)
    available = [stage for stage in stage_reports.values() if stage["available"]]
    errors = [stage for stage in available if stage["status"] in {"error", "failed"}]
    incomplete = [stage for stage in available if not stage["complete"]]
    report_complete = bool(available) and not errors and not incomplete
    ready = bool(available) and not errors and (resolved_mode == "diagnostic" or not incomplete)
    all_stages_explicit = len(available) == len(STAGE_CODES)
    explicit_confirmatory = all(stage["confirmatory"] for stage in available) and all_stages_explicit
    confirmatory = bool(
        resolved_mode == "complete"
        and report_complete
        and explicit_confirmatory
        and gates.get("expansion_decision") == "expand"
    )
    confirmatory_reasons = []
    if not explicit_confirmatory:
        confirmatory_reasons.append("Confirmatory status was not explicitly present on every B/R/C/S input.")
    if gates.get("expansion_decision") != "expand":
        confirmatory_reasons.append("The existing expansion gates did not authorize confirmatory release.")
    if resolved_mode != "complete":
        confirmatory_reasons.append("Diagnostic/incomplete reporting is never confirmatory.")
    report = {
        "schema_version": SCORING_PIPELINE_SCHEMA_VERSION,
        "manifest_type": "benchmark_campaign_scoring_report",
        "status": "frozen",
        "frozen": True,
        "report_mode": resolved_mode,
        "complete": report_complete,
        "ready": ready,
        "confirmatory": confirmatory,
        "confirmatory_reasons": confirmatory_reasons,
        "stages": stage_reports,
        "stage_summaries": {code: stage_reports[code]["summary"] for code in STAGE_CODES},
        "stage_exclusions": {code: stage_reports[code]["exclusions"] for code in STAGE_CODES},
        "stage_uncertainty": {code: stage_reports[code]["uncertainty"] for code in STAGE_CODES},
        "gates": gates,
        "selection_policy": "No scientific effect size, sign, layer, scale, or subset is used for orchestration decisions.",
        "inputs": {
            "run_config": None if run_config_resolved is None else str(run_config_resolved),
            "construct_specs": [str(path) for path in spec_paths],
            "stages": {code: stage.to_mapping() for code, stage in stage_inputs.items()},
            "metadata": _jsonable(metadata or {}),
        },
    }
    report["input_identity_sha256"] = canonical_hash(_jsonable(report["inputs"]))
    report["report_sha256"] = canonical_hash(
        _jsonable({key: value for key, value in report.items() if key != "report_sha256"})
    )
    return report


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON pipeline config without mutating it."""

    config_path = Path(path).resolve()
    payload = _load_object(config_path, label="scoring pipeline config")
    payload["__config_path"] = config_path
    return payload


def build_scoring_report_from_config(
    path: str | Path,
    *,
    mode: str | None = None,
    adapters: StageAdapterRegistry | Mapping[str, StageValidator] | None = None,
    campaign_path: str | Path | None = None,
    stage_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pipeline report from the compact JSON CLI configuration."""

    payload = load_pipeline_config(path)
    config_path = Path(payload.pop("__config_path"))
    stages = dict(payload.get("stages", payload.get("stage_inputs", {})))
    for key in STAGE_CODES:
        if key in payload and key not in stages:
            stages[key] = payload[key]
    stages.update(stage_overrides or {})
    selected_mode = mode or str(payload.get("mode", "complete"))
    selected_campaign = campaign_path or payload.get("campaign_path") or payload.get("campaign")
    run_config = payload.get("run_config_path") or payload.get("run_config")
    construct_specs = payload.get("construct_spec_paths") or payload.get("construct_specs") or payload.get("construct_spec") or ()
    def config_path_value(value: Any) -> Any:
        if isinstance(value, (str, Path)):
            candidate = Path(value)
            return candidate if candidate.is_absolute() else config_path.parent / candidate
        if isinstance(value, list):
            return [config_path_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(config_path_value(item) for item in value)
        return value

    selected_campaign = config_path_value(selected_campaign)
    run_config = config_path_value(run_config)
    construct_specs = config_path_value(construct_specs)
    if isinstance(construct_specs, (str, Path)):
        construct_specs = (construct_specs,)
    return build_scoring_report(
        stages,
        mode=selected_mode,
        adapters=adapters,
        campaign_path=selected_campaign,
        run_config_path=run_config,
        construct_spec_paths=construct_specs,
        config_path=config_path,
        metadata=payload.get("metadata"),
    )


def write_scoring_report(
    report: Mapping[str, Any],
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one frozen report and refuse accidental replacement by default."""

    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing scoring report: {output_path}")
    payload = dict(report)
    payload["status"] = "frozen"
    payload["frozen"] = True
    payload["report_sha256"] = canonical_hash(
        _jsonable({key: value for key, value in payload.items() if key != "report_sha256"})
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


# Descriptive aliases for callers that use "score" or "run" terminology.
score_benchmark_campaign = build_scoring_report
run_scoring_pipeline = build_scoring_report


__all__ = [
    "SCORING_PIPELINE_SCHEMA_VERSION",
    "STAGE_CODES",
    "ScoringContext",
    "StageAdapter",
    "StageAdapterRegistry",
    "StageInput",
    "StageResult",
    "StageValidator",
    "build_scoring_report",
    "build_scoring_report_from_config",
    "default_stage_adapters",
    "evaluate_expansion_gates",
    "load_pipeline_config",
    "normalize_stage_code",
    "run_scoring_pipeline",
    "score_benchmark_campaign",
    "write_scoring_report",
]
