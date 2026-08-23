"""Versioned, JSON-friendly schemas for the multi-construct benchmark.

The schemas deliberately separate shared execution settings from construct-
specific scientific meaning. A run can therefore batch prompts from several
constructs through one model/activation pass while keeping directions,
behavioral outcomes, and steering results partitioned by construct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "0.1.0"
KNOWN_SPLITS = frozenset(
    {
        "direction_train",
        "direction_validation",
        "direction_heldout",
        "behavior_eval",
        "steering_eval",
        "calibration",
    }
)
DEFAULT_REQUIRED_SPLITS = (
    "direction_train",
    "direction_validation",
    "direction_heldout",
    "behavior_eval",
    "steering_eval",
)
PROMPT_ROLES = frozenset({"probe", "behavior", "steering", "calibration"})
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})


def _nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object.")
    return dict(value)


def _string_list(value: Any, *, field_name: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings.")
    items = tuple(_nonempty_string(item, field_name=field_name) for item in value)
    if not allow_empty and not items:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return items


def _validate_id(value: Any, *, field_name: str) -> str:
    identifier = _nonempty_string(value, field_name=field_name)
    if not _ID_PATTERN.fullmatch(identifier):
        raise ValueError(
            f"{field_name}={identifier!r} must use lowercase letters, numbers, and underscores "
            "and start with a letter."
        )
    return identifier


def _validate_schema_version(value: Any) -> str:
    version = _nonempty_string(value, field_name="schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version={version!r}; supported versions are "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    return version


def _condition_ids(conditions: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    return tuple(str(condition["condition_id"]) for condition in conditions)


@dataclass(frozen=True)
class ConstructSpec:
    """Scientific definition of one construct and its independent task."""

    construct_id: str
    version: str
    family: str
    title: str
    description: str
    contrast_conditions: tuple[dict[str, Any], ...]
    probe_prompt_template: str
    independent_behavior_task: dict[str, Any]
    expected_direction: dict[str, Any]
    parsing_rules: dict[str, Any]
    required_splits: tuple[str, ...] = DEFAULT_REQUIRED_SPLITS
    paired_splits: tuple[str, ...] = (
        "direction_train",
        "direction_validation",
        "direction_heldout",
    )
    controls: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ConstructSpec":
        payload = _mapping(data, field_name="construct_spec")
        construct_id = _validate_id(payload.get("construct_id"), field_name="construct_id")
        version = _nonempty_string(payload.get("version", "v1"), field_name="version")
        family = _validate_id(payload.get("family"), field_name="family")
        title = _nonempty_string(payload.get("title"), field_name="title")
        description = _nonempty_string(payload.get("description"), field_name="description")
        schema_version = _validate_schema_version(payload.get("schema_version", SCHEMA_VERSION))

        raw_conditions = payload.get("contrast_conditions")
        if not isinstance(raw_conditions, (list, tuple)) or len(raw_conditions) != 2:
            raise ValueError("contrast_conditions must contain exactly two condition objects.")
        conditions: list[dict[str, Any]] = []
        for index, raw_condition in enumerate(raw_conditions):
            condition = _mapping(raw_condition, field_name=f"contrast_conditions[{index}]")
            condition_id = _validate_id(
                condition.get("condition_id"),
                field_name=f"contrast_conditions[{index}].condition_id",
            )
            condition["condition_id"] = condition_id
            condition["label"] = _nonempty_string(
                condition.get("label"), field_name=f"contrast_conditions[{index}].label"
            )
            condition["definition"] = _nonempty_string(
                condition.get("definition"), field_name=f"contrast_conditions[{index}].definition"
            )
            conditions.append(condition)
        condition_tuple = tuple(conditions)
        if len(set(_condition_ids(condition_tuple))) != 2:
            raise ValueError("contrast_conditions condition_id values must be distinct.")

        probe_prompt_template = _nonempty_string(
            payload.get("probe_prompt_template"), field_name="probe_prompt_template"
        )
        behavior_task = _mapping(
            payload.get("independent_behavior_task"), field_name="independent_behavior_task"
        )
        for field_name in ("task_id", "prompt_template", "primary_outcome", "response_format"):
            _nonempty_string(
                behavior_task.get(field_name),
                field_name=f"independent_behavior_task.{field_name}",
            )

        expected_direction = _mapping(payload.get("expected_direction"), field_name="expected_direction")
        readout_direction = _mapping(
            expected_direction.get("readout"), field_name="expected_direction.readout"
        )
        behavior_direction = _mapping(
            expected_direction.get("behavior"), field_name="expected_direction.behavior"
        )
        condition_set = set(_condition_ids(condition_tuple))
        for field_name in ("positive_condition", "negative_condition"):
            condition_value = _nonempty_string(
                readout_direction.get(field_name),
                field_name=f"expected_direction.readout.{field_name}",
            )
            if condition_value not in condition_set:
                raise ValueError(f"{field_name} must name one of the contrast conditions.")
        for field_name in ("outcome", "estimand"):
            _nonempty_string(
                behavior_direction.get(field_name),
                field_name=f"expected_direction.behavior.{field_name}",
            )
        expected_direction["readout"] = readout_direction
        expected_direction["behavior"] = behavior_direction

        parsing_rules = _mapping(payload.get("parsing_rules"), field_name="parsing_rules")
        _nonempty_string(parsing_rules.get("parser_id"), field_name="parsing_rules.parser_id")

        required_splits = _string_list(
            payload.get("required_splits", DEFAULT_REQUIRED_SPLITS),
            field_name="required_splits",
            allow_empty=False,
        )
        unknown_required = set(required_splits) - KNOWN_SPLITS
        if unknown_required:
            raise ValueError(f"required_splits contains unknown values: {sorted(unknown_required)}")
        paired_splits = _string_list(
            payload.get(
                "paired_splits",
                ("direction_train", "direction_validation", "direction_heldout"),
            ),
            field_name="paired_splits",
            allow_empty=False,
        )
        if not set(paired_splits).issubset(required_splits):
            raise ValueError("paired_splits must be a subset of required_splits.")
        controls = _string_list(payload.get("controls", []), field_name="controls")
        metadata = _mapping(payload.get("metadata", {}), field_name="metadata")

        return cls(
            construct_id=construct_id,
            version=version,
            family=family,
            title=title,
            description=description,
            contrast_conditions=condition_tuple,
            probe_prompt_template=probe_prompt_template,
            independent_behavior_task=behavior_task,
            expected_direction=expected_direction,
            parsing_rules=parsing_rules,
            required_splits=required_splits,
            paired_splits=paired_splits,
            controls=controls,
            metadata=metadata,
            schema_version=schema_version,
        )

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return _condition_ids(self.contrast_conditions)

    @property
    def positive_condition_id(self) -> str:
        return str(self.expected_direction["readout"]["positive_condition"])

    @property
    def negative_condition_id(self) -> str:
        return str(self.expected_direction["readout"]["negative_condition"])

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "construct_id": self.construct_id,
            "version": self.version,
            "family": self.family,
            "title": self.title,
            "description": self.description,
            "contrast_conditions": [dict(condition) for condition in self.contrast_conditions],
            "probe_prompt_template": self.probe_prompt_template,
            "independent_behavior_task": dict(self.independent_behavior_task),
            "expected_direction": dict(self.expected_direction),
            "parsing_rules": dict(self.parsing_rules),
            "required_splits": list(self.required_splits),
            "paired_splits": list(self.paired_splits),
            "controls": list(self.controls),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RunConfig:
    """Shared execution settings for a multi-construct run."""

    run_id: str
    construct_ids: tuple[str, ...]
    model: dict[str, Any]
    activation: dict[str, Any]
    steering: dict[str, Any]
    output_root: str
    seed: int
    analysis_spec_id: str
    execution: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunConfig":
        payload = _mapping(data, field_name="run_config")
        run_id = _validate_id(payload.get("run_id"), field_name="run_id")
        construct_ids = _string_list(
            payload.get("construct_ids"), field_name="construct_ids", allow_empty=False
        )
        model = _mapping(payload.get("model"), field_name="model")
        _nonempty_string(model.get("model_id"), field_name="model.model_id")
        activation = _mapping(payload.get("activation"), field_name="activation")
        layers = activation.get("layers")
        if not isinstance(layers, (list, tuple)) or not layers:
            raise ValueError("activation.layers must be a non-empty list.")
        if any(not isinstance(layer, int) or layer < 1 for layer in layers):
            raise ValueError("activation.layers must contain positive integers.")
        _nonempty_string(activation.get("activation_site"), field_name="activation.activation_site")
        _nonempty_string(activation.get("token_mode"), field_name="activation.token_mode")
        _nonempty_string(activation.get("storage_dtype"), field_name="activation.storage_dtype")
        batch_size = activation.get("batch_size")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("activation.batch_size must be a positive integer.")

        steering = _mapping(payload.get("steering"), field_name="steering")
        scales = steering.get("scales")
        if not isinstance(scales, (list, tuple)) or not scales:
            raise ValueError("steering.scales must be a non-empty list.")
        if any(not isinstance(scale, (int, float)) for scale in scales):
            raise ValueError("steering.scales must contain numbers.")
        position_mode = _nonempty_string(
            steering.get("position_mode"), field_name="steering.position_mode"
        )
        if position_mode not in {"all", "last"}:
            raise ValueError("steering.position_mode must be 'all' or 'last'.")
        intervention_timing = _nonempty_string(
            steering.get("intervention_timing"), field_name="steering.intervention_timing"
        )
        if intervention_timing not in {"prefill_only", "generation_only", "every_step", "fixed_window"}:
            raise ValueError(
                "steering.intervention_timing must be one of: prefill_only, generation_only, "
                "every_step, fixed_window."
            )
        direction_source = _nonempty_string(
            steering.get("direction_source"), field_name="steering.direction_source"
        )
        if direction_source != "direction_train_only":
            raise ValueError("steering.direction_source must be 'direction_train_only'.")

        output_root = _nonempty_string(payload.get("output_root"), field_name="output_root")
        seed = payload.get("seed")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer.")
        analysis_spec_id = _validate_id(
            payload.get("analysis_spec_id", "rsc_benchmark_core_v1"), field_name="analysis_spec_id"
        )
        execution = _mapping(payload.get("execution", {}), field_name="execution")
        schema_version = _validate_schema_version(payload.get("schema_version", SCHEMA_VERSION))
        return cls(
            run_id=run_id,
            construct_ids=construct_ids,
            model=model,
            activation=activation,
            steering=steering,
            output_root=output_root,
            seed=seed,
            analysis_spec_id=analysis_spec_id,
            execution=execution,
            schema_version=schema_version,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "construct_ids": list(self.construct_ids),
            "model": dict(self.model),
            "activation": dict(self.activation),
            "steering": dict(self.steering),
            "output_root": self.output_root,
            "seed": self.seed,
            "analysis_spec_id": self.analysis_spec_id,
            "execution": dict(self.execution),
        }


@dataclass(frozen=True)
class AnalysisSpec:
    """Frozen estimands, controls, and uncertainty rules for a run."""

    analysis_id: str
    version: str
    primary_readout: dict[str, Any]
    primary_steering: dict[str, Any]
    controls: dict[str, Any]
    uncertainty: dict[str, Any]
    exclusions: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AnalysisSpec":
        payload = _mapping(data, field_name="analysis_spec")
        analysis_id = _validate_id(payload.get("analysis_id"), field_name="analysis_id")
        version = _nonempty_string(payload.get("version", "v1"), field_name="version")
        fields = {}
        for field_name in (
            "primary_readout",
            "primary_steering",
            "controls",
            "uncertainty",
            "exclusions",
        ):
            fields[field_name] = _mapping(payload.get(field_name), field_name=field_name)
        schema_version = _validate_schema_version(payload.get("schema_version", SCHEMA_VERSION))
        return cls(
            analysis_id=analysis_id,
            version=version,
            primary_readout=fields["primary_readout"],
            primary_steering=fields["primary_steering"],
            controls=fields["controls"],
            uncertainty=fields["uncertainty"],
            exclusions=fields["exclusions"],
            schema_version=schema_version,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "version": self.version,
            "primary_readout": dict(self.primary_readout),
            "primary_steering": dict(self.primary_steering),
            "controls": dict(self.controls),
            "uncertainty": dict(self.uncertainty),
            "exclusions": dict(self.exclusions),
        }
