"""Versioned, JSON-friendly schemas for the multi-construct benchmark.

The schemas deliberately separate shared execution settings from construct-
specific scientific meaning. A run can therefore batch prompts from several
constructs through one model/activation pass while keeping directions,
behavioral outcomes, and steering results partitioned by construct.
"""

from __future__ import annotations

import math
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
        "collateral_eval",
    }
)
DEFAULT_REQUIRED_SPLITS = (
    "direction_train",
    "direction_validation",
    "direction_heldout",
    "behavior_eval",
    "steering_eval",
)
PROMPT_ROLES = frozenset({"probe", "behavior", "steering", "calibration", "collateral"})
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})
DEFAULT_STORAGE_SETTINGS = {
    "workspace_root_env": "RSC_BENCH_WORKSPACE_ROOT",
    "archive_uri_env": "RSC_BENCH_ARCHIVE_URI",
    "sync_endpoint_env": "RSC_BENCH_S3_ENDPOINT_URL",
    "sync_tool": "aws",
    "sync_on_finalize": True,
    "keep_local_copy": True,
}


def _default_execution_run_modes() -> dict[str, Any]:
    """Return conservative defaults for model-side execution modes.

    Prompt inventories are generated separately. These modes describe how a
    frozen inventory is selected for model-side work: ``test`` is a bounded
    engineering run and ``full`` is the complete, confirmatory path.
    """

    return {
        "test": {
            "purpose": "engineering_smoke",
            "confirmatory": False,
            "max_runtime_minutes": 60,
            "prompt_selection": {
                "strategy": "balanced_pair_preserving",
                "max_pairs_per_paired_split": 2,
                "max_items_per_single_split": 2,
            },
        },
        "full": {
            "purpose": "confirmatory",
            "confirmatory": True,
            "max_runtime_minutes": None,
            "prompt_selection": {"strategy": "all"},
        },
    }


def _validate_execution_run_modes(value: Any, *, default_mode: Any = "test") -> tuple[dict[str, Any], str]:
    raw_modes = _default_execution_run_modes() if value is None else _mapping(value, field_name="execution.run_modes")
    required_modes = {"test", "full"}
    missing_modes = required_modes - set(raw_modes)
    if missing_modes:
        raise ValueError(f"execution.run_modes is missing required mode(s): {sorted(missing_modes)}")

    validated_modes: dict[str, Any] = {}
    for mode_name, raw_mode in raw_modes.items():
        mode_id = _validate_id(mode_name, field_name="execution.run_modes key")
        mode = _mapping(raw_mode, field_name=f"execution.run_modes.{mode_id}")
        purpose = _nonempty_string(mode.get("purpose"), field_name=f"execution.run_modes.{mode_id}.purpose")
        confirmatory = mode.get("confirmatory")
        if not isinstance(confirmatory, bool):
            raise ValueError(
                f"execution.run_modes.{mode_id}.confirmatory must be a boolean."
            )
        max_runtime = mode.get("max_runtime_minutes")
        if max_runtime is not None and (
            not isinstance(max_runtime, (int, float))
            or isinstance(max_runtime, bool)
            or not math.isfinite(float(max_runtime))
            or float(max_runtime) <= 0
        ):
            raise ValueError(
                f"execution.run_modes.{mode_id}.max_runtime_minutes must be positive or null."
            )
        selection = _mapping(
            mode.get("prompt_selection"),
            field_name=f"execution.run_modes.{mode_id}.prompt_selection",
        )
        strategy = _nonempty_string(
            selection.get("strategy"),
            field_name=f"execution.run_modes.{mode_id}.prompt_selection.strategy",
        )
        if strategy not in {"all", "balanced_pair_preserving"}:
            raise ValueError(
                f"execution.run_modes.{mode_id}.prompt_selection.strategy must be "
                "'all' or 'balanced_pair_preserving'."
            )
        if strategy == "balanced_pair_preserving":
            for field_name in ("max_pairs_per_paired_split", "max_items_per_single_split"):
                limit = selection.get(field_name)
                if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                    raise ValueError(
                        f"execution.run_modes.{mode_id}.prompt_selection.{field_name} "
                        "must be a positive integer."
                    )
        if mode_id == "test":
            if confirmatory:
                raise ValueError("execution.run_modes.test.confirmatory must be false.")
            if max_runtime is None:
                raise ValueError("execution.run_modes.test.max_runtime_minutes is required.")
            if strategy != "balanced_pair_preserving":
                raise ValueError(
                    "execution.run_modes.test.prompt_selection.strategy must be "
                    "'balanced_pair_preserving'."
                )
        if mode_id == "full":
            engineering_only = mode.get("engineering_only", False)
            if not isinstance(engineering_only, bool):
                raise ValueError(
                    "execution.run_modes.full.engineering_only must be a boolean."
                )
            if not confirmatory and not (
                engineering_only and purpose == "full_coverage_engineering"
            ):
                raise ValueError(
                    "execution.run_modes.full.confirmatory must be true unless the mode is "
                    "explicitly marked engineering_only with purpose='full_coverage_engineering'."
                )
            if max_runtime is not None:
                raise ValueError("execution.run_modes.full.max_runtime_minutes must be null.")
            if strategy != "all":
                raise ValueError("execution.run_modes.full.prompt_selection.strategy must be 'all'.")
        else:
            engineering_only = mode.get("engineering_only", False)
            if not isinstance(engineering_only, bool):
                raise ValueError(
                    f"execution.run_modes.{mode_id}.engineering_only must be a boolean."
                )
            if engineering_only:
                raise ValueError(
                    f"execution.run_modes.{mode_id}.engineering_only is only valid for the full mode."
                )
        mode["purpose"] = purpose
        mode["confirmatory"] = confirmatory
        mode["engineering_only"] = engineering_only
        mode["max_runtime_minutes"] = max_runtime
        mode["prompt_selection"] = selection
        validated_modes[mode_id] = mode

    default_mode = _validate_id(
        default_mode,
        field_name="execution.default_run_mode",
    )
    if default_mode not in validated_modes:
        raise ValueError(
            f"execution.default_run_mode={default_mode!r} is not defined in execution.run_modes."
        )
    return validated_modes, default_mode


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


def _validate_behavior_task(value: Any, *, field_name: str) -> dict[str, Any]:
    task = _mapping(value, field_name=field_name)
    for name in ("task_id", "prompt_template", "primary_outcome", "response_format"):
        _nonempty_string(task.get(name), field_name=f"{field_name}.{name}")
    item_schema = _mapping(
        task.get("item_metadata_schema"),
        field_name=f"{field_name}.item_metadata_schema",
    )
    properties = _mapping(
        item_schema.get("properties"),
        field_name=f"{field_name}.item_metadata_schema.properties",
    )
    required = _string_list(
        item_schema.get("required"),
        field_name=f"{field_name}.item_metadata_schema.required",
        allow_empty=False,
    )
    if set(required) != set(properties):
        raise ValueError(
            f"{field_name}.item_metadata_schema must require every declared property exactly once."
        )
    for property_name, raw_property in properties.items():
        _validate_id(property_name, field_name="item_metadata_schema property")
        property_schema = _mapping(
            raw_property,
            field_name=f"{field_name}.item_metadata_schema.properties.{property_name}",
        )
        property_type = _nonempty_string(
            property_schema.get("type"),
            field_name=f"{field_name}.item_metadata_schema.properties.{property_name}.type",
        )
        if property_type not in {"string", "integer", "number", "boolean"}:
            raise ValueError(f"Unsupported item metadata type={property_type!r}.")
    item_schema["required"] = list(required)
    item_schema["properties"] = properties
    task["item_metadata_schema"] = item_schema
    return task


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
    collateral_behavior_task: dict[str, Any] | None = None
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
        behavior_task = _validate_behavior_task(
            payload.get("independent_behavior_task"),
            field_name="independent_behavior_task",
        )
        raw_collateral_task = payload.get("collateral_behavior_task")
        collateral_task = (
            None
            if raw_collateral_task is None
            else _validate_behavior_task(
                raw_collateral_task,
                field_name="collateral_behavior_task",
            )
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
            collateral_behavior_task=collateral_task,
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
        result = {
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
        if self.collateral_behavior_task is not None:
            result["collateral_behavior_task"] = dict(self.collateral_behavior_task)
        return result


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
    storage: dict[str, Any]
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
        if len(set(layers)) != len(layers):
            raise ValueError("activation.layers must not contain duplicates.")
        layer_selection = _nonempty_string(
            activation.get("layer_selection", "validation_max_margin"),
            field_name="activation.layer_selection",
        )
        if layer_selection not in {"validation_max_margin", "fixed"}:
            raise ValueError("activation.layer_selection must be 'validation_max_margin' or 'fixed'.")
        activation["layer_selection"] = layer_selection
        _nonempty_string(activation.get("activation_site"), field_name="activation.activation_site")
        _nonempty_string(activation.get("token_mode"), field_name="activation.token_mode")
        storage_dtype = _nonempty_string(
            activation.get("storage_dtype"), field_name="activation.storage_dtype"
        )
        if storage_dtype not in {"float16", "float32"}:
            raise ValueError("activation.storage_dtype must be 'float16' or 'float32'.")
        activation["storage_dtype"] = storage_dtype
        batch_size = activation.get("batch_size")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("activation.batch_size must be a positive integer.")

        steering = _mapping(payload.get("steering"), field_name="steering")
        scales = steering.get("scales")
        if not isinstance(scales, (list, tuple)) or not scales:
            raise ValueError("steering.scales must be a non-empty list.")
        if any(not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) for scale in scales):
            raise ValueError("steering.scales must contain finite numbers.")
        if 0.0 not in {float(scale) for scale in scales}:
            raise ValueError("steering.scales must include a zero-dose condition.")
        if not any(float(scale) > 0 for scale in scales) or not any(float(scale) < 0 for scale in scales):
            raise ValueError("steering.scales must include positive and negative doses.")
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
        calibration_method = _nonempty_string(
            steering.get("calibration"), field_name="steering.calibration"
        )
        if calibration_method not in {"neutral", "within_condition"}:
            raise ValueError("steering.calibration must be 'neutral' or 'within_condition'.")
        random_direction_count = steering.get("random_direction_count")
        if not isinstance(random_direction_count, int) or random_direction_count < 1:
            raise ValueError("steering.random_direction_count must be a positive integer.")
        if intervention_timing == "fixed_window":
            window = steering.get("fixed_window")
            if (
                not isinstance(window, (list, tuple))
                or len(window) != 2
                or any(not isinstance(index, int) for index in window)
                or window[0] < 0
                or window[1] <= window[0]
            ):
                raise ValueError("fixed_window timing requires steering.fixed_window=[start, end].")

        output_root = _nonempty_string(payload.get("output_root"), field_name="output_root")
        seed = payload.get("seed")
        if not isinstance(seed, int):
            raise ValueError("seed must be an integer.")
        analysis_spec_id = _validate_id(
            payload.get("analysis_spec_id", "rsc_benchmark_core_v1"), field_name="analysis_spec_id"
        )
        execution = _mapping(payload.get("execution", {}), field_name="execution")
        validated_run_modes, default_run_mode = _validate_execution_run_modes(
            execution.get("run_modes"),
            default_mode=execution.get("default_run_mode", "test"),
        )
        execution["run_modes"] = validated_run_modes
        execution["default_run_mode"] = default_run_mode
        raw_storage = _mapping(payload.get("storage", {}), field_name="storage")
        storage = dict(DEFAULT_STORAGE_SETTINGS)
        storage.update(raw_storage)
        for field_name in (
            "workspace_root_env",
            "archive_uri_env",
            "sync_endpoint_env",
            "sync_tool",
        ):
            _nonempty_string(storage.get(field_name), field_name=f"storage.{field_name}")
        if storage["sync_tool"] != "aws":
            raise ValueError("storage.sync_tool must be 'aws'.")
        for field_name in ("sync_on_finalize", "keep_local_copy"):
            if not isinstance(storage[field_name], bool):
                raise ValueError(f"storage.{field_name} must be a boolean.")
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
            storage=storage,
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
            "storage": dict(self.storage),
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
