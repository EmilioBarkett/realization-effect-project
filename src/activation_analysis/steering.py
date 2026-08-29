from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from .model_loading import (
    component_attribute,
    decode_tokens,
    load_model,
    load_tokenizer_or_processor,
    move_batch_to_device,
)
from .constrained_generation import build_numeric_logits_processor
from .tokenization import (
    enforce_token_length_limit,
    format_model_prompt,
    inspect_token_lengths,
    tokenize_padded_without_truncation,
    validate_padded_encoding_lengths,
)


SUPPORTED_POSITION_MODES = {"all", "last"}
SUPPORTED_INTERVENTION_TIMINGS = {"prefill_only", "generation_only", "every_step", "fixed_window"}
SUPPORTED_TRACKING_SOURCES = {
    "injection_direction_train_only",
    "independent_train_only",
    "same_vector_persistence_diagnostic",
}


@dataclass(frozen=True)
class SteeringConfig:
    """Generation-time residual stream steering configuration.

    Layer numbers are 1-based transformer block indices, matching the residual
    stream logger and activation-vector output naming used in this repo.
    """

    direction_path: Path
    layer: int
    scale: float
    position_mode: str = "last"
    normalize_direction: bool = True
    intervention_timing: str = "every_step"
    fixed_window_start: int | None = None
    fixed_window_end: int | None = None
    direction_id: str = "injected_direction"
    direction_source: str = "injection_direction_train_only"
    direction_role: str = "injection_immediate"
    requested_dose: float | None = None
    calibration_projection_scale: float | None = None


@dataclass(frozen=True)
class SteeringVectorInfo:
    path: str
    hidden_size: int
    raw_norm: float
    normalized: bool


@dataclass(frozen=True)
class TrackingDirection:
    """A direction used to read a layer after the intervention.

    ``independent_train_only`` is the primary downstream readout source.  The
    same-vector source is retained only as an explicitly labelled diagnostic;
    it must not be confused with a construct-state readout at the downstream
    layer.
    """

    layer: int
    direction_path: Path
    direction_id: str
    source: str = "independent_train_only"
    role: str = "downstream_construct_state"


@dataclass(frozen=True)
class InjectionObservation:
    """Scalar pre/post arithmetic for one hooked forward pass."""

    layer: int
    direction_id: str
    direction_source: str
    direction_role: str
    forward_index: int
    phase: str
    token_position: int
    injection_applied: bool
    pre_projection: float
    post_projection: float
    observed_shift: float
    requested_dose: float | None
    physical_scale: float
    calibrated_projection_scale: float | None
    expected_shift: float
    expected_observed_difference: float


@dataclass(frozen=True)
class ProjectionObservation:
    """A scalar projection recorded at an injection or downstream layer."""

    layer: int
    direction_id: str
    direction_source: str
    direction_role: str
    direction_path: str
    forward_index: int
    phase: str
    token_position: int
    projection: float
    injection_applied: bool


@dataclass
class SteeringTrace:
    """Small, JSON-serializable steering manipulation-check trace.

    Hooks append only Python scalars to this object.  No activation tensor is
    retained after a hook returns.
    """

    injection_layer: int
    intervention_timing: str
    position_mode: str
    direction_id: str
    direction_source: str
    direction_role: str
    requested_dose: float | None
    physical_scale: float
    calibration_projection_scale: float | None
    injection_direction: SteeringVectorInfo | None = None
    tracking_directions: list[dict[str, Any]] = field(default_factory=list)
    injection_observations: list[InjectionObservation] = field(default_factory=list)
    projection_observations: list[ProjectionObservation] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        """Return a stable mapping suitable for a JSONL raw-run record."""

        direction_info = None
        if self.injection_direction is not None:
            direction_info = {
                "path": self.injection_direction.path,
                "hidden_size": self.injection_direction.hidden_size,
                "raw_norm": self.injection_direction.raw_norm,
                "normalized": self.injection_direction.normalized,
            }
        return {
            "injection_layer": self.injection_layer,
            "intervention_timing": self.intervention_timing,
            "position_mode": self.position_mode,
            "direction_id": self.direction_id,
            "direction_source": self.direction_source,
            "direction_role": self.direction_role,
            "requested_dose": self.requested_dose,
            "physical_scale": self.physical_scale,
            "calibration_projection_scale": self.calibration_projection_scale,
            "injection_direction": direction_info,
            "tracking_directions": list(self.tracking_directions),
            "injection_observations": [
                {
                    "layer": observation.layer,
                    "direction_id": observation.direction_id,
                    "direction_source": observation.direction_source,
                    "direction_role": observation.direction_role,
                    "forward_index": observation.forward_index,
                    "phase": observation.phase,
                    "token_position": observation.token_position,
                    "injection_applied": observation.injection_applied,
                    "pre_projection": observation.pre_projection,
                    "post_projection": observation.post_projection,
                    "observed_shift": observation.observed_shift,
                    "requested_dose": observation.requested_dose,
                    "physical_scale": observation.physical_scale,
                    "calibrated_projection_scale": observation.calibrated_projection_scale,
                    "expected_shift": observation.expected_shift,
                    "expected_observed_difference": observation.expected_observed_difference,
                }
                for observation in self.injection_observations
            ],
            "projection_observations": [
                {
                    "layer": observation.layer,
                    "direction_id": observation.direction_id,
                    "direction_source": observation.direction_source,
                    "direction_role": observation.direction_role,
                    "direction_path": observation.direction_path,
                    "forward_index": observation.forward_index,
                    "phase": observation.phase,
                    "token_position": observation.token_position,
                    "projection": observation.projection,
                    "injection_applied": observation.injection_applied,
                }
                for observation in self.projection_observations
            ],
        }


def _validate_intervention_timing(config: SteeringConfig) -> None:
    if config.intervention_timing not in SUPPORTED_INTERVENTION_TIMINGS:
        supported = ", ".join(sorted(SUPPORTED_INTERVENTION_TIMINGS))
        raise ValueError(f"intervention_timing must be one of: {supported}.")
    if config.intervention_timing == "fixed_window":
        if config.fixed_window_start is None or config.fixed_window_end is None:
            raise ValueError("fixed_window timing requires fixed_window_start and fixed_window_end.")
        if config.fixed_window_start < 0 or config.fixed_window_end <= config.fixed_window_start:
            raise ValueError("fixed_window must satisfy 0 <= start < end.")
    elif config.fixed_window_start is not None or config.fixed_window_end is not None:
        raise ValueError("fixed_window_start/end may only be set for fixed_window timing.")


def should_inject_at_forward(
    intervention_timing: str,
    forward_index: int,
    *,
    fixed_window_start: int | None = None,
    fixed_window_end: int | None = None,
) -> bool:
    """Return whether to inject on a generate() forward pass.

    Forward index 0 is the prompt-prefill pass. Later indices are autoregressive
    generation passes. Fixed windows use a half-open interval over these indices.
    """

    config = SteeringConfig(
        direction_path=Path("unused.npy"),
        layer=1,
        scale=0.0,
        intervention_timing=intervention_timing,
        fixed_window_start=fixed_window_start,
        fixed_window_end=fixed_window_end,
    )
    _validate_intervention_timing(config)
    if forward_index < 0:
        raise ValueError("forward_index must be non-negative.")
    if intervention_timing == "prefill_only":
        return forward_index == 0
    if intervention_timing == "generation_only":
        return forward_index > 0
    if intervention_timing == "every_step":
        return True
    assert fixed_window_start is not None and fixed_window_end is not None
    return fixed_window_start <= forward_index < fixed_window_end


class ResidualSteeringGenerator:
    """Generate text while injecting a saved direction into one residual layer."""

    def __init__(
        self,
        model_id: str | Path,
        tokenizer_id: str | Path | None = None,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        device: str = "auto",
        dtype: str = "auto",
        device_map: str | None = None,
        attn_implementation: str | None = None,
        block_path: str | None = None,
    ) -> None:
        self._torch, self._transformers = self._import_dependencies()
        self.model_id = str(model_id)
        self.tokenizer_id = str(tokenizer_id or model_id)
        self.revision = revision
        self.block_path = block_path
        self.resolved_block_path: str | None = None

        self.tokenizer, self.tokenizer_loader = load_tokenizer_or_processor(
            self._transformers,
            self.tokenizer_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

        model_dtype = self._resolve_dtype(dtype)
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": model_dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map:
            model_kwargs["device_map"] = device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = load_model(self._transformers, self.model_id, model_kwargs)
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        if device_map:
            self.device = str(next(self.model.parameters()).device)
            self.resolved_device = f"device_map:{device_map}; input_device:{self.device}"
        else:
            self.device = self._resolve_device(device)
            self.resolved_device = self.device
            self.model = self.model.to(self.device)
        self.model.eval()

        self.num_transformer_layers = int(self._config_value("num_hidden_layers"))
        self.d_model = int(self._config_value("hidden_size"))
        self._direction_cache: dict[tuple[str, bool], tuple[Any, SteeringVectorInfo]] = {}
        self.last_tokenization_report: dict[str, Any] | None = None

    def _import_dependencies(self) -> tuple[Any, Any]:
        try:
            import torch
            import transformers
        except Exception as exc:  # pragma: no cover - environment-specific.
            raise RuntimeError(
                "ResidualSteeringGenerator requires torch and transformers. "
                "Install the activation-analysis interpretability dependencies first."
            ) from exc
        return torch, transformers

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        if getattr(self._torch.backends, "mps", None) and self._torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, requested: str) -> Any:
        if requested == "auto":
            if self._torch.cuda.is_available():
                return self._torch.bfloat16
            if getattr(self._torch.backends, "mps", None) and self._torch.backends.mps.is_available():
                return self._torch.bfloat16
            return self._torch.float32
        requested = requested.lower()
        if requested in {"bf16", "bfloat16"}:
            return self._torch.bfloat16
        if requested in {"fp16", "float16"}:
            return self._torch.float16
        if requested in {"fp32", "float32"}:
            return self._torch.float32
        raise ValueError(f"Unsupported dtype value: {requested}")

    def _config_value(self, name: str) -> Any:
        if hasattr(self.model.config, name):
            return getattr(self.model.config, name)
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, name):
            return getattr(text_config, name)
        raise RuntimeError(f"Model config does not expose '{name}'.")

    @staticmethod
    def _candidate_block_paths() -> list[tuple[str, ...]]:
        return [
            ("model", "layers"),
            ("model", "language_model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "decoder", "layers"),
            ("transformer", "h"),
            ("transformer", "blocks"),
            ("gpt_neox", "layers"),
            ("decoder", "layers"),
            ("layers",),
        ]

    def _resolve_module_path(self, path: tuple[str, ...]) -> Any:
        node = self.model
        for name in path:
            if not hasattr(node, name):
                joined = ".".join(path)
                raise AttributeError(f"'{type(node).__name__}' has no attribute '{name}' while resolving {joined}")
            node = getattr(node, name)
        return node

    def _blocks_from_path(self, path: tuple[str, ...]) -> list[Any]:
        node = self._resolve_module_path(path)
        try:
            blocks = list(node)
        except TypeError as exc:
            joined = ".".join(path)
            raise TypeError(f"Resolved path {joined} is not iterable.") from exc
        if not blocks:
            joined = ".".join(path)
            raise ValueError(f"Resolved path {joined} did not contain any blocks.")
        return blocks

    def _resolve_transformer_blocks(self) -> list[Any]:
        if self.block_path:
            path = tuple(part for part in self.block_path.split(".") if part)
            if not path:
                raise ValueError("--block-path must not be empty when provided.")
            try:
                blocks = self._blocks_from_path(path)
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to resolve explicit transformer block path '{self.block_path}'. "
                    "Pass a dotted module path such as 'model.layers' or 'transformer.h'."
                ) from exc
            self.resolved_block_path = ".".join(path)
            return blocks

        failures: list[str] = []
        for path in self._candidate_block_paths():
            try:
                blocks = self._blocks_from_path(path)
            except Exception as exc:
                failures.append(f"{'.'.join(path)} ({type(exc).__name__})")
                continue
            self.resolved_block_path = ".".join(path)
            return blocks
        inspected = ", ".join(failures)
        raise RuntimeError(
            "Unable to resolve transformer block modules for this model architecture. "
            f"Inspected paths: {inspected}. "
            "Use --block-path with the dotted module path to the transformer block list."
        )

    def resolve_block_path(self) -> str:
        self._resolve_transformer_blocks()
        return self.resolved_block_path or ""

    def load_direction(self, config: SteeringConfig) -> tuple[Any, SteeringVectorInfo]:
        if config.position_mode not in SUPPORTED_POSITION_MODES:
            supported = ", ".join(sorted(SUPPORTED_POSITION_MODES))
            raise ValueError(f"position_mode must be one of: {supported}.")
        cache_key = (str(config.direction_path.resolve()), config.normalize_direction)
        cached = self._direction_cache.get(cache_key)
        if cached is not None:
            return cached
        array = np.load(config.direction_path)
        if array.ndim != 1:
            raise ValueError(f"Expected a 1D steering vector, got shape {array.shape}.")
        if array.shape[0] != self.d_model:
            raise ValueError(
                f"Direction hidden size {array.shape[0]} does not match model hidden size {self.d_model}."
            )
        direction = self._torch.as_tensor(array, dtype=self._torch.float32)
        raw_norm = float(self._torch.linalg.vector_norm(direction).item())
        if raw_norm == 0:
            raise ValueError("Steering direction has zero norm.")
        if config.normalize_direction:
            direction = direction / raw_norm
        info = SteeringVectorInfo(
            path=str(config.direction_path),
            hidden_size=int(array.shape[0]),
            raw_norm=raw_norm,
            normalized=config.normalize_direction,
        )
        self._direction_cache[cache_key] = (direction, info)
        return direction, info

    @staticmethod
    def _coerce_tracking_direction(layer: int, value: Any) -> TrackingDirection:
        if isinstance(value, TrackingDirection):
            tracking = value
        elif isinstance(value, (str, Path)):
            tracking = TrackingDirection(
                layer=layer,
                direction_path=Path(value),
                direction_id=f"layer_{layer:02d}_direction",
            )
        elif isinstance(value, Mapping):
            path = value.get("direction_path", value.get("path"))
            direction_id = value.get("direction_id")
            if path is None or not direction_id:
                raise ValueError(
                    "Tracking-direction mappings require 'path'/'direction_path' and 'direction_id'."
                )
            tracking = TrackingDirection(
                layer=int(value.get("layer", layer)),
                direction_path=Path(path),
                direction_id=str(direction_id),
                source=str(value.get("source", "independent_train_only")),
                role=str(value.get("role", "downstream_construct_state")),
            )
        else:
            raise TypeError(
                "tracking_directions values must be TrackingDirection, path-like, or a mapping."
            )
        if tracking.layer != layer:
            raise ValueError(
                f"Tracking direction layer key {layer} does not match its declared layer {tracking.layer}."
            )
        if tracking.layer < 1:
            raise ValueError("Tracking direction layers are 1-based and must be >= 1.")
        if tracking.source not in SUPPORTED_TRACKING_SOURCES:
            supported = ", ".join(sorted(SUPPORTED_TRACKING_SOURCES))
            raise ValueError(f"tracking direction source must be one of: {supported}.")
        if not tracking.direction_id.strip():
            raise ValueError("Tracking direction direction_id must not be empty.")
        return tracking

    def _expected_shift(self, config: SteeringConfig, info: SteeringVectorInfo, *, applied: bool) -> float:
        if not applied:
            return 0.0
        if config.requested_dose is not None and config.calibration_projection_scale is not None:
            return float(config.requested_dose * config.calibration_projection_scale)
        return float(config.scale * (1.0 if info.normalized else info.raw_norm))

    def _tracked_position(
        self,
        tensor: Any,
        *,
        forward_index: int,
        prefill_token_position: int | None,
    ) -> tuple[int, int]:
        position_index = int(tensor.shape[1]) - 1
        if prefill_token_position is None:
            return position_index, position_index
        if forward_index == 0:
            return position_index, int(prefill_token_position)
        return position_index, int(prefill_token_position + forward_index)

    def _projection(self, tensor: Any, direction: Any, *, position_index: int) -> float:
        direction_cast = direction.to(device=tensor.device, dtype=tensor.dtype)
        projection = self._torch.sum(tensor[0, position_index, :] * direction_cast)
        # Convert immediately to a host scalar so hooks never retain an
        # activation or a device tensor after returning.
        return float(projection.detach().float().cpu().item())

    @contextmanager
    def steering_hooks(
        self,
        config: SteeringConfig,
        *,
        tracking_directions: Mapping[int, TrackingDirection | Path | str | Mapping[str, Any]] | None = None,
        trace: SteeringTrace | None = None,
        prefill_token_position: int | None = None,
    ) -> Iterator[SteeringVectorInfo]:
        if config.layer < 1:
            raise ValueError("Layer numbers are 1-based and must be >= 1.")
        blocks = self._resolve_transformer_blocks()
        if config.layer > len(blocks):
            raise ValueError(f"Requested layer {config.layer}, but model has only {len(blocks)} blocks.")
        _validate_intervention_timing(config)
        direction, info = self.load_direction(config)
        if not np.isfinite(float(config.scale)):
            raise ValueError("Steering scale must be finite.")
        if config.requested_dose is not None and not np.isfinite(float(config.requested_dose)):
            raise ValueError("requested_dose must be finite when provided.")
        if config.calibration_projection_scale is not None and (
            not np.isfinite(float(config.calibration_projection_scale))
            or float(config.calibration_projection_scale) <= 0
        ):
            raise ValueError("calibration_projection_scale must be finite and greater than zero.")

        normalized_tracking: dict[int, TrackingDirection] = {
            config.layer: TrackingDirection(
                layer=config.layer,
                direction_path=config.direction_path,
                direction_id=config.direction_id,
                source=config.direction_source,
                role=config.direction_role,
            )
        }
        for layer, value in (tracking_directions or {}).items():
            layer_number = int(layer)
            # The injection layer is always the condition-specific direction
            # (target, shuffled, or random).  Do not let the plan's target
            # readout entry overwrite that control direction.
            if layer_number == config.layer:
                continue
            tracking = self._coerce_tracking_direction(layer_number, value)
            normalized_tracking[layer_number] = tracking
        for layer in normalized_tracking:
            if layer > len(blocks):
                raise ValueError(f"Requested tracking layer {layer}, but model has only {len(blocks)} blocks.")

        loaded_tracking: dict[int, tuple[TrackingDirection, Any, SteeringVectorInfo]] = {}
        for layer, tracking in sorted(normalized_tracking.items()):
            if layer == config.layer:
                loaded_tracking[layer] = (tracking, direction, info)
            else:
                tracking_config = SteeringConfig(
                    direction_path=tracking.direction_path,
                    layer=layer,
                    scale=0.0,
                    position_mode=config.position_mode,
                    normalize_direction=config.normalize_direction,
                )
                downstream_direction, downstream_info = self.load_direction(tracking_config)
                loaded_tracking[layer] = (tracking, downstream_direction, downstream_info)

        if trace is not None:
            trace.injection_direction = info
            trace.tracking_directions = [
                {
                    "layer": layer,
                    "direction_id": tracking.direction_id,
                    "direction_source": tracking.source,
                    "direction_role": tracking.role,
                    "direction_path": str(tracking.direction_path),
                    "hidden_size": tracking_info.hidden_size,
                    "raw_norm": tracking_info.raw_norm,
                    "normalized": tracking_info.normalized,
                }
                for layer, (tracking, _tracking_direction, tracking_info) in loaded_tracking.items()
            ]

        handles = []
        try:
            for layer, (tracking, tracking_direction, tracking_info) in loaded_tracking.items():
                if layer == config.layer:
                    hook = self._make_injection_tracking_hook(
                        config=config,
                        direction=direction,
                        info=info,
                        trace=trace,
                        prefill_token_position=prefill_token_position,
                    )
                else:
                    hook = self._make_projection_tracking_hook(
                        config=config,
                        tracking=tracking,
                        direction=tracking_direction,
                        info=tracking_info,
                        trace=trace,
                        prefill_token_position=prefill_token_position,
                    )
                handles.append(blocks[layer - 1].register_forward_hook(hook))
            yield info
        finally:
            for handle in reversed(handles):
                handle.remove()

    def _make_injection_tracking_hook(
        self,
        *,
        config: SteeringConfig,
        direction: Any,
        info: SteeringVectorInfo,
        trace: SteeringTrace | None,
        prefill_token_position: int | None,
    ):
        forward_index = 0

        def _hook(_module, _inputs, output):
            nonlocal forward_index
            tensor = self._extract_block_tensor(output)
            position_index, token_position = self._tracked_position(
                tensor,
                forward_index=forward_index,
                prefill_token_position=prefill_token_position,
            )
            pre_projection = self._projection(tensor, direction, position_index=position_index)
            injection_applied = should_inject_at_forward(
                config.intervention_timing,
                forward_index,
                fixed_window_start=config.fixed_window_start,
                fixed_window_end=config.fixed_window_end,
            )
            replacement = (
                self._inject(
                    tensor,
                    direction=direction,
                    scale=config.scale,
                    position_mode=config.position_mode,
                )
                if injection_applied
                else tensor
            )
            post_projection = self._projection(replacement, direction, position_index=position_index)
            observed_shift = post_projection - pre_projection
            expected_shift = self._expected_shift(config, info, applied=injection_applied)
            if trace is not None:
                if trace.injection_direction is None:
                    trace.injection_direction = info
                trace.injection_observations.append(
                    InjectionObservation(
                        layer=config.layer,
                        direction_id=config.direction_id,
                        direction_source=config.direction_source,
                        direction_role=config.direction_role,
                        forward_index=forward_index,
                        phase="prefill" if forward_index == 0 else "generation",
                        token_position=token_position,
                        injection_applied=injection_applied,
                        pre_projection=pre_projection,
                        post_projection=post_projection,
                        observed_shift=observed_shift,
                        requested_dose=config.requested_dose,
                        physical_scale=float(config.scale),
                        calibrated_projection_scale=config.calibration_projection_scale,
                        expected_shift=expected_shift,
                        expected_observed_difference=expected_shift - observed_shift,
                    )
                )
                trace.projection_observations.append(
                    ProjectionObservation(
                        layer=config.layer,
                        direction_id=config.direction_id,
                        direction_source=config.direction_source,
                        direction_role=config.direction_role,
                        direction_path=info.path,
                        forward_index=forward_index,
                        phase="prefill" if forward_index == 0 else "generation",
                        token_position=token_position,
                        projection=post_projection,
                        injection_applied=injection_applied,
                    )
                )
            forward_index += 1
            return self._replace_block_tensor(output, replacement)

        return _hook

    def _make_projection_tracking_hook(
        self,
        *,
        config: SteeringConfig,
        tracking: TrackingDirection,
        direction: Any,
        info: SteeringVectorInfo,
        trace: SteeringTrace | None,
        prefill_token_position: int | None,
    ):
        forward_index = 0

        def _hook(_module, _inputs, output):
            nonlocal forward_index
            tensor = self._extract_block_tensor(output)
            position_index, token_position = self._tracked_position(
                tensor,
                forward_index=forward_index,
                prefill_token_position=prefill_token_position,
            )
            projection = self._projection(tensor, direction, position_index=position_index)
            if trace is not None:
                trace.projection_observations.append(
                    ProjectionObservation(
                        layer=tracking.layer,
                        direction_id=tracking.direction_id,
                        direction_source=tracking.source,
                        direction_role=tracking.role,
                        direction_path=info.path,
                        forward_index=forward_index,
                        phase="prefill" if forward_index == 0 else "generation",
                        token_position=token_position,
                        projection=projection,
                        injection_applied=should_inject_at_forward(
                            config.intervention_timing,
                            forward_index,
                            fixed_window_start=config.fixed_window_start,
                            fixed_window_end=config.fixed_window_end,
                        ),
                    )
                )
            forward_index += 1
            return output

        return _hook

    def _make_hook(
        self,
        *,
        direction: Any,
        scale: float,
        position_mode: str,
        intervention_timing: str,
        fixed_window_start: int | None,
        fixed_window_end: int | None,
    ):
        forward_index = 0

        def _hook(_module, _inputs, output):
            nonlocal forward_index
            tensor = self._extract_block_tensor(output)
            if should_inject_at_forward(
                intervention_timing,
                forward_index,
                fixed_window_start=fixed_window_start,
                fixed_window_end=fixed_window_end,
            ):
                replacement = self._inject(
                    tensor,
                    direction=direction,
                    scale=scale,
                    position_mode=position_mode,
                )
            else:
                replacement = tensor
            forward_index += 1
            return self._replace_block_tensor(output, replacement)

        return _hook

    def _extract_block_tensor(self, block_output: Any) -> Any:
        tensor = block_output[0] if isinstance(block_output, (tuple, list)) else block_output
        if not hasattr(tensor, "shape"):
            raise RuntimeError("Transformer block output is not a tensor.")
        if len(tensor.shape) != 3:
            raise RuntimeError(
                "Expected transformer block output shape [batch, seq, hidden], "
                f"got {tuple(tensor.shape)}"
            )
        return tensor

    def _replace_block_tensor(self, original_output: Any, replacement: Any) -> Any:
        if isinstance(original_output, tuple):
            return (replacement, *original_output[1:])
        if isinstance(original_output, list):
            return [replacement, *original_output[1:]]
        return replacement

    def _inject(self, tensor: Any, *, direction: Any, scale: float, position_mode: str) -> Any:
        steer = direction.to(device=tensor.device, dtype=tensor.dtype) * scale
        steered = tensor.clone()
        if position_mode == "all":
            steered = steered + steer.view(1, 1, -1)
        elif position_mode == "last":
            steered[:, -1, :] = steered[:, -1, :] + steer
        else:
            supported = ", ".join(sorted(SUPPORTED_POSITION_MODES))
            raise ValueError(f"position_mode must be one of: {supported}.")
        return steered

    def format_prompt(
        self,
        prompt: str,
        *,
        prompt_format: str,
        system_prompt: str,
        enable_thinking: bool | None = None,
    ) -> str:
        return format_model_prompt(
            self.tokenizer,
            prompt,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )

    def generate(
        self,
        prompt: str,
        *,
        prompt_format: str = "completion",
        system_prompt: str = "",
        steering_config: SteeringConfig | None = None,
        max_new_tokens: int = 32,
        min_new_tokens: int = 4,
        max_length: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.0,
        enable_thinking: bool | None = None,
        parser_id: str | None = None,
        constrained_numeric: bool = True,
        tracking_directions: Mapping[int, TrackingDirection | Path | str | Mapping[str, Any]] | None = None,
        return_trace: bool = False,
    ) -> tuple[str, SteeringVectorInfo | None] | tuple[str, SteeringVectorInfo | None, SteeringTrace | None]:
        if tracking_directions and steering_config is None:
            raise ValueError("tracking_directions require a steering_config so the injection is identified.")
        formatted_prompt = self.format_prompt(
            prompt,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        token_report = inspect_token_lengths(
            self.tokenizer,
            [formatted_prompt],
            max_length=max_length,
            prompt_ids=["generated_prompt"],
            tokenizer_id=self.tokenizer_id,
            revision=self.revision,
        )
        enforce_token_length_limit(token_report)
        encoded = tokenize_padded_without_truncation(
            self.tokenizer,
            [formatted_prompt],
            return_tensors="pt",
        )
        validate_padded_encoding_lengths(encoded, token_report)
        self.last_tokenization_report = token_report.to_mapping()
        encoded = move_batch_to_device(encoded, self.device)
        input_length = int(encoded["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": component_attribute(self.tokenizer, "pad_token_id")
            or component_attribute(self.tokenizer, "eos_token_id"),
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        if constrained_numeric and parser_id:
            numeric_processor = build_numeric_logits_processor(
                self.tokenizer,
                parser_id=parser_id,
                start_length=input_length,
            )
            if numeric_processor is not None:
                processor_list = getattr(self._transformers, "LogitsProcessorList", None)
                generation_kwargs["logits_processor"] = (
                    processor_list([numeric_processor])
                    if callable(processor_list)
                    else [numeric_processor]
                )

        trace = None
        if steering_config is not None:
            trace = SteeringTrace(
                injection_layer=steering_config.layer,
                intervention_timing=steering_config.intervention_timing,
                position_mode=steering_config.position_mode,
                direction_id=steering_config.direction_id,
                direction_source=steering_config.direction_source,
                direction_role=steering_config.direction_role,
                requested_dose=steering_config.requested_dose,
                physical_scale=float(steering_config.scale),
                calibration_projection_scale=steering_config.calibration_projection_scale,
            )
            with self.steering_hooks(
                steering_config,
                tracking_directions=tracking_directions,
                trace=trace,
                prefill_token_position=input_length - 1,
            ) as info:
                with self._torch.no_grad():
                    output = self.model.generate(**generation_kwargs)
        else:
            with self._torch.no_grad():
                output = self.model.generate(**generation_kwargs)
            info = None

        # Some multimodal/processor-backed generation APIs return a generation
        # output object rather than the raw sequence tensor.
        sequences = getattr(output, "sequences", output)
        new_tokens = sequences[0, input_length:]
        text = decode_tokens(self.tokenizer, new_tokens, skip_special_tokens=True).strip()
        if return_trace:
            return text, info, trace
        return text, info
