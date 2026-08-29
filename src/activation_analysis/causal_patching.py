"""Matched-episode residual-stream causal patching.

This module implements the first causal-pathway analysis in the benchmark.
It is deliberately narrower than a complete mechanistic-interpretability
system: two matched induction episodes share an identical downstream task,
and one condition's residual state is interchanged into the other condition at
a registered probe-to-task boundary during prompt prefill.

The public runner stores only scalar diagnostics and generated text.  It does
not retain activation tensors after a run, which keeps the artifact suitable
for a manifest-backed model-side execution.  Component/path patching and
ablation are separate later methods and are not silently substituted here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_loading import component_attribute, decode_tokens
from .tokenization import (
    enforce_token_length_limit,
    format_model_prompt,
    inspect_token_lengths,
    tokenize_padded_without_truncation,
    validate_padded_encoding_lengths,
)


CAUSAL_PATCH_SCHEMA_VERSION = "0.1.0"
BOUNDARY_MODES = {"last_induction_token"}
INTERCHANGE_TIMINGS = {"prefill_only"}
INTERCHANGE_VARIANTS = {
    "natural_state_replacement",
    "donor_minus_recipient_delta",
}


class _EarlyStopForward(RuntimeError):
    """Stop a source capture after the final requested block."""


@dataclass(frozen=True)
class MatchedEpisode:
    """A pair of induction contexts followed by one identical task.

    ``positive_induction_prompt`` and ``negative_induction_prompt`` are the
    only text that differs between the two source conditions.  The runner
    appends the same ``downstream_prompt`` to both before tokenization.  The
    last token of each induction prefix is the only registered patch target.
    """

    request_id: str
    positive_induction_prompt: str
    negative_induction_prompt: str
    downstream_prompt: str
    construct_id: str = ""
    positive_source_prompt_id: str = ""
    negative_source_prompt_id: str = ""
    downstream_prompt_id: str = ""
    positive_condition: str = "positive"
    negative_condition: str = "negative"
    downstream_task_id: str = ""
    boundary_separator: str = "\n\n"
    prompt_format: str = "completion"
    system_prompt: str = ""
    boundary_mode: str = "last_induction_token"
    intervention_timing: str = "prefill_only"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text_fields = {
            "request_id": self.request_id,
            "positive_induction_prompt": self.positive_induction_prompt,
            "negative_induction_prompt": self.negative_induction_prompt,
            "downstream_prompt": self.downstream_prompt,
            "positive_condition": self.positive_condition,
            "negative_condition": self.negative_condition,
        }
        for field_name, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if not isinstance(self.boundary_separator, str) or not self.boundary_separator:
            raise ValueError("boundary_separator must be a non-empty string.")
        if self.positive_condition == self.negative_condition:
            raise ValueError("positive_condition and negative_condition must differ.")
        if self.prompt_format not in {"completion", "chat"}:
            raise ValueError("prompt_format must be 'completion' or 'chat'.")
        if self.boundary_mode not in BOUNDARY_MODES:
            supported = ", ".join(sorted(BOUNDARY_MODES))
            raise ValueError(f"boundary_mode must be one of: {supported}.")
        if self.intervention_timing not in INTERCHANGE_TIMINGS:
            supported = ", ".join(sorted(INTERCHANGE_TIMINGS))
            raise ValueError(f"intervention_timing must be one of: {supported}.")
        for field_name in (
            "positive_source_prompt_id",
            "negative_source_prompt_id",
            "downstream_prompt_id",
            "downstream_task_id",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string when provided.")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")

    @property
    def resolved_positive_source_prompt_id(self) -> str:
        return self.positive_source_prompt_id or f"{self.request_id}__positive_source"

    @property
    def resolved_negative_source_prompt_id(self) -> str:
        return self.negative_source_prompt_id or f"{self.request_id}__negative_source"

    @property
    def resolved_downstream_prompt_id(self) -> str:
        return self.downstream_prompt_id or f"{self.request_id}__downstream"

    @property
    def resolved_downstream_task_id(self) -> str:
        return self.downstream_task_id or self.resolved_downstream_prompt_id

    def to_mapping(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "construct_id": self.construct_id,
            "positive_source_prompt_id": self.resolved_positive_source_prompt_id,
            "negative_source_prompt_id": self.resolved_negative_source_prompt_id,
            "downstream_prompt_id": self.resolved_downstream_prompt_id,
            "downstream_task_id": self.resolved_downstream_task_id,
            "positive_condition": self.positive_condition,
            "negative_condition": self.negative_condition,
            "positive_induction_prompt": self.positive_induction_prompt,
            "negative_induction_prompt": self.negative_induction_prompt,
            "downstream_prompt": self.downstream_prompt,
            "boundary_separator": self.boundary_separator,
            "prompt_format": self.prompt_format,
            "system_prompt": self.system_prompt,
            "boundary_mode": self.boundary_mode,
            "intervention_timing": self.intervention_timing,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MatchedEpisode":
        """Build a request from canonical JSONL data.

        ``positive_prompt``/``negative_prompt`` are accepted as migration
        aliases, but all emitted records use the explicit induction names.
        """

        if not isinstance(raw, Mapping):
            raise ValueError("Matched episode input must be a JSON object.")
        positive = raw.get("positive_induction_prompt", raw.get("positive_prompt"))
        negative = raw.get("negative_induction_prompt", raw.get("negative_prompt"))
        required = {
            "request_id": raw.get("request_id"),
            "positive_induction_prompt": positive,
            "negative_induction_prompt": negative,
            "downstream_prompt": raw.get("downstream_prompt"),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Matched episode is missing required fields: {missing}.")
        return cls(
            request_id=str(required["request_id"]),
            positive_induction_prompt=str(positive),
            negative_induction_prompt=str(negative),
            downstream_prompt=str(required["downstream_prompt"]),
            construct_id=str(raw.get("construct_id", "")),
            positive_source_prompt_id=str(raw.get("positive_source_prompt_id", "")),
            negative_source_prompt_id=str(raw.get("negative_source_prompt_id", "")),
            downstream_prompt_id=str(raw.get("downstream_prompt_id", "")),
            positive_condition=str(raw.get("positive_condition", "positive")),
            negative_condition=str(raw.get("negative_condition", "negative")),
            downstream_task_id=str(raw.get("downstream_task_id", "")),
            boundary_separator=str(raw.get("boundary_separator", "\n\n")),
            prompt_format=str(raw.get("prompt_format", "completion")),
            system_prompt=str(raw.get("system_prompt", "")),
            boundary_mode=str(raw.get("boundary_mode", "last_induction_token")),
            intervention_timing=str(raw.get("intervention_timing", "prefill_only")),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class InterchangeObservation:
    """One donor-to-receiver residual interchange observation."""

    request_id: str
    construct_id: str
    layer: int
    patch_direction: str
    donor_condition: str
    receiver_condition: str
    donor_prompt_id: str
    receiver_prompt_id: str
    downstream_prompt_id: str
    intervention_variant: str
    intervention_timing: str
    boundary_mode: str
    donor_boundary_position: int
    receiver_boundary_position: int
    donor_token_length: int
    receiver_token_length: int
    receiver_baseline_output: str
    patched_output: str
    patch_applied: bool
    forward_calls: int
    receiver_pre_norm: float
    donor_norm: float
    replacement_delta_l2: float
    receiver_post_norm: float

    @property
    def record_id(self) -> str:
        return f"{self.request_id}__{self.patch_direction}__layer_{self.layer:02d}"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "request_id": self.request_id,
            "construct_id": self.construct_id,
            "layer": self.layer,
            "patch_direction": self.patch_direction,
            "donor_condition": self.donor_condition,
            "receiver_condition": self.receiver_condition,
            "donor_prompt_id": self.donor_prompt_id,
            "receiver_prompt_id": self.receiver_prompt_id,
            "downstream_prompt_id": self.downstream_prompt_id,
            "intervention_variant": self.intervention_variant,
            "intervention_timing": self.intervention_timing,
            "boundary_mode": self.boundary_mode,
            "donor_boundary_position": self.donor_boundary_position,
            "receiver_boundary_position": self.receiver_boundary_position,
            "donor_token_length": self.donor_token_length,
            "receiver_token_length": self.receiver_token_length,
            "receiver_baseline_output": self.receiver_baseline_output,
            "patched_output": self.patched_output,
            "patch_applied": self.patch_applied,
            "forward_calls": self.forward_calls,
            "receiver_pre_norm": self.receiver_pre_norm,
            "donor_norm": self.donor_norm,
            "replacement_delta_l2": self.replacement_delta_l2,
            "receiver_post_norm": self.receiver_post_norm,
        }


@dataclass(frozen=True)
class _EncodedEpisode:
    prompt_id: str
    formatted_prompt: str
    encoded: Mapping[str, Any]
    source_encoded: Mapping[str, Any]
    boundary_position: int
    source_boundary_position: int
    boundary_char_start: int
    boundary_char_end: int
    token_length: int


def compose_matched_episode(
    induction_prompt: str,
    downstream_prompt: str,
    *,
    boundary_separator: str = "\n\n",
) -> str:
    """Compose an induction prefix and downstream task without mutation."""

    if not induction_prompt.strip() or not downstream_prompt.strip():
        raise ValueError("Both induction_prompt and downstream_prompt must be non-empty.")
    if not boundary_separator:
        raise ValueError("boundary_separator must be non-empty.")
    return f"{induction_prompt}{boundary_separator}{downstream_prompt}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_rows(value: Any) -> list[list[Any]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("Expected a list or tensor-like value.")
    if value and not isinstance(value[0], list):
        value = [value]
    if not all(isinstance(row, list) for row in value):
        raise ValueError("Expected a two-dimensional list or tensor-like value.")
    return value


def _row_value(value: Any, row_index: int) -> list[Any]:
    rows = _as_rows(value)
    if row_index >= len(rows):
        raise ValueError(f"Tokenizer output has no row {row_index}.")
    return rows[row_index]


def _offset_pair(value: Any, row_index: int, position: int) -> tuple[int, int]:
    rows = _as_rows(value)
    try:
        pair = rows[row_index][position]
        return int(pair[0]), int(pair[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("Tokenizer offset_mapping is malformed.") from exc


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


class MatchedEpisodeResidualPatcher:
    """Run matched residual interchange on an already-loaded model.

    The constructor accepts an existing model/tokenizer so RunPod callers can
    share one loaded model with other model-side passes.  The CLI uses the
    repository's ``ResidualSteeringGenerator`` solely as the common loader;
    no steering direction is used by this class.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "auto",
        block_path: str | None = None,
        model_id: str = "",
        tokenizer_id: str = "",
        revision: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment-specific.
            raise RuntimeError(
                "MatchedEpisodeResidualPatcher requires torch. Install the optional "
                "activation-analysis interpretability dependencies first."
            ) from exc
        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.tokenizer_id = str(tokenizer_id)
        self.revision = revision
        self.enable_thinking = enable_thinking
        self.block_path = block_path
        self.resolved_block_path: str | None = None
        self.device = self._resolve_device(device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _resolve_device(self, requested: str) -> Any:
        if requested != "auto":
            return self._torch.device(requested)
        if hasattr(self.model, "parameters"):
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        if getattr(self._torch.backends, "mps", None) and self._torch.backends.mps.is_available():
            return self._torch.device("mps")
        return self._torch.device("cpu")

    def _resolve_module_path(self, path: tuple[str, ...]) -> Any:
        node = self.model
        for name in path:
            if not hasattr(node, name):
                raise AttributeError(f"Model has no attribute '{name}' while resolving {'.'.join(path)}.")
            node = getattr(node, name)
        return node

    def _blocks_from_path(self, path: tuple[str, ...]) -> list[Any]:
        node = self._resolve_module_path(path)
        try:
            blocks = list(node)
        except TypeError as exc:
            raise TypeError(f"Resolved block path {'.'.join(path)} is not iterable.") from exc
        if not blocks:
            raise ValueError(f"Resolved block path {'.'.join(path)} is empty.")
        return blocks

    def resolve_transformer_blocks(self) -> list[Any]:
        if self.block_path:
            path = tuple(part for part in self.block_path.split(".") if part)
            if not path:
                raise ValueError("block_path must not be empty.")
            try:
                blocks = self._blocks_from_path(path)
            except Exception as exc:
                raise RuntimeError(f"Unable to resolve block_path='{self.block_path}'.") from exc
            self.resolved_block_path = ".".join(path)
            return blocks

        failures: list[str] = []
        for path in _candidate_block_paths():
            try:
                blocks = self._blocks_from_path(path)
            except Exception as exc:
                failures.append(f"{'.'.join(path)} ({type(exc).__name__})")
                continue
            self.resolved_block_path = ".".join(path)
            return blocks
        raise RuntimeError(
            "Unable to resolve transformer blocks. "
            f"Inspected paths: {', '.join(failures)}. Pass block_path explicitly."
        )

    @staticmethod
    def _extract_block_tensor(block_output: Any) -> Any:
        tensor = block_output[0] if isinstance(block_output, (tuple, list)) else block_output
        if not hasattr(tensor, "shape") or len(tensor.shape) != 3:
            shape = getattr(tensor, "shape", None)
            raise RuntimeError(f"Expected block output [batch, seq, hidden], got {shape}.")
        return tensor

    @staticmethod
    def _replace_block_tensor(original_output: Any, replacement: Any) -> Any:
        if isinstance(original_output, tuple):
            return (replacement, *original_output[1:])
        if isinstance(original_output, list):
            return [replacement, *original_output[1:]]
        return replacement

    def _format_episode(self, episode_text: str, episode: MatchedEpisode) -> str:
        formatted = format_model_prompt(
            self.tokenizer,
            episode_text,
            prompt_format=episode.prompt_format,
            system_prompt=episode.system_prompt,
            enable_thinking=self.enable_thinking,
        )
        if not isinstance(formatted, str) or not formatted:
            raise ValueError("Tokenizer chat formatting must return a non-empty string.")
        return formatted

    def _boundary_char_span(
        self,
        formatted_prompt: str,
        induction_prompt: str,
        *,
        prompt_format: str,
    ) -> tuple[int, int]:
        if prompt_format == "completion":
            start = 0
        else:
            first = formatted_prompt.find(induction_prompt)
            last = formatted_prompt.rfind(induction_prompt)
            if first < 0 or first != last:
                raise ValueError(
                    "Could not locate the induction text exactly once in the formatted chat prompt; "
                    "refusing an ambiguous patch boundary."
                )
            start = first
        return start, start + len(induction_prompt)

    def _find_boundary_position(
        self,
        formatted_prompt: str,
        induction_prompt: str,
        *,
        prompt_format: str,
        attention_mask: Any,
        offset_mapping: Any,
        row_index: int,
    ) -> tuple[int, int, int]:
        char_start, char_end = self._boundary_char_span(
            formatted_prompt,
            induction_prompt,
            prompt_format=prompt_format,
        )
        mask_row = _row_value(attention_mask, row_index)
        candidates: list[int] = []
        for position, active in enumerate(mask_row):
            if int(active) == 0:
                continue
            token_start, token_end = _offset_pair(offset_mapping, row_index, position)
            if token_end > token_start and token_start >= char_start and token_end <= char_end:
                candidates.append(position)
        if not candidates:
            raise ValueError(
                "No complete token lies inside the induction prefix. Refusing to patch an "
                "ambiguous or tokenizer-merged boundary; choose a safer boundary separator."
            )
        return candidates[-1], char_start, char_end

    def _encode_episodes(
        self,
        episode: MatchedEpisode,
        *,
        max_length: int,
    ) -> tuple[_EncodedEpisode, _EncodedEpisode, dict[str, Any]]:
        positive_raw = compose_matched_episode(
            episode.positive_induction_prompt,
            episode.downstream_prompt,
            boundary_separator=episode.boundary_separator,
        )
        negative_raw = compose_matched_episode(
            episode.negative_induction_prompt,
            episode.downstream_prompt,
            boundary_separator=episode.boundary_separator,
        )
        formatted = [
            self._format_episode(positive_raw, episode),
            self._format_episode(negative_raw, episode),
        ]
        prompt_ids = [
            episode.resolved_positive_source_prompt_id,
            episode.resolved_negative_source_prompt_id,
        ]
        report = inspect_token_lengths(
            self.tokenizer,
            formatted,
            max_length=max_length,
            prompt_ids=prompt_ids,
            tokenizer_id=self.tokenizer_id or None,
            revision=self.revision,
        )
        enforce_token_length_limit(report)
        encoded = tokenize_padded_without_truncation(
            self.tokenizer,
            formatted,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping", None)
        if offset_mapping is None:
            raise ValueError(
                "The tokenizer did not return offset_mapping; a fixed residual patch boundary "
                "cannot be verified without offsets."
            )
        validate_padded_encoding_lengths(encoded, report)
        attention_mask = encoded["attention_mask"]
        positive_source_position, positive_start, positive_end = self._find_boundary_position(
            formatted[0],
            episode.positive_induction_prompt,
            prompt_format=episode.prompt_format,
            attention_mask=attention_mask,
            offset_mapping=offset_mapping,
            row_index=0,
        )
        negative_source_position, negative_start, negative_end = self._find_boundary_position(
            formatted[1],
            episode.negative_induction_prompt,
            prompt_format=episode.prompt_format,
            attention_mask=attention_mask,
            offset_mapping=offset_mapping,
            row_index=1,
        )
        model_encoded = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        positive_generation_encoded, positive_position = self._encode_single_generation_episode(
            formatted[0],
            episode.positive_induction_prompt,
            episode,
            prompt_id=prompt_ids[0],
            max_length=max_length,
        )
        negative_generation_encoded, negative_position = self._encode_single_generation_episode(
            formatted[1],
            episode.negative_induction_prompt,
            episode,
            prompt_id=prompt_ids[1],
            max_length=max_length,
        )
        positive = _EncodedEpisode(
            prompt_id=prompt_ids[0],
            formatted_prompt=formatted[0],
            encoded=positive_generation_encoded,
            source_encoded={key: value[0:1] for key, value in model_encoded.items()},
            boundary_position=positive_position,
            source_boundary_position=positive_source_position,
            boundary_char_start=positive_start,
            boundary_char_end=positive_end,
            token_length=int(report.lengths[0]),
        )
        negative = _EncodedEpisode(
            prompt_id=prompt_ids[1],
            formatted_prompt=formatted[1],
            encoded=negative_generation_encoded,
            source_encoded={key: value[1:2] for key, value in model_encoded.items()},
            boundary_position=negative_position,
            source_boundary_position=negative_source_position,
            boundary_char_start=negative_start,
            boundary_char_end=negative_end,
            token_length=int(report.lengths[1]),
        )
        tokenization = report.to_mapping()
        tokenization["boundary_positions"] = {
            "positive": positive_position,
            "negative": negative_position,
        }
        tokenization["source_batch_boundary_positions"] = {
            "positive": positive_source_position,
            "negative": negative_source_position,
        }
        tokenization["boundary_char_spans"] = {
            "positive": [positive_start, positive_end],
            "negative": [negative_start, negative_end],
        }
        return positive, negative, tokenization

    def _encode_single_generation_episode(
        self,
        formatted_prompt: str,
        induction_prompt: str,
        episode: MatchedEpisode,
        *,
        prompt_id: str,
        max_length: int,
    ) -> tuple[dict[str, Any], int]:
        """Encode one unpadded prompt for baseline and patched generation."""

        report = inspect_token_lengths(
            self.tokenizer,
            [formatted_prompt],
            max_length=max_length,
            prompt_ids=[prompt_id],
            tokenizer_id=self.tokenizer_id or None,
            revision=self.revision,
        )
        enforce_token_length_limit(report)
        encoded = tokenize_padded_without_truncation(
            self.tokenizer,
            [formatted_prompt],
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping", None)
        if offset_mapping is None:
            raise ValueError("The tokenizer did not return offsets for the unpadded generation prompt.")
        validate_padded_encoding_lengths(encoded, report)
        position, _start, _end = self._find_boundary_position(
            formatted_prompt,
            induction_prompt,
            prompt_format=episode.prompt_format,
            attention_mask=encoded["attention_mask"],
            offset_mapping=offset_mapping,
            row_index=0,
        )
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }, position

    def _capture_source_states(
        self,
        positive: _EncodedEpisode,
        negative: _EncodedEpisode,
        layers: Sequence[int],
    ) -> dict[int, Any]:
        blocks = self.resolve_transformer_blocks()
        requested = sorted(set(int(layer) for layer in layers))
        if not requested or requested[0] < 1:
            raise ValueError("layers must contain positive 1-based layer numbers.")
        if requested[-1] > len(blocks):
            raise ValueError(f"Requested layer {requested[-1]}, but model has {len(blocks)} blocks.")
        encoded = {
            key: self._torch.cat([positive.source_encoded[key], negative.source_encoded[key]], dim=0)
            for key in positive.source_encoded
        }
        boundary_positions = [positive.source_boundary_position, negative.source_boundary_position]
        captured: dict[int, Any] = {}
        handles = []

        def make_hook(layer_number: int):
            def _hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = self._extract_block_tensor(output)
                if any(position >= int(tensor.shape[1]) for position in boundary_positions):
                    raise RuntimeError("A source boundary position is outside the model sequence.")
                captured[layer_number] = tensor[
                    [0, 1],
                    boundary_positions,
                    :,
                ].detach().to(dtype=self._torch.float32, device="cpu")
                if layer_number == requested[-1]:
                    raise _EarlyStopForward()

            return _hook

        for layer in requested:
            handles.append(blocks[layer - 1].register_forward_hook(make_hook(layer)))
        try:
            with self._torch.inference_mode():
                try:
                    self.model(
                        **encoded,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )
                except _EarlyStopForward:
                    pass
        finally:
            for handle in handles:
                handle.remove()
        missing = [layer for layer in requested if layer not in captured]
        if missing:
            raise RuntimeError(f"Source capture missed requested layers: {missing}.")
        return captured

    def _generate(
        self,
        encoded: Mapping[str, Any],
        *,
        max_new_tokens: int,
        min_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            raise ValueError("min_new_tokens must satisfy 0 <= min_new_tokens <= max_new_tokens.")
        input_length = int(encoded["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
        }
        pad_token_id = component_attribute(self.tokenizer, "pad_token_id")
        eos_token_id = component_attribute(self.tokenizer, "eos_token_id")
        if pad_token_id is not None or eos_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id if pad_token_id is not None else eos_token_id
        if do_sample:
            if temperature <= 0:
                raise ValueError("temperature must be positive when do_sample=True.")
            generation_kwargs["temperature"] = temperature
        with self._torch.inference_mode():
            output = self.model.generate(**generation_kwargs)
        sequences = getattr(output, "sequences", output)
        if hasattr(sequences, "ndim") and sequences.ndim != 2:
            raise RuntimeError(f"Expected generated token IDs [batch, sequence], got {sequences.shape}.")
        row = sequences[0, input_length:]
        return decode_tokens(self.tokenizer, row, skip_special_tokens=True).strip()

    def _generate_with_patch(
        self,
        receiver: _EncodedEpisode,
        donor_state: Any,
        *,
        layer: int,
        variant: str,
        max_new_tokens: int,
        min_new_tokens: int,
        do_sample: bool,
        temperature: float,
        random_seed: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if variant not in INTERCHANGE_VARIANTS:
            supported = ", ".join(sorted(INTERCHANGE_VARIANTS))
            raise ValueError(f"variant must be one of: {supported}.")
        blocks = self.resolve_transformer_blocks()
        if layer < 1 or layer > len(blocks):
            raise ValueError(f"Requested layer {layer}, but model has {len(blocks)} blocks.")
        donor = donor_state.to(device=self.device, dtype=self._torch.float32)
        if donor.ndim != 1:
            raise ValueError(f"donor_state must have shape [hidden], got {tuple(donor.shape)}.")
        calls = 0
        patch_applied = False
        patch_metrics: dict[str, Any] = {
            "patch_applied": False,
            "receiver_pre_norm": None,
            "donor_norm": float(self._torch.linalg.vector_norm(donor).item()),
            "replacement_delta_l2": None,
            "receiver_post_norm": None,
        }

        def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
            nonlocal calls, patch_applied
            calls += 1
            tensor = self._extract_block_tensor(output)
            if calls == 1:
                position = receiver.boundary_position
                if position >= int(tensor.shape[1]):
                    raise RuntimeError("Receiver boundary position is outside the model sequence.")
                replacement = tensor.clone()
                before = tensor[0, position, :].detach()
                donor_cast = donor.to(dtype=tensor.dtype)
                if variant == "natural_state_replacement":
                    after = donor_cast
                else:
                    # This is algebraically equivalent to replacement at one
                    # token, but the recorded variant makes the intervention
                    # contract explicit for later comparisons with additive
                    # steering and component patching.
                    after = before + (donor_cast - before)
                replacement[0, position, :] = after
                patch_metrics["receiver_pre_norm"] = float(
                    self._torch.linalg.vector_norm(before.float()).item()
                )
                patch_metrics["replacement_delta_l2"] = float(
                    self._torch.linalg.vector_norm((after - before).float()).item()
                )
                patch_metrics["receiver_post_norm"] = float(
                    self._torch.linalg.vector_norm(after.float()).item()
                )
                patch_applied = True
                return self._replace_block_tensor(output, replacement)
            return output

        handle = blocks[layer - 1].register_forward_hook(_hook)
        try:
            text = self._generate(
                receiver.encoded,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        finally:
            handle.remove()
        if not patch_applied:
            raise RuntimeError("The residual patch hook did not run during prompt prefill.")
        patch_metrics["forward_calls"] = calls
        patch_metrics["random_seed"] = random_seed
        return text, patch_metrics

    def run_episode(
        self,
        episode: MatchedEpisode,
        *,
        layers: Sequence[int],
        max_length: int = 1024,
        max_new_tokens: int = 16,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 0.0,
        intervention_variant: str = "natural_state_replacement",
        include_same_condition_controls: bool = True,
    ) -> dict[str, Any]:
        """Run baselines and bidirectional boundary swaps for one episode.

        The two baseline generations and all patch generations use the same
        downstream prompt.  Patching is performed only on the first (prefill)
        call at the receiver's own boundary position; no activation is added
        during answer generation.
        """

        if episode.intervention_timing != "prefill_only":
            raise ValueError("C1 residual interchange currently supports prefill_only only.")
        requested_layers = sorted(set(int(layer) for layer in layers))
        if not requested_layers or requested_layers[0] < 1:
            raise ValueError("layers must contain positive 1-based layer numbers.")
        if intervention_variant not in INTERCHANGE_VARIANTS:
            supported = ", ".join(sorted(INTERCHANGE_VARIANTS))
            raise ValueError(f"intervention_variant must be one of: {supported}.")
        positive, negative, tokenization = self._encode_episodes(episode, max_length=max_length)
        source_states = self._capture_source_states(positive, negative, requested_layers)
        positive_baseline = self._generate(
            positive.encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        negative_baseline = self._generate(
            negative.encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        observations: list[InterchangeObservation] = []
        directions = [
            (
                "positive_to_negative",
                episode.positive_condition,
                episode.negative_condition,
                positive,
                negative,
                episode.resolved_positive_source_prompt_id,
                episode.resolved_negative_source_prompt_id,
                negative_baseline,
                0,
                1,
            ),
            (
                "negative_to_positive",
                episode.negative_condition,
                episode.positive_condition,
                negative,
                positive,
                episode.resolved_negative_source_prompt_id,
                episode.resolved_positive_source_prompt_id,
                positive_baseline,
                1,
                0,
            ),
        ]
        if include_same_condition_controls:
            directions.extend(
                [
                    (
                        "positive_to_positive",
                        episode.positive_condition,
                        episode.positive_condition,
                        positive,
                        positive,
                        episode.resolved_positive_source_prompt_id,
                        episode.resolved_positive_source_prompt_id,
                        positive_baseline,
                        0,
                        0,
                    ),
                    (
                        "negative_to_negative",
                        episode.negative_condition,
                        episode.negative_condition,
                        negative,
                        negative,
                        episode.resolved_negative_source_prompt_id,
                        episode.resolved_negative_source_prompt_id,
                        negative_baseline,
                        1,
                        1,
                    ),
                ]
            )

        for layer in requested_layers:
            for (
                direction,
                donor_condition,
                receiver_condition,
                donor_episode,
                receiver_episode,
                donor_prompt_id,
                receiver_prompt_id,
                receiver_baseline,
                donor_index,
                receiver_index,
            ) in directions:
                patched, metrics = self._generate_with_patch(
                    receiver_episode,
                    source_states[layer][donor_index],
                    layer=layer,
                    variant=intervention_variant,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                observations.append(
                    InterchangeObservation(
                        request_id=episode.request_id,
                        construct_id=episode.construct_id,
                        layer=layer,
                        patch_direction=direction,
                        donor_condition=donor_condition,
                        receiver_condition=receiver_condition,
                        donor_prompt_id=donor_prompt_id,
                        receiver_prompt_id=receiver_prompt_id,
                        downstream_prompt_id=episode.resolved_downstream_prompt_id,
                        intervention_variant=intervention_variant,
                        intervention_timing=episode.intervention_timing,
                        boundary_mode=episode.boundary_mode,
                        donor_boundary_position=(
                            positive.boundary_position if donor_index == 0 else negative.boundary_position
                        ),
                        receiver_boundary_position=(
                            positive.boundary_position if receiver_index == 0 else negative.boundary_position
                        ),
                        donor_token_length=(positive.token_length if donor_index == 0 else negative.token_length),
                        receiver_token_length=(
                            positive.token_length if receiver_index == 0 else negative.token_length
                        ),
                        receiver_baseline_output=receiver_baseline,
                        patched_output=patched,
                        patch_applied=bool(metrics["patch_applied"] if "patch_applied" in metrics else True),
                        forward_calls=int(metrics["forward_calls"]),
                        receiver_pre_norm=float(metrics["receiver_pre_norm"]),
                        donor_norm=float(metrics["donor_norm"]),
                        replacement_delta_l2=float(metrics["replacement_delta_l2"]),
                        receiver_post_norm=float(metrics["receiver_post_norm"]),
                    )
                )

        return {
            "schema_version": CAUSAL_PATCH_SCHEMA_VERSION,
            "request": episode.to_mapping(),
            "model": {
                "model_id": self.model_id,
                "tokenizer_id": self.tokenizer_id,
                "revision": self.revision,
                "device": str(self.device),
                "block_path": self.resolved_block_path,
            },
            "prompt_fingerprints": {
                "positive_episode_sha256": _sha256_text(positive.formatted_prompt),
                "negative_episode_sha256": _sha256_text(negative.formatted_prompt),
                "downstream_prompt_sha256": _sha256_text(episode.downstream_prompt),
                "downstream_prompt_identical": True,
            },
            "tokenization": tokenization,
            "baselines": {
                "positive": {
                    "prompt_id": positive.prompt_id,
                    "boundary_position": positive.boundary_position,
                    "token_length": positive.token_length,
                    "output": positive_baseline,
                },
                "negative": {
                    "prompt_id": negative.prompt_id,
                    "boundary_position": negative.boundary_position,
                    "token_length": negative.token_length,
                    "output": negative_baseline,
                },
            },
            "intervention": {
                "layers": requested_layers,
                "boundary_mode": episode.boundary_mode,
                "intervention_timing": episode.intervention_timing,
                "variant": intervention_variant,
                "include_same_condition_controls": include_same_condition_controls,
            },
            "observations": [observation.to_mapping() for observation in observations],
        }


def expected_observation_ids(
    request_ids: Sequence[str],
    layers: Sequence[int],
    *,
    include_same_condition_controls: bool = True,
) -> list[str]:
    """Return the manifest identity set for a causal-interchange run."""

    directions = ["positive_to_negative", "negative_to_positive"]
    if include_same_condition_controls:
        directions.extend(["positive_to_positive", "negative_to_negative"])
    return [
        f"{request_id}__{direction}__layer_{int(layer):02d}"
        for request_id in request_ids
        for direction in directions
        for layer in sorted(set(int(value) for value in layers))
    ]


def load_matched_episode_jsonl(path: Path) -> list[MatchedEpisode]:
    """Load and validate one frozen matched-episode JSONL inventory."""

    if not path.is_file():
        raise ValueError(f"Matched-episode inventory does not exist: {path}")
    episodes: list[MatchedEpisode] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on matched-episode line {line_number}.") from exc
        episode = MatchedEpisode.from_mapping(raw)
        if episode.request_id in seen:
            raise ValueError(f"Duplicate matched-episode request_id={episode.request_id!r}.")
        seen.add(episode.request_id)
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"Matched-episode inventory is empty: {path}")
    return episodes


def residual_interchange_manifest_path(output: Path) -> Path:
    """Return the adjacent manifest path for a residual-interchange output."""

    return output.with_suffix(output.suffix + ".manifest.json")


def validate_residual_interchange_output(
    output: Path,
    *,
    allow_incomplete_diagnostic: bool = False,
) -> dict[str, Any]:
    """Validate a causal output and refuse truncated runs by default.

    This validator is intentionally independent of the model runner.  A
    downstream scorer can therefore fail closed before treating a partial
    output as causal evidence.
    """

    manifest_path = residual_interchange_manifest_path(output)
    if not output.is_file():
        raise ValueError(f"Residual-interchange output does not exist: {output}")
    if not manifest_path.is_file():
        raise ValueError(f"Cannot validate {output} without adjacent manifest {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path} is not valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object.")
    required = {
        "schema_version",
        "manifest_type",
        "request_ids",
        "layers",
        "completed_request_ids",
        "expected_record_ids",
        "expected_request_count",
        "expected_observation_count",
        "completed_request_count",
        "completed_observation_count",
        "complete",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"{manifest_path} is missing required fields: {missing}.")
    if manifest["schema_version"] != CAUSAL_PATCH_SCHEMA_VERSION:
        raise ValueError(f"Unsupported residual-interchange schema version: {manifest['schema_version']!r}.")
    if manifest["manifest_type"] != "residual_interchange_output":
        raise ValueError(f"{manifest_path} is not a residual-interchange output manifest.")
    request_ids = manifest["request_ids"]
    record_ids = manifest["expected_record_ids"]
    if (
        not isinstance(request_ids, list)
        or not request_ids
        or any(not isinstance(value, str) or not value for value in request_ids)
        or len(set(request_ids)) != len(request_ids)
    ):
        raise ValueError(f"{manifest_path} request_ids are invalid.")
    if (
        not isinstance(record_ids, list)
        or not record_ids
        or any(not isinstance(value, str) or not value for value in record_ids)
        or len(set(record_ids)) != len(record_ids)
    ):
        raise ValueError(f"{manifest_path} expected_record_ids are invalid.")
    expected_records = set(record_ids)
    if manifest["expected_request_count"] != len(request_ids):
        raise ValueError(f"{manifest_path} expected_request_count is inconsistent.")
    if manifest["expected_observation_count"] != len(expected_records):
        raise ValueError(f"{manifest_path} expected_observation_count is inconsistent.")
    layers = manifest["layers"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError(f"{manifest_path} layers are invalid.")
    execution = manifest.get("execution", {})
    include_same = bool(execution.get("include_same_condition_controls", True))
    expected_from_matrix = set(
        expected_observation_ids(
            request_ids,
            layers,
            include_same_condition_controls=include_same,
        )
    )
    if expected_records != expected_from_matrix:
        raise ValueError(f"{manifest_path} expected_record_ids do not match the request/layer matrix.")
    completed_request_ids = manifest["completed_request_ids"]
    if (
        not isinstance(completed_request_ids, list)
        or any(value not in request_ids for value in completed_request_ids)
        or len(set(completed_request_ids)) != len(completed_request_ids)
    ):
        raise ValueError(f"{manifest_path} completed_request_ids are invalid.")
    complete = manifest["complete"]
    if not isinstance(complete, bool):
        raise ValueError(f"{manifest_path} complete must be boolean.")
    if not complete and not allow_incomplete_diagnostic:
        raise ValueError(
            f"Refusing to score incomplete residual-interchange output {output}; "
            "pass allow_incomplete_diagnostic=True for engineering inspection only."
        )
    seen_requests: set[str] = set()
    seen_records: set[str] = set()
    for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{output} has invalid JSON on line {line_number}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{output} line {line_number} must be a JSON object.")
        request = row.get("request")
        request_id = request.get("request_id") if isinstance(request, dict) else None
        if request_id not in request_ids or request_id in seen_requests:
            raise ValueError(f"{output} line {line_number} has an invalid or duplicate request_id.")
        observations = row.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"{output} line {line_number} has no observations list.")
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError(f"{output} line {line_number} has a malformed observation.")
            record_id = observation.get("record_id")
            if record_id not in expected_records or record_id in seen_records:
                raise ValueError(f"{output} line {line_number} has an invalid or duplicate record_id.")
            if observation.get("request_id") != request_id:
                raise ValueError(f"Observation {record_id!r} has the wrong request_id.")
            seen_records.add(record_id)
        seen_requests.add(request_id)
    if manifest["completed_request_count"] != len(seen_requests):
        raise ValueError(f"{manifest_path} completed_request_count disagrees with the output.")
    if set(completed_request_ids) != seen_requests:
        raise ValueError(f"{manifest_path} completed_request_ids disagree with the output.")
    if manifest["completed_observation_count"] != len(seen_records):
        raise ValueError(f"{manifest_path} completed_observation_count disagrees with the output.")
    if complete:
        if seen_requests != set(request_ids) or seen_records != expected_records:
            raise ValueError(f"{output} is marked complete but is missing expected records.")
        raw_hash = manifest.get("raw_output_sha256")
        if not raw_hash or _sha256_file(output) != raw_hash:
            raise ValueError(f"{output} does not match raw_output_sha256 in its manifest.")
    return manifest


__all__ = [
    "BOUNDARY_MODES",
    "CAUSAL_PATCH_SCHEMA_VERSION",
    "INTERCHANGE_TIMINGS",
    "INTERCHANGE_VARIANTS",
    "InterchangeObservation",
    "MatchedEpisode",
    "MatchedEpisodeResidualPatcher",
    "compose_matched_episode",
    "expected_observation_ids",
    "load_matched_episode_jsonl",
    "residual_interchange_manifest_path",
    "validate_residual_interchange_output",
]
