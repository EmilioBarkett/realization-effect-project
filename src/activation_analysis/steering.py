from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SUPPORTED_POSITION_MODES = {"all", "last"}


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


@dataclass(frozen=True)
class SteeringVectorInfo:
    path: str
    hidden_size: int
    raw_norm: float
    normalized: bool


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
        self.block_path = block_path
        self.resolved_block_path: str | None = None

        self.tokenizer = self._transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        self.model = self._load_model(model_kwargs)
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

    def _load_model(self, model_kwargs: dict[str, Any]) -> Any:
        errors: list[Exception] = []
        for loader_name in (
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModel",
        ):
            loader = getattr(self._transformers, loader_name, None)
            if loader is None:
                continue
            try:
                return loader.from_pretrained(self.model_id, **model_kwargs)
            except Exception as exc:
                errors.append(exc)
        details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors[-3:])
        raise RuntimeError(f"Unable to load model '{self.model_id}'. {details}")

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

    @contextmanager
    def steering_hooks(self, config: SteeringConfig) -> Iterator[SteeringVectorInfo]:
        if config.layer < 1:
            raise ValueError("Layer numbers are 1-based and must be >= 1.")
        blocks = self._resolve_transformer_blocks()
        if config.layer > len(blocks):
            raise ValueError(f"Requested layer {config.layer}, but model has only {len(blocks)} blocks.")
        direction, info = self.load_direction(config)
        handle = blocks[config.layer - 1].register_forward_hook(
            self._make_hook(
                direction=direction,
                scale=config.scale,
                position_mode=config.position_mode,
            )
        )
        try:
            yield info
        finally:
            handle.remove()

    def _make_hook(self, *, direction: Any, scale: float, position_mode: str):
        def _hook(_module, _inputs, output):
            tensor = self._extract_block_tensor(output)
            steered = self._inject(tensor, direction=direction, scale=scale, position_mode=position_mode)
            return self._replace_block_tensor(output, steered)

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

    def format_prompt(self, prompt: str, *, prompt_format: str, system_prompt: str) -> str:
        if prompt_format == "completion":
            return prompt
        if prompt_format == "chat":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError("Tokenizer does not support apply_chat_template; use --prompt-format completion.")
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

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
    ) -> tuple[str, SteeringVectorInfo | None]:
        formatted_prompt = self.format_prompt(
            prompt,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )
        encoded = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        if steering_config is None or steering_config.scale == 0:
            with self._torch.no_grad():
                output = self.model.generate(**generation_kwargs)
            info = None
        else:
            with self.steering_hooks(steering_config) as info:
                with self._torch.no_grad():
                    output = self.model.generate(**generation_kwargs)

        new_tokens = output[0, input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), info
