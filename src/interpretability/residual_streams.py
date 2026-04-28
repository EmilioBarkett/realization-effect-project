from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class _EarlyStopForward(RuntimeError):
    """Internal control-flow exception used to stop after the final requested layer."""


@dataclass(frozen=True)
class BatchResiduals:
    prompt_ids: list[str]
    token_ids: list[list[int]]
    token_positions: list[list[int]]
    hidden_states_by_layer: dict[int, Any]


class ResidualStreamLogger:
    """Capture residual stream tensors from selected transformer blocks.

    This is adapted from the metageniuses forward-pass adapter, with the biological
    sequence assumptions removed so it can operate on text prompts from this project.
    Layer numbers are 1-based transformer block indices.
    """

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
        stop_after_last_requested_layer: bool = True,
    ) -> None:
        self._torch, self._transformers = self._import_dependencies()
        self.model_id = str(model_id)
        self.tokenizer_id = str(tokenizer_id or model_id)
        self.stop_after_last_requested_layer = stop_after_last_requested_layer

        self.tokenizer = self._transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_dtype = self._resolve_dtype(dtype)
        model_kwargs = {
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": model_dtype,
            "low_cpu_mem_usage": True,
        }
        self.model = self._load_model(model_kwargs)
        self.device = self._resolve_device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.num_transformer_layers = int(self._config_value("num_hidden_layers"))
        self.d_model = int(self._config_value("hidden_size"))

    def _import_dependencies(self) -> tuple[Any, Any]:
        try:
            import torch
            import transformers
        except Exception as exc:  # pragma: no cover - environment-specific.
            raise RuntimeError(
                "ResidualStreamLogger requires torch and transformers. "
                "Install them in the active environment before running extraction."
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

    def _resolve_dtype(self, requested: str):
        if requested == "auto":
            if self._torch.cuda.is_available():
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

    def _resolve_transformer_blocks(self) -> list[Any]:
        candidate_paths = [
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
        for path in candidate_paths:
            node = self.model
            try:
                for name in path:
                    node = getattr(node, name)
            except AttributeError:
                continue
            try:
                blocks = list(node)
            except TypeError:
                continue
            if blocks:
                return blocks
        raise RuntimeError(
            "Unable to resolve transformer block modules for this model architecture."
        )

    def _extract_block_tensor(self, block_output: Any):
        if isinstance(block_output, (tuple, list)):
            if not block_output:
                raise RuntimeError("Transformer block output tuple was empty.")
            block_output = block_output[0]
        if not hasattr(block_output, "shape"):
            raise RuntimeError("Transformer block output is not a tensor.")
        if len(block_output.shape) != 3:
            raise RuntimeError(
                "Expected transformer block output shape [batch, seq, hidden], "
                f"got {tuple(block_output.shape)}"
            )
        return block_output

    def extract_batch(
        self,
        prompts: list[str],
        prompt_ids: list[str],
        layers: list[int],
        *,
        max_length: int = 512,
    ) -> BatchResiduals:
        if len(prompts) != len(prompt_ids):
            raise ValueError("prompts and prompt_ids must have the same length.")
        if not layers:
            raise ValueError("At least one layer is required.")

        blocks = self._resolve_transformer_blocks()
        max_layer = max(layers)
        if min(layers) < 1:
            raise ValueError("Layer numbers are 1-based and must be >= 1.")
        if max_layer > len(blocks):
            raise ValueError(
                f"Requested layer {max_layer}, but model has only {len(blocks)} blocks."
            )

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        captured_layers: dict[int, Any] = {}
        hook_handles = []

        def make_hook(layer_number: int):
            def _hook(_module, _inputs, output):
                tensor = self._extract_block_tensor(output)
                captured_layers[layer_number] = tensor.detach().to(
                    dtype=self._torch.float32,
                    device="cpu",
                )
                if self.stop_after_last_requested_layer and layer_number == max_layer:
                    raise _EarlyStopForward()

            return _hook

        for layer in sorted(set(layers)):
            handle = blocks[layer - 1].register_forward_hook(make_hook(layer))
            hook_handles.append(handle)

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
                for handle in hook_handles:
                    handle.remove()

        input_ids = encoded["input_ids"].detach().cpu()
        attention_mask = encoded["attention_mask"].detach().cpu()
        token_ids: list[list[int]] = []
        token_positions: list[list[int]] = []
        for batch_idx in range(input_ids.shape[0]):
            positions = attention_mask[batch_idx].nonzero(as_tuple=False).flatten()
            token_positions.append(positions.tolist())
            token_ids.append(input_ids[batch_idx, positions].tolist())

        missing_layers = [layer for layer in layers if layer not in captured_layers]
        if missing_layers:
            raise RuntimeError(f"Missing captured layers: {missing_layers}")

        return BatchResiduals(
            prompt_ids=prompt_ids,
            token_ids=token_ids,
            token_positions=token_positions,
            hidden_states_by_layer={layer: captured_layers[layer] for layer in layers},
        )
