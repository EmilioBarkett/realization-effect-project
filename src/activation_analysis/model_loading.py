"""Shared Transformers loading helpers for text and multimodal causal models.

The benchmark is text-only, but some current open models expose their language
model through ``AutoProcessor`` and ``AutoModelForMultimodalLM``. Keeping the
fallbacks here makes tokenizer preflight, residual logging, and steering use
the same component and avoids changing the tokenization contract between
stages.
When a processor wraps a text tokenizer, text-only runs deliberately use that
nested tokenizer so a processor cannot reinterpret ordinary prompt strings as
multimodal content.
"""

from __future__ import annotations

from typing import Any


def _nested_component(component: Any) -> Any | None:
    """Return a processor's underlying tokenizer when one is exposed."""

    nested = getattr(component, "tokenizer", None)
    return nested if nested is not None and nested is not component else None


def component_attribute(component: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from a tokenizer or its underlying processor tokenizer."""

    value = getattr(component, name, default)
    if value is not None:
        return value
    nested = _nested_component(component)
    return getattr(nested, name, default) if nested is not None else default


def ensure_padding_token(component: Any) -> None:
    """Ensure generation has a padding ID without mutating a processor unnecessarily."""

    if component_attribute(component, "pad_token_id") is not None:
        return
    eos_token_id = component_attribute(component, "eos_token_id")
    eos_token = component_attribute(component, "eos_token")
    if eos_token_id is None and eos_token is None:
        return
    target = _nested_component(component) or component
    if eos_token is not None:
        try:
            target.pad_token = eos_token
        except (AttributeError, TypeError):
            pass
    if component_attribute(component, "pad_token_id") is None and eos_token_id is not None:
        try:
            target.pad_token_id = eos_token_id
        except (AttributeError, TypeError):
            pass


def load_tokenizer_or_processor(
    transformers: Any,
    identifier: str,
    *,
    revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> tuple[Any, str]:
    """Load the exact text-processing component used by model execution.

    Qwen3.8 is processor-capable, but this benchmark's frozen prompts are
    text-only. Prefer a plain tokenizer when the repository provides one, and
    retain a processor fallback for repositories that expose only a processor.
    If that fallback exposes a nested text tokenizer, return the nested
    tokenizer rather than the multimodal wrapper. This keeps chat formatting,
    token-length checks, model inputs, and decoding on one text-only contract.
    """

    kwargs = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    # Mistral's current tokenizer metadata carries a legacy regex.  Passing
    # this flag makes the tokenizer used for length checks and model execution
    # agree with the corrected reference implementation.
    if "mistral" in identifier.lower():
        kwargs["fix_mistral_regex"] = True
    loader_names = ("AutoTokenizer", "AutoProcessor")
    errors: list[Exception] = []
    for loader_name in loader_names:
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            component = loader.from_pretrained(identifier, **kwargs)
            if loader_name == "AutoProcessor":
                nested = _nested_component(component)
                if nested is not None:
                    component = nested
                    loader_name = "AutoProcessor.tokenizer"
            ensure_padding_token(component)
            return component, loader_name
        except Exception as exc:
            errors.append(exc)
    details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors[-3:])
    raise RuntimeError(
        f"Unable to load tokenizer or processor '{identifier}'. "
        f"Tried {', '.join(loader_names)}. {details}"
    )


def load_model(transformers: Any, model_id: str, model_kwargs: dict[str, Any]) -> Any:
    """Load a causal or multimodal causal model with explicit fallbacks."""

    errors: list[Exception] = []
    for loader_name in (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
        "AutoModelForConditionalGeneration",
        "AutoModel",
    ):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            return loader.from_pretrained(model_id, **model_kwargs)
        except Exception as exc:
            errors.append(exc)
    details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors[-4:])
    raise RuntimeError(f"Unable to load model '{model_id}'. {details}")


def move_batch_to_device(encoded: Any, device: str) -> dict[str, Any]:
    """Move tensor-like model inputs while preserving non-tensor metadata."""

    items = encoded.items() if hasattr(encoded, "items") else []
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in items}


def decode_tokens(component: Any, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
    """Decode generated IDs through a tokenizer or processor tokenizer."""

    decoder = getattr(component, "decode", None)
    if not callable(decoder):
        nested = _nested_component(component)
        decoder = getattr(nested, "decode", None) if nested is not None else None
    if not callable(decoder):
        raise RuntimeError("The loaded tokenizer/processor does not expose decode().")
    return str(decoder(token_ids, skip_special_tokens=skip_special_tokens))


__all__ = [
    "component_attribute",
    "decode_tokens",
    "ensure_padding_token",
    "load_model",
    "load_tokenizer_or_processor",
    "move_batch_to_device",
]
