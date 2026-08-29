from __future__ import annotations

from types import SimpleNamespace

import pytest

from activation_analysis.model_loading import (
    component_attribute,
    decode_tokens,
    load_model,
    load_tokenizer_or_processor,
    move_batch_to_device,
)


class _Loader:
    def __init__(self, value=None, error: Exception | None = None, calls: list[str] | None = None) -> None:
        self.value = value
        self.error = error
        self.calls = calls if calls is not None else []

    def from_pretrained(self, identifier, **kwargs):
        del kwargs
        self.calls.append(str(identifier))
        if self.error is not None:
            raise self.error
        return self.value


def test_processor_fallback_returns_nested_text_tokenizer() -> None:
    nested = SimpleNamespace(
        pad_token_id=None,
        eos_token_id=9,
        eos_token="<eos>",
        decode=lambda values, skip_special_tokens: f"{values}:{skip_special_tokens}",
    )
    processor = SimpleNamespace(tokenizer=nested)
    processor_loader = _Loader(processor)
    tokenizer_loader = _Loader(error=RuntimeError("text tokenizer unavailable"))
    transformers = SimpleNamespace(
        AutoProcessor=processor_loader,
        AutoTokenizer=tokenizer_loader,
    )

    loaded, loader_name = load_tokenizer_or_processor(
        transformers,
        "Qwen/Qwen3.8-27B",
        revision="rev",
        local_files_only=True,
        trust_remote_code=False,
    )

    assert loaded is nested
    assert loader_name == "AutoProcessor.tokenizer"
    assert processor_loader.calls == ["Qwen/Qwen3.8-27B"]
    assert tokenizer_loader.calls == ["Qwen/Qwen3.8-27B"]
    assert component_attribute(loaded, "pad_token_id") == 9
    assert nested.pad_token == "<eos>"
    assert decode_tokens(loaded, [1, 2]) == "[1, 2]:True"


def test_tokenizer_loader_falls_back_to_processor_for_other_models() -> None:
    processor = SimpleNamespace()
    tokenizer_loader = _Loader(error=RuntimeError("not a tokenizer"))
    processor_loader = _Loader(processor)
    transformers = SimpleNamespace(
        AutoTokenizer=tokenizer_loader,
        AutoProcessor=processor_loader,
    )

    loaded, loader_name = load_tokenizer_or_processor(
        transformers,
        "some/multimodal-model",
        revision=None,
        local_files_only=False,
        trust_remote_code=True,
    )

    assert loaded is processor
    assert loader_name == "AutoProcessor"
    assert tokenizer_loader.calls == ["some/multimodal-model"]
    assert processor_loader.calls == ["some/multimodal-model"]


def test_mistral_loader_requests_corrected_tokenizer_regex() -> None:
    class _KwargLoader:
        def __init__(self) -> None:
            self.kwargs = None

        def from_pretrained(self, identifier, **kwargs):
            del identifier
            self.kwargs = kwargs
            return SimpleNamespace(pad_token_id=0)

    loader = _KwargLoader()
    transformers = SimpleNamespace(AutoTokenizer=loader)

    load_tokenizer_or_processor(
        transformers,
        "mistralai/Mistral-Small-24B-Instruct-2501",
        revision="rev",
        local_files_only=True,
        trust_remote_code=False,
    )

    assert loader.kwargs is not None
    assert loader.kwargs["fix_mistral_regex"] is True


def test_model_loader_tries_multimodal_fallbacks_in_order() -> None:
    loaded_model = object()
    causal_loader = _Loader(error=RuntimeError("causal class mismatch"))
    image_loader = _Loader(error=RuntimeError("image class mismatch"))
    multimodal_loader = _Loader(value=loaded_model)
    transformers = SimpleNamespace(
        AutoModelForCausalLM=causal_loader,
        AutoModelForImageTextToText=image_loader,
        AutoModelForMultimodalLM=multimodal_loader,
    )

    assert load_model(transformers, "Qwen/Qwen3.8-27B", {"revision": "rev"}) is loaded_model
    assert causal_loader.calls == ["Qwen/Qwen3.8-27B"]
    assert image_loader.calls == ["Qwen/Qwen3.8-27B"]
    assert multimodal_loader.calls == ["Qwen/Qwen3.8-27B"]


def test_move_batch_preserves_non_tensor_values() -> None:
    class _Tensor:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def to(self, device: str):
            self.devices.append(device)
            return self

    tensor = _Tensor()
    metadata = "text-only"
    moved = move_batch_to_device({"input_ids": tensor, "metadata": metadata}, "cuda")

    assert moved["input_ids"] is tensor
    assert tensor.devices == ["cuda"]
    assert moved["metadata"] == metadata


def test_model_loader_reports_all_fallback_errors() -> None:
    transformers = SimpleNamespace(
        AutoModelForCausalLM=_Loader(error=ValueError("first")),
        AutoModel=SimpleNamespace(from_pretrained=lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("last"))),
    )
    with pytest.raises(RuntimeError, match="Unable to load model") as error:
        load_model(transformers, "broken/model", {})
    assert "first" in str(error.value)
    assert "last" in str(error.value)
