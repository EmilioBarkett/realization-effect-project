from __future__ import annotations

import json
import urllib.request

import pytest

from activation_analysis.openai_prompt_generation import (
    DEFAULT_OPENAI_MODEL,
    build_openai_responses_body,
    call_openai_responses,
)


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tiny_generation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompts"],
            "properties": {"prompts": {"type": "array"}},
        },
    },
}


def test_build_openai_responses_body_translates_schema_and_omits_seed() -> None:
    body = build_openai_responses_body(
        "anthropic/claude-sonnet-4.6",
        [{"role": "system", "content": "Return JSON."}, {"role": "user", "content": "Hi"}],
        {
            "response_schema": SCHEMA,
            "reasoning_effort": "xhigh",
            "max_tokens": 321,
            "temperature": 0.7,
            "seed": 44,
        },
    )

    assert body["model"] == DEFAULT_OPENAI_MODEL
    assert body["input"][1]["content"] == "Hi"
    assert body["store"] is False
    assert body["reasoning"] == {"effort": "xhigh"}
    assert body["max_output_tokens"] == 321
    assert "temperature" not in body
    assert "top_p" not in body
    assert "seed" not in body
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "tiny_generation",
        "strict": True,
        "schema": SCHEMA["json_schema"]["schema"],
    }


def test_call_openai_responses_normalizes_response_and_records_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "resp_test_123",
                    "model": DEFAULT_OPENAI_MODEL,
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "summary": []},
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": '{"prompts": []}'}],
                        },
                    ],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 25,
                        "total_tokens": 125,
                        "output_tokens_details": {"reasoning_tokens": 9},
                    },
                }
            ).encode("utf-8")

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = call_openai_responses(
        "plan-model-id",
        [{"role": "user", "content": "Generate one item."}],
        {
            "response_schema": SCHEMA,
            "reasoning_effort": "xhigh",
            "max_output_tokens": 500,
            "timeout": 17,
            "seed": 99,
            "input_usd_per_million_tokens": 0.2,
            "output_usd_per_million_tokens": 1.2,
        },
    )

    request = captured["request"]
    assert isinstance(request, urllib.request.Request)
    assert request.get_header("Authorization") == "Bearer test-openai-key"
    assert captured["timeout"] == 17.0
    assert result["choices"][0]["message"]["content"] == '{"prompts": []}'
    assert result["usage"]["total_tokens"] == 125
    metadata = result["_generation_metadata"]
    assert metadata["provider"] == "openai"
    assert metadata["api"] == "responses"
    assert metadata["response_id"] == "resp_test_123"
    assert metadata["usage"] == {
        "input_tokens": 100,
        "output_tokens": 25,
        "total_tokens": 125,
        "output_tokens_details": {"reasoning_tokens": 9},
    }
    assert metadata["input_tokens"] == 100
    assert metadata["output_tokens"] == 25
    assert metadata["total_tokens"] == 125
    assert metadata["reasoning_effort"] == "xhigh"
    assert metadata["seed_supported"] is False
    assert metadata["requested_seed"] == 99
    assert metadata["store"] is False
    assert metadata["actual_cost_usd"] == pytest.approx(0.00005)


def test_call_openai_responses_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Set OPENAI_API_KEY"):
        call_openai_responses("ignored", [], {"response_schema": SCHEMA})
