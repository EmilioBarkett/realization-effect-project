"""OpenAI Responses transport for construct prompt generation.

The benchmark generation code currently parses a small, Chat Completions-shaped
response (``choices[0].message.content``).  This module keeps that interface
stable while speaking the Responses API and translating the repository's
OpenRouter-style JSON-schema option into the Responses ``text.format`` shape.

The transport is intentionally stdlib-only.  It does not import the OpenAI SDK,
which keeps no-API tests and the base development environment independent of a
particular SDK version.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_OUTPUT_TOKENS = 8000
OPENAI_TIMEOUT_OVERRIDE_ENV = "OPENAI_REQUEST_TIMEOUT_SECONDS"


def _default_response_schema() -> dict[str, Any]:
    """Return a permissive JSON-schema envelope for direct transport use.

    The construct generator always supplies its construct-specific schema.  A
    permissive object is still useful for callers using this transport directly
    and keeps the JSON-schema requirement explicit rather than silently falling
    back to free-form text.
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "prompt_generation_response",
            "strict": False,
            "schema": {"type": "object"},
        },
    }


def _responses_text_format(response_schema: Any) -> dict[str, Any]:
    """Convert the repository/OpenRouter schema envelope to Responses format."""

    raw = _default_response_schema() if response_schema is None else response_schema
    if not isinstance(raw, Mapping):
        raise ValueError("response_schema must be a mapping.")
    if raw.get("type") != "json_schema":
        raise ValueError("response_schema must have type='json_schema'.")

    # Existing generation schemas use OpenRouter's ``json_schema`` wrapper;
    # accept the already-unwrapped Responses form as well for direct callers.
    config = raw.get("json_schema", raw)
    if not isinstance(config, Mapping):
        raise ValueError("response_schema.json_schema must be a mapping.")
    name = config.get("name")
    schema = config.get("schema")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("response_schema must define a non-empty schema name.")
    if not isinstance(schema, Mapping):
        raise ValueError("response_schema must define a JSON schema object.")

    result: dict[str, Any] = {
        "type": "json_schema",
        "name": name,
        "schema": dict(schema),
        "strict": bool(config.get("strict", True)),
    }
    if config.get("description") is not None:
        result["description"] = str(config["description"])
    return result


def _reasoning_effort(options: Mapping[str, Any]) -> str:
    value = options.get("reasoning_effort")
    if value is None and isinstance(options.get("reasoning"), Mapping):
        value = options["reasoning"].get("effort")
    return str(value or DEFAULT_REASONING_EFFORT)


def build_openai_responses_body(
    model_id: str,
    messages: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a Responses API request body without performing network I/O.

    ``model_id`` remains part of the RequestFn contract, but the OpenAI
    transport defaults to the explicitly selected Luna model.  Callers can
    override it with ``openai_model`` (or ``model``) when running a controlled
    comparison.  This avoids accidentally sending an OpenRouter model slug to
    the OpenAI endpoint.
    """

    selected_model = str(
        options.get("openai_model")
        or options.get("model")
        or DEFAULT_OPENAI_MODEL
    )
    max_output_tokens = options.get("max_output_tokens", options.get("max_tokens"))
    if max_output_tokens is None:
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS

    body: dict[str, Any] = {
        "model": selected_model,
        "input": [dict(message) for message in messages],
        "store": False,
        "reasoning": {"effort": _reasoning_effort(options)},
        "text": {"format": _responses_text_format(options.get("response_schema"))},
        "max_output_tokens": int(max_output_tokens),
    }

    # GPT-5.6 Luna rejects temperature/top_p while reasoning is enabled.
    # Seeds are likewise unsupported. Preserve those requested values only in
    # provenance metadata; do not send parameters the selected model rejects.
    return body


def _response_text(response: Mapping[str, Any]) -> str:
    """Extract visible assistant text from a Responses API payload."""

    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("OpenAI Responses response did not include output text.")

    parts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping):
            continue
        # A response can include reasoning/tool items before the assistant
        # message.  Only collect visible message content, never reasoning text.
        if item.get("type") not in (None, "message"):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    text = "".join(parts)
    if not text:
        raise ValueError("OpenAI Responses response did not include assistant text.")
    return text


def _usage_summary(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in usage:
            summary[key] = usage[key]
    for key in ("input_tokens_details", "output_tokens_details"):
        if isinstance(usage.get(key), Mapping):
            summary[key] = dict(usage[key])
    return summary


def _estimated_cost_usd(usage: Mapping[str, Any], options: Mapping[str, Any]) -> float | None:
    """Estimate cost only when rates are explicitly provided by the caller."""

    raw_cost = usage.get("cost")
    if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
        return float(raw_cost)
    input_rate = options.get("input_usd_per_million_tokens")
    output_rate = options.get("output_usd_per_million_tokens")
    if not isinstance(input_rate, (int, float)) or isinstance(input_rate, bool):
        return None
    if not isinstance(output_rate, (int, float)) or isinstance(output_rate, bool):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, (int, float)) or not isinstance(output_tokens, (int, float)):
        return None
    return (float(input_tokens) * float(input_rate) + float(output_tokens) * float(output_rate)) / 1_000_000


def _normalise_response(response: Mapping[str, Any], *, model: str, model_id: str, options: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a Responses payload into the repository's parser contract."""

    content = _response_text(response)
    raw_usage = response.get("usage")
    usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
    usage_summary = _usage_summary(usage)
    status = response.get("status")
    incomplete_details = response.get("incomplete_details")
    metadata: dict[str, Any] = {
        "provider": "openai",
        "api": "responses",
        "model": model,
        "requested_model_id": model_id,
        "response_id": response.get("id"),
        "status": status,
        "usage": usage_summary,
        **{key: usage_summary[key] for key in ("input_tokens", "output_tokens", "total_tokens") if key in usage_summary},
        "reasoning_effort": _reasoning_effort(options),
        "seed_supported": False,
        "requested_seed": options.get("seed"),
        "store": False,
    }
    if isinstance(incomplete_details, Mapping):
        metadata["incomplete_details"] = dict(incomplete_details)
        metadata["incomplete_reason"] = incomplete_details.get("reason")
    if isinstance(status, str) and status.strip().lower() in {"incomplete", "failed", "cancelled"}:
        metadata["incomplete"] = True
        metadata.setdefault("incomplete_reason", response.get("incomplete_details", {}).get("reason") if isinstance(response.get("incomplete_details"), Mapping) else status)
    estimated_cost = _estimated_cost_usd(usage, options)
    if estimated_cost is not None:
        metadata["actual_cost_usd"] = estimated_cost

    return {
        "id": response.get("id"),
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": response.get("status"),
            }
        ],
        "usage": usage,
        "_generation_metadata": metadata,
    }


def call_openai_responses(
    model_id: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Call OpenAI Responses and return a Chat-Completions-shaped payload.

    The API key is read from ``OPENAI_API_KEY`` by default (or the environment
    variable named by ``api_key_env``).  ``options['api_key']`` is accepted for
    the existing generation CLI's injected-credential contract.  Neither the
    key nor the raw response is written to disk by this transport.
    """

    api_key_env = str(options.get("api_key_env", "OPENAI_API_KEY"))
    api_key = str(options.get("api_key") or os.environ.get(api_key_env, "")).strip()
    if not api_key:
        raise ValueError(f"Set {api_key_env} before making an OpenAI Responses request.")

    body = build_openai_responses_body(model_id, messages, options)
    selected_model = str(body["model"])
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    request = urllib.request.Request(
        str(options.get("endpoint_url", OPENAI_RESPONSES_URL)),
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    configured_timeout = float(options.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    environment_timeout = os.environ.get(OPENAI_TIMEOUT_OVERRIDE_ENV)
    if environment_timeout:
        try:
            configured_timeout = max(configured_timeout, float(environment_timeout))
        except ValueError:
            raise ValueError(
                f"{OPENAI_TIMEOUT_OVERRIDE_ENV} must be a positive number when set."
            )
    if configured_timeout <= 0:
        raise ValueError("OpenAI request timeout must be positive.")
    try:
        with urllib.request.urlopen(
            request,
            timeout=configured_timeout,
        ) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenAI Responses HTTP {exc.code}: {error_body}") from exc

    if not isinstance(raw_response, Mapping):
        raise ValueError("OpenAI Responses response must be a JSON object.")
    return _normalise_response(
        raw_response,
        model=selected_model,
        model_id=model_id,
        options=options,
    )


__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_REASONING_EFFORT",
    "OPENAI_RESPONSES_URL",
    "build_openai_responses_body",
    "call_openai_responses",
]
