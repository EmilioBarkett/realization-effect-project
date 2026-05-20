"""OpenRouter/OpenAI API helpers for realization-effect experiments."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a casino visitor making decisions about gambling at a slot machine. "
    "You will be given information about your recent gambling history and asked two "
    "questions: how much to wager in your next slot machine session, and which type "
    "of slot machine to choose. Respond with exactly two integers, each on its own "
    "line: first your total session wager in CHF (1–1000), then your machine risk "
    "preference (1 = very conservative/low-volatility, 5 = very aggressive/"
    "high-volatility). Do not explain your reasoning."
)


def create_openrouter_client(api_key: str) -> OpenAI:
    """Create an OpenAI-compatible OpenRouter client."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _extract_response_text(response: Any) -> str:
    """Extract plain text from a Responses API object."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_type = getattr(content, "type", "")
            if content_type in {"output_text", "text"}:
                text = getattr(content, "text", "")
                if text:
                    parts.append(text)

    return "\n".join(parts).strip()


def _is_retryable_error(error: Exception) -> bool:
    """Return True for transient API failures that should be retried."""
    retryable_names = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }
    if error.__class__.__name__ in retryable_names:
        return True

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 429} or status_code >= 500

    return False


def call_model(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_retries: int,
) -> tuple[str, str]:
    """Call the Responses API with retry/backoff and return text + request id."""
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=prompt,
                temperature=temperature,
            )
            response_text = _extract_response_text(response)
            request_id = str(
                getattr(response, "_request_id", None)
                or getattr(response, "id", None)
                or ""
            )
            return response_text, request_id
        except Exception as error:
            last_error = error
            if attempt >= max_retries or not _is_retryable_error(error):
                raise

            time.sleep(2**attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("call_model failed without a captured exception")
