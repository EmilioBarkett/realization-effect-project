"""Fail-closed tokenization helpers for model-side benchmark runs.

The activation and steering paths must never silently replace a frozen prompt
with a truncated one.  This module performs a no-truncation inspection first,
then provides the padded encoding used by the model runner after the limit has
been checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class TokenLengthLimitError(ValueError):
    """Raised when one or more frozen prompts exceed the configured limit."""


@dataclass(frozen=True)
class TokenLengthReport:
    """Token lengths observed without truncation."""

    prompt_ids: tuple[str, ...]
    lengths: tuple[int, ...]
    max_length: int
    tokenizer_id: str | None = None
    revision: str | None = None

    @property
    def over_limit_indices(self) -> tuple[int, ...]:
        return tuple(index for index, length in enumerate(self.lengths) if length > self.max_length)

    @property
    def over_limit_prompt_ids(self) -> tuple[str, ...]:
        return tuple(self.prompt_ids[index] for index in self.over_limit_indices)

    def to_mapping(self) -> dict[str, Any]:
        values = np.asarray(self.lengths, dtype=np.float64)
        summary: dict[str, Any] = {
            "count": int(values.size),
            "min": int(values.min()) if values.size else None,
            "max": int(values.max()) if values.size else None,
            "mean": float(values.mean()) if values.size else None,
            "p50": float(np.percentile(values, 50)) if values.size else None,
            "p90": float(np.percentile(values, 90)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "p99": float(np.percentile(values, 99)) if values.size else None,
            "over_limit_count": len(self.over_limit_indices),
        }
        return {
            "tokenizer_id": self.tokenizer_id,
            "revision": self.revision,
            "max_length": self.max_length,
            "truncation": False,
            "length_summary": summary,
            "prompt_lengths": [
                {
                    "prompt_id": prompt_id,
                    "token_length": length,
                    "over_limit": length > self.max_length,
                }
                for prompt_id, length in zip(self.prompt_ids, self.lengths, strict=True)
            ],
            "over_limit_prompt_ids": list(self.over_limit_prompt_ids),
        }


def format_model_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    prompt_format: str,
    system_prompt: str = "",
) -> str:
    """Apply the same completion/chat formatting used by model execution."""

    if prompt_format == "completion":
        return prompt
    if prompt_format == "chat":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                "Tokenizer does not support apply_chat_template; use prompt_format='completion'."
            )
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError("prompt_format must be 'completion' or 'chat'.")


def _call_tokenizer(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    padding: bool,
    return_tensors: str | None,
    return_offsets_mapping: bool = False,
) -> Any:
    """Call a tokenizer with truncation explicitly disabled.

    A few lightweight test tokenizers and older tokenizer implementations do
    not accept ``return_offsets_mapping`` or ``max_length=None``.  The
    fallbacks preserve the important invariant: ``truncation=False`` is always
    sent, and no fallback enables truncation.
    """

    base_kwargs: dict[str, Any] = {
        "padding": padding,
        "truncation": False,
        "max_length": None,
        "return_tensors": return_tensors,
    }
    attempts = [
        {**base_kwargs, "return_offsets_mapping": return_offsets_mapping},
        dict(base_kwargs),
        {key: value for key, value in base_kwargs.items() if key != "max_length"},
        {
            key: value
            for key, value in base_kwargs.items()
            if key not in {"max_length", "return_tensors"}
        },
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return tokenizer(prompts, **kwargs)
        except (NotImplementedError, TypeError, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _as_rows(value: Any) -> list[list[Any]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("Tokenizer input_ids must be a list or tensor.")
    if value and not isinstance(value[0], list):
        value = [value]
    if not all(isinstance(row, list) for row in value):
        raise ValueError("Tokenizer input_ids rows must be lists.")
    return value


def _lengths_from_encoding(encoded: Any, expected_count: int) -> list[int]:
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Tokenizer output is missing input_ids.") from exc
    rows = _as_rows(input_ids)
    if len(rows) != expected_count:
        raise ValueError(
            f"Tokenizer returned {len(rows)} input rows for {expected_count} prompts."
        )

    attention_mask = None
    try:
        attention_mask = encoded["attention_mask"]
    except (KeyError, TypeError):
        pass
    if attention_mask is None:
        return [len(row) for row in rows]
    masks = _as_rows(attention_mask)
    if len(masks) != expected_count:
        raise ValueError("Tokenizer attention_mask row count does not match input_ids.")
    return [sum(int(value) for value in row) for row in masks]


def inspect_token_lengths(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
    prompt_ids: Sequence[str] | None = None,
    tokenizer_id: str | None = None,
    revision: str | None = None,
) -> TokenLengthReport:
    """Inspect exact token lengths with truncation disabled."""

    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer.")
    materialized = [str(prompt) for prompt in prompts]
    if not materialized:
        raise ValueError("At least one prompt is required for tokenization preflight.")
    ids = tuple(str(value) for value in (prompt_ids or [f"prompt_{index:05d}" for index in range(len(materialized))]))
    if len(ids) != len(materialized):
        raise ValueError("prompt_ids and prompts must have the same length.")
    encoded = _call_tokenizer(
        tokenizer,
        materialized,
        padding=False,
        return_tensors=None,
    )
    lengths = tuple(_lengths_from_encoding(encoded, len(materialized)))
    return TokenLengthReport(
        prompt_ids=ids,
        lengths=lengths,
        max_length=max_length,
        tokenizer_id=tokenizer_id,
        revision=revision,
    )


def enforce_token_length_limit(report: TokenLengthReport) -> None:
    """Fail closed when the frozen input would be truncated."""

    if not report.over_limit_indices:
        return
    examples = [
        f"{prompt_id}={report.lengths[index]}"
        for index, prompt_id in zip(report.over_limit_indices[:5], report.over_limit_prompt_ids[:5], strict=True)
    ]
    raise TokenLengthLimitError(
        f"{len(report.over_limit_indices)} prompt(s) exceed max_length={report.max_length}; "
        "refusing silent truncation. Examples: "
        + ", ".join(examples)
    )


def tokenize_padded_without_truncation(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    return_tensors: str = "pt",
    return_offsets_mapping: bool = False,
) -> Any:
    """Encode already-checked prompts for model input without truncation."""

    return _call_tokenizer(
        tokenizer,
        prompts,
        padding=True,
        return_tensors=return_tensors,
        return_offsets_mapping=return_offsets_mapping,
    )


def validate_padded_encoding_lengths(encoded: Any, report: TokenLengthReport) -> None:
    """Ensure the padded model encoding still contains every checked token."""

    lengths = _lengths_from_encoding(encoded, len(report.prompt_ids))
    if tuple(lengths) != report.lengths:
        raise ValueError(
            "Padded tokenizer output changed token lengths relative to the no-truncation preflight."
        )


__all__ = [
    "TokenLengthLimitError",
    "TokenLengthReport",
    "enforce_token_length_limit",
    "format_model_prompt",
    "inspect_token_lengths",
    "tokenize_padded_without_truncation",
    "validate_padded_encoding_lengths",
]
