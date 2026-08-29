"""Shared constrained decoding for registered numeric response contracts.

The model-side behavior tasks use strict integer parsers.  A parser can reject
an explanation after the fact, but it cannot prevent an otherwise correct
answer from being truncated or padded with reasoning.  This module provides a
small Transformers-compatible logits processor that allows only complete
registered numeric answers followed by EOS.  It is intentionally independent
of a particular model family so Mistral and Qwen use the same response
channel.

The pure prefix helpers are usable without importing torch and are covered by
the base-environment tests.  The logits processor imports torch only when a
generation call actually invokes it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


NUMERIC_PARSER_VALUES: dict[str, tuple[int, ...]] = {
    "single_integer_allocation_0_to_100_v1": tuple(range(101)),
    "single_integer_probability_v1": tuple(range(101)),
    "single_integer_choice_1_or_2_v1": (1, 2),
}


def allowed_values_for_parser(parser_id: str) -> tuple[int, ...] | None:
    """Return the registered numeric choices for ``parser_id``.

    ``None`` means the parser is not a single numeric response contract and
    must be handled by its existing generation path.  Two-integer contracts
    are deliberately not silently approximated by this single-integer
    constraint.
    """

    return NUMERIC_PARSER_VALUES.get(str(parser_id))


def _tokenize_candidate(tokenizer: Any, text: str) -> tuple[int, ...]:
    """Encode one candidate without adding special tokens."""

    if callable(tokenizer):
        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except (TypeError, ValueError, NotImplementedError):
            encoded = None
        if encoded is not None:
            if isinstance(encoded, dict):
                encoded = encoded.get("input_ids")
            elif hasattr(encoded, "get"):
                encoded = encoded.get("input_ids")
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            if isinstance(encoded, (list, tuple)):
                return tuple(int(value) for value in encoded)

    encoder = getattr(tokenizer, "encode", None)
    if callable(encoder):
        encoded = encoder(text, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if isinstance(encoded, (list, tuple)):
            return tuple(int(value) for value in encoded)
    raise TypeError("Tokenizer must support callable(text, add_special_tokens=False) or encode().")


def numeric_token_sequences(
    tokenizer: Any,
    values: Iterable[int],
    *,
    allow_leading_space: bool = True,
) -> tuple[tuple[int, ...], ...]:
    """Encode the accepted numeric strings, retaining whitespace variants."""

    candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for value in values:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"Numeric constrained values must be integers, got {value!r}.")
        strings = [str(int(value))]
        if allow_leading_space:
            strings.append(f" {int(value)}")
        for candidate in strings:
            sequence = _tokenize_candidate(tokenizer, candidate)
            if not sequence:
                raise ValueError(f"Tokenizer produced an empty sequence for numeric candidate {candidate!r}.")
            if sequence not in seen:
                seen.add(sequence)
                candidates.append(sequence)
    if not candidates:
        raise ValueError("At least one numeric constrained value is required.")
    return tuple(candidates)


def _normalise_eos_ids(eos_token_id: int | Sequence[int] | None) -> tuple[int, ...]:
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        return (eos_token_id,)
    return tuple(int(value) for value in eos_token_id)


def allowed_next_token_ids(
    generated_suffix: Sequence[int],
    candidate_sequences: Iterable[Sequence[int]],
    eos_token_id: int | Sequence[int] | None,
) -> tuple[int, ...]:
    """Return the only token IDs allowed after a generated prefix.

    EOS is allowed only once the current prefix is a complete candidate.  If
    the tokenizer has a candidate-prefix collision (for example ``1`` and
    ``10``), both EOS and the valid continuation remain available.
    """

    prefix = tuple(int(value) for value in generated_suffix)
    allowed: set[int] = set()
    complete = False
    for raw_sequence in candidate_sequences:
        sequence = tuple(int(value) for value in raw_sequence)
        if sequence[: len(prefix)] != prefix:
            continue
        if len(sequence) == len(prefix):
            complete = True
        elif len(sequence) > len(prefix):
            allowed.add(sequence[len(prefix)])
    if complete:
        allowed.update(_normalise_eos_ids(eos_token_id))
    return tuple(sorted(allowed))


class NumericRangeLogitsProcessor:
    """A Transformers-compatible processor for one numeric response field."""

    def __init__(
        self,
        *,
        candidate_sequences: Iterable[Sequence[int]],
        start_length: int,
        eos_token_id: int | Sequence[int] | None,
    ) -> None:
        if isinstance(start_length, bool) or start_length < 0:
            raise ValueError("start_length must be a non-negative integer.")
        self.candidate_sequences = tuple(tuple(int(value) for value in sequence) for sequence in candidate_sequences)
        if not self.candidate_sequences or any(not sequence for sequence in self.candidate_sequences):
            raise ValueError("candidate_sequences must contain non-empty token sequences.")
        self.start_length = int(start_length)
        self.eos_token_ids = _normalise_eos_ids(eos_token_id)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Mask logits that cannot complete one registered numeric choice."""

        import torch

        if getattr(input_ids, "ndim", None) != 2 or getattr(scores, "ndim", None) != 2:
            raise ValueError("NumericRangeLogitsProcessor expects rank-2 input_ids and scores.")
        if int(input_ids.shape[0]) != int(scores.shape[0]):
            raise ValueError("input_ids and scores must have the same batch size.")
        masked = scores.new_full(scores.shape, torch.finfo(scores.dtype).min)
        vocabulary_size = int(scores.shape[1])
        for row_index in range(int(input_ids.shape[0])):
            row = input_ids[row_index, self.start_length :].detach().cpu().tolist()
            allowed = allowed_next_token_ids(row, self.candidate_sequences, self.eos_token_ids)
            allowed = tuple(token_id for token_id in allowed if 0 <= token_id < vocabulary_size)
            if not allowed:
                # A malformed prefix must terminate as an invalid response;
                # never reopen the full vocabulary after the constraint loses
                # its prefix match.
                allowed = self.eos_token_ids
            if not allowed:
                raise ValueError("Numeric constrained generation has no valid next token or EOS token.")
            indices = torch.as_tensor(allowed, device=scores.device, dtype=torch.long)
            masked[row_index].index_copy_(0, indices, scores[row_index].index_select(0, indices))
        return masked


def build_numeric_logits_processor(
    tokenizer: Any,
    *,
    parser_id: str,
    start_length: int,
) -> NumericRangeLogitsProcessor | None:
    """Build the shared processor for a registered single-integer parser."""

    values = allowed_values_for_parser(parser_id)
    if values is None:
        return None
    sequences = numeric_token_sequences(tokenizer, values)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        nested = getattr(tokenizer, "tokenizer", None)
        eos_token_id = getattr(nested, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError(f"Tokenizer for parser {parser_id!r} does not define eos_token_id.")
    return NumericRangeLogitsProcessor(
        candidate_sequences=sequences,
        start_length=start_length,
        eos_token_id=eos_token_id,
    )


__all__ = [
    "NUMERIC_PARSER_VALUES",
    "NumericRangeLogitsProcessor",
    "allowed_next_token_ids",
    "allowed_values_for_parser",
    "build_numeric_logits_processor",
    "numeric_token_sequences",
]
