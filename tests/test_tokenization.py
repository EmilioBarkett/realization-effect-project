from __future__ import annotations

import pytest

from activation_analysis.tokenization import (
    TokenLengthLimitError,
    enforce_token_length_limit,
    format_model_prompt,
    inspect_token_lengths,
    tokenize_padded_without_truncation,
    validate_padded_encoding_lengths,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, prompts, **kwargs):
        self.calls.append(dict(kwargs))
        values = [prompts] if isinstance(prompts, str) else list(prompts)
        lengths = [len(str(prompt).split()) for prompt in values]
        if kwargs.get("padding"):
            width = max(lengths, default=0)
        else:
            width = None
        input_ids = []
        attention_mask = []
        for length in lengths:
            row = list(range(length))
            mask = [1] * length
            if width is not None:
                row.extend([0] * (width - length))
                mask.extend([0] * (width - length))
            input_ids.append(row)
            attention_mask.append(mask)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def test_token_length_preflight_disables_truncation_and_matches_padded_encoding() -> None:
    tokenizer = _FakeTokenizer()
    report = inspect_token_lengths(
        tokenizer,
        ["one two", "one two three"],
        max_length=3,
        prompt_ids=["short", "long"],
        tokenizer_id="fake/tokenizer",
        revision="rev",
    )

    assert report.lengths == (2, 3)
    assert report.over_limit_prompt_ids == ()
    encoded = tokenize_padded_without_truncation(tokenizer, ["one two", "one two three"])
    validate_padded_encoding_lengths(encoded, report)
    assert all(call["truncation"] is False for call in tokenizer.calls)
    assert report.to_mapping()["length_summary"]["max"] == 3


def test_token_length_preflight_fails_closed_before_silent_truncation() -> None:
    tokenizer = _FakeTokenizer()
    report = inspect_token_lengths(
        tokenizer,
        ["one two three four"],
        max_length=3,
        prompt_ids=["too_long"],
    )

    with pytest.raises(TokenLengthLimitError, match="too_long=4"):
        enforce_token_length_limit(report)


def test_padded_encoding_must_preserve_preflight_lengths() -> None:
    tokenizer = _FakeTokenizer()
    report = inspect_token_lengths(tokenizer, ["one two", "one two three"], max_length=3)

    with pytest.raises(ValueError, match="changed token lengths"):
        validate_padded_encoding_lengths(
            {"input_ids": [[0, 1], [0, 1, 2]], "attention_mask": [[1, 1], [1, 1, 0]]},
            report,
        )


def test_model_prompt_formatting_is_shared_by_preflight_and_execution() -> None:
    tokenizer = _FakeTokenizer()
    assert format_model_prompt(tokenizer, "Decide.", prompt_format="completion") == "Decide."
    assert format_model_prompt(
        tokenizer,
        "Decide.",
        prompt_format="chat",
        system_prompt="Follow the response contract.",
    ) == "system: Follow the response contract.\nuser: Decide."
