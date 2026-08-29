from __future__ import annotations

import pytest

from activation_analysis.constrained_generation import (
    NumericRangeLogitsProcessor,
    allowed_next_token_ids,
    allowed_values_for_parser,
    build_numeric_logits_processor,
    numeric_token_sequences,
)


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        value = text.strip()
        prefix = 1000 if text.startswith(" ") else 0
        return {"input_ids": [prefix + int(value)]}


def test_registered_parser_values_are_exact() -> None:
    assert allowed_values_for_parser("single_integer_allocation_0_to_100_v1") == tuple(range(101))
    assert allowed_values_for_parser("single_integer_probability_v1") == tuple(range(101))
    assert allowed_values_for_parser("single_integer_choice_1_or_2_v1") == (1, 2)
    assert allowed_values_for_parser("two_integers_sum_100_v1") is None


def test_prefix_constraint_allows_only_candidates_and_eos_after_completion() -> None:
    sequences = ((10,), (11, 20))
    assert allowed_next_token_ids([], sequences, 99) == (10, 11)
    assert allowed_next_token_ids([10], sequences, 99) == (99,)
    assert allowed_next_token_ids([11], sequences, 99) == (20,)
    assert allowed_next_token_ids([11, 20], sequences, 99) == (99,)
    assert allowed_next_token_ids([42], sequences, 99) == ()


def test_numeric_token_sequences_include_leading_space_variants() -> None:
    sequences = numeric_token_sequences(_Tokenizer(), (0, 100))
    assert sequences == ((0,), (1000,), (100,), (1100,))


def test_processor_masks_every_non_numeric_next_token() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = _Tokenizer()
    processor = build_numeric_logits_processor(
        tokenizer,
        parser_id="single_integer_choice_1_or_2_v1",
        start_length=2,
    )
    assert isinstance(processor, NumericRangeLogitsProcessor)
    scores = torch.arange(0.0, 1200.0).reshape(1, 1200)
    result = processor(torch.tensor([[7, 8]]), scores)
    allowed = {1, 2, 1001, 1002}
    assert all(
        result[0, token_id].item() == scores[0, token_id].item()
        for token_id in allowed
    )
    assert all(
        result[0, token_id].item() < -1e30
        for token_id in (0, 3, 99, 500)
    )


def test_processor_allows_eos_only_after_a_complete_candidate() -> None:
    torch = pytest.importorskip("torch")
    processor = NumericRangeLogitsProcessor(
        candidate_sequences=((5,),),
        start_length=1,
        eos_token_id=99,
    )
    scores = torch.zeros((1, 100))
    prefix_result = processor(torch.tensor([[8]]), scores)
    assert prefix_result[0, 5].item() == 0.0
    assert prefix_result[0, 99].item() < -1e30
    completed_result = processor(torch.tensor([[8, 5]]), scores)
    assert completed_result[0, 99].item() == 0.0
    assert completed_result[0, 5].item() < -1e30
