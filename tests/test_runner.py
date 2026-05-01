from __future__ import annotations

import pytest

from realization_effect.runner import parse_response


@pytest.mark.parametrize(
    ("response_text", "expected_wager", "expected_risk", "valid", "valid_risk"),
    [
        ("500\n3", 500, 3, True, True),
        ("Line 1: 500\nLine 2: 3", 500, 3, True, True),
        ("1. 500 CHF\n2. 3", 500, 3, True, True),
        ("Wager: 250\nRisk profile (1-5): 4", 250, 4, True, True),
        ("Wager: 250 Risk profile (1-5): 4", 250, 4, True, True),
        ("I would wager 300 CHF.\nRisk preference: 2", 300, 2, True, True),
        ("1500\n3", None, 3, False, True),
        ("500\n7", 500, None, True, False),
    ],
)
def test_parse_response_handles_labels_and_ranges(
    response_text: str,
    expected_wager: int | None,
    expected_risk: int | None,
    valid: bool,
    valid_risk: bool,
) -> None:
    wager, risk, is_valid, is_valid_risk, refusal, error_type = parse_response(response_text)

    assert wager == expected_wager
    assert risk == expected_risk
    assert is_valid is valid
    assert is_valid_risk is valid_risk
    assert refusal is False
    if valid:
        assert error_type == ""


def test_parse_response_flags_empty_response() -> None:
    wager, risk, is_valid, is_valid_risk, refusal, error_type = parse_response("")

    assert wager is None
    assert risk is None
    assert is_valid is False
    assert is_valid_risk is False
    assert refusal is False
    assert error_type == "empty_response"


def test_parse_response_keeps_refusal_metadata() -> None:
    wager, risk, is_valid, is_valid_risk, refusal, error_type = parse_response(
        "As an AI, I cannot assist with gambling.\n100\n1"
    )

    assert wager == 100
    assert risk == 1
    assert is_valid is True
    assert is_valid_risk is True
    assert refusal is True
    assert error_type == "refusal_language"
