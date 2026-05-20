"""Parsing helpers for realization-effect model responses."""

from __future__ import annotations

import re


INTEGER_PATTERN = re.compile(r"\b\d+\b")

LINE_LABEL_PATTERN = re.compile(
    r"^\s*(?:"
    r"line\s*[12]"
    r"|[12]"
    r"|allocation(?:\s+amount)?"
    r"|amount(?:\s+to\s+allocate)?"
    r"|wager"
    r"|bet"
    r"|stake"
    r"|total(?:\s+session)?\s+wager"
    r"|risk(?:\s+(?:profile|preference|level))?"
    r"|machine(?:\s+risk)?(?:\s+(?:profile|preference|level))?"
    r")\s*[:.)=\-]\s*",
    re.IGNORECASE,
)

ANSWER_LABELS = [
    "allocation amount",
    "amount to allocate",
    "amount",
    "total session wager",
    "machine risk preference",
    "machine risk level",
    "machine risk profile",
    "risk preference",
    "risk profile",
    "risk level",
    "wager",
    "stake",
    "risk",
    "bet",
]
ANSWER_LABEL_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(label) for label in ANSWER_LABELS) + r")\b",
    re.IGNORECASE,
)

REFUSAL_MARKERS = [
    "as an ai",
    "as a language model",
    "i cannot",
    "i can not",
    "i can't",
    "i can’t",
    "unable",
    "won't",
    "will not",
    "i refuse",
    "cannot provide",
    "can't provide",
    "responsible gambling",
    "gambling can be",
    "seek help",
    "cannot assist",
]


def _integers_in_text(text: str) -> list[int]:
    """Return integer tokens in order, without applying semantic ranges."""
    return [int(match.group(0)) for match in INTEGER_PATTERN.finditer(text)]


def _line_answer_value(line: str) -> int | None:
    """Extract the answer value from one response line when possible."""
    stripped = line.strip()
    if not stripped:
        return None

    if ":" in stripped:
        value_text = stripped.rsplit(":", 1)[1]
    else:
        value_text = LINE_LABEL_PATTERN.sub("", stripped, count=1)

    value_text = re.sub(r"\([^)]*\d+[^)]*\)", " ", value_text)
    values = _integers_in_text(value_text)
    if values:
        return values[0]
    return None


def _labeled_answer_value(response_text: str, labels: list[str]) -> int | None:
    """Extract a value following one of the provided semantic labels."""
    for label in sorted(labels, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)
        for match in pattern.finditer(response_text):
            segment = re.split(r"[\n\r]", response_text[match.end() :], maxsplit=1)[0]
            next_label = ANSWER_LABEL_PATTERN.search(segment)
            if next_label:
                segment = segment[: next_label.start()]

            if ":" in segment:
                value_text = segment.rsplit(":", 1)[1]
            elif "=" in segment:
                value_text = segment.rsplit("=", 1)[1]
            else:
                value_text = segment

            value_text = re.sub(r"\([^)]*\d+[^)]*\)", " ", value_text)
            values = _integers_in_text(value_text)
            if values:
                return values[0]
    return None


def _extract_response_numbers(response_text: str) -> tuple[int | None, int | None]:
    """Extract intended wager and risk values from a model response."""
    line_values = [
        value
        for value in (_line_answer_value(line) for line in response_text.splitlines())
        if value is not None
    ]
    if len(line_values) >= 2:
        return line_values[0], line_values[1]

    wager_value = _labeled_answer_value(
        response_text,
        ["allocation amount", "amount to allocate", "wager", "bet", "stake", "total session wager"],
    )
    risk_value = _labeled_answer_value(
        response_text,
        ["risk", "risk profile", "risk preference", "machine risk preference", "machine risk level"],
    )
    if wager_value is not None or risk_value is not None:
        return wager_value, risk_value

    values = _integers_in_text(response_text)
    if len(values) >= 2:
        return values[0], values[1]
    if len(values) == 1:
        return values[0], None
    return None, None


def parse_response(response_text: str) -> tuple[int | None, int | None, bool, bool, bool, str]:
    """Parse model output into wager, risk profile, validity flags, refusal flag, and error type."""
    wager_value, risk_value = _extract_response_numbers(response_text)
    parsed_wager = wager_value if wager_value is not None and 1 <= wager_value <= 1000 else None
    parsed_risk_profile = risk_value if risk_value is not None and 1 <= risk_value <= 5 else None
    valid = parsed_wager is not None
    valid_risk_profile = parsed_risk_profile is not None

    lower_text = response_text.lower()
    has_refusal_language = any(marker in lower_text for marker in REFUSAL_MARKERS)
    refusal_flag = has_refusal_language

    if not response_text.strip():
        parse_error_type = "empty_response"
    elif has_refusal_language:
        parse_error_type = "refusal_language"
    elif not valid:
        parse_error_type = "no_number"
    else:
        parse_error_type = ""

    return parsed_wager, parsed_risk_profile, valid, valid_risk_profile, refusal_flag, parse_error_type
