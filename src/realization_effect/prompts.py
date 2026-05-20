"""Prompt builders for realization-effect behavioral trials."""

from __future__ import annotations

from typing import Callable


# Treat small realized outcomes as near-neutral wording.
NEAR_ZERO_REALIZED_THRESHOLD = 50

TWO_INTEGER_INSTRUCTION = (
    "Respond with two integers on separate lines:\n"
    "Line 1: your total session wager in CHF (1–1000)\n"
    "Line 2: your machine risk preference (1–5, where 1 = very conservative/"
    "low-volatility and 5 = very aggressive/high-volatility)"
)

PAPER_PROMPT_TEMPLATE = (
    "You are currently on a casino visit. {history_sentence} Your balance remains "
    "on your playing card. How much do you want to wager in your next session, and "
    "which type of slot machine would you choose?\n" + TWO_INTEGER_INSTRUCTION
)

REALIZED_PROMPT_TEMPLATE = (
    "You visited this casino on a previous occasion. At the end of that visit you "
    "cashed out and {history_sentence} That visit is now over. You have just "
    "arrived for a new visit. How much do you want to wager in your first session, "
    "and which type of slot machine would you choose?\n" + TWO_INTEGER_INSTRUCTION
)


def _build_prompt_absolute(outcome_type: str, amount: int) -> str:
    """Build absolute framing prompt text for one trial."""
    if outcome_type == "paper":
        if amount == 0:
            history_sentence = "So far during this visit you are even and have neither won nor lost."
        elif amount > 0:
            history_sentence = f"So far during this visit you have won {amount} CHF across previous sessions."
        else:
            history_sentence = f"So far during this visit you have lost {abs(amount)} CHF across previous sessions."
        return PAPER_PROMPT_TEMPLATE.format(history_sentence=history_sentence)

    if outcome_type == "realized":
        if amount == 0:
            history_sentence = "you finished exactly even (0 CHF)."
        elif abs(amount) <= NEAR_ZERO_REALIZED_THRESHOLD:
            if amount > 0:
                history_sentence = f"you finished roughly even, with a small gain of {amount} CHF."
            else:
                history_sentence = f"you finished roughly even, with a small loss of {abs(amount)} CHF."
        elif amount > 0:
            history_sentence = f"you received {amount} CHF."
        else:
            history_sentence = f"you paid {abs(amount)} CHF."

        return REALIZED_PROMPT_TEMPLATE.format(history_sentence=history_sentence)

    raise ValueError(f"Unsupported outcome_type: {outcome_type}")


def _build_prompt_balance(outcome_type: str, amount: int) -> str:
    """Build balance framing prompt text for one trial."""
    if outcome_type == "paper":
        if amount > 0:
            balance_sentence = f"your current balance on your playing card is {amount} CHF above where you began."
        elif amount < 0:
            balance_sentence = f"your current balance on your playing card is {abs(amount)} CHF below where you began."
        else:
            balance_sentence = "your current balance on your playing card is exactly the same as where you began."

        return (
            "You are on a casino visit. Relative to the start of this visit, "
            f"{balance_sentence}\n\n"
            "This visit is still ongoing, and your balance remains on your playing card.\n\n"
            "You are about to begin an upcoming slot machine session.\n\n"
            "How much do you want to wager in this upcoming session, and which machine risk "
            "level would you choose?\n" + TWO_INTEGER_INSTRUCTION
        )

    if outcome_type == "realized":
        if amount > 0:
            result_sentence = f"your final result was {amount} CHF above where you started."
        elif amount < 0:
            result_sentence = f"your final result was {abs(amount)} CHF below where you started."
        else:
            result_sentence = "your final result was exactly the same as where you started."

        return (
            "You are beginning a new casino visit. On your previous casino visit, "
            f"{result_sentence}\n\n"
            "That previous visit ended, and you cashed out your playing card balance.\n\n"
            "You are about to begin an upcoming slot machine session in this new visit.\n\n"
            "How much do you want to wager in this upcoming session, and which machine risk "
            "level would you choose?\n" + TWO_INTEGER_INSTRUCTION
        )

    raise ValueError(f"Unsupported outcome_type: {outcome_type}")


def _build_prompt_qualitative(outcome_type: str, amount: int) -> str:
    """Build qualitative non-numeric framing prompt for one trial."""
    if outcome_type == "paper":
        if amount == 0:
            history_sentence = "So far during this visit you are exactly even and have neither won nor lost."
        elif amount > 0:
            if amount <= 80:
                history_sentence = "So far during this visit you have won a modest amount."
            else:
                history_sentence = "So far during this visit you have won a substantial amount."
        else:
            if amount >= -96:
                history_sentence = "So far during this visit you have lost a small amount."
            elif amount >= -309:
                history_sentence = "So far during this visit you have lost a moderate amount."
            else:
                history_sentence = "So far during this visit you have lost a substantial amount."
        return (
            "You are currently on a casino visit. "
            f"{history_sentence} "
            "Your balance remains on your playing card. "
            "How much do you want to wager in your next session, and which type of "
            "slot machine would you choose?\n" + TWO_INTEGER_INSTRUCTION
        )

    if outcome_type == "realized":
        if amount == 0:
            result_sentence = "you finished exactly even."
        elif amount > 0:
            result_sentence = "you came out ahead."
        elif amount >= -62:
            result_sentence = "you lost a small amount."
        elif amount >= -787:
            result_sentence = "you lost a moderate amount."
        elif amount >= -2790:
            result_sentence = "you lost a large amount."
        else:
            result_sentence = "you lost a very large amount."
        return (
            "You visited this casino on a previous occasion. At the end of that visit "
            f"you cashed out and {result_sentence} That visit is now over. You have "
            "just arrived for a new visit. How much do you want to wager in your first "
            "session, and which type of slot machine would you choose?\n"
            + TWO_INTEGER_INSTRUCTION
        )

    raise ValueError(f"Unsupported outcome_type: {outcome_type}")


PROMPT_BUILDERS: dict[str, Callable[[str, int], str]] = {
    "absolute": _build_prompt_absolute,
    "balance": _build_prompt_balance,
    "qualitative": _build_prompt_qualitative,
}


def build_prompt(outcome_type: str, amount: int, prompt_version: str = "absolute") -> str:
    """Build the condition-specific user prompt for one trial."""
    builder = PROMPT_BUILDERS.get(prompt_version)
    if builder is None:
        supported = ", ".join(sorted(PROMPT_BUILDERS.keys()))
        raise ValueError(f"Unsupported prompt_version '{prompt_version}'. Supported: {supported}")
    return builder(outcome_type, amount)
