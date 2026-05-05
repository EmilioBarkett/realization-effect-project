#!/usr/bin/env python3
"""Run a realization-effect replication experiment with an OpenRouter model."""

import argparse
import csv
import hashlib
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

GENERATION_PROMPT_VERSION = "generation"
DEFAULT_GENERATION_PLAN = Path("configs/emotion_activation/final_inference_prompt_generation_v1.json")
DEFAULT_GENERATION_OUTPUT = Path("experiments/emotion_activation/prompts/final/final_inference_prompts_v1.csv")

# Constant instructions prompt used for every trial.
SYSTEM_PROMPT = (
    "You are a casino visitor making decisions about gambling at a slot machine. "
    "You will be given information about your recent gambling history and asked two "
    "questions: how much to wager in your next slot machine session, and which type "
    "of slot machine to choose. Respond with exactly two integers, each on its own "
    "line: first your total session wager in CHF (1–1000), then your machine risk "
    "preference (1 = very conservative/low-volatility, 5 = very aggressive/"
    "high-volatility). Do not explain your reasoning."
)

# Regex for integer tokens in model responses. Range checks happen after
# extraction so labels such as "Line 1: 500" do not get mistaken for answers.
INTEGER_PATTERN = re.compile(r"\b\d+\b")

# Regex for stripping common ordered-list / answer labels from response lines.
LINE_LABEL_PATTERN = re.compile(
    r"^\s*(?:"
    r"line\s*[12]"
    r"|[12]"
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

# Heuristic markers for refusals/moralizing language.
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

# Treat small realized outcomes as near-neutral wording.
NEAR_ZERO_REALIZED_THRESHOLD = 50

# Prompt templates by outcome type.
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

# Output schema.
RESULT_FIELDS = [
    "trial_id",
    "condition",
    "outcome_type",
    "amount",
    "run_number",
    "model",
    "temperature",
    "prompt_version",
    "prompt_text",
    "response_text",
    "parsed_wager",
    "log_wager",
    "valid",
    "risk_profile",
    "valid_risk_profile",
    "refusal_flag",
    "parse_error_type",
    "timestamp",
    "request_id",
]


def load_conditions(csv_path: Path) -> List[Dict[str, object]]:
    """Load and validate the condition table from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Conditions file not found: {csv_path}")

    conditions: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"condition", "outcome_type", "amount"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("conditions.csv must contain: condition,outcome_type,amount")

        for row in reader:
            condition = row["condition"].strip()
            outcome_type = row["outcome_type"].strip().lower()
            amount = int(row["amount"])

            if outcome_type not in {"paper", "realized"}:
                raise ValueError(f"Invalid outcome_type '{outcome_type}' in condition '{condition}'")

            conditions.append(
                {
                    "condition": condition,
                    "outcome_type": outcome_type,
                    "amount": amount,
                }
            )

    if not conditions:
        raise ValueError("No rows found in conditions.csv")

    return conditions


def _build_prompt_absolute(outcome_type: str, amount: int) -> str:
    """Build absolute framing prompt text for one trial."""
    if outcome_type == "paper":
        if amount == 0:
            history_sentence = "So far during this visit you are even and have neither won nor lost."
        elif amount > 0:
            history_sentence = f"So far during this visit you have won {amount} CHF across previous sessions."
        else:
            history_sentence = (
                f"So far during this visit you have lost {abs(amount)} CHF across previous sessions."
            )
        return PAPER_PROMPT_TEMPLATE.format(history_sentence=history_sentence)

    if outcome_type == "realized":
        if amount == 0:
            history_sentence = "you finished exactly even (0 CHF)."
        elif abs(amount) <= NEAR_ZERO_REALIZED_THRESHOLD:
            if amount > 0:
                history_sentence = (
                    f"you finished roughly even, with a small gain of {amount} CHF."
                )
            else:
                history_sentence = (
                    f"you finished roughly even, with a small loss of {abs(amount)} CHF."
                )
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
            balance_sentence = (
                f"your current balance on your playing card is {amount} CHF above where you began."
            )
        elif amount < 0:
            balance_sentence = (
                f"your current balance on your playing card is {abs(amount)} CHF below where you began."
            )
        else:
            balance_sentence = (
                "your current balance on your playing card is exactly the same as where you began."
            )

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
    """Build qualitative (non-numeric) framing prompt for one trial.

    Amounts are described in relative terms only — no CHF figures — to test
    whether the realization effect holds when the reference point is implicit
    rather than exact. Thresholds follow the paper's quintile boundaries.
    """
    if outcome_type == "paper":
        if amount == 0:
            history_sentence = "So far during this visit you are exactly even and have neither won nor lost."
        elif amount > 0:
            if amount <= 80:  # Q4: small gain
                history_sentence = "So far during this visit you have won a modest amount."
            else:             # Q5: large gain
                history_sentence = "So far during this visit you have won a substantial amount."
        else:
            if amount >= -96:   # Q3: small paper loss
                history_sentence = "So far during this visit you have lost a small amount."
            elif amount >= -309:  # Q2: medium paper loss
                history_sentence = "So far during this visit you have lost a moderate amount."
            else:                 # Q1: large paper loss
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
        elif amount > 0:     # Q5: realized gain
            result_sentence = "you came out ahead."
        elif amount >= -62:  # Q4: near-zero realized loss (baseline)
            result_sentence = "you lost a small amount."
        elif amount >= -787: # Q3: medium realized loss
            result_sentence = "you lost a moderate amount."
        elif amount >= -2790:  # Q2: large realized loss
            result_sentence = "you lost a large amount."
        else:                  # Q1: extreme realized loss
            result_sentence = "you lost a very large amount."
        return (
            "You visited this casino on a previous occasion. At the end of that visit "
            f"you cashed out and {result_sentence} That visit is now over. You have "
            "just arrived for a new visit. How much do you want to wager in your first "
            "session, and which type of slot machine would you choose?\n"
            + TWO_INTEGER_INSTRUCTION
        )

    raise ValueError(f"Unsupported outcome_type: {outcome_type}")


PROMPT_BUILDERS = {
    "absolute": _build_prompt_absolute,
    "balance": _build_prompt_balance,
    "qualitative": _build_prompt_qualitative,
}


def build_prompt(outcome_type: str, amount: int, prompt_version: str = "absolute") -> str:
    """Build the condition-specific user prompt for one trial."""
    builder = PROMPT_BUILDERS.get(prompt_version)
    if builder is None:
        supported = ", ".join(sorted(PROMPT_BUILDERS.keys()))
        raise ValueError(
            f"Unsupported prompt_version '{prompt_version}'. Supported: {supported}"
        )
    return builder(outcome_type=outcome_type, amount=amount)


def _run_generation_prompt_mode(args: argparse.Namespace) -> None:
    """Route `--prompt-version generation` to the OpenRouter prompt generator."""

    from emotion_activation.openrouter_prompt_generation import (
        generate_prompt_csv,
        load_generation_plan,
        pilot_plan_one_job_per_cell,
    )

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before generating prompts.")

    plan = load_generation_plan(args.generation_plan)
    if args.generation_pilot_all_cells:
        plan = pilot_plan_one_job_per_cell(
            plan,
            count_per_model=args.generation_pilot_count_per_cell,
        )
    model_aliases = set(args.models) if args.models is not None else None
    output_path = (
        args.generation_output
        if args.generation_output is not None
        else (
            DEFAULT_GENERATION_OUTPUT
            if args.output == Path("results/results.csv")
            else args.output
        )
    )

    written = generate_prompt_csv(
        plan,
        output_path,
        api_key=api_key,
        model_aliases=model_aliases,
        limit_jobs=args.generation_limit_jobs,
        resume=args.generation_resume,
    )
    print(f"wrote {written} generated prompts to {output_path}")


def _extract_response_text(response: Any) -> str:
    """Extract plain text from a Responses API object."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
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
) -> Tuple[str, str]:
    """Call the Responses API with retry/backoff and return text + request id."""
    last_error: Optional[Exception] = None

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

            backoff_seconds = 2 ** attempt
            time.sleep(backoff_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("call_model failed without a captured exception")


def _integers_in_text(text: str) -> List[int]:
    """Return integer tokens in order, without applying semantic ranges."""
    return [int(match.group(0)) for match in INTEGER_PATTERN.finditer(text)]


def _line_answer_value(line: str) -> Optional[int]:
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


def _labeled_answer_value(response_text: str, labels: List[str]) -> Optional[int]:
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


def _extract_response_numbers(response_text: str) -> Tuple[Optional[int], Optional[int]]:
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
        ["wager", "bet", "stake", "total session wager"],
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


def parse_response(response_text: str) -> Tuple[Optional[int], Optional[int], bool, bool, bool, str]:
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


def _normalize_temperature(value: Any) -> str:
    """Normalize temperature to a stable string for resume keys."""
    try:
        return format(float(value), ".12g")
    except (TypeError, ValueError):
        return ""


def _slug_for_filename(value: str) -> str:
    """Convert arbitrary text to a filesystem-friendly token."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "value"


def build_block_output_path(
    output_path: Path,
    model: str,
    temperature: float,
    prompt_version: str,
) -> Path:
    """Return a deterministic per-block CSV path under results/blocks/."""
    temp_key = _normalize_temperature(temperature)
    block_key = f"{model}|{temp_key}|{prompt_version}"
    key_hash = hashlib.sha1(block_key.encode("utf-8")).hexdigest()[:10]
    filename = (
        "block__"
        f"model-{_slug_for_filename(model)}__"
        f"temp-{_slug_for_filename(temp_key)}__"
        f"prompt-{_slug_for_filename(prompt_version)}__"
        f"{key_hash}.csv"
    )
    return output_path.parent / "blocks" / filename


def _list_block_csv_paths(blocks_dir: Path) -> List[Path]:
    """List block CSVs in deterministic filename order."""
    if not blocks_dir.exists():
        return []
    return sorted(
        (path for path in blocks_dir.glob("block__*.csv") if path.is_file()),
        key=lambda path: path.name,
    )


def _row_identity_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    """Build a stable identity key for deduplicating merged rows."""
    condition = (row.get("condition") or "").strip()
    model = (row.get("model") or "").strip()
    prompt_version = (row.get("prompt_version") or "").strip()
    temp_key = _normalize_temperature(row.get("temperature"))

    try:
        run_number = int(str(row.get("run_number", "")).strip())
    except ValueError:
        run_number = 0

    if condition and run_number > 0 and model and temp_key and prompt_version:
        return (
            "run",
            condition,
            str(run_number),
            model,
            temp_key,
            prompt_version,
        )

    request_id = (row.get("request_id") or "").strip()
    if request_id:
        return ("request_id", request_id)

    # Fallback identity for malformed legacy rows without stable run/request IDs.
    return (
        "row",
        condition,
        str(row.get("run_number", "")).strip(),
        model,
        temp_key,
        prompt_version,
        str(row.get("timestamp", "")).strip(),
        str(row.get("response_text", "")),
    )


def _row_sort_key(row: Dict[str, str]) -> Tuple[str, str, str, str, int, str]:
    """Deterministic ordering for merged rows before trial_id renumbering."""
    condition = (row.get("condition") or "").strip()
    model = (row.get("model") or "").strip()
    temp_key = _normalize_temperature(row.get("temperature"))
    prompt_version = (row.get("prompt_version") or "").strip()
    request_id = (row.get("request_id") or "").strip()

    try:
        run_number = int(str(row.get("run_number", "")).strip())
    except ValueError:
        run_number = 0

    return (model, temp_key, prompt_version, condition, run_number, request_id)


def _row_quality_score(row: Dict[str, str]) -> Tuple[int, int, int, int, int]:
    """Score rows so richer/more complete duplicates win during reconciliation."""
    has_request_id = int(bool((row.get("request_id") or "").strip()))
    has_timestamp = int(bool((row.get("timestamp") or "").strip()))
    has_wager = int(bool(str(row.get("parsed_wager", "")).strip()))
    has_risk_profile = int(bool(str(row.get("risk_profile", "")).strip()))
    response_length = len(str(row.get("response_text", "")))
    return (has_request_id, has_timestamp, has_wager, has_risk_profile, response_length)


def _parse_positive_run_number(value: Any) -> int:
    """Parse run_number and return a positive integer or 0 when invalid."""
    try:
        run_number = int(str(value).strip())
    except ValueError:
        return 0
    return run_number if run_number > 0 else 0


def _block_selector_key(row: Dict[str, str]) -> Optional[Tuple[str, str, str]]:
    """Return a block selector key (model/temp/prompt_version) for a row."""
    model = (row.get("model") or "").strip()
    temp_key = _normalize_temperature(row.get("temperature"))
    prompt_version = (row.get("prompt_version") or "").strip()
    if model and temp_key and prompt_version:
        return (model, temp_key, prompt_version)
    return None


def _rounded_trial_target(min_trials_per_condition: int, round_trials_step: int) -> int:
    """Round down a per-condition trial count to a stable, human-friendly target."""
    if min_trials_per_condition <= 0:
        return 0
    if round_trials_step <= 1:
        return min_trials_per_condition
    if min_trials_per_condition < round_trials_step:
        return min_trials_per_condition
    return (min_trials_per_condition // round_trials_step) * round_trials_step


def _select_balanced_block_rows(
    block_rows: List[Dict[str, str]],
    round_trials_step: int,
) -> Tuple[List[Dict[str, str]], int, int]:
    """Keep an equal rounded number of runs per condition for merged outputs."""
    if not block_rows:
        return [], 0, 0

    best_row_by_condition_run: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in block_rows:
        condition = (row.get("condition") or "").strip()
        run_number = _parse_positive_run_number(row.get("run_number"))
        if not condition or run_number <= 0:
            continue
        key = (condition, run_number)
        current = best_row_by_condition_run.get(key)
        if current is None or _row_quality_score(row) > _row_quality_score(current):
            best_row_by_condition_run[key] = row

    if not best_row_by_condition_run:
        return list(block_rows), 0, 0

    run_numbers_by_condition: Dict[str, List[int]] = defaultdict(list)
    for condition, run_number in best_row_by_condition_run:
        run_numbers_by_condition[condition].append(run_number)

    min_trials_per_condition = min(len(run_numbers) for run_numbers in run_numbers_by_condition.values())
    target_trials_per_condition = _rounded_trial_target(
        min_trials_per_condition=min_trials_per_condition,
        round_trials_step=round_trials_step,
    )

    selected_condition_runs: Set[Tuple[str, int]] = set()
    if target_trials_per_condition > 0:
        for condition, run_numbers in run_numbers_by_condition.items():
            for run_number in sorted(set(run_numbers))[:target_trials_per_condition]:
                selected_condition_runs.add((condition, run_number))

    selected_rows: List[Dict[str, str]] = []
    malformed_rows: List[Dict[str, str]] = []
    for row in block_rows:
        condition = (row.get("condition") or "").strip()
        run_number = _parse_positive_run_number(row.get("run_number"))
        if not condition or run_number <= 0:
            malformed_rows.append(row)
            continue
        if (condition, run_number) in selected_condition_runs:
            selected_rows.append(row)

    # Preserve malformed legacy rows so reconciliation remains backward-compatible.
    selected_rows.extend(malformed_rows)
    return selected_rows, min_trials_per_condition, target_trials_per_condition


def merge_block_csvs(
    block_csv_paths: List[Path],
    output_path: Path,
    round_trials_step: int = 1,
) -> int:
    """Merge existing output + block CSVs, keeping one row per unique run."""
    if not block_csv_paths and (not output_path.exists() or output_path.stat().st_size == 0):
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_identity: Dict[Tuple[str, ...], Dict[str, str]] = {}

    def ingest_row(row: Dict[str, str]) -> None:
        identity = _row_identity_key(row)
        current = rows_by_identity.get(identity)
        if current is None or _row_quality_score(row) > _row_quality_score(current):
            rows_by_identity[identity] = row

    block_rows_by_selector: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for block_csv_path in block_csv_paths:
        if not block_csv_path.exists() or block_csv_path.stat().st_size == 0:
            continue
        with block_csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
                selector = _block_selector_key(normalized)
                if selector is None:
                    continue
                block_rows_by_selector[selector].append(normalized)

    selected_block_rows: List[Dict[str, str]] = []
    for selector in sorted(block_rows_by_selector.keys()):
        model, temp_key, prompt_version = selector
        block_rows = block_rows_by_selector[selector]
        if round_trials_step <= 1:
            selected_block_rows.extend(block_rows)
            continue
        filtered_rows, min_trials, target_trials = _select_balanced_block_rows(
            block_rows=block_rows,
            round_trials_step=round_trials_step,
        )
        selected_block_rows.extend(filtered_rows)
        dropped_rows = len(block_rows) - len(filtered_rows)
        if dropped_rows > 0 or min_trials != target_trials:
            print(
                "Adjusted merged block rows to maintain even per-condition trial counts: "
                f"model={model}, temperature={temp_key}, prompt_version={prompt_version}, "
                f"min_trials_per_condition={min_trials}, merged_trials_per_condition={target_trials}, "
                f"dropped_rows={max(dropped_rows, 0)}"
            )

    block_selectors = set(block_rows_by_selector.keys())
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
                selector = _block_selector_key(normalized)
                # When a block CSV exists, block rows are authoritative for that selector.
                if selector is not None and selector in block_selectors:
                    continue
                ingest_row(normalized)

    for row in selected_block_rows:
        ingest_row(row)

    merged_rows = 0
    merged_trial_id = 0
    ordered_rows = sorted(rows_by_identity.values(), key=_row_sort_key)

    with output_path.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for merged_row in ordered_rows:
            merged_trial_id += 1
            merged_row["trial_id"] = merged_trial_id
            writer.writerow(merged_row)
            merged_rows += 1

        output_handle.flush()
        os.fsync(output_handle.fileno())

    return merged_rows


def default_grouped_output_path(output_path: Path) -> Path:
    """Return the grouped-results companion path for a merged output CSV."""
    return output_path.with_name(f"{output_path.stem}_grouped{output_path.suffix}")


def reconcile_merged_outputs(
    output_path: Path,
    grouped_output_path: Optional[Path] = None,
    round_trials_step: int = 1,
) -> Tuple[int, int]:
    """Merge block CSVs into output_path and refresh grouped companion CSV."""
    blocks_dir = output_path.parent / "blocks"
    block_csv_paths = _list_block_csv_paths(blocks_dir)
    if not block_csv_paths:
        return 0, 0

    merged_rows = merge_block_csvs(
        block_csv_paths=block_csv_paths,
        output_path=output_path,
        round_trials_step=round_trials_step,
    )

    grouped_path = grouped_output_path or default_grouped_output_path(output_path)
    grouped_path.parent.mkdir(parents=True, exist_ok=True)
    if grouped_path.resolve(strict=False) != output_path.resolve(strict=False):
        shutil.copyfile(output_path, grouped_path)

    return len(block_csv_paths), merged_rows


def _load_resume_state(
    results_path: Path,
) -> Tuple[Set[Tuple[str, int, str, str, str]], int]:
    """Read existing results to determine completed runs and next trial_id."""
    completed_runs: Set[Tuple[str, int, str, str, str]] = set()
    max_trial_id = 0

    if not results_path.exists() or results_path.stat().st_size == 0:
        return completed_runs, max_trial_id

    with results_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = (row.get("condition") or "").strip()

            try:
                run_number = int(row.get("run_number", "0"))
            except ValueError:
                run_number = 0

            model = (row.get("model") or "").strip()
            temp_key = _normalize_temperature(row.get("temperature"))
            prompt_version = (row.get("prompt_version") or "").strip()

            try:
                trial_id = int(row.get("trial_id", "0"))
            except ValueError:
                trial_id = 0

            if condition and run_number > 0 and model and temp_key and prompt_version:
                completed_runs.add((condition, run_number, model, temp_key, prompt_version))

            max_trial_id = max(max_trial_id, trial_id)

    return completed_runs, max_trial_id


def run_experiment(
    conditions_path: Path,
    output_path: Path,
    n_trials: int,
    model: str,
    temperature: float,
    target_model: Optional[str] = None,
    target_temperature: Optional[float] = None,
    target_prompt_version: Optional[str] = None,
    sleep_seconds: float = 0.0,
    prompt_version: str = "absolute",
    shuffle: bool = False,
    max_retries: int = 5,
    seed: Optional[int] = None,
) -> None:
    """Execute the experiment and append each row immediately to disk."""
    if n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    if max_retries < 0:
        raise ValueError("max_retries must be zero or a positive integer")
    if not math.isfinite(temperature) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a finite number between 0 and 2")
    if target_temperature is not None and (
        not math.isfinite(target_temperature) or target_temperature < 0 or target_temperature > 2
    ):
        raise ValueError("target_temperature must be a finite number between 0 and 2")

    conditions = load_conditions(conditions_path)
    rng = random.Random(seed) if seed is not None else random.Random()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_runs, max_trial_id = _load_resume_state(output_path)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    temp_key = _normalize_temperature(temperature)
    target_model_value = target_model.strip() if target_model else model
    target_temperature_value = temperature if target_temperature is None else target_temperature
    target_temp_key = _normalize_temperature(target_temperature_value)
    target_prompt_version_value = (
        target_prompt_version.strip() if target_prompt_version else prompt_version
    )

    if (
        target_model is not None
        or target_temperature is not None
        or target_prompt_version is not None
    ) and (
        model != target_model_value
        or temp_key != target_temp_key
        or prompt_version != target_prompt_version_value
    ):
        raise ValueError(
            "Target filters must match --model, --temperature, and --prompt-version "
            "for safe resume of that experiment block."
        )

    completed_by_condition: Dict[str, Set[int]] = {
        str(condition["condition"]): set() for condition in conditions
    }
    for condition_name, run_number, row_model, row_temp_key, row_prompt_version in completed_runs:
        if (
            row_model == target_model_value
            and row_temp_key == target_temp_key
            and row_prompt_version == target_prompt_version_value
            and condition_name in completed_by_condition
            and 1 <= run_number <= n_trials
        ):
            completed_by_condition[condition_name].add(run_number)

    missing_by_condition: Dict[str, List[int]] = {}
    print(
        "Resume summary for block: "
        f"model={target_model_value}, temperature={target_temperature_value}, "
        f"prompt_version={target_prompt_version_value}"
    )
    for condition in conditions:
        condition_name = str(condition["condition"])
        completed_numbers = completed_by_condition[condition_name]
        missing_run_numbers = [
            run_number for run_number in range(1, n_trials + 1) if run_number not in completed_numbers
        ]
        missing_by_condition[condition_name] = missing_run_numbers

        missing_display = "none" if not missing_run_numbers else str(missing_run_numbers)
        print(
            f"  {condition_name}: {len(completed_numbers)} unique completed; "
            f"missing run_numbers={missing_display}"
        )

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_api_key:
        raise EnvironmentError(
            "Missing OPENROUTER_API_KEY in environment. "
            "Set it in your shell before running this script."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    trial_id = max_trial_id
    total_new = 0

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

        trial_plan: List[Tuple[Dict[str, object], int]] = []
        for condition in conditions:
            condition_name = str(condition["condition"])
            for run_number in missing_by_condition[condition_name]:
                trial_plan.append((condition, run_number))

        if shuffle:
            rng.shuffle(trial_plan)

        for condition, run_number in trial_plan:
            condition_name = str(condition["condition"])
            outcome_type = str(condition["outcome_type"])
            amount = int(condition["amount"])

            run_key = (condition_name, run_number, model, temp_key, prompt_version)
            if run_key in completed_runs:
                continue

            prompt_text = build_prompt(
                outcome_type=outcome_type,
                amount=amount,
                prompt_version=prompt_version,
            )
            response_text, request_id = call_model(
                client=client,
                prompt=prompt_text,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
            )
            parsed_wager, parsed_risk_profile, valid, valid_risk_profile, refusal_flag, parse_error_type = parse_response(response_text)
            log_wager = math.log(parsed_wager) if valid and parsed_wager is not None else ""

            trial_id += 1
            row = {
                "trial_id": trial_id,
                "condition": condition_name,
                "outcome_type": outcome_type,
                "amount": amount,
                "run_number": run_number,
                "model": model,
                "temperature": temperature,
                "prompt_version": prompt_version,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "parsed_wager": parsed_wager if parsed_wager is not None else "",
                "log_wager": log_wager,
                "valid": valid,
                "risk_profile": parsed_risk_profile if parsed_risk_profile is not None else "",
                "valid_risk_profile": valid_risk_profile,
                "refusal_flag": refusal_flag,
                "parse_error_type": parse_error_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
            }

            # Persist every trial row so interrupted runs can be resumed safely.
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

            completed_runs.add(run_key)
            total_new += 1

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"Completed. Added {total_new} new trial rows to {output_path}.")


def run_experiment_grid(
    conditions_path: Path,
    output_path: Path,
    n_trials: int,
    models: List[str],
    temperatures: List[float],
    target_model: Optional[str] = None,
    target_temperature: Optional[float] = None,
    target_prompt_version: Optional[str] = None,
    sleep_seconds: float = 0.0,
    prompt_version: str = "absolute",
    prompt_versions: Optional[List[str]] = None,
    shuffle: bool = False,
    max_retries: int = 5,
    seed: Optional[int] = None,
    max_workers: int = 1,
    round_trials_step: int = 1,
) -> None:
    """Run all model/temperature/prompt_version blocks, optionally in parallel."""
    if not models:
        raise ValueError("models must contain at least one model")
    if not temperatures:
        raise ValueError("temperatures must contain at least one temperature")
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    if round_trials_step <= 0:
        raise ValueError("round_trials_step must be a positive integer")

    versions = prompt_versions if prompt_versions is not None else [prompt_version]
    cleaned_versions = [version.strip() for version in versions if version and version.strip()]
    if not cleaned_versions:
        raise ValueError("prompt_versions must contain at least one non-empty value")

    target_model_value = target_model.strip() if target_model else None
    target_prompt_version_value = (
        target_prompt_version.strip() if target_prompt_version else None
    )
    target_temp_key = _normalize_temperature(target_temperature)

    block_specs: List[Tuple[str, float, str, Path]] = []
    for model in models:
        for temperature in temperatures:
            for block_prompt_version in cleaned_versions:
                if target_model_value is not None and model != target_model_value:
                    continue
                if (
                    target_temperature is not None
                    and _normalize_temperature(temperature) != target_temp_key
                ):
                    continue
                if (
                    target_prompt_version_value is not None
                    and block_prompt_version != target_prompt_version_value
                ):
                    continue

                block_output_path = build_block_output_path(
                    output_path=output_path,
                    model=model,
                    temperature=temperature,
                    prompt_version=block_prompt_version,
                )
                block_specs.append((model, temperature, block_prompt_version, block_output_path))

    deduped_block_specs: List[Tuple[str, float, str, Path]] = []
    seen_block_paths: Set[str] = set()
    for block_spec in block_specs:
        block_path_key = str(block_spec[3])
        if block_path_key in seen_block_paths:
            continue
        seen_block_paths.add(block_path_key)
        deduped_block_specs.append(block_spec)
    block_specs = deduped_block_specs

    block_specs.sort(
        key=lambda block: (block[0], _normalize_temperature(block[1]), block[2], block[3].name)
    )
    if not block_specs:
        print("No experiment blocks matched the requested filters.")
        return
    else:
        print(
            f"Running {len(block_specs)} block(s) with max_workers="
            f"{min(max_workers, len(block_specs))}."
        )
        print(
            "Write process: each block appends to its own CSV under "
            f"{output_path.parent / 'blocks'}, then all blocks are reconciled once "
            f"into {output_path} at the end."
        )

    def _run_single_block(block_spec: Tuple[str, float, str, Path]) -> None:
        block_model, block_temperature, block_prompt_version, block_output_path = block_spec
        print(
            "Starting block: "
            f"model={block_model}, temperature={block_temperature}, "
            f"prompt_version={block_prompt_version}, output={block_output_path}"
        )
        block_error: Optional[Exception] = None
        try:
            run_experiment(
                conditions_path=conditions_path,
                output_path=block_output_path,
                n_trials=n_trials,
                model=block_model,
                temperature=block_temperature,
                target_model=target_model,
                target_temperature=target_temperature,
                target_prompt_version=target_prompt_version,
                sleep_seconds=sleep_seconds,
                prompt_version=block_prompt_version,
                shuffle=shuffle,
                max_retries=max_retries,
                seed=seed,
            )
        except Exception as error:
            block_error = error
        if block_error is not None:
            raise block_error

    failed_blocks: List[Tuple[Tuple[str, float, str, Path], Exception]] = []
    if len(block_specs) == 1 or max_workers == 1:
        for block_spec in block_specs:
            try:
                _run_single_block(block_spec)
            except Exception as error:
                failed_blocks.append((block_spec, error))
                break
    elif block_specs:
        worker_count = min(max_workers, len(block_specs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_block = {
                executor.submit(_run_single_block, block_spec): block_spec
                for block_spec in block_specs
            }
            for future in as_completed(future_to_block):
                block_spec = future_to_block[future]
                try:
                    future.result()
                except Exception as error:
                    failed_blocks.append((block_spec, error))

    grouped_output_path = default_grouped_output_path(output_path)
    merged_block_count, merged_rows = reconcile_merged_outputs(
        output_path=output_path,
        grouped_output_path=grouped_output_path,
        round_trials_step=round_trials_step,
    )
    if merged_block_count == 0:
        blocks_dir = output_path.parent / "blocks"
        print(f"No block CSV files found in {blocks_dir}; skipping merge.")
    else:
        print(
            "Final reconciliation complete: "
            f"merged {merged_block_count} block file(s) into {output_path} "
            f"with {merged_rows} total row(s), and refreshed {grouped_output_path}."
        )

    if failed_blocks:
        summary_lines = []
        for block_spec, error in failed_blocks:
            block_model, block_temperature, block_prompt_version, _ = block_spec
            summary_lines.append(
                "  - "
                f"model={block_model}, temperature={block_temperature}, "
                f"prompt_version={block_prompt_version}: {error}"
            )
        raise RuntimeError(
            "One or more experiment blocks failed:\n" + "\n".join(summary_lines)
        ) from failed_blocks[0][1]


def main() -> None:
    """Parse CLI arguments and run the experiment."""
    parser = argparse.ArgumentParser(description="Run realization-effect LLM experiment.")
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("configs/realization_effect/conditions.csv"),
        help="Path to conditions CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/results.csv"),
        help="Path to output results CSV file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of runs per condition (default: 100).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="One or more models to query (via OpenRouter).",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="One or more sampling temperatures passed to the API.",
    )
    # Backward-compatible single-value aliases.
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Optional resume filter for model.",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=None,
        help="Optional resume filter for temperature.",
    )
    parser.add_argument(
        "--target-prompt-version",
        type=str,
        default=None,
        help="Optional resume filter for prompt version.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="absolute",
        help=(
            "Prompt wording version (default: absolute). Use 'generation' to "
            "generate SAE inference prompts instead of behavioral results."
        ),
    )
    parser.add_argument(
        "--prompt-versions",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of prompt wording versions to run as separate blocks.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle condition order before running.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for transient API failures (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of experiment blocks to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--round-trials-step",
        type=int,
        default=1,
        help=(
            "Round merged per-condition trial counts down to this step size "
            "(default: 1, disabled; use values >1 to round down)."
        ),
    )
    parser.add_argument(
        "--generation-plan",
        type=Path,
        default=DEFAULT_GENERATION_PLAN,
        help="Prompt-generation plan JSON used when --prompt-version generation.",
    )
    parser.add_argument(
        "--generation-output",
        type=Path,
        default=None,
        help=(
            "Generated prompt CSV path. Defaults to "
            "experiments/emotion_activation/prompts/final/final_inference_prompts_v1.csv "
            "when --prompt-version generation."
        ),
    )
    parser.add_argument(
        "--generation-limit-jobs",
        type=int,
        default=None,
        help="Limit generation jobs for a small pilot.",
    )
    parser.add_argument(
        "--generation-pilot-all-cells",
        action="store_true",
        help=(
            "Run one representative job per generation cell and selected model. "
            "This samples the full taxonomy instead of the first expanded jobs."
        ),
    )
    parser.add_argument(
        "--generation-pilot-count-per-cell",
        type=int,
        default=1,
        help="Prompts per sampled cell when --generation-pilot-all-cells is set.",
    )
    parser.add_argument(
        "--generation-resume",
        action="store_true",
        help="Append missing prompt-generation batches to an existing CSV.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    args = parser.parse_args()

    prompt_versions = (
        args.prompt_versions if args.prompt_versions is not None else [args.prompt_version]
    )
    if GENERATION_PROMPT_VERSION in prompt_versions:
        if len(prompt_versions) != 1:
            raise SystemExit("--prompt-version generation cannot be mixed with behavioral prompt versions.")
        if args.model is not None:
            args.models = [args.model]
        _run_generation_prompt_mode(args)
        return

    models = args.models if args.models is not None else [args.model or "openai/o4-mini"]
    temperatures = (
        args.temperatures if args.temperatures is not None else [args.temperature or 1.0]
    )

    run_experiment_grid(
        conditions_path=args.conditions,
        output_path=args.output,
        n_trials=args.n_trials,
        models=models,
        temperatures=temperatures,
        target_model=args.target_model,
        target_temperature=args.target_temperature,
        target_prompt_version=args.target_prompt_version,
        sleep_seconds=args.sleep_seconds,
        prompt_version=args.prompt_version,
        prompt_versions=prompt_versions,
        shuffle=args.shuffle,
        max_retries=args.max_retries,
        seed=args.seed,
        max_workers=args.max_workers,
        round_trials_step=args.round_trials_step,
    )


if __name__ == "__main__":
    main()
