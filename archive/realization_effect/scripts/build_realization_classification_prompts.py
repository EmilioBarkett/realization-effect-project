#!/usr/bin/env python3
"""Build a realization-classification prompt set from the paired prompt CSV."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import PROMPT_SUFFIX  # noqa: E402


CLASSIFICATION_INSTRUCTION = (
    "Task: Determine whether the described outcome is already realized "
    "(cashed out or position closed) or still paper (open/unrealized).\n"
    "Answer now.\n"
    "Return exactly one label: REALIZED or PAPER.\n"
    "Label:"
)


def _classification_prompt(prompt_text: str) -> str:
    prompt = prompt_text
    instruction_marker = "Answer now."
    if instruction_marker in prompt:
        prompt = prompt.split(instruction_marker, maxsplit=1)[0].rstrip()
    if prompt.endswith(PROMPT_SUFFIX):
        prompt = prompt.removesuffix(PROMPT_SUFFIX).rstrip()
    prompt = re.sub(
        r"Answer with exactly two integers on separate lines:[^.]*\.",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    return f"{prompt}\n\n{CLASSIFICATION_INSTRUCTION}\n"


def _expected_label(pair_role: str) -> str:
    role = pair_role.strip().lower()
    if role == "realized_closed":
        return "REALIZED"
    if role == "paper_open":
        return "PAPER"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a classification variant of realization_vector_v1 prompts.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "experiments/activation_analysis/prompts/activation_vectors/"
            "realization_vector_v1_realization_classification.csv"
        ),
    )
    args = parser.parse_args()

    with args.input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"No columns found in {args.input_csv}")
        fieldnames = list(reader.fieldnames)

    for extra in ("classification_label_expected", "classification_prompt_variant"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    transformed: list[dict[str, str]] = []
    for row in rows:
        out = dict(row)
        out["prompt_text"] = _classification_prompt(row.get("prompt_text", ""))
        out["behavior_target"] = "realization_classification"
        out["classification_label_expected"] = _expected_label(row.get("pair_role", ""))
        out["classification_prompt_variant"] = "single_label_realized_vs_paper_v1"
        transformed.append(out)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(transformed)

    print(
        {
            "input": str(args.input_csv),
            "output": str(args.output_csv),
            "rows": len(transformed),
        }
    )


if __name__ == "__main__":
    main()
