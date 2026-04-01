#!/usr/bin/env python3
"""Generate and export all prompt texts for inspection.

Writes one row per (condition × prompt_version) combination to a CSV so you
can review exactly what each model sees before running the experiment.

Usage:
  python generate_prompts.py                          # all versions, stdout
  python generate_prompts.py --output prompts.csv     # save to file
  python generate_prompts.py --version qualitative    # one version only
  python generate_prompts.py --conditions my_conds.csv --output prompts.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from run_experiment import PROMPT_BUILDERS, SYSTEM_PROMPT, build_prompt, load_conditions

OUTPUT_FIELDS = [
    "condition",
    "outcome_type",
    "amount",
    "prompt_version",
    "system_prompt",
    "user_prompt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all prompt texts to CSV.")
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("conditions.csv"),
        help="Path to conditions CSV (default: conditions.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to stdout.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        choices=list(PROMPT_BUILDERS.keys()),
        help="Restrict to one prompt version. Default: all versions.",
    )
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    versions = [args.version] if args.version else list(PROMPT_BUILDERS.keys())

    rows = []
    for condition in conditions:
        for version in versions:
            user_prompt = build_prompt(
                outcome_type=str(condition["outcome_type"]),
                amount=int(condition["amount"]),
                prompt_version=version,
            )
            rows.append(
                {
                    "condition": condition["condition"],
                    "outcome_type": condition["outcome_type"],
                    "amount": condition["amount"],
                    "prompt_version": version,
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                }
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} prompts to {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
