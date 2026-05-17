#!/usr/bin/env python3
"""Validate behavior-evaluation prompts before local or cloud model runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROMPT_PREFIX = "Read the following short scenario.\n\nScenario:\n"
PROMPT_SUFFIX = "\n\nDo not answer yet. Continue processing the scenario."
FIRST_PERSON_PATTERN = re.compile(r"\b(?:i|me|my|mine|we|our|ours)\b", re.IGNORECASE)
EXPLICIT_INTERNAL_LABELS = [
    "paper_open",
    "realized_closed",
    "realization",
    "vector",
    "dataset",
]


def _has_behavior_instruction(prompt_text: str) -> bool:
    normalized = re.sub(r"\s+", " ", prompt_text.lower())
    return (
        "two integers" in normalized
        and bool(re.search(r"\b1\s*(?:-|to)\s*1000\b", normalized))
        and bool(re.search(r"\b1\s*(?:-|to)\s*5\b", normalized))
    )


def _scenario(prompt_text: str) -> str:
    return prompt_text.removeprefix(PROMPT_PREFIX).removesuffix(PROMPT_SUFFIX)


def _word_count(prompt_text: str) -> int:
    return len(re.findall(r"\b\w+\b", _scenario(prompt_text)))


def _add_issue(issues: list[dict[str, str]], row: dict[str, str], issue: str, detail: str = "") -> None:
    issues.append(
        {
            "prompt_id": row.get("prompt_id", ""),
            "pair_id": row.get("pair_id", ""),
            "pair_role": row.get("pair_role", ""),
            "domain": row.get("domain", ""),
            "source_llm": row.get("source_llm", ""),
            "issue": issue,
            "detail": detail,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate behavior_eval prompts for cloud behavior runs.")
    parser.add_argument(
        "--prompt-csv",
        type=Path,
        default=Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("results/final/activation_vectors/behavior_prompt_validation.json"),
    )
    parser.add_argument("--max-pair-word-delta", type=int, default=25)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    with args.prompt_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "behavior_eval"]
    if not rows:
        raise ValueError(f"No behavior_eval rows found in {args.prompt_csv}")

    issues: list[dict[str, str]] = []
    seen_prompt_ids: set[str] = set()
    duplicate_prompt_ids: set[str] = set()
    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        prompt_id = row.get("prompt_id", "")
        prompt_text = row.get("prompt_text", "")
        if prompt_id in seen_prompt_ids:
            duplicate_prompt_ids.add(prompt_id)
        seen_prompt_ids.add(prompt_id)
        by_pair[row.get("pair_id", "")].append(row)

        if not prompt_text.startswith(PROMPT_PREFIX):
            _add_issue(issues, row, "bad_prefix")
        if not prompt_text.endswith(PROMPT_SUFFIX):
            _add_issue(issues, row, "bad_suffix")
        if not _has_behavior_instruction(prompt_text):
            _add_issue(issues, row, "missing_behavior_instruction")
        scenario = _scenario(prompt_text)
        if FIRST_PERSON_PATTERN.search(scenario):
            _add_issue(issues, row, "first_person")
        lower_scenario = scenario.lower()
        leaked = [label for label in EXPLICIT_INTERNAL_LABELS if label in lower_scenario]
        if leaked:
            _add_issue(issues, row, "internal_label_leak", ",".join(leaked))

    for prompt_id in sorted(duplicate_prompt_ids):
        issues.append(
            {
                "prompt_id": prompt_id,
                "pair_id": "",
                "pair_role": "",
                "domain": "",
                "source_llm": "",
                "issue": "duplicate_prompt_id",
                "detail": "",
            }
        )

    for pair_id, pair_rows in by_pair.items():
        roles = Counter(row.get("pair_role", "") for row in pair_rows)
        if roles != {"paper_open": 1, "realized_closed": 1}:
            for row in pair_rows:
                _add_issue(issues, row, "bad_pair_roles", json.dumps(dict(roles), sort_keys=True))
            continue
        paper = next(row for row in pair_rows if row.get("pair_role") == "paper_open")
        realized = next(row for row in pair_rows if row.get("pair_role") == "realized_closed")
        for key in ("domain", "outcome_valence", "amount_bucket", "source_llm"):
            if paper.get(key, "") != realized.get(key, ""):
                _add_issue(issues, paper, "pair_metadata_mismatch", key)
                _add_issue(issues, realized, "pair_metadata_mismatch", key)
        word_delta = abs(_word_count(paper["prompt_text"]) - _word_count(realized["prompt_text"]))
        if word_delta > args.max_pair_word_delta:
            _add_issue(issues, paper, "pair_word_delta", str(word_delta))
            _add_issue(issues, realized, "pair_word_delta", str(word_delta))

    summary = {
        "prompt_csv": str(args.prompt_csv),
        "behavior_rows": len(rows),
        "pairs": len(by_pair),
        "issues": len(issues),
        "issues_by_type": dict(Counter(issue["issue"] for issue in issues)),
        "rows_by_domain": dict(Counter(row.get("domain", "") for row in rows)),
        "rows_by_source_llm": dict(Counter(row.get("source_llm", "") for row in rows)),
        "sample_issues": issues[:25],
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if args.fail_on_issues and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
