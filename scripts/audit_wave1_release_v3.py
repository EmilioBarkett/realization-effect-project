#!/usr/bin/env python3
"""Audit Wave 1 v3 review samples before full API generation.

The audit is deliberately independent of the generator's acceptance path.  It
checks task-role/split contracts, response instructions, category metadata,
collateral lexical separation, and the exact review coverage expected for the
release.  A nonzero exit status means the review sample must be regenerated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from construct_benchmark.config import load_construct_spec
from construct_benchmark.generation import downstream_prompt_text_issues
from construct_benchmark.prompts import load_prompt_records, validate_prompt_records


ROOT = Path(__file__).resolve().parents[1]
WAVE1 = (
    "realization_account_closure",
    "evidence_diagnosticity",
    "source_reliability",
    "persistence_continuation",
)
SUPPLEMENT_CONSTRUCTS = WAVE1[:3]


def _spec(construct_id: str, collateral: bool):
    suffix = "v3" if collateral else "v2"
    return load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs" / f"{construct_id}_{suffix}.json"
    )


def _records(directory: Path, construct_id: str) -> list[Any]:
    path = directory / f"{construct_id}.csv"
    if not path.exists():
        raise ValueError(f"Missing review output: {path}")
    return load_prompt_records(path)


def _audit_review(
    primary_dir: Path,
    collateral_dir: Path,
    *,
    primary_expected_count: int = 1,
    collateral_expected_count: int = 1,
) -> dict[str, Any]:
    failures: list[str] = []
    summary: dict[str, Any] = {"primary": {}, "collateral": {}}
    for construct_id in SUPPLEMENT_CONSTRUCTS:
        rows = _records(primary_dir, construct_id)
        spec = _spec(construct_id, collateral=False)
        validate_prompt_records(rows, {construct_id: spec}, require_all_splits=False)
        counts: dict[str, int] = {}
        issues: list[str] = []
        for row in rows:
            counts[row.split] = counts.get(row.split, 0) + 1
            issues.extend(
                f"{row.prompt_id}: {issue}"
                for issue in downstream_prompt_text_issues(row.prompt_text)
            )
        expected = {
            "behavior_eval": primary_expected_count,
            "steering_eval": primary_expected_count,
            "calibration": primary_expected_count,
        }
        if counts != expected:
            failures.append(f"{construct_id} primary review counts={counts}, expected={expected}")
        if issues:
            failures.extend(issues)
        summary["primary"][construct_id] = {"record_count": len(rows), "by_split": counts, "issues": issues}

    collateral_forbidden = {
        "realization_account_closure": ("account", "settled", "realization"),
        "evidence_diagnosticity": ("diagnostic", "hypothesis", "posterior", "evidence"),
        "source_reliability": ("source", "authority", "testimony", "reliable"),
        "persistence_continuation": ("goal", "abandon", "setback", "continue"),
    }
    for construct_id in WAVE1:
        rows = _records(collateral_dir, construct_id)
        spec = _spec(construct_id, collateral=True)
        validate_prompt_records(rows, {construct_id: spec}, require_all_splits=False)
        issues: list[str] = []
        for row in rows:
            if row.split != "collateral_eval" or row.prompt_role != "collateral":
                issues.append(f"{row.prompt_id}: wrong collateral split/role")
            issues.extend(
                f"{row.prompt_id}: {issue}"
                for issue in downstream_prompt_text_issues(row.prompt_text)
            )
            lowered = row.prompt_text.casefold()
            for term in collateral_forbidden[construct_id]:
                if term in lowered:
                    issues.append(f"{row.prompt_id}: target anchor present: {term}")
            task_metadata = dict(row.metadata.get("task_metadata") or {})
            expected_fields = {"correct_option", "difficulty", "option_order", "domain_family"}
            if set(task_metadata) != expected_fields:
                issues.append(f"{row.prompt_id}: collateral metadata fields={sorted(task_metadata)}")
        if len(rows) != collateral_expected_count:
            failures.append(
                f"{construct_id} collateral review count={len(rows)}, expected={collateral_expected_count}"
            )
        if collateral_expected_count > 1:
            correct_counts = {
                value: sum(
                    row.metadata.get("task_metadata", {}).get("correct_option") == value
                    for row in rows
                )
                for value in (1, 2)
            }
            expected_correct = collateral_expected_count // 2
            if correct_counts != {1: expected_correct, 2: expected_correct}:
                failures.append(f"{construct_id} collateral correct-option balance={correct_counts}")
        if issues:
            failures.extend(issues)
        summary["collateral"][construct_id] = {"record_count": len(rows), "issues": issues}
    return {"pass": not failures, "failures": failures, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-review-dir", type=Path, required=True)
    parser.add_argument("--collateral-review-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-expected-count", type=int, default=1)
    parser.add_argument("--collateral-expected-count", type=int, default=1)
    args = parser.parse_args()
    if args.primary_expected_count < 1 or args.collateral_expected_count < 1:
        raise SystemExit("expected counts must be positive")
    report = _audit_review(
        args.primary_review_dir.resolve(),
        args.collateral_review_dir.resolve(),
        primary_expected_count=args.primary_expected_count,
        collateral_expected_count=args.collateral_expected_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
