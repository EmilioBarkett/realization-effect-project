#!/usr/bin/env python3
"""Partition mixed experiment results into canonical and legacy CSVs.

Legacy rows are identified as either:
1) condition labels not present in the active conditions.csv,
2) rows belonging to a (model, temperature, prompt_version) block whose
   condition coverage does not match the active conditions.csv schema, or
3) rows belonging to blocks whose prompt text does not match the active
   prompt templates for their (condition, prompt_version).
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


REQUIRED_FIELDS = [
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


def load_expected_conditions(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "condition" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'condition' column")
        conditions = [str(row["condition"]).strip() for row in reader if str(row["condition"]).strip()]
    if not conditions:
        raise ValueError(f"No conditions found in {path}")
    return conditions


def load_condition_specs(path: Path) -> Dict[str, Tuple[str, int]]:
    """Load condition -> (outcome_type, amount) from conditions CSV."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"condition", "outcome_type", "amount"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{path} must contain columns: condition,outcome_type,amount"
            )

        specs: Dict[str, Tuple[str, int]] = {}
        for row in reader:
            condition = str(row.get("condition", "")).strip()
            outcome_type = str(row.get("outcome_type", "")).strip()
            amount_raw = str(row.get("amount", "")).strip()
            if not condition:
                continue
            try:
                amount = int(amount_raw)
            except ValueError as error:
                raise ValueError(
                    f"Invalid integer amount '{amount_raw}' for condition '{condition}'"
                ) from error
            specs[condition] = (outcome_type, amount)

    if not specs:
        raise ValueError(f"No condition specs found in {path}")
    return specs


def normalize_prompt_text(prompt_text: str) -> str:
    """Normalize whitespace so formatting-only differences do not trigger drift."""
    return re.sub(r"\s+", " ", (prompt_text or "").strip())


def load_prompt_builders() -> Tuple[Dict[str, object], object]:
    """Load prompt builders from run_experiment to compare prompt templates."""
    try:
        from .runner import PROMPT_BUILDERS, build_prompt
    except Exception as error:  # pragma: no cover - defensive for local environments.
        raise RuntimeError(
            "Unable to import prompt builders from run_experiment.py. "
            "Install dependencies and run with the project interpreter."
        ) from error
    return PROMPT_BUILDERS, build_prompt


def build_expected_prompt_map(
    *,
    condition_specs: Dict[str, Tuple[str, int]],
    prompt_versions: Set[str],
) -> Dict[Tuple[str, str], str]:
    """Build normalized expected prompt text by (condition, prompt_version)."""
    prompt_builders, build_prompt_fn = load_prompt_builders()
    supported_prompt_versions = set(prompt_builders.keys())
    unknown_versions = sorted(prompt_versions - supported_prompt_versions)
    if unknown_versions:
        joined = ", ".join(unknown_versions)
        raise ValueError(
            f"Unsupported prompt_version value(s) in input data: {joined}"
        )

    expected: Dict[Tuple[str, str], str] = {}
    for condition, (outcome_type, amount) in condition_specs.items():
        for prompt_version in sorted(prompt_versions):
            prompt_text = build_prompt_fn(
                outcome_type=outcome_type,
                amount=amount,
                prompt_version=prompt_version,
            )
            expected[(condition, prompt_version)] = normalize_prompt_text(prompt_text)
    return expected


def block_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        (row.get("model") or "").strip(),
        (row.get("temperature") or "").strip(),
        (row.get("prompt_version") or "").strip(),
    )


def partition_results(
    input_csv: Path,
    conditions_csv: Path,
    canonical_csv: Path,
    legacy_csv: Path,
    report_path: Path,
    enforce_prompt_template_check: bool = True,
) -> None:
    expected_conditions = load_expected_conditions(conditions_csv)
    condition_specs = load_condition_specs(conditions_csv)
    expected_set = set(expected_conditions)

    rows: List[Dict[str, str]] = []
    block_conditions: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    observed_prompt_versions: Set[str] = set()

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_csv} is missing a header row")
        missing_required = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing_required:
            raise ValueError(f"{input_csv} missing required columns: {missing_required}")

        for row in reader:
            normalized = {field: row.get(field, "") for field in REQUIRED_FIELDS}
            rows.append(normalized)
            condition = (normalized.get("condition") or "").strip()
            if condition:
                block_conditions[block_key(normalized)].add(condition)
            prompt_version = (normalized.get("prompt_version") or "").strip()
            if prompt_version:
                observed_prompt_versions.add(prompt_version)

    expected_prompt_map: Dict[Tuple[str, str], str] = {}
    if enforce_prompt_template_check and observed_prompt_versions:
        expected_prompt_map = build_expected_prompt_map(
            condition_specs=condition_specs,
            prompt_versions=observed_prompt_versions,
        )

    legacy_blocks_schema: Set[Tuple[str, str, str]] = set()
    for key, conditions in block_conditions.items():
        if len(conditions) != len(expected_set) or not conditions.issubset(expected_set):
            legacy_blocks_schema.add(key)

    legacy_blocks_prompt: Set[Tuple[str, str, str]] = set()
    if expected_prompt_map:
        for row in rows:
            condition = (row.get("condition") or "").strip()
            prompt_version = (row.get("prompt_version") or "").strip()
            key = block_key(row)
            expected_prompt = expected_prompt_map.get((condition, prompt_version))
            if expected_prompt is None:
                legacy_blocks_prompt.add(key)
                continue

            observed_prompt = normalize_prompt_text(row.get("prompt_text", ""))
            if observed_prompt != expected_prompt:
                legacy_blocks_prompt.add(key)

    canonical_rows: List[Dict[str, str]] = []
    legacy_rows: List[Dict[str, str]] = []
    for row in rows:
        condition = (row.get("condition") or "").strip()
        key = block_key(row)
        is_legacy = (
            condition not in expected_set
            or key in legacy_blocks_schema
            or key in legacy_blocks_prompt
        )
        if is_legacy:
            legacy_rows.append(row)
        else:
            canonical_rows.append(row)

    canonical_csv.parent.mkdir(parents=True, exist_ok=True)
    legacy_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with canonical_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(canonical_rows)

    with legacy_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(legacy_rows)

    legacy_blocks_all = legacy_blocks_schema | legacy_blocks_prompt
    sorted_legacy_blocks = sorted(legacy_blocks_all, key=lambda item: (item[0], item[1], item[2]))
    sorted_prompt_drift_blocks = sorted(
        legacy_blocks_prompt, key=lambda item: (item[0], item[1], item[2])
    )
    sorted_schema_blocks = sorted(
        legacy_blocks_schema, key=lambda item: (item[0], item[1], item[2])
    )
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Input rows: {len(rows)}\n")
        handle.write(f"Canonical rows: {len(canonical_rows)}\n")
        handle.write(f"Legacy rows: {len(legacy_rows)}\n")
        handle.write(f"Expected condition count: {len(expected_set)}\n")
        handle.write(f"Legacy blocks: {len(sorted_legacy_blocks)}\n")
        handle.write(f"Legacy blocks (schema mismatch): {len(sorted_schema_blocks)}\n")
        handle.write(f"Legacy blocks (prompt mismatch): {len(sorted_prompt_drift_blocks)}\n")
        handle.write(
            f"Prompt template check enabled: {'yes' if enforce_prompt_template_check else 'no'}\n"
        )
        handle.write("\nLegacy block keys (model | temperature | prompt_version):\n")
        for model, temperature, prompt_version in sorted_legacy_blocks:
            handle.write(f"- {model} | {temperature} | {prompt_version}\n")

        if sorted_prompt_drift_blocks:
            handle.write(
                "\nPrompt-mismatch block keys (model | temperature | prompt_version):\n"
            )
            for model, temperature, prompt_version in sorted_prompt_drift_blocks:
                handle.write(f"- {model} | {temperature} | {prompt_version}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition mixed results into canonical and legacy CSVs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/results.csv"),
        help="Merged input results CSV.",
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("configs/realization_effect/conditions.csv"),
        help="Active conditions CSV used to define canonical schema.",
    )
    parser.add_argument(
        "--canonical-output",
        type=Path,
        default=Path("results/results.csv"),
        help="Output path for canonical rows (default rewrites results/results.csv in place).",
    )
    parser.add_argument(
        "--legacy-output",
        type=Path,
        default=Path("results/legacy/results_legacy.csv"),
        help="Output path for legacy rows.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("results/legacy/partition_report.txt"),
        help="Output path for partition summary report.",
    )
    parser.add_argument(
        "--skip-prompt-template-check",
        action="store_true",
        help="Disable prompt-template drift detection.",
    )
    args = parser.parse_args()

    partition_results(
        input_csv=args.input,
        conditions_csv=args.conditions,
        canonical_csv=args.canonical_output,
        legacy_csv=args.legacy_output,
        report_path=args.report_output,
        enforce_prompt_template_check=not args.skip_prompt_template_check,
    )
    print(
        f"Wrote canonical={args.canonical_output} and legacy={args.legacy_output}; "
        f"summary={args.report_output}"
    )


if __name__ == "__main__":
    main()
