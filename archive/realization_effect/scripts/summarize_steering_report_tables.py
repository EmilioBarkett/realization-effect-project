#!/usr/bin/env python3
"""Create report-ready steering summary tables from a steering_eval CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _float(value: str) -> float:
    return float(value)


def _mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def _median(values: list[float]) -> float:
    return median(values) if values else float("nan")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _both_valid(row: dict[str, str]) -> bool:
    return _is_true(row["valid_amount"]) and _is_true(row["valid_risk_profile"])


def _strict_valid(row: dict[str, str]) -> bool:
    return _both_valid(row) and _is_true(row["response_exactly_two_integers"])


def _scale_summary(
    rows_by_scale: dict[float, list[dict[str, str]]],
    *,
    dataset_id: str,
    input_path: Path,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for scale in sorted(rows_by_scale):
        scale_rows = rows_by_scale[scale]
        both_valid = [row for row in scale_rows if _both_valid(row)]
        strict_valid = [row for row in scale_rows if _strict_valid(row)]
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "scale": int(scale) if scale.is_integer() else scale,
                "rows": len(scale_rows),
                "both_valid_rows": len(both_valid),
                "strict_valid_rows": len(strict_valid),
                "mean_amount_all_valid": _mean([_float(row["parsed_amount"]) for row in both_valid]),
                "mean_risk_all_valid": _mean([_float(row["risk_profile"]) for row in both_valid]),
                "mean_amount_strict_valid": _mean([_float(row["parsed_amount"]) for row in strict_valid]),
                "mean_risk_strict_valid": _mean([_float(row["risk_profile"]) for row in strict_valid]),
                "input_path": str(input_path),
            }
        )
    return summary_rows


def _matched_deltas(
    rows_by_scale: dict[float, list[dict[str, str]]],
    *,
    baseline_scale: float,
    dataset_id: str,
) -> list[dict[str, object]]:
    baseline_by_prompt = {row["prompt_id"]: row for row in rows_by_scale[baseline_scale]}
    delta_rows: list[dict[str, object]] = []
    for scale in sorted(rows_by_scale):
        if scale == baseline_scale:
            continue
        matched = [
            (row, baseline_by_prompt[row["prompt_id"]])
            for row in rows_by_scale[scale]
            if row["prompt_id"] in baseline_by_prompt
        ]
        subsets = {
            "all_valid_matched": [
                (row, base)
                for row, base in matched
                if _both_valid(row) and _both_valid(base)
            ],
            "strict_valid_matched": [
                (row, base)
                for row, base in matched
                if _strict_valid(row) and _strict_valid(base)
            ],
        }
        for subset, subset_rows in subsets.items():
            amount_deltas = [
                _float(row["parsed_amount"]) - _float(base["parsed_amount"])
                for row, base in subset_rows
            ]
            risk_deltas = [
                _float(row["risk_profile"]) - _float(base["risk_profile"])
                for row, base in subset_rows
            ]
            delta_rows.append(
                {
                    "dataset_id": dataset_id,
                    "scale": int(scale) if scale.is_integer() else scale,
                    "comparison": "steered_minus_in_run_unsteered_baseline",
                    "subset": subset,
                    "n": len(subset_rows),
                    "mean_amount_delta": _mean(amount_deltas),
                    "median_amount_delta": _median(amount_deltas),
                    "mean_risk_delta": _mean(risk_deltas),
                    "median_risk_delta": _median(risk_deltas),
                }
            )
    return delta_rows


def _compliance(
    rows_by_scale: dict[float, list[dict[str, str]]],
    *,
    dataset_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_specs = [
        ("scale", lambda row: "all"),
        ("domain", lambda row: row["domain"]),
        ("source_llm", lambda row: row["source_llm"]),
        ("pair_role", lambda row: row["pair_role"]),
    ]
    for scale in sorted(rows_by_scale):
        scale_rows = rows_by_scale[scale]
        for group_type, group_fn in group_specs:
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in scale_rows:
                grouped[group_fn(row)].append(row)
            for group in sorted(grouped):
                group_rows = grouped[group]
                strict_count = sum(_strict_valid(row) for row in group_rows)
                noncompliant = len(group_rows) - strict_count
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "scale": int(scale) if scale.is_integer() else scale,
                        "group_type": group_type,
                        "group": group,
                        "rows": len(group_rows),
                        "strict_two_integer_rows": strict_count,
                        "noncompliant_rows": noncompliant,
                        "noncompliance_rate": noncompliant / len(group_rows),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", default="gemma_layer18_train_only_realization_steering")
    parser.add_argument("--baseline-scale", type=float, default=0.0)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    rows_by_scale: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_scale[float(row["steering_scale"])].append(row)
    if args.baseline_scale not in rows_by_scale:
        raise ValueError(f"Missing baseline scale {args.baseline_scale}")

    _write_rows(
        args.output_dir / "steering_scale_summary.csv",
        [
            "dataset_id",
            "scale",
            "rows",
            "both_valid_rows",
            "strict_valid_rows",
            "mean_amount_all_valid",
            "mean_risk_all_valid",
            "mean_amount_strict_valid",
            "mean_risk_strict_valid",
            "input_path",
        ],
        _scale_summary(rows_by_scale, dataset_id=args.dataset_id, input_path=args.input),
    )
    _write_rows(
        args.output_dir / "steering_matched_deltas.csv",
        [
            "dataset_id",
            "scale",
            "comparison",
            "subset",
            "n",
            "mean_amount_delta",
            "median_amount_delta",
            "mean_risk_delta",
            "median_risk_delta",
        ],
        _matched_deltas(rows_by_scale, baseline_scale=args.baseline_scale, dataset_id=args.dataset_id),
    )
    _write_rows(
        args.output_dir / "compliance_by_scale_source.csv",
        [
            "dataset_id",
            "scale",
            "group_type",
            "group",
            "rows",
            "strict_two_integer_rows",
            "noncompliant_rows",
            "noncompliance_rate",
        ],
        _compliance(rows_by_scale, dataset_id=args.dataset_id),
    )


if __name__ == "__main__":
    main()
