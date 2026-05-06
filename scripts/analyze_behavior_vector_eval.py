#!/usr/bin/env python3
"""Summarize free-generation behavior evals for activation-vector experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _float_or_none(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _group_means(pair_deltas: list[dict[str, str]], key: str) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pair_deltas:
        groups[row[key]].append(row)
    return {
        group: {
            "pairs": len(rows),
            "mean_amount_delta": _mean([float(row["amount_delta"]) for row in rows]),
            "mean_risk_delta": _mean([float(row["risk_delta"]) for row in rows]),
            "mean_projection_delta": _mean([float(row["projection_delta"]) for row in rows]),
        }
        for group, rows in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze free-generated behavior eval outputs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/final/activation_vectors/realization_vector_v1_layer18/behavior_eval.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/final/activation_vectors/realization_vector_v1_layer18/"
            "evaluation/behavior_eval_summary.json"
        ),
    )
    parser.add_argument(
        "--pair-output",
        type=Path,
        default=Path(
            "results/final/activation_vectors/realization_vector_v1_layer18/"
            "evaluation/behavior_pair_deltas.csv"
        ),
    )
    args = parser.parse_args()

    with args.input.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_pair: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][row["pair_role"]] = row

    pair_deltas: list[dict[str, str]] = []
    for pair_id, pair_rows in sorted(by_pair.items()):
        paper = pair_rows.get("paper_open")
        realized = pair_rows.get("realized_closed")
        if paper is None or realized is None:
            continue
        if not (
            _is_true(paper.get("valid_amount", ""))
            and _is_true(realized.get("valid_amount", ""))
            and _is_true(paper.get("valid_risk_profile", ""))
            and _is_true(realized.get("valid_risk_profile", ""))
        ):
            continue
        paper_amount = float(paper["parsed_amount"])
        realized_amount = float(realized["parsed_amount"])
        paper_risk = float(paper["risk_profile"])
        realized_risk = float(realized["risk_profile"])
        paper_projection = float(paper["projection"])
        realized_projection = float(realized["projection"])
        pair_deltas.append(
            {
                "pair_id": pair_id,
                "domain": paper.get("domain", ""),
                "outcome_valence": paper.get("outcome_valence", ""),
                "amount_bucket": paper.get("amount_bucket", ""),
                "source_llm": paper.get("source_llm", ""),
                "paper_projection": str(paper_projection),
                "realized_projection": str(realized_projection),
                "projection_delta": str(realized_projection - paper_projection),
                "paper_amount": str(paper_amount),
                "realized_amount": str(realized_amount),
                "amount_delta": str(realized_amount - paper_amount),
                "paper_risk": str(paper_risk),
                "realized_risk": str(realized_risk),
                "risk_delta": str(realized_risk - paper_risk),
            }
        )

    valid_amount_rows = [row for row in rows if _is_true(row.get("valid_amount", ""))]
    valid_risk_rows = [row for row in rows if _is_true(row.get("valid_risk_profile", ""))]
    both_valid_rows = [
        row
        for row in rows
        if _is_true(row.get("valid_amount", "")) and _is_true(row.get("valid_risk_profile", ""))
    ]
    projection_amount = [
        (float(row["projection"]), float(row["parsed_amount"]))
        for row in both_valid_rows
        if _float_or_none(row.get("projection", "")) is not None
    ]
    projection_risk = [
        (float(row["projection"]), float(row["risk_profile"]))
        for row in both_valid_rows
        if _float_or_none(row.get("projection", "")) is not None
    ]

    amount_deltas = [float(row["amount_delta"]) for row in pair_deltas]
    risk_deltas = [float(row["risk_delta"]) for row in pair_deltas]
    projection_deltas = [float(row["projection_delta"]) for row in pair_deltas]
    summary = {
        "input": str(args.input),
        "rows": len(rows),
        "valid_amount_rows": len(valid_amount_rows),
        "valid_risk_rows": len(valid_risk_rows),
        "both_valid_rows": len(both_valid_rows),
        "complete_valid_pairs": len(pair_deltas),
        "mean_amount_delta_realized_minus_paper": _mean(amount_deltas),
        "mean_risk_delta_realized_minus_paper": _mean(risk_deltas),
        "mean_projection_delta_realized_minus_paper": _mean(projection_deltas),
        "row_projection_amount_correlation": _pearson(
            [item[0] for item in projection_amount],
            [item[1] for item in projection_amount],
        ),
        "row_projection_risk_correlation": _pearson(
            [item[0] for item in projection_risk],
            [item[1] for item in projection_risk],
        ),
        "pair_projection_delta_amount_delta_correlation": _pearson(projection_deltas, amount_deltas),
        "pair_projection_delta_risk_delta_correlation": _pearson(projection_deltas, risk_deltas),
        "by_domain": _group_means(pair_deltas, "domain"),
        "by_outcome_valence": _group_means(pair_deltas, "outcome_valence"),
        "by_amount_bucket": _group_means(pair_deltas, "amount_bucket"),
        "by_source_llm": _group_means(pair_deltas, "source_llm"),
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.pair_output.parent.mkdir(parents=True, exist_ok=True)
    with args.pair_output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(pair_deltas[0].keys()) if pair_deltas else ["pair_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(pair_deltas)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
