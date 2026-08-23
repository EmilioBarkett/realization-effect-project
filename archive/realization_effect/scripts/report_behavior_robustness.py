#!/usr/bin/env python3
"""Robustness report for paired behavior-vector eval results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _trimmed(values: list[float], fraction: float) -> list[float]:
    ordered = sorted(values)
    trim_count = int(len(ordered) * fraction)
    if trim_count == 0:
        return ordered
    return ordered[trim_count:-trim_count]


def _winsorized(values: list[float], fraction: float) -> list[float]:
    ordered = sorted(values)
    trim_count = int(len(ordered) * fraction)
    if trim_count == 0:
        return ordered
    lower = ordered[trim_count]
    upper = ordered[-trim_count - 1]
    return [min(max(value, lower), upper) for value in values]


def _bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    sample_size = len(values)
    means = [
        sum(values[rng.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(iterations)
    ]
    means.sort()
    return {
        "lower_95": means[int(iterations * 0.025)],
        "median": means[int(iterations * 0.5)],
        "upper_95": means[int(iterations * 0.975)],
    }


def _sign_test_two_sided(values: list[float]) -> dict[str, float | int | None]:
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    nonzero = positive + negative
    if nonzero == 0:
        p_value = None
    else:
        smaller = min(positive, negative)
        p_value = 2 * sum(math.comb(nonzero, k) * 0.5**nonzero for k in range(smaller + 1))
        p_value = min(1.0, p_value)
    return {
        "positive": positive,
        "negative": negative,
        "zero": sum(value == 0 for value in values),
        "nonzero": nonzero,
        "two_sided_p": p_value,
    }


def _delta_summary(values: list[float], *, bootstrap_iterations: int, seed: int) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": _quantile(values, 0.5),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "quantiles": {
            "p01": _quantile(values, 0.01),
            "p05": _quantile(values, 0.05),
            "p10": _quantile(values, 0.10),
            "p25": _quantile(values, 0.25),
            "p75": _quantile(values, 0.75),
            "p90": _quantile(values, 0.90),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
        },
        "trimmed_means": {
            "trim_05": _mean(_trimmed(values, 0.05)),
            "trim_10": _mean(_trimmed(values, 0.10)),
            "trim_20": _mean(_trimmed(values, 0.20)),
        },
        "winsorized_means": {
            "winsor_05": _mean(_winsorized(values, 0.05)),
            "winsor_10": _mean(_winsorized(values, 0.10)),
            "winsor_20": _mean(_winsorized(values, 0.20)),
        },
        "bootstrap_mean_ci": _bootstrap_mean_ci(
            values,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "sign_test": _sign_test_two_sided(values),
        "large_delta_counts": {
            "abs_ge_50": sum(abs(value) >= 50 for value in values),
            "abs_ge_100": sum(abs(value) >= 100 for value in values),
            "abs_ge_250": sum(abs(value) >= 250 for value in values),
            "abs_ge_500": sum(abs(value) >= 500 for value in values),
        },
    }


def _group_summary(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(float(row["amount_delta"]))
    summaries = {}
    for group, values in sorted(grouped.items()):
        counts = Counter("positive" if value > 0 else "negative" if value < 0 else "zero" for value in values)
        summaries[group] = {
            "pairs": len(values),
            "mean": _mean(values),
            "median": _quantile(values, 0.5),
            "positive": counts["positive"],
            "negative": counts["negative"],
            "zero": counts["zero"],
        }
    return summaries


def _remove_extremes(values: list[float]) -> dict[str, float]:
    output = {}
    for count in (5, 10, 20, 30):
        without_top_positive = sorted(values, reverse=True)[count:]
        without_top_absolute = sorted(values, key=abs, reverse=True)[count:]
        output[f"remove_top_{count}_positive"] = _mean(without_top_positive)
        output[f"remove_top_{count}_absolute"] = _mean(without_top_absolute)
    return output


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_extreme_pairs(rows: list[dict[str, str]], output: Path, count: int) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["amount_delta"]))
    extremes = sorted_rows[:count] + list(reversed(sorted_rows[-count:]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["pair_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(extremes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report whether behavior deltas are robust or outlier-driven.")
    parser.add_argument("--pair-deltas", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--extreme-output", type=Path, default=None)
    parser.add_argument("--extreme-count", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = _load_rows(args.pair_deltas)
    amount_deltas = [float(row["amount_delta"]) for row in rows]
    risk_deltas = [float(row["risk_delta"]) for row in rows]
    positive_sum = sum(value for value in amount_deltas if value > 0)
    negative_sum = sum(value for value in amount_deltas if value < 0)
    net_sum = sum(amount_deltas)
    top_positive = sorted((value for value in amount_deltas if value > 0), reverse=True)[:10]

    report = {
        "input": str(args.pair_deltas),
        "amount_delta": _delta_summary(
            amount_deltas,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "risk_delta": _delta_summary(
            risk_deltas,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        ),
        "outlier_sensitivity": _remove_extremes(amount_deltas),
        "sum_positive_amount_deltas": positive_sum,
        "sum_negative_amount_deltas": negative_sum,
        "net_amount_delta": net_sum,
        "share_of_net_from_top_10_positive_deltas": sum(top_positive) / net_sum if net_sum else None,
        "by_domain": _group_summary(rows, "domain"),
        "by_outcome_valence": _group_summary(rows, "outcome_valence"),
        "by_amount_bucket": _group_summary(rows, "amount_bucket"),
        "by_source_llm": _group_summary(rows, "source_llm"),
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.extreme_output:
        _write_extreme_pairs(rows, args.extreme_output, args.extreme_count)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
