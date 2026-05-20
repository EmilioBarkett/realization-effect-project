"""Condition-table loading for realization-effect experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def load_conditions(csv_path: Path) -> list[dict[str, Any]]:
    """Load and validate the condition table from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Conditions file not found: {csv_path}")

    conditions: list[dict[str, Any]] = []
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
