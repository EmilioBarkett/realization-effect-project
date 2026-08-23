#!/usr/bin/env python3
"""Refresh parsed behavior columns from saved free-generation responses."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from realization_effect.runner import parse_response  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Reparse behavior eval response_text columns in place.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/final/activation_vectors/realization_vector_v1_layer18/behavior_eval.csv"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.input
    with args.input.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {args.input}")
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row in rows:
        amount, risk, valid_amount, valid_risk, refusal, error_type = parse_response(row.get("response_text", ""))
        row["parsed_amount"] = "" if amount is None else str(amount)
        row["valid_amount"] = str(valid_amount)
        row["risk_profile"] = "" if risk is None else str(risk)
        row["valid_risk_profile"] = str(valid_risk)
        row["refusal_flag"] = str(refusal)
        row["parse_error_type"] = error_type

    output.parent.mkdir(parents=True, exist_ok=True)
    if output == args.input:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False, dir=str(output.parent)) as tmp:
            tmp_path = Path(tmp.name)
            writer = csv.DictWriter(tmp, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(output)
    else:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    print(f"reparsed {len(rows)} rows -> {output}")


if __name__ == "__main__":
    main()
