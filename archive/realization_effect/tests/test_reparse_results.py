from __future__ import annotations

import csv
import math

from realization_effect.runner import RESULT_FIELDS
from realization_effect.reparse_results import reparse_csv


def test_reparse_csv_repairs_stored_parser_values(tmp_path) -> None:
    path = tmp_path / "results.csv"
    row = {field: "" for field in RESULT_FIELDS}
    row.update(
        {
            "trial_id": "1",
            "response_text": "Line 1: 500\nLine 2: 3",
            "parsed_wager": "1",
            "log_wager": "0.0",
            "valid": "True",
            "risk_profile": "2",
            "valid_risk_profile": "True",
            "refusal_flag": "False",
            "parse_error_type": "",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    summary = reparse_csv(path, write=True, backup=False)

    assert summary.rows == 1
    assert summary.changed_rows == 1
    with path.open("r", newline="", encoding="utf-8") as handle:
        repaired = next(csv.DictReader(handle))
    assert repaired["parsed_wager"] == "500"
    assert math.isclose(float(repaired["log_wager"]), math.log(500))
    assert repaired["risk_profile"] == "3"
