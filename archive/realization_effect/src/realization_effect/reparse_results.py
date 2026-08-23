#!/usr/bin/env python3
"""Audit or repair parsed answer columns in realization-effect result CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .runner import parse_response


PARSED_FIELDS = {
    "parsed_wager",
    "log_wager",
    "valid",
    "risk_profile",
    "valid_risk_profile",
    "refusal_flag",
    "parse_error_type",
}


@dataclass(frozen=True)
class ReparseSummary:
    path: Path
    rows: int
    changed_rows: int
    label_like_rows: int


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return str(number)


def _is_changed(existing: dict[str, str], repaired: dict[str, str]) -> bool:
    for field in PARSED_FIELDS:
        if _normalize_cell(existing.get(field, "")) != _normalize_cell(repaired.get(field, "")):
            return True
    return False


def _label_like(response_text: str) -> bool:
    lower = response_text.lower().lstrip()
    return lower.startswith(("line 1", "1.", "wager:")) or "line 1:" in lower


def _repaired_values(response_text: str) -> dict[str, str]:
    wager, risk_profile, valid, valid_risk_profile, refusal_flag, parse_error_type = parse_response(
        response_text
    )
    return {
        "parsed_wager": str(wager) if wager is not None else "",
        "log_wager": str(math.log(wager)) if valid and wager is not None else "",
        "valid": _bool_text(valid),
        "risk_profile": str(risk_profile) if risk_profile is not None else "",
        "valid_risk_profile": _bool_text(valid_risk_profile),
        "refusal_flag": _bool_text(refusal_flag),
        "parse_error_type": parse_error_type,
    }


def reparse_csv(path: Path, *, write: bool = False, backup: bool = True) -> ReparseSummary:
    """Audit or rewrite one result CSV using the current parser."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row")
        missing = ({"response_text"} | PARSED_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        fieldnames = list(reader.fieldnames)
        repaired_rows: list[dict[str, str]] = []
        rows = 0
        changed_rows = 0
        label_like_rows = 0

        for row in reader:
            rows += 1
            response_text = row.get("response_text", "")
            if _label_like(response_text):
                label_like_rows += 1

            repaired = _repaired_values(response_text)
            if _is_changed(row, repaired):
                changed_rows += 1
                if write:
                    row.update(repaired)
            repaired_rows.append(row)

    if write and changed_rows:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copyfile(path, backup_path)

        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(repaired_rows)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)

    return ReparseSummary(
        path=path,
        rows=rows,
        changed_rows=changed_rows,
        label_like_rows=label_like_rows,
    )


def expand_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.csv")))
        else:
            expanded.append(path)
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit or repair parsed answer columns in realization-effect result CSVs."
    )
    parser.add_argument("paths", type=Path, nargs="+", help="Result CSV files or directories to scan")
    parser.add_argument("--write", action="store_true", help="Rewrite parsed columns in place")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak files when using --write",
    )
    args = parser.parse_args()

    totals = ReparseSummary(path=Path("<total>"), rows=0, changed_rows=0, label_like_rows=0)
    for path in expand_paths(args.paths):
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        summary = reparse_csv(path, write=args.write, backup=not args.no_backup)
        print(
            f"{summary.path}: rows={summary.rows} "
            f"changed_rows={summary.changed_rows} label_like_rows={summary.label_like_rows}"
        )
        totals = ReparseSummary(
            path=totals.path,
            rows=totals.rows + summary.rows,
            changed_rows=totals.changed_rows + summary.changed_rows,
            label_like_rows=totals.label_like_rows + summary.label_like_rows,
        )

    print(
        f"TOTAL: rows={totals.rows} changed_rows={totals.changed_rows} "
        f"label_like_rows={totals.label_like_rows}"
    )


if __name__ == "__main__":
    main()
