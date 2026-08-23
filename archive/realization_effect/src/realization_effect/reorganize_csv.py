#!/usr/bin/env python3
"""Reorganize CSV rows by model, temperature, prompt version, or any other columns.

Examples:
  python scripts/reorganize_csv.py results/results_merged.csv \
      --group-by model temperature prompt_version \
      --sort-by run_number trial_id \
      --renumber-column trial_id \
      --output results/results_merged_grouped.csv

  python scripts/reorganize_csv.py results/results_merged.csv \
      --group-by model temperature \
      --split-dir results/grouped_by_model_temp
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group/reorder a CSV by columns of your choice and optionally split "
            "rows into one CSV per group."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        required=True,
        metavar="COLUMN",
        help="Columns used to group rows (for example: model temperature prompt_version).",
    )
    parser.add_argument(
        "--sort-by",
        nargs="*",
        default=[],
        metavar="COLUMN",
        help="Optional extra sort columns within each group.",
    )
    parser.add_argument(
        "--descending",
        nargs="*",
        default=[],
        metavar="COLUMN",
        help="Columns that should be sorted in descending order.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path for regrouped rows. "
            "Default: <input_stem>_grouped.csv in the same folder."
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="If set, also writes one CSV file per group into this folder.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding used for read/write (default: utf-8).",
    )
    parser.add_argument(
        "--renumber-column",
        default=None,
        metavar="COLUMN",
        help=(
            "Optional column to reassign as sequential row numbers after regrouping "
            "(for example: trial_id)."
        ),
    )
    parser.add_argument(
        "--renumber-start",
        type=int,
        default=1,
        metavar="N",
        help="Starting value for --renumber-column (default: 1).",
    )
    return parser.parse_args()


def load_rows(path: Path, encoding: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open("r", newline="", encoding=encoding) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV must include a header row.")

        rows = list(reader)
        return reader.fieldnames, rows


def validate_columns(requested: Iterable[str], fieldnames: Sequence[str]) -> None:
    missing = [name for name in requested if name not in fieldnames]
    if missing:
        available = ", ".join(fieldnames)
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Unknown column(s): {missing_text}. Available columns: {available}"
        )


def smart_sort_value(value: str) -> Tuple[int, object]:
    text = (value or "").strip()
    if text == "":
        return (2, "")

    lower = text.lower()
    if lower in {"true", "false"}:
        return (0, lower == "true")

    try:
        return (0, float(text))
    except ValueError:
        return (1, lower)


def reorder_rows(
    rows: List[Dict[str, str]],
    group_by: Sequence[str],
    sort_by: Sequence[str],
    descending: Sequence[str],
) -> List[Dict[str, str]]:
    order = list(group_by) + list(sort_by)
    if not order:
        return list(rows)

    descending_set = set(descending)
    reordered = list(rows)

    # Stable sorts from least-significant to most-significant key.
    for column in reversed(order):
        reordered.sort(
            key=lambda row, c=column: smart_sort_value(row.get(c, "")),
            reverse=column in descending_set,
        )

    return reordered


def renumber_rows(
    rows: Sequence[Dict[str, str]],
    renumber_column: str,
    renumber_start: int,
) -> None:
    for index, row in enumerate(rows, start=renumber_start):
        row[renumber_column] = str(index)


def default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_grouped{input_csv.suffix}")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_filename_piece(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "blank"


def write_split_files(
    split_dir: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Dict[str, str]],
    group_by: Sequence[str],
    encoding: str,
) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        key = tuple(row.get(column, "") for column in group_by)
        grouped[key].append(row)

    manifest_path = split_dir / "group_manifest.csv"
    with manifest_path.open("w", newline="", encoding=encoding) as handle:
        manifest_fields = list(group_by) + ["row_count", "file"]
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()

        for key, group_rows in grouped.items():
            parts = [f"{col}-{sanitize_filename_piece(val)}" for col, val in zip(group_by, key)]
            filename = "__".join(parts) + ".csv"
            group_file = split_dir / filename
            write_csv(group_file, fieldnames, group_rows, encoding)

            manifest_row = {col: val for col, val in zip(group_by, key)}
            manifest_row["row_count"] = str(len(group_rows))
            manifest_row["file"] = filename
            writer.writerow(manifest_row)

    return len(grouped)


def main() -> None:
    args = parse_args()

    fieldnames, rows = load_rows(args.input_csv, args.encoding)

    requested_columns = list(args.group_by) + list(args.sort_by) + list(args.descending)
    if args.renumber_column:
        requested_columns.append(args.renumber_column)
    validate_columns(requested_columns, fieldnames)

    reordered = reorder_rows(
        rows=rows,
        group_by=args.group_by,
        sort_by=args.sort_by,
        descending=args.descending,
    )
    if args.renumber_column:
        renumber_rows(
            rows=reordered,
            renumber_column=args.renumber_column,
            renumber_start=args.renumber_start,
        )

    output_path = args.output or default_output_path(args.input_csv)
    write_csv(output_path, fieldnames, reordered, args.encoding)

    print(f"Wrote regrouped CSV: {output_path} ({len(reordered)} rows)")
    if args.renumber_column:
        end_value = args.renumber_start + len(reordered) - 1
        print(
            f"Renumbered '{args.renumber_column}' from {args.renumber_start} to {end_value}"
        )

    if args.split_dir:
        group_count = write_split_files(
            split_dir=args.split_dir,
            fieldnames=fieldnames,
            rows=reordered,
            group_by=args.group_by,
            encoding=args.encoding,
        )
        print(f"Wrote {group_count} grouped files to: {args.split_dir}")


if __name__ == "__main__":
    main()
