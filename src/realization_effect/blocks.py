"""Block CSV naming, resume, merge, and reconciliation helpers."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULT_FIELDS = [
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


def _normalize_temperature(value: Any) -> str:
    """Normalize temperature to a stable string for resume keys."""
    try:
        return format(float(value), ".12g")
    except (TypeError, ValueError):
        return ""


def _slug_for_filename(value: str) -> str:
    """Convert arbitrary text to a filesystem-friendly token."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "value"


def build_block_output_path(
    output_path: Path,
    model: str,
    temperature: float,
    prompt_version: str,
) -> Path:
    """Return a deterministic per-block CSV path under results/blocks/."""
    temp_key = _normalize_temperature(temperature)
    block_key = f"{model}|{temp_key}|{prompt_version}"
    key_hash = hashlib.sha1(block_key.encode("utf-8")).hexdigest()[:10]
    filename = (
        "block__"
        f"model-{_slug_for_filename(model)}__"
        f"temp-{_slug_for_filename(temp_key)}__"
        f"prompt-{_slug_for_filename(prompt_version)}__"
        f"{key_hash}.csv"
    )
    return output_path.parent / "blocks" / filename


def _list_block_csv_paths(blocks_dir: Path) -> list[Path]:
    """List block CSVs in deterministic filename order."""
    if not blocks_dir.exists():
        return []
    return sorted(
        (path for path in blocks_dir.glob("block__*.csv") if path.is_file()),
        key=lambda path: path.name,
    )


def _row_identity_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Build a stable identity key for deduplicating merged rows."""
    condition = (row.get("condition") or "").strip()
    model = (row.get("model") or "").strip()
    prompt_version = (row.get("prompt_version") or "").strip()
    temp_key = _normalize_temperature(row.get("temperature"))

    try:
        run_number = int(str(row.get("run_number", "")).strip())
    except ValueError:
        run_number = 0

    if condition and run_number > 0 and model and temp_key and prompt_version:
        return (
            "run",
            condition,
            str(run_number),
            model,
            temp_key,
            prompt_version,
        )

    request_id = (row.get("request_id") or "").strip()
    if request_id:
        return ("request_id", request_id)

    return (
        "row",
        condition,
        str(row.get("run_number", "")).strip(),
        model,
        temp_key,
        prompt_version,
        str(row.get("timestamp", "")).strip(),
        str(row.get("response_text", "")),
    )


def _row_sort_key(row: dict[str, str]) -> tuple[str, str, str, str, int, str]:
    """Deterministic ordering for merged rows before trial_id renumbering."""
    condition = (row.get("condition") or "").strip()
    model = (row.get("model") or "").strip()
    temp_key = _normalize_temperature(row.get("temperature"))
    prompt_version = (row.get("prompt_version") or "").strip()
    request_id = (row.get("request_id") or "").strip()

    try:
        run_number = int(str(row.get("run_number", "")).strip())
    except ValueError:
        run_number = 0

    return (model, temp_key, prompt_version, condition, run_number, request_id)


def _row_quality_score(row: dict[str, str]) -> tuple[int, int, int, int, int]:
    """Score rows so richer/more complete duplicates win during reconciliation."""
    has_request_id = int(bool((row.get("request_id") or "").strip()))
    has_timestamp = int(bool((row.get("timestamp") or "").strip()))
    has_wager = int(bool(str(row.get("parsed_wager", "")).strip()))
    has_risk_profile = int(bool(str(row.get("risk_profile", "")).strip()))
    response_length = len(str(row.get("response_text", "")))
    return (has_request_id, has_timestamp, has_wager, has_risk_profile, response_length)


def _parse_positive_run_number(value: Any) -> int:
    """Parse run_number and return a positive integer or 0 when invalid."""
    try:
        run_number = int(str(value).strip())
    except ValueError:
        return 0
    return run_number if run_number > 0 else 0


def _block_selector_key(row: dict[str, str]) -> tuple[str, str, str] | None:
    """Return a block selector key (model/temp/prompt_version) for a row."""
    model = (row.get("model") or "").strip()
    temp_key = _normalize_temperature(row.get("temperature"))
    prompt_version = (row.get("prompt_version") or "").strip()
    if model and temp_key and prompt_version:
        return (model, temp_key, prompt_version)
    return None


def _rounded_trial_target(min_trials_per_condition: int, round_trials_step: int) -> int:
    """Round down a per-condition trial count to a stable, human-friendly target."""
    if min_trials_per_condition <= 0:
        return 0
    if round_trials_step <= 1:
        return min_trials_per_condition
    if min_trials_per_condition < round_trials_step:
        return min_trials_per_condition
    return (min_trials_per_condition // round_trials_step) * round_trials_step


def _select_balanced_block_rows(
    block_rows: list[dict[str, str]],
    round_trials_step: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Keep an equal rounded number of runs per condition for merged outputs."""
    if not block_rows:
        return [], 0, 0

    best_row_by_condition_run: dict[tuple[str, int], dict[str, str]] = {}
    for row in block_rows:
        condition = (row.get("condition") or "").strip()
        run_number = _parse_positive_run_number(row.get("run_number"))
        if not condition or run_number <= 0:
            continue
        key = (condition, run_number)
        current = best_row_by_condition_run.get(key)
        if current is None or _row_quality_score(row) > _row_quality_score(current):
            best_row_by_condition_run[key] = row

    if not best_row_by_condition_run:
        return list(block_rows), 0, 0

    run_numbers_by_condition: dict[str, list[int]] = defaultdict(list)
    for condition, run_number in best_row_by_condition_run:
        run_numbers_by_condition[condition].append(run_number)

    min_trials_per_condition = min(len(run_numbers) for run_numbers in run_numbers_by_condition.values())
    target_trials_per_condition = _rounded_trial_target(
        min_trials_per_condition=min_trials_per_condition,
        round_trials_step=round_trials_step,
    )

    selected_condition_runs: set[tuple[str, int]] = set()
    if target_trials_per_condition > 0:
        for condition, run_numbers in run_numbers_by_condition.items():
            for run_number in sorted(set(run_numbers))[:target_trials_per_condition]:
                selected_condition_runs.add((condition, run_number))

    selected_rows: list[dict[str, str]] = []
    malformed_rows: list[dict[str, str]] = []
    for row in block_rows:
        condition = (row.get("condition") or "").strip()
        run_number = _parse_positive_run_number(row.get("run_number"))
        if not condition or run_number <= 0:
            malformed_rows.append(row)
            continue
        if (condition, run_number) in selected_condition_runs:
            selected_rows.append(row)

    selected_rows.extend(malformed_rows)
    return selected_rows, min_trials_per_condition, target_trials_per_condition


def merge_block_csvs(
    block_csv_paths: list[Path],
    output_path: Path,
    round_trials_step: int = 1,
) -> int:
    """Merge existing output + block CSVs, keeping one row per unique run."""
    if not block_csv_paths and (not output_path.exists() or output_path.stat().st_size == 0):
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_identity: dict[tuple[str, ...], dict[str, str]] = {}

    def ingest_row(row: dict[str, str]) -> None:
        identity = _row_identity_key(row)
        current = rows_by_identity.get(identity)
        if current is None or _row_quality_score(row) > _row_quality_score(current):
            rows_by_identity[identity] = row

    block_rows_by_selector: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for block_csv_path in block_csv_paths:
        if not block_csv_path.exists() or block_csv_path.stat().st_size == 0:
            continue
        with block_csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
                selector = _block_selector_key(normalized)
                if selector is None:
                    continue
                block_rows_by_selector[selector].append(normalized)

    selected_block_rows: list[dict[str, str]] = []
    for selector in sorted(block_rows_by_selector.keys()):
        model, temp_key, prompt_version = selector
        block_rows = block_rows_by_selector[selector]
        if round_trials_step <= 1:
            selected_block_rows.extend(block_rows)
            continue
        filtered_rows, min_trials, target_trials = _select_balanced_block_rows(
            block_rows=block_rows,
            round_trials_step=round_trials_step,
        )
        selected_block_rows.extend(filtered_rows)
        dropped_rows = len(block_rows) - len(filtered_rows)
        if dropped_rows > 0 or min_trials != target_trials:
            print(
                "Adjusted merged block rows to maintain even per-condition trial counts: "
                f"model={model}, temperature={temp_key}, prompt_version={prompt_version}, "
                f"min_trials_per_condition={min_trials}, merged_trials_per_condition={target_trials}, "
                f"dropped_rows={max(dropped_rows, 0)}"
            )

    block_selectors = set(block_rows_by_selector.keys())
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
                selector = _block_selector_key(normalized)
                if selector is not None and selector in block_selectors:
                    continue
                ingest_row(normalized)

    for row in selected_block_rows:
        ingest_row(row)

    merged_rows = 0
    merged_trial_id = 0
    ordered_rows = sorted(rows_by_identity.values(), key=_row_sort_key)

    with output_path.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for merged_row in ordered_rows:
            merged_trial_id += 1
            merged_row["trial_id"] = str(merged_trial_id)
            writer.writerow(merged_row)
            merged_rows += 1

        output_handle.flush()
        os.fsync(output_handle.fileno())

    return merged_rows


def default_grouped_output_path(output_path: Path) -> Path:
    """Return the grouped-results companion path for a merged output CSV."""
    return output_path.with_name(f"{output_path.stem}_grouped{output_path.suffix}")


def reconcile_merged_outputs(
    output_path: Path,
    grouped_output_path: Path | None = None,
    round_trials_step: int = 1,
) -> tuple[int, int]:
    """Merge block CSVs into output_path and refresh grouped companion CSV."""
    blocks_dir = output_path.parent / "blocks"
    block_csv_paths = _list_block_csv_paths(blocks_dir)
    if not block_csv_paths:
        return 0, 0

    merged_rows = merge_block_csvs(
        block_csv_paths=block_csv_paths,
        output_path=output_path,
        round_trials_step=round_trials_step,
    )

    grouped_path = grouped_output_path or default_grouped_output_path(output_path)
    grouped_path.parent.mkdir(parents=True, exist_ok=True)
    if grouped_path.resolve(strict=False) != output_path.resolve(strict=False):
        shutil.copyfile(output_path, grouped_path)

    return len(block_csv_paths), merged_rows


def load_resume_state(
    results_path: Path,
) -> tuple[set[tuple[str, int, str, str, str]], int]:
    """Read existing results to determine completed runs and next trial_id."""
    completed_runs: set[tuple[str, int, str, str, str]] = set()
    max_trial_id = 0

    if not results_path.exists() or results_path.stat().st_size == 0:
        return completed_runs, max_trial_id

    with results_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = (row.get("condition") or "").strip()

            try:
                run_number = int(row.get("run_number", "0"))
            except ValueError:
                run_number = 0

            model = (row.get("model") or "").strip()
            temp_key = _normalize_temperature(row.get("temperature"))
            prompt_version = (row.get("prompt_version") or "").strip()

            try:
                trial_id = int(row.get("trial_id", "0"))
            except ValueError:
                trial_id = 0

            if condition and run_number > 0 and model and temp_key and prompt_version:
                completed_runs.add((condition, run_number, model, temp_key, prompt_version))

            max_trial_id = max(max_trial_id, trial_id)

    return completed_runs, max_trial_id
