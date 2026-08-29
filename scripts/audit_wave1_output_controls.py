#!/usr/bin/env python3
"""Audit Wave 1 steering output accessibility and collateral-control coverage.

The audit consumes an existing, manifest-backed steering JSONL.  It performs
no inference and never loads model weights.  Output accessibility and parser
compliance are measured from the registered downstream task.  Unrelated
behavior is reported separately: if no registered collateral task is present,
the result is explicitly ``not_collected`` rather than being inferred from
same-task compliance or response length.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

_SRC = Path(__file__).resolve().parents[1] / "src"
import sys

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.behavior import (  # noqa: E402
    orient_primary_outcome,
    parse_behavior_output,
    primary_outcome,
)
from construct_benchmark.config import load_construct_spec  # noqa: E402
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402


SCHEMA_VERSION = "0.1.0"
COLLATERAL_KEY_FRAGMENTS = ("collateral", "unrelated")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Steering output does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path} line {line_number}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object on {path} line {line_number}.")
            rows.append(row)
    return rows


def _manifest_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(raw_path.suffix + ".manifest.json")


def _validate_manifest(raw_path: Path, rows: list[dict[str, Any]], construct_id: str) -> dict[str, Any]:
    manifest_path = _manifest_path(raw_path)
    if not manifest_path.is_file():
        raise ValueError(f"Missing adjacent steering manifest: {manifest_path}")
    manifest = _load_object(manifest_path)
    if manifest.get("manifest_type") != "construct_steering_output":
        raise ValueError(f"Unexpected steering manifest type in {manifest_path}.")
    if manifest.get("construct_id") != construct_id:
        raise ValueError(f"Steering manifest construct_id does not match {construct_id}.")
    if manifest.get("complete") is not True:
        raise ValueError("Output accessibility audit requires a complete steering manifest.")
    expected = manifest.get("expected_record_count")
    completed = manifest.get("completed_record_count")
    if not isinstance(expected, int) or completed != expected or len(rows) != expected:
        raise ValueError(
            f"Steering output/manifest count mismatch: rows={len(rows)}, expected={expected}, completed={completed}."
        )
    raw_hash = manifest.get("raw_generations_sha256")
    if not raw_hash or file_sha256(raw_path) != raw_hash:
        raise ValueError("Steering output does not match the manifest raw_generations_sha256.")
    record_ids = [row.get("record_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in record_ids):
        raise ValueError("Steering output contains a row without record_id.")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Steering output contains duplicate record_id values.")
    return manifest


def _contains_collateral_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if any(fragment in key_text for fragment in COLLATERAL_KEY_FRAGMENTS):
                return True
            if _contains_collateral_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_collateral_key(item) for item in value)
    return False


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def audit_output(
    *,
    raw_path: Path,
    construct_spec_path: Path,
    minimum_primary_valid_rate: float = 0.95,
    collateral_summary_path: Path | None = None,
) -> dict[str, Any]:
    spec = load_construct_spec(construct_spec_path)
    rows = _read_jsonl(raw_path)
    manifest = _validate_manifest(raw_path, rows, spec.construct_id)

    group_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "parser_valid": 0,
            "primary_valid": 0,
            "nonempty_text": 0,
            "output_characters": [],
            "primary_values": [],
        }
    )
    invalid_examples: list[dict[str, Any]] = []
    collateral_rows = 0
    injected_rows = 0
    injection_shift_values: list[float] = []
    intervention_timings: Counter[str] = Counter()
    all_doses: set[float] = set()
    direction_kinds: Counter[str] = Counter()

    for row in rows:
        direction_kind = str(row.get("direction_kind", "missing"))
        dose = float(row["dose"]) if row.get("dose") is not None else None
        group = f"{direction_kind}:{dose:g}" if dose is not None else direction_kind
        stats = group_stats[group]
        stats["rows"] += 1
        direction_kinds[direction_kind] += 1
        if dose is not None:
            all_doses.add(dose)
        text = row.get("output_text")
        if isinstance(text, str) and text.strip():
            stats["nonempty_text"] += 1
            stats["output_characters"].append(len(text))
        parser_id = str(row.get("parser_id") or spec.parsing_rules["parser_id"])
        metadata = dict(row.get("task_metadata") or {})
        task_id = str(row.get("task_id") or spec.independent_behavior_task["task_id"])
        parsed = parse_behavior_output(text or "", parser_id=parser_id, item_metadata=metadata, task_id=task_id)
        if parsed.valid:
            stats["parser_valid"] += 1
            try:
                outcome = primary_outcome(parsed, str(spec.independent_behavior_task["primary_outcome"]))
                directed = orient_primary_outcome(spec.construct_id, outcome, metadata)
                if directed is not None:
                    stats["primary_valid"] += 1
                    stats["primary_values"].append(float(directed))
                elif len(invalid_examples) < 5:
                    invalid_examples.append({"record_id": row.get("record_id"), "reason": "orientation returned None"})
            except (TypeError, ValueError) as exc:
                if len(invalid_examples) < 5:
                    invalid_examples.append({"record_id": row.get("record_id"), "reason": str(exc)})
        elif len(invalid_examples) < 5:
            invalid_examples.append({"record_id": row.get("record_id"), "reason": parsed.error or "invalid parser output"})

        if _contains_collateral_key(row):
            collateral_rows += 1
        if row.get("injection_applied") is True:
            injected_rows += 1
        if row.get("observed_shift") is not None:
            try:
                injection_shift_values.append(float(row["observed_shift"]))
            except (TypeError, ValueError):
                pass
        if row.get("intervention_timing") is not None:
            intervention_timings[str(row["intervention_timing"])] += 1

    groups: dict[str, Any] = {}
    for group, stats in sorted(group_stats.items()):
        rows_count = stats["rows"]
        groups[group] = {
            "rows": rows_count,
            "parser_valid_rows": stats["parser_valid"],
            "primary_valid_rows": stats["primary_valid"],
            "parser_valid_rate": stats["parser_valid"] / rows_count if rows_count else None,
            "primary_valid_rate": stats["primary_valid"] / rows_count if rows_count else None,
            "nonempty_text_rate": stats["nonempty_text"] / rows_count if rows_count else None,
            "mean_output_characters": _safe_mean(stats["output_characters"]),
            "mean_directed_primary_outcome": _safe_mean(stats["primary_values"]),
        }

    required_direction_kinds = {"target", "shuffled", "random"}
    missing_kinds = sorted(required_direction_kinds - set(direction_kinds))
    group_rates = [float(item["primary_valid_rate"]) for item in groups.values() if item["primary_valid_rate"] is not None]
    accessibility_pass = bool(
        not missing_kinds
        and group_rates
        and min(group_rates) >= minimum_primary_valid_rate
        and injected_rows > 0
        and intervention_timings
    )
    if collateral_summary_path is not None:
        collateral_payload = _load_object(collateral_summary_path)
        collateral_behavior = collateral_payload.get("behavior", collateral_payload)
        collateral_constructs = (
            collateral_behavior.get("constructs", {})
            if isinstance(collateral_behavior, Mapping)
            else {}
        )
        collateral_entry = collateral_constructs.get(spec.construct_id)
        if not isinstance(collateral_entry, Mapping):
            raise ValueError(
                f"Collateral summary {collateral_summary_path} has no entry for {spec.construct_id}."
            )
        collateral_total = int(collateral_entry.get("total_rows", 0))
        collateral_valid = int(collateral_entry.get("valid_primary_rows", 0))
        collateral_rate = (
            collateral_valid / collateral_total if collateral_total else None
        )
        collateral_status = (
            "not_collected"
            if collateral_total <= 0
            else (
                "collected"
                if collateral_rate is not None and collateral_rate >= minimum_primary_valid_rate
                else "collected_unusable"
            )
        )
        collateral = {
            "status": collateral_status,
            "source_summary": str(collateral_summary_path),
            "record_count": collateral_total,
            "valid_primary_rows": collateral_valid,
            "invalid_rows": int(collateral_entry.get("invalid_rows", 0)),
            "correctness_rate": collateral_rate,
            "minimum_valid_rate": minimum_primary_valid_rate,
            "interpretation": "Independent collateral-task correctness, scored separately from the construct task."
            if collateral_status == "collected"
            else "The independent collateral task was run, but its valid correctness denominator is below the registered audit threshold.",
        }
    elif collateral_rows:
        collateral = {
            "status": "collected_proxy_rows_present",
            "registered_collateral_rows": collateral_rows,
            "interpretation": "Collateral-shaped metadata or rows were present and require task-specific scoring review.",
        }
    else:
        collateral = {
            "status": "not_collected",
            "registered_collateral_rows": 0,
            "interpretation": "The frozen Wave 1 steering inventory contains no unrelated/collateral behavior task. Same-task compliance and response length are not substitutes.",
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "wave1_output_accessibility_collateral_audit",
        "status": "complete",
        "confirmatory": False,
        "construct_id": spec.construct_id,
        "raw_output": str(raw_path),
        "raw_output_sha256": file_sha256(raw_path),
        "output_manifest": str(_manifest_path(raw_path)),
        "source_manifest": {
            "complete": manifest.get("complete"),
            "expected_record_count": manifest.get("expected_record_count"),
            "source_confirmatory": bool(manifest.get("confirmatory", False)),
            "reclassified_non_confirmatory": True,
            "intervention_timing": manifest.get("intervention_timing"),
            "injection_layer": manifest.get("injection_layer"),
            "tracking_layers": manifest.get("tracking_layers"),
        },
        "row_count": len(rows),
        "direction_kinds": dict(sorted(direction_kinds.items())),
        "doses": sorted(all_doses),
        "groups": groups,
        "accessibility": {
            "minimum_primary_valid_rate": minimum_primary_valid_rate,
            "pass": accessibility_pass,
            "invalid_example_count_capped": len(invalid_examples),
            "invalid_examples": invalid_examples,
            "overall_primary_valid_rate": sum(item["primary_valid_rows"] for item in groups.values()) / len(rows) if rows else None,
        },
        "control_coverage": {
            "required_direction_kinds": sorted(required_direction_kinds),
            "missing_direction_kinds": missing_kinds,
            "pass": not missing_kinds,
        },
        "manipulation_proxy": {
            "injection_applied_rows": injected_rows,
            "observed_shift_count": len(injection_shift_values),
            "mean_observed_shift": _safe_mean(injection_shift_values),
            "intervention_timings": dict(sorted(intervention_timings.items())),
        },
        "same_task_proxies": {
            "description": "These are output-accessibility/compliance proxies, not unrelated collateral outcomes.",
            "by_direction_kind": {
                kind: {
                    "rows": sum(value["rows"] for key, value in groups.items() if key.startswith(f"{kind}:")),
                    "primary_valid_rows": sum(value["primary_valid_rows"] for key, value in groups.items() if key.startswith(f"{kind}:")),
                }
                for kind in sorted(direction_kinds)
            },
        },
        "collateral": collateral,
    }
    report["report_sha256"] = canonical_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--construct-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-primary-valid-rate", type=float, default=0.95)
    parser.add_argument(
        "--collateral-summary",
        type=Path,
        default=None,
        help="Completed prompt-only collateral score for this construct.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = audit_output(
        raw_path=args.raw_generations.resolve(),
        construct_spec_path=args.construct_spec.resolve(),
        minimum_primary_valid_rate=args.minimum_primary_valid_rate,
        collateral_summary_path=(
            None if args.collateral_summary is None else args.collateral_summary.resolve()
        ),
    )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accessibility"]["pass"] and report["control_coverage"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
