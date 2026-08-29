#!/usr/bin/env python3
"""Compose and audit the non-confirmatory Wave 1 v4 preflight inventory.

The v3 Wave 1 inventory is immutable.  This command replaces only the
evidence-diagnosticity downstream rows with the independently generated v4
release, retains all v3 probe rows and the other construct rows verbatim, and
writes a new frozen inventory with an explicit audit report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import load_construct_specs  # noqa: E402
from construct_benchmark.generation import downstream_prompt_text_issues  # noqa: E402
from construct_benchmark.manifests import file_sha256  # noqa: E402
from construct_benchmark.prompts import PromptRecord, load_prompt_records, validate_prompt_records, write_prompt_records  # noqa: E402


WAVE1_IDS = (
    "realization_account_closure",
    "evidence_diagnosticity",
    "source_reliability",
    "persistence_continuation",
)
EVIDENCE_ID = "evidence_diagnosticity"
VECTOR_SPLITS = {"direction_train", "direction_validation", "direction_heldout"}
REPLACED_SPLITS = {"behavior_eval", "steering_eval", "calibration", "collateral_eval"}
TRADEOFF_FIELDS = (
    "high_information_option",
    "cost_contrast",
    "delay_contrast",
    "stakes_contrast",
    "option_order",
)


def _normalise(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _task_metadata(record: PromptRecord) -> dict[str, Any]:
    value = record.metadata.get("task_metadata", {})
    return dict(value) if isinstance(value, dict) else {}


def audit_v4_records(records: Iterable[PromptRecord]) -> dict[str, Any]:
    """Audit v4 evidence rows for the preregistered non-dominance design."""

    evidence = [record for record in records if record.construct_id == EVIDENCE_ID]
    failures: list[str] = []
    issues: list[str] = []
    counts = Counter(record.split for record in evidence)
    expected_counts = {"behavior_eval": 32, "steering_eval": 32, "calibration": 16, "collateral_eval": 16}
    if counts != expected_counts:
        failures.append(f"evidence v4 split counts={dict(sorted(counts.items()))}, expected={expected_counts}")

    seen_text: dict[str, str] = {}
    for record in evidence:
        issues.extend(
            f"{record.prompt_id}: {issue}"
            for issue in downstream_prompt_text_issues(record.prompt_text, expected_output_format=record.expected_output_format)
        )
        if record.split in {"behavior_eval", "steering_eval", "calibration"}:
            metadata = _task_metadata(record)
            for field in TRADEOFF_FIELDS:
                if field not in metadata:
                    issues.append(f"{record.prompt_id}: missing tradeoff metadata {field}")
            if record.split in {"behavior_eval", "steering_eval"}:
                if metadata.get("high_information_option") not in {"option_a", "option_b"}:
                    issues.append(f"{record.prompt_id}: invalid high_information_option")
                for forbidden in (
                    "targeted test",
                    "routine test",
                    "high-information",
                    "low-information",
                    "diagnosticity",
                ):
                    if forbidden in record.prompt_text.casefold():
                        issues.append(f"{record.prompt_id}: end-user prompt contains forbidden label {forbidden!r}")
        normalised = _normalise(record.prompt_text)
        if normalised in seen_text:
            issues.append(f"{record.prompt_id}: duplicate normalized prompt text with {seen_text[normalised]}")
        else:
            seen_text[normalised] = record.prompt_id

    for split in ("behavior_eval", "steering_eval"):
        rows = [record for record in evidence if record.split == split]
        combinations = {
            tuple(_task_metadata(record).get(field) for field in TRADEOFF_FIELDS)
            for record in rows
        }
        if len(combinations) != 32:
            failures.append(f"evidence v4 {split} has {len(combinations)} tradeoff combinations, expected 32")
        for field in TRADEOFF_FIELDS:
            field_counts = Counter(_task_metadata(record).get(field) for record in rows)
            if field_counts and min(field_counts.values()) != max(field_counts.values()):
                failures.append(f"evidence v4 {split} {field} is unbalanced: {dict(field_counts)}")
        high_counts = Counter(_task_metadata(record).get("high_information_option") for record in rows)
        if high_counts != Counter({"option_a": 16, "option_b": 16}):
            failures.append(f"evidence v4 {split} high-information option balance={dict(high_counts)}")

    calibration = [_task_metadata(record) for record in evidence if record.split == "calibration"]
    for metadata in calibration:
        for field in ("diagnostic_benefit_contrast", "high_information_option", "cost_contrast", "delay_contrast", "stakes_contrast"):
            if metadata.get(field) != "matched":
                failures.append(f"calibration item has non-matched {field}={metadata.get(field)!r}")
                break

    collateral = [record for record in evidence if record.split == "collateral_eval"]
    correct_counts = Counter(_task_metadata(record).get("correct_option") for record in collateral)
    if correct_counts != Counter({1: 8, 2: 8}):
        failures.append(f"evidence v4 collateral correct-option balance={dict(correct_counts)}")
    if issues:
        failures.extend(issues)
    return {
        "pass": not failures,
        "failures": failures,
        "record_count": len(evidence),
        "split_counts": dict(sorted(counts.items())),
        "tradeoff_fields": list(TRADEOFF_FIELDS),
        "full_factorial_size": 32,
        "high_information_option_balance": {
            split: dict(
                Counter(
                    _task_metadata(record).get("high_information_option")
                    for record in evidence
                    if record.split == split
                )
            )
            for split in ("behavior_eval", "steering_eval")
        },
    }


def compose_inventory(*, base_inventory: Path, evidence_inventory: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    base_records = load_prompt_records(base_inventory)
    evidence_records = load_prompt_records(evidence_inventory)
    selected_evidence = [
        record
        for record in evidence_records
        if record.construct_id == EVIDENCE_ID and record.split in REPLACED_SPLITS
    ]
    if len(selected_evidence) != 96:
        raise ValueError(f"Expected 96 v4 evidence downstream rows, found {len(selected_evidence)}.")
    retained = [
        record
        for record in base_records
        if not (record.construct_id == EVIDENCE_ID and record.split in REPLACED_SPLITS)
    ]
    combined = retained + selected_evidence
    spec_paths = [
        _ROOT / "configs/construct_benchmark/constructs" / f"{construct_id}_v3.json"
        for construct_id in WAVE1_IDS
    ]
    spec_paths[1] = _ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v4.json"
    specs = load_construct_specs(spec_paths)
    validate_prompt_records(combined, specs, require_all_splits=True)
    prompt_ids = [record.prompt_id for record in combined]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("The composed v4 inventory contains duplicate prompt IDs.")
    audit = audit_v4_records(selected_evidence)
    if not audit["pass"]:
        raise ValueError("The v4 evidence audit failed: " + "; ".join(audit["failures"][:5]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined.csv"
    write_prompt_records(combined, output_path)
    manifest = {
        "schema_version": "0.1.0",
        "manifest_type": "wave_execution_prompt_inventory",
        "release_id": "wave1_preflight_v4",
        "inventory_version": "wave1_preflight_v4",
        "status": "frozen",
        "frozen": True,
        "confirmatory": False,
        "scope": "wave1_preflight_repaired_downstream_release",
        "wave": 1,
        "construct_ids": list(WAVE1_IDS),
        "record_count": len(combined),
        "parent_inventory": {"path": str(base_inventory.resolve()), "sha256": file_sha256(base_inventory)},
        "v4_generation_source": {"path": str(evidence_inventory.resolve()), "sha256": file_sha256(evidence_inventory)},
        "generation_provider": "openai_luna",
        "preserved_v3_rule": "Retain all v3 probe rows and all non-evidence downstream rows verbatim; replace only evidence downstream rows.",
        "constructs": {
            construct_id: {
                "record_count": sum(record.construct_id == construct_id for record in combined),
                "split_counts": dict(sorted(Counter(record.split for record in combined if record.construct_id == construct_id).items())),
            }
            for construct_id in WAVE1_IDS
        },
        "prompt_audit": audit,
        "output_path": "combined.csv",
        "output_sha256": file_sha256(output_path),
    }
    (output_dir / "inventory_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "inventory_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-inventory", type=Path, required=True)
    parser.add_argument("--evidence-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = compose_inventory(
        base_inventory=args.base_inventory.resolve(),
        evidence_inventory=args.evidence_inventory.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
