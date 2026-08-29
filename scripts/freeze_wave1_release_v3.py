#!/usr/bin/env python3
"""Freeze the reviewed, expanded Wave 1 prompt inventory.

The original v2 inventory is treated as an immutable input.  This command
adds only the reviewed primary supplements and the independent collateral
items, validates the combined contract, and writes a new versioned inventory
plus provenance and hashes.  It never modifies the v2 inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from construct_benchmark.config import load_construct_spec
from construct_benchmark.generation import downstream_prompt_text_issues
from construct_benchmark.manifests import canonical_hash
from construct_benchmark.prompts import load_prompt_records, validate_prompt_records, write_prompt_records


ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTS = (
    "realization_account_closure",
    "evidence_diagnosticity",
    "source_reliability",
    "persistence_continuation",
)
SUPPLEMENTS = CONSTRUCTS[:3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _spec(construct_id: str):
    return load_construct_spec(
        ROOT / "configs/construct_benchmark/constructs" / f"{construct_id}_v3.json"
    )


def _load(path: Path) -> list[Any]:
    return load_prompt_records(path)


def _repair_legacy_downstream_contracts(records: list[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Repair only redundant terminal response text in the copied v2 rows.

    The v2 source file remains immutable.  This release copy retains the
    original prompt IDs and records a before/after hash for every bounded
    repair, so old engineering outputs remain separately reproducible.
    """

    repaired: list[Any] = []
    receipts: list[dict[str, Any]] = []
    terminal_contract = re.compile(
        r"\n\s*\n(?:return|respond|report|output|provide|reply|answer)\b.*$",
        re.IGNORECASE | re.DOTALL,
    )
    for row in records:
        issues = downstream_prompt_text_issues(row.prompt_text)
        text = row.prompt_text
        if row.split in {"behavior_eval", "steering_eval", "calibration"} and "multiple response contracts" in " ".join(issues):
            match = terminal_contract.search(text)
            if match is not None and match.start() > 0:
                candidate = text[: match.start()].rstrip()
                if not downstream_prompt_text_issues(candidate):
                    metadata = dict(row.metadata)
                    metadata["release_v3_legacy_repair"] = {
                        "kind": "remove_redundant_terminal_response_contract",
                        "source_inventory_version": "wave1_repaired_v2",
                    }
                    receipts.append(
                        {
                            "prompt_id": row.prompt_id,
                            "construct_id": row.construct_id,
                            "split": row.split,
                            "before_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "after_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
                        }
                    )
                    row = replace(row, prompt_text=candidate, metadata=metadata)
        repaired.append(row)
    return repaired, receipts


def _audit_records(records: list[Any], specs: dict[str, Any]) -> dict[str, Any]:
    validate_prompt_records(records, specs)
    failures: list[str] = []
    for row in records:
        if row.prompt_role in {"behavior", "steering", "calibration", "collateral"}:
            failures.extend(
                f"{row.prompt_id}: {issue}"
                for issue in downstream_prompt_text_issues(row.prompt_text)
            )
    counts = Counter((row.construct_id, row.split) for row in records)
    expected = {
        "realization_account_closure": {
            "direction_train": 200, "direction_validation": 80, "direction_heldout": 80,
            "behavior_eval": 32, "steering_eval": 32, "calibration": 32, "collateral_eval": 32,
        },
        "evidence_diagnosticity": {
            "direction_train": 200, "direction_validation": 80, "direction_heldout": 80,
            "behavior_eval": 32, "steering_eval": 32, "calibration": 32, "collateral_eval": 32,
        },
        "source_reliability": {
            "direction_train": 200, "direction_validation": 80, "direction_heldout": 80,
            "behavior_eval": 32, "steering_eval": 32, "calibration": 32, "collateral_eval": 32,
        },
        "persistence_continuation": {
            "direction_train": 200, "direction_validation": 80, "direction_heldout": 80,
            "behavior_eval": 80, "steering_eval": 80, "calibration": 80, "collateral_eval": 32,
        },
    }
    for construct_id, split_counts in expected.items():
        for split, target in split_counts.items():
            actual = counts[(construct_id, split)]
            if actual != target:
                failures.append(f"{construct_id}/{split}: {actual} records, expected {target}")
    collateral_counts = Counter(
        row.metadata.get("task_metadata", {}).get("correct_option")
        for row in records
        if row.split == "collateral_eval"
    )
    if collateral_counts != Counter({1: 64, 2: 64}):
        failures.append(f"collateral correct_option balance={dict(collateral_counts)}")
    return {
        "pass": not failures,
        "failures": failures,
        "counts_by_construct_split": {
            construct_id: {
                split: counts[(construct_id, split)]
                for split in expected[construct_id]
            }
            for construct_id in CONSTRUCTS
        },
        "collateral_correct_option_counts": dict(sorted(collateral_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-inventory", required=True)
    parser.add_argument("--primary-supplement-dir", required=True)
    parser.add_argument("--collateral-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--review-audit", required=True)
    args = parser.parse_args()

    original_path = _path(args.original_inventory).resolve()
    primary_dir = _path(args.primary_supplement_dir).resolve()
    collateral_dir = _path(args.collateral_dir).resolve()
    output_dir = _path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty frozen output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original, legacy_repairs = _repair_legacy_downstream_contracts(_load(original_path))
    additions: list[Any] = []
    source_files = [original_path]
    for construct_id in SUPPLEMENTS:
        path = primary_dir / f"{construct_id}.csv"
        additions.extend(_load(path))
        source_files.append(path)
    for construct_id in CONSTRUCTS:
        path = collateral_dir / f"{construct_id}.csv"
        additions.extend(_load(path))
        source_files.append(path)
    records = [*original, *additions]
    # The release is a new immutable execution copy.  Keep source lineage in
    # metadata, but give every row one authoritative inventory version so the
    # distributed worker does not reject the intentional v1/v2 source mix.
    records = [
        replace(
            row,
            metadata={
                **row.metadata,
                "inventory_version": "wave1_release_v3",
            },
        )
        for row in records
    ]
    specs = {construct_id: _spec(construct_id) for construct_id in CONSTRUCTS}
    audit = _audit_records(records, specs)
    audit_path = output_dir / "inventory_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not audit["pass"]:
        raise ValueError(f"Expanded inventory failed audit: {audit['failures'][:3]}")

    combined_path = output_dir / "combined.csv"
    write_prompt_records(records, combined_path)
    spec_hashes = {
        construct_id: _sha256(ROOT / "configs/construct_benchmark/constructs" / f"{construct_id}_v3.json")
        for construct_id in CONSTRUCTS
    }
    manifest = {
        "manifest_type": "wave1_release_v3_frozen_prompt_inventory",
        "schema_version": "0.1.0",
        "inventory_version": "wave1_release_v3",
        "frozen": True,
        "confirmatory_eligibility": "pending_wave1_gates",
        "source_original_inventory": str(original_path),
        "source_original_inventory_sha256": _sha256(original_path),
        "source_additions": [
            {"path": str(path), "sha256": _sha256(path)} for path in source_files[1:]
        ],
        "construct_spec_sha256": spec_hashes,
        "review_audit_path": str(_path(args.review_audit).resolve()),
        "review_audit_sha256": _sha256(_path(args.review_audit).resolve()),
        "combined_csv_sha256": _sha256(combined_path),
        "record_count": len(records),
        "audit": audit,
        "preservation_note": "The original v2 engineering inventory remains unchanged and separately addressable.",
        "legacy_release_repairs": legacy_repairs,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    (output_dir / "inventory_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "record_count": len(records), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
