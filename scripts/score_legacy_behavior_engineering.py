#!/usr/bin/env python3
"""Score a complete historical behavior output under the current gate.

The recovered Mistral baseline was created before ``RunConfig`` normalization
was introduced.  Its manifest stores the canonical hash of the raw embedded
JSON config, while the current loader hashes the normalized dataclass mapping.
This compatibility scorer validates the raw snapshot hash explicitly, checks
all row identities and model/spec provenance, and reclassifies the result as
non-confirmatory engineering evidence.  It is intentionally not a path for
confirmatory runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
import sys

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.behavior_baseline import (  # noqa: E402
    output_manifest_path,
    read_behavior_output,
    score_behavior_rows,
)
from construct_benchmark.behavioral_variation import audit_prompt_only_variation  # noqa: E402
from construct_benchmark.config import load_construct_specs  # noqa: E402
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def score_legacy_output(
    *,
    raw_output: Path,
    run_config_snapshot: Path,
    construct_spec_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    raw_rows = read_behavior_output(raw_output)
    manifest = _load_object(output_manifest_path(raw_output))
    config_payload = _load_object(run_config_snapshot)
    specs = load_construct_specs(construct_spec_paths)
    if manifest.get("manifest_type") != "construct_behavior_output":
        raise ValueError("Recovered behavior output has an unexpected manifest type.")
    if manifest.get("complete") is not True:
        raise ValueError("Recovered behavior output is not complete.")
    expected = manifest.get("expected_record_count")
    if not isinstance(expected, int) or len(raw_rows) != expected or manifest.get("completed_record_count") != expected:
        raise ValueError("Recovered behavior output count does not match its manifest.")
    if file_sha256(raw_output) != manifest.get("raw_generations_sha256"):
        raise ValueError("Recovered behavior output does not match its raw hash.")
    raw_config_hash = canonical_hash(config_payload)
    if manifest.get("run_config_hash") != raw_config_hash:
        raise ValueError("Recovered behavior manifest does not match its embedded raw config snapshot.")
    construct_ids = tuple(config_payload.get("construct_ids", ()))
    if set(construct_ids) != set(specs) or len(construct_ids) != len(specs):
        raise ValueError("Embedded config construct_ids and requested specs disagree.")
    expected_spec_hashes = manifest.get("construct_spec_hashes", {})
    for construct_id, spec in specs.items():
        if expected_spec_hashes.get(construct_id) != canonical_hash(spec.to_mapping()):
            raise ValueError(f"Recovered behavior spec hash mismatch for {construct_id}.")

    expected_ids = set(manifest.get("expected_record_ids", ()))
    seen: set[str] = set()
    for row in raw_rows:
        record_id = row.get("record_id")
        prompt_id = row.get("prompt_id")
        construct_id = row.get("construct_id")
        if record_id in seen or record_id not in expected_ids or record_id != f"{prompt_id}__prompt_only":
            raise ValueError(f"Recovered behavior output has an invalid record identity: {record_id!r}.")
        if construct_id not in specs:
            raise ValueError(f"Recovered behavior output has unknown construct: {construct_id!r}.")
        for field, expected_value in (
            ("split", manifest.get("split")),
            ("intervention", "none"),
            ("prompt_inventory_sha256", manifest.get("prompt_inventory_sha256")),
            ("run_config_hash", manifest.get("run_config_hash")),
            ("construct_spec_hash", expected_spec_hashes[construct_id]),
        ):
            if row.get(field) != expected_value:
                raise ValueError(f"Recovered behavior row {record_id!r} has incompatible {field}.")
        if row.get("model") != manifest.get("model"):
            raise ValueError(f"Recovered behavior row {record_id!r} has incompatible model metadata.")
        seen.add(record_id)
    if seen != expected_ids:
        raise ValueError("Recovered behavior output is missing manifest-registered records.")

    parsed_rows, behavior_summary = score_behavior_rows(raw_rows, specs)
    variation = {
        construct_id: audit_prompt_only_variation(raw_rows, spec)
        for construct_id, spec in specs.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = output_dir / "parsed_generations.csv"
    with parsed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parsed_rows[0]))
        writer.writeheader()
        writer.writerows(parsed_rows)
    report = {
        "schema_version": "0.1.0",
        "manifest_type": "prompt_only_behavior_score",
        "confirmatory": False,
        "raw_output": str(raw_output),
        "raw_record_count": len(raw_rows),
        "manifest_complete": True,
        "behavior": behavior_summary,
        "variation_gate": variation,
        "pass": all(item["pass"] for item in variation.values()),
        "provenance": {
            "run_id": manifest.get("run_id"),
            "prompt_inventory_sha256": manifest.get("prompt_inventory_sha256"),
            "run_config_hash": manifest.get("run_config_hash"),
            "run_config_snapshot": str(run_config_snapshot),
            "run_config_hash_method": "canonical_hash(raw_embedded_json_snapshot)",
            "output_manifest": str(output_manifest_path(raw_output)),
            "reclassified_non_confirmatory": True,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--run-config-snapshot", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = score_legacy_output(
        raw_output=args.raw_output.resolve(),
        run_config_snapshot=args.run_config_snapshot.resolve(),
        construct_spec_paths=[path.resolve() for path in args.construct_spec],
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
