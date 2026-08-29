#!/usr/bin/env python3
"""Score a completed prompt-only behavior baseline and its variation gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.behavior_baseline import (  # noqa: E402
    read_behavior_output,
    score_behavior_rows,
    validate_behavior_output_manifest,
)
from construct_benchmark.behavioral_variation import audit_prompt_only_variation  # noqa: E402
from construct_benchmark.config import load_construct_specs, load_run_config  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a manifest-backed prompt-only behavior baseline without steering."
    )
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-diagnostic", action="store_true")
    parser.add_argument("--minimum-valid", type=int, default=None)
    parser.add_argument("--minimum-distinct", type=int, default=None)
    parser.add_argument("--minimum-sd", type=float, default=None)
    parser.add_argument("--maximum-invalid", type=int, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_config = load_run_config(args.run_config)
    construct_specs = load_construct_specs(args.construct_spec)
    raw_rows = read_behavior_output(args.raw_generations)
    manifest, complete = validate_behavior_output_manifest(
        args.raw_generations,
        raw_rows,
        run_config=run_config,
        construct_specs=construct_specs,
        allow_incomplete_diagnostic=args.allow_incomplete_diagnostic,
    )
    overrides = {
        key: value
        for key, value in {
            "minimum_zero_dose_valid": args.minimum_valid,
            "minimum_zero_dose_distinct": args.minimum_distinct,
            "minimum_zero_dose_sample_sd": args.minimum_sd,
            "maximum_zero_dose_invalid": args.maximum_invalid,
        }.items()
        if value is not None
    }
    parsed_rows, summary = score_behavior_rows(raw_rows, construct_specs)
    split = str(manifest.get("split", "behavior_eval"))
    if split == "collateral_eval":
        variation = {
            construct_id: {
                "construct_id": construct_id,
                "status": "not_applicable",
                "pass": True,
                "reason": "Collateral tasks are scored for unrelated-task correctness, not prompt-only construct variation.",
            }
            for construct_id in construct_specs
        }
    else:
        variation = {
            construct_id: audit_prompt_only_variation(raw_rows, spec, thresholds=overrides)
            for construct_id, spec in construct_specs.items()
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = args.output_dir / "parsed_generations.csv"
    if parsed_rows:
        with parsed_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(parsed_rows[0]))
            writer.writeheader()
            writer.writerows(parsed_rows)
    report = {
        "manifest_type": "prompt_only_behavior_score",
        # A complete diagnostic/test run is not confirmatory merely because it
        # has all expected rows. Preserve the run manifest's pre-registered
        # status and only report confirmatory when the source run was marked so.
        "confirmatory": bool(manifest.get("confirmatory", False))
        and complete
        and not args.allow_incomplete_diagnostic,
        "raw_output": str(args.raw_generations),
        "raw_record_count": len(raw_rows),
        "manifest_complete": complete,
        "behavior": summary,
        "variation_gate": variation,
        "pass": complete and all(item["pass"] for item in variation.values()),
        "provenance": {
            "run_id": manifest["run_id"],
            "prompt_inventory_sha256": manifest["prompt_inventory_sha256"],
            "run_config_hash": manifest["run_config_hash"],
            "output_manifest": str(args.raw_generations.with_suffix(args.raw_generations.suffix + ".manifest.json")),
            "diagnostic_override": args.allow_incomplete_diagnostic,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
