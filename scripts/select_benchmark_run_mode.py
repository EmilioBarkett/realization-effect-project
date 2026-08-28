#!/usr/bin/env python3
"""Materialize a deterministic prompt inventory for a configured run mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import load_construct_specs, load_run_config, validate_run_constructs  # noqa: E402
from construct_benchmark.campaign import confirmatory_execution_report  # noqa: E402
from construct_benchmark.manifests import file_sha256  # noqa: E402
from construct_benchmark.prompts import load_prompt_records, write_prompt_records  # noqa: E402
from construct_benchmark.run_modes import select_prompt_records  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a complete or bounded, hashable prompt inventory for a configured "
            "benchmark run mode without making API or model calls."
        )
    )
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="Complete frozen prompt inventory.")
    parser.add_argument("--mode", choices=("test", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Selected CSV or JSONL inventory.")
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--confirmatory-campaign",
        type=Path,
        default=None,
        help="Campaign manifest required before selecting a gated full inventory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selection without writing files.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_config = load_run_config(args.run_config)
    construct_specs = load_construct_specs(args.construct_spec)
    validate_run_constructs(run_config, construct_specs)
    campaign_path = args.confirmatory_campaign
    configured_campaign = run_config.execution.get("confirmatory_campaign_path")
    if campaign_path is None and isinstance(configured_campaign, str) and configured_campaign.strip():
        campaign_path = args.run_config.parent / configured_campaign
    if args.mode == "full" and campaign_path is not None:
        release_report = confirmatory_execution_report(campaign_path, mode="full")
        if not release_report["ready"]:
            blockers = ", ".join(
                str(check["name"]) for check in release_report["blocking_checks"]
            )
            raise SystemExit(
                "Confirmatory campaign release is blocked; refusing to select a full inventory. "
                f"Blocking checks: {blockers}"
            )
    source_records = load_prompt_records(args.prompts)
    selected_records, manifest = select_prompt_records(
        source_records,
        run_config=run_config,
        construct_specs=construct_specs,
        mode=args.mode,
    )
    manifest["source_prompt_inventory_sha256"] = file_sha256(args.prompts)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing selected inventory: {args.output}")
    if args.manifest_output.exists():
        raise SystemExit(f"Refusing to overwrite existing selection manifest: {args.manifest_output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_prompt_records(selected_records, args.output)
    manifest["selected_prompt_inventory_sha256"] = file_sha256(args.output)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": str(args.output),
                "manifest": str(args.manifest_output),
                "selected_prompt_count": len(selected_records),
                "confirmatory": manifest["confirmatory"],
                "max_runtime_minutes": manifest["max_runtime_minutes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
