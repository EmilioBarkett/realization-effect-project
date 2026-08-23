#!/usr/bin/env python3
"""Validate a multi-construct prompt inventory and emit its run manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import (
    load_analysis_spec,
    load_construct_specs,
    load_run_config,
)
from construct_benchmark.manifests import build_run_plan, write_run_plan
from construct_benchmark.prompts import load_prompt_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and plan a shared activation run across multiple constructs."
    )
    parser.add_argument(
        "--construct-spec",
        action="append",
        type=Path,
        required=True,
        help="Construct specification JSON; repeat once per construct.",
    )
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=None, help="Combined canonical CSV or JSONL inventory.")
    parser.add_argument("--write-plan", type=Path, default=None, help="Output multi-construct run manifest JSON.")
    args = parser.parse_args()

    construct_specs = load_construct_specs(args.construct_spec)
    run_config = load_run_config(args.run_config)
    analysis_spec = load_analysis_spec(args.analysis_spec)
    prompt_records = load_prompt_records(args.prompts) if args.prompts else None
    plan = build_run_plan(
        run_config,
        construct_specs,
        analysis_spec,
        prompt_inventory_path=args.prompts,
        prompt_records=prompt_records,
    )
    if args.write_plan:
        write_run_plan(args.write_plan, plan)
        print(f"wrote run plan: {args.write_plan}")
    print(
        json.dumps(
            {
                "run_id": plan["run_id"],
                "construct_count": plan["construct_count"],
                "construct_ids": plan["run_config"]["construct_ids"],
                "shared_activation_output": plan["shared_execution"]["activation_output"],
                "construct_outputs": {
                    entry["construct_id"]: entry["output_layout"]
                    for entry in plan["constructs"]
                },
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
