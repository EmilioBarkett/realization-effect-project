#!/usr/bin/env python3
"""Run the benchmark control plane on deterministic fake local data.

This command never calls OpenRouter, loads model weights, or requires a GPU.
Its outputs are software smoke-test artifacts, not scientific measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import (  # noqa: E402
    load_analysis_spec,
    load_construct_specs,
    load_run_config,
    validate_analysis_spec,
    validate_run_constructs,
)
from construct_benchmark.fake import run_fake_construct  # noqa: E402
from construct_benchmark.manifests import canonical_hash  # noqa: E402
from construct_benchmark.prompts import validate_prompt_records, write_prompt_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-model benchmark smoke test on deterministic fake data.")
    parser.add_argument("--construct-spec", type=Path, action="append", required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=250)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()

    run_config = load_run_config(args.run_config)
    analysis_spec = load_analysis_spec(args.analysis_spec)
    validate_analysis_spec(run_config, analysis_spec)
    specs = load_construct_specs(args.construct_spec)
    validate_run_constructs(run_config, specs)

    all_records = []
    construct_summaries = {}
    for construct_id in run_config.construct_ids:
        records, summary = run_fake_construct(
            specs[construct_id],
            run_config,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        all_records.extend(records)
        construct_summaries[construct_id] = summary
    inventory_summary = validate_prompt_records(all_records, specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.output_dir / "prompt_inventory.csv"
    write_prompt_records(all_records, prompt_path)
    summary = {
        "run_id": run_config.run_id,
        "construct_ids": list(run_config.construct_ids),
        "fake_model": True,
        "not_empirical": True,
        "external_calls": {"openrouter": False, "runpod": False, "model_weights": False},
        "inventory": inventory_summary,
        "constructs": construct_summaries,
        "provenance": {
            "run_config_hash": canonical_hash(run_config.to_mapping()),
            "analysis_spec_hash": canonical_hash(analysis_spec.to_mapping()),
            "construct_spec_hashes": {
                construct_id: canonical_hash(spec.to_mapping()) for construct_id, spec in specs.items()
            },
            "prompt_inventory_path": str(prompt_path),
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
    }
    summary_path = args.output_dir / "fake_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
