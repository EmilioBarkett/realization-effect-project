#!/usr/bin/env python3
"""Validate one real model's frozen behavioral/accessibility preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from construct_benchmark.model_preflight import (  # noqa: E402
    load_preflight_gate_config,
    validate_preflight,
)
from construct_benchmark.config import load_construct_specs  # noqa: E402
from construct_benchmark.manifests import file_sha256  # noqa: E402


def _assignments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Expected CONSTRUCT_ID=PATH, got {value!r}.")
        construct_id, path = value.split("=", 1)
        if not construct_id or not path or construct_id in result:
            raise SystemExit(f"Invalid or duplicate steering output assignment: {value!r}.")
        result[construct_id] = Path(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--behavior-output", type=Path, required=True)
    parser.add_argument("--collateral-output", type=Path, required=True)
    parser.add_argument(
        "--steering-output",
        action="append",
        required=True,
        metavar="CONSTRUCT_ID=PATH",
    )
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=ROOT / "configs/construct_benchmark/gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    specs = load_construct_specs(args.construct_spec)
    gate_config = load_preflight_gate_config(args.gate_config)
    gate_config["gate_config_sha256"] = file_sha256(args.gate_config)
    thresholds = {}
    if args.thresholds is not None and args.gate_config == parser.get_default("gate_config"):
        config = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if isinstance(config.get("thresholds"), dict):
            thresholds = dict(config["thresholds"])
        else:
            stages = dict(config.get("stages", {}))
            behavior = dict(stages.get("behavior_eval", {}))
            collateral = dict(stages.get("collateral_eval", {}))
            steering = dict(stages.get("steering_eval", {}))
            thresholds = {
                "behavior_minimum_valid_rate": behavior.get("minimum_valid_rate", 1.0),
                "behavior_maximum_invalid_items": behavior.get("maximum_invalid_items", 0),
                "behavior_minimum_distinct_outcomes": behavior.get("minimum_distinct_outcomes", 3),
                "behavior_minimum_sample_sd": behavior.get("minimum_sample_sd", 2.0),
                "collateral_minimum_valid_rate": collateral.get("minimum_valid_rate", 0.95),
                "collateral_minimum_correctness_rate": collateral.get("minimum_correctness_rate", 0.75),
                "steering_minimum_valid_rate": steering.get("minimum_valid_rate", 0.95),
            }
    steering_outputs = _assignments(args.steering_output)
    report = validate_preflight(
        selection_manifest=selection,
        construct_specs=specs,
        behavior_output=args.behavior_output,
        collateral_output=args.collateral_output,
        steering_outputs=steering_outputs,
        thresholds=thresholds,
        gate_config=gate_config,
    )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["release_decision"] == "pass_preflight" else 1


if __name__ == "__main__":
    raise SystemExit(main())
