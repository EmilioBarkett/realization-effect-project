#!/usr/bin/env python3
"""Freeze a small, outcome-independent model-side preflight selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from construct_benchmark.config import load_construct_specs  # noqa: E402
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402
from construct_benchmark.model_preflight import prepare_selection_manifest  # noqa: E402
from construct_benchmark.model_preflight import load_preflight_gate_config  # noqa: E402
from construct_benchmark.prompts import load_prompt_records, validate_prompt_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-inventory", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=ROOT / "configs/construct_benchmark/gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--minimum-items", type=int, default=None)
    parser.add_argument("--target-items", type=int, default=None)
    parser.add_argument("--maximum-items", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    specs = load_construct_specs(args.construct_spec)
    records = load_prompt_records(args.prompt_inventory)
    validate_prompt_records(records, specs)
    gate_config = load_preflight_gate_config(args.gate_config)
    configured_construct_ids = list(gate_config["construct_ids"])
    if list(specs) != configured_construct_ids:
        raise SystemExit(
            "Construct specs must match the v2 gate order: "
            f"{configured_construct_ids}"
        )
    allowed_models = {
        str(entry.get("model_id"))
        for entry in gate_config.get("models", [])
        if isinstance(entry, dict) and entry.get("model_id")
    }
    if allowed_models and args.model_id not in allowed_models:
        raise SystemExit(f"Model {args.model_id!r} is not registered in the v2 gate config.")
    inventory_name = str(args.prompt_inventory).casefold()
    release = dict(gate_config.get("prompt_release", {}))
    required_tokens = [str(value).casefold() for value in release.get("required_path_tokens", [])]
    forbidden_tokens = [str(value).casefold() for value in release.get("forbidden_path_tokens", [])]
    missing_tokens = [token for token in required_tokens if token not in inventory_name]
    forbidden_present = [token for token in forbidden_tokens if token in inventory_name]
    if missing_tokens or forbidden_present:
        raise SystemExit(
            "Prompt inventory is not an approved v2 release: "
            f"missing path tokens={missing_tokens}, forbidden path tokens={forbidden_present}"
        )
    bounds = dict(gate_config["item_bounds"])
    supplied_bounds = {
        "minimum": args.minimum_items,
        "target": args.target_items,
        "maximum": args.maximum_items,
    }
    for key, supplied in supplied_bounds.items():
        if supplied is not None and int(supplied) != int(bounds[key]):
            raise SystemExit(f"--{key}-items does not match the frozen v2 gate config.")
    model = {"model_id": args.model_id, "revision": args.revision}
    if args.tokenizer_id is not None:
        model["tokenizer_id"] = args.tokenizer_id
    manifest = prepare_selection_manifest(
        records,
        source_inventory=args.prompt_inventory,
        model=model,
        construct_ids=specs,
        seed=args.seed,
        minimum_items=int(bounds["minimum"]),
        target_items=int(bounds["target"]),
        maximum_items=int(bounds["maximum"]),
        gate_config=gate_config,
        gate_config_sha256=file_sha256(args.gate_config),
    )
    manifest["output"] = str(args.output.resolve())
    manifest["selection_sha256"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "selection_sha256"}
    )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
