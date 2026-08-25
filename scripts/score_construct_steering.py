#!/usr/bin/env python3
"""Parse raw steering generations and compute the primary target-direction effect."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.behavior import (  # noqa: E402
    BehaviorObservation,
    orient_primary_outcome,
    parse_behavior_output,
    primary_outcome,
)
from construct_benchmark.config import load_construct_spec  # noqa: E402
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402
from construct_benchmark.uncertainty import bootstrap_state_transfer_ci  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score construct steering generations.")
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--construct-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()

    spec = load_construct_spec(args.construct_spec)
    rows = []
    observations = []
    for line_number, line in enumerate(args.raw_generations.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on raw generation line {line_number}.") from exc
        if raw.get("construct_id") != spec.construct_id:
            raise ValueError(f"Raw generation line {line_number} has the wrong construct_id.")
        task_metadata = dict(raw.get("task_metadata") or {})
        parsed = parse_behavior_output(
            raw.get("output_text", ""),
            parser_id=str(raw.get("parser_id") or spec.parsing_rules["parser_id"]),
            item_metadata=task_metadata,
        )
        outcome = None
        directed_outcome = None
        error = parsed.error
        if parsed.valid:
            try:
                outcome = primary_outcome(parsed, spec.independent_behavior_task["primary_outcome"])
                directed_outcome = orient_primary_outcome(spec.construct_id, outcome, task_metadata)
            except ValueError as exc:
                error = str(exc)
        valid_primary = parsed.valid and directed_outcome is not None and error is None
        row = {
            "condition_id": raw["condition_id"],
            "prompt_id": raw["prompt_id"],
            "direction_kind": raw["direction_kind"],
            "direction_index": raw["direction_index"],
            "dose": raw["dose"],
            "physical_scale": raw["physical_scale"],
            "parser_valid": parsed.valid,
            "primary_valid": valid_primary,
            "outcome": outcome,
            "directed_outcome": directed_outcome,
            "error": error or "",
            "task_metadata_json": json.dumps(task_metadata, sort_keys=True),
        }
        rows.append(row)
        if raw["direction_kind"] == "target":
            observations.append(
                BehaviorObservation(
                    item_id=str(raw["prompt_id"]),
                    scale=float(raw["dose"]),
                    outcome=directed_outcome,
                    valid=valid_primary,
                )
            )
    target_doses = sorted({observation.scale for observation in observations})
    if 0.0 not in target_doses:
        raise ValueError("Target-direction rows do not contain a zero-dose condition.")
    effect, effect_ci = bootstrap_state_transfer_ci(
        observations,
        positive_scale=max(target_doses),
        negative_scale=min(target_doses),
        zero_scale=0.0,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = args.output_dir / "parsed_generations.csv"
    with parsed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "construct_id": spec.construct_id,
        "primary_outcome": spec.independent_behavior_task["primary_outcome"],
        "target_direction_effect": asdict(effect),
        "uncertainty": effect_ci.to_mapping(),
        "control_rows": {
            kind: sum(row["direction_kind"] == kind for row in rows)
            for kind in ("shuffled", "random")
        },
        "provenance": {
            "construct_spec_hash": canonical_hash(spec.to_mapping()),
            "raw_generations_sha256": file_sha256(args.raw_generations),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
