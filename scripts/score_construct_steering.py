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
from construct_benchmark.manipulation import (  # noqa: E402
    score_expected_observed_shift,
    summarize_manipulation_records,
)
from construct_benchmark.uncertainty import bootstrap_state_transfer_ci  # noqa: E402


def _output_manifest_path(raw_generations: Path) -> Path:
    return raw_generations.with_suffix(raw_generations.suffix + ".manifest.json")


def _read_raw_rows(raw_generations: Path, *, construct_id: str) -> list[dict]:
    if not raw_generations.is_file():
        raise ValueError(f"Raw steering output does not exist: {raw_generations}")
    rows: list[dict] = []
    for line_number, line in enumerate(
        raw_generations.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on raw generation line {line_number}.") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Raw generation line {line_number} is not a JSON object.")
        if raw.get("construct_id") != construct_id:
            raise ValueError(f"Raw generation line {line_number} has the wrong construct_id.")
        rows.append(raw)
    return rows


def _load_and_validate_output_manifest(
    raw_generations: Path,
    raw_rows: list[dict],
    *,
    construct_id: str,
    construct_spec_hash: str,
    allow_incomplete_diagnostic: bool = False,
) -> tuple[dict, bool]:
    """Require a complete, identity-consistent steering output by default.

    The diagnostic override permits a deliberate partial-run inspection, but
    still rejects malformed rows, incompatible provenance, duplicate records,
    and records outside the frozen manifest's identity set.
    """

    manifest_path = _output_manifest_path(raw_generations)
    if not manifest_path.is_file():
        raise ValueError(
            f"Cannot score {raw_generations} without its adjacent output manifest {manifest_path}."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path} is not valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object.")

    required_fields = (
        "manifest_type",
        "construct_id",
        "steering_plan_sha256",
        "prompt_inventory_sha256",
        "run_config_hash",
        "construct_spec_hash",
        "model",
        "injection_layer",
        "tracking_layers",
        "tracking_directions",
        "expected_condition_ids",
        "expected_record_ids",
        "activation_site",
        "position_mode",
        "intervention_timing",
        "dtype",
        "expected_record_count",
        "complete",
    )
    missing_fields = [field for field in required_fields if field not in manifest]
    if missing_fields:
        raise ValueError(f"{manifest_path} is missing required fields: {missing_fields}.")
    if manifest["manifest_type"] != "construct_steering_output":
        raise ValueError(f"{manifest_path} is not a construct steering output manifest.")
    if manifest["construct_id"] != construct_id:
        raise ValueError(f"{manifest_path} has the wrong construct_id.")
    if manifest["construct_spec_hash"] != construct_spec_hash:
        raise ValueError(f"{manifest_path} construct specification hash differs from the requested spec.")

    model = manifest["model"]
    if not isinstance(model, dict) or not model.get("model_id"):
        raise ValueError(f"{manifest_path} is missing model metadata.")
    tracking_layers = manifest["tracking_layers"]
    if not isinstance(tracking_layers, list):
        raise ValueError(f"{manifest_path} tracking_layers must be a list.")
    try:
        tracking_layers = [int(layer) for layer in tracking_layers]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{manifest_path} contains invalid tracking layers.") from exc
    if tracking_layers != sorted(set(tracking_layers)) or not tracking_layers:
        raise ValueError(f"{manifest_path} tracking_layers must be sorted, unique, and non-empty.")
    tracking_directions = manifest["tracking_directions"]
    if not isinstance(tracking_directions, dict) or set(tracking_directions) != {
        str(layer) for layer in tracking_layers
    }:
        raise ValueError(f"{manifest_path} tracking_directions do not match tracking_layers.")

    condition_ids = manifest["expected_condition_ids"]
    record_ids = manifest["expected_record_ids"]
    if not isinstance(condition_ids, list) or not condition_ids:
        raise ValueError(f"{manifest_path} expected_condition_ids are invalid.")
    if not all(isinstance(condition_id, str) and condition_id for condition_id in condition_ids):
        raise ValueError(f"{manifest_path} expected_condition_ids are invalid.")
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError(f"{manifest_path} expected_condition_ids are invalid.")
    expected_record_ids = {
        f"{condition_id}__tracking_layer_{layer:02d}"
        for condition_id in condition_ids
        for layer in tracking_layers
    }
    if not isinstance(record_ids, list) or not all(isinstance(record_id, str) for record_id in record_ids):
        raise ValueError(f"{manifest_path} expected_record_ids are inconsistent with the manifest matrix.")
    if len(record_ids) != len(set(record_ids)) or set(record_ids) != expected_record_ids:
        raise ValueError(f"{manifest_path} expected_record_ids are inconsistent with the manifest matrix.")
    expected_record_count = manifest["expected_record_count"]
    if (
        not isinstance(expected_record_count, int)
        or isinstance(expected_record_count, bool)
        or expected_record_count != len(expected_record_ids)
    ):
        raise ValueError(f"{manifest_path} expected_record_count is inconsistent with the manifest matrix.")

    complete = manifest["complete"]
    if not isinstance(complete, bool):
        raise ValueError(f"{manifest_path} complete must be a boolean.")
    if not complete and not allow_incomplete_diagnostic:
        raise ValueError(
            f"Refusing to score incomplete steering output {raw_generations}; "
            "pass --allow-incomplete-diagnostic for diagnostic scoring only."
        )
    completed_record_count = manifest.get("completed_record_count")
    if complete:
        if completed_record_count != expected_record_count:
            raise ValueError(f"{manifest_path} completed_record_count does not equal expected_record_count.")
        if len(raw_rows) != expected_record_count:
            raise ValueError(
                f"Raw steering output has {len(raw_rows)} rows but the completed manifest expects "
                f"{expected_record_count}."
            )
    elif len(raw_rows) > expected_record_count:
        raise ValueError("Raw steering output has more rows than the incomplete manifest permits.")

    raw_hash = manifest.get("raw_generations_sha256")
    if complete and not raw_hash:
        raise ValueError(f"{manifest_path} is complete but has no raw_generations_sha256.")
    if raw_hash and file_sha256(raw_generations) != raw_hash:
        raise ValueError(f"{raw_generations} does not match the manifest raw_generations_sha256.")

    row_fields = (
        "construct_id",
        "steering_plan_sha256",
        "prompt_inventory_sha256",
        "run_config_hash",
        "construct_spec_hash",
        "injection_layer",
        "activation_site",
        "position_mode",
        "intervention_timing",
        "dtype",
    )
    seen_record_ids: set[str] = set()
    for row in raw_rows:
        record_id = row.get("record_id")
        condition_id = row.get("condition_id")
        tracking_layer = row.get("tracking_layer")
        if not isinstance(record_id, str) or not isinstance(condition_id, str):
            raise ValueError("Raw steering output contains a row without record_id or condition_id.")
        try:
            tracking_layer = int(tracking_layer)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Raw steering output record {record_id!r} has an invalid tracking_layer.") from exc
        expected_record_id = f"{condition_id}__tracking_layer_{tracking_layer:02d}"
        if record_id != expected_record_id or record_id not in expected_record_ids:
            raise ValueError(f"Raw steering output contains an unexpected record identity: {record_id!r}.")
        if record_id in seen_record_ids:
            raise ValueError(f"Raw steering output contains duplicate record identity: {record_id!r}.")
        seen_record_ids.add(record_id)
        if tracking_layer not in tracking_layers:
            raise ValueError(f"Raw steering output record {record_id!r} uses an unregistered tracking layer.")
        for field in row_fields:
            expected = manifest[field]
            actual = tracking_layer if field == "tracking_layer" else row.get(field)
            if actual != expected:
                raise ValueError(
                    f"Raw steering output record {record_id!r} is incompatible with the output manifest: "
                    f"{field} differs."
                )
        if row.get("model") != model:
            raise ValueError(f"Raw steering output record {record_id!r} has different model metadata.")
        if "revision" in model and row.get("model_revision") != model.get("revision"):
            raise ValueError(f"Raw steering output record {record_id!r} has different model revision metadata.")

    if complete and seen_record_ids != expected_record_ids:
        missing = sorted(expected_record_ids - seen_record_ids)
        raise ValueError(f"Raw steering output is missing expected records: {missing[:3]}.")
    return manifest, complete


def main() -> None:
    parser = argparse.ArgumentParser(description="Score construct steering generations.")
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--construct-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument(
        "--allow-incomplete-diagnostic",
        action="store_true",
        help="Permit scoring a manifest-marked partial run for diagnostics only.",
    )
    args = parser.parse_args()

    spec = load_construct_spec(args.construct_spec)
    raw_rows = _read_raw_rows(args.raw_generations, construct_id=spec.construct_id)
    output_manifest, manifest_complete = _load_and_validate_output_manifest(
        args.raw_generations,
        raw_rows,
        construct_id=spec.construct_id,
        construct_spec_hash=canonical_hash(spec.to_mapping()),
        allow_incomplete_diagnostic=args.allow_incomplete_diagnostic,
    )
    manipulation_records = []
    behavior_rows = []
    for raw in raw_rows:
        enriched = dict(raw)
        if raw.get("pre_projection") is not None and raw.get("post_projection") is not None:
            expected_shift = raw.get("expected_shift")
            if expected_shift is None:
                expected_shift = raw.get("physical_scale", 0.0)
            enriched.update(
                score_expected_observed_shift(
                    float(raw["pre_projection"]),
                    float(raw["post_projection"]),
                    float(expected_shift),
                )
            )
        manipulation_records.append(enriched)
        if (
            raw.get("tracking_layer") is None
            or raw.get("tracking_role") == "injection_immediate"
            or raw.get("tracking_layer") == raw.get("injection_layer")
        ):
            behavior_rows.append(raw)

    if not behavior_rows:
        raise ValueError("Raw steering output contains no injection-layer behavior rows.")

    rows = []
    observations = []
    behavior_condition_ids = set()
    for raw in behavior_rows:
        condition_id = str(raw["condition_id"])
        if condition_id in behavior_condition_ids:
            raise ValueError(f"Raw steering output contains duplicate behavior record for {condition_id!r}.")
        behavior_condition_ids.add(condition_id)
        task_metadata = dict(raw.get("task_metadata") or {})
        parsed = parse_behavior_output(
            raw.get("output_text", ""),
            parser_id=str(raw.get("parser_id") or spec.parsing_rules["parser_id"]),
            item_metadata=task_metadata,
            task_id=str(raw.get("task_id") or spec.independent_behavior_task["task_id"]),
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
            "tracking_layer": raw.get("tracking_layer", ""),
            "tracking_role": raw.get("tracking_role", ""),
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
    manipulation_path = args.output_dir / "manipulation_checks.csv"
    manipulation_fields = [
        "record_id",
        "condition_id",
        "prompt_id",
        "direction_kind",
        "direction_index",
        "dose",
        "physical_scale",
        "injection_layer",
        "tracking_layer",
        "tracking_direction_id",
        "tracking_direction_source",
        "tracking_role",
        "intervention_timing",
        "phase",
        "token_position",
        "injection_applied",
        "pre_projection",
        "post_projection",
        "observed_shift",
        "expected_shift",
        "expected_observed_difference",
        "absolute_error",
        "relative_error",
        "sign_agreement",
        "injection_calibration_projection_scale",
        "tracking_calibration",
        "tracking_calibration_projection_scale",
        "projection",
        "downstream_projection",
    ]
    with manipulation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manipulation_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manipulation_records)
    manipulation_summary = summarize_manipulation_records(manipulation_records)
    summary = {
        "construct_id": spec.construct_id,
        "confirmatory": manifest_complete and not args.allow_incomplete_diagnostic,
        "primary_outcome": spec.independent_behavior_task["primary_outcome"],
        "raw_record_count": len(raw_rows),
        "behavior_record_count": len(behavior_rows),
        "tracking_layers": sorted(
            {
                int(row["tracking_layer"])
                for row in raw_rows
                if row.get("tracking_layer") is not None
            }
        ),
        "target_direction_effect": asdict(effect),
        "uncertainty": effect_ci.to_mapping(),
        "manipulation_checks": manipulation_summary,
        "control_rows": {
            kind: sum(row["direction_kind"] == kind for row in rows)
            for kind in ("shuffled", "random")
        },
        "provenance": {
            "construct_spec_hash": canonical_hash(spec.to_mapping()),
            "raw_generations_sha256": file_sha256(args.raw_generations),
            "output_manifest": {
                "path": str(_output_manifest_path(args.raw_generations)),
                "complete": manifest_complete,
                "expected_record_count": output_manifest["expected_record_count"],
                "actual_record_count": len(raw_rows),
                "diagnostic_override": args.allow_incomplete_diagnostic,
            },
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
