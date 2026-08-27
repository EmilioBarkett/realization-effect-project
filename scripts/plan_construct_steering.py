#!/usr/bin/env python3
"""Freeze a construct's randomized steering/control condition matrix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.calibration import CalibrationResult  # noqa: E402
from construct_benchmark.config import load_construct_spec, load_run_config  # noqa: E402
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402
from construct_benchmark.prompts import load_prompt_records  # noqa: E402
from construct_benchmark.steering import (  # noqa: E402
    build_steering_conditions,
    random_control_direction,
    shuffled_label_direction,
)


def _existing_artifact_path(raw_path: str | Path, *, summary_path: Path) -> Path:
    """Resolve an artifact path stored in a readout summary."""

    path = Path(raw_path)
    if path.is_file():
        return path
    relative_to_summary = summary_path.parent / path
    if relative_to_summary.is_file():
        return relative_to_summary
    raise SystemExit(f"Readout summary references a missing direction artifact: {raw_path}")


def _build_tracking_directions(
    *,
    summary: dict,
    summary_path: Path,
    run_config,
    construct_id: str,
    selected_layer: int,
    injection_direction: Path,
) -> tuple[list[int], dict[str, dict]]:
    """Register the injection layer and later independently-read layers.

    Candidate directions are produced for every layer analyzed by the readout
    command.  If an older summary lacks a later candidate artifact, the
    fallback is retained as a same-vector diagnostic and is labelled as such;
    it is never presented as an independent downstream construct-state
    readout.
    """

    registered_layers = sorted(int(layer) for layer in run_config.activation["layers"])
    if selected_layer not in registered_layers:
        raise SystemExit("Readout selected a layer that is not registered in the run configuration.")
    tracking_layers = [layer for layer in registered_layers if layer >= selected_layer]
    candidate_artifacts = summary.get("candidate_directions", {})
    tracking_directions: dict[str, dict] = {}
    selected_calibration = summary.get("calibration")
    if not isinstance(selected_calibration, dict):
        selected_calibration = None
    for layer in tracking_layers:
        if layer == selected_layer:
            tracking_directions[str(layer)] = {
                "layer": layer,
                "direction_id": f"{construct_id}__injected_direction__layer_{layer:02d}",
                "path": str(injection_direction),
                "source": "injection_direction_train_only",
                "role": "injection_immediate",
                "source_split": "direction_train",
                "direction_sha256": file_sha256(injection_direction),
                "calibration": selected_calibration,
            }
            continue
        candidate = candidate_artifacts.get(str(layer))
        if isinstance(candidate, dict) and candidate.get("path"):
            path = _existing_artifact_path(candidate["path"], summary_path=summary_path)
            if candidate.get("source_split") != "direction_train":
                raise SystemExit(
                    f"Candidate direction for layer {layer} is not sourced from direction_train."
                )
            declared_hash = candidate.get("direction_sha256")
            if declared_hash and file_sha256(path) != declared_hash:
                raise SystemExit(f"Candidate direction hash does not match the readout summary: layer {layer}")
            candidate_calibration = candidate.get("calibration")
            if not isinstance(candidate_calibration, dict) or candidate_calibration.get("projection_scale") is None:
                raise SystemExit(
                    f"Candidate direction for layer {layer} is missing its frozen training calibration."
                )
            try:
                calibration_scale = float(candidate_calibration["projection_scale"])
            except (TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Candidate direction for layer {layer} has an invalid training calibration."
                ) from exc
            if not np.isfinite(calibration_scale) or calibration_scale <= 0:
                raise SystemExit(
                    f"Candidate direction for layer {layer} has a non-positive training calibration."
                )
            tracking_directions[str(layer)] = {
                "layer": layer,
                "direction_id": f"{construct_id}__construct_state__layer_{layer:02d}",
                "path": str(path),
                "source": "independent_train_only",
                "role": "downstream_construct_state",
                "source_split": "direction_train",
                "direction_sha256": file_sha256(path),
                "calibration": candidate_calibration,
            }
        else:
            tracking_directions[str(layer)] = {
                "layer": layer,
                "direction_id": f"{construct_id}__same_vector_diagnostic__layer_{layer:02d}",
                "path": str(injection_direction),
                "source": "same_vector_persistence_diagnostic",
                "role": "same_vector_persistence_diagnostic",
                "source_split": "direction_train",
                "direction_sha256": file_sha256(injection_direction),
                "calibration": None,
            }
    return tracking_layers, tracking_directions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic steering condition plan.")
    parser.add_argument("--prompt-inventory", type=Path, required=True)
    parser.add_argument("--construct-spec", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--readout-summary", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--pair-differences", type=Path, required=True)
    parser.add_argument("--direction-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = load_construct_spec(args.construct_spec)
    run_config = load_run_config(args.run_config)
    if spec.construct_id not in run_config.construct_ids:
        raise SystemExit(f"Run config does not contain construct_id={spec.construct_id!r}.")
    records = [
        record
        for record in load_prompt_records(args.prompt_inventory)
        if record.construct_id == spec.construct_id and record.split == "steering_eval"
    ]
    if not records:
        raise SystemExit(f"No steering_eval prompts found for construct_id={spec.construct_id!r}.")
    summary = json.loads(args.readout_summary.read_text(encoding="utf-8"))
    if summary.get("construct_id") != spec.construct_id:
        raise SystemExit("Readout summary construct_id does not match the construct specification.")
    selected_layer = int(
        summary.get("selected_layer", summary.get("layer", run_config.activation["layers"][0]))
    )
    if selected_layer not in set(run_config.activation["layers"]):
        raise SystemExit("Readout selected a layer that is not registered in the run configuration.")
    calibration = CalibrationResult(**summary["calibration"])
    steering = run_config.steering
    target_direction = np.load(args.direction).astype(np.float32, copy=False)
    pair_differences = np.load(args.pair_differences).astype(np.float32, copy=False)
    if pair_differences.ndim != 2 or pair_differences.shape[1] != target_direction.shape[0]:
        raise SystemExit("Pair differences and target direction have incompatible shapes.")
    tracking_layers, tracking_directions = _build_tracking_directions(
        summary=summary,
        summary_path=args.readout_summary,
        run_config=run_config,
        construct_id=spec.construct_id,
        selected_layer=selected_layer,
        injection_direction=args.direction,
    )
    args.direction_output_dir.mkdir(parents=True, exist_ok=True)
    shuffled_path = args.direction_output_dir / "shuffled_direction.npy"
    np.save(shuffled_path, shuffled_label_direction(pair_differences, seed=run_config.seed + 10_000))
    random_paths = []
    for index in range(int(steering["random_direction_count"])):
        path = args.direction_output_dir / f"random_direction_{index:02d}.npy"
        np.save(
            path,
            random_control_direction(
                target_direction.shape[0],
                seed=run_config.seed + 20_000 + index,
                orthogonal_to=target_direction,
            ),
        )
        random_paths.append(path)
    conditions = build_steering_conditions(
        [record.prompt_id for record in records],
        calibration,
        doses=steering["scales"],
        intervention_timing=steering["intervention_timing"],
        seed=run_config.seed,
        include_shuffled=True,
        random_direction_count=int(steering["random_direction_count"]),
    )
    payload = {
        "schema_version": run_config.schema_version,
        "plan_type": "construct_steering_conditions",
        "run_id": run_config.run_id,
        "construct_id": spec.construct_id,
        "model": run_config.model,
        "candidate_layers": list(run_config.activation["layers"]),
        "layer": selected_layer,
        "tracking_layers": tracking_layers,
        "tracking_directions": tracking_directions,
        "layer_selection": summary.get("layer_selection", {}),
        "activation_site": run_config.activation["activation_site"],
        "position_mode": steering["position_mode"],
        "intervention_timing": steering["intervention_timing"],
        "fixed_window": steering.get("fixed_window"),
        "calibration": asdict(calibration),
        "direction_paths": {
            "target": str(args.direction),
            "shuffled": str(shuffled_path),
            "random": [str(path) for path in random_paths],
        },
        "condition_count": len(conditions),
        "conditions": [asdict(condition) for condition in conditions],
        "provenance": {
            "run_config_hash": canonical_hash(run_config.to_mapping()),
            "construct_spec_hash": canonical_hash(spec.to_mapping()),
            "prompt_inventory_sha256": file_sha256(args.prompt_inventory),
            "readout_summary_sha256": file_sha256(args.readout_summary),
            "direction_sha256": file_sha256(args.direction),
            "pair_differences_sha256": file_sha256(args.pair_differences),
            "tracking_direction_hashes": {
                layer: entry["direction_sha256"] for layer, entry in tracking_directions.items()
            },
            "control_direction_hashes": {
                "shuffled": file_sha256(shuffled_path),
                "random": [file_sha256(path) for path in random_paths],
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("run_id", "construct_id", "condition_count")}, indent=2))


if __name__ == "__main__":
    main()
