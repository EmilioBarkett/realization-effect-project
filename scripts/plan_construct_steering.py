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
from construct_benchmark.run_modes import resolve_run_mode  # noqa: E402
from construct_benchmark.steering import (  # noqa: E402
    build_steering_conditions,
    random_control_direction,
    shuffled_label_direction,
)


def _existing_artifact_path(raw_path: str | Path, *, summary_path: Path) -> Path:
    """Resolve an artifact path stored in a readout summary."""

    try:
        path = Path(raw_path)
    except TypeError as exc:
        raise SystemExit(f"Readout summary references an invalid artifact path: {raw_path!r}") from exc
    candidates = [path]
    if not path.is_absolute():
        candidates.append(summary_path.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(f"Readout summary references a missing direction artifact: {raw_path}")


def _declared_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SystemExit(f"{label} must declare a 64-character lowercase SHA-256 hash.")
    return value


def _validate_selected_artifact_binding(
    *,
    summary: dict,
    summary_path: Path,
    supplied_direction: Path,
    supplied_pair_differences: Path,
) -> tuple[Path, Path]:
    """Bind supplied steering inputs to the selected train-only readout artifacts."""

    selected = summary.get("direction")
    if not isinstance(selected, dict):
        raise SystemExit("Readout summary is missing its selected direction artifact record.")
    if selected.get("source_split") != "direction_train":
        raise SystemExit("Readout summary selected artifacts must be sourced from direction_train.")

    declared_direction = _existing_artifact_path(
        selected.get("path", ""),
        summary_path=summary_path,
    )
    declared_pair_differences = _existing_artifact_path(
        selected.get("pair_differences_path", ""),
        summary_path=summary_path,
    )
    expected_direction_hash = _declared_sha256(
        selected.get("direction_sha256"),
        label="Readout summary selected direction",
    )
    expected_pair_differences_hash = _declared_sha256(
        selected.get("pair_differences_sha256"),
        label="Readout summary selected pair differences",
    )
    if file_sha256(declared_direction) != expected_direction_hash:
        raise SystemExit("Selected direction artifact hash does not match the readout summary.")
    if file_sha256(declared_pair_differences) != expected_pair_differences_hash:
        raise SystemExit("Selected pair-differences artifact hash does not match the readout summary.")

    supplied_direction = _existing_artifact_path(supplied_direction, summary_path=summary_path)
    supplied_pair_differences = _existing_artifact_path(supplied_pair_differences, summary_path=summary_path)
    if supplied_direction != declared_direction:
        raise SystemExit("Supplied target direction does not match the readout summary's selected artifact path.")
    if supplied_pair_differences != declared_pair_differences:
        raise SystemExit(
            "Supplied pair-differences artifact does not match the readout summary's selected artifact path."
        )
    if file_sha256(supplied_direction) != expected_direction_hash:
        raise SystemExit("Supplied target direction hash does not match the readout summary.")
    if file_sha256(supplied_pair_differences) != expected_pair_differences_hash:
        raise SystemExit("Supplied pair-differences hash does not match the readout summary.")
    return supplied_direction, supplied_pair_differences


def _load_validated_array(path: Path, *, label: str, dimensions: int) -> np.ndarray:
    """Load one numeric steering artifact with fail-closed shape/value checks."""

    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{label} could not be loaded as a NumPy array: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise SystemExit(f"{label} must be a NumPy array.")
    if value.ndim != dimensions or value.size == 0:
        raise SystemExit(f"{label} must be a non-empty {dimensions}-dimensional array.")
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise SystemExit(f"{label} must contain real numeric values.")
    if not np.isfinite(value).all():
        raise SystemExit(f"{label} must contain only finite values.")
    if not np.any(value != 0):
        raise SystemExit(f"{label} must not be all zero.")
    try:
        converted = value.astype(np.float32, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"{label} could not be converted to float32: {exc}") from exc
    if not np.isfinite(converted).all():
        raise SystemExit(f"{label} must remain finite after float32 conversion.")
    if not np.any(converted != 0):
        raise SystemExit(f"{label} must remain non-zero after float32 conversion.")
    return converted


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
    parser.add_argument("--mode", choices=("test", "full"), default=None)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="On-disk dtype for control direction arrays; steering arithmetic reloads float32.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = load_construct_spec(args.construct_spec)
    run_config = load_run_config(args.run_config)
    mode_id, mode_config = resolve_run_mode(run_config, args.mode)
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
    direction_path, pair_differences_path = _validate_selected_artifact_binding(
        summary=summary,
        summary_path=args.readout_summary,
        supplied_direction=args.direction,
        supplied_pair_differences=args.pair_differences,
    )
    target_direction = _load_validated_array(
        direction_path,
        label="Target direction",
        dimensions=1,
    )
    pair_differences = _load_validated_array(
        pair_differences_path,
        label="Pair differences",
        dimensions=2,
    )
    if pair_differences.shape[0] < 2:
        raise SystemExit("Pair differences must contain at least two pair rows.")
    if pair_differences.ndim != 2 or pair_differences.shape[1] != target_direction.shape[0]:
        raise SystemExit("Pair differences and target direction have incompatible shapes.")
    tracking_layers, tracking_directions = _build_tracking_directions(
        summary=summary,
        summary_path=args.readout_summary,
        run_config=run_config,
        construct_id=spec.construct_id,
        selected_layer=selected_layer,
        injection_direction=direction_path,
    )
    args.direction_output_dir.mkdir(parents=True, exist_ok=True)
    storage_dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    shuffled_path = args.direction_output_dir / "shuffled_direction.npy"
    np.save(
        shuffled_path,
        shuffled_label_direction(pair_differences, seed=run_config.seed + 10_000).astype(storage_dtype, copy=False),
    )
    random_paths = []
    for index in range(int(steering["random_direction_count"])):
        path = args.direction_output_dir / f"random_direction_{index:02d}.npy"
        np.save(
            path,
            random_control_direction(
                target_direction.shape[0],
                seed=run_config.seed + 20_000 + index,
                orthogonal_to=target_direction,
            ).astype(storage_dtype, copy=False),
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
        "mode": mode_id,
        "purpose": mode_config["purpose"],
        "confirmatory": bool(mode_config["confirmatory"]),
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
        "direction_storage_dtype": args.storage_dtype,
        "direction_paths": {
            "target": str(direction_path),
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
            "direction_sha256": file_sha256(direction_path),
            "pair_differences_sha256": file_sha256(pair_differences_path),
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
