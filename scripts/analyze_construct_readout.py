#!/usr/bin/env python3
"""Build one train-only direction and evaluate its frozen held-out readout."""

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

from activation_analysis.vector_analysis import collect_prompt_mean_activations, write_csv  # noqa: E402
from construct_benchmark.calibration import estimate_projection_scale  # noqa: E402
from construct_benchmark.config import load_construct_spec  # noqa: E402
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402
from construct_benchmark.readout import (  # noqa: E402
    estimate_train_direction,
    evaluate_heldout_readout,
    evaluate_validation_readout,
)
from construct_benchmark.uncertainty import bootstrap_readout_margin_ci  # noqa: E402


def _csv_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    return result or None


def _parse_layers(layer: int | None, layers: str | None) -> list[int]:
    if (layer is None) == (layers is None):
        raise SystemExit("Provide exactly one of --layer or --layers.")
    values = [layer] if layer is not None else [int(item.strip()) for item in str(layers).split(",") if item.strip()]
    if not values or any(value < 1 for value in values) or len(set(values)) != len(values):
        raise SystemExit("Layers must be distinct positive integers.")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a construct-scoped held-out projection readout.")
    parser.add_argument("--activation-run", type=Path, required=True)
    parser.add_argument("--construct-spec", type=Path, required=True)
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layer", type=int, default=None, help="One frozen layer for a diagnostic run.")
    layer_group.add_argument("--layers", default=None, help="Comma-separated candidates selected on validation.")
    parser.add_argument(
        "--layer-selection",
        choices=("validation_max_margin", "fixed"),
        default="validation_max_margin",
        help="Selection rule when multiple candidate layers are supplied.",
    )
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--token-regions", default="scenario,task")
    parser.add_argument("--calibration-method", choices=("neutral", "within_condition"), default="neutral")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument(
        "--allow-incomplete-run",
        action="store_true",
        help="Allow diagnostic analysis of an activation run stopped before its selected inventory completed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    layers = _parse_layers(args.layer, args.layers)
    spec = load_construct_spec(args.construct_spec)
    activation_manifest_path = args.activation_run / "manifest.json"
    activation_manifest = json.loads(activation_manifest_path.read_text(encoding="utf-8"))
    execution = activation_manifest.get("execution", {})
    if execution.get("complete") is False and not args.allow_incomplete_run:
        raise SystemExit(
            "Activation run stopped before its selected inventory completed; "
            "pass --allow-incomplete-run for diagnostic analysis only."
        )
    activations = collect_prompt_mean_activations(
        args.activation_run,
        layers=set(layers),
        token_regions=_csv_set(args.token_regions),
        activation_site=args.activation_site,
    )
    if not activations:
        raise SystemExit("No activations matched the requested layer/token-region filters.")

    candidate_results = []
    estimates = {}
    calibrations = {}
    layer_activations = {}
    for layer in layers:
        selected = [
            activation
            for activation in activations
            if activation.layer == layer
            or (activation.layer is None and int(activation.metadata.get("layer", layer)) == layer)
        ]
        if not selected:
            raise SystemExit(f"No activations found for candidate layer {layer}.")
        estimate = estimate_train_direction(
            selected,
            construct_id=spec.construct_id,
            positive_condition_id=spec.positive_condition_id,
            negative_condition_id=spec.negative_condition_id,
        )
        calibration = estimate_projection_scale(
            selected,
            estimate.direction,
            construct_id=spec.construct_id,
            method=args.calibration_method,
        )
        validation = evaluate_validation_readout(
            selected,
            estimate,
            projection_scale=calibration.projection_scale,
        )
        estimates[layer] = estimate
        calibrations[layer] = calibration
        layer_activations[layer] = selected
        candidate_results.append(
            {
                "layer": layer,
                "mean_standardized_margin": validation.mean_standardized_margin,
                "pair_accuracy": validation.pair_accuracy,
                "pair_count": validation.pair_count,
            }
        )

    if args.layer_selection == "fixed":
        selected_layer = layers[0]
    else:
        selected_layer = min(
            layers,
            key=lambda layer: (
                -next(item["mean_standardized_margin"] for item in candidate_results if item["layer"] == layer),
                layer,
            ),
        )
    estimate = estimates[selected_layer]
    calibration = calibrations[selected_layer]
    selected_activations = layer_activations[selected_layer]
    readout = evaluate_heldout_readout(
        selected_activations,
        estimate,
        projection_scale=calibration.projection_scale,
    )
    readout_ci = bootstrap_readout_margin_ci(
        readout.margins,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    direction_path = args.output_dir / "mean_direction.npy"
    pair_differences_path = args.output_dir / "pair_differences.npy"
    np.save(direction_path, estimate.direction)
    np.save(pair_differences_path, estimate.pair_differences)
    write_csv(
        args.output_dir / "heldout_pair_margins.csv",
        [asdict(margin) for margin in readout.margins],
        [
            "pair_id",
            "positive_prompt_id",
            "negative_prompt_id",
            "positive_projection",
            "negative_projection",
            "standardized_margin",
        ],
    )
    summary = {
        "construct_id": spec.construct_id,
        "candidate_layers": layers,
        "selected_layer": selected_layer,
        "layer_selection": {
            "rule": args.layer_selection,
            "selection_split": (
                "direction_validation" if args.layer_selection == "validation_max_margin" else "none"
            ),
            "selection_metric": "mean_standardized_margin",
            "candidates": candidate_results,
        },
        "activation_site": args.activation_site,
        "token_regions": sorted(_csv_set(args.token_regions) or []),
        "direction": {
            "source_split": "direction_train",
            "pair_count": estimate.pair_count,
            "positive_condition_id": estimate.positive_condition_id,
            "negative_condition_id": estimate.negative_condition_id,
            "norm": float(np.linalg.norm(estimate.direction)),
            "path": str(direction_path),
            "pair_differences_path": str(pair_differences_path),
        },
        "calibration": asdict(calibration),
        "readout": {
            "split": readout.split,
            "pair_count": readout.pair_count,
            "mean_standardized_margin": readout.mean_standardized_margin,
            "pair_accuracy": readout.pair_accuracy,
            "uncertainty": readout_ci.to_mapping(),
        },
        "provenance": {
            "construct_spec_hash": canonical_hash(spec.to_mapping()),
            "activation_manifest_sha256": file_sha256(activation_manifest_path),
            "execution": {
                "run_mode": execution.get("run_mode"),
                "confirmatory": execution.get("confirmatory"),
                "complete": execution.get("complete", True),
                "allow_incomplete_run": args.allow_incomplete_run,
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
