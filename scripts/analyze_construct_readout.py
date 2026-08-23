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
from construct_benchmark.readout import estimate_train_direction, evaluate_heldout_readout  # noqa: E402


def _csv_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    result = {item.strip() for item in value.split(",") if item.strip()}
    return result or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a construct-scoped held-out projection readout.")
    parser.add_argument("--activation-run", type=Path, required=True)
    parser.add_argument("--construct-spec", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--token-regions", default="scenario,task")
    parser.add_argument("--calibration-method", choices=("neutral", "within_condition"), default="neutral")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.layer < 1:
        raise SystemExit("--layer must be a positive 1-based layer number.")
    spec = load_construct_spec(args.construct_spec)
    activations = collect_prompt_mean_activations(
        args.activation_run,
        layers={args.layer},
        token_regions=_csv_set(args.token_regions),
        activation_site=args.activation_site,
    )
    estimate = estimate_train_direction(
        activations,
        construct_id=spec.construct_id,
        positive_condition_id=spec.positive_condition_id,
        negative_condition_id=spec.negative_condition_id,
    )
    calibration = estimate_projection_scale(
        activations,
        estimate.direction,
        construct_id=spec.construct_id,
        method=args.calibration_method,
    )
    readout = evaluate_heldout_readout(
        activations,
        estimate,
        projection_scale=calibration.projection_scale,
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
        "layer": args.layer,
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
        },
        "provenance": {
            "construct_spec_hash": canonical_hash(spec.to_mapping()),
            "activation_manifest_sha256": file_sha256(args.activation_run / "manifest.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
