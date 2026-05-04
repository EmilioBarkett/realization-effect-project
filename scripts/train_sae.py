#!/usr/bin/env python3
"""Train the first local SAE scaffold from validated activation runs."""

from pathlib import Path
import argparse
import json
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sae.config import SAEDatasetConfig, SAETrainingConfig
from sae.training import train_sae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local sparse autoencoder over activation vectors.")
    parser.add_argument("--dataset-config", required=True, help="SAE dataset config JSON.")
    parser.add_argument("--training-config", help="Optional SAE training config JSON.")
    parser.add_argument("--output-dir", help="Override training output directory.")
    parser.add_argument("--max-steps", type=int, help="Override max training steps.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--d-sae", type=int, help="Override SAE hidden width.")
    parser.add_argument("--top-k", type=int, help="Override top-k sparse activation count.")
    parser.add_argument("--normalization", help="Override normalization: mean_center_global_norm or none.")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, or mps.")
    args = parser.parse_args()

    dataset_config = SAEDatasetConfig.from_json(args.dataset_config)
    training_config = (
        SAETrainingConfig.from_json(args.training_config)
        if args.training_config
        else SAETrainingConfig()
    )
    overrides = training_config.to_json_dict()
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.d_sae is not None:
        overrides["d_sae"] = args.d_sae
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.normalization:
        overrides["normalization"] = args.normalization
    if args.device:
        overrides["device"] = args.device
    training_config = SAETrainingConfig.from_json_dict(overrides)

    result = train_sae(dataset_config, training_config)
    print(
        json.dumps(
            {
                "steps": result.steps,
                "model_config": result.model_config.to_json_dict(),
                "final_metrics": result.final_metrics,
                "checkpoint_path": str(result.checkpoint_path),
                "manifest_path": str(result.manifest_path),
                "normalization_stats_path": str(result.normalization_stats_path),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
