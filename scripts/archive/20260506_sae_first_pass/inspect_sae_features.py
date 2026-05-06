#!/usr/bin/env python3
"""Inspect trained SAE feature activations over an activation dataset."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sae.config import SAEDatasetConfig
from sae.inspection import inspect_sae_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect trained SAE feature associations.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--examples-per-feature", type=int, default=8)
    args = parser.parse_args()

    summary = inspect_sae_features(
        dataset_config=SAEDatasetConfig.from_json(args.dataset_config),
        checkpoint_path=Path(args.checkpoint),
        normalization_stats_path=Path(args.normalization_stats),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        device=args.device,
        min_group_size=args.min_group_size,
        top_n=args.top_n,
        examples_per_feature=args.examples_per_feature,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
