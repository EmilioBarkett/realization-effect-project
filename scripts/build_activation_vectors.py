#!/usr/bin/env python3
"""Build paired activation-vector directions from residual-stream runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.vector_analysis import (
    build_pair_directions,
    collect_prompt_mean_activations,
    write_csv,
    write_json,
)


def _parse_csv_set(value: str | None, *, as_int: bool = False):
    if not value:
        return None
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if as_int:
        return {int(part) for part in parts}
    return parts


def _metadata_value(metadata: dict, key: str) -> str:
    value = metadata.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def _filter_activations(activations, *, include_splits: set[str] | None, exclude_splits: set[str] | None):
    if include_splits is None and exclude_splits is None:
        return activations
    filtered = []
    for activation in activations:
        split = _metadata_value(activation.metadata, "split")
        if include_splits is not None and split not in include_splits:
            continue
        if exclude_splits is not None and split in exclude_splits:
            continue
        filtered.append(activation)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Build activation-vector directions.")
    parser.add_argument("--activation-run", required=True, help="Residual-stream run directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--layers", default=None, help="Comma-separated layer filter.")
    parser.add_argument("--token-regions", default="scenario", help="Comma-separated token regions.")
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument("--positive-role", default="realized_closed")
    parser.add_argument("--negative-role", default="paper_open")
    parser.add_argument("--include-splits", default=None, help="Comma-separated metadata split values to include.")
    parser.add_argument("--exclude-splits", default=None, help="Comma-separated metadata split values to exclude.")
    args = parser.parse_args()

    activations = collect_prompt_mean_activations(
        args.activation_run,
        layers=_parse_csv_set(args.layers, as_int=True),
        token_regions=_parse_csv_set(args.token_regions),
        activation_site=args.activation_site,
    )
    selected_activations = _filter_activations(
        activations,
        include_splits=_parse_csv_set(args.include_splits),
        exclude_splits=_parse_csv_set(args.exclude_splits),
    )
    pair_rows, mean_direction = build_pair_directions(
        selected_activations,
        positive_role=args.positive_role,
        negative_role=args.negative_role,
    )
    if mean_direction is None:
        raise SystemExit("No complete prompt pairs found in activation metadata.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "mean_direction.npy", mean_direction)
    write_csv(
        output_dir / "pair_directions.csv",
        pair_rows,
        [
            "pair_id",
            "positive_prompt_id",
            "negative_prompt_id",
            "positive_role",
            "negative_role",
            "domain",
            "split",
            "outcome_valence",
            "amount_bucket",
            "risk_context",
            "behavior_target",
            "direction_norm",
        ],
    )
    write_json(
        output_dir / "summary.json",
        {
            "activation_run": args.activation_run,
            "prompt_count": len(selected_activations),
            "source_prompt_count": len(activations),
            "pair_count": len(pair_rows),
            "direction_file": str(output_dir / "mean_direction.npy"),
            "direction_norm": float(np.linalg.norm(mean_direction)),
            "positive_role": args.positive_role,
            "negative_role": args.negative_role,
            "include_splits": sorted(_parse_csv_set(args.include_splits) or []),
            "exclude_splits": sorted(_parse_csv_set(args.exclude_splits) or []),
            "aggregation": "prompt_mean_over_selected_token_vectors",
        },
    )
    print(f"built {len(pair_rows)} pair directions in {output_dir}")


if __name__ == "__main__":
    main()
