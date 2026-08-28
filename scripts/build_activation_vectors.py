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


DEFAULT_DIRECTION_SPLIT = "direction_train"


def _numpy_storage_dtype(value: str) -> np.dtype:
    if value == "float16":
        return np.dtype(np.float16)
    if value == "float32":
        return np.dtype(np.float32)
    raise ValueError("storage_dtype must be one of: float16, float32.")


def _parse_csv_set(value: str | None, *, as_int: bool = False):
    if not value:
        return None
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if as_int:
        return {int(part) for part in parts}
    return parts


def _validated_include_splits(value: str | None, *, allow_nontrain: bool) -> set[str]:
    splits = _parse_csv_set(value)
    if not splits:
        raise ValueError("--include-splits must contain at least one split name.")
    nontrain_splits = splits - {DEFAULT_DIRECTION_SPLIT}
    if nontrain_splits and not allow_nontrain:
        raise ValueError(
            "Direction construction is train-only by default. Pass --allow-nontrain-splits only "
            "for a registered diagnostic analysis."
        )
    return splits


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


def _filter_construct(activations, construct_id: str | None):
    construct_ids = {
        str(activation.metadata.get("construct_id", "")).strip()
        for activation in activations
    }
    construct_ids.discard("")
    if construct_id is None and len(construct_ids) > 1:
        raise ValueError(
            "Activation run contains multiple constructs. Pass --construct-id to build one "
            "construct-scoped direction at a time."
        )
    if construct_id is None:
        return activations
    return [
        activation
        for activation in activations
        if str(activation.metadata.get("construct_id", "")).strip() == construct_id
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build activation-vector directions.")
    parser.add_argument("--activation-run", required=True, help="Residual-stream run directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--construct-id", default=None, help="Construct namespace to analyze.")
    parser.add_argument("--layers", default=None, help="Comma-separated layer filter.")
    parser.add_argument("--token-regions", default="scenario", help="Comma-separated token regions.")
    parser.add_argument("--activation-site", default="resid_post")
    parser.add_argument(
        "--storage-dtype",
        default="float16",
        choices=("float16", "float32"),
        help="On-disk dtype for direction arrays; computation remains float32.",
    )
    parser.add_argument("--positive-role", default="realized_closed")
    parser.add_argument("--negative-role", default="paper_open")
    parser.add_argument(
        "--include-splits",
        default=DEFAULT_DIRECTION_SPLIT,
        help=(
            "Comma-separated metadata split values to include. Defaults to direction_train; "
            "pass an explicit value only for registered diagnostic analyses."
        ),
    )
    parser.add_argument(
        "--allow-nontrain-splits",
        action="store_true",
        help="Explicitly permit non-training splits for a registered diagnostic analysis.",
    )
    parser.add_argument("--exclude-splits", default=None, help="Comma-separated metadata split values to exclude.")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    activations = collect_prompt_mean_activations(
        args.activation_run,
        layers=_parse_csv_set(args.layers, as_int=True),
        token_regions=_parse_csv_set(args.token_regions),
        activation_site=args.activation_site,
    )
    include_splits = _validated_include_splits(
        args.include_splits,
        allow_nontrain=args.allow_nontrain_splits,
    )
    selected_activations = _filter_activations(
        activations,
        include_splits=include_splits,
        exclude_splits=_parse_csv_set(args.exclude_splits),
    )
    selected_activations = _filter_construct(selected_activations, args.construct_id)
    pair_rows, mean_direction = build_pair_directions(
        selected_activations,
        positive_role=args.positive_role,
        negative_role=args.negative_role,
    )
    if mean_direction is None:
        raise SystemExit("No complete prompt pairs found in activation metadata.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "mean_direction.npy",
        mean_direction.astype(_numpy_storage_dtype(args.storage_dtype), copy=False),
    )
    write_csv(
        output_dir / "pair_directions.csv",
        pair_rows,
        [
            "construct_id",
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
            "construct_id": args.construct_id,
            "prompt_count": len(selected_activations),
            "source_prompt_count": len(activations),
            "pair_count": len(pair_rows),
            "direction_file": str(output_dir / "mean_direction.npy"),
            "direction_norm": float(np.linalg.norm(mean_direction)),
            "positive_role": args.positive_role,
            "negative_role": args.negative_role,
            "include_splits": sorted(include_splits),
            "allow_nontrain_splits": args.allow_nontrain_splits,
            "exclude_splits": sorted(_parse_csv_set(args.exclude_splits) or []),
            "aggregation": "prompt_mean_over_selected_token_vectors",
            "storage_dtype": args.storage_dtype,
        },
    )
    print(f"built {len(pair_rows)} pair directions in {output_dir}")


if __name__ == "__main__":
    main()
