#!/usr/bin/env python3
"""Inspect activation-run vectors available for downstream analysis."""

from pathlib import Path
import argparse
import json
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sae.config import SAEDatasetConfig
from sae.dataset import summarize_activation_dataset


def _parse_csv_set(value: str | None, *, as_int: bool = False):
    if not value:
        return None
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if as_int:
        return {int(part) for part in parts}
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize activation vectors available for analysis.")
    parser.add_argument("run_dirs", nargs="*", help="Activation run directories.")
    parser.add_argument("--config", help="Optional archived SAE dataset config JSON.")
    parser.add_argument("--layers", help="Comma-separated layer filter, overriding config layers.")
    parser.add_argument("--token-regions", help="Comma-separated token-region filter.")
    parser.add_argument("--activation-site", default=None, help="Activation site filter; defaults to config/resid_post.")
    parser.add_argument("--max-vectors", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        config = SAEDatasetConfig.from_json(args.config)
        run_dirs = [str(path) for path in config.activation_runs]
        layers = set(config.layers)
        token_regions = set(config.token_regions) if config.token_regions is not None else None
        prompt_metadata_filters = (
            {key: set(value) for key, value in config.prompt_metadata_filters.items()}
            if config.prompt_metadata_filters is not None
            else None
        )
        activation_site = args.activation_site if args.activation_site is not None else config.activation_site
        max_vectors = args.max_vectors if args.max_vectors is not None else config.max_vectors
    else:
        run_dirs = args.run_dirs
        layers = None
        token_regions = None
        prompt_metadata_filters = None
        activation_site = args.activation_site or "resid_post"
        max_vectors = args.max_vectors

    if args.layers:
        layers = _parse_csv_set(args.layers, as_int=True)
    if args.token_regions:
        token_regions = _parse_csv_set(args.token_regions)
    if not run_dirs:
        raise SystemExit("Provide at least one run directory or --config.")

    summary = summarize_activation_dataset(
        run_dirs,
        layers=layers,
        token_regions=token_regions,
        prompt_metadata_filters=prompt_metadata_filters,
        activation_site=activation_site,
        max_vectors=max_vectors,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
