#!/usr/bin/env python3
"""Inspect activation-run vectors available for downstream analysis."""

from pathlib import Path
import argparse
import json
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.activation_store import summarize_activation_dataset


def _parse_csv_set(value: str | None, *, as_int: bool = False):
    if not value:
        return None
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if as_int:
        return {int(part) for part in parts}
    return parts


def _load_dataset_config(path: str | Path) -> dict:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    activation_runs = data.get("activation_runs")
    if not isinstance(activation_runs, list) or not activation_runs:
        raise ValueError(f"{config_path} must define a non-empty activation_runs list.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize activation vectors available for analysis.")
    parser.add_argument("run_dirs", nargs="*", help="Activation run directories.")
    parser.add_argument("--config", help="Optional activation dataset config JSON.")
    parser.add_argument("--layers", help="Comma-separated layer filter, overriding config layers.")
    parser.add_argument("--token-regions", help="Comma-separated token-region filter.")
    parser.add_argument("--activation-site", default=None, help="Activation site filter; defaults to config/resid_post.")
    parser.add_argument("--max-vectors", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        config = _load_dataset_config(args.config)
        run_dirs = [str(path) for path in config["activation_runs"]]
        configured_layers = config.get("layers")
        layers = {int(layer) for layer in configured_layers} if configured_layers is not None else None
        configured_regions = config.get("token_regions")
        token_regions = set(configured_regions) if configured_regions is not None else None
        configured_filters = config.get("prompt_metadata_filters")
        prompt_metadata_filters = (
            {str(key): {str(item) for item in values} for key, values in configured_filters.items()}
            if isinstance(configured_filters, dict)
            else None
        )
        activation_site = args.activation_site if args.activation_site is not None else config.get("activation_site", "resid_post")
        max_vectors = args.max_vectors if args.max_vectors is not None else config.get("max_vectors")
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
