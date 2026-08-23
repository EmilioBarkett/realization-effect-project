#!/usr/bin/env python3
"""Validate the frozen construct registry against its specified JSON specs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import load_construct_specs  # noqa: E402
from construct_benchmark.registry import (  # noqa: E402
    load_construct_registry,
    validate_registry_against_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the RSC construct registry.")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--construct-spec",
        type=Path,
        nargs="*",
        default=None,
        help="Optional specified construct specs; defaults to paths listed in the registry.",
    )
    args = parser.parse_args()

    registry = load_construct_registry(args.registry)
    spec_paths = args.construct_spec
    if spec_paths is None:
        spec_paths = [
            args.registry.parent / entry.spec_path
            for entry in registry.entries
            if entry.status == "specified"
        ]
    specs = load_construct_specs(spec_paths)
    summary = validate_registry_against_specs(registry, specs)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
