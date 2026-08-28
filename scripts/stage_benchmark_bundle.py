#!/usr/bin/env python3
"""Stage frozen benchmark inputs for a portable local or RunPod execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.staging import stage_bundle, validate_staging_bundle  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, default=[])
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    files: dict[str, Path] = {
        "inventory": args.inventory,
        "run_config": args.run_config,
        "analysis_spec": args.analysis_spec,
    }
    for spec in sorted(args.construct_spec, key=lambda item: item.name):
        stem = spec.stem.replace(".", "_")
        files[f"construct_{stem}"] = spec
    if args.validate_only:
        manifest = validate_staging_bundle(args.output_dir)
    else:
        manifest = stage_bundle(args.output_dir, files, bundle_id=args.bundle_id)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
