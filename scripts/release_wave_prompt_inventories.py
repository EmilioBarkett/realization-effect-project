#!/usr/bin/env python3
"""Release frozen Wave 2--4 prompt inventories as confirmatory inputs.

The command copies exact, hash-verified engineering inventories into new
immutable release directories. It does not release model-side execution;
the campaign validator continues to enforce the Wave 1 and precision gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.campaign import release_wave_prompt_inventories  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Release frozen Wave 2-4 prompt inventories as confirmatory inputs."
    )
    parser.add_argument("--waves", nargs="+", type=int, choices=(2, 3, 4), default=(2, 3, 4))
    parser.add_argument(
        "--registry",
        type=Path,
        default=_ROOT / "configs/construct_benchmark/construct_registry_v1.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_ROOT / "results/benchmark/prompt_inventories",
        help="Root containing the frozen engineering wave inventories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_ROOT / "results/benchmark/prompt_inventories",
    )
    parser.add_argument(
        "--released-by",
        required=True,
        help="Human or project authority authorizing prompt-input release.",
    )
    parser.add_argument(
        "--release-statement",
        required=True,
        help="Short attestation explaining the scope of this prompt-input release.",
    )
    parser.add_argument("--release-date", default=date.today().isoformat())
    return parser


def main() -> None:
    args = _parser().parse_args()
    source_manifest_paths = {
        wave: args.source_root / f"wave{wave}_four_construct_full_luna_v1" / "inventory_manifest.json"
        for wave in dict.fromkeys(args.waves)
    }
    summaries = release_wave_prompt_inventories(
        waves=args.waves,
        registry_path=args.registry,
        source_manifest_paths=source_manifest_paths,
        output_root=args.output_root,
        released_by=args.released_by,
        release_statement=args.release_statement,
        release_date=args.release_date,
    )
    print(
        json.dumps(
            {
                "status": "released",
                "confirmatory_prompt_inputs": True,
                "model_execution_release": False,
                "waves": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
