#!/usr/bin/env python3
"""Validate a gated Waves 2--4 execution campaign without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.campaign import confirmatory_execution_report  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a Waves 2-4 confirmatory campaign without loading model weights."
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=_ROOT / "configs/construct_benchmark/confirmatory_campaigns/waves2_4_confirmatory_v1.json",
    )
    parser.add_argument("--mode", choices=("test", "full"), default="test")
    parser.add_argument("--waves", nargs="+", type=int, choices=(2, 3, 4), default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = confirmatory_execution_report(args.campaign, mode=args.mode, waves=args.waves)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
