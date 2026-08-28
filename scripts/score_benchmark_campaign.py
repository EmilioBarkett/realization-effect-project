#!/usr/bin/env python3
"""Validate and assemble model-independent B/R/C/S campaign scores.

The JSON config names available stage artifacts.  Existing repository
validators and scorers are used by the default adapters; this command only
writes a separate frozen orchestration report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.scoring_pipeline import (  # noqa: E402
    build_scoring_report_from_config,
    normalize_stage_code,
    write_scoring_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--pipeline-config", dest="config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("complete", "diagnostic"), default=None)
    parser.add_argument("--campaign", type=Path, default=None, help="Override campaign path in the config.")
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="CODE=PATH",
        help="Override one stage input, for example B=/tmp/behavior.jsonl.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _stage_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--stage must use CODE=PATH: {value!r}")
        code, path = value.split("=", 1)
        code = normalize_stage_code(code)
        if not path.strip():
            raise ValueError(f"--stage path is empty for {code}.")
        if code in overrides:
            raise ValueError(f"Duplicate --stage override for {code}.")
        overrides[code] = path
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_scoring_report_from_config(
            args.config,
            mode=args.mode,
            campaign_path=args.campaign,
            stage_overrides=_stage_overrides(args.stage),
        )
        report = write_scoring_report(report, args.output, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["report_mode"] == "complete" and not report["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
