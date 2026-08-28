#!/usr/bin/env python3
"""Fail-closed zero-dose variation audit for a completed steering test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.behavioral_variation import audit_zero_dose_variation  # noqa: E402
from construct_benchmark.config import load_construct_spec  # noqa: E402
from construct_benchmark.manifests import canonical_hash  # noqa: E402

try:
    from scripts.score_construct_steering import (  # type: ignore
        _load_and_validate_output_manifest,
        _read_raw_rows,
    )
except ModuleNotFoundError:  # pragma: no cover - direct CLI fallback
    from score_construct_steering import (  # type: ignore
        _load_and_validate_output_manifest,
        _read_raw_rows,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Require usable target zero-dose behavioral variation before a full run."
    )
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--construct-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--minimum-valid", type=int, default=None)
    parser.add_argument("--minimum-distinct", type=int, default=None)
    parser.add_argument("--minimum-sd", type=float, default=None)
    parser.add_argument("--maximum-invalid", type=int, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        spec = load_construct_spec(args.construct_spec)
        raw_rows = _read_raw_rows(args.raw_generations, construct_id=spec.construct_id)
        manifest, complete = _load_and_validate_output_manifest(
            args.raw_generations,
            raw_rows,
            construct_id=spec.construct_id,
            construct_spec_hash=canonical_hash(spec.to_mapping()),
            allow_incomplete_diagnostic=False,
        )
        overrides = {
            key: value
            for key, value in {
                "minimum_zero_dose_valid": args.minimum_valid,
                "minimum_zero_dose_distinct": args.minimum_distinct,
                "minimum_zero_dose_sample_sd": args.minimum_sd,
                "maximum_zero_dose_invalid": args.maximum_invalid,
            }.items()
            if value is not None
        }
        report = audit_zero_dose_variation(raw_rows, spec, thresholds=overrides)
        report["manifest"] = {
            "complete": complete,
            "expected_record_count": manifest["expected_record_count"],
            "actual_record_count": len(raw_rows),
            "raw_generations": str(args.raw_generations),
        }
        report["construct_spec"] = str(args.construct_spec)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
