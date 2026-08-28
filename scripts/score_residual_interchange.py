#!/usr/bin/env python3
"""Validate and summarize a matched-episode residual-interchange run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.causal_patching import validate_residual_interchange_output  # noqa: E402


def _read_rows(output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def build_summary(output: Path, *, allow_incomplete_diagnostic: bool = False) -> dict[str, Any]:
    manifest = validate_residual_interchange_output(
        output,
        allow_incomplete_diagnostic=allow_incomplete_diagnostic,
    )
    rows = _read_rows(output)
    observations = [
        observation
        for row in rows
        for observation in row.get("observations", [])
    ]
    cross_condition = [
        observation
        for observation in observations
        if observation.get("patch_direction") in {"positive_to_negative", "negative_to_positive"}
    ]
    same_condition = [
        observation
        for observation in observations
        if observation.get("patch_direction") in {"positive_to_positive", "negative_to_negative"}
    ]
    return {
        "schema_version": manifest["schema_version"],
        "manifest_type": manifest["manifest_type"],
        "complete": bool(manifest["complete"]),
        "confirmatory": bool(manifest.get("confirmatory", False)),
        "request_count": len(rows),
        "observation_count": len(observations),
        "cross_condition_observation_count": len(cross_condition),
        "same_condition_observation_count": len(same_condition),
        "cross_condition_output_changed_count": sum(
            observation.get("patched_output") != observation.get("receiver_baseline_output")
            for observation in cross_condition
        ),
        "same_condition_output_changed_count": sum(
            observation.get("patched_output") != observation.get("receiver_baseline_output")
            for observation in same_condition
        ),
        "layers": manifest["layers"],
        "model": manifest.get("model"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--allow-incomplete-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    summary = build_summary(
        args.raw_output,
        allow_incomplete_diagnostic=args.allow_incomplete_diagnostic,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
