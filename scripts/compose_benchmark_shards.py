#!/usr/bin/env python3
"""Validate and compose complete distributed benchmark worker outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.composition import (  # noqa: E402
    compose_worker_outputs,
    validate_worker_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed composition of immutable benchmark worker output files."
    )
    parser.add_argument(
        "--worker-output",
        action="append",
        dest="worker_outputs",
        default=[],
        help="Worker JSONL/CSV output; repeat once per worker.",
    )
    parser.add_argument(
        "--worker-manifest",
        action="append",
        dest="worker_manifests",
        default=[],
        help="Optional manifests matching --worker-output order.",
    )
    parser.add_argument(
        "--worker",
        action="append",
        default=[],
        metavar="OUTPUT::MANIFEST",
        help="Explicit output/manifest pair; repeat once per worker.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--parent-manifest", type=Path, default=None)
    parser.add_argument("--expected-request-ids", default=None)
    parser.add_argument("--expected-observation-ids", default=None)
    parser.add_argument("--run-mode", dest="target_run_mode", default=None)
    confirmatory = parser.add_mutually_exclusive_group()
    confirmatory.add_argument("--confirmatory", dest="target_confirmatory", action="store_true")
    confirmatory.add_argument("--non-confirmatory", dest="target_confirmatory", action="store_false")
    parser.set_defaults(target_confirmatory=None)
    parser.add_argument("--allow-incomplete", action="store_true", help="Diagnostics only; composition still requires complete inputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print a report without writing.")
    return parser


def _workers(args: argparse.Namespace) -> list[object]:
    workers: list[object] = []
    if args.worker_manifests and len(args.worker_manifests) != len(args.worker_outputs):
        raise SystemExit("--worker-manifest must be repeated exactly once per --worker-output.")
    for index, output in enumerate(args.worker_outputs):
        manifest = args.worker_manifests[index] if args.worker_manifests else None
        workers.append({"output": output, "manifest": manifest})
    for item in args.worker:
        if "::" not in item:
            raise SystemExit("Each --worker value must use OUTPUT::MANIFEST.")
        output, manifest = item.split("::", 1)
        if not output or not manifest:
            raise SystemExit("Each --worker value must include both output and manifest paths.")
        workers.append((output, manifest))
    if not workers:
        raise SystemExit("Provide at least one --worker-output or --worker pair.")
    return workers


def _json_default(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main() -> None:
    args = _parser().parse_args()
    workers = _workers(args)
    common = {
        "expected_request_ids": args.expected_request_ids,
        "expected_observation_ids": args.expected_observation_ids,
        "parent_manifest": args.parent_manifest,
        "target_run_mode": args.target_run_mode,
        "target_confirmatory": args.target_confirmatory,
        "allow_incomplete": args.allow_incomplete,
    }
    if args.dry_run:
        report = validate_worker_outputs(workers, **common)
        report.pop("_validated_workers", None)
    else:
        report = compose_worker_outputs(
            workers,
            args.output,
            manifest_output=args.manifest_output,
            **common,
        )
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
