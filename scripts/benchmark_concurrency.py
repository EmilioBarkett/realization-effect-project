#!/usr/bin/env python3
"""Benchmark a frozen workload across the registered concurrency rollout.

The command is model-independent.  ``--measurements`` is a JSON fixture or a
measurement export keyed by worker count; without it, a deterministic local
runner is used for smoke testing.  No model, API, or GPU is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.concurrency_benchmark import (  # noqa: E402
    ConcurrencyPolicy,
    benchmark_concurrency,
    freeze_workload,
    write_concurrency_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True, help="Frozen JSON request list/object.")
    parser.add_argument(
        "--expected-request-count",
        type=int,
        default=100,
        help="Required frozen workload size (default: 100); override only for an explicitly labelled smoke fixture.",
    )
    parser.add_argument("--measurements", type=Path, default=None, help="Optional JSON measurements keyed by worker count.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--worker-counts",
        nargs="+",
        default=["1,3,4"],
        help="Registered rollout counts, comma- or space-separated, for example 1,3,4,5.",
    )
    parser.add_argument("--include-five-worker", "--include-five", action="store_true")
    parser.add_argument("--hourly-rate", type=float, default=None)
    parser.add_argument("--material-improvement", type=float, default=0.10)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--max-retry-rate", type=float, default=0.10)
    parser.add_argument("--max-peak-vram-gb", type=float, default=None)
    parser.add_argument("--max-cost-per-request", type=float, default=None)
    parser.add_argument("--allow-output-identity-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc


def _measurement_map(path: Path | None) -> dict[int, Any]:
    if path is None:
        return {}
    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("measurements"), dict):
        payload = payload["measurements"]
    if isinstance(payload, list):
        payload = {
            str(item["worker_count"]): item
            for item in payload
            if isinstance(item, dict) and item.get("worker_count") is not None
        }
    if not isinstance(payload, dict):
        raise SystemExit("--measurements must contain an object keyed by worker count.")
    return {int(key): value for key, value in payload.items()}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workload = freeze_workload(_load_json(args.workload))
    measurements = _measurement_map(args.measurements)

    def runner(frozen_workload, worker_count):
        if measurements:
            if worker_count not in measurements:
                raise ValueError(f"No measurement was supplied for worker_count={worker_count}.")
            return measurements[worker_count]
        count = len(frozen_workload.request_ids)
        return {
            "elapsed_seconds": 60.0 / (1.0 + 0.75 * worker_count),
            "requested_requests": count,
            "valid_requests": count,
            "observations": 2 * count,
            "output_identities": {request_id: request_id for request_id in frozen_workload.request_ids},
            "hourly_rate": args.hourly_rate or 1.0,
            "stable": True,
        }

    try:
        worker_count_tokens = [
            token.strip()
            for value in args.worker_counts
            for token in value.split(",")
            if token.strip()
        ]
        worker_counts = tuple(int(item) for item in worker_count_tokens)
        policy = ConcurrencyPolicy(
            worker_counts=worker_counts,
            include_five_worker=args.include_five_worker,
            hourly_rate=args.hourly_rate,
            material_improvement=args.material_improvement,
            max_failure_rate=args.max_failure_rate,
            max_retry_rate=args.max_retry_rate,
            max_peak_vram_gb=args.max_peak_vram_gb,
            max_cost_per_request=args.max_cost_per_request,
            require_output_identity_match=not args.allow_output_identity_mismatch,
        )
        report = benchmark_concurrency(
            workload,
            runner,
            policy=policy,
            expected_request_count=args.expected_request_count,
        )
        report = write_concurrency_report(report, args.output, overwrite=args.overwrite)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["selection"]["selected_worker_count"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
