#!/usr/bin/env python3
"""Run a manifest-backed, bounded parallel benchmark campaign.

The command is intentionally a stage orchestrator rather than a scientific
stage implementation.  Use ``--fake-model`` for deterministic local checks,
``--adapter gpu`` for the real construct-pure GPU worker, ``--adapter
module:callable`` for a Python stage adapter, or
``--worker-command`` for an argv-only stage command using the tokens
``{shard_manifest}``, ``{worker_manifest}``, ``{output_path}``,
``{checkpoint}``, ``{worker_id}``, ``{stage}``, and ``{log_path}``.
Shutdown commands additionally support ``{status}``, ``{reason}``,
``{campaign}``, ``{terminal_report}``, and ``{output}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.parallel_executor import (  # noqa: E402
    ParallelExecutor,
    ParallelExecutorError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reusable parallel benchmark stage with durable recovery manifests."
    )
    parser.add_argument("--campaign", default="parallel_campaign", help="Campaign ID or JSON campaign file.")
    parser.add_argument("--inventory", type=Path, required=True, help="JSON, JSONL, or CSV request inventory.")
    parser.add_argument("--run-config", type=Path, default=None, help="Optional run-config JSON used for identity.")
    parser.add_argument("--stage", default="benchmark", help="Opaque stage ID passed to the adapter.")
    parser.add_argument("--worker-count", type=int, default=None, help="Concurrent worker slots; otherwise use run-config topology or 1.")
    parser.add_argument("--resume", action="store_true", help="Resume the matching campaign state/checkpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Plan manifests without launching workers.")
    parser.add_argument("--fake-model", action="store_true", help="Use the deterministic no-GPU fake adapter.")
    parser.add_argument("--adapter", default=None, help="Use gpu, or a Python adapter as module:callable.")
    parser.add_argument(
        "--worker-command",
        "--command",
        nargs="+",
        default=None,
        help="argv-only worker command; use fixed {shard_manifest}/{output_path} tokens.",
    )
    parser.add_argument("--stagger", type=float, default=0.0, help="Seconds between worker launches/model loads.")
    parser.add_argument("--stall", type=float, default=0.0, help="Fake/Python worker stall duration for fault tests.")
    parser.add_argument("--stall-after", type=int, default=None, help="Stall after this many fake requests.")
    parser.add_argument("--crash-after", type=int, default=None, help="Inject a fake worker exit after N requests.")
    parser.add_argument("--crash-once", action="store_true", help="Make --crash-after apply only on the first attempt.")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum worker restarts after an initial failure.")
    parser.add_argument("--idle-timeout", type=float, default=900.0, help="Seconds without useful output progress.")
    parser.add_argument("--poll-interval", type=float, default=0.2, help="Worker supervision polling interval.")
    parser.add_argument("--budget", "--hard-ceiling", dest="budget", type=float, default=None, help="Hard GPU budget ceiling in USD.")
    parser.add_argument("--reserve", "--budget-reserve", dest="reserve", type=float, default=0.0, help="USD held in reserve and never committed to launches.")
    parser.add_argument("--rate", "--gpu-hourly-rate", dest="rate", type=float, default=0.0, help="Configured GPU hourly rate in USD.")
    parser.add_argument("--estimate-seconds", type=float, default=3600.0, help="Per-worker launch cost estimate horizon.")
    parser.add_argument(
        "--shutdown-command",
        "--pod-stop-command",
        nargs="+",
        default=None,
        help="Optional argv-only terminal shutdown command; runs after a durable success/failure report, never in dry-run.",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=30.0,
        help="Maximum seconds allowed for the optional shutdown command.",
    )
    parser.add_argument("--confirmatory", action="store_true", help="Mark this stage run as confirmatory in identities.")
    parser.add_argument("--run-mode", default=None, help="Opaque run mode, otherwise read from run-config or use test.")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True, help="Campaign output directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = ParallelExecutor(
            campaign=args.campaign,
            inventory=args.inventory,
            run_config=args.run_config,
            stage=args.stage,
            worker_count=args.worker_count,
            resume=args.resume,
            dry_run=args.dry_run,
            fake_model=args.fake_model,
            adapter=args.adapter,
            worker_command=args.worker_command,
            stagger_seconds=args.stagger,
            stall_seconds=args.stall,
            stall_after=args.stall_after,
            crash_after=args.crash_after,
            crash_once=args.crash_once,
            max_retries=args.max_retries,
            idle_timeout_seconds=args.idle_timeout,
            poll_interval_seconds=args.poll_interval,
            hard_ceiling_usd=args.budget,
            reserve_usd=args.reserve,
            gpu_hourly_rate_usd=args.rate,
            worker_estimate_seconds=args.estimate_seconds,
            shutdown_command=args.shutdown_command,
            shutdown_timeout_seconds=args.shutdown_timeout,
            confirmatory=args.confirmatory if args.confirmatory else None,
            run_mode=args.run_mode,
            output=args.output,
        ).run()
    except (ParallelExecutorError, ValueError, FileNotFoundError, OSError) as exc:
        error = {"status": "error", "error": str(exc), "output": str(args.output)}
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") in {"success", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
