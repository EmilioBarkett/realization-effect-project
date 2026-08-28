#!/usr/bin/env python3
"""Inspect durable parallel campaign and worker manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.parallel_executor import InvalidCheckpointError, inspect_campaign  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a parallel benchmark campaign without launching workers.")
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true", help="Refresh until the campaign reaches a terminal state.")
    parser.add_argument("--interval", type=float, default=1.0, help="Watch refresh interval in seconds.")
    parser.add_argument("--max-cycles", type=int, default=None, help="Bound --watch refreshes for automation/tests.")
    parser.add_argument("--json", action="store_true", help="Emit one JSON snapshot per refresh.")
    return parser


def _print_snapshot(snapshot: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    progress = snapshot.get("progress", {})
    print(
        f"{snapshot.get('campaign_id')}: {snapshot.get('status')} "
        f"requests={progress.get('completed_request_count_observed', 0)}/"
        f"{progress.get('expected_request_count', '?')} "
        f"observations={progress.get('completed_observation_count_observed', 0)}/"
        f"{progress.get('expected_observation_count', '?')} "
        f"stale={','.join(snapshot.get('stale_workers', [])) or 'none'}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cycles = 0
    while True:
        try:
            snapshot = inspect_campaign(args.output)
        except (InvalidCheckpointError, OSError, ValueError) as exc:
            print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
            return 2
        _print_snapshot(snapshot, as_json=args.json)
        cycles += 1
        if not args.watch or snapshot.get("status") in {"success", "failure", "dry_run"}:
            return 0
        if args.max_cycles is not None and cycles >= args.max_cycles:
            return 0
        if args.interval > 0:
            time.sleep(min(args.interval, 60.0))


if __name__ == "__main__":
    raise SystemExit(main())
