#!/usr/bin/env python3
"""Provision, inspect, or stop the single exact-B300 campaign pod.

The only credential accepted by this command is ``RUNPOD_2_API_KEY`` in the
local controller environment. It is never passed into the pod command or
written to controller state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.runpod_controller import (  # noqa: E402
    B300_GPU_TYPE,
    RUNPOD_API_KEY_ENV,
    RUNPOD_BASE_URL,
    RunPodController,
    RunPodError,
    load_spec,
)


class _DryRunTransport:
    def request(self, *_args: object, **_kwargs: object) -> object:
        raise RunPodError("dry-run transport cannot make network requests")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=RUNPOD_BASE_URL, help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    availability = subparsers.add_parser("availability", help="List matching B300 pod/machine records.")
    availability.add_argument("--data-center-id", action="append", dest="data_center_ids", default=None)

    create = subparsers.add_parser("create", help="Create exactly one B300 pod with /workspace mounted.")
    create.add_argument("--spec", type=Path, required=True, help="Versioned controller spec JSON.")
    create.add_argument("--state", type=Path, required=True, help="Durable controller state path.")
    create.add_argument("--dry-run", action="store_true", help="Validate and print a credential-safe request plan.")

    inspect = subparsers.add_parser("inspect", help="Inspect readiness and exact GPU identity.")
    inspect.add_argument("--pod-id", default=None)
    inspect.add_argument("--state", type=Path, default=None)

    stop = subparsers.add_parser("stop", help="Stop only the campaign-owned pod.")
    stop.add_argument("--pod-id", default=None)
    stop.add_argument("--state", type=Path, required=True)
    stop.add_argument("--dry-run", action="store_true", help="Validate ownership without making a request.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            spec = load_spec(args.spec)
            controller = RunPodController(
                transport=_DryRunTransport() if args.dry_run else None,
                state_path=args.state,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
            )
            result = controller.create_b300_pod(spec, dry_run=args.dry_run)
        elif args.command == "availability":
            controller = RunPodController(base_url=args.base_url, timeout_seconds=args.timeout)
            result = controller.query_availability(data_center_ids=args.data_center_ids)
        elif args.command == "inspect":
            controller = RunPodController(
                state_path=args.state,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
            )
            result = controller.inspect_pod(args.pod_id)
        else:
            if args.dry_run:
                result = {
                    "status": "dry_run",
                    "pod_id": args.pod_id,
                    "credential_env": RUNPOD_API_KEY_ENV,
                    "expected_gpu_type": B300_GPU_TYPE,
                    "request_method": "POST",
                    "request_path": "/pods/{pod_id}/stop",
                }
            else:
                controller = RunPodController(
                    state_path=args.state,
                    base_url=args.base_url,
                    timeout_seconds=args.timeout,
                )
                result = controller.stop_pod(args.pod_id)
    except (RunPodError, RuntimeError, ValueError, OSError) as exc:
        result = {"status": "error", "error": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"dry_run", "created", "ready", "ok", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
