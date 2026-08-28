#!/usr/bin/env python3
"""Stop one RunPod pod through the v1 REST API.

This is an explicit, opt-in lifecycle command for a local controller.  It
reads only ``RUNPOD_2_API_KEY``; the legacy ``RUNPOD_API_KEY`` is deliberately
not a fallback.  Response bodies and credentials are never printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable


RUNPOD_API_KEY_ENV = "RUNPOD_2_API_KEY"
RUNPOD_POD_ID_ENV = "RUNPOD_POD_ID"
RUNPOD_STOP_URL = "https://rest.runpod.io/v1/pods/{pod_id}/stop"
_POD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def normalize_pod_id(value: str | None) -> str:
    """Validate a pod ID before placing it in a URL path."""

    candidate = str(value or "").strip()
    if not candidate or not _POD_ID_PATTERN.fullmatch(candidate):
        raise ValueError("pod ID must contain only letters, digits, underscores, or hyphens")
    return candidate


def _endpoint(pod_id: str) -> str:
    return RUNPOD_STOP_URL.format(pod_id=normalize_pod_id(pod_id))


def stop_pod(
    pod_id: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Issue a shell-free POST stop request and return a sanitized summary."""

    safe_pod_id = normalize_pod_id(pod_id)
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a finite positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    credential = api_key if api_key is not None else os.environ.get(RUNPOD_API_KEY_ENV, "")
    if not str(credential).strip():
        raise RuntimeError(f"{RUNPOD_API_KEY_ENV} is not set")

    request = urllib.request.Request(
        _endpoint(safe_pod_id),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {credential}",
        },
        method="POST",
    )
    open_url = opener or urllib.request.urlopen
    result: dict[str, Any] = {
        "status": "unknown",
        "pod_id": safe_pod_id,
        "endpoint": _endpoint(safe_pod_id),
        "http_status": None,
        "error": None,
    }
    try:
        response = open_url(request, timeout=timeout)
        with response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            status = int(status)
        result["http_status"] = status
        result["status"] = "stopped" if 200 <= status < 300 else "error"
        if result["status"] != "stopped":
            result["error"] = "RunPod returned a non-success HTTP status"
    except urllib.error.HTTPError as exc:
        result["http_status"] = int(exc.code)
        result["status"] = "error"
        result["error"] = "RunPod returned an HTTP error"
    except (urllib.error.URLError, TimeoutError, OSError):
        result["status"] = "error"
        result["error"] = "RunPod stop request failed"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pod-id",
        default=None,
        help=f"RunPod pod ID; defaults to {RUNPOD_POD_ID_ENV}.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the pod ID and print the endpoint without requiring a credential or making a request.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        raw_pod_id = args.pod_id if args.pod_id is not None else os.environ.get(RUNPOD_POD_ID_ENV)
        pod_id = normalize_pod_id(raw_pod_id)
        if args.dry_run:
            result = {
                "status": "dry_run",
                "pod_id": pod_id,
                "endpoint": _endpoint(pod_id),
                "credential_env": RUNPOD_API_KEY_ENV,
                "request_method": "POST",
            }
        else:
            result = stop_pod(pod_id, timeout_seconds=args.timeout)
    except (RuntimeError, ValueError) as exc:
        result = {"status": "error", "error": str(exc)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"dry_run", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
