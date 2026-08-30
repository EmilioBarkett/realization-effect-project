"""Shell-free terminal lifecycle hooks for paid benchmark runtimes.

The executor must be able to leave a truthful terminal report even when a
provider-specific shutdown command fails.  This module keeps that policy
independent from RunPod's HTTP API: callers provide an argv command, and the
helper runs it without a shell, records only a safe status summary, and never
captures command output (which could contain credentials).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
from typing import Any, Callable, Mapping, Sequence


DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_TOKEN_NAMES = ("status", "reason", "campaign", "output", "terminal_report")
_INHERITED_RUNPOD_ENV_TO_REMOVE = ("RUNPOD_API_KEY", "RUNPOD_CONFIG")


def normalize_terminal_command(command: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalize a provider hook to an argv tuple and reject shell syntax."""

    if command is None:
        return None
    if isinstance(command, str):
        try:
            values = tuple(shlex.split(command, posix=True))
        except ValueError as exc:
            raise ValueError(f"shutdown command is not valid argv: {exc}") from exc
    else:
        values = tuple(str(item) for item in command)
    if not values or not values[0].strip():
        raise ValueError("shutdown command must contain an executable.")
    if any("\x00" in value for value in values):
        raise ValueError("shutdown command must not contain NUL bytes.")
    return values


def _command_with_context(
    command: Sequence[str],
    context: Mapping[str, Any],
) -> tuple[str, ...]:
    replacements = {name: str(context.get(name, "")) for name in _TOKEN_NAMES}
    result: list[str] = []
    for argument in command:
        value = str(argument)
        for name, replacement in replacements.items():
            value = value.replace("{" + name + "}", replacement)
        result.append(value)
    return tuple(result)


def _command_hash(command: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(command), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_terminal_shutdown(
    command: str | Sequence[str] | None,
    *,
    context: Mapping[str, Any],
    timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run an optional terminal hook and return a non-sensitive status.

    ``runner`` is injectable for deterministic tests.  The default runner is
    ``subprocess.run`` with ``shell=False`` and both output streams discarded.
    A non-zero exit, timeout, or launch error is reported but never raised;
    the scientific terminal status remains the caller's source of truth.
    """

    normalized = normalize_terminal_command(command)
    if normalized is None:
        return {
            "configured": False,
            "attempted": False,
            "status": "disabled",
            "argv_sha256": None,
        }
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("shutdown timeout must be a finite positive number.") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("shutdown timeout must be a finite positive number.")

    argv = _command_with_context(normalized, context)
    digest = _command_hash(argv)
    child_env = dict(os.environ)
    for name in _INHERITED_RUNPOD_ENV_TO_REMOVE:
        child_env.pop(name, None)
    result: dict[str, Any] = {
        "configured": True,
        "attempted": True,
        "status": "unknown",
        "argv_sha256": digest,
        "executable": argv[0],
        "argument_count": len(argv),
        "timeout_seconds": timeout,
        "return_code": None,
        "error": None,
    }
    invoke = runner or subprocess.run
    try:
        completed = invoke(
            list(argv),
            check=False,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            env=child_env,
        )
        return_code = getattr(completed, "returncode", completed if isinstance(completed, int) else None)
        result["return_code"] = return_code
        result["status"] = "succeeded" if return_code == 0 else "failed"
        if return_code != 0:
            result["error"] = "shutdown command returned a non-zero exit status"
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "shutdown command exceeded its timeout"
    except OSError as exc:
        result["status"] = "error"
        result["error"] = type(exc).__name__
    except Exception as exc:  # pragma: no cover - defensive provider boundary
        result["status"] = "error"
        result["error"] = type(exc).__name__
    return result


__all__ = [
    "DEFAULT_SHUTDOWN_TIMEOUT_SECONDS",
    "normalize_terminal_command",
    "run_terminal_shutdown",
]
