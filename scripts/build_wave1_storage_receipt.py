#!/usr/bin/env python3
"""Record durable storage and checksum provenance without copying raw data.

The receipt is intentionally metadata-only.  It records where the raw run
lives, which checksum manifest was verified, whether an archive URI is
configured, and which small analysis artifacts were produced.  It never reads
model weights or uploads/copies the raw activation and generation files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
import sys

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_receipt(*, run_root: Path, gate_root: Path, base_commit: str, overlay_root: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    checksum_path = run_root / "raw_run_checksums.sha256"
    verification_path = run_root / "raw_run_checksum_verification.log"
    if not checksum_path.is_file() or not verification_path.is_file():
        raise ValueError("The complete checksum and verification files are required before writing a receipt.")
    checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    verification_lines = verification_path.read_text(encoding="utf-8").splitlines()
    failures = [line for line in verification_lines if not line.endswith(": OK")]
    if not checksum_lines or len(verification_lines) != len(checksum_lines) or failures:
        raise ValueError("Checksum verification is incomplete or contains failures.")

    categories = {
        "activations": run_root / "activations",
        "generations": run_root / "raw",
        "residual_interchange": run_root / "causal_768",
        "construct_steering": run_root / "constructs",
        "checkpoints": run_root / "checkpoints",
    }
    archive_configured = bool(os.environ.get("RSC_BENCH_ARCHIVE_URI"))
    source_files = [path for path in run_root.rglob("*") if path.is_file()]
    source_bytes = sum(path.stat().st_size for path in source_files)
    receipt: dict[str, Any] = {
        "schema_version": "0.1.0",
        "manifest_type": "wave1_raw_storage_receipt",
        "status": "verified",
        "confirmatory": False,
        "base_snapshot_commit": base_commit,
        "raw_run_root": str(run_root),
        "source_tree_file_count": len(source_files),
        "source_tree_bytes": source_bytes,
        "checksum_manifest": {
            "path": str(checksum_path),
            "sha256": file_sha256(checksum_path),
            "entry_count": len(checksum_lines),
            "verification_log": str(verification_path),
            "verification_log_sha256": file_sha256(verification_path),
            "verification_entry_count": len(verification_lines),
            "verification_failures": len(failures),
        },
        "raw_categories": {
            name: {"path": str(path), "exists": path.exists(), "local_sync": False}
            for name, path in categories.items()
        },
        "local_transfer_policy": "Raw activations, generations, residual-interchange observations, and checkpoints remain on the RunPod persistent volume and are not copied to the laptop.",
        "archive": {
            "archive_uri_configured": archive_configured,
            "status": "configured_not_attempted" if archive_configured else "not_configured",
            "uri_value_recorded": False,
        },
        "analysis": {
            "model_loaded": False,
            "inference_requested": False,
            "cpu_scoring_completed_beyond_source_summaries": True,
            "temporary_analysis_pod": "stop_after_receipt_and_sync",
        },
    }
    receipt["receipt_sha256"] = canonical_hash(receipt)

    overlay_files: dict[str, str] = {}
    if overlay_root is not None and overlay_root.exists():
        for path in sorted(item for item in overlay_root.rglob("*") if item.is_file()):
            overlay_files[str(path.relative_to(overlay_root))] = file_sha256(path)
    process_listing = _command_output(["ps", "-eo", "args="])
    model_markers = (
        "run_parallel_benchmark.py",
        "run_construct_steering.py",
        "log_residual_streams.py",
    )
    model_processes_detected = any(
        marker in line
        for line in process_listing.splitlines()
        for marker in model_markers
    )
    environment: dict[str, Any] = {
        "schema_version": "0.1.0",
        "manifest_type": "wave1_cpu_analysis_environment",
        "status": "verified",
        "confirmatory": False,
        "base_snapshot_commit": base_commit,
        "python": _command_output(["python3", "--version"]),
        "packages": {name: _package_version(name) for name in ("numpy", "pandas", "scipy")},
        "gpu_query": _command_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"]),
        "model_processes_detected": model_processes_detected,
        "overlay_root": None if overlay_root is None else str(overlay_root),
        "overlay_files": overlay_files,
        "raw_data_local_sync": False,
    }
    environment["environment_sha256"] = canonical_hash(environment)
    return receipt, environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--overlay-root", type=Path, default=None)
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    gate_root = args.gate_root.resolve()
    receipt, environment = build_receipt(
        run_root=run_root,
        gate_root=gate_root,
        base_commit=args.base_commit,
        overlay_root=None if args.overlay_root is None else args.overlay_root.resolve(),
    )
    gate_root.mkdir(parents=True, exist_ok=True)
    (gate_root / "raw_storage_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (gate_root / "cpu_analysis_environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    continuation_path = gate_root / "continuation_manifest.json"
    if continuation_path.is_file():
        continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
        continuation["raw_storage_receipt"] = "raw_storage_receipt.json"
        continuation["cpu_analysis_environment"] = "cpu_analysis_environment.json"
        continuation["raw_checksum_manifest"] = str(run_root / "raw_run_checksums.sha256")
        continuation["raw_checksum_manifest_sha256"] = receipt["checksum_manifest"]["sha256"]
        continuation["raw_checksum_verification_failures"] = receipt["checksum_manifest"]["verification_failures"]
        continuation.pop("manifest_sha256", None)
        continuation["manifest_sha256"] = _json_hash(continuation)
        continuation_path.write_text(json.dumps(continuation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": receipt, "environment": environment}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
