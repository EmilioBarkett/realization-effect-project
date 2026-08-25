#!/usr/bin/env python3
"""Finalize, checksum, and optionally archive one benchmark run.

The command is intentionally separate from model execution. It can therefore
be used after a local smoke run or as the final step on a RunPod worker. When
an archive URI is configured, finalization syncs the run to an S3-compatible
prefix and then syncs the small completion metadata once more.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.schemas import RunConfig  # noqa: E402
from construct_benchmark.storage import (  # noqa: E402
    build_storage_manifest,
    build_sync_command,
    layout_for_existing_run,
    resolve_archive_uri,
    run_sync_command,
    verify_checksums,
    write_archive_receipt,
    write_checksums,
    write_json,
    write_storage_manifest,
)


def _load_run_config(run_root: Path, config_path: Path | None) -> RunConfig:
    if config_path is not None:
        from construct_benchmark.config import load_run_config

        return load_run_config(config_path)
    plan_path = run_root / "run_plan.json"
    if not plan_path.is_file():
        raise SystemExit(f"Missing run plan: {plan_path}; provide --run-config if using an older run.")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        return RunConfig.from_mapping(plan["run_config"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not load run configuration from {plan_path}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize and optionally archive a benchmark run.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--archive-uri", default=None, help="Override the configured s3:// archive prefix.")
    parser.add_argument("--sync", action="store_true", help="Require an archive sync after finalization.")
    parser.add_argument("--no-sync", action="store_true", help="Disable automatic archive sync for this run.")
    parser.add_argument(
        "--require-archive",
        action="store_true",
        help="Fail if no archive URI is configured instead of leaving the run local-only.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the archive command without writing or syncing.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.sync and args.no_sync:
        raise SystemExit("--sync and --no-sync are mutually exclusive.")

    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Run root does not exist: {run_root}")
    run_config = _load_run_config(run_root, args.run_config)
    layout = layout_for_existing_run(run_config, run_root)
    archive_uri = resolve_archive_uri(run_config, args.archive_uri)
    if args.require_archive and archive_uri is None:
        raise SystemExit(
            f"No archive URI configured; set {run_config.storage['archive_uri_env']} or pass --archive-uri."
        )

    automatic_sync = bool(run_config.storage["sync_on_finalize"]) and archive_uri is not None
    should_sync = False if args.no_sync else (args.sync or automatic_sync)
    if should_sync and archive_uri is None:
        raise SystemExit(
            f"Archive sync requested but no URI is configured; set {run_config.storage['archive_uri_env']} "
            "or pass --archive-uri."
        )

    endpoint_url = os.environ.get(str(run_config.storage["sync_endpoint_env"]))
    command = (
        build_sync_command(
            run_config,
            layout,
            archive_uri,
            dry_run=args.dry_run,
            endpoint_url=endpoint_url,
        )
        if should_sync and archive_uri is not None
        else None
    )

    if args.dry_run:
        checksum_summary = None
        if layout.checksums_path.is_file():
            checksum_summary = verify_checksums(run_root)
        print(
            json.dumps(
                {
                    "run_id": run_config.run_id,
                    "run_root": str(run_root),
                    "would_sync": should_sync,
                    "archive_destination": (
                        command[4] if command is not None and len(command) > 4 else None
                    ),
                    "command": command,
                    "checksums": checksum_summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    write_json(
        run_root / "run_status.json",
        {
            "schema_version": run_config.schema_version,
            "manifest_type": "benchmark_run_status",
            "run_id": run_config.run_id,
            "status": "finalizing",
        },
    )
    checksums_path = write_checksums(run_root)
    checksum_summary = verify_checksums(run_root)
    if not checksum_summary["valid"]:
        raise SystemExit(f"Checksum verification failed before archive: {checksum_summary['failures']}")

    archive_status = "pending" if should_sync else ("not_configured" if archive_uri is None else "not_requested")
    write_storage_manifest(
        layout.storage_manifest_path,
        build_storage_manifest(
            run_config,
            layout,
            status="finalized",
            plan_path=layout.plan_path,
            archive_uri=archive_uri,
            archive_status=archive_status,
        ),
    )

    if should_sync and archive_uri is not None and command is not None:
        run_sync_command(command)
        completed_at = datetime.now(timezone.utc).isoformat()
        write_archive_receipt(
            layout,
            archive_uri=archive_uri,
            command=command,
            completed_at=completed_at,
        )
        write_storage_manifest(
            layout.storage_manifest_path,
            build_storage_manifest(
                run_config,
                layout,
                status="finalized",
                plan_path=layout.plan_path,
                archive_uri=archive_uri,
                archive_status="synced",
                archived_at=completed_at,
            ),
        )
        # The first sync copied the scientific artifacts. This second sync
        # publishes the completion receipt and final archive status.
        run_sync_command(
            build_sync_command(
                run_config,
                layout,
                archive_uri,
                endpoint_url=endpoint_url,
            )
        )

    write_json(
        run_root / "run_status.json",
        {
            "schema_version": run_config.schema_version,
            "manifest_type": "benchmark_run_status",
            "run_id": run_config.run_id,
            "status": "finalized",
            "archive_status": "synced" if should_sync else archive_status,
            "checksums": str(checksums_path),
        },
    )
    print(
        json.dumps(
            {
                "run_id": run_config.run_id,
                "status": "finalized",
                "run_root": str(run_root),
                "checked_files": checksum_summary["checked_files"],
                "archive_status": "synced" if should_sync else archive_status,
                "archive_destination": command[4] if command is not None and len(command) > 4 else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
