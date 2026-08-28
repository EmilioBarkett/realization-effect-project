#!/usr/bin/env python3
"""Prepare a portable benchmark workspace and freeze its run plan.

This command performs no API calls and does not load model weights. It creates
the shared multi-construct directory layout, snapshots the scientific config,
copies an optional frozen prompt inventory, and writes a run plan plus storage
manifest. The same command works from a local checkout or a RunPod network
volume when ``RSC_BENCH_WORKSPACE_ROOT`` is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import (  # noqa: E402
    load_analysis_spec,
    load_construct_specs,
    load_run_config,
)
from construct_benchmark.campaign import confirmatory_execution_report  # noqa: E402
from construct_benchmark.manifests import build_run_plan, file_sha256, write_run_plan  # noqa: E402
from construct_benchmark.prompts import load_prompt_records  # noqa: E402
from construct_benchmark.storage import (  # noqa: E402
    build_storage_manifest,
    prepare_run_directories,
    resolve_archive_uri,
    resolve_storage_layout,
    write_storage_manifest,
)


def _write_snapshot(path: Path, payload: object) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise SystemExit(f"Refusing to overwrite an existing config snapshot: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _copy_frozen_input(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Prompt inventory does not exist: {source}")
    if source == destination:
        return destination
    if destination.exists():
        if file_sha256(source) != file_sha256(destination):
            raise SystemExit(
                f"Refusing to overwrite a different frozen prompt inventory: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible local or RunPod benchmark workspace."
    )
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("test", "full"), default=None)
    parser.add_argument("--prompts", type=Path, default=None, help="Optional frozen CSV/JSONL inventory.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help="Workspace root; defaults to the configured environment variable or the current directory.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse an existing compatible run directory.")
    parser.add_argument(
        "--confirmatory-campaign",
        type=Path,
        default=None,
        help="Campaign manifest required before preparing a gated full run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print paths without writing files.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    construct_specs = load_construct_specs(args.construct_spec)
    run_config = load_run_config(args.run_config)
    analysis_spec = load_analysis_spec(args.analysis_spec)
    layout = resolve_storage_layout(run_config, args.workspace_root)

    if layout.run_root.exists() and any(layout.run_root.iterdir()) and not args.resume and not args.dry_run:
        raise SystemExit(
            f"Run directory is not empty: {layout.run_root}. Use --resume only when the frozen inputs match."
        )

    prompt_path = layout.prompts_root / "combined.csv"
    if args.prompts is not None:
        suffix = args.prompts.suffix if args.prompts.suffix in {".csv", ".jsonl"} else ".csv"
        prompt_path = layout.prompts_root / f"combined{suffix}"
        if args.prompts.exists():
            if args.dry_run:
                prompt_path = prompt_path.resolve()
            else:
                _copy_frozen_input(args.prompts, prompt_path)
        else:
            raise SystemExit(f"Prompt inventory does not exist: {args.prompts}")

    prompt_records = load_prompt_records(prompt_path) if prompt_path.exists() else None
    plan = build_run_plan(
        run_config,
        construct_specs,
        analysis_spec,
        prompt_inventory_path=prompt_path,
        prompt_records=prompt_records,
        output_root=layout.output_root,
        run_mode=args.run_mode,
    )
    campaign_path = args.confirmatory_campaign
    configured_campaign = run_config.execution.get("confirmatory_campaign_path")
    if campaign_path is None and isinstance(configured_campaign, str) and configured_campaign.strip():
        campaign_path = args.run_config.parent / configured_campaign
    if plan["run_mode"]["confirmatory"] and campaign_path is not None:
        release_report = confirmatory_execution_report(campaign_path, mode="full")
        if not release_report["ready"]:
            blockers = ", ".join(
                str(check["name"]) for check in release_report["blocking_checks"]
            )
            raise SystemExit(
                "Confirmatory campaign release is blocked; refusing to prepare a full run. "
                f"Blocking checks: {blockers}"
            )
    archive_uri = resolve_archive_uri(run_config)

    if not args.dry_run:
        prepare_run_directories(run_config, layout)
        _write_snapshot(layout.config_root / "run_config.json", run_config.to_mapping())
        _write_snapshot(layout.config_root / "analysis_spec.json", analysis_spec.to_mapping())
        for construct_id, spec in construct_specs.items():
            _write_snapshot(
                layout.config_root / "constructs" / f"{construct_id}.json",
                spec.to_mapping(),
            )
        write_run_plan(layout.plan_path, plan)
        write_storage_manifest(
            layout.storage_manifest_path,
            build_storage_manifest(
                run_config,
                layout,
                status="prepared",
                plan_path=layout.plan_path,
                archive_uri=archive_uri,
            ),
        )

    print(
        json.dumps(
            {
                "run_id": run_config.run_id,
                "run_mode": plan["run_mode"]["mode"],
                "status": "validated_only" if args.dry_run else "prepared",
                "construct_ids": list(run_config.construct_ids),
                "workspace_root": str(layout.workspace_root),
                "run_root": str(layout.run_root),
                "run_plan": str(layout.plan_path),
                "prompt_inventory": str(prompt_path),
                "archive_uri_configured": archive_uri is not None,
                "archive_uri_env": run_config.storage["archive_uri_env"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
