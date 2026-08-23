#!/usr/bin/env python3
"""Reconcile block outputs into canonical results and partition legacy rows."""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

from .partition import partition_results
from .runner import default_grouped_output_path, reconcile_merged_outputs


def copy_extra_blocks(*, source_dirs: List[Path], destination_blocks_dir: Path) -> Dict[str, int]:
    """Copy block CSVs from extra directories into the canonical blocks directory."""
    destination_blocks_dir.mkdir(parents=True, exist_ok=True)
    copied_by_dir: Dict[str, int] = {}

    for source_dir in source_dirs:
        copied = 0
        if source_dir.exists():
            for source_path in sorted(source_dir.glob("block__*.csv")):
                destination_path = destination_blocks_dir / source_path.name
                shutil.copyfile(source_path, destination_path)
                copied += 1
        copied_by_dir[str(source_dir)] = copied
    return copied_by_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally copy extra block files into canonical results/blocks, "
            "reconcile results/results.csv, and partition legacy rows."
        )
    )
    parser.add_argument(
        "--canonical-output",
        type=Path,
        default=Path("results/results.csv"),
        help="Canonical merged results CSV to refresh.",
    )
    parser.add_argument(
        "--round-trials-step",
        type=int,
        default=1,
        help="Round-down step for balanced per-condition merge target (default: 1).",
    )
    parser.add_argument(
        "--extra-blocks-dir",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Optional extra block directories to fold into canonical blocks."
        ),
    )
    parser.add_argument(
        "--skip-partition",
        action="store_true",
        help="Skip legacy partition step after reconcile.",
    )
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("configs/realization_effect/conditions.csv"),
        help="Conditions CSV used for canonical-vs-legacy partitioning.",
    )
    parser.add_argument(
        "--legacy-output",
        type=Path,
        default=Path("tests/fixtures/noncanonical/legacy_archive/results_legacy.csv"),
        help="Output path for legacy rows.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("tests/fixtures/noncanonical/legacy_archive/partition_report.txt"),
        help="Output path for partition summary report.",
    )
    parser.add_argument(
        "--skip-prompt-template-check",
        action="store_true",
        help="Disable prompt-template drift detection during partitioning.",
    )
    args = parser.parse_args()

    canonical_blocks_dir = args.canonical_output.parent / "blocks"
    copied_by_dir = copy_extra_blocks(
        source_dirs=args.extra_blocks_dir,
        destination_blocks_dir=canonical_blocks_dir,
    )
    copied_total = sum(copied_by_dir.values())

    grouped_output_path = default_grouped_output_path(args.canonical_output)
    block_count, merged_rows = reconcile_merged_outputs(
        output_path=args.canonical_output,
        grouped_output_path=grouped_output_path,
        round_trials_step=args.round_trials_step,
    )

    print(f"Copied {copied_total} extra block file(s) into {canonical_blocks_dir}.")
    for source_dir, copied in copied_by_dir.items():
        print(f"  - {source_dir}: {copied}")
    print(
        f"Reconciled {block_count} block file(s) into {args.canonical_output} "
        f"with {merged_rows} total row(s)."
    )
    print(f"Refreshed grouped output at {grouped_output_path}.")

    if args.skip_partition:
        return

    partition_results(
        input_csv=args.canonical_output,
        conditions_csv=args.conditions,
        canonical_csv=args.canonical_output,
        legacy_csv=args.legacy_output,
        report_path=args.report_output,
        enforce_prompt_template_check=not args.skip_prompt_template_check,
    )
    if grouped_output_path.resolve(strict=False) != args.canonical_output.resolve(strict=False):
        shutil.copyfile(args.canonical_output, grouped_output_path)
    print(
        "Partitioned canonical vs legacy rows: "
        f"canonical={args.canonical_output}, legacy={args.legacy_output}, report={args.report_output}"
    )
    print(f"Synced grouped output to canonical rows at {grouped_output_path}.")


if __name__ == "__main__":
    main()
