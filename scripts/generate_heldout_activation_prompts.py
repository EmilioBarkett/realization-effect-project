#!/usr/bin/env python3
"""Generate held-out realization-vector prompts and audit overlap locally."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import (  # noqa: E402
    generate_prompt_csv,
    iter_generation_jobs,
    load_generation_plan,
)


DEFAULT_PLAN = Path("configs/activation_analysis/realization_vector_heldout_generation_v1.json")
DEFAULT_REFERENCE = Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv")
DEFAULT_OVERLAP_OUTPUT = Path("results/audits/heldout_prompt_overlap.csv")


def estimate_generation(plan: dict, *, model_aliases: set[str] | None, limit_jobs: int | None) -> tuple[int, int, int]:
    rows_per_unit = 2 if str(plan.get("generation_mode", "single_prompt")) == "paired_contrast" else 1
    max_per_request = int(plan.get("generation", {}).get("max_prompts_per_request", 0) or 0)
    jobs = list(iter_generation_jobs(plan, model_aliases=model_aliases, limit_jobs=limit_jobs))
    prompt_rows = sum(job.count * rows_per_unit for job in jobs)
    if max_per_request > 0:
        api_requests = sum(math.ceil(job.count / max_per_request) for job in jobs)
    else:
        api_requests = len(jobs)
    return len(jobs), prompt_rows, api_requests


def run_overlap_audit(reference: Path, candidate: Path, output: Path, *, fail_on_overlap: bool) -> None:
    command = [
        sys.executable,
        "scripts/audit_prompt_overlap.py",
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
    ]
    if fail_on_overlap:
        command.append("--fail-on-overlap")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out activation prompts through OpenRouter.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None, help="Optional model aliases from the held-out plan.")
    parser.add_argument("--limit-jobs", type=int, default=None, help="Limit API jobs for a smoke run.")
    parser.add_argument("--resume", action="store_true", help="Append missing batches to an existing CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate prompt rows and API requests without calling LLMs.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--skip-audit", action="store_true", help="Skip local overlap audit after generation.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--overlap-output", type=Path, default=DEFAULT_OVERLAP_OUTPUT)
    parser.add_argument("--fail-on-overlap", action="store_true")
    args = parser.parse_args()

    plan = load_generation_plan(args.plan)
    output = args.output or Path(str(plan["default_output"]))
    model_aliases = set(args.models) if args.models else None
    job_count, prompt_rows, api_requests = estimate_generation(
        plan,
        model_aliases=model_aliases,
        limit_jobs=args.limit_jobs,
    )
    print(
        f"plan={args.plan} output={output} jobs={job_count} "
        f"prompt_rows={prompt_rows} estimated_api_requests={api_requests}"
    )

    if args.dry_run:
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before generating prompts.")

    written = generate_prompt_csv(
        plan,
        output,
        api_key=api_key,
        model_aliases=model_aliases,
        limit_jobs=args.limit_jobs,
        resume=args.resume,
    )
    print(f"wrote {written} held-out prompts to {output}")

    if not args.skip_audit:
        run_overlap_audit(args.reference, output, args.overlap_output, fail_on_overlap=args.fail_on_overlap)


if __name__ == "__main__":
    main()
