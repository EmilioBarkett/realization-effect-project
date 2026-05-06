from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

from activation_analysis.openrouter_prompt_generation import (
    generate_prompt_csv,
    load_generation_plan,
    pilot_plan_one_job_per_cell,
)


def run_generation_prompt_mode(
    args: Namespace,
    *,
    default_generation_output: Path,
) -> None:
    """Run OpenRouter prompt generation from the realization experiment CLI."""

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before generating prompts.")

    plan = load_generation_plan(args.generation_plan)
    if args.generation_pilot_all_cells:
        plan = pilot_plan_one_job_per_cell(
            plan,
            count_per_model=args.generation_pilot_count_per_cell,
        )
    model_aliases = set(args.models) if args.models is not None else None
    output_path = (
        args.generation_output
        if args.generation_output is not None
        else (
            Path(plan["default_output"])
            if plan.get("default_output")
            else (
                default_generation_output
                if args.output == Path("results/results.csv")
                else args.output
            )
        )
    )

    written = generate_prompt_csv(
        plan,
        output_path,
        api_key=api_key,
        model_aliases=model_aliases,
        limit_jobs=args.generation_limit_jobs,
        resume=args.generation_resume,
    )
    print(f"wrote {written} generated prompts to {output_path}")
