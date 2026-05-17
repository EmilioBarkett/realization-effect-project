#!/usr/bin/env python3
"""Generate synthetic prompts for activation-vector analysis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import (
    generate_prompt_csv,
    load_generation_plan,
    merge_prompt_csvs,
    pilot_plan_one_job_per_cell,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate activation-analysis prompt CSVs.")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/activation_analysis/realization_vector_generation_v1.json"),
        help="Generation plan JSON.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Generated prompt CSV path.")
    parser.add_argument(
        "--output-template",
        default=None,
        help=(
            "Model-specific output path template containing {model}, for example "
            "experiments/.../realization_vector_v1__{model}.csv."
        ),
    )
    parser.add_argument("--models", nargs="+", default=None, help="Model aliases from the plan.")
    parser.add_argument(
        "--merge-inputs",
        nargs="+",
        type=Path,
        default=None,
        help="Merge completed model-specific CSVs into --output without API calls.",
    )
    parser.add_argument("--limit-jobs", type=int, default=None, help="Limit generation jobs.")
    parser.add_argument("--pilot-all-cells", action="store_true", help="Sample one job per cell/model.")
    parser.add_argument("--pilot-count-per-cell", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Append missing batches to an existing CSV.")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args()

    if args.merge_inputs:
        if args.output is None:
            raise SystemExit("Provide --output when using --merge-inputs.")
        merged = merge_prompt_csvs(args.merge_inputs, args.output)
        print(f"merged {merged} prompts to {args.output}")
        return

    if args.output_template and "{model}" not in args.output_template:
        raise SystemExit("--output-template must contain {model}.")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before generating prompts.")

    plan = load_generation_plan(args.plan)
    if args.pilot_all_cells:
        plan = pilot_plan_one_job_per_cell(plan, count_per_model=args.pilot_count_per_cell)

    selected_models = args.models
    if args.output_template:
        if not selected_models:
            selected_models = [str(model["alias"]) for model in plan["models"]]
        total_written = 0
        for model_alias in selected_models:
            output = Path(args.output_template.format(model=model_alias))
            written = generate_prompt_csv(
                plan,
                output,
                api_key=api_key,
                model_aliases={model_alias},
                limit_jobs=args.limit_jobs,
                resume=args.resume,
            )
            total_written += written
            print(f"wrote {written} generated prompts for {model_alias} to {output}")
        print(f"wrote {total_written} generated prompts across {len(selected_models)} model file(s)")
        return

    output = args.output
    if output is None:
        default_output = plan.get("default_output")
        if not default_output:
            raise SystemExit("Provide --output or set default_output in the generation plan.")
        output = Path(str(default_output))

    written = generate_prompt_csv(
        plan,
        output,
        api_key=api_key,
        model_aliases=set(args.models) if args.models else None,
        limit_jobs=args.limit_jobs,
        resume=args.resume,
    )
    print(f"wrote {written} generated prompts to {output}")


if __name__ == "__main__":
    main()
