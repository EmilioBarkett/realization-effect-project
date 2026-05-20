#!/usr/bin/env python3
"""CLI entrypoint for realization-effect behavioral and prompt-generation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from activation_analysis.generation_cli import run_generation_prompt_mode
from realization_effect.api import (
    SYSTEM_PROMPT,
    _extract_response_text,
    _is_retryable_error,
    call_model,
    create_openrouter_client,
)
from realization_effect.blocks import (
    RESULT_FIELDS,
    _block_selector_key,
    _list_block_csv_paths,
    _normalize_temperature,
    _parse_positive_run_number,
    _rounded_trial_target,
    _row_identity_key,
    _row_quality_score,
    _row_sort_key,
    _select_balanced_block_rows,
    _slug_for_filename,
    build_block_output_path,
    default_grouped_output_path,
    load_resume_state,
    merge_block_csvs,
    reconcile_merged_outputs,
)
from realization_effect.conditions import load_conditions
from realization_effect.orchestration import run_experiment, run_experiment_grid
from realization_effect.parsing import (
    ANSWER_LABEL_PATTERN,
    ANSWER_LABELS,
    INTEGER_PATTERN,
    LINE_LABEL_PATTERN,
    REFUSAL_MARKERS,
    _extract_response_numbers,
    _integers_in_text,
    _labeled_answer_value,
    _line_answer_value,
    parse_response,
)
from realization_effect.prompts import (
    NEAR_ZERO_REALIZED_THRESHOLD,
    PAPER_PROMPT_TEMPLATE,
    PROMPT_BUILDERS,
    REALIZED_PROMPT_TEMPLATE,
    TWO_INTEGER_INSTRUCTION,
    _build_prompt_absolute,
    _build_prompt_balance,
    _build_prompt_qualitative,
    build_prompt,
)


GENERATION_PROMPT_VERSION = "generation"
DEFAULT_GENERATION_PLAN = Path("configs/activation_analysis/realization_vector_generation_v1.json")
DEFAULT_GENERATION_OUTPUT = Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv")


def main() -> None:
    """Parse CLI arguments and run the requested experiment workflow."""
    parser = argparse.ArgumentParser(description="Run realization-effect LLM experiment.")
    parser.add_argument(
        "--conditions",
        type=Path,
        default=Path("configs/realization_effect/conditions.csv"),
        help="Path to conditions CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/results.csv"),
        help="Path to output results CSV file.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of runs per condition (default: 100).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="One or more models to query (via OpenRouter).",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="One or more sampling temperatures passed to the API.",
    )
    parser.add_argument("--model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="Optional resume filter for model.",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=None,
        help="Optional resume filter for temperature.",
    )
    parser.add_argument(
        "--target-prompt-version",
        type=str,
        default=None,
        help="Optional resume filter for prompt version.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="absolute",
        help=(
            "Prompt wording version (default: absolute). Use 'generation' to "
            "generate activation-analysis prompts instead of behavioral results."
        ),
    )
    parser.add_argument(
        "--prompt-versions",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of prompt wording versions to run as separate blocks.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle condition order before running.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for transient API failures (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of experiment blocks to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--round-trials-step",
        type=int,
        default=1,
        help=(
            "Round merged per-condition trial counts down to this step size "
            "(default: 1, disabled; use values >1 to round down)."
        ),
    )
    parser.add_argument(
        "--generation-plan",
        type=Path,
        default=DEFAULT_GENERATION_PLAN,
        help="Prompt-generation plan JSON used when --prompt-version generation.",
    )
    parser.add_argument(
        "--generation-output",
        type=Path,
        default=None,
        help=(
            "Generated prompt CSV path. Defaults to "
            "the generation plan's default_output, or realization_vector_v1.csv "
            "when --prompt-version generation."
        ),
    )
    parser.add_argument(
        "--generation-limit-jobs",
        type=int,
        default=None,
        help="Limit generation jobs for a small pilot.",
    )
    parser.add_argument(
        "--generation-pilot-all-cells",
        action="store_true",
        help=(
            "Run one representative job per generation cell and selected model. "
            "This samples the full taxonomy instead of the first expanded jobs."
        ),
    )
    parser.add_argument(
        "--generation-pilot-count-per-cell",
        type=int,
        default=1,
        help="Prompts per sampled cell when --generation-pilot-all-cells is set.",
    )
    parser.add_argument(
        "--generation-resume",
        action="store_true",
        help="Append missing prompt-generation batches to an existing CSV.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    args = parser.parse_args()

    prompt_versions = args.prompt_versions if args.prompt_versions is not None else [args.prompt_version]
    if GENERATION_PROMPT_VERSION in prompt_versions:
        if len(prompt_versions) != 1:
            raise SystemExit("--prompt-version generation cannot be mixed with behavioral prompt versions.")
        if args.model is not None:
            args.models = [args.model]
        run_generation_prompt_mode(
            args,
            default_generation_output=DEFAULT_GENERATION_OUTPUT,
        )
        return

    models = args.models if args.models is not None else [args.model or "openai/o4-mini"]
    temperatures = args.temperatures if args.temperatures is not None else [args.temperature or 1.0]

    run_experiment_grid(
        conditions_path=args.conditions,
        output_path=args.output,
        n_trials=args.n_trials,
        models=models,
        temperatures=temperatures,
        target_model=args.target_model,
        target_temperature=args.target_temperature,
        target_prompt_version=args.target_prompt_version,
        sleep_seconds=args.sleep_seconds,
        prompt_version=args.prompt_version,
        prompt_versions=prompt_versions,
        shuffle=args.shuffle,
        max_retries=args.max_retries,
        seed=args.seed,
        max_workers=args.max_workers,
        round_trials_step=args.round_trials_step,
    )


__all__ = [
    "ANSWER_LABEL_PATTERN",
    "ANSWER_LABELS",
    "DEFAULT_GENERATION_OUTPUT",
    "DEFAULT_GENERATION_PLAN",
    "GENERATION_PROMPT_VERSION",
    "INTEGER_PATTERN",
    "LINE_LABEL_PATTERN",
    "NEAR_ZERO_REALIZED_THRESHOLD",
    "PAPER_PROMPT_TEMPLATE",
    "PROMPT_BUILDERS",
    "REALIZED_PROMPT_TEMPLATE",
    "REFUSAL_MARKERS",
    "RESULT_FIELDS",
    "SYSTEM_PROMPT",
    "TWO_INTEGER_INSTRUCTION",
    "_block_selector_key",
    "_build_prompt_absolute",
    "_build_prompt_balance",
    "_build_prompt_qualitative",
    "_extract_response_numbers",
    "_extract_response_text",
    "_integers_in_text",
    "_is_retryable_error",
    "_labeled_answer_value",
    "_line_answer_value",
    "_list_block_csv_paths",
    "_normalize_temperature",
    "_parse_positive_run_number",
    "_rounded_trial_target",
    "_row_identity_key",
    "_row_quality_score",
    "_row_sort_key",
    "_select_balanced_block_rows",
    "_slug_for_filename",
    "build_block_output_path",
    "build_prompt",
    "call_model",
    "create_openrouter_client",
    "default_grouped_output_path",
    "load_conditions",
    "load_resume_state",
    "main",
    "merge_block_csvs",
    "parse_response",
    "reconcile_merged_outputs",
    "run_experiment",
    "run_experiment_grid",
]


if __name__ == "__main__":
    main()
