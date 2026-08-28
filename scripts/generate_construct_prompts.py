#!/usr/bin/env python3
"""Generate canonical construct prompt inventories or run a no-API dry run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import load_construct_spec  # noqa: E402
from construct_benchmark.generation import (  # noqa: E402
    dry_run_summary,
    generate_prompt_records,
    load_generation_plan,
    resolve_generation_mode,
    write_generation_result,
)
from activation_analysis.openai_prompt_generation import call_openai_responses  # noqa: E402


VECTOR_SPLITS = {"direction_train", "direction_validation", "direction_heldout"}


def _load_plan_and_spec(plan_path: Path, spec_path: Path | None):
    raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw_plan, dict):
        raise ValueError(f"{plan_path} must contain a JSON object.")
    if spec_path is None:
        raw_spec_path = raw_plan.get("construct_spec_path")
        if not isinstance(raw_spec_path, str) or not raw_spec_path.strip():
            raise ValueError(f"{plan_path} must define construct_spec_path when --construct-spec is omitted.")
        spec_path = (plan_path.parent / raw_spec_path).resolve()
    spec = load_construct_spec(spec_path)
    return load_generation_plan(plan_path, spec), spec


def _summary_path(args: argparse.Namespace) -> Path | None:
    if args.summary_output is not None:
        return args.summary_output
    if args.dry_run:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical construct prompt inventories.")
    parser.add_argument("--plan", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--construct-spec",
        type=Path,
        nargs="*",
        default=None,
        help="Optional spec paths matching --plan order; otherwise use each plan's construct_spec_path.",
    )
    parser.add_argument(
        "--mode",
        choices=("review", "full"),
        default=None,
        help=(
            "Named API-generation mode. review emits one item per model/cell and is explicitly partial; "
            "full emits the complete frozen inventory."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("all", "vector"),
        default="all",
        help="Generate the complete plan or only paired train/validation/held-out vector prompts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Expand plans without making API calls.")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Output CSV/JSONL for one plan.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Per-construct output directory.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional model aliases.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum number of concurrent generation requests for each construct.",
    )
    parser.add_argument(
        "--count-per-model-per-cell",
        type=int,
        default=None,
        help="Override every cell's count for an explicit pilot; requires --allow-partial when writing.",
    )
    parser.add_argument("--limit-jobs", type=int, default=None, help="Generate only a partial job prefix.")
    parser.add_argument("--allow-partial", action="store_true", help="Permit writing a partial inventory.")
    parser.add_argument(
        "--input-usd-per-million-tokens",
        type=float,
        default=None,
        help="Optional input-token price used only for dry-run cost estimates.",
    )
    parser.add_argument(
        "--output-usd-per-million-tokens",
        type=float,
        default=None,
        help="Optional output-token price used only for dry-run cost estimates.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="API-key environment variable for the active OpenAI Luna transport.",
    )
    args = parser.parse_args()

    if args.construct_spec is not None and len(args.construct_spec) not in {1, len(args.plan)}:
        raise SystemExit("--construct-spec must contain one path or exactly one path per --plan.")
    if args.output is not None and len(args.plan) != 1:
        raise SystemExit("--output can only be used with one --plan; use --output-dir for multiple plans.")
    if not args.dry_run and args.output is None and args.output_dir is None:
        raise SystemExit("Provide --output or --output-dir when not using --dry-run.")
    if args.limit_jobs is not None and args.limit_jobs < 1:
        raise SystemExit("--limit-jobs must be positive.")
    if args.workers < 1:
        raise SystemExit("--workers must be positive.")
    if args.count_per_model_per_cell is not None and args.count_per_model_per_cell < 1:
        raise SystemExit("--count-per-model-per-cell must be positive.")
    if (args.input_usd_per_million_tokens is None) != (args.output_usd_per_million_tokens is None):
        raise SystemExit(
            "Provide both --input-usd-per-million-tokens and --output-usd-per-million-tokens for a cost estimate."
        )

    specs = []
    plans = []
    for index, plan_path in enumerate(args.plan):
        spec_path = None
        if args.construct_spec is not None:
            spec_path = args.construct_spec[0] if len(args.construct_spec) == 1 else args.construct_spec[index]
        plan, spec = _load_plan_and_spec(plan_path, spec_path)
        plans.append(plan)
        specs.append(spec)

    selected_splits = VECTOR_SPLITS if args.scope == "vector" else None

    if args.dry_run:
        selected_mode = args.mode or "full"
        mode_configs = [resolve_generation_mode(plan, selected_mode)[1] for plan in plans]
        if selected_mode == "full" and any(
            value is not None for value in (args.count_per_model_per_cell, args.limit_jobs)
        ):
            raise SystemExit("full generation mode cannot use count or job limits; use review mode for a partial run.")
        mode_count_override = (
            int(mode_configs[0]["count_per_model_per_cell"])
            if selected_mode == "review" and args.count_per_model_per_cell is None
            else args.count_per_model_per_cell
        )
        selected_aliases = set(args.models) if args.models else None
        summaries = [
            dry_run_summary(
                plan,
                model_aliases=selected_aliases,
                count_per_model_override=mode_count_override,
                splits=selected_splits,
                input_usd_per_million_tokens=args.input_usd_per_million_tokens,
                output_usd_per_million_tokens=args.output_usd_per_million_tokens,
            )
            for plan in plans
        ]
        for summary, mode_config in zip(summaries, mode_configs, strict=True):
            summary["run_mode"] = selected_mode
            summary["run_mode_purpose"] = mode_config["purpose"]
            summary["generation_scope"] = args.scope
        aggregate = {
            "run_mode": selected_mode,
            "generation_scope": args.scope,
            "plan_count": len(summaries),
            "construct_ids": [summary["construct_id"] for summary in summaries],
            "complete_plan": all(summary["complete_plan"] for summary in summaries),
            "request_count": sum(summary["request_count"] for summary in summaries),
            "expected_record_count": sum(summary["expected_record_count"] for summary in summaries),
            "estimated_input_tokens": sum(summary["estimated_input_tokens"] for summary in summaries),
            "estimated_output_tokens": sum(summary["estimated_output_tokens"] for summary in summaries),
            "estimated_total_tokens": sum(summary["estimated_total_tokens"] for summary in summaries),
            "estimated_cost_usd": (
                sum(summary["estimated_cost_usd"] for summary in summaries)
                if all(summary["estimated_cost_usd"] is not None for summary in summaries)
                else None
            ),
            "records_by_split": {},
            "records_by_model": {},
        }
        for summary in summaries:
            for key in ("records_by_split", "records_by_model"):
                for name, count in summary[key].items():
                    aggregate[key][name] = aggregate[key].get(name, 0) + count
        output = {"plans": summaries, "aggregate": aggregate}
        print(json.dumps(output, indent=2, sort_keys=True))
        summary_path = _summary_path(args)
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    if args.mode is None:
        raise SystemExit("Choose an API generation mode with --mode review or --mode full.")
    mode_configs = [resolve_generation_mode(plan, args.mode)[1] for plan in plans]
    if args.mode == "full" and (
        args.count_per_model_per_cell is not None
        or args.limit_jobs is not None
        or args.models is not None
    ):
        raise SystemExit(
            "full generation mode cannot use model, count, or job limits; "
            "use review mode for a partial run."
        )
    mode_count_override = (
        int(mode_configs[0]["count_per_model_per_cell"])
        if args.mode == "review" and args.count_per_model_per_cell is None
        else args.count_per_model_per_cell
    )
    api_key = os.environ.get(args.api_key_env)
    selected_aliases = set(args.models) if args.models else None
    planned_aliases = [set(model["alias"] for model in plan["models"]) for plan in plans]
    selection_is_complete = selected_aliases is None or all(selected_aliases == aliases for aliases in planned_aliases)
    requested_partial = (
        args.limit_jobs is not None
        or mode_count_override is not None
        or not selection_is_complete
    )
    mode_allows_partial = any(bool(mode_config["partial"]) for mode_config in mode_configs)
    if requested_partial and not (args.allow_partial or mode_allows_partial):
        raise SystemExit(
            "The requested run is partial (limited jobs, model subset, or per-cell count override); "
            "pass --allow-partial or choose review mode to make that incompleteness explicit."
        )
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before generating prompts.")

    output_paths: list[Path] = []
    summaries = []
    for plan, spec in zip(plans, specs, strict=True):
        result = generate_prompt_records(
            plan,
            spec,
            api_key=api_key,
            request_fn=call_openai_responses,
            workers=args.workers,
            model_aliases=selected_aliases,
            count_per_model_override=mode_count_override,
            limit_jobs=args.limit_jobs,
            splits=selected_splits,
        )
        if not result.complete and args.scope == "all" and not (args.allow_partial or mode_allows_partial):
            raise SystemExit(
                "The requested generation is partial; pass --allow-partial or choose review mode "
                "to write it explicitly."
            )
        if args.output is not None:
            output_path = args.output
        else:
            assert args.output_dir is not None
            output_path = args.output_dir / f"{spec.construct_id}.csv"
        write_generation_result(result, output_path)
        output_paths.append(output_path)
        summaries.append(
            {
                "output": str(output_path),
                "run_mode": args.mode,
                "run_mode_purpose": mode_configs[len(summaries)]["purpose"],
                "generation_scope": args.scope,
                **result.summary(),
            }
        )

    output = {"run_mode": args.mode, "generation_scope": args.scope, "plans": summaries}
    print(json.dumps(output, indent=2, sort_keys=True))
    summary_path = _summary_path(args)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
