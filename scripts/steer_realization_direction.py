#!/usr/bin/env python3
"""Run generation-time realization-direction steering experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
from statistics import mean
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import PROMPT_SUFFIX  # noqa: E402
from activation_analysis.steering import ResidualSteeringGenerator, SteeringConfig  # noqa: E402
from realization_effect.runner import parse_response  # noqa: E402


SYSTEM_PROMPT = (
    "You are completing a controlled decision task for a research experiment. "
    "Return only the requested numeric answer. Do not explain your reasoning."
)


FIELDNAMES = [
    "run_id",
    "row_index",
    "prompt_id",
    "pair_id",
    "pair_role",
    "domain",
    "outcome_valence",
    "amount_bucket",
    "risk_context",
    "behavior_target",
    "source_llm",
    "model_id",
    "backend",
    "prompt_format",
    "generation_prompt",
    "steering_label",
    "steering_scale",
    "steering_layer",
    "steering_position_mode",
    "direction_path",
    "direction_hidden_size",
    "direction_raw_norm",
    "direction_normalized",
    "resolved_block_path",
    "response_text",
    "response_integer_count",
    "response_exactly_two_integers",
    "parsed_amount",
    "valid_amount",
    "risk_profile",
    "valid_risk_profile",
    "refusal_flag",
    "parse_error_type",
    "timestamp",
]


INTEGER_PATTERN = re.compile(r"-?\d+")


def _load_prompt_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["prompt_id"]: row
            for row in csv.DictReader(handle)
            if row.get("prompt_id") and row.get("prompt_text")
        }


def _load_behavior_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "behavior_eval"]
    if not rows:
        raise ValueError(f"No behavior_eval rows found in {path}")
    return rows


def _behavior_generation_prompt(prompt_text: str, *, answer_contract: str = "two_line") -> str:
    prompt = prompt_text
    if prompt.endswith(PROMPT_SUFFIX):
        prompt = prompt.removesuffix(PROMPT_SUFFIX).rstrip()
    if answer_contract == "plain":
        return prompt + "\n\nAnswer now.\n"
    if answer_contract != "two_line":
        raise ValueError(f"Unsupported answer contract: {answer_contract}")
    return (
        prompt
        + "\n\nAnswer now. Return only the two requested integers on separate lines, "
        "with no labels and no explanation.\n"
    )


def _default_run_name(model_id: str) -> str:
    model_slug = "".join(char if char.isalnum() else "_" for char in model_id.lower()).strip("_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_slug or 'model'}__realization_steering__{timestamp}"


def _parse_scales(scales: str | None, scale_values: list[float] | None) -> list[float]:
    if scale_values:
        values = scale_values
    elif scales:
        values = [float(value.strip()) for value in scales.split(",") if value.strip()]
    else:
        values = [-150.0, -75.0, 0.0, 75.0, 150.0]
    if not values:
        raise argparse.ArgumentTypeError("At least one steering scale is required.")
    return sorted(set(values))


def _steering_label(scale: float) -> str:
    if scale > 0:
        return "plus_realized_closed"
    if scale < 0:
        return "minus_realized_closed"
    return "unsteered"


def _response_integers(response_text: str) -> list[int]:
    return [int(match.group(0)) for match in INTEGER_PATTERN.finditer(response_text)]


def _read_existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (row["prompt_id"], row["steering_scale"])
            for row in csv.DictReader(handle)
            if row.get("prompt_id") and row.get("steering_scale")
        }


def _value(row: dict[str, str], prompt_row: dict[str, str], key: str) -> str:
    return row.get(key) or prompt_row.get(key, "")


def _float_or_none(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_scale: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_prompt_scale: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        scale = row["steering_scale"]
        by_scale[scale].append(row)
        by_prompt_scale[(row["prompt_id"], scale)] = row

    scale_summary = {}
    for scale, scale_rows in sorted(by_scale.items(), key=lambda item: float(item[0])):
        valid_amounts = [float(row["parsed_amount"]) for row in scale_rows if _is_true(row.get("valid_amount", ""))]
        valid_risks = [float(row["risk_profile"]) for row in scale_rows if _is_true(row.get("valid_risk_profile", ""))]
        exactly_two_integers = [
            row for row in scale_rows if _is_true(row.get("response_exactly_two_integers", ""))
        ]
        both_valid = [
            row
            for row in scale_rows
            if _is_true(row.get("valid_amount", "")) and _is_true(row.get("valid_risk_profile", ""))
        ]
        scale_summary[scale] = {
            "rows": len(scale_rows),
            "valid_amount_rows": len(valid_amounts),
            "valid_risk_rows": len(valid_risks),
            "both_valid_rows": len(both_valid),
            "response_exactly_two_integer_rows": len(exactly_two_integers),
            "mean_amount": _mean(valid_amounts),
            "mean_risk": _mean(valid_risks),
        }

    baseline_scale = "0.0" if "0.0" in by_scale else "0"
    baseline_deltas = {}
    if baseline_scale in by_scale:
        for scale in sorted(by_scale, key=float):
            if scale == baseline_scale:
                continue
            amount_deltas: list[float] = []
            risk_deltas: list[float] = []
            for row in by_scale[scale]:
                baseline = by_prompt_scale.get((row["prompt_id"], baseline_scale))
                if baseline is None:
                    continue
                if _is_true(row.get("valid_amount", "")) and _is_true(baseline.get("valid_amount", "")):
                    steered_amount = _float_or_none(row.get("parsed_amount", ""))
                    baseline_amount = _float_or_none(baseline.get("parsed_amount", ""))
                    if steered_amount is not None and baseline_amount is not None:
                        amount_deltas.append(steered_amount - baseline_amount)
                if _is_true(row.get("valid_risk_profile", "")) and _is_true(baseline.get("valid_risk_profile", "")):
                    steered_risk = _float_or_none(row.get("risk_profile", ""))
                    baseline_risk = _float_or_none(baseline.get("risk_profile", ""))
                    if steered_risk is not None and baseline_risk is not None:
                        risk_deltas.append(steered_risk - baseline_risk)
            baseline_deltas[scale] = {
                "paired_amount_rows": len(amount_deltas),
                "paired_risk_rows": len(risk_deltas),
                "mean_amount_delta_from_unsteered": _mean(amount_deltas),
                "mean_risk_delta_from_unsteered": _mean(risk_deltas),
            }

    return {
        "rows": len(rows),
        "scale_summary": scale_summary,
        "baseline_scale": baseline_scale if baseline_scale in by_scale else None,
        "deltas_from_unsteered": baseline_deltas,
    }


def _write_summary(output: Path, summary_output: Path) -> None:
    with output.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary_output.write_text(
        json.dumps(_summarize_rows(rows), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _manifest(
    args: argparse.Namespace,
    *,
    output: Path,
    summary_output: Path,
    run_id: str,
    prompt_count: int,
    scales: list[float],
    generator: ResidualSteeringGenerator,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "summary_output": str(summary_output),
        "prompt_csv": str(args.prompt_csv),
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "backend": "transformers",
        "prompt_format": args.prompt_format,
        "answer_contract": args.answer_contract,
        "prompt_count": prompt_count,
        "limit": args.limit,
        "scales": scales,
        "direction": str(args.direction),
        "layer": args.layer,
        "position_mode": args.position_mode,
        "normalize_direction": not args.no_normalize_direction,
        "device": generator.resolved_device,
        "device_map": args.device_map or "",
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation or "",
        "resolved_block_path": generator.resolved_block_path or "",
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "system_prompt": args.system_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma realization-direction steering over behavior prompts.")
    parser.add_argument("--model-id", default="models/gemma-3-4b-pt")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument(
        "--prompt-csv",
        type=Path,
        default=Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv"),
    )
    parser.add_argument(
        "--direction",
        type=Path,
        default=Path("results/final/activation_vectors/realization_vector_v1_layer18/mean_direction.npy"),
    )
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument(
        "--position-mode",
        choices=["all", "last"],
        default="last",
        help="Token positions to steer in each forward call. 'last' is the recommended generation smoke mode.",
    )
    parser.add_argument(
        "--scales",
        default=None,
        help="Comma-separated scales. Use --scales=-150,-75,0,75,150 when the first value is negative.",
    )
    parser.add_argument(
        "--scale",
        dest="scale_values",
        action="append",
        type=float,
        help="Add one scale value. Can be repeated, e.g. --scale -150 --scale 0 --scale 150.",
    )
    parser.add_argument("--no-normalize-direction", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Defaults to <output-dir>/<run-name>/steering_eval.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/test/activation_vectors/steering_runs"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--prompt-format", choices=["completion", "chat"], default="completion")
    parser.add_argument("--answer-contract", choices=["two_line", "plain"], default="two_line")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default="auto", help="Use auto, bf16, fp16, or fp32.")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--block-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    scales = _parse_scales(args.scales, args.scale_values)
    prompt_rows = _load_prompt_rows(args.prompt_csv)
    behavior_rows = _load_behavior_rows(args.prompt_csv)
    if args.limit is not None:
        behavior_rows = behavior_rows[: args.limit]

    run_id = args.run_name or _default_run_name(args.model_id)
    output = args.output or (args.output_dir / run_id / "steering_eval.csv")
    summary_output = output.parent / "steering_summary.json"
    manifest_path = output.parent / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    generator = ResidualSteeringGenerator(
        args.model_id,
        tokenizer_id=args.tokenizer_id,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        block_path=args.block_path,
    )
    generator.resolve_block_path()

    existing_keys = _read_existing_keys(output) if args.resume else set()
    append = args.resume and output.exists()
    manifest_path.write_text(
        json.dumps(
            _manifest(
                args,
                output=output,
                summary_output=summary_output,
                run_id=run_id,
                prompt_count=len(behavior_rows),
                scales=scales,
                generator=generator,
            ),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    mode = "a" if append else "w"
    written = 0
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        total = len(behavior_rows) * len(scales)
        for index, row in enumerate(behavior_rows, start=1):
            prompt_id = row["prompt_id"]
            prompt_row = prompt_rows[prompt_id]
            generation_prompt = _behavior_generation_prompt(
                prompt_row["prompt_text"],
                answer_contract=args.answer_contract,
            )
            for scale in scales:
                scale_text = str(float(scale))
                if (prompt_id, scale_text) in existing_keys:
                    continue
                steering_config = SteeringConfig(
                    direction_path=args.direction,
                    layer=args.layer,
                    scale=float(scale),
                    position_mode=args.position_mode,
                    normalize_direction=not args.no_normalize_direction,
                )
                response_text, vector_info = generator.generate(
                    generation_prompt,
                    prompt_format=args.prompt_format,
                    system_prompt=args.system_prompt,
                    steering_config=steering_config,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                )
                if vector_info is None:
                    _, vector_info = generator.load_direction(steering_config)
                amount, risk, valid_amount, valid_risk, refusal, error_type = parse_response(response_text)
                response_integers = _response_integers(response_text)
                writer.writerow(
                    {
                        "run_id": run_id,
                        "row_index": index,
                        "prompt_id": prompt_id,
                        "pair_id": _value(row, prompt_row, "pair_id"),
                        "pair_role": _value(row, prompt_row, "pair_role"),
                        "domain": _value(row, prompt_row, "domain"),
                        "outcome_valence": _value(row, prompt_row, "outcome_valence"),
                        "amount_bucket": _value(row, prompt_row, "amount_bucket"),
                        "risk_context": _value(row, prompt_row, "risk_context"),
                        "behavior_target": _value(row, prompt_row, "behavior_target"),
                        "source_llm": _value(row, prompt_row, "source_llm"),
                        "model_id": args.model_id,
                        "backend": "transformers",
                        "prompt_format": args.prompt_format,
                        "generation_prompt": generation_prompt,
                        "steering_label": _steering_label(float(scale)),
                        "steering_scale": scale_text,
                        "steering_layer": args.layer,
                        "steering_position_mode": args.position_mode,
                        "direction_path": vector_info.path,
                        "direction_hidden_size": vector_info.hidden_size,
                        "direction_raw_norm": vector_info.raw_norm,
                        "direction_normalized": vector_info.normalized,
                        "resolved_block_path": generator.resolved_block_path or "",
                        "response_text": response_text,
                        "response_integer_count": len(response_integers),
                        "response_exactly_two_integers": len(response_integers) == 2,
                        "parsed_amount": "" if amount is None else amount,
                        "valid_amount": valid_amount,
                        "risk_profile": "" if risk is None else risk,
                        "valid_risk_profile": valid_risk,
                        "refusal_flag": refusal,
                        "parse_error_type": error_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                handle.flush()
                written += 1
                print(f"wrote steering {written}/{total}: {prompt_id} scale={scale_text}", flush=True)

    _write_summary(output, summary_output)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output": str(output),
                "summary_output": str(summary_output),
                "manifest": str(manifest_path),
                "model_id": args.model_id,
                "prompt_count": len(behavior_rows),
                "scales": scales,
                "written": written,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
