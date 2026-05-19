#!/usr/bin/env python3
"""Run realization-direction steering on direct realized-vs-paper classification."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from contextlib import nullcontext
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import PROMPT_SUFFIX  # noqa: E402
from activation_analysis.steering import ResidualSteeringGenerator, SteeringConfig  # noqa: E402


SYSTEM_PROMPT = (
    "You are completing a controlled realization-classification task for a research experiment. "
    "Return only the requested label."
)

LABEL_PATTERN = re.compile(r"[A-Za-z_]+|[01]")
LABEL_WORD_PATTERN = re.compile(r"\b(REALIZED|PAPER|REALIZED_CLOSED|PAPER_OPEN|OPEN|UNREALIZED)\b", re.IGNORECASE)
LABEL_MAP = {
    "REALIZED": "REALIZED",
    "REALIZED_CLOSED": "REALIZED",
    "CLOSED": "REALIZED",
    "1": "REALIZED",
    "PAPER": "PAPER",
    "PAPER_OPEN": "PAPER",
    "OPEN": "PAPER",
    "UNREALIZED": "PAPER",
    "0": "PAPER",
}

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
    "expected_label",
    "classification_mode",
    "score_realized",
    "score_paper",
    "score_realized_per_token",
    "score_paper_per_token",
    "score_realized_calibrated",
    "score_paper_calibrated",
    "response_text",
    "response_token_count",
    "response_single_token",
    "parsed_label",
    "valid_label",
    "correct_label",
    "refusal_flag",
    "parse_error_type",
    "timestamp",
]


def _classification_prompt(prompt_text: str) -> str:
    prompt = prompt_text
    instruction_marker = "Answer now."
    if instruction_marker in prompt:
        prompt = prompt.split(instruction_marker, maxsplit=1)[0].rstrip()
    if prompt.endswith(PROMPT_SUFFIX):
        prompt = prompt.removesuffix(PROMPT_SUFFIX).rstrip()
    prompt = re.sub(
        r"Answer with exactly two integers on separate lines:[^.]*\.",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    return (
        f"{prompt}\n\n"
        "Task: Determine whether the described outcome is already realized "
        "(cashed out or position closed) or still paper (open/unrealized).\n"
        "Answer now.\n"
        "Return exactly one label: REALIZED or PAPER.\n"
        "Label:"
    )


def _manual_gemma_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _classification_calibration_prompt() -> str:
    return (
        "Answer now.\n"
        "Write exactly one label based on the scenario.\n"
        "Allowed labels: REALIZED or PAPER.\n"
        "Label:"
    )


def _candidate_ids(generator: ResidualSteeringGenerator, label: str) -> list[int]:
    ids = generator.tokenizer.encode(f" {label}", add_special_tokens=False)
    if not ids:
        raise ValueError(f"Candidate label '{label}' produced no token ids.")
    return ids


def _score_label(
    generator: ResidualSteeringGenerator,
    *,
    prompt: str,
    prompt_format: str,
    system_prompt: str,
    label_ids: list[int],
    steering_config: SteeringConfig | None,
) -> float:
    torch = generator._torch
    formatted_prompt = generator.format_prompt(prompt, prompt_format=prompt_format, system_prompt=system_prompt)
    encoded = generator.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encoded["input_ids"].to(generator.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=generator.device)
    else:
        attention_mask = attention_mask.to(generator.device)

    score = 0.0
    hook_ctx = (
        generator.steering_hooks(steering_config)
        if steering_config is not None and steering_config.scale != 0
        else nullcontext(None)
    )
    with hook_ctx:
        with torch.no_grad():
            for token_id in label_ids:
                outputs = generator.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                score += float(log_probs[0, token_id].item())

                next_token = torch.tensor([[token_id]], device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                next_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, next_mask], dim=1)
    return score


def _score_per_token(score: float, label_ids: list[int]) -> float:
    if not label_ids:
        return score
    return score / float(len(label_ids))


def _calibrated_score(score_per_token: float, prior_score_per_token: float) -> float:
    return score_per_token - prior_score_per_token


def _default_run_name(model_id: str) -> str:
    model_slug = "".join(char if char.isalnum() else "_" for char in model_id.lower()).strip("_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_slug or 'model'}__realization_classification_steering__{timestamp}"


def _expected_label(pair_role: str) -> str:
    role = pair_role.strip().lower()
    if role == "realized_closed":
        return "REALIZED"
    if role == "paper_open":
        return "PAPER"
    return ""


def _parse_scales(scales: str | None, scale_values: list[float] | None) -> list[float]:
    if scale_values:
        values = scale_values
    elif scales:
        values = [float(value.strip()) for value in scales.split(",") if value.strip()]
    else:
        values = [-50.0, 0.0, 50.0]
    if not values:
        raise argparse.ArgumentTypeError("At least one steering scale is required.")
    return sorted(set(values))


def _steering_label(scale: float) -> str:
    if scale > 0:
        return "plus_realized_closed"
    if scale < 0:
        return "minus_realized_closed"
    return "unsteered"


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


def _read_existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (row["prompt_id"], row["steering_scale"])
            for row in csv.DictReader(handle)
            if row.get("prompt_id") and row.get("steering_scale")
        }


def _response_tokens(response_text: str) -> list[str]:
    return [token.group(0) for token in LABEL_PATTERN.finditer(response_text)]


def _parse_label(response_text: str) -> tuple[str, bool, str]:
    text = response_text.strip()
    if not text:
        return "", False, "empty_response"

    label_hits = [LABEL_MAP.get(match.group(1).upper(), "") for match in LABEL_WORD_PATTERN.finditer(text)]
    label_hits = [label for label in label_hits if label]
    unique_hits = list(dict.fromkeys(label_hits))
    if len(unique_hits) == 1:
        return unique_hits[0], True, ""
    if len(unique_hits) > 1:
        return "", False, "ambiguous_label"

    tokens = _response_tokens(text)
    if not tokens:
        return "", False, "empty_response"
    canonical = LABEL_MAP.get(tokens[0].upper(), "")
    if not canonical:
        return "", False, "invalid_label"
    return canonical, True, ""


def _value(row: dict[str, str], prompt_row: dict[str, str], key: str) -> str:
    return row.get(key) or prompt_row.get(key, "")


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _float_bool_mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [1.0 for row in rows if _is_true(row.get(key, ""))]
    return (sum(values) / len(rows)) if rows else None


def _summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_scale: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_prompt_scale: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        scale = row["steering_scale"]
        by_scale[scale].append(row)
        by_prompt_scale[(row["prompt_id"], scale)] = row

    scale_summary = {}
    for scale, scale_rows in sorted(by_scale.items(), key=lambda item: float(item[0])):
        valid_rows = [row for row in scale_rows if _is_true(row.get("valid_label", ""))]
        correct_rows = [row for row in scale_rows if _is_true(row.get("correct_label", ""))]
        realized_preds = [row for row in valid_rows if row.get("parsed_label", "") == "REALIZED"]
        single_token_rows = [row for row in scale_rows if _is_true(row.get("response_single_token", ""))]
        scale_summary[scale] = {
            "rows": len(scale_rows),
            "valid_label_rows": len(valid_rows),
            "single_token_rows": len(single_token_rows),
            "accuracy": (len(correct_rows) / len(valid_rows)) if valid_rows else None,
            "realized_prediction_rate": (len(realized_preds) / len(valid_rows)) if valid_rows else None,
        }

    baseline_scale = "0.0" if "0.0" in by_scale else "0"
    baseline_deltas = {}
    if baseline_scale in by_scale:
        for scale in sorted(by_scale, key=float):
            if scale == baseline_scale:
                continue
            paired_valid = 0
            correctness_shift = []
            realized_shift = []
            for row in by_scale[scale]:
                baseline = by_prompt_scale.get((row["prompt_id"], baseline_scale))
                if baseline is None:
                    continue
                if _is_true(row.get("valid_label", "")) and _is_true(baseline.get("valid_label", "")):
                    paired_valid += 1
                    correctness_shift.append(
                        (1.0 if _is_true(row.get("correct_label", "")) else 0.0)
                        - (1.0 if _is_true(baseline.get("correct_label", "")) else 0.0)
                    )
                    realized_shift.append(
                        (1.0 if row.get("parsed_label", "") == "REALIZED" else 0.0)
                        - (1.0 if baseline.get("parsed_label", "") == "REALIZED" else 0.0)
                    )
            baseline_deltas[scale] = {
                "paired_valid_rows": paired_valid,
                "mean_accuracy_delta_from_unsteered": _mean(correctness_shift),
                "mean_realized_prediction_rate_delta_from_unsteered": _mean(realized_shift),
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
        "task": "realization_classification",
        "output": str(output),
        "summary_output": str(summary_output),
        "prompt_csv": str(args.prompt_csv),
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "backend": "transformers",
        "prompt_format": args.prompt_format,
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
        "score_calibration": args.score_calibration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realization-label steering positive control on behavior_eval prompts.")
    parser.add_argument("--model-id", default="models/gemma-3-4b-pt")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument(
        "--prompt-csv",
        type=Path,
        default=Path(
            "experiments/activation_analysis/prompts/activation_vectors/"
            "realization_vector_v1_realization_classification.csv"
        ),
    )
    parser.add_argument(
        "--direction",
        type=Path,
        default=Path("results/final/activation_vectors/realization_vector_v1_layer18/mean_direction.npy"),
    )
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--position-mode", choices=["all", "last"], default="last")
    parser.add_argument("--scales", default=None)
    parser.add_argument("--scale", dest="scale_values", action="append", type=float)
    parser.add_argument("--no-normalize-direction", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/test/activation_vectors/steering_runs"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--prompt-format", choices=["completion", "chat"], default="completion")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--block-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--classification-mode", choices=["score", "generate"], default="score")
    parser.add_argument("--score-calibration", choices=["none", "pmi"], default="pmi")
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
    manual_chat_wrapper = False
    if args.prompt_format == "chat" and not getattr(generator.tokenizer, "chat_template", None):
        if "gemma" in args.model_id.lower():
            manual_chat_wrapper = True
            print(
                "chat template unavailable for tokenizer; using manual gemma chat wrapper with completion mode",
                flush=True,
            )
        else:
            print("chat template unavailable for tokenizer; falling back to completion prompt format", flush=True)
            args.prompt_format = "completion"
    realized_ids = _candidate_ids(generator, "REALIZED")
    paper_ids = _candidate_ids(generator, "PAPER")
    calibration_prompt = _classification_calibration_prompt()
    prior_prompt_format = "completion" if manual_chat_wrapper else args.prompt_format
    prior_prompt_text = (
        _manual_gemma_chat_prompt(args.system_prompt, calibration_prompt)
        if manual_chat_wrapper
        else calibration_prompt
    )

    prior_scores_by_scale: dict[str, tuple[float, float]] = {}
    if args.classification_mode == "score" and args.score_calibration == "pmi":
        for scale in scales:
            steering_config = SteeringConfig(
                direction_path=args.direction,
                layer=args.layer,
                scale=float(scale),
                position_mode=args.position_mode,
                normalize_direction=not args.no_normalize_direction,
            )
            prior_realized = _score_label(
                generator,
                prompt=prior_prompt_text,
                prompt_format=prior_prompt_format,
                system_prompt=args.system_prompt,
                label_ids=realized_ids,
                steering_config=steering_config,
            )
            prior_paper = _score_label(
                generator,
                prompt=prior_prompt_text,
                prompt_format=prior_prompt_format,
                system_prompt=args.system_prompt,
                label_ids=paper_ids,
                steering_config=steering_config,
            )
            prior_scores_by_scale[str(float(scale))] = (
                _score_per_token(prior_realized, realized_ids),
                _score_per_token(prior_paper, paper_ids),
            )

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
            generation_prompt = _classification_prompt(prompt_row["prompt_text"])
            prompt_for_generation = (
                _manual_gemma_chat_prompt(args.system_prompt, generation_prompt)
                if manual_chat_wrapper
                else generation_prompt
            )
            prompt_format_for_generation = "completion" if manual_chat_wrapper else args.prompt_format
            expected_label = _expected_label(_value(row, prompt_row, "pair_role"))
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
                    prompt_for_generation,
                    prompt_format=prompt_format_for_generation,
                    system_prompt=args.system_prompt,
                    steering_config=steering_config,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                ) if args.classification_mode == "generate" else ("", None)
                if vector_info is None:
                    _, vector_info = generator.load_direction(steering_config)

                score_realized = None
                score_paper = None
                score_realized_per_token = None
                score_paper_per_token = None
                score_realized_calibrated = None
                score_paper_calibrated = None
                if args.classification_mode == "score":
                    score_realized = _score_label(
                        generator,
                        prompt=prompt_for_generation,
                        prompt_format=prompt_format_for_generation,
                        system_prompt=args.system_prompt,
                        label_ids=realized_ids,
                        steering_config=steering_config,
                    )
                    score_paper = _score_label(
                        generator,
                        prompt=prompt_for_generation,
                        prompt_format=prompt_format_for_generation,
                        system_prompt=args.system_prompt,
                        label_ids=paper_ids,
                        steering_config=steering_config,
                    )
                    score_realized_per_token = _score_per_token(score_realized, realized_ids)
                    score_paper_per_token = _score_per_token(score_paper, paper_ids)
                    if args.score_calibration == "pmi":
                        prior_realized_per_token, prior_paper_per_token = prior_scores_by_scale[scale_text]
                        score_realized_calibrated = _calibrated_score(
                            score_realized_per_token,
                            prior_realized_per_token,
                        )
                        score_paper_calibrated = _calibrated_score(
                            score_paper_per_token,
                            prior_paper_per_token,
                        )
                        parsed_label = "REALIZED" if score_realized_calibrated >= score_paper_calibrated else "PAPER"
                    else:
                        parsed_label = "REALIZED" if score_realized_per_token >= score_paper_per_token else "PAPER"
                    valid_label = True
                    parse_error_type = ""
                    response_text = f"SCORED_CLASSIFICATION: {parsed_label}"
                    response_tokens = [parsed_label]
                    single_token = True
                else:
                    parsed_label, valid_label, parse_error_type = _parse_label(response_text)
                    response_tokens = _response_tokens(response_text)
                    single_token = len(response_tokens) == 1
                refusal_flag = "true" if valid_label is False and "refus" in response_text.lower() else "false"
                correct_label = bool(valid_label and expected_label and parsed_label == expected_label)

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
                        "behavior_target": "realization_classification",
                        "source_llm": _value(row, prompt_row, "source_llm"),
                        "model_id": args.model_id,
                        "backend": "transformers",
                        "prompt_format": prompt_format_for_generation,
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
                        "expected_label": expected_label,
                        "classification_mode": args.classification_mode,
                        "score_realized": "" if score_realized is None else score_realized,
                        "score_paper": "" if score_paper is None else score_paper,
                        "score_realized_per_token": (
                            "" if score_realized_per_token is None else score_realized_per_token
                        ),
                        "score_paper_per_token": "" if score_paper_per_token is None else score_paper_per_token,
                        "score_realized_calibrated": (
                            "" if score_realized_calibrated is None else score_realized_calibrated
                        ),
                        "score_paper_calibrated": (
                            "" if score_paper_calibrated is None else score_paper_calibrated
                        ),
                        "response_text": response_text,
                        "response_token_count": len(response_tokens),
                        "response_single_token": single_token,
                        "parsed_label": parsed_label,
                        "valid_label": valid_label,
                        "correct_label": correct_label,
                        "refusal_flag": refusal_flag,
                        "parse_error_type": parse_error_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                handle.flush()
                written += 1
                print(f"wrote classification steering {written}/{total}: {prompt_id} scale={scale_text}", flush=True)

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
