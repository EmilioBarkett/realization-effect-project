#!/usr/bin/env python3
"""Run local behavior generation for activation-vector behavior prompts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import PROMPT_SUFFIX  # noqa: E402
from realization_effect.runner import parse_response  # noqa: E402


FIELDNAMES = [
    "prompt_id",
    "pair_id",
    "pair_role",
    "domain",
    "outcome_valence",
    "amount_bucket",
    "risk_context",
    "behavior_target",
    "source_llm",
    "projection",
    "token_count",
    "model_id",
    "generation_prompt",
    "response_text",
    "parsed_amount",
    "valid_amount",
    "risk_profile",
    "valid_risk_profile",
    "refusal_flag",
    "parse_error_type",
    "timestamp",
]


def _load_behavior_projection_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "behavior_eval"]
    if not rows:
        raise ValueError(f"No behavior_eval rows found in {path}")
    return rows


def _load_prompt_texts(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["prompt_id"]: row["prompt_text"]
            for row in csv.DictReader(handle)
            if row.get("prompt_id") and row.get("prompt_text")
        }


def _behavior_generation_prompt(prompt_text: str) -> str:
    prompt = prompt_text
    if prompt.endswith(PROMPT_SUFFIX):
        prompt = prompt.removesuffix(PROMPT_SUFFIX).rstrip()
    return (
        prompt
        + "\n\nAnswer now. Return only the two requested integers on separate lines, "
        "with no labels and no explanation.\n"
    )


def _read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["prompt_id"] for row in csv.DictReader(handle) if row.get("prompt_id")}


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.bfloat16
        return torch.float32
    requested = requested.lower()
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    if requested in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {requested}")


def _generate_text(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_length = int(encoded["input_ids"].shape[1])
    generation_kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**generation_kwargs)
    new_tokens = output[0, input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local behavior generation for activation-vector prompts.")
    parser.add_argument("--model-id", default="models/gemma-3-4b-pt")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument(
        "--prompt-csv",
        type=Path,
        default=Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv"),
    )
    parser.add_argument(
        "--projection-csv",
        type=Path,
        default=Path(
            "results/final/activation_vectors/realization_vector_v1_layer18/"
            "evaluation/prompt_projections.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final/activation_vectors/realization_vector_v1_layer18/behavior_eval.csv"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", help="Use auto, bf16, fp16, or fp32. auto uses bf16 on CUDA/MPS.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    behavior_rows = _load_behavior_projection_rows(args.projection_csv)
    prompt_texts = _load_prompt_texts(args.prompt_csv)
    if args.limit is not None:
        behavior_rows = behavior_rows[: args.limit]

    existing_ids = _read_existing_ids(args.output) if args.resume else set()
    append = args.resume and args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer_id = args.tokenizer_id or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dtype = _resolve_dtype(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    device = _resolve_device(torch, args.device)
    model = model.to(device)
    model.eval()

    mode = "a" if append else "w"
    written = 0
    with args.output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        for index, row in enumerate(behavior_rows, start=1):
            prompt_id = row["prompt_id"]
            if prompt_id in existing_ids:
                continue
            prompt_text = prompt_texts[prompt_id]
            generation_prompt = _behavior_generation_prompt(prompt_text)
            response_text = _generate_text(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                device=device,
                prompt=generation_prompt,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
            )
            amount, risk, valid_amount, valid_risk, refusal, error_type = parse_response(response_text)
            writer.writerow(
                {
                    "prompt_id": prompt_id,
                    "pair_id": row.get("pair_id", ""),
                    "pair_role": row.get("pair_role", ""),
                    "domain": row.get("domain", ""),
                    "outcome_valence": row.get("outcome_valence", ""),
                    "amount_bucket": row.get("amount_bucket", ""),
                    "risk_context": row.get("risk_context", ""),
                    "behavior_target": row.get("behavior_target", ""),
                    "source_llm": row.get("source_llm", ""),
                    "projection": row.get("projection", ""),
                    "token_count": row.get("token_count", ""),
                    "model_id": args.model_id,
                    "generation_prompt": generation_prompt,
                    "response_text": response_text,
                    "parsed_amount": "" if amount is None else amount,
                    "valid_amount": valid_amount,
                    "risk_profile": "" if risk is None else risk,
                    "valid_risk_profile": valid_risk,
                    "refusal_flag": refusal,
                    "parse_error_type": error_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            written += 1
            print(f"wrote behavior {index}/{len(behavior_rows)}: {prompt_id}", flush=True)

    summary = {
        "output": str(args.output),
        "model_id": args.model_id,
        "prompt_count": len(behavior_rows),
        "written": written,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
