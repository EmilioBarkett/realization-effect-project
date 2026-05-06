#!/usr/bin/env python3
"""Score discrete behavior choices for activation-vector behavior prompts."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import PROMPT_SUFFIX  # noqa: E402


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
    "choice_amount",
    "choice_risk",
    "choice_logprob_sum",
    "choice_logprob_mean",
    "choice_token_count",
    "score_mode",
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


def _behavior_scoring_prompt(prompt_text: str) -> str:
    prompt = prompt_text
    if prompt.endswith(PROMPT_SUFFIX):
        prompt = prompt.removesuffix(PROMPT_SUFFIX).rstrip()
    return prompt + "\n\nAnswer:\n"


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
        return torch.float32
    requested = requested.lower()
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    if requested in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {requested}")


def _candidate_completion(amount: int, risk: int) -> str:
    return f"{amount}\n{risk}"


def _score_candidates(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    prompt: str,
    candidates: list[tuple[int, int]],
    batch_size: int,
    max_length: int,
) -> list[dict[str, float | int]]:
    del batch_size
    encoded_prompt = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded_prompt["input_ids"].to(device)
    scored: list[dict[str, float | int]] = []

    with torch.no_grad():
        prefix_outputs = model(input_ids=input_ids, use_cache=True)
    base_logits = prefix_outputs.logits[:, -1, :]
    base_past = prefix_outputs.past_key_values

    for amount, risk in candidates:
        completion_ids = tokenizer(
            _candidate_completion(amount, risk),
            add_special_tokens=False,
        )["input_ids"]
        logits_for_next = base_logits
        past = base_past
        logprob_sum = 0.0
        token_count = 0

        with torch.no_grad():
            for token_index, token_id in enumerate(completion_ids):
                log_probs = torch.log_softmax(logits_for_next[0].float(), dim=-1)
                logprob_sum += float(log_probs[token_id].item())
                token_count += 1
                if token_index == len(completion_ids) - 1:
                    break
                token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
                outputs = model(input_ids=token_tensor, past_key_values=past, use_cache=True)
                logits_for_next = outputs.logits[:, -1, :]
                past = outputs.past_key_values

        logprob_mean = logprob_sum / token_count if token_count else float("-inf")
        scored.append(
            {
                "amount": amount,
                "risk": risk,
                "logprob_sum": logprob_sum,
                "logprob_mean": logprob_mean,
                "token_count": token_count,
            }
        )
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Score discrete behavior choices for activation-vector prompts.")
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
        default=Path("results/final/activation_vectors/realization_vector_v1_layer18/behavior_choice_scores.csv"),
    )
    parser.add_argument("--amount-grid", default="50,100,200,300,500,750,1000")
    parser.add_argument("--risk-grid", default="1,2,3,4,5")
    parser.add_argument("--score-mode", choices=["mean", "sum"], default="mean")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--choice-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    amount_grid = [int(value) for value in args.amount_grid.split(",") if value.strip()]
    risk_grid = [int(value) for value in args.risk_grid.split(",") if value.strip()]
    candidates = list(itertools.product(amount_grid, risk_grid))
    behavior_rows = _load_behavior_projection_rows(args.projection_csv)
    if args.limit is not None:
        behavior_rows = behavior_rows[: args.limit]
    prompt_texts = _load_prompt_texts(args.prompt_csv)
    existing_ids = _read_existing_ids(args.output) if args.resume else set()

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    written = 0
    score_key = "logprob_mean" if args.score_mode == "mean" else "logprob_sum"
    with args.output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        for index, row in enumerate(behavior_rows, start=1):
            prompt_id = row["prompt_id"]
            if prompt_id in existing_ids:
                continue
            prompt = _behavior_scoring_prompt(prompt_texts[prompt_id])
            scored = _score_candidates(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                device=device,
                prompt=prompt,
                candidates=candidates,
                batch_size=args.choice_batch_size,
                max_length=args.max_length,
            )
            best = max(scored, key=lambda item: float(item[score_key]))
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
                    "choice_amount": best["amount"],
                    "choice_risk": best["risk"],
                    "choice_logprob_sum": best["logprob_sum"],
                    "choice_logprob_mean": best["logprob_mean"],
                    "choice_token_count": best["token_count"],
                    "score_mode": args.score_mode,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            written += 1
            print(
                f"scored behavior {index}/{len(behavior_rows)}: "
                f"{prompt_id} -> {best['amount']}/{best['risk']}",
                flush=True,
            )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_id": args.model_id,
                "prompt_count": len(behavior_rows),
                "written": written,
                "candidate_count": len(candidates),
                "score_mode": args.score_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
