#!/usr/bin/env python3
"""Run behavior generation for activation-vector behavior prompts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.openrouter_prompt_generation import PROMPT_SUFFIX  # noqa: E402
from realization_effect.runner import parse_response  # noqa: E402


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
    "projection",
    "token_count",
    "model_id",
    "backend",
    "prompt_format",
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


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = (
    "You are completing a controlled decision task for a research experiment. "
    "Return only the requested numeric answer. Do not explain your reasoning."
)


def _load_behavior_rows_from_projection(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "behavior_eval"]
    if not rows:
        raise ValueError(f"No behavior_eval rows found in {path}")
    return rows


def _load_behavior_rows_from_prompt_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "behavior_eval"]
    if not rows:
        raise ValueError(f"No behavior_eval rows found in {path}")
    return rows


def _load_prompt_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["prompt_id"]: row
            for row in csv.DictReader(handle)
            if row.get("prompt_id") and row.get("prompt_text")
        }


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


def _read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["prompt_id"] for row in csv.DictReader(handle) if row.get("prompt_id")}


def _value(row: dict[str, str], prompt_row: dict[str, str], key: str) -> str:
    return row.get(key) or prompt_row.get(key, "")


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
    prompt_format: str,
    system_prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    max_length: int,
    do_sample: bool,
    temperature: float,
) -> str:
    if prompt_format == "chat":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template; use --prompt-format completion.")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif prompt_format != "completion":
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
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


def _extract_openrouter_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response did not include choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response choice did not include a message.")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
        return "".join(parts).strip()
    return ""


def _openrouter_generate_text(
    *,
    model_id: str,
    prompt: str,
    prompt_format: str,
    system_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    api_key: str,
    timeout: float,
    retries: int,
    reasoning_effort: str | None,
) -> str:
    if prompt_format == "chat":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    elif prompt_format == "completion":
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature if do_sample else 0,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/EmilioBarkett/realization-effect-project",
        "X-Title": "Realization Effect Behavior Eval",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return _extract_openrouter_text(json.loads(response.read().decode("utf-8")))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _default_run_name(model_id: str) -> str:
    model_slug = "".join(char if char.isalnum() else "_" for char in model_id.lower()).strip("_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_slug or 'model'}__behavior_eval__{timestamp}"


def _manifest(args: argparse.Namespace, *, output: Path, run_id: str, prompt_count: int, device: str, dtype: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "prompt_csv": str(args.prompt_csv),
        "projection_csv": "" if args.projection_csv is None else str(args.projection_csv),
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "backend": args.backend,
        "prompt_format": args.prompt_format,
        "answer_contract": args.answer_contract,
        "prompt_count": prompt_count,
        "limit": args.limit,
        "device": device,
        "device_map": args.device_map or "",
        "dtype": dtype,
        "attn_implementation": args.attn_implementation or "",
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "system_prompt": args.system_prompt,
        "api_key_env": args.api_key_env if args.backend == "openrouter" else "",
        "openrouter_reasoning_effort": args.openrouter_reasoning_effort or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behavior generation for activation-vector prompts.")
    parser.add_argument("--model-id", default="models/gemma-3-4b-pt")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--backend", choices=["transformers", "openrouter"], default="transformers")
    parser.add_argument(
        "--prompt-csv",
        type=Path,
        default=Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv"),
    )
    parser.add_argument(
        "--projection-csv",
        type=Path,
        default=None,
        help="Optional projection CSV. If omitted, behavior rows are loaded directly from --prompt-csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Defaults to <output-dir>/behavior_eval.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final/activation_vectors/behavior_runs"),
        help="Directory for run outputs when --output is omitted.",
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
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional Hugging Face device_map, e.g. auto for larger cloud models.",
    )
    parser.add_argument("--dtype", default="auto", help="Use auto, bf16, fp16, or fp32. auto uses bf16 on CUDA/MPS.")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument(
        "--openrouter-reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default=None,
        help="Optional OpenRouter reasoning effort; use none for Qwen3.5 non-thinking mode.",
    )
    args = parser.parse_args()

    prompt_rows = _load_prompt_rows(args.prompt_csv)
    behavior_rows = (
        _load_behavior_rows_from_projection(args.projection_csv)
        if args.projection_csv is not None
        else _load_behavior_rows_from_prompt_csv(args.prompt_csv)
    )
    if args.limit is not None:
        behavior_rows = behavior_rows[: args.limit]

    run_id = args.run_name or _default_run_name(args.model_id)
    output = args.output or (args.output_dir / run_id / "behavior_eval.csv")
    existing_ids = _read_existing_ids(output) if args.resume else set()
    append = args.resume and output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)

    model: Any = None
    tokenizer: Any = None
    torch: Any = None
    device = ""
    manifest_device = ""
    resolved_dtype = ""
    api_key = ""
    if args.backend == "transformers":
        import torch as torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_module
        tokenizer_id = args.tokenizer_id or args.model_id
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model_dtype = _resolve_dtype(torch, args.dtype)
        model_kwargs: dict[str, Any] = {
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
            "torch_dtype": model_dtype,
            "low_cpu_mem_usage": True,
        }
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        if args.device_map:
            device = str(next(model.parameters()).device)
            manifest_device = f"device_map:{args.device_map}; input_device:{device}"
        else:
            device = _resolve_device(torch, args.device)
            model = model.to(device)
            manifest_device = device
        model.eval()
        resolved_dtype = str(model_dtype).replace("torch.", "")
    elif args.backend == "openrouter":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise SystemExit(f"Set {args.api_key_env} before running --backend openrouter.")
        manifest_device = "openrouter"
        resolved_dtype = "remote"
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    manifest_path = output.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                args,
                output=output,
                run_id=run_id,
                prompt_count=len(behavior_rows),
                device=manifest_device,
                dtype=resolved_dtype,
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
        for index, row in enumerate(behavior_rows, start=1):
            prompt_id = row["prompt_id"]
            if prompt_id in existing_ids:
                continue
            prompt_row = prompt_rows[prompt_id]
            generation_prompt = _behavior_generation_prompt(
                prompt_row["prompt_text"],
                answer_contract=args.answer_contract,
            )
            if args.backend == "transformers":
                response_text = _generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    torch=torch,
                    device=device,
                    prompt=generation_prompt,
                    prompt_format=args.prompt_format,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                )
            else:
                response_text = _openrouter_generate_text(
                    model_id=args.model_id,
                    prompt=generation_prompt,
                    prompt_format=args.prompt_format,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    api_key=api_key,
                    timeout=args.request_timeout,
                    retries=args.request_retries,
                    reasoning_effort=args.openrouter_reasoning_effort,
                )
            amount, risk, valid_amount, valid_risk, refusal, error_type = parse_response(response_text)
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
                    "projection": row.get("projection", ""),
                    "token_count": row.get("token_count", ""),
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "prompt_format": args.prompt_format,
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
            handle.flush()
            written += 1
            print(f"wrote behavior {index}/{len(behavior_rows)}: {prompt_id}", flush=True)

    summary = {
        "run_id": run_id,
        "output": str(output),
        "manifest": str(manifest_path),
        "model_id": args.model_id,
        "prompt_count": len(behavior_rows),
        "written": written,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
