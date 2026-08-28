#!/usr/bin/env python3
"""Measure frozen prompt lengths with the exact model tokenizer.

The command loads tokenizer files only; it does not load model weights or make
generation calls.  It writes a report even when prompts exceed the configured
limit, then exits nonzero so a model-side run cannot silently truncate them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.tokenization import (  # noqa: E402
    format_model_prompt,
    inspect_token_lengths,
)
from activation_analysis.model_loading import load_tokenizer_or_processor  # noqa: E402
from construct_benchmark.config import load_construct_specs, load_run_config, validate_run_constructs  # noqa: E402
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402
from construct_benchmark.prompts import load_prompt_records, validate_prompt_records  # noqa: E402


def _load_tokenizer(run_config, *, local_files_only: bool, trust_remote_code: bool):
    try:
        import transformers
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Tokenizer preflight requires transformers. Install the optional interp dependencies first."
        ) from exc
    tokenizer_id = run_config.model.get("tokenizer_id") or run_config.model["model_id"]
    tokenizer, loader_name = load_tokenizer_or_processor(
        transformers,
        str(tokenizer_id),
        revision=run_config.model.get("revision"),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer, str(tokenizer_id), loader_name


def _group_report(records, lengths: tuple[int, ...], max_length: int) -> dict[str, Any]:
    by_group: dict[tuple[str, str], list[int]] = {}
    for record, length in zip(records, lengths, strict=True):
        by_group.setdefault((record.construct_id, record.split), []).append(length)
    groups = {}
    for (construct_id, split), values in sorted(by_group.items()):
        groups[f"{construct_id}:{split}"] = {
            "count": len(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "over_limit_count": sum(value > max_length for value in values),
        }
    return groups


def build_tokenizer_preflight_report(
    *,
    run_config_path: Path,
    prompt_inventory_path: Path,
    construct_spec_paths: list[Path],
    output: Path,
    prompt_format: str = "completion",
    system_prompt: str = "",
    max_length: int | None = None,
    limit: int | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Build and write a tokenizer report without loading model weights."""

    run_config = load_run_config(run_config_path)
    construct_specs = load_construct_specs(construct_spec_paths)
    validate_run_constructs(run_config, construct_specs)
    records = load_prompt_records(prompt_inventory_path)
    validate_prompt_records(records, construct_specs)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when provided.")
        records = records[:limit]
    if not records:
        raise ValueError("No prompts selected for tokenizer preflight.")

    tokenizer, tokenizer_id, tokenizer_loader = _load_tokenizer(
        run_config,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    formatted_prompts = [
        format_model_prompt(
            tokenizer,
            record.prompt_text,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
        )
        for record in records
    ]
    resolved_max_length = int(max_length or run_config.activation["max_length"])
    token_report = inspect_token_lengths(
        tokenizer,
        formatted_prompts,
        max_length=resolved_max_length,
        prompt_ids=[record.prompt_id for record in records],
        tokenizer_id=tokenizer_id,
        revision=run_config.model.get("revision"),
    )
    payload = {
        "schema_version": run_config.schema_version,
        "manifest_type": "tokenizer_preflight",
        "run_config": str(run_config_path.resolve()),
        "run_config_hash": canonical_hash(run_config.to_mapping()),
        "prompt_inventory": str(prompt_inventory_path.resolve()),
        "prompt_inventory_sha256": file_sha256(prompt_inventory_path),
        "construct_ids": list(run_config.construct_ids),
        "prompt_format": prompt_format,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "selected_prompt_count": len(records),
        "tokenizer": {
            "tokenizer_id": tokenizer_id,
            "loader": tokenizer_loader,
            "revision": run_config.model.get("revision"),
        },
        **token_report.to_mapping(),
        "groups": _group_report(records, token_report.lengths, resolved_max_length),
    }
    payload["ready"] = not bool(token_report.over_limit_indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed tokenizer length preflight for a frozen inventory.")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-format", choices=("completion", "chat"), default="completion")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = build_tokenizer_preflight_report(
            run_config_path=args.run_config,
            prompt_inventory_path=args.prompts,
            construct_spec_paths=args.construct_spec,
            output=args.output,
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
            max_length=args.max_length,
            limit=args.limit,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["ready"]:
            raise SystemExit(1)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
