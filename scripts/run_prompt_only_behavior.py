#!/usr/bin/env python3
"""Run independent downstream behavior prompts without activation steering.

This is the cheap model-side baseline used before direction construction and
the full steering battery.  It loads one model per process, writes one
manifest-backed JSONL row per selected behavior prompt, and can resume only
against the same frozen inventory and run configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.steering import ResidualSteeringGenerator  # noqa: E402
from construct_benchmark.behavior_baseline import (  # noqa: E402
    build_behavior_output_manifest,
    output_manifest_path,
    read_behavior_output,
    select_behavior_records,
    select_preflight_behavior_records,
    validate_behavior_output_manifest,
)
from construct_benchmark.config import (  # noqa: E402
    load_construct_specs,
    load_run_config,
    validate_run_constructs,
)
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402
from construct_benchmark.prompts import load_prompt_records, validate_prompt_records  # noqa: E402
from construct_benchmark.run_modes import resolve_run_mode  # noqa: E402


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _existing_rows(
    output: Path,
    *,
    run_config,
    construct_specs,
    allow_incomplete_diagnostic: bool,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any] | None]:
    if not output.exists():
        return [], set(), None
    rows = read_behavior_output(output)
    manifest, _complete = validate_behavior_output_manifest(
        output,
        rows,
        run_config=run_config,
        construct_specs=construct_specs,
        allow_incomplete_diagnostic=allow_incomplete_diagnostic,
    )
    return rows, {str(row["prompt_id"]) for row in rows}, manifest


def execute_prompt_only_behavior(
    *,
    run_config,
    construct_specs: Mapping[str, Any],
    prompt_inventory: Path,
    output: Path,
    mode: str | None = None,
    prompt_format: str = "chat",
    system_prompt: str = "",
    enable_thinking: bool | None = None,
    max_new_tokens: int = 32,
    min_new_tokens: int = 1,
    max_length: int | None = None,
    device: str = "auto",
    dtype: str = "auto",
    device_map: str | None = None,
    block_path: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    max_runtime_minutes: float | None = None,
    resume: bool = False,
    prompt_split: str = "behavior_eval",
    constrained_numeric_generation: bool = True,
    preflight_selection: Mapping[str, Any] | None = None,
    generator_factory: Callable[..., Any] = ResidualSteeringGenerator,
) -> dict[str, Any]:
    """Execute or resume a prompt-only behavior run."""

    validate_run_constructs(run_config, dict(construct_specs))
    records = load_prompt_records(prompt_inventory)
    validate_prompt_records(records, construct_specs)
    effective_mode = "test" if preflight_selection is not None and mode is None else mode
    if preflight_selection is None:
        selected, selection_manifest = select_behavior_records(
            records,
            run_config=run_config,
            construct_specs=construct_specs,
            split=prompt_split,
            mode=effective_mode,
        )
    else:
        selected, selection_manifest = select_preflight_behavior_records(
            records,
            run_config=run_config,
            construct_specs=construct_specs,
            preflight_selection=preflight_selection,
            split=prompt_split,
        )
    resolved_mode, mode_config = resolve_run_mode(run_config, effective_mode)
    resolved_max_length = int(max_length or run_config.activation["max_length"])
    if resolved_max_length < 1:
        raise ValueError("max_length must be positive.")
    runtime_limit = max_runtime_minutes
    if runtime_limit is None and mode_config["max_runtime_minutes"] is not None:
        runtime_limit = float(mode_config["max_runtime_minutes"])
    if runtime_limit is not None and runtime_limit <= 0:
        raise ValueError("max_runtime_minutes must be positive when provided.")

    output = output.resolve()
    manifest = build_behavior_output_manifest(
        run_config=run_config,
        construct_specs=construct_specs,
        output=output,
        prompt_inventory_sha256=file_sha256(prompt_inventory),
        selection_manifest=selection_manifest,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
        system_prompt_sha256=_sha256_text(system_prompt),
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        max_length=resolved_max_length,
        dtype=dtype,
        device=device,
        device_map=device_map,
        block_path=block_path,
    )
    manifest.update(
        {
            "mode": resolved_mode,
            "purpose": mode_config["purpose"],
            "confirmatory": bool(mode_config["confirmatory"]),
            "max_runtime_minutes": runtime_limit,
            "run_config_hash": canonical_hash(run_config.to_mapping()),
            "constrained_numeric_generation": bool(constrained_numeric_generation),
        }
    )
    manifest_path = output_manifest_path(output)

    if output.exists() and not resume:
        raise FileExistsError(f"{output} already exists; use --resume or choose a new output path.")
    existing_rows: list[dict[str, Any]] = []
    completed_prompt_ids: set[str] = set()
    if resume:
        if not output.exists():
            raise FileNotFoundError(f"Cannot resume missing behavior output: {output}")
        existing_rows, completed_prompt_ids, existing_manifest = _existing_rows(
            output,
            run_config=run_config,
            construct_specs=construct_specs,
            allow_incomplete_diagnostic=True,
        )
        if existing_manifest is None:
            raise ValueError("Resume did not produce an output manifest.")
        expected_fields = (
            "selection",
            "prompt_inventory_sha256",
            "mode",
            "purpose",
            "confirmatory",
            "max_runtime_minutes",
            "prompt_format",
            "enable_thinking",
            "system_prompt_sha256",
            "max_new_tokens",
            "min_new_tokens",
            "max_length",
            "dtype",
            "device",
            "device_map",
            "block_path",
            "constrained_numeric_generation",
        )
        for field in expected_fields:
            if existing_manifest.get(field) != manifest.get(field):
                raise ValueError(f"Existing behavior manifest differs in {field}; use a new output path.")
        manifest = existing_manifest
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    expected_prompt_ids = set(str(value) for value in manifest["expected_prompt_ids"])
    unexpected = completed_prompt_ids - expected_prompt_ids
    if unexpected:
        raise ValueError(f"Existing behavior output contains prompts outside the frozen selection: {sorted(unexpected)[:3]}")
    if completed_prompt_ids == expected_prompt_ids:
        manifest["completed_record_count"] = len(completed_prompt_ids)
        manifest["complete"] = True
        manifest["raw_generations_sha256"] = file_sha256(output)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {
            "output": str(output),
            "new_rows": 0,
            "completed_records": len(completed_prompt_ids),
            "total_records": len(expected_prompt_ids),
            "complete": True,
        }

    generator = generator_factory(
        run_config.model["model_id"],
        run_config.model.get("tokenizer_id"),
        revision=run_config.model.get("revision"),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device=device,
        dtype=dtype,
        device_map=device_map,
        block_path=block_path,
    )
    selected_by_id = {record.prompt_id: record for record in selected}
    started = time.monotonic()
    stopped_by_runtime = False
    written = 0
    mode_flag = "a" if resume else "w"
    with output.open(mode_flag, encoding="utf-8") as handle:
        for prompt_id in manifest["expected_prompt_ids"]:
            prompt_id = str(prompt_id)
            if prompt_id in completed_prompt_ids:
                continue
            if runtime_limit is not None and time.monotonic() - started >= runtime_limit * 60:
                stopped_by_runtime = True
                break
            record = selected_by_id[prompt_id]
            output_text, _info = generator.generate(
                record.prompt_text,
                prompt_format=prompt_format,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                parser_id=record.parser_id,
                constrained_numeric=constrained_numeric_generation,
                steering_config=None,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                max_length=resolved_max_length,
                do_sample=False,
            )
            row = {
                "record_id": f"{record.prompt_id}__prompt_only",
                "prompt_id": record.prompt_id,
                "construct_id": record.construct_id,
                "split": record.split,
                "prompt_role": record.prompt_role,
                "prompt_family": record.prompt_family,
                "condition_id": record.condition_id,
                "intervention": "none",
                "direction_kind": "none",
                "dose": 0.0,
                "physical_scale": 0.0,
                "task_id": record.task_id,
                "parser_id": record.parser_id,
                "expected_output_format": record.expected_output_format,
                "task_metadata": record.metadata.get("task_metadata", {}),
                "output_text": output_text,
                "model": dict(run_config.model),
                "model_revision": run_config.model.get("revision"),
                "prompt_inventory_sha256": manifest["prompt_inventory_sha256"],
                "run_config_hash": manifest["run_config_hash"],
                "construct_spec_hash": manifest["construct_spec_hashes"][record.construct_id],
                "prompt_format": prompt_format,
                "system_prompt_sha256": manifest["system_prompt_sha256"],
                "max_length": resolved_max_length,
                "dtype": dtype,
                "device": getattr(generator, "resolved_device", device),
                "tokenization": getattr(generator, "last_tokenization_report", None),
            }
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            completed_prompt_ids.add(prompt_id)
            written += 1

    complete = completed_prompt_ids == expected_prompt_ids
    manifest["completed_record_count"] = len(completed_prompt_ids)
    manifest["complete"] = complete
    manifest["stopped_by_runtime"] = stopped_by_runtime or not complete
    manifest["runtime_seconds"] = round(time.monotonic() - started, 3)
    if output.is_file():
        manifest["raw_generations_sha256"] = file_sha256(output)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "new_rows": written,
        "completed_records": len(completed_prompt_ids),
        "total_records": len(expected_prompt_ids),
        "complete": complete,
        "stopped_by_runtime": manifest["stopped_by_runtime"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the independent prompt-only behavior baseline.")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, required=True)
    parser.add_argument("--prompt-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preflight-selection-manifest",
        type=Path,
        default=None,
        help="Use the exact outcome-independent IDs from a v2 preflight selection.",
    )
    parser.add_argument("--mode", choices=("test", "full"), default=None)
    parser.add_argument(
        "--split",
        dest="prompt_split",
        choices=("behavior_eval", "steering_eval", "calibration", "collateral_eval"),
        default="behavior_eval",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--block-path", default=None)
    parser.add_argument("--prompt-format", choices=("completion", "chat"), default="chat")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument(
        "--disable-thinking",
        action="store_false",
        dest="enable_thinking",
        default=None,
        help="Request a text-only response without hidden reasoning when the chat template supports it.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-runtime-minutes", type=float, default=None)
    parser.add_argument(
        "--disable-constrained-numeric-generation",
        action="store_false",
        dest="constrained_numeric_generation",
        default=True,
        help="Disable the shared tokenizer-aware numeric constraint (diagnostic use only).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_new_tokens < 1 or args.min_new_tokens < 1 or args.min_new_tokens > args.max_new_tokens:
        raise SystemExit("Require 1 <= --min-new-tokens <= --max-new-tokens.")
    run_config = load_run_config(args.run_config)
    construct_specs = load_construct_specs(args.construct_spec)
    preflight_selection = None
    if args.preflight_selection_manifest is not None:
        try:
            preflight_selection = json.loads(
                args.preflight_selection_manifest.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"{args.preflight_selection_manifest} is not valid JSON."
            ) from exc
        if not isinstance(preflight_selection, dict):
            raise SystemExit("--preflight-selection-manifest must contain a JSON object.")
    result = execute_prompt_only_behavior(
        run_config=run_config,
        construct_specs=construct_specs,
        prompt_inventory=args.prompt_inventory,
        output=args.output,
        mode=args.mode,
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        block_path=args.block_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        max_runtime_minutes=args.max_runtime_minutes,
        resume=args.resume,
        prompt_split=args.prompt_split,
        constrained_numeric_generation=args.constrained_numeric_generation,
        preflight_selection=preflight_selection,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
