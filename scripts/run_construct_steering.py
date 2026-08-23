#!/usr/bin/env python3
"""Execute a frozen construct steering plan on a local or RunPod model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.steering import ResidualSteeringGenerator, SteeringConfig  # noqa: E402
from construct_benchmark.manifests import file_sha256  # noqa: E402
from construct_benchmark.prompts import load_prompt_records  # noqa: E402


def _direction_path(plan: dict, condition: dict) -> Path:
    paths = plan["direction_paths"]
    kind = condition["direction_kind"]
    if kind == "target":
        return Path(paths["target"])
    if kind == "shuffled":
        return Path(paths["shuffled"])
    if kind == "random":
        return Path(paths["random"][int(condition["direction_index"])])
    raise ValueError(f"Unsupported direction_kind={kind!r}.")


def _completed_condition_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} has invalid JSON on line {line_number}.") from exc
        completed.add(str(row["condition_id"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen residual-stream steering condition matrix.")
    parser.add_argument("--steering-plan", type=Path, required=True)
    parser.add_argument("--prompt-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--block-path", default=None)
    parser.add_argument("--prompt-format", choices=("completion", "chat"), default="chat")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    if args.output.exists() and not args.resume:
        raise SystemExit(f"{args.output} already exists; use --resume or choose a new output path.")
    plan = json.loads(args.steering_plan.read_text(encoding="utf-8"))
    steering_plan_sha256 = file_sha256(args.steering_plan)
    prompt_inventory_sha256 = file_sha256(args.prompt_inventory)
    if prompt_inventory_sha256 != plan["provenance"]["prompt_inventory_sha256"]:
        raise SystemExit("Prompt inventory hash does not match the frozen steering plan.")
    prompts = {
        record.prompt_id: record
        for record in load_prompt_records(args.prompt_inventory)
        if record.construct_id == plan["construct_id"] and record.split == "steering_eval"
    }
    missing_prompts = sorted({condition["prompt_id"] for condition in plan["conditions"]} - set(prompts))
    if missing_prompts:
        raise SystemExit(f"Steering plan references missing prompt IDs: {missing_prompts[:5]}")
    for direction_path in {
        _direction_path(plan, condition) for condition in plan["conditions"]
    }:
        if not direction_path.is_file():
            raise SystemExit(f"Missing direction artifact: {direction_path}")
    if file_sha256(Path(plan["direction_paths"]["target"])) != plan["provenance"]["direction_sha256"]:
        raise SystemExit("Target direction hash does not match the frozen steering plan.")
    if (
        file_sha256(Path(plan["direction_paths"]["shuffled"]))
        != plan["provenance"]["control_direction_hashes"]["shuffled"]
    ):
        raise SystemExit("Shuffled direction hash does not match the frozen steering plan.")
    for path, expected_hash in zip(
        plan["direction_paths"]["random"],
        plan["provenance"]["control_direction_hashes"]["random"],
        strict=True,
    ):
        if file_sha256(Path(path)) != expected_hash:
            raise SystemExit(f"Random direction hash does not match the frozen steering plan: {path}")

    model = plan["model"]
    if model["model_id"] == "REPLACE_WITH_LOCAL_MODEL":
        raise SystemExit("Replace the placeholder model_id in the run configuration before execution.")
    generator = ResidualSteeringGenerator(
        model["model_id"],
        model.get("tokenizer_id"),
        revision=model.get("revision"),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        block_path=args.block_path,
    )
    completed = _completed_condition_ids(args.output) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for condition in plan["conditions"]:
            if condition["condition_id"] in completed:
                continue
            prompt = prompts[condition["prompt_id"]]
            config = SteeringConfig(
                direction_path=_direction_path(plan, condition),
                layer=int(plan["layer"]),
                scale=float(condition["physical_scale"]),
                position_mode=str(plan["position_mode"]),
                intervention_timing=str(condition["intervention_timing"]),
                fixed_window_start=(plan.get("fixed_window") or [None, None])[0],
                fixed_window_end=(plan.get("fixed_window") or [None, None])[1],
            )
            output_text, direction_info = generator.generate(
                prompt.prompt_text,
                prompt_format=args.prompt_format,
                system_prompt=args.system_prompt,
                steering_config=config,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                max_length=args.max_length,
                do_sample=False,
            )
            row = {
                **condition,
                "construct_id": plan["construct_id"],
                "prompt_inventory_sha256": prompt_inventory_sha256,
                "steering_plan_sha256": steering_plan_sha256,
                "parser_id": prompt.parser_id,
                "expected_output_format": prompt.expected_output_format,
                "task_metadata": prompt.metadata.get("task_metadata", {}),
                "output_text": output_text,
                "direction": None if direction_info is None else direction_info.__dict__,
            }
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            written += 1
    print(json.dumps({"output": str(args.output), "new_rows": written, "total_conditions": len(plan["conditions"])}))


if __name__ == "__main__":
    main()
