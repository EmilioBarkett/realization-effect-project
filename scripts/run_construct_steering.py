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


def _tracking_directions(plan: dict) -> dict[int, dict]:
    raw = plan.get("tracking_directions")
    if isinstance(raw, dict) and raw:
        result = {int(layer): dict(value) for layer, value in raw.items()}
        for layer, value in result.items():
            if int(value.get("layer", layer)) != layer:
                raise ValueError(f"Tracking direction layer key {layer} does not match its declaration.")
            if not value.get("path") or not value.get("direction_id"):
                raise ValueError(f"Tracking direction layer {layer} is missing path or direction_id.")
        return dict(sorted(result.items()))
    layer = int(plan["layer"])
    return {
        layer: {
            "layer": layer,
            "direction_id": f"injected_direction__layer_{layer:02d}",
            "path": str(plan["direction_paths"]["target"]),
            "source": "injection_direction_train_only",
            "role": "injection_immediate",
        }
    }


def _validate_direction_artifacts(plan: dict, tracking: dict[int, dict]) -> None:
    paths = {_direction_path(plan, condition) for condition in plan["conditions"]}
    paths.update(Path(entry["path"]) for entry in tracking.values())
    for direction_path in paths:
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
    expected_tracking_hashes = plan["provenance"].get("tracking_direction_hashes", {})
    for layer, entry in tracking.items():
        expected_hash = expected_tracking_hashes.get(str(layer), entry.get("direction_sha256"))
        if expected_hash and file_sha256(Path(entry["path"])) != expected_hash:
            raise SystemExit(f"Tracking direction hash does not match the frozen steering plan: layer {layer}")


def _read_existing_rows(
    path: Path,
    *,
    plan_sha256: str,
    prompt_inventory_sha256: str,
    plan: dict,
    dtype: str,
) -> tuple[list[dict], set[tuple[str, int]]]:
    if not path.exists():
        return [], set()
    rows: list[dict] = []
    completed: set[tuple[str, int]] = set()
    expected_model = plan["model"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} has invalid JSON on line {line_number}.") from exc
        for field, expected in (
            ("steering_plan_sha256", plan_sha256),
            ("prompt_inventory_sha256", prompt_inventory_sha256),
            ("construct_id", plan["construct_id"]),
            ("injection_layer", int(plan["layer"])),
            ("position_mode", str(plan["position_mode"])),
            ("intervention_timing", str(plan["intervention_timing"])),
            ("activation_site", str(plan.get("activation_site", ""))),
            ("dtype", dtype),
        ):
            if row.get(field) != expected:
                raise ValueError(
                    f"{path} is incompatible with the frozen steering run: {field}={row.get(field)!r}, "
                    f"expected {expected!r}."
                )
        if row.get("model") != expected_model:
            raise ValueError(f"{path} is incompatible with the frozen steering run: model provenance differs.")
        condition_id = str(row["condition_id"])
        tracking_layer = row.get("tracking_layer")
        if tracking_layer is None:
            raise ValueError(f"{path} line {line_number} has no tracking_layer; do not mix legacy outputs.")
        key = (condition_id, int(tracking_layer))
        if key in completed:
            raise ValueError(f"{path} contains duplicate record identity {key!r}.")
        completed.add(key)
        rows.append(row)
    return rows, completed


def _output_manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _select_zero_dose_plan(plan: dict) -> dict:
    """Derive the cheap target zero-dose subset without changing the source plan."""

    conditions = [
        dict(condition)
        for condition in plan.get("conditions", [])
        if condition.get("direction_kind") == "target"
        and float(condition.get("dose", float("nan"))) == 0.0
    ]
    if not conditions:
        raise ValueError("Steering plan contains no target zero-dose conditions.")
    selected = dict(plan)
    selected["conditions"] = conditions
    selected["execution_scope"] = "target_zero_dose_behavior_gate"
    selected["source_condition_count"] = len(plan.get("conditions", []))
    selected["selected_condition_count"] = len(conditions)
    return selected


def _build_output_manifest(
    *,
    plan: dict,
    plan_sha256: str,
    prompt_inventory_sha256: str,
    output: Path,
    tracking: dict[int, dict],
    args,
) -> dict:
    return {
        "schema_version": plan.get("schema_version"),
        "manifest_type": "construct_steering_output",
        "run_id": plan.get("run_id"),
        "construct_id": plan["construct_id"],
        "output": str(output),
        "steering_plan_sha256": plan_sha256,
        "prompt_inventory_sha256": prompt_inventory_sha256,
        "run_config_hash": plan.get("provenance", {}).get("run_config_hash"),
        "construct_spec_hash": plan.get("provenance", {}).get("construct_spec_hash"),
        "model": dict(plan["model"]),
        "injection_layer": int(plan["layer"]),
        "tracking_layers": list(tracking),
        "tracking_directions": {str(layer): dict(entry) for layer, entry in tracking.items()},
        "expected_condition_ids": [str(condition["condition_id"]) for condition in plan["conditions"]],
        "expected_record_ids": [
            f"{condition['condition_id']}__tracking_layer_{tracking_layer:02d}"
            for condition in plan["conditions"]
            for tracking_layer in tracking
        ],
        "activation_site": str(plan.get("activation_site", "")),
        "position_mode": str(plan["position_mode"]),
        "intervention_timing": str(plan["intervention_timing"]),
        "prompt_format": args.prompt_format,
        "system_prompt_sha256": file_sha256_from_text(args.system_prompt),
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "block_path": args.block_path,
        "execution_scope": str(plan.get("execution_scope", "full_condition_matrix")),
        "source_condition_count": int(plan.get("source_condition_count", len(plan["conditions"]))),
        "selected_condition_count": int(plan.get("selected_condition_count", len(plan["conditions"]))),
        "confirmatory": bool(plan.get("confirmatory", False)),
        "expected_record_count": len(plan["conditions"]) * len(tracking),
        "completed_record_count": 0,
        "complete": False,
    }


def file_sha256_from_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_output_manifest(path: Path, expected: dict) -> None:
    if not path.is_file():
        raise ValueError(
            f"Cannot resume {expected['output']} without its output manifest {path}. "
            "Use a new output path rather than mixing untracked rows."
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON.") from exc
    fields = (
        "manifest_type",
        "run_id",
        "construct_id",
        "steering_plan_sha256",
        "prompt_inventory_sha256",
        "run_config_hash",
        "construct_spec_hash",
        "model",
        "injection_layer",
        "tracking_layers",
        "tracking_directions",
        "expected_condition_ids",
        "expected_record_ids",
        "activation_site",
        "position_mode",
        "intervention_timing",
        "prompt_format",
        "system_prompt_sha256",
        "max_new_tokens",
        "min_new_tokens",
        "max_length",
        "dtype",
        "device",
        "device_map",
        "block_path",
        "execution_scope",
        "source_condition_count",
        "selected_condition_count",
        "confirmatory",
        "expected_record_count",
    )
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ValueError(
                f"{path} is incompatible with the frozen steering run: {field} differs."
            )


def _prefill_projection(trace: dict, *, layer: int) -> dict | None:
    observations = [
        observation
        for observation in trace.get("projection_observations", [])
        if int(observation.get("layer", -1)) == layer
        and int(observation.get("forward_index", -1)) == 0
    ]
    return observations[0] if observations else None


def _prefill_injection(trace: dict) -> dict | None:
    observations = [
        observation
        for observation in trace.get("injection_observations", [])
        if int(observation.get("forward_index", -1)) == 0
    ]
    return observations[0] if observations else None


def _trace_rows(
    *,
    condition: dict,
    prompt,
    output_text: str,
    trace: dict,
    tracking: dict[int, dict],
    plan: dict,
    plan_sha256: str,
    prompt_inventory_sha256: str,
    model: dict,
    dtype: str,
    device: str,
    resolved_block_path: str,
) -> list[dict]:
    injection = _prefill_injection(trace)
    if injection is None:
        raise RuntimeError(f"Condition {condition['condition_id']} produced no prefill injection observation.")
    rows = []
    for tracking_layer, tracking_entry in tracking.items():
        projection = _prefill_projection(trace, layer=tracking_layer)
        if projection is None:
            raise RuntimeError(
                f"Condition {condition['condition_id']} produced no prefill projection for layer {tracking_layer}."
            )
        is_injection_layer = tracking_layer == int(plan["layer"])
        observed_tracking_direction_id = (
            trace["direction_id"] if is_injection_layer else tracking_entry["direction_id"]
        )
        observed_tracking_source = (
            trace["direction_source"] if is_injection_layer else tracking_entry.get("source", "")
        )
        observed_tracking_role = (
            trace["direction_role"] if is_injection_layer else tracking_entry.get("role", "")
        )
        observed_tracking_path = (
            (trace.get("injection_direction") or {}).get("path", tracking_entry["path"])
            if is_injection_layer
            else tracking_entry["path"]
        )
        tracking_calibration = (
            {"projection_scale": trace.get("calibration_projection_scale")}
            if is_injection_layer
            else tracking_entry.get("calibration")
        )
        tracking_calibration_scale = (
            tracking_calibration.get("projection_scale")
            if isinstance(tracking_calibration, dict)
            else None
        )
        row = {
            **condition,
            "record_id": f"{condition['condition_id']}__tracking_layer_{tracking_layer:02d}",
            "construct_id": plan["construct_id"],
            "prompt_inventory_sha256": prompt_inventory_sha256,
            "steering_plan_sha256": plan_sha256,
            "run_config_hash": plan.get("provenance", {}).get("run_config_hash"),
            "construct_spec_hash": plan.get("provenance", {}).get("construct_spec_hash"),
            "model": dict(model),
            "model_revision": model.get("revision"),
            "activation_site": str(plan.get("activation_site", "")),
            "injection_layer": int(plan["layer"]),
            "tracking_layer": tracking_layer,
            "tracking_direction_id": observed_tracking_direction_id,
            "tracking_direction_source": observed_tracking_source,
            "tracking_role": observed_tracking_role,
            "tracking_direction_path": observed_tracking_path,
            "direction_id": trace["direction_id"],
            "direction_source": trace["direction_source"],
            "direction_role": trace["direction_role"],
            "position_mode": trace["position_mode"],
            "intervention_timing": trace["intervention_timing"],
            "phase": projection["phase"],
            "token_position": projection["token_position"],
            "prefill_forward_index": projection["forward_index"],
            "injection_applied": projection["injection_applied"],
            "pre_projection": injection["pre_projection"] if is_injection_layer else None,
            "post_projection": injection["post_projection"] if is_injection_layer else None,
            "observed_shift": injection["observed_shift"] if is_injection_layer else None,
            "expected_shift": injection["expected_shift"] if is_injection_layer else None,
            "expected_observed_difference": (
                injection["expected_observed_difference"] if is_injection_layer else None
            ),
            "calibrated_projection_scale": trace["calibration_projection_scale"],
            "injection_calibration_projection_scale": trace["calibration_projection_scale"],
            "tracking_calibration": tracking_calibration,
            "tracking_calibration_projection_scale": tracking_calibration_scale,
            "projection": projection["projection"],
            "downstream_projection": projection["projection"] if not is_injection_layer else None,
            "output_text": output_text,
            "parser_id": prompt.parser_id,
            "expected_output_format": prompt.expected_output_format,
            "task_metadata": prompt.metadata.get("task_metadata", {}),
            "direction": trace.get("injection_direction"),
            "trace": trace,
            "dtype": dtype,
            "device": device,
            "resolved_block_path": resolved_block_path,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen residual-stream steering condition matrix.")
    parser.add_argument("--steering-plan", type=Path, required=True)
    parser.add_argument("--prompt-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--zero-dose-only",
        action="store_true",
        help="Run only target-direction zero-dose conditions for the pre-full-run behavior gate.",
    )
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
    if args.zero_dose_only:
        plan = _select_zero_dose_plan(plan)
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
    tracking = _tracking_directions(plan)
    declared_tracking_layers = plan.get("tracking_layers")
    if declared_tracking_layers is not None and sorted(int(layer) for layer in declared_tracking_layers) != list(tracking):
        raise SystemExit("Steering plan tracking_layers does not agree with tracking_directions.")
    _validate_direction_artifacts(plan, tracking)
    model = plan["model"]
    if model["model_id"] == "REPLACE_WITH_LOCAL_MODEL":
        raise SystemExit("Replace the placeholder model_id in the run configuration before execution.")

    output_manifest = _build_output_manifest(
        plan=plan,
        plan_sha256=steering_plan_sha256,
        prompt_inventory_sha256=prompt_inventory_sha256,
        output=args.output,
        tracking=tracking,
        args=args,
    )
    output_manifest_path = _output_manifest_path(args.output)
    if args.resume:
        _validate_output_manifest(output_manifest_path, output_manifest)
    else:
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        output_manifest_path.write_text(
            json.dumps(output_manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    _, completed = (
        _read_existing_rows(
            args.output,
            plan_sha256=steering_plan_sha256,
            prompt_inventory_sha256=prompt_inventory_sha256,
            plan=plan,
            dtype=args.dtype,
        )
        if args.resume
        else ([], set())
    )
    expected_keys = {
        (str(condition["condition_id"]), tracking_layer)
        for condition in plan["conditions"]
        for tracking_layer in tracking
    }
    unexpected_keys = completed - expected_keys
    if unexpected_keys:
        raise ValueError(
            f"{args.output} contains records not present in the frozen steering plan: "
            f"{sorted(unexpected_keys)[:3]}"
        )
    if args.resume and completed == expected_keys:
        output_manifest["completed_record_count"] = len(completed)
        output_manifest["complete"] = True
        output_manifest["raw_generations_sha256"] = file_sha256(args.output)
        output_manifest_path.write_text(
            json.dumps(output_manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "new_rows": 0,
                    "completed_records": len(completed),
                    "total_records": len(expected_keys),
                }
            )
        )
        return
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for condition in plan["conditions"]:
            condition_keys = {
                (str(condition["condition_id"]), tracking_layer) for tracking_layer in tracking
            }
            if condition_keys.issubset(completed):
                continue
            prompt = prompts[condition["prompt_id"]]
            direction_kind = str(condition["direction_kind"])
            direction_source = {
                "target": "injection_direction_train_only",
                "shuffled": "shuffled_label_control",
                "random": "orthogonal_random_control",
            }.get(direction_kind, f"{direction_kind}_control")
            calibration = plan.get("calibration", {})
            config = SteeringConfig(
                direction_path=_direction_path(plan, condition),
                layer=int(plan["layer"]),
                scale=float(condition["physical_scale"]),
                position_mode=str(plan["position_mode"]),
                intervention_timing=str(condition["intervention_timing"]),
                fixed_window_start=(plan.get("fixed_window") or [None, None])[0],
                fixed_window_end=(plan.get("fixed_window") or [None, None])[1],
                direction_id=(
                    f"{plan['construct_id']}__{direction_kind}_{int(condition['direction_index']):02d}"
                    f"__layer_{int(plan['layer']):02d}"
                ),
                direction_source=direction_source,
                direction_role="injection_immediate",
                requested_dose=float(condition["dose"]),
                calibration_projection_scale=(
                    float(calibration["projection_scale"]) if calibration.get("projection_scale") is not None else None
                ),
            )
            output_text, _direction_info, trace = generator.generate(
                prompt.prompt_text,
                prompt_format=args.prompt_format,
                system_prompt=args.system_prompt,
                steering_config=config,
                tracking_directions=tracking,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                max_length=args.max_length,
                do_sample=False,
                return_trace=True,
            )
            if trace is None:
                raise RuntimeError(f"Condition {condition['condition_id']} returned no steering trace.")
            rows = _trace_rows(
                condition=condition,
                prompt=prompt,
                output_text=output_text,
                trace=trace.to_mapping(),
                tracking=tracking,
                plan=plan,
                plan_sha256=steering_plan_sha256,
                prompt_inventory_sha256=prompt_inventory_sha256,
                model=model,
                dtype=args.dtype,
                device=generator.resolved_device,
                resolved_block_path=generator.resolved_block_path or args.block_path or "",
            )
            for row in rows:
                key = (str(row["condition_id"]), int(row["tracking_layer"]))
                if key in completed:
                    continue
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                completed.add(key)
                written += 1
    output_manifest["completed_record_count"] = len(completed)
    output_manifest["complete"] = completed == expected_keys
    if args.output.is_file():
        output_manifest["raw_generations_sha256"] = file_sha256(args.output)
    output_manifest_path.write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "new_rows": written,
                "completed_records": len(completed),
                "total_records": len(expected_keys),
                "total_conditions": len(plan["conditions"]),
                "tracking_layers": list(tracking),
            }
        )
    )


if __name__ == "__main__":
    main()
