#!/usr/bin/env python3
"""Run manifest-backed matched-episode residual interchange.

This is the first causal-pathway runner for the benchmark.  It deliberately
does not replace the primary behavioral steering runner: it records baselines
and bidirectional residual-state swaps at a fixed probe-to-task boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.causal_patching import (  # noqa: E402
    CAUSAL_PATCH_SCHEMA_VERSION,
    INTERCHANGE_VARIANTS,
    MatchedEpisodeResidualPatcher,
    expected_observation_ids,
    load_matched_episode_jsonl,
)
from activation_analysis.steering import ResidualSteeringGenerator  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _parse_layers(value: str) -> list[int]:
    try:
        layers = sorted(set(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers or layers[0] < 1:
        raise argparse.ArgumentTypeError("layers must contain positive 1-based integers")
    return layers


def _load_existing_records(
    output: Path,
    *,
    expected_request_ids: set[str],
    expected_record_ids: set[str],
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    if not output.exists():
        return [], set(), set()
    rows: list[dict[str, Any]] = []
    completed_requests: set[str] = set()
    completed_records: set[str] = set()
    for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{output} has invalid JSON on line {line_number}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{output} line {line_number} must be a JSON object.")
        request_id = row.get("request", {}).get("request_id")
        if not isinstance(request_id, str) or request_id not in expected_request_ids:
            raise ValueError(f"{output} line {line_number} contains an unknown request_id.")
        if request_id in completed_requests:
            raise ValueError(f"{output} contains duplicate request_id={request_id!r}.")
        observations = row.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"{output} line {line_number} has no observations.")
        row_records: set[str] = set()
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError(f"{output} line {line_number} contains a malformed observation.")
            record_id = observation.get("record_id")
            if not isinstance(record_id, str) or record_id not in expected_record_ids:
                raise ValueError(f"{output} line {line_number} contains an unexpected record_id.")
            if record_id in completed_records or record_id in row_records:
                raise ValueError(f"{output} contains duplicate record_id={record_id!r}.")
            if observation.get("request_id") != request_id:
                raise ValueError(f"{output} record {record_id!r} has the wrong request_id.")
            row_records.add(record_id)
        rows.append(row)
        completed_requests.add(request_id)
        completed_records.update(row_records)
    return rows, completed_requests, completed_records


def _request_record_ids(
    request_id: str,
    *,
    layers: Sequence[int],
    include_same_condition_controls: bool,
) -> set[str]:
    return {
        record_id
        for record_id in expected_observation_ids(
            [request_id],
            layers,
            include_same_condition_controls=include_same_condition_controls,
        )
    }


def _build_manifest(
    *,
    output: Path,
    requests_path: Path,
    episodes: Sequence[Any],
    layers: Sequence[int],
    model: dict[str, Any],
    args: argparse.Namespace,
    resolved_block_path: str | None,
) -> dict[str, Any]:
    request_ids = [episode.request_id for episode in episodes]
    record_ids = expected_observation_ids(
        request_ids,
        layers,
        include_same_condition_controls=args.include_same_condition_controls,
    )
    return {
        "schema_version": CAUSAL_PATCH_SCHEMA_VERSION,
        "manifest_type": "residual_interchange_output",
        "analysis_scope": "engineering_causal_diagnosis",
        "confirmatory": False,
        "output": str(output),
        "requests_path": str(requests_path),
        "requests_sha256": _sha256_file(requests_path),
        "request_ids": request_ids,
        "layers": list(layers),
        "model": model,
        "block_path": resolved_block_path,
        "prompt_contract": {
            "boundary_mode": "last_induction_token",
            "intervention_timing": "prefill_only",
            "downstream_task_shared_between_conditions": True,
            "prompt_formats": sorted({episode.prompt_format for episode in episodes}),
        },
        "execution": {
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "intervention_variant": args.intervention_variant,
            "include_same_condition_controls": args.include_same_condition_controls,
            "device": args.device,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
        },
        "expected_record_ids": record_ids,
        "expected_request_count": len(request_ids),
        "expected_observation_count": len(record_ids),
        "completed_request_ids": [],
        "completed_request_count": 0,
        "completed_observation_count": 0,
        "complete": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "manifest_type",
            "requests_path",
            "requests_sha256",
            "request_ids",
            "layers",
            "model",
            "block_path",
            "prompt_contract",
            "execution",
            "expected_record_ids",
            "expected_request_count",
            "expected_observation_count",
        )
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_existing_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Cannot resume without adjacent manifest: {path}")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON.") from exc
    if not isinstance(actual, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    if _manifest_identity(actual) != _manifest_identity(expected):
        raise ValueError("Existing residual-interchange manifest is incompatible with the requested run.")
    return actual


def _refresh_manifest(
    manifest: dict[str, Any],
    *,
    output: Path,
    completed_request_ids: set[str],
    completed_record_ids: set[str],
) -> None:
    expected_requests = set(manifest["request_ids"])
    expected_records = set(manifest["expected_record_ids"])
    if not completed_request_ids <= expected_requests:
        raise ValueError("Completed request set contains an unexpected request.")
    if not completed_record_ids <= expected_records:
        raise ValueError("Completed record set contains an unexpected record.")
    manifest["completed_request_ids"] = [
        request_id for request_id in manifest["request_ids"] if request_id in completed_request_ids
    ]
    manifest["completed_request_count"] = len(completed_request_ids)
    manifest["completed_observation_count"] = len(completed_record_ids)
    manifest["complete"] = completed_request_ids == expected_requests and completed_record_ids == expected_records
    if manifest["complete"]:
        manifest["raw_output_sha256"] = _sha256_file(output)
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    else:
        manifest.pop("raw_output_sha256", None)
        manifest.pop("completed_at", None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--revision")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layers", type=_parse_layers)
    layer_group.add_argument("--all-layers", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--prompt-format", help="Require all input requests to use this format.")
    parser.add_argument("--system-prompt", default=None, help="Require all input requests to use this system prompt.")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--intervention-variant", choices=sorted(INTERCHANGE_VARIANTS), default="natural_state_replacement")
    parser.add_argument(
        "--include-same-condition-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record same-condition donor controls in addition to bidirectional cross-condition swaps.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device-map")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--block-path")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_length < 1 or args.max_new_tokens < 1 or args.min_new_tokens < 0:
        raise SystemExit("max_length and max_new_tokens must be positive; min_new_tokens must be non-negative")
    if args.min_new_tokens > args.max_new_tokens:
        raise SystemExit("min_new_tokens cannot exceed max_new_tokens")
    if args.do_sample and args.temperature <= 0:
        raise SystemExit("temperature must be positive when --do-sample is used")

    requests_path = args.requests.resolve()
    output = args.output.resolve()
    manifest_path = _manifest_path(output)
    episodes = load_matched_episode_jsonl(requests_path)
    if args.prompt_format is not None:
        mismatches = [episode.request_id for episode in episodes if episode.prompt_format != args.prompt_format]
        if mismatches:
            raise SystemExit(f"Input requests do not all use --prompt-format={args.prompt_format!r}: {mismatches[:3]}")
    if args.system_prompt is not None:
        mismatches = [episode.request_id for episode in episodes if episode.system_prompt != args.system_prompt]
        if mismatches:
            raise SystemExit("Input requests do not all use the requested --system-prompt.")

    loader = ResidualSteeringGenerator(
        args.model_id,
        args.tokenizer_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        block_path=args.block_path,
    )
    patcher = MatchedEpisodeResidualPatcher(
        loader.model,
        loader.tokenizer,
        device=loader.device,
        block_path=loader.block_path or loader.resolve_block_path(),
        model_id=loader.model_id,
        tokenizer_id=loader.tokenizer_id,
        revision=loader.revision,
    )
    blocks = patcher.resolve_transformer_blocks()
    if args.all_layers:
        layers = list(range(1, len(blocks) + 1))
    else:
        assert args.layers is not None
        layers = list(args.layers)
        if layers[-1] > len(blocks):
            raise SystemExit(f"Requested layer {layers[-1]}, but the model has {len(blocks)} blocks.")

    model_metadata = {
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "revision": args.revision,
    }
    expected_manifest = _build_manifest(
        output=output,
        requests_path=requests_path,
        episodes=episodes,
        layers=layers,
        model=model_metadata,
        args=args,
        resolved_block_path=patcher.resolved_block_path,
    )
    if output.exists() or manifest_path.exists():
        if not args.resume:
            raise SystemExit(
                f"Output or manifest already exists ({output}, {manifest_path}); use a new path or --resume."
            )
        manifest = _validate_existing_manifest(manifest_path, expected_manifest)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        manifest = expected_manifest
        _write_manifest(manifest_path, manifest)

    rows, completed_requests, completed_records = _load_existing_records(
        output,
        expected_request_ids=set(manifest["request_ids"]),
        expected_record_ids=set(manifest["expected_record_ids"]),
    )
    _refresh_manifest(
        manifest,
        output=output,
        completed_request_ids=completed_requests,
        completed_record_ids=completed_records,
    )
    _write_manifest(manifest_path, manifest)
    if manifest["complete"]:
        print(json.dumps({"status": "already_complete", "output": str(output)}))
        return 0

    episodes_by_id = {episode.request_id: episode for episode in episodes}
    try:
        with output.open("a", encoding="utf-8") as handle:
            for episode in episodes:
                if episode.request_id in completed_requests:
                    continue
                result = patcher.run_episode(
                    episode,
                    layers=layers,
                    max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    intervention_variant=args.intervention_variant,
                    include_same_condition_controls=args.include_same_condition_controls,
                )
                row_record_ids = {observation["record_id"] for observation in result["observations"]}
                expected_for_request = _request_record_ids(
                    episode.request_id,
                    layers=layers,
                    include_same_condition_controls=args.include_same_condition_controls,
                )
                if row_record_ids != expected_for_request:
                    raise RuntimeError(
                        f"Request {episode.request_id} produced an invalid observation identity set."
                    )
                handle.write(json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                rows.append(result)
                completed_requests.add(episode.request_id)
                completed_records.update(row_record_ids)
                _refresh_manifest(
                    manifest,
                    output=output,
                    completed_request_ids=completed_requests,
                    completed_record_ids=completed_records,
                )
                _write_manifest(manifest_path, manifest)
                print(
                    json.dumps(
                        {
                            "status": "completed_request",
                            "request_id": episode.request_id,
                            "completed_requests": len(completed_requests),
                            "expected_requests": len(episodes),
                        }
                    ),
                    flush=True,
                )
    finally:
        # A stopped process leaves a truthful incomplete manifest that can be
        # resumed after the model or pod is restarted.
        _refresh_manifest(
            manifest,
            output=output,
            completed_request_ids=completed_requests,
            completed_record_ids=completed_records,
        )
        _write_manifest(manifest_path, manifest)

    # Keep the local variable intentionally explicit: it makes a future
    # extension for per-request validation straightforward and documents that
    # all loaded request IDs were consumed or resumed.
    if set(episodes_by_id) != set(manifest["request_ids"]):
        raise RuntimeError("Loaded request IDs changed while executing the run.")
    print(
        json.dumps(
            {
                "status": "complete" if manifest["complete"] else "incomplete",
                "output": str(output),
                "manifest": str(manifest_path),
                "completed_requests": manifest["completed_request_count"],
                "completed_observations": manifest["completed_observation_count"],
            }
        )
    )
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
