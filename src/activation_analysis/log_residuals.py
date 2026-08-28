from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .emotion_probes import load_emotion_probe_records
from .residual_streams import BatchResiduals, ResidualStreamLogger, SUPPORTED_ACTIVATION_SITES


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt_text: str
    metadata: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _parse_layers(value: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer must be provided.")
    if any(layer < 1 for layer in layers):
        raise argparse.ArgumentTypeError("Layers are 1-based and must be >= 1.")
    return layers


def _resolve_requested_layers(
    requested_layers: list[int] | None,
    *,
    all_layers: bool,
    num_transformer_layers: int,
) -> list[int]:
    """Resolve the explicit benchmark layer list or the opt-in all-layer trace."""

    if num_transformer_layers < 1:
        raise ValueError("num_transformer_layers must be >= 1.")
    if all_layers:
        return list(range(1, num_transformer_layers + 1))
    if not requested_layers:
        raise ValueError("Provide requested layers unless all_layers is enabled.")
    if min(requested_layers) < 1 or max(requested_layers) > num_transformer_layers:
        raise ValueError(
            f"Requested layers must be between 1 and {num_transformer_layers}; "
            f"got {requested_layers}."
        )
    return list(requested_layers)


def _parse_token_region_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    regions = {part.strip() for part in value.split(",") if part.strip()}
    if not regions:
        raise argparse.ArgumentTypeError("At least one token region must be provided.")
    return regions


def _load_prompt_records(args: argparse.Namespace) -> list[PromptRecord]:
    if args.emotion_config:
        return [
            PromptRecord(
                prompt_id=record.prompt_id,
                prompt_text=record.prompt_text,
                metadata=record.metadata,
            )
            for record in load_emotion_probe_records(Path(args.emotion_config))
        ]

    if args.prompt_csv:
        return _load_prompt_csv(
            Path(args.prompt_csv),
            args.prompt_column,
            args.id_column,
        )
    raise ValueError("Provide --prompt-csv or --emotion-config; activation logging does not generate prompts inline.")


def _with_token_regions(record: PromptRecord, strategy: str) -> PromptRecord:
    if strategy == "none":
        return record
    if strategy != "auto":
        raise ValueError("token region strategy must be one of: auto, none.")
    metadata = dict(record.metadata)
    metadata.setdefault("prompt_regions", _infer_prompt_regions(record.prompt_text, metadata))
    return PromptRecord(
        prompt_id=record.prompt_id,
        prompt_text=record.prompt_text,
        metadata=metadata,
    )


def _infer_prompt_regions(prompt_text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    metadata = metadata or {}
    if str(metadata.get("construct_id", "")).strip():
        return _infer_construct_prompt_regions(prompt_text, metadata)
    if "Scenario:\n" in prompt_text and "\n\nDo not answer yet." in prompt_text:
        return _infer_emotion_prompt_regions(prompt_text)
    if str(metadata.get("emotion", "")).strip():
        return _infer_emotion_prompt_regions(prompt_text)
    return _infer_realization_prompt_regions(prompt_text)


def _infer_construct_prompt_regions(prompt_text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_role = str(metadata.get("prompt_role", "")).strip()
    if prompt_role != "probe":
        return [_region("task", 0, len(prompt_text))]
    scenario_marker = "Scenario:\n"
    scenario_marker_start = prompt_text.find(scenario_marker)
    if scenario_marker_start == -1:
        return [_region("scenario", 0, len(prompt_text))]
    scenario_start = scenario_marker_start + len(scenario_marker)
    processing_markers = ("\n\nContinue processing", "\n\nDo not answer yet.")
    processing_start = next(
        (position for marker in processing_markers if (position := prompt_text.find(marker, scenario_start)) != -1),
        len(prompt_text),
    )
    regions = []
    if scenario_start > 0:
        regions.append(_region("wrapper", 0, scenario_start))
    regions.append(_region("scenario", scenario_start, processing_start))
    if processing_start < len(prompt_text):
        regions.append(_region("processing_instruction", processing_start, len(prompt_text)))
    return regions


def _region(label: str, start: int, end: int) -> dict[str, Any]:
    return {"label": label, "start": start, "end": end}


def _infer_emotion_prompt_regions(prompt_text: str) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    scenario_marker = "Scenario:\n"
    processing_marker = "\n\nDo not answer yet."
    scenario_start = prompt_text.find(scenario_marker)
    processing_start = prompt_text.find(processing_marker)

    if scenario_start == -1 or processing_start == -1:
        return [_region("prompt", 0, len(prompt_text))]

    scenario_content_start = scenario_start + len(scenario_marker)
    if scenario_start > 0:
        regions.append(_region("wrapper", 0, scenario_content_start))
    regions.append(_region("scenario", scenario_content_start, processing_start))
    regions.append(_region("processing_instruction", processing_start, len(prompt_text)))
    return regions


def _infer_realization_prompt_regions(prompt_text: str) -> list[dict[str, Any]]:
    decision_marker = "How much do you want"
    response_marker = "\nRespond with"
    decision_start = prompt_text.find(decision_marker)
    response_start = prompt_text.find(response_marker)

    if decision_start == -1:
        return [_region("prompt", 0, len(prompt_text))]

    regions = [_region("scenario", 0, decision_start)]
    if response_start == -1:
        regions.append(_region("decision_question", decision_start, len(prompt_text)))
    else:
        regions.append(_region("decision_question", decision_start, response_start))
        regions.append(_region("response_instruction", response_start, len(prompt_text)))
    return regions


def _load_prompt_csv(
    path: Path,
    prompt_column: str,
    id_column: str | None,
) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or prompt_column not in reader.fieldnames:
            raise ValueError(f"{path} must contain prompt column '{prompt_column}'.")
        for index, row in enumerate(reader):
            prompt_id = row.get(id_column or "") or row.get("prompt_id") or f"prompt_{index:05d}"
            prompt_text = row[prompt_column]
            metadata = {key: value for key, value in row.items() if key != prompt_column}
            raw_metadata = metadata.pop("metadata_json", None)
            if raw_metadata:
                try:
                    parsed_metadata = json.loads(raw_metadata)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} row {index + 2} has invalid metadata_json.") from exc
                if not isinstance(parsed_metadata, dict):
                    raise ValueError(f"{path} row {index + 2} metadata_json must contain an object.")
                for key, value in metadata.items():
                    if key not in parsed_metadata:
                        parsed_metadata[key] = value
                metadata = parsed_metadata
            for field_name in ("prompt_regions", "prompt_regions_json"):
                raw_regions = metadata.get(field_name)
                if isinstance(raw_regions, str) and raw_regions.strip():
                    try:
                        metadata[field_name] = json.loads(raw_regions)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path} row {index + 2} has invalid {field_name}.") from exc
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    metadata=metadata,
                )
            )
    if not records:
        raise ValueError(f"No prompt rows found in {path}.")
    return records


def _batched(records: list[PromptRecord], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield start // batch_size, records[start : start + batch_size]


def _write_batch(
    output_dir: Path,
    batch_index: int,
    records: list[PromptRecord],
    batch: BatchResiduals,
    layers: list[int],
    storage_dtype: str = "float16",
    include_token_regions: set[str] | None = None,
) -> list[dict[str, Any]]:
    shard_records: list[dict[str, Any]] = []
    np_storage_dtype = _numpy_storage_dtype(storage_dtype)
    token_keep_indices = _token_keep_indices(batch.token_regions, include_token_regions)
    for layer in layers:
        layer_dir = output_dir / "activations" / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = layer_dir / f"batch_{batch_index:06d}.npy"
        index_path = layer_dir / f"batch_{batch_index:06d}.jsonl"

        source_array = batch.hidden_states_by_layer[layer].numpy()
        array = _filter_activation_array(source_array, token_keep_indices).astype(np_storage_dtype, copy=False)
        np.save(tensor_path, array)
        with index_path.open("w", encoding="utf-8") as handle:
            for record, token_ids, token_positions, token_regions, keep_indices in zip(
                records,
                batch.token_ids,
                batch.token_positions,
                batch.token_regions,
                token_keep_indices,
                strict=True,
            ):
                filtered_token_ids = [token_ids[index] for index in keep_indices]
                filtered_token_positions = [token_positions[index] for index in keep_indices]
                filtered_token_regions = [token_regions[index] for index in keep_indices]
                row = {
                    "prompt_id": record.prompt_id,
                    "activation_site": batch.activation_site,
                    "token_mode": batch.token_mode,
                    "token_ids": filtered_token_ids,
                    "token_positions": filtered_token_positions,
                    "token_regions": filtered_token_regions,
                    "num_tokens": len(filtered_token_ids),
                    "metadata": record.metadata,
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        shard_records.append(
            {
                "layer": layer,
                "tensor_file": str(tensor_path.relative_to(output_dir)),
                "index_file": str(index_path.relative_to(output_dir)),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    return shard_records


def _token_keep_indices(token_regions: list[list[str]], include_token_regions: set[str] | None) -> list[list[int]]:
    if include_token_regions is None:
        return [list(range(len(regions))) for regions in token_regions]
    return [
        [index for index, region in enumerate(regions) if region in include_token_regions]
        for regions in token_regions
    ]


def _filter_activation_array(array: np.ndarray, keep_indices_by_row: list[list[int]]) -> np.ndarray:
    if array.shape[0] != len(keep_indices_by_row):
        raise ValueError("Token-region filter row count does not match activation batch size.")
    max_selected = max((len(indices) for indices in keep_indices_by_row), default=0)
    filtered = np.zeros((array.shape[0], max_selected, array.shape[2]), dtype=array.dtype)
    for batch_row, keep_indices in enumerate(keep_indices_by_row):
        if keep_indices:
            filtered[batch_row, : len(keep_indices), :] = array[batch_row, keep_indices, :]
    return filtered


def _numpy_storage_dtype(value: str) -> np.dtype:
    if value == "float16":
        return np.dtype(np.float16)
    if value == "float32":
        return np.dtype(np.float32)
    raise ValueError("storage_dtype must be one of: float16, float32.")


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    logger: ResidualStreamLogger,
    layers: list[int],
    shards: list[dict[str, Any]],
    total_prompts: int,
    selected_prompts: int | None = None,
    complete: bool | None = None,
    stopped_by_runtime: bool = False,
    runtime_seconds: float | None = None,
) -> None:
    stats: dict[str, Any] = {"total_prompts": total_prompts, "total_shards": len(shards)}
    run_mode = getattr(args, "run_mode", None)
    max_runtime_minutes = getattr(args, "max_runtime_minutes", None)
    if run_mode is not None or max_runtime_minutes is not None:
        stats.update(
            {
                "selected_prompts": selected_prompts if selected_prompts is not None else total_prompts,
                "complete": bool(complete) if complete is not None else True,
                "stopped_by_runtime": bool(stopped_by_runtime),
                "max_runtime_minutes": max_runtime_minutes,
                "runtime_seconds": runtime_seconds,
            }
        )
    manifest = {
        "schema_version": "0.1.0",
        "created_at": _utc_now(),
        "model": {
            "model_id": logger.model_id,
            "tokenizer_id": logger.tokenizer_id,
            "num_transformer_layers": logger.num_transformer_layers,
            "d_model": logger.d_model,
        },
        "extraction": {
            "run_name": args.run_name,
            "layers": layers,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "activation_site": args.activation_site,
            "token_mode": args.token_mode,
            "token_region_strategy": args.token_region_strategy,
            "include_token_regions": sorted(args.include_token_regions) if args.include_token_regions else None,
            "storage_dtype": args.storage_dtype,
            "block_path": args.block_path,
            "resolved_block_path": logger.resolved_block_path,
            "local_files_only": args.local_files_only,
            "dtype": args.dtype,
            "device": args.device,
            "device_map": getattr(args, "device_map", None),
            "attn_implementation": getattr(args, "attn_implementation", None),
            "resolved_device": getattr(logger, "resolved_device", logger.device),
        },
        "instrumentation": {
            "mode": "residual_all_layers" if getattr(args, "all_layers", False) else "benchmark",
            "residual_only": True,
            "component_traces": [],
            "all_transformer_layers": bool(getattr(args, "all_layers", False)),
            "interpretability_scope": "residual_stream_only",
            "note": (
                "An all-layer residual trace supports later localization and causal tracing, "
                "but does not capture attention-head, MLP, or feature-level circuit internals."
            ),
        },
        "input": {
            "emotion_config": args.emotion_config,
            "prompt_csv": args.prompt_csv,
            "prompt_column": args.prompt_column,
            "id_column": args.id_column,
            "limit": args.limit,
        },
        "tokenization": {
            "truncation": False,
            "max_length": args.max_length,
            "checked_prompt_count": getattr(logger, "tokenization_prompt_count", total_prompts),
            "max_observed_token_length": getattr(logger, "tokenization_max_observed_length", None),
            "over_limit_count": getattr(logger, "tokenization_over_limit_count", 0),
        },
        "stats": stats,
        "shards": shards,
    }
    if run_mode is not None or max_runtime_minutes is not None:
        manifest["execution"] = {
            "run_mode": run_mode,
            "confirmatory": run_mode == "full" if run_mode is not None else None,
            "max_runtime_minutes": max_runtime_minutes,
            "selected_prompts": selected_prompts if selected_prompts is not None else total_prompts,
            "processed_prompts": total_prompts,
            "complete": bool(complete) if complete is not None else True,
            "stopped_by_runtime": bool(stopped_by_runtime),
            "runtime_seconds": runtime_seconds,
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sanitize_run_part(value: str) -> str:
    value = value.strip().replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "unknown"


def _build_run_name(args: argparse.Namespace, records: list[PromptRecord]) -> str:
    all_layers = bool(getattr(args, "all_layers", False))
    layers = "all" if all_layers else "-".join(str(layer) for layer in (args.layers or []))
    prompt_source = args.prompt_csv or args.emotion_config or "prompt-records"
    fingerprint_payload = {
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "revision": args.revision,
        "layers": "all" if all_layers else args.layers,
        "token_mode": args.token_mode,
        "activation_site": args.activation_site,
        "token_region_strategy": args.token_region_strategy,
        "storage_dtype": args.storage_dtype,
        "include_token_regions": sorted(args.include_token_regions) if args.include_token_regions else None,
        "prompt_source": prompt_source,
        "prompt_ids": [record.prompt_id for record in records],
        "prompt_text_sha256": hashlib.sha256(
            "\n".join(record.prompt_text for record in records).encode("utf-8")
        ).hexdigest(),
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:10]
    model_name = _sanitize_run_part(str(args.model_id).rstrip("/").split("/")[-1])
    if args.emotion_config:
        prompt_name = _sanitize_run_part(Path(args.emotion_config).stem)
    elif args.prompt_csv:
        prompt_name = _sanitize_run_part(Path(args.prompt_csv).stem)
    else:
        prompt_name = "prompt-records"
    return (
        f"{model_name}__prompt-{prompt_name}__layers-{layers}"
        f"__site-{args.activation_site}__tokens-{args.token_mode}"
        f"__store-{args.storage_dtype}__{digest}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Log selected residual stream layers for activation-analysis prompts.")
    parser.add_argument("--model-id", required=True, help="HF model id or local model directory")
    parser.add_argument("--tokenizer-id", help="Optional tokenizer id or local tokenizer directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output directory. If omitted, writes to results/test/residual_streams/<deterministic-run-name>.",
    )
    parser.add_argument("--run-name", default=None, help="Optional deterministic run directory name.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--layers", type=_parse_layers, help="Comma-separated layers, 1-based")
    layer_group.add_argument(
        "--all-layers",
        action="store_true",
        help=(
            "Opt in to a residual-only trace at every transformer layer. This is an "
            "interpretability extension, not a full attention/MLP circuit trace."
        ),
    )
    parser.add_argument("--emotion-config", help="Structured probe config JSON; mutually exclusive with --prompt-csv.")
    parser.add_argument("--prompt-csv", help="Frozen CSV with a prompt text column and optional metadata.")
    parser.add_argument("--prompt-column", default="prompt_text")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--activation-site",
        default="resid_post",
        choices=sorted(SUPPORTED_ACTIVATION_SITES),
        help=(
            "Activation site to log. 'resid_post' is the residual stream after "
            "the transformer block; 'block_output' is an alias for the same "
            "current hook."
        ),
    )
    parser.add_argument("--token-mode", default="nonpad", choices=["all", "nonpad", "final"], help="Which token activations to save.")
    parser.add_argument(
        "--storage-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Numeric dtype used for saved activation .npy shards.",
    )
    parser.add_argument(
        "--token-region-strategy",
        default="auto",
        choices=["auto", "none"],
        help="How to label selected tokens in metadata without filtering saved activations.",
    )
    parser.add_argument(
        "--include-token-regions",
        type=_parse_token_region_filter,
        default=None,
        help=(
            "Comma-separated token region labels to keep when writing activation shards, "
            "for example scenario,decision_question. Defaults to writing all selected tokens."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--run-mode",
        choices=("test", "full"),
        default=None,
        help="Optional named benchmark mode to record in the activation manifest.",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="Optional wall-clock budget; finishes the current batch and stops before the next one.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default=None, help="Optional HF device_map, e.g. auto for cloud multi-GPU runs.")
    parser.add_argument("--attn-implementation", default=None, help="Optional HF attention implementation, e.g. flash_attention_2.")
    parser.add_argument("--block-path", default=None, help="Dotted path to transformer blocks, e.g. model.layers.")
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided.")
    if args.max_runtime_minutes is not None and args.max_runtime_minutes <= 0:
        raise ValueError("--max-runtime-minutes must be positive when provided.")
    if args.run_mode == "test" and args.max_runtime_minutes is None:
        raise ValueError("--run-mode test requires --max-runtime-minutes.")
    if bool(args.prompt_csv) == bool(args.emotion_config):
        raise ValueError("Provide exactly one of --prompt-csv or --emotion-config.")

    records = _load_prompt_records(args)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("No prompts selected for extraction.")
    records = [_with_token_regions(record, args.token_region_strategy) for record in records]

    args.run_name = args.run_name or _build_run_name(args, records)
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/test/residual_streams") / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Pass --overwrite or choose a different --run-name/--output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_started_at = time.monotonic()
    logger = ResidualStreamLogger(
        args.model_id,
        tokenizer_id=args.tokenizer_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        block_path=args.block_path,
        stop_after_last_requested_layer=not args.no_early_stop,
    )

    layers = _resolve_requested_layers(
        args.layers,
        all_layers=args.all_layers,
        num_transformer_layers=logger.num_transformer_layers,
    )

    shards: list[dict[str, Any]] = []
    processed_records: list[PromptRecord] = []
    stopped_by_runtime = False
    prompt_path = output_dir / "prompts.jsonl"
    prompt_path.write_text("", encoding="utf-8")
    for batch_index, batch_records in _batched(records, args.batch_size):
        if (
            args.max_runtime_minutes is not None
            and (time.monotonic() - run_started_at) >= args.max_runtime_minutes * 60
        ):
            stopped_by_runtime = True
            break
        batch = logger.extract_batch(
            [record.prompt_text for record in batch_records],
            [record.prompt_id for record in batch_records],
            layers,
            max_length=args.max_length,
            token_mode=args.token_mode,
            activation_site=args.activation_site,
            token_region_spans=[record.metadata.get("prompt_regions", []) for record in batch_records],
        )
        shards.extend(
            _write_batch(
                output_dir,
                batch_index,
                batch_records,
                batch,
                layers,
                storage_dtype=args.storage_dtype,
                include_token_regions=args.include_token_regions,
            )
        )
        with prompt_path.open("a", encoding="utf-8") as handle:
            for record in batch_records:
                handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
        processed_records.extend(batch_records)
        print(f"wrote batch {batch_index + 1} ({len(batch_records)} prompts)", flush=True)

    if not processed_records:
        raise RuntimeError("The runtime budget expired before any activation batch completed.")
    complete = len(processed_records) == len(records)
    runtime_seconds = round(time.monotonic() - run_started_at, 3)
    _write_manifest(
        output_dir,
        args,
        logger,
        layers,
        shards,
        total_prompts=len(processed_records),
        selected_prompts=len(records),
        complete=complete,
        stopped_by_runtime=stopped_by_runtime or not complete,
        runtime_seconds=runtime_seconds,
    )
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "selected_prompts": len(records),
                "processed_prompts": len(processed_records),
                "complete": complete,
                "stopped_by_runtime": stopped_by_runtime or not complete,
                "runtime_seconds": runtime_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
