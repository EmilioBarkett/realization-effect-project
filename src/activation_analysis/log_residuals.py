from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from realization_effect.runner import build_prompt, load_conditions

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

    behavioral_summaries = _load_behavioral_summaries(
        Path(args.results_csv),
        prompt_version=args.prompt_version,
        enabled=not args.no_results_join,
    )
    if args.prompt_csv:
        return _load_prompt_csv(
            Path(args.prompt_csv),
            args.prompt_column,
            args.id_column,
            behavioral_summaries=behavioral_summaries,
        )

    conditions = load_conditions(Path(args.conditions_csv))
    records: list[PromptRecord] = []
    for index, condition in enumerate(conditions):
        condition_name = str(condition.get("condition") or f"condition_{index:05d}")
        metadata = {
            **condition,
            "prompt_version": args.prompt_version,
        }
        if condition_name in behavioral_summaries:
            metadata["behavioral_results"] = behavioral_summaries[condition_name]
        prompt_text = build_prompt(
            outcome_type=str(condition["outcome_type"]),
            amount=int(condition["amount"]),
            prompt_version=args.prompt_version,
        )
        records.append(
            PromptRecord(
                prompt_id=condition_name,
                prompt_text=prompt_text,
                metadata=metadata,
            )
        )
    return records


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
    if "Scenario:\n" in prompt_text and "\n\nDo not answer yet." in prompt_text:
        return _infer_emotion_prompt_regions(prompt_text)
    if str(metadata.get("emotion", "")).strip():
        return _infer_emotion_prompt_regions(prompt_text)
    return _infer_realization_prompt_regions(prompt_text)


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
    behavioral_summaries: dict[str, dict[str, Any]] | None = None,
) -> list[PromptRecord]:
    behavioral_summaries = behavioral_summaries or {}
    records: list[PromptRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or prompt_column not in reader.fieldnames:
            raise ValueError(f"{path} must contain prompt column '{prompt_column}'.")
        for index, row in enumerate(reader):
            prompt_id = row.get(id_column or "") or row.get("prompt_id") or f"prompt_{index:05d}"
            prompt_text = row[prompt_column]
            metadata = {key: value for key, value in row.items() if key != prompt_column}
            condition = str(metadata.get("condition", "")).strip()
            if condition in behavioral_summaries:
                metadata["behavioral_results"] = behavioral_summaries[condition]
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


def _load_behavioral_summaries(
    path: Path,
    *,
    prompt_version: str,
    enabled: bool,
) -> dict[str, dict[str, Any]]:
    if not enabled or not path.exists():
        return {}

    groups: dict[str, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"condition", "prompt_version", "parsed_wager", "risk_profile", "model"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            return {}

        for row in reader:
            if str(row.get("prompt_version", "")).strip() != prompt_version:
                continue
            condition = str(row.get("condition", "")).strip()
            if not condition:
                continue
            group = groups.setdefault(
                condition,
                {
                    "condition": condition,
                    "prompt_version": prompt_version,
                    "rows": 0,
                    "valid_wager_rows": 0,
                    "wager_sum": 0.0,
                    "log_wager_sum": 0.0,
                    "valid_risk_rows": 0,
                    "risk_profile_sum": 0.0,
                    "models": set(),
                },
            )
            group["rows"] += 1
            model = str(row.get("model", "")).strip()
            if model:
                group["models"].add(model)
            try:
                wager = float(row.get("parsed_wager", ""))
                log_wager = float(row.get("log_wager", ""))
            except ValueError:
                pass
            else:
                group["valid_wager_rows"] += 1
                group["wager_sum"] += wager
                group["log_wager_sum"] += log_wager
            try:
                risk_profile = float(row.get("risk_profile", ""))
            except ValueError:
                pass
            else:
                group["valid_risk_rows"] += 1
                group["risk_profile_sum"] += risk_profile

    summaries: dict[str, dict[str, Any]] = {}
    for condition, group in groups.items():
        valid_wager_rows = int(group["valid_wager_rows"])
        valid_risk_rows = int(group["valid_risk_rows"])
        summaries[condition] = {
            "condition": condition,
            "prompt_version": prompt_version,
            "rows": int(group["rows"]),
            "valid_wager_rows": valid_wager_rows,
            "mean_wager": group["wager_sum"] / valid_wager_rows if valid_wager_rows else None,
            "mean_log_wager": group["log_wager_sum"] / valid_wager_rows if valid_wager_rows else None,
            "valid_risk_rows": valid_risk_rows,
            "mean_risk_profile": group["risk_profile_sum"] / valid_risk_rows if valid_risk_rows else None,
            "models": sorted(group["models"]),
        }
    return summaries


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
) -> None:
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
            "resolved_device": logger.device,
        },
        "input": {
            "conditions_csv": args.conditions_csv,
            "emotion_config": args.emotion_config,
            "prompt_csv": args.prompt_csv,
            "results_csv": None if args.no_results_join else args.results_csv,
            "prompt_version": args.prompt_version,
            "prompt_column": args.prompt_column,
            "id_column": args.id_column,
            "limit": args.limit,
        },
        "stats": {
            "total_prompts": total_prompts,
            "total_shards": len(shards),
        },
        "shards": shards,
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
    layers = "-".join(str(layer) for layer in args.layers)
    prompt_source = args.prompt_csv or args.conditions_csv
    if args.emotion_config:
        prompt_source = args.emotion_config
    fingerprint_payload = {
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "revision": args.revision,
        "layers": args.layers,
        "token_mode": args.token_mode,
        "activation_site": args.activation_site,
        "token_region_strategy": args.token_region_strategy,
        "storage_dtype": args.storage_dtype,
        "include_token_regions": sorted(args.include_token_regions) if args.include_token_regions else None,
        "prompt_source": prompt_source,
        "prompt_version": args.prompt_version,
        "prompt_ids": [record.prompt_id for record in records],
        "prompt_text_sha256": hashlib.sha256(
            "\n".join(record.prompt_text for record in records).encode("utf-8")
        ).hexdigest(),
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:10]
    model_name = _sanitize_run_part(Path(str(args.model_id)).name)
    if args.emotion_config:
        prompt_name = _sanitize_run_part(Path(args.emotion_config).stem)
    else:
        prompt_name = _sanitize_run_part(args.prompt_version if not args.prompt_csv else Path(args.prompt_csv).stem)
    return (
        f"{model_name}__prompt-{prompt_name}__layers-{layers}"
        f"__site-{args.activation_site}__tokens-{args.token_mode}"
        f"__store-{args.storage_dtype}__{digest}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log selected residual stream layers for realization-effect prompts."
    )
    parser.add_argument("--model-id", required=True, help="HF model id or local model directory")
    parser.add_argument("--tokenizer-id", help="Optional tokenizer id or local tokenizer directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output directory. If omitted, writes to results/test/residual_streams/<deterministic-run-name>.",
    )
    parser.add_argument("--run-name", default=None, help="Optional deterministic run directory name.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    parser.add_argument("--layers", type=_parse_layers, required=True, help="Comma-separated layers, 1-based")
    parser.add_argument("--conditions-csv", default="configs/realization_effect/conditions.csv")
    parser.add_argument("--results-csv", default="results/results.csv", help="Optional behavioral results CSV for condition-level metadata joins.")
    parser.add_argument("--no-results-join", action="store_true", help="Do not attach behavioral result summaries to prompt metadata.")
    parser.add_argument("--emotion-config", help="Emotion probe config JSON; overrides --conditions-csv and --prompt-csv.")
    parser.add_argument("--prompt-version", default="absolute", choices=["absolute", "balance", "qualitative"])
    parser.add_argument("--prompt-csv", help="CSV with a prompt text column; overrides --conditions-csv")
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
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--block-path", default=None, help="Dotted path to transformer blocks, e.g. model.layers.")
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided.")

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

    logger = ResidualStreamLogger(
        args.model_id,
        tokenizer_id=args.tokenizer_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
        block_path=args.block_path,
        stop_after_last_requested_layer=not args.no_early_stop,
    )

    prompt_path = output_dir / "prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    shards: list[dict[str, Any]] = []
    for batch_index, batch_records in _batched(records, args.batch_size):
        batch = logger.extract_batch(
            [record.prompt_text for record in batch_records],
            [record.prompt_id for record in batch_records],
            args.layers,
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
                args.layers,
                storage_dtype=args.storage_dtype,
                include_token_regions=args.include_token_regions,
            )
        )
        print(f"wrote batch {batch_index + 1} ({len(batch_records)} prompts)", flush=True)

    _write_manifest(output_dir, args, logger, args.layers, shards, total_prompts=len(records))
    print(f"wrote manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
