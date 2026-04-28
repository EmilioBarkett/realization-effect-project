from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from realization_effect.runner import build_prompt, load_conditions

from .residual_streams import BatchResiduals, ResidualStreamLogger


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


def _load_prompt_records(args: argparse.Namespace) -> list[PromptRecord]:
    if args.prompt_csv:
        return _load_prompt_csv(Path(args.prompt_csv), args.prompt_column, args.id_column)

    conditions = load_conditions(Path(args.conditions_csv))
    records: list[PromptRecord] = []
    for index, condition in enumerate(conditions):
        prompt_text = build_prompt(
            outcome_type=str(condition["outcome_type"]),
            amount=int(condition["amount"]),
            prompt_version=args.prompt_version,
        )
        prompt_id = str(condition.get("condition") or f"condition_{index:05d}")
        records.append(
            PromptRecord(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                metadata={
                    **condition,
                    "prompt_version": args.prompt_version,
                },
            )
        )
    return records


def _load_prompt_csv(path: Path, prompt_column: str, id_column: str | None) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or prompt_column not in reader.fieldnames:
            raise ValueError(f"{path} must contain prompt column '{prompt_column}'.")
        for index, row in enumerate(reader):
            prompt_id = row.get(id_column or "") or row.get("prompt_id") or f"prompt_{index:05d}"
            prompt_text = row[prompt_column]
            metadata = {key: value for key, value in row.items() if key != prompt_column}
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
) -> list[dict[str, Any]]:
    shard_records: list[dict[str, Any]] = []
    for layer in layers:
        layer_dir = output_dir / "activations" / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = layer_dir / f"batch_{batch_index:06d}.npy"
        index_path = layer_dir / f"batch_{batch_index:06d}.jsonl"

        np.save(tensor_path, batch.hidden_states_by_layer[layer].numpy())
        with index_path.open("w", encoding="utf-8") as handle:
            for record, token_ids, token_positions in zip(
                records,
                batch.token_ids,
                batch.token_positions,
                strict=True,
            ):
                row = {
                    "prompt_id": record.prompt_id,
                    "token_ids": token_ids,
                    "token_positions": token_positions,
                    "num_tokens": len(token_ids),
                    "metadata": record.metadata,
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        shard_records.append(
            {
                "layer": layer,
                "tensor_file": str(tensor_path.relative_to(output_dir)),
                "index_file": str(index_path.relative_to(output_dir)),
                "shape": list(batch.hidden_states_by_layer[layer].shape),
            }
        )
    return shard_records


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
            "layers": layers,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "local_files_only": args.local_files_only,
            "dtype": args.dtype,
            "device": args.device,
        },
        "input": {
            "conditions_csv": args.conditions_csv,
            "prompt_csv": args.prompt_csv,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log selected residual stream layers for realization-effect prompts."
    )
    parser.add_argument("--model-id", required=True, help="HF model id or local model directory")
    parser.add_argument("--tokenizer-id", help="Optional tokenizer id or local tokenizer directory")
    parser.add_argument("--output-dir", default="results/residual_streams")
    parser.add_argument("--layers", type=_parse_layers, required=True, help="Comma-separated layers, 1-based")
    parser.add_argument("--conditions-csv", default="configs/realization_effect/conditions.csv")
    parser.add_argument("--prompt-version", default="absolute", choices=["absolute", "balance", "qualitative"])
    parser.add_argument("--prompt-csv", help="CSV with a prompt text column; overrides --conditions-csv")
    parser.add_argument("--prompt-column", default="prompt_text")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_prompt_records(args)
    if args.limit is not None:
        records = records[: args.limit]

    logger = ResidualStreamLogger(
        args.model_id,
        tokenizer_id=args.tokenizer_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        dtype=args.dtype,
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
        )
        shards.extend(_write_batch(output_dir, batch_index, batch_records, batch, args.layers))
        print(f"wrote batch {batch_index + 1} ({len(batch_records)} prompts)", flush=True)

    _write_manifest(output_dir, args, logger, args.layers, shards, total_prompts=len(records))
    print(f"wrote manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
