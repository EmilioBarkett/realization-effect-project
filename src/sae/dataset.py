from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from emotion_activation.activation_store import ActivationRun, load_activation_run


@dataclass(frozen=True)
class ActivationVectorRecord:
    """One residual-stream vector plus metadata needed for SAE analysis."""

    vector: np.ndarray
    metadata: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_activation_vectors(
    run_dir: str | Path,
    *,
    layers: set[int] | None = None,
    token_regions: set[str] | None = None,
    prompt_metadata_filters: dict[str, set[str]] | None = None,
    activation_site: str | None = "resid_post",
    max_vectors: int | None = None,
) -> Iterator[ActivationVectorRecord]:
    """Yield token-level activation vectors from a residual-stream run.

    The logger writes tensors by layer and batch. This iterator flattens those
    shards into one record per real token while preserving prompt, token, layer,
    region, and run metadata for filtering and SAE interpretation.
    """

    run = load_activation_run(run_dir)
    yielded = 0
    for shard in run.shards:
        if layers is not None and shard.layer not in layers:
            continue

        array = np.load(shard.tensor_path, mmap_mode="r")
        index_rows = _read_jsonl(shard.index_path)
        for batch_row, row in enumerate(index_rows):
            if activation_site is not None and row.get("activation_site", activation_site) != activation_site:
                continue
            prompt_metadata = row.get("metadata", {})
            if prompt_metadata_filters is not None:
                if any(
                    str(prompt_metadata.get(key, "")).strip() not in allowed_values
                    for key, allowed_values in prompt_metadata_filters.items()
                ):
                    continue

            token_ids = row.get("token_ids", [])
            token_positions = row.get("token_positions", [])
            token_region_labels = row.get("token_regions") or ["unknown"] * len(token_ids)
            num_tokens = int(row.get("num_tokens", len(token_ids)))
            if num_tokens > array.shape[1]:
                raise ValueError(
                    f"{shard.index_path} row {batch_row + 1} has {num_tokens} tokens, "
                    f"but tensor sequence length is {array.shape[1]}."
                )
            if len(token_region_labels) < num_tokens:
                raise ValueError(
                    f"{shard.index_path} row {batch_row + 1} has fewer token region labels "
                    "than token vectors."
                )

            for token_index in range(num_tokens):
                region = str(token_region_labels[token_index])
                if token_regions is not None and region not in token_regions:
                    continue

                metadata = {
                    "run_dir": str(run.path),
                    "layer": shard.layer,
                    "tensor_file": str(shard.tensor_path.relative_to(run.path)),
                    "index_file": str(shard.index_path.relative_to(run.path)),
                    "batch_row": batch_row,
                    "token_index": token_index,
                    "prompt_id": row.get("prompt_id"),
                    "activation_site": row.get("activation_site"),
                    "token_mode": row.get("token_mode"),
                    "token_id": token_ids[token_index] if token_index < len(token_ids) else None,
                    "token_position": (
                        token_positions[token_index] if token_index < len(token_positions) else None
                    ),
                    "token_region": region,
                    "prompt_metadata": prompt_metadata,
                }
                yield ActivationVectorRecord(
                    vector=np.asarray(array[batch_row, token_index, :]),
                    metadata=metadata,
                )
                yielded += 1
                if max_vectors is not None and yielded >= max_vectors:
                    return


def summarize_activation_dataset(
    run_dirs: list[str | Path],
    *,
    layers: set[int] | None = None,
    token_regions: set[str] | None = None,
    prompt_metadata_filters: dict[str, set[str]] | None = None,
    activation_site: str | None = "resid_post",
    max_vectors: int | None = None,
) -> dict[str, Any]:
    """Count vectors by layer and token region without training an SAE."""

    counts_by_layer: dict[int, int] = {}
    counts_by_region: dict[str, int] = {}
    total_vectors = 0
    hidden_size: int | None = None

    for run_dir in run_dirs:
        for record in iter_activation_vectors(
            run_dir,
            layers=layers,
            token_regions=token_regions,
            prompt_metadata_filters=prompt_metadata_filters,
            activation_site=activation_site,
            max_vectors=None if max_vectors is None else max_vectors - total_vectors,
        ):
            layer = int(record.metadata["layer"])
            region = str(record.metadata["token_region"])
            counts_by_layer[layer] = counts_by_layer.get(layer, 0) + 1
            counts_by_region[region] = counts_by_region.get(region, 0) + 1
            hidden_size = int(record.vector.shape[0])
            total_vectors += 1
            if max_vectors is not None and total_vectors >= max_vectors:
                break
        if max_vectors is not None and total_vectors >= max_vectors:
            break

    return {
        "total_vectors": total_vectors,
        "hidden_size": hidden_size,
        "counts_by_layer": {str(key): counts_by_layer[key] for key in sorted(counts_by_layer)},
        "counts_by_region": {
            key: counts_by_region[key]
            for key in sorted(counts_by_region)
        },
    }


def load_activation_runs(paths: list[str | Path]) -> list[ActivationRun]:
    return [load_activation_run(path) for path in paths]
