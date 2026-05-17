from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class ActivationShard:
    layer: int
    tensor_path: Path
    index_path: Path
    shape: tuple[int, int, int]
    dtype: str | None = None


@dataclass(frozen=True)
class ActivationRun:
    path: Path
    manifest: dict[str, Any]
    prompts: list[dict[str, Any]]
    shards: list[ActivationShard]

    def iter_layer_arrays(self, layer: int) -> Iterator[tuple[ActivationShard, np.ndarray]]:
        for shard in self.shards:
            if shard.layer != layer:
                continue
            yield shard, np.load(shard.tensor_path, mmap_mode="r")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            rows.append(row)
    return rows


def load_activation_run(path: str | Path) -> ActivationRun:
    run_path = Path(path)
    manifest_path = run_path / "manifest.json"
    prompts_path = run_path / "prompts.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Missing prompts file: {prompts_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts = _read_jsonl(prompts_path)

    shards: list[ActivationShard] = []
    for shard in manifest.get("shards", []):
        if not isinstance(shard, dict):
            raise ValueError("Manifest shard entries must be objects.")
        try:
            layer = int(shard["layer"])
            tensor_file = Path(str(shard["tensor_file"]))
            index_file = Path(str(shard["index_file"]))
            shape = tuple(int(value) for value in shard["shape"])
            dtype = str(shard["dtype"]) if shard.get("dtype") is not None else None
        except KeyError as exc:
            raise ValueError(f"Manifest shard is missing required key: {exc}") from exc
        if len(shape) != 3:
            raise ValueError(f"Shard shape for layer {layer} must be 3D, got {shape}.")
        shards.append(
            ActivationShard(
                layer=layer,
                tensor_path=run_path / tensor_file,
                index_path=run_path / index_file,
                shape=shape,
                dtype=dtype,
            )
        )

    return ActivationRun(path=run_path, manifest=manifest, prompts=prompts, shards=shards)


def validate_activation_run(path: str | Path) -> list[str]:
    run = load_activation_run(path)
    errors: list[str] = []

    extraction = run.manifest.get("extraction", {})
    model = run.manifest.get("model", {})
    stats = run.manifest.get("stats", {})
    expected_layers = set(extraction.get("layers", []))
    expected_token_mode = extraction.get("token_mode")
    expected_activation_site = extraction.get("activation_site", "resid_post")
    expected_storage_dtype = extraction.get("storage_dtype", "float32")
    expected_d_model = model.get("d_model")

    if stats.get("total_prompts") != len(run.prompts):
        errors.append(
            f"stats.total_prompts={stats.get('total_prompts')} but prompts.jsonl has {len(run.prompts)} rows"
        )
    if stats.get("total_shards") != len(run.shards):
        errors.append(
            f"stats.total_shards={stats.get('total_shards')} but manifest lists {len(run.shards)} shards"
        )
    if expected_layers and {shard.layer for shard in run.shards} - expected_layers:
        errors.append("manifest contains shard layers outside extraction.layers")

    for shard in run.shards:
        if not shard.tensor_path.exists():
            errors.append(f"missing tensor file: {shard.tensor_path}")
            continue
        if not shard.index_path.exists():
            errors.append(f"missing index file: {shard.index_path}")
            continue

        try:
            array = np.load(shard.tensor_path, mmap_mode="r")
        except Exception as exc:
            errors.append(f"could not load tensor file {shard.tensor_path}: {exc}")
            continue

        if tuple(array.shape) != shard.shape:
            errors.append(
                f"{shard.tensor_path} shape {tuple(array.shape)} does not match manifest {shard.shape}"
            )
        shard_dtype = shard.dtype or str(expected_storage_dtype)
        if str(array.dtype) != shard_dtype:
            errors.append(f"{shard.tensor_path} dtype is {array.dtype}, expected {shard_dtype}")
        if str(array.dtype) != str(expected_storage_dtype):
            errors.append(
                f"{shard.tensor_path} dtype is {array.dtype}, "
                f"but manifest extraction.storage_dtype is {expected_storage_dtype}"
            )
        if expected_d_model is not None and array.shape[2] != int(expected_d_model):
            errors.append(
                f"{shard.tensor_path} hidden size {array.shape[2]} does not match d_model={expected_d_model}"
            )

        index_rows = _read_jsonl(shard.index_path)
        if len(index_rows) != array.shape[0]:
            errors.append(
                f"{shard.index_path} has {len(index_rows)} rows but tensor batch size is {array.shape[0]}"
            )

        for row_number, row in enumerate(index_rows, start=1):
            token_ids = row.get("token_ids", [])
            token_positions = row.get("token_positions", [])
            token_regions = row.get("token_regions")
            if row.get("token_mode") != expected_token_mode:
                errors.append(f"{shard.index_path}:{row_number} token_mode does not match manifest")
            if row.get("activation_site", expected_activation_site) != expected_activation_site:
                errors.append(f"{shard.index_path}:{row_number} activation_site does not match manifest")
            if len(token_ids) != len(token_positions):
                errors.append(f"{shard.index_path}:{row_number} token_ids/token_positions length mismatch")
            if token_regions is not None and len(token_regions) != len(token_ids):
                errors.append(f"{shard.index_path}:{row_number} token_regions/token_ids length mismatch")
            if row.get("num_tokens") != len(token_ids):
                errors.append(f"{shard.index_path}:{row_number} num_tokens does not match token_ids")
            if len(token_ids) > array.shape[1]:
                errors.append(f"{shard.index_path}:{row_number} has more tokens than tensor sequence length")

    return errors
