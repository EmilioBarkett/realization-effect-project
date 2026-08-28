from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

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


@dataclass(frozen=True)
class ActivationVectorRecord:
    """One token-level activation vector with reproducible provenance."""

    vector: np.ndarray
    metadata: dict[str, Any]
    run_path: Path
    layer: int
    shard_index: int
    row_index: int
    token_index: int
    token_position: int | None
    token_region: str | None


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


def _prompt_metadata_by_id(run: ActivationRun) -> dict[str, dict[str, Any]]:
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(run.prompts, start=1):
        prompt_id = row.get("prompt_id")
        if prompt_id is None:
            raise ValueError(f"prompts.jsonl row {row_number} is missing prompt_id.")
        prompt_metadata = row.get("metadata", {})
        if not isinstance(prompt_metadata, dict):
            raise ValueError(f"prompts.jsonl row {row_number} metadata must be an object.")
        merged = dict(prompt_metadata)
        merged["prompt_id"] = str(prompt_id)
        if row.get("prompt_text") is not None:
            merged["prompt_text"] = row["prompt_text"]
        metadata_by_id[str(prompt_id)] = merged
    return metadata_by_id


def _record_metadata(
    prompt_metadata: dict[str, Any],
    index_row: dict[str, Any],
    *,
    prompt_id: str,
) -> dict[str, Any]:
    index_metadata = index_row.get("metadata", {})
    if not isinstance(index_metadata, dict):
        raise ValueError(f"Activation index metadata for {prompt_id} must be an object.")

    merged_prompt_metadata = dict(prompt_metadata)
    merged_prompt_metadata.update(index_metadata)
    merged_prompt_metadata["prompt_id"] = prompt_id

    metadata = dict(merged_prompt_metadata)
    metadata["prompt_metadata"] = dict(merged_prompt_metadata)
    return metadata


def _matches_prompt_metadata(
    metadata: dict[str, Any],
    filters: Mapping[str, Iterable[Any]] | None,
) -> bool:
    if filters is None:
        return True
    nested = metadata.get("prompt_metadata", {})
    if not isinstance(nested, dict):
        nested = {}
    for key, allowed_values in filters.items():
        if isinstance(allowed_values, (str, bytes)):
            allowed = {str(allowed_values)}
        else:
            allowed = {str(value) for value in allowed_values}
        value = metadata.get(key, nested.get(key))
        if str(value) not in allowed:
            return False
    return True


def _validate_index_row(row: dict[str, Any], path: Path, row_number: int, sequence_length: int) -> None:
    token_ids = row.get("token_ids", [])
    token_positions = row.get("token_positions", [])
    token_regions = row.get("token_regions")
    if not isinstance(token_ids, list) or not isinstance(token_positions, list):
        raise ValueError(f"{path}:{row_number} token_ids and token_positions must be lists.")
    if len(token_ids) != len(token_positions):
        raise ValueError(f"{path}:{row_number} token_ids/token_positions length mismatch.")
    if token_regions is not None:
        if not isinstance(token_regions, list):
            raise ValueError(f"{path}:{row_number} token_regions must be a list when present.")
        if len(token_regions) != len(token_ids):
            raise ValueError(f"{path}:{row_number} token_regions/token_ids length mismatch.")
    if row.get("num_tokens") is not None and row["num_tokens"] != len(token_ids):
        raise ValueError(f"{path}:{row_number} num_tokens does not match token_ids.")
    if len(token_ids) > sequence_length:
        raise ValueError(f"{path}:{row_number} has more tokens than tensor sequence length.")


def iter_activation_vectors(
    run_dir: str | Path,
    *,
    layers: set[int] | None = None,
    token_regions: set[str] | None = None,
    prompt_metadata_filters: Mapping[str, Iterable[Any]] | None = None,
    activation_site: str | None = "resid_post",
    max_vectors: int | None = None,
) -> Iterator[ActivationVectorRecord]:
    """Yield token-level vectors from an activation run using memory mapping.

    The tensor row and token position are taken from the shard index, while
    prompt metadata are merged into a flat record for convenient filtering and
    retained under ``prompt_metadata`` for provenance.
    """

    if max_vectors is not None and max_vectors < 0:
        raise ValueError("max_vectors must be non-negative when provided.")
    if max_vectors == 0:
        return

    run = load_activation_run(run_dir)
    prompt_metadata_by_id = _prompt_metadata_by_id(run)
    yielded = 0

    for shard_index, shard in enumerate(run.shards):
        if layers is not None and shard.layer not in layers:
            continue
        if not shard.tensor_path.exists():
            raise FileNotFoundError(f"Missing tensor file: {shard.tensor_path}")
        if not shard.index_path.exists():
            raise FileNotFoundError(f"Missing index file: {shard.index_path}")

        array = np.load(shard.tensor_path, mmap_mode="r")
        if array.ndim != 3:
            raise ValueError(f"{shard.tensor_path} must be a 3D array, got shape {array.shape}.")
        index_rows = _read_jsonl(shard.index_path)
        if len(index_rows) != array.shape[0]:
            raise ValueError(
                f"{shard.index_path} has {len(index_rows)} rows but tensor batch size is {array.shape[0]}."
            )

        for row_index, index_row in enumerate(index_rows):
            prompt_id = str(index_row.get("prompt_id") or "")
            if not prompt_id:
                raise ValueError(f"{shard.index_path}:{row_index + 1} is missing prompt_id.")
            row_activation_site = str(
                index_row.get(
                    "activation_site",
                    run.manifest.get("extraction", {}).get("activation_site", "resid_post"),
                )
            )
            if activation_site is not None and row_activation_site != activation_site:
                continue
            prompt_metadata = prompt_metadata_by_id.get(prompt_id, {})
            metadata = _record_metadata(prompt_metadata, index_row, prompt_id=prompt_id)
            _validate_index_row(index_row, shard.index_path, row_index + 1, array.shape[1])
            if not _matches_prompt_metadata(metadata, prompt_metadata_filters):
                continue

            token_ids = index_row.get("token_ids", [])
            token_positions = index_row.get("token_positions", [])
            token_regions_for_row = index_row.get("token_regions")
            for token_index, token_position_value in enumerate(token_positions):
                token_region = (
                    str(token_regions_for_row[token_index])
                    if token_regions_for_row is not None
                    else None
                )
                if token_regions is not None and token_region not in token_regions:
                    continue
                token_position = int(token_position_value) if token_position_value is not None else None
                record_metadata = dict(metadata)
                record_metadata.update(
                    {
                        "layer": shard.layer,
                        "activation_site": row_activation_site,
                        "token_mode": index_row.get("token_mode"),
                        "token_id": token_ids[token_index],
                        "token_index": token_index,
                        "token_position": token_position,
                        "token_region": token_region,
                    }
                )
                yield ActivationVectorRecord(
                    vector=np.asarray(array[row_index, token_index, :]),
                    metadata=record_metadata,
                    run_path=run.path,
                    layer=shard.layer,
                    shard_index=shard_index,
                    row_index=row_index,
                    token_index=token_index,
                    token_position=token_position,
                    token_region=token_region,
                )
                yielded += 1
                if max_vectors is not None and yielded >= max_vectors:
                    return


def summarize_activation_dataset(
    run_dirs: Iterable[str | Path],
    *,
    layers: set[int] | None = None,
    token_regions: set[str] | None = None,
    prompt_metadata_filters: Mapping[str, Iterable[Any]] | None = None,
    activation_site: str | None = "resid_post",
    max_vectors: int | None = None,
) -> dict[str, Any]:
    """Summarize filtered vectors without materializing activation tensors."""

    total_vectors = 0
    hidden_size: int | None = None
    counts_by_layer: dict[str, int] = {}
    counts_by_region: dict[str, int] = {}

    for run_dir in run_dirs:
        for record in iter_activation_vectors(
            run_dir,
            layers=layers,
            token_regions=token_regions,
            prompt_metadata_filters=prompt_metadata_filters,
            activation_site=activation_site,
            max_vectors=None if max_vectors is None else max_vectors - total_vectors,
        ):
            vector_hidden_size = int(record.vector.shape[0])
            if hidden_size is None:
                hidden_size = vector_hidden_size
            elif hidden_size != vector_hidden_size:
                raise ValueError(
                    f"Activation hidden size changed from {hidden_size} to {vector_hidden_size}."
                )
            total_vectors += 1
            layer_key = str(record.layer)
            counts_by_layer[layer_key] = counts_by_layer.get(layer_key, 0) + 1
            region_key = record.token_region or ""
            counts_by_region[region_key] = counts_by_region.get(region_key, 0) + 1
            if max_vectors is not None and total_vectors >= max_vectors:
                return {
                    "total_vectors": total_vectors,
                    "hidden_size": hidden_size,
                    "counts_by_layer": dict(sorted(counts_by_layer.items())),
                    "counts_by_region": dict(sorted(counts_by_region.items())),
                }

    return {
        "total_vectors": total_vectors,
        "hidden_size": hidden_size,
        "counts_by_layer": dict(sorted(counts_by_layer.items())),
        "counts_by_region": dict(sorted(counts_by_region.items())),
    }


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
    tokenization = run.manifest.get("tokenization")
    if tokenization is not None:
        if not isinstance(tokenization, dict):
            errors.append("manifest tokenization must be an object")
        else:
            if tokenization.get("truncation") is not False:
                errors.append("manifest tokenization.truncation must be false")
            expected_max_length = extraction.get("max_length")
            if (
                expected_max_length is not None
                and tokenization.get("max_length") != expected_max_length
            ):
                errors.append("manifest tokenization.max_length does not match extraction.max_length")
            checked_prompt_count = tokenization.get("checked_prompt_count")
            if checked_prompt_count != len(run.prompts):
                errors.append(
                    "manifest tokenization.checked_prompt_count does not match prompts.jsonl"
                )
            over_limit_count = tokenization.get("over_limit_count")
            if over_limit_count != 0:
                errors.append("manifest tokenization.over_limit_count must be zero")
            max_observed = tokenization.get("max_observed_token_length")
            if (
                expected_max_length is not None
                and max_observed is not None
                and max_observed > expected_max_length
            ):
                errors.append("manifest tokenization.max_observed_token_length exceeds max_length")

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
