"""Fail-closed validation and composition of distributed worker outputs.

Workers write immutable JSONL/CSV output plus an adjacent JSON manifest.  This
module validates each pair before reading anything into a combined artifact:
identity sets, completeness, provenance, model/runtime metadata, and the
confirmatory boundary are all checked before composition.  Source files are
never modified.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .distributed_contracts import (
    DISTRIBUTED_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    canonical_hash,
    canonical_json,
    file_sha256,
    load_json_object,
    provenance_version_families,
)


COMPOSITION_MANIFEST_TYPE = "benchmark_composition"
WORKER_MANIFEST_TYPES = frozenset(
    {
        "benchmark_worker_output",
        "benchmark_worker_result",
        "distributed_worker_output",
        "worker_output",
        "construct_steering_output",
        "construct_behavior_output",
        "residual_interchange_output",
    }
)


def _default_manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Worker output does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Worker output {path}:{line_number} is blank or truncated.")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Worker output {path}:{line_number} is not valid JSON.") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Worker output {path}:{line_number} must contain a JSON object.")
                rows.append(row)
        if not rows:
            raise ValueError(f"Worker output is empty: {path}")
        return rows
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Worker output is not valid JSON: {path}") from exc
        if isinstance(payload, Mapping):
            for key in ("rows", "records", "outputs", "observations"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list) or not payload or not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"Worker output must contain a non-empty JSON array of objects: {path}")
        return [dict(row) for row in payload]

    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Worker output is missing a CSV header: {path}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Worker output has duplicate CSV header names: {path}")
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Worker output {path}:{line_number} has extra CSV fields.")
            if any(value is None for value in row.values()):
                raise ValueError(f"Worker output {path}:{line_number} has missing CSV fields.")
            if not any(str(value).strip() for value in row.values()):
                raise ValueError(f"Worker output {path}:{line_number} is blank or truncated.")
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"Worker output is empty: {path}")
    return rows


def _parse_json_if_needed(value: Any, *, field_name: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if field_name.endswith("_json") or stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            if field_name.endswith("_json"):
                raise ValueError(f"{field_name} contains invalid JSON.") from exc
    return value


def _mapping_sources(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = [payload]
    for key, value in payload.items():
        parsed = _parse_json_if_needed(value, field_name=str(key))
        if isinstance(parsed, Mapping):
            sources.append(parsed)
    for key in ("model", "tokenizer", "execution", "runtime", "generation", "decoding", "causal"):
        value = payload.get(key)
        parsed = _parse_json_if_needed(value, field_name=key)
        if isinstance(parsed, Mapping) and parsed not in sources:
            sources.append(parsed)
    return sources


def _present_value(payload: Mapping[str, Any], aliases: Sequence[str]) -> tuple[bool, Any]:
    normalized = {str(alias).casefold().replace("-", "_") for alias in aliases}
    for source in _mapping_sources(payload):
        for key, value in source.items():
            key_normalized = str(key).strip().casefold().replace("-", "_")
            if key_normalized in normalized:
                return True, _parse_json_if_needed(value, field_name=str(key))
    return False, None


def _required_value(payload: Mapping[str, Any], aliases: Sequence[str], *, label: str) -> Any:
    present, value = _present_value(payload, aliases)
    if not present or value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Worker manifest is missing required {label}.")
    return value


def _text(value: Any, *, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} must be a non-empty string.")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must be a non-empty string.")
    return result


def _strict_ids(value: Any, *, label: str) -> tuple[str, ...]:
    value = _parse_json_if_needed(value, field_name=label)
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"{label} must be a list of non-empty strings.")
    result = tuple(_text(item, label=label) for item in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate IDs.")
    return result


def _ids_from_manifest(payload: Mapping[str, Any], aliases: Sequence[str], *, label: str) -> tuple[str, ...]:
    present, value = _present_value(payload, aliases)
    if present:
        return _strict_ids(value, label=label)
    return ()


def _expected_observation_owners(payload: Mapping[str, Any], *, label: str) -> dict[str, str]:
    """Read an optional observation-to-request map from a worker manifest."""

    present, value = _present_value(payload, ("expected_observations",))
    if not present:
        return {}
    value = _parse_json_if_needed(value, field_name="expected_observations")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} expected_observations must be a list.")
    if not value:
        raise ValueError(f"{label} expected_observations must not be empty.")
    if all(not isinstance(item, Mapping) for item in value):
        # Some detached manifests provide only a redundant list of IDs.  The
        # explicit owner check is available when the richer mapping is used.
        return {}
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} expected_observations must use one uniform item shape.")
    owners: dict[str, str] = {}
    for item in value:
        observation = item.get("observation_id", item.get("record_id", item.get("id")))
        request = item.get("request_id", item.get("prompt_id", item.get("request")))
        observation_id = _text(observation, label="expected observation_id")
        request_id = _text(request, label="expected observation request_id")
        if observation_id in owners:
            raise ValueError(f"{label} expected_observations contains duplicate IDs.")
        owners[observation_id] = request_id
    return owners


def _row_request_id(row: Mapping[str, Any], *, row_number: int) -> str:
    for key in ("request_id", "prompt_id", "request", "prompt"):
        if key not in row or row[key] in (None, ""):
            continue
        value = _parse_json_if_needed(row[key], field_name=key)
        if isinstance(value, Mapping):
            for nested_key in ("request_id", "prompt_id", "id"):
                if value.get(nested_key) not in (None, ""):
                    return _text(value[nested_key], label=f"output row {row_number} request_id")
            continue
        return _text(value, label=f"output row {row_number} request_id")
    raise ValueError(f"Worker output row {row_number} is missing request_id.")


def _observation_ids_from_value(value: Any, *, row_number: int, nested: bool = False) -> list[str]:
    value = _parse_json_if_needed(value, field_name="observation_ids")
    if isinstance(value, Mapping):
        for key in ("observation_id", "expected_observation_id", "record_id", "id"):
            if value.get(key) not in (None, ""):
                return [_text(value[key], label=f"output row {row_number} observation_id")]
        raise ValueError(f"Worker output row {row_number} has a malformed observation object.")
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Worker output row {row_number} has an empty observation list.")
        result: list[str] = []
        for item in value:
            result.extend(_observation_ids_from_value(item, row_number=row_number, nested=True))
        return result
    if nested and isinstance(value, str):
        return [_text(value, label=f"output row {row_number} observation_id")]
    return [_text(value, label=f"output row {row_number} observation_id")]


def _row_observation_ids(row: Mapping[str, Any], *, row_number: int, request_id: str) -> list[str]:
    if "observations" in row and row["observations"] not in (None, ""):
        raw_observations = _parse_json_if_needed(row["observations"], field_name="observations")
        if not isinstance(raw_observations, (list, tuple)) or not raw_observations:
            raise ValueError(f"Worker output row {row_number} has malformed observations.")
        result: list[str] = []
        for item in raw_observations:
            if isinstance(item, Mapping):
                nested_request = item.get("request_id")
                if nested_request not in (None, "") and str(nested_request).strip() != request_id:
                    raise ValueError(
                        f"Worker output row {row_number} observation belongs to a different request."
                    )
            result.extend(_observation_ids_from_value(item, row_number=row_number, nested=True))
        return result
    for key in (
        "observation_id",
        "expected_observation_id",
        "record_id",
        "observation_ids",
        "expected_observation_ids",
    ):
        if key in row and row[key] not in (None, ""):
            return _observation_ids_from_value(row[key], row_number=row_number)
    raise ValueError(
        f"Worker output row {row_number} for request {request_id!r} is missing observation_id(s)."
    )


def _output_identity_sets(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str], dict[str, str]]:
    request_ids: set[str] = set()
    observation_ids: set[str] = set()
    observation_owner: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        request_id = _row_request_id(row, row_number=row_number)
        request_ids.add(request_id)
        row_observations = _row_observation_ids(row, row_number=row_number, request_id=request_id)
        if len(set(row_observations)) != len(row_observations):
            raise ValueError(f"Worker output row {row_number} contains duplicate observation IDs.")
        for observation_id in row_observations:
            previous = observation_owner.get(observation_id)
            if previous is not None:
                raise ValueError(
                    f"Worker output contains duplicate observation ID {observation_id!r} "
                    f"for requests {previous!r} and {request_id!r}."
                )
            observation_owner[observation_id] = request_id
            observation_ids.add(observation_id)
    return request_ids, observation_ids, observation_owner


def _version_families(payload: Mapping[str, Any]) -> frozenset[str]:
    return provenance_version_families(payload)


def _canonical_identity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.loads(canonical_json(value))
    if isinstance(value, (list, tuple)):
        return [_canonical_identity_value(item) for item in value]
    return value


def _int_or_tuple(value: Any, *, label: str) -> int | tuple[int, ...]:
    value = _parse_json_if_needed(value, field_name=label)
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer or list of integers.")
    if isinstance(value, (list, tuple)):
        try:
            result = tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer or list of integers.") from exc
        if not result:
            raise ValueError(f"{label} must not be empty.")
        return tuple(sorted(result))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer or list of integers.") from exc


def _required_bool(payload: Mapping[str, Any], aliases: Sequence[str], *, label: str) -> bool:
    value = _required_value(payload, aliases, label=label)
    if not isinstance(value, bool):
        if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
            return value.strip().casefold() == "true"
        raise ValueError(f"Worker manifest {label} must be a boolean.")
    return value


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = _text(
        _required_value(manifest, ("schema_version",), label="schema_version"),
        label="schema_version",
    )
    parent_hash = _text(
        _required_value(
            manifest,
            (
                "parent_inventory_sha256",
                "parent_prompt_inventory_sha256",
                "inventory_sha256",
                "requests_sha256",
            ),
            label="parent inventory hash",
        ),
        label="parent inventory hash",
    )
    prompt_present, prompt_value = _present_value(
        manifest,
        ("prompt_hash", "prompt_inventory_sha256", "prompt_sha256", "prompt_input_sha256"),
    )
    prompt_hash = _text(prompt_value if prompt_present else parent_hash, label="prompt hash")
    model_revision = _text(
        _required_value(
            manifest,
            ("model_revision", "model_revision_id", "revision", "model_id"),
            label="model revision",
        ),
        label="model revision",
    )
    tokenizer_revision = _text(
        _required_value(
            manifest,
            ("tokenizer_revision", "tokenizer_revision_id", "tokenizer_id"),
            label="tokenizer revision",
        ),
        label="tokenizer revision",
    )
    model_present, model_value = _present_value(manifest, ("model", "model_id"))
    tokenizer_present, tokenizer_value = _present_value(manifest, ("tokenizer", "tokenizer_id"))
    dtype = _text(
        _required_value(manifest, ("dtype", "activation_dtype", "storage_dtype"), label="dtype"),
        label="dtype",
    )
    layer_value = _required_value(
        manifest,
        ("layer", "injection_layer", "layers", "activation_layer"),
        label="layer",
    )
    layer = _int_or_tuple(layer_value, label="layer")
    activation_site = _text(
        _required_value(manifest, ("activation_site", "site"), label="activation site"),
        label="activation site",
    )
    decoding_present, decoding_value = _present_value(
        manifest,
        ("decoding", "decoding_config", "generation", "generation_config", "execution"),
    )
    if not decoding_present:
        raise ValueError("Worker manifest is missing required decoding metadata.")
    token_limit_value = _required_value(
        manifest,
        ("token_limit", "max_length", "max_input_tokens", "input_token_limit"),
        label="token limit",
    )
    token_limit = _int_or_tuple(token_limit_value, label="token limit")
    prompt_format = _text(
        _required_value(manifest, ("prompt_format", "format"), label="prompt format"),
        label="prompt format",
    )
    system_present, system_value = _present_value(
        manifest,
        ("system_message", "system_prompt", "system_prompt_sha256", "system_message_sha256"),
    )
    if not system_present:
        raise ValueError("Worker manifest is missing required system message metadata.")
    system_message = _canonical_identity_value(system_value)
    run_mode = _text(
        _required_value(manifest, ("run_mode", "mode"), label="run mode"),
        label="run mode",
    )
    confirmatory = _required_bool(manifest, ("confirmatory",), label="confirmatory")
    engineering_only_value = manifest.get("engineering_only", False)
    if not isinstance(engineering_only_value, bool):
        raise ValueError("Worker manifest engineering_only must be a boolean.")
    if run_mode == "full" and not confirmatory and not engineering_only_value:
        raise ValueError(
            "Non-confirmatory full worker manifests must declare engineering_only=true."
        )
    if run_mode != "full" and engineering_only_value:
        raise ValueError("engineering_only is only valid for full worker manifests.")
    run_config_present, run_config_value = _present_value(manifest, ("run_config_hash", "config_hash"))
    run_config_hash = _text(run_config_value if run_config_present else "UNSPECIFIED", label="run config hash")
    causal_present, causal_value = _present_value(manifest, ("causal_identity", "causal"))
    causal_identity = _canonical_identity_value(causal_value) if causal_present else None
    causal_limit_present, causal_limit_value = _present_value(
        manifest,
        ("causal_token_limit", "causal_max_length", "causal_input_token_limit"),
    )
    if causal_limit_present:
        causal_token_limit = _int_or_tuple(causal_limit_value, label="causal token limit")
    elif isinstance(causal_identity, Mapping) and "token_limit" in causal_identity:
        causal_token_limit = _int_or_tuple(causal_identity["token_limit"], label="causal token limit")
    else:
        causal_token_limit = None
    version_families = _version_families(manifest)
    if len(version_families) > 1:
        raise ValueError(f"Worker manifest mixes incompatible v1/v2 markers: {sorted(version_families)}.")
    artifact_version_present, artifact_version = _present_value(
        manifest,
        ("artifact_version", "contract_version", "prompt_version", "inventory_version", "version"),
    )
    return {
        "schema_version": schema_version,
        "parent_inventory_sha256": parent_hash,
        "prompt_hash": prompt_hash,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "model": _canonical_identity_value(model_value) if model_present else model_revision,
        "tokenizer": _canonical_identity_value(tokenizer_value) if tokenizer_present else tokenizer_revision,
        "dtype": dtype,
        "layer": layer,
        "activation_site": activation_site,
        "decoding": _canonical_identity_value(decoding_value),
        "token_limit": token_limit,
        "prompt_format": prompt_format,
        "system_message": system_message,
        "run_mode": run_mode,
        "confirmatory": confirmatory,
        "engineering_only": engineering_only_value,
        "run_config_hash": run_config_hash,
        "causal_identity": causal_identity,
        "causal_token_limit": causal_token_limit,
        "version_family": next(iter(version_families), None),
        "artifact_version": _canonical_identity_value(artifact_version) if artifact_version_present else None,
    }


def _manifest_type_is_worker(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return normalized in WORKER_MANIFEST_TYPES or "worker" in normalized or "output" in normalized


def _manifest_path_for(output: Path, manifest_path: str | Path | None) -> Path:
    return Path(manifest_path) if manifest_path is not None else _default_manifest_path(output)


@dataclass(frozen=True)
class ValidatedWorkerOutput:
    output_path: Path
    manifest_path: Path
    rows: tuple[dict[str, Any], ...]
    manifest: Mapping[str, Any]
    request_ids: frozenset[str]
    observation_ids: frozenset[str]
    observation_owner: Mapping[str, str]
    identity: Mapping[str, Any]
    output_sha256: str
    manifest_sha256: str


def validate_worker_output(
    output: str | Path,
    manifest: str | Path | Mapping[str, Any] | None = None,
    *,
    allow_incomplete: bool = False,
) -> ValidatedWorkerOutput:
    """Validate one worker output and its adjacent/explicit manifest."""

    output_path = Path(output).resolve()
    rows = _read_rows(output_path)
    if isinstance(manifest, Mapping):
        manifest_value = dict(manifest)
        manifest_path = Path("<in-memory-manifest>")
        manifest_hash = canonical_hash(manifest_value)
    else:
        manifest_path = _manifest_path_for(output_path, manifest).resolve()
        manifest_value = load_json_object(manifest_path, label="worker output manifest")
        manifest_hash = file_sha256(manifest_path)
    manifest_type = manifest_value.get("manifest_type")
    if not _manifest_type_is_worker(manifest_type) or manifest_type == "benchmark_shard":
        raise ValueError(f"{manifest_path} is not a worker output manifest.")
    complete = manifest_value.get("complete")
    if not isinstance(complete, bool):
        raise ValueError(f"{manifest_path} complete must be a boolean.")
    if not complete and not allow_incomplete:
        raise ValueError(f"Refusing incomplete worker output {output_path}; complete=false.")

    request_ids, observation_ids, observation_owner = _output_identity_sets(rows)
    expected_requests = _ids_from_manifest(
        manifest_value,
        ("expected_request_ids", "owned_request_ids", "request_ids", "expected_prompt_ids"),
        label="expected request IDs",
    )
    expected_observations = _ids_from_manifest(
        manifest_value,
        (
            "expected_observation_ids",
            "owned_observation_ids",
            "observation_ids",
            "expected_record_ids",
        ),
        label="expected observation IDs",
    )
    if not expected_observations:
        present, value = _present_value(manifest_value, ("expected_observations",))
        if present:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{manifest_path} expected_observations must be a list.")
            expected_observations = _strict_ids(
                [
                    item.get("observation_id", item.get("record_id", item.get("id")))
                    if isinstance(item, Mapping)
                    else item
                    for item in value
                ],
                label="expected observation IDs",
            )
    if not expected_requests:
        raise ValueError(f"{manifest_path} is missing expected request IDs.")
    if not expected_observations:
        raise ValueError(f"{manifest_path} is missing expected observation IDs.")
    expected_request_set = set(expected_requests)
    expected_observation_set = set(expected_observations)
    if request_ids != expected_request_set:
        raise ValueError(
            f"Worker output request IDs differ from {manifest_path}; "
            f"unknown={sorted(request_ids - expected_request_set)[:5]}, "
            f"missing={sorted(expected_request_set - request_ids)[:5]}."
        )
    if observation_ids != expected_observation_set:
        raise ValueError(
            f"Worker output observation IDs differ from {manifest_path}; "
            f"unknown={sorted(observation_ids - expected_observation_set)[:5]}, "
            f"missing={sorted(expected_observation_set - observation_ids)[:5]}."
        )
    expected_observation_owner = _expected_observation_owners(
        manifest_value,
        label=str(manifest_path),
    )
    if expected_observation_owner and expected_observation_owner != observation_owner:
        mismatches = sorted(
            observation_id
            for observation_id in set(expected_observation_owner) | set(observation_owner)
            if expected_observation_owner.get(observation_id) != observation_owner.get(observation_id)
        )
        raise ValueError(
            f"Worker output observation ownership differs from {manifest_path}; "
            f"mismatched_observations={mismatches[:5]}."
        )
    for count_key, actual in (
        ("expected_request_count", len(expected_requests)),
        ("expected_observation_count", len(expected_observations)),
        ("completed_request_count", len(request_ids)),
        ("completed_observation_count", len(observation_ids)),
    ):
        if count_key in manifest_value:
            try:
                declared = int(manifest_value[count_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{manifest_path} {count_key} must be an integer.") from exc
            if declared != actual:
                raise ValueError(
                    f"{manifest_path} {count_key}={declared} does not match the validated output ({actual})."
                )
    completed_request_ids = _ids_from_manifest(
        manifest_value,
        ("completed_request_ids", "completed_prompt_ids"),
        label="completed request IDs",
    )
    if completed_request_ids and set(completed_request_ids) != request_ids:
        raise ValueError(f"{manifest_path} completed request IDs do not match the output rows.")
    completed_observation_ids = _ids_from_manifest(
        manifest_value,
        ("completed_observation_ids", "completed_record_ids"),
        label="completed observation IDs",
    )
    if completed_observation_ids and set(completed_observation_ids) != observation_ids:
        raise ValueError(f"{manifest_path} completed observation IDs do not match the output rows.")

    declared_output_hash_present, declared_output_hash = _present_value(
        manifest_value,
        ("output_sha256", "raw_output_sha256", "worker_output_sha256"),
    )
    output_hash = file_sha256(output_path)
    if declared_output_hash_present and str(declared_output_hash) != output_hash:
        raise ValueError(f"{output_path} does not match its manifest output hash.")

    identity = _manifest_identity(manifest_value)
    return ValidatedWorkerOutput(
        output_path=output_path,
        manifest_path=manifest_path,
        rows=tuple(dict(row) for row in rows),
        manifest=manifest_value,
        request_ids=frozenset(request_ids),
        observation_ids=frozenset(observation_ids),
        observation_owner=dict(observation_owner),
        identity=identity,
        output_sha256=output_hash,
        manifest_sha256=manifest_hash,
    )


def _worker_input(value: Any) -> tuple[Path, str | Path | Mapping[str, Any] | None]:
    if isinstance(value, (str, Path)):
        return Path(value), None
    if isinstance(value, Mapping):
        output = value.get("output", value.get("output_path", value.get("path")))
        if output is None:
            raise ValueError("Worker input object is missing output/output_path.")
        return Path(output), value.get("manifest", value.get("manifest_path"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return Path(value[0]), value[1]
    raise ValueError("Worker input must be an output path, (output, manifest), or mapping.")


def _parent_expected_ids(
    parent_manifest: str | Path | Mapping[str, Any] | None,
    *,
    request: bool,
) -> set[str] | None:
    if parent_manifest is None:
        return None
    payload = dict(parent_manifest) if isinstance(parent_manifest, Mapping) else load_json_object(parent_manifest, label="parent manifest")
    aliases = (
        ("expected_request_ids", "parent_request_ids", "request_ids", "owned_request_ids")
        if request
        else ("expected_observation_ids", "parent_observation_ids", "observation_ids", "owned_observation_ids")
    )
    found: set[str] = set()
    present = False
    for alias in aliases:
        if alias in payload:
            present = True
            found.update(_strict_ids(payload[alias], label=alias))
    shards = payload.get("shards")
    if isinstance(shards, list):
        for shard in shards:
            if isinstance(shard, Mapping):
                for alias in aliases:
                    if alias in shard:
                        present = True
                        found.update(_strict_ids(shard[alias], label=alias))
    if not present:
        return None
    if not found:
        raise ValueError("Parent manifest declares an empty expected identity set.")
    return found


def _coerce_id_set(value: Iterable[str] | str | Path | None, *, label: str) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if isinstance(payload, Mapping):
                for key in ("ids", label, f"expected_{label}", f"parent_{label}"):
                    if key in payload:
                        payload = payload[key]
                        break
            value = payload
        else:
            value = [part for part in str(value).split(",") if part.strip()]
    values = _strict_ids(list(value), label=label)
    return set(values)


def _manifest_construct_ids(manifest: Mapping[str, Any]) -> list[str]:
    present, value = _present_value(manifest, ("construct_ids",))
    if present:
        return list(_strict_ids(value, label="construct_ids"))
    present, value = _present_value(manifest, ("construct_id",))
    return [_text(value, label="construct_id")] if present else []


def validate_worker_outputs(
    workers: Iterable[Any],
    *,
    expected_request_ids: Iterable[str] | str | Path | None = None,
    expected_observation_ids: Iterable[str] | str | Path | None = None,
    parent_manifest: str | Path | Mapping[str, Any] | None = None,
    target_run_mode: str | None = None,
    target_confirmatory: bool | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate all worker outputs and return a composition-ready report."""

    materialized = list(workers)
    if not materialized:
        raise ValueError("At least one worker output is required.")
    validated: list[ValidatedWorkerOutput] = []
    seen_outputs: set[Path] = set()
    seen_manifests: set[Path] = set()
    for worker in materialized:
        output_path, manifest_path = _worker_input(worker)
        resolved_output = output_path.resolve()
        if resolved_output in seen_outputs:
            raise ValueError(f"Worker output is listed more than once: {resolved_output}")
        seen_outputs.add(resolved_output)
        if isinstance(manifest_path, (str, Path)) and Path(manifest_path).resolve() in seen_manifests:
            raise ValueError(f"Worker manifest is listed more than once: {manifest_path}")
        if isinstance(manifest_path, (str, Path)):
            seen_manifests.add(Path(manifest_path).resolve())
        validated.append(validate_worker_output(resolved_output, manifest_path, allow_incomplete=allow_incomplete))

    first_identity = dict(validated[0].identity)
    version_families = {item.identity.get("version_family") for item in validated if item.identity.get("version_family")}
    if len(version_families) > 1:
        raise ValueError(f"Worker outputs mix incompatible v1/v2 artifacts: {sorted(version_families)}.")
    causal_limits = {
        item.identity.get("causal_token_limit")
        for item in validated
        if item.identity.get("causal_token_limit") is not None
    }
    if len(causal_limits) > 1:
        pretty = sorted(str(value) for value in causal_limits)
        raise ValueError(
            f"Worker outputs have mixed causal token-limit identity ({', '.join(pretty)}); refusing composition."
        )
    for item in validated[1:]:
        for field_name, expected in first_identity.items():
            actual = item.identity.get(field_name)
            if actual != expected:
                if field_name == "causal_token_limit":
                    raise ValueError(
                        "Worker manifests disagree on causal token-limit identity; "
                        f"expected {expected!r}, found {actual!r}."
                    )
                if field_name == "version_family":
                    raise ValueError("Worker manifests mix v1/v2 prompt versions.")
                raise ValueError(
                    f"Worker manifests disagree on {field_name}: expected {expected!r}, found {actual!r}."
                )

    all_requests: set[str] = set()
    all_observations: set[str] = set()
    duplicate_requests: set[str] = set()
    duplicate_observations: set[str] = set()
    for item in validated:
        duplicate_requests.update(all_requests.intersection(item.request_ids))
        duplicate_observations.update(all_observations.intersection(item.observation_ids))
        all_requests.update(item.request_ids)
        all_observations.update(item.observation_ids)
    if duplicate_requests:
        raise ValueError(f"Worker outputs overlap request IDs: {sorted(duplicate_requests)[:5]}.")
    if duplicate_observations:
        raise ValueError(f"Worker outputs overlap observation IDs: {sorted(duplicate_observations)[:5]}.")

    expected_request_set = _coerce_id_set(expected_request_ids, label="request_ids")
    expected_observation_set = _coerce_id_set(expected_observation_ids, label="observation_ids")
    parent_request_set = _parent_expected_ids(parent_manifest, request=True)
    parent_observation_set = _parent_expected_ids(parent_manifest, request=False)
    expected_request_set = expected_request_set or parent_request_set
    expected_observation_set = expected_observation_set or parent_observation_set
    if expected_request_set is not None and all_requests != expected_request_set:
        raise ValueError(
            f"Composed request IDs differ from the parent; "
            f"unknown={sorted(all_requests - expected_request_set)[:5]}, "
            f"missing={sorted(expected_request_set - all_requests)[:5]}."
        )
    if expected_observation_set is not None and all_observations != expected_observation_set:
        raise ValueError(
            f"Composed observation IDs differ from the parent; "
            f"unknown={sorted(all_observations - expected_observation_set)[:5]}, "
            f"missing={sorted(expected_observation_set - all_observations)[:5]}."
        )

    run_mode = first_identity["run_mode"]
    confirmatory = bool(first_identity["confirmatory"])
    engineering_only = bool(first_identity.get("engineering_only", False))
    if target_run_mode is not None and run_mode != target_run_mode:
        if target_run_mode == "full" and run_mode == "test":
            raise ValueError(
                "Refusing test-to-confirmatory promotion: worker run_mode=test cannot become full."
            )
        raise ValueError(f"Worker run_mode={run_mode!r} does not match requested {target_run_mode!r}.")
    if target_confirmatory is True and (not confirmatory or run_mode != "full"):
        raise ValueError(
            "Refusing test-to-confirmatory promotion: all worker outputs must be full and confirmatory."
        )
    if target_confirmatory is False and confirmatory:
        raise ValueError("Refusing to relabel confirmatory worker outputs as non-confirmatory.")
    if run_mode == "test" and confirmatory:
        raise ValueError("Worker manifests mark test output as confirmatory.")
    if run_mode == "full" and not confirmatory and not engineering_only:
        raise ValueError("Non-confirmatory full worker outputs require engineering_only=true.")
    if not allow_incomplete and any(item.manifest.get("complete") is not True for item in validated):
        raise ValueError("Incomplete terminal worker inputs are not composable.")

    constructs = sorted({construct_id for item in validated for construct_id in _manifest_construct_ids(item.manifest)})
    report = {
        "schema_version": DISTRIBUTED_SCHEMA_VERSION,
        "manifest_type": "benchmark_composition_report",
        "complete": all(item.manifest.get("complete") is True for item in validated),
        "worker_count": len(validated),
        "request_ids": sorted(all_requests),
        "observation_ids": sorted(all_observations),
        "request_count": len(all_requests),
        "observation_count": len(all_observations),
        "construct_ids": constructs,
        "parent_inventory_sha256": first_identity["parent_inventory_sha256"],
        "prompt_hash": first_identity["prompt_hash"],
        "run_config_hash": first_identity["run_config_hash"],
        "run_mode": run_mode,
        "confirmatory": confirmatory,
        "engineering_only": engineering_only,
        "identity": dict(first_identity),
        "source_manifests": [
            {
                "path": str(item.manifest_path),
                "sha256": item.manifest_sha256,
                "output_path": str(item.output_path),
                "output_sha256": item.output_sha256,
                "construct_ids": _manifest_construct_ids(item.manifest),
                "request_ids": sorted(item.request_ids),
                "observation_ids": sorted(item.observation_ids),
            }
            for item in validated
        ],
        "source_manifest_sha256s": [item.manifest_sha256 for item in validated],
    }
    return {**report, "_validated_workers": validated}


def _serialize_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        return "".join(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n" for row in rows)
    if suffix == ".json":
        return json.dumps([dict(row) for row in rows], ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        raise ValueError("Cannot compose rows without fields.")
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = {
            key: canonical_json(value) if isinstance(value, (Mapping, list, tuple)) else value
            for key, value in row.items()
        }
        writer.writerow(serialized)
    return buffer.getvalue()


def compose_worker_outputs(
    workers: Iterable[Any],
    output: str | Path,
    *,
    manifest_output: str | Path | None = None,
    expected_request_ids: Iterable[str] | str | Path | None = None,
    expected_observation_ids: Iterable[str] | str | Path | None = None,
    parent_manifest: str | Path | Mapping[str, Any] | None = None,
    target_run_mode: str | None = None,
    target_confirmatory: bool | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate and atomically compose complete worker outputs."""

    report = validate_worker_outputs(
        workers,
        expected_request_ids=expected_request_ids,
        expected_observation_ids=expected_observation_ids,
        parent_manifest=parent_manifest,
        target_run_mode=target_run_mode,
        target_confirmatory=target_confirmatory,
        allow_incomplete=allow_incomplete,
    )
    if not report["complete"]:
        raise ValueError("Cannot write a complete composition from incomplete worker inputs.")
    validated: list[ValidatedWorkerOutput] = report.pop("_validated_workers")
    output_path = Path(output)
    manifest_path = Path(manifest_output) if manifest_output is not None else _default_manifest_path(output_path)
    if output_path.resolve() == manifest_path.resolve():
        raise ValueError("Composition output and manifest output must be different files.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing composition output: {output_path}")
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing composition manifest: {manifest_path}")

    rows_with_keys: list[tuple[str, str, dict[str, Any]]] = []
    for item in validated:
        for row in item.rows:
            request_id = _row_request_id(row, row_number=1)
            observation_ids = _row_observation_ids(row, row_number=1, request_id=request_id)
            rows_with_keys.append((request_id, min(observation_ids), row))
    rows_with_keys.sort(key=lambda value: (value[0], value[1], canonical_json(value[2])))
    rows = [row for _, _, row in rows_with_keys]
    output_text = _serialize_rows(rows, output_path)
    atomic_write_text(output_path, output_text, label="composition output")
    output_hash = file_sha256(output_path)

    composition_manifest: dict[str, Any] = {
        "schema_version": DISTRIBUTED_SCHEMA_VERSION,
        "manifest_type": COMPOSITION_MANIFEST_TYPE,
        "immutable": True,
        "complete": True,
        "output_path": str(output_path.resolve()),
        "output_sha256": output_hash,
        "manifest_path": str(manifest_path.resolve()),
        "parent_inventory_sha256": report["parent_inventory_sha256"],
        "prompt_hash": report["prompt_hash"],
        "run_config_hash": report["run_config_hash"],
        "run_mode": report["run_mode"],
        "confirmatory": report["confirmatory"],
        "engineering_only": report["engineering_only"],
        "construct_ids": report["construct_ids"],
        "expected_request_ids": report["request_ids"],
        "expected_observation_ids": report["observation_ids"],
        "expected_request_count": report["request_count"],
        "expected_observation_count": report["observation_count"],
        "completed_request_ids": report["request_ids"],
        "completed_observation_ids": report["observation_ids"],
        "completed_request_count": report["request_count"],
        "completed_observation_count": report["observation_count"],
        "identity": report["identity"],
        "source_manifests": report["source_manifests"],
        "source_manifest_sha256s": report["source_manifest_sha256s"],
        "composition_algorithm": "sorted-request-observation-rows-v1",
    }
    composition_manifest["composition_sha256"] = canonical_hash(composition_manifest)
    atomic_write_json(manifest_path, composition_manifest, label="composition manifest")
    return {
        **report,
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "output_sha256": output_hash,
        "composition_sha256": composition_manifest["composition_sha256"],
    }


def compose_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for :func:`compose_worker_outputs`."""

    return compose_worker_outputs(*args, **kwargs)


def compose_shards(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Short alias for callers that refer to the physical shard stage."""

    return compose_worker_outputs(*args, **kwargs)


def validate_composition_inputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for validating worker inputs without writing a composition."""

    return validate_worker_outputs(*args, **kwargs)


__all__ = [
    "COMPOSITION_MANIFEST_TYPE",
    "ValidatedWorkerOutput",
    "compose_outputs",
    "compose_shards",
    "compose_worker_outputs",
    "validate_composition_inputs",
    "validate_worker_output",
    "validate_worker_outputs",
]
