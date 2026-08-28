"""Real model-side worker for one construct-pure GPU shard.

The worker is launched by :class:`ParallelExecutor` as one process per
construct-pure shard. Residual logging is implemented directly so model
loading happens once per process and CPU/persistent-storage writes happen
after every batch. Readout, steering, and causal stages are separate,
resumable stage invocations whose commands are explicitly supplied by the
run configuration; they never silently reuse another stage's artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_json, load_run_config
from .parallel_executor import (
    AdapterError,
    InvalidCheckpointError,
    _worker_payload,
    _write_jsonl_row,
    _write_worker_manifest,
    load_shard_manifest,
    load_worker_manifest,
    read_output_progress,
)


GPU_WORKER_SCHEMA_VERSION = "gpu_worker_v1"
GPU_STAGES = frozenset({"residual_logging", "readout", "steering", "causal_interchange"})
_STAGE_ALIASES = {"residual": "residual_logging", "causal": "causal_interchange"}
B300_GPU_TYPE = "NVIDIA B300"


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(dict(payload), handle, indent=2, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _canonical_stage(stage: str) -> str:
    value = str(stage).strip()
    value = _STAGE_ALIASES.get(value, value)
    if value not in GPU_STAGES:
        raise AdapterError(f"GPU worker stage must be one of {sorted(GPU_STAGES)}")
    return value


def _safe_relative(path: Path, root: Path, *, label: str) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise AdapterError(f"{label} must remain inside the worker-owned output directory") from exc


def _validate_output_ownership(
    *,
    output: Path,
    worker_manifest_path: Path,
    worker_id: str,
) -> None:
    if output.is_symlink() or worker_manifest_path.is_symlink():
        raise AdapterError("GPU worker refuses symlinked output or checkpoint paths")
    output = output.resolve()
    manifest = worker_manifest_path.resolve()
    if output.name != "output.jsonl" or manifest.name != "worker_manifest.json":
        raise AdapterError("GPU worker requires the executor's canonical output/checkpoint filenames")
    if output.parent != manifest.parent or output.parent.name != worker_id:
        raise AdapterError("GPU worker output and checkpoint must share the worker-owned directory")


def _construct_id(shard: Mapping[str, Any]) -> str:
    constructs = shard.get("construct_ids")
    if not isinstance(constructs, list) or len(constructs) != 1 or not str(constructs[0]).strip():
        raise AdapterError(
            "GPU workers require exactly one construct-pure shard; mixed or missing construct IDs are refused."
        )
    return str(constructs[0]).strip()


def _validate_version_family(shard: Mapping[str, Any]) -> None:
    families: set[str] = set()
    for request in shard.get("requests", []):
        if not isinstance(request, Mapping):
            continue
        for key in ("version_family", "prompt_version", "inventory_version", "version"):
            value = request.get(key)
            if value not in (None, ""):
                families.add(str(value).strip())
        metadata = request.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("version_family", "prompt_version", "inventory_version", "version"):
                value = metadata.get(key)
                if value not in (None, ""):
                    families.add(str(value).strip())
    declared = shard.get("version_families")
    if isinstance(declared, list):
        families.update(str(value).strip() for value in declared if str(value).strip())
    if len(families) > 1:
        raise AdapterError(f"GPU worker refuses mixed prompt/request version families: {sorted(families)}")


def _parallel_config(run_config_payload: Mapping[str, Any]) -> dict[str, Any]:
    execution = run_config_payload.get("execution")
    if not isinstance(execution, Mapping):
        return {}
    value = execution.get("parallel_executor")
    return dict(value) if isinstance(value, Mapping) else {}


def _gpu_config(run_config_payload: Mapping[str, Any]) -> dict[str, Any]:
    parallel = _parallel_config(run_config_payload)
    value = parallel.get("gpu_worker")
    return dict(value) if isinstance(value, Mapping) else {}


def _stage_config(run_config_payload: Mapping[str, Any], stage: str) -> dict[str, Any]:
    gpu_config = _gpu_config(run_config_payload)
    stages = gpu_config.get("stages")
    if not isinstance(stages, Mapping):
        return {}
    value = stages.get(stage)
    return dict(value) if isinstance(value, Mapping) else {}


def _request_prompt(request: Mapping[str, Any]) -> str:
    for key in ("prompt_text", "prompt", "text", "input"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise AdapterError(f"Request {request.get('request_id')!r} has no canonical prompt_text")


def _activation_record(request: Mapping[str, Any], construct_id: str) -> Any:
    from activation_analysis.log_residuals import PromptRecord as ActivationPromptRecord

    request_id = str(request["request_id"])
    metadata_value = request.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    for key in (
        "construct_id",
        "split",
        "prompt_role",
        "pair_id",
        "pair_role",
        "condition_id",
        "task_id",
        "prompt_version",
        "inventory_version",
        "version",
    ):
        if key in request and request[key] not in (None, ""):
            metadata.setdefault(key, request[key])
    metadata["construct_id"] = construct_id
    return ActivationPromptRecord(
        prompt_id=request_id,
        prompt_text=_request_prompt(request),
        metadata=metadata,
    )


def _cuda_telemetry(
    logger: Any,
    *,
    replica_count: int,
    processed_prompts: int,
    elapsed_seconds: float | None = None,
    loaded_model_vram_gb: float | None = None,
) -> dict[str, Any] | None:
    torch = getattr(logger, "_torch", None)
    if torch is None or not torch.cuda.is_available():
        return None
    device = getattr(logger, "device", "cuda")
    try:
        device_index = torch.cuda.current_device() if str(device).startswith("cuda") else 0
        properties = torch.cuda.get_device_properties(device_index)
        elapsed = None if elapsed_seconds is None else max(0.0, float(elapsed_seconds))
        throughput = None
        if elapsed is not None and elapsed > 0:
            throughput = float(processed_prompts) / elapsed
        reserved_vram_gb = float(torch.cuda.memory_reserved(device_index)) / (1024**3)
        peak_reserved_vram_gb = float(torch.cuda.max_memory_reserved(device_index)) / (1024**3)
        return {
            "replica_count": replica_count,
            "processed_prompts": processed_prompts,
            "device_index": int(device_index),
            "device_name": str(properties.name),
            "total_vram_gb": float(properties.total_memory) / (1024**3),
            "allocated_vram_gb": float(torch.cuda.memory_allocated(device_index)) / (1024**3),
            "reserved_vram_gb": reserved_vram_gb,
            "loaded_model_vram_gb": (
                reserved_vram_gb if loaded_model_vram_gb is None else float(loaded_model_vram_gb)
            ),
            "peak_allocated_vram_gb": float(torch.cuda.max_memory_allocated(device_index)) / (1024**3),
            "peak_reserved_vram_gb": peak_reserved_vram_gb,
            "peak_vram_gb": peak_reserved_vram_gb,
            "elapsed_seconds": elapsed,
            "throughput_items_per_second": throughput,
        }
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _runtime_compatibility_preflight(*, device: str, torch_module: Any | None = None) -> dict[str, Any]:
    """Verify the visible runtime is one exact B300 before model loading.

    The controller verifies the provider-side SKU. This second check verifies
    what the worker can actually see inside the container, before
    ``ResidualStreamLogger`` constructs a tokenizer or downloads a model.
    """

    if torch_module is None:
        try:
            import torch as torch_module
        except Exception as exc:  # pragma: no cover - environment-specific
            raise AdapterError("GPU worker requires torch for the B300 runtime preflight") from exc
    requested_device = str(device).strip().lower()
    if requested_device == "cuda":
        device_index = 0
    elif requested_device.startswith("cuda:"):
        try:
            device_index = int(requested_device.split(":", 1)[1])
        except (TypeError, ValueError) as exc:
            raise AdapterError("GPU worker device must be 'cuda' or 'cuda:0'") from exc
    else:
        raise AdapterError("GPU worker residual logging requires a CUDA device ('cuda' or 'cuda:0')")
    if device_index != 0:
        raise AdapterError("GPU worker requires the single visible B300 at cuda:0")
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise AdapterError("GPU worker cannot start: CUDA is unavailable")
    try:
        visible_gpu_count = int(cuda.device_count())
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise AdapterError("GPU worker could not determine the visible CUDA device count") from exc
    if visible_gpu_count != 1:
        raise AdapterError(
            f"GPU worker requires exactly one visible CUDA GPU for the single-pod topology; found {visible_gpu_count}"
        )
    try:
        properties = cuda.get_device_properties(device_index)
        device_name = str(properties.name)
        total_vram_gb = float(properties.total_memory) / (1024**3)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise AdapterError("GPU worker could not inspect the visible CUDA GPU") from exc
    if "B300" not in device_name.upper():
        raise AdapterError(
            f"GPU worker saw runtime GPU {device_name!r}; refusing anything other than {B300_GPU_TYPE!r}"
        )
    return {
        "expected_gpu_type": B300_GPU_TYPE,
        "device_name": device_name,
        "device_index": device_index,
        "visible_gpu_count": visible_gpu_count,
        "total_vram_gb": total_vram_gb,
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_version": str(getattr(getattr(torch_module, "version", None), "cuda", "unknown")),
    }


def _stage_manifest(
    *,
    path: Path,
    stage: str,
    shard: Mapping[str, Any],
    worker_manifest: Mapping[str, Any],
    run_config_payload: Mapping[str, Any],
    artifact_root: Path,
    complete: bool,
    processed_request_ids: Sequence[str],
    telemetry: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model = run_config_payload.get("model")
    model = dict(model) if isinstance(model, Mapping) else {}
    payload: dict[str, Any] = {
        "schema_version": GPU_WORKER_SCHEMA_VERSION,
        "manifest_type": "gpu_worker_stage",
        "stage": stage,
        "complete": complete,
        "construct_id": _construct_id(shard),
        "shard_id": str(shard["shard_id"]),
        "worker_id": str(worker_manifest["worker_id"]),
        "shard_manifest_sha256": worker_manifest.get("execution_identity", {}).get("shard_manifest_sha256"),
        "run_config_hash": worker_manifest.get("execution_identity", {}).get("run_config_hash"),
        "run_mode": worker_manifest.get("execution_identity", {}).get("run_mode"),
        "confirmatory": worker_manifest.get("execution_identity", {}).get("confirmatory"),
        "model": model,
        "activation": dict(run_config_payload.get("activation", {}))
        if isinstance(run_config_payload.get("activation"), Mapping)
        else {},
        "artifact_root": str(artifact_root),
        "processed_request_ids": list(processed_request_ids),
        "created_at_epoch": time.time(),
        "inference_mode": True,
        "use_cache": False,
    }
    if telemetry is not None:
        payload["gpu_telemetry"] = dict(telemetry)
    if extra:
        payload.update(dict(extra))
    _atomic_replace_json(path, payload)
    return payload


def _load_existing_stage_manifest(
    path: Path,
    *,
    stage: str,
    construct_id: str,
    run_config_hash: str | None,
) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidCheckpointError(f"GPU stage manifest is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise InvalidCheckpointError(f"GPU stage manifest must be an object: {path}")
    if payload.get("stage") != stage or payload.get("construct_id") != construct_id:
        raise InvalidCheckpointError("GPU stage manifest does not match the immutable shard stage")
    if run_config_hash is not None and payload.get("run_config_hash") != run_config_hash:
        raise InvalidCheckpointError("GPU stage manifest has a different run-config hash")
    raw_shards = payload.get("shards", [])
    if not isinstance(raw_shards, list) or not all(isinstance(item, Mapping) for item in raw_shards):
        raise InvalidCheckpointError("GPU stage manifest shards must be a list of objects")
    shards = [dict(item) for item in raw_shards]
    batch_indices: list[int] = []
    for item in shards:
        tensor_file = item.get("tensor_file")
        if isinstance(tensor_file, str):
            match = re.search(r"batch_(\d+)\.npy$", tensor_file)
            if match:
                batch_indices.append(int(match.group(1)))
    return shards, (max(batch_indices) + 1 if batch_indices else 0)


def _heartbeat(
    *,
    worker_manifest_path: Path,
    output: Path,
    shard: Mapping[str, Any],
    manifest: Mapping[str, Any],
    stage: str,
    completed_requests: set[str],
    completed_observations: set[str],
    status: str = "running",
    error: str | None = None,
) -> None:
    payload = _worker_payload(
        worker_id=str(manifest["worker_id"]),
        shard=shard,
        status=status,
        output_path=output,
        worker_manifest_path=worker_manifest_path,
        stage=stage,
        retry_count=int(manifest.get("retry_count", 0)),
        execution_identity=manifest.get("execution_identity", {}),
        now=time.time(),
        completed_request_ids=completed_requests,
        completed_observation_ids=completed_observations,
        error=error,
        terminal_reason=None,
        pid=os.getpid(),
    )
    _write_worker_manifest(worker_manifest_path, payload)


def _run_residual_logging(
    *,
    shard: Mapping[str, Any],
    worker_manifest: Mapping[str, Any],
    worker_manifest_path: Path,
    output: Path,
    run_config_path: Path,
    run_config_payload: Mapping[str, Any],
    stage: str,
) -> None:
    from activation_analysis.log_residuals import _with_token_regions, _write_batch
    from activation_analysis.residual_streams import ResidualStreamLogger

    construct_id = _construct_id(shard)
    run_config = load_run_config(run_config_path)
    model = dict(run_config.model)
    revision = model.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise AdapterError("GPU residual logging requires an exact pinned model revision")
    activation = dict(run_config.activation)
    gpu_config = _gpu_config(run_config_payload)
    local_files_only = bool(gpu_config.get("local_files_only", False))
    trust_remote_code = bool(gpu_config.get("trust_remote_code", False))
    device = str(gpu_config.get("device", "cuda"))
    dtype = str(gpu_config.get("dtype", "bf16"))
    device_map_value = gpu_config.get("device_map")
    device_map = str(device_map_value) if device_map_value not in (None, "") else None
    attn_implementation_value = gpu_config.get("attn_implementation")
    attn_implementation = str(attn_implementation_value) if attn_implementation_value not in (None, "") else None
    block_path_value = gpu_config.get("block_path")
    block_path = str(block_path_value) if block_path_value not in (None, "") else None
    records = [
        _with_token_regions(_activation_record(request, construct_id), str(activation.get("token_region_strategy", "auto")))
        for request in shard["requests"]
    ]
    request_map = {str(request["request_id"]): request for request in shard["requests"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=True)
    progress = read_output_progress(output, shard)
    completed_requests = set(progress.completed_request_ids)
    completed_observations = set(progress.completed_observation_ids)
    artifact_root = output.parent / "artifacts" / "residual_logging"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_manifest_path = artifact_root / "manifest.json"
    run_config_hash = worker_manifest.get("execution_identity", {}).get("run_config_hash")
    existing_shards, next_batch_index = _load_existing_stage_manifest(
        artifact_manifest_path,
        stage=stage,
        construct_id=construct_id,
        run_config_hash=run_config_hash,
    )
    _heartbeat(
        worker_manifest_path=worker_manifest_path,
        output=output,
        shard=shard,
        manifest=worker_manifest,
        stage=stage,
        completed_requests=completed_requests,
        completed_observations=completed_observations,
    )
    if len(completed_requests) == len(shard["request_ids"]):
        if not artifact_manifest_path.is_file():
            raise InvalidCheckpointError("Completed residual output has no residual artifact manifest")
        return

    pending_records = [record for record in records if record.prompt_id not in completed_requests]
    batch_size = int(activation.get("batch_size", 1))
    if batch_size < 1:
        raise AdapterError("activation.batch_size must be positive")
    layers = [int(layer) for layer in activation.get("layers", [])]
    include_regions_raw = activation.get("include_token_regions")
    include_regions = set(str(item) for item in include_regions_raw) if isinstance(include_regions_raw, list) else None
    runtime_compatibility = _runtime_compatibility_preflight(device=device)
    logger = ResidualStreamLogger(
        model["model_id"],
        tokenizer_id=str(model.get("tokenizer_id", model["model_id"])),
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device=device,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        block_path=block_path,
    )
    torch = getattr(logger, "_torch", None)
    loaded_model_vram_gb: float | None = None
    if torch is not None and torch.cuda.is_available():
        try:
            loaded_model_vram_gb = float(torch.cuda.memory_reserved(0)) / (1024**3)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            loaded_model_vram_gb = None
        torch.cuda.reset_peak_memory_stats()
    shard_records: list[dict[str, Any]] = list(existing_shards)
    telemetry: dict[str, Any] | None = None
    inference_started = time.monotonic()
    processed_this_run = 0
    for start in range(0, len(pending_records), batch_size):
        batch_index = next_batch_index
        next_batch_index += 1
        batch_records = pending_records[start : start + batch_size]
        _heartbeat(
            worker_manifest_path=worker_manifest_path,
            output=output,
            shard=shard,
            manifest=worker_manifest,
            stage=stage,
            completed_requests=completed_requests,
            completed_observations=completed_observations,
        )
        batch = logger.extract_batch(
            [record.prompt_text for record in batch_records],
            [record.prompt_id for record in batch_records],
            layers,
            max_length=int(activation.get("max_length", 512)),
            token_mode=str(activation.get("token_mode", "nonpad")),
            activation_site=str(activation.get("activation_site", "resid_post")),
            token_region_spans=[record.metadata.get("prompt_regions", []) for record in batch_records],
        )
        batch_shards = _write_batch(
            artifact_root,
            batch_index,
            batch_records,
            batch,
            layers,
            storage_dtype=str(activation.get("storage_dtype", "float16")),
            include_token_regions=include_regions,
        )
        shard_records.extend(batch_shards)
        processed_this_run += len(batch_records)
        for record in batch_records:
            request_id = record.prompt_id
            request = request_map[request_id]
            observations = [str(item) for item in request["observation_ids"]]
            _write_jsonl_row(
                output,
                {
                    "request_id": request_id,
                    "observation_ids": observations,
                    "worker_id": str(worker_manifest["worker_id"]),
                    "stage": stage,
                    "artifact_manifest": str(artifact_manifest_path),
                    "output": {"kind": "residual_logging", "construct_id": construct_id},
                },
            )
            completed_requests.add(request_id)
            completed_observations.update(observations)
        telemetry = _cuda_telemetry(
            logger,
            replica_count=int(worker_manifest.get("execution_identity", {}).get("replica_count", 1)),
            processed_prompts=processed_this_run,
            elapsed_seconds=time.monotonic() - inference_started,
            loaded_model_vram_gb=loaded_model_vram_gb,
        )
        _stage_manifest(
            path=artifact_manifest_path,
            stage=stage,
            shard=shard,
            worker_manifest=worker_manifest,
            run_config_payload=run_config_payload,
            artifact_root=artifact_root,
            complete=False,
            processed_request_ids=sorted(completed_requests),
            telemetry=telemetry,
            extra={"runtime_compatibility": runtime_compatibility, "shards": list(shard_records)},
        )
        _heartbeat(
            worker_manifest_path=worker_manifest_path,
            output=output,
            shard=shard,
            manifest=worker_manifest,
            stage=stage,
            completed_requests=completed_requests,
            completed_observations=completed_observations,
        )
    _stage_manifest(
        path=artifact_manifest_path,
        stage=stage,
        shard=shard,
        worker_manifest=worker_manifest,
        run_config_payload=run_config_payload,
        artifact_root=artifact_root,
        complete=True,
        processed_request_ids=sorted(completed_requests),
        telemetry=telemetry,
        extra={
            "runtime_compatibility": runtime_compatibility,
            "shards": list(shard_records),
            "activation_manifest_path": str(artifact_manifest_path),
        },
    )


def _stage_command(
    *,
    stage: str,
    stage_config: Mapping[str, Any],
    shard: Mapping[str, Any],
    worker_manifest_path: Path,
    output: Path,
    artifact_root: Path,
    run_config_path: Path,
) -> list[str]:
    configured = stage_config.get("command")
    if configured is None:
        raise AdapterError(
            f"GPU stage {stage!r} requires an explicit stage command in execution.parallel_executor.gpu_worker.stages"
        )
    if isinstance(configured, str):
        import shlex

        command = shlex.split(configured, posix=(os.name != "nt"))
    elif isinstance(configured, Sequence) and not isinstance(configured, (bytes, bytearray)):
        command = [str(item) for item in configured]
    else:
        raise AdapterError(f"GPU stage {stage!r} command must be an argv list or string")
    if not command:
        raise AdapterError(f"GPU stage {stage!r} command must not be empty")
    construct_id = _construct_id(shard)
    tokens = {
        "stage": stage,
        "construct_id": construct_id,
        "shard_manifest": str(Path(str(shard["manifest_path"])).resolve())
        if shard.get("manifest_path")
        else "",
        "worker_manifest": str(worker_manifest_path),
        "output_path": str(output),
        "artifact_dir": str(artifact_root),
        "run_config": str(run_config_path),
    }
    rendered: list[str] = []
    for argument in command:
        value = str(argument)
        for name, replacement in tokens.items():
            value = value.replace("{" + name + "}", replacement)
        rendered.append(value)
    return rendered


def _run_external_stage(
    *,
    stage: str,
    shard: Mapping[str, Any],
    worker_manifest: Mapping[str, Any],
    worker_manifest_path: Path,
    output: Path,
    run_config_path: Path,
    run_config_payload: Mapping[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=True)
    progress = read_output_progress(output, shard)
    completed_requests = set(progress.completed_request_ids)
    completed_observations = set(progress.completed_observation_ids)
    artifact_root = output.parent / "artifacts" / stage
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_manifest_path = artifact_root / "manifest.json"
    if len(completed_requests) == len(shard["request_ids"]):
        if not artifact_manifest_path.is_file():
            raise InvalidCheckpointError(f"Completed {stage} output has no stage artifact manifest")
        return
    stage_config = _stage_config(run_config_payload, stage)
    prerequisite = stage_config.get("prerequisite_manifest")
    if not isinstance(prerequisite, str) or not prerequisite.strip() or not Path(prerequisite).is_file():
        raise AdapterError(f"GPU stage {stage!r} requires an existing prerequisite_manifest")
    _heartbeat(
        worker_manifest_path=worker_manifest_path,
        output=output,
        shard=shard,
        manifest=worker_manifest,
        stage=stage,
        completed_requests=completed_requests,
        completed_observations=completed_observations,
    )
    command = _stage_command(
        stage=stage,
        stage_config=stage_config,
        shard=shard,
        worker_manifest_path=worker_manifest_path,
        output=output,
        artifact_root=artifact_root,
        run_config_path=run_config_path,
    )
    subprocess.run(command, check=True, shell=False)
    for request in shard["requests"]:
        request_id = str(request["request_id"])
        observations = [str(item) for item in request["observation_ids"]]
        if request_id in completed_requests:
            continue
        _write_jsonl_row(
            output,
            {
                "request_id": request_id,
                "observation_ids": observations,
                "worker_id": str(worker_manifest["worker_id"]),
                "stage": stage,
                "artifact_manifest": str(artifact_manifest_path),
                "output": {"kind": stage, "construct_id": _construct_id(shard)},
            },
        )
        completed_requests.add(request_id)
        completed_observations.update(observations)
    _stage_manifest(
        path=artifact_manifest_path,
        stage=stage,
        shard=shard,
        worker_manifest=worker_manifest,
        run_config_payload=run_config_payload,
        artifact_root=artifact_root,
        complete=True,
        processed_request_ids=sorted(completed_requests),
        extra={"prerequisite_manifest": str(Path(prerequisite).resolve())},
    )


def run_gpu_worker(
    *,
    shard_manifest_path: str | Path,
    worker_manifest_path: str | Path,
    output_path: str | Path,
    stage: str,
    run_config_path: str | Path,
) -> int:
    shard_path = Path(shard_manifest_path).resolve()
    worker_path = Path(worker_manifest_path).resolve()
    output = Path(output_path).resolve()
    config_path = Path(run_config_path).resolve()
    canonical_stage = _canonical_stage(stage)
    try:
        shard = load_shard_manifest(shard_path)
        _construct_id(shard)
        _validate_version_family(shard)
        worker_manifest = load_worker_manifest(worker_path, shard=shard, expected_output_path=output)
        worker_id = str(worker_manifest["worker_id"])
        _validate_output_ownership(output=output, worker_manifest_path=worker_path, worker_id=worker_id)
        if worker_manifest["stage"] != stage and worker_manifest["stage"] != canonical_stage:
            raise InvalidCheckpointError("GPU worker stage does not match the immutable worker manifest")
        if not config_path.is_file():
            raise AdapterError(f"GPU worker run config does not exist: {config_path}")
        run_config_payload = load_json(config_path)
        run_config = load_run_config(config_path)
        expected_hash = worker_manifest.get("execution_identity", {}).get("run_config_hash")
        if expected_hash and expected_hash != _canonical_hash(run_config_payload):
            raise InvalidCheckpointError("GPU worker run config hash differs from the immutable worker identity")
        identity = worker_manifest.get("execution_identity", {})
        if isinstance(identity, Mapping):
            for field in ("run_mode", "confirmatory", "stage"):
                if field in identity and field in shard and identity[field] != shard[field]:
                    raise InvalidCheckpointError(f"GPU worker identity mismatch for {field}")
        if str(run_config.run_id).strip() == "":
            raise AdapterError("GPU worker run config has no run ID")
        if canonical_stage == "residual_logging":
            _run_residual_logging(
                shard=shard,
                worker_manifest=worker_manifest,
                worker_manifest_path=worker_path,
                output=output,
                run_config_path=config_path,
                run_config_payload=run_config_payload,
                stage=canonical_stage,
            )
        else:
            _run_external_stage(
                stage=canonical_stage,
                shard=shard,
                worker_manifest=worker_manifest,
                worker_manifest_path=worker_path,
                output=output,
                run_config_path=config_path,
                run_config_payload=run_config_payload,
            )
        progress = read_output_progress(output, shard)
        if progress.completed_request_ids != frozenset(shard["request_ids"]):
            raise InvalidCheckpointError("GPU stage returned before every request was durably acknowledged")
        _heartbeat(
            worker_manifest_path=worker_path,
            output=output,
            shard=shard,
            manifest=worker_manifest,
            stage=stage,
            completed_requests=set(progress.completed_request_ids),
            completed_observations=set(progress.completed_observation_ids),
            status="complete",
        )
        return 0
    except Exception as exc:
        try:
            shard = load_shard_manifest(shard_path)
            previous = json.loads(worker_path.read_text(encoding="utf-8"))
            if not isinstance(previous, Mapping):
                previous = {}
            progress = read_output_progress(output, shard)
            failed = _worker_payload(
                worker_id=str(previous.get("worker_id", shard.get("worker_id", "worker"))),
                shard=shard,
                status="failed",
                output_path=output,
                worker_manifest_path=worker_path,
                stage=stage,
                retry_count=int(previous.get("retry_count", 0)),
                execution_identity=previous.get("execution_identity", {}),
                now=time.time(),
                completed_request_ids=progress.completed_request_ids,
                completed_observation_ids=progress.completed_observation_ids,
                error=f"{type(exc).__name__}: {exc}",
                terminal_reason="worker_exception",
                pid=os.getpid(),
            )
            _write_worker_manifest(worker_path, failed)
        except Exception:
            pass
        print(f"gpu worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    from .distributed_contracts import canonical_hash

    return canonical_hash(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--worker-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_gpu_worker(
        shard_manifest_path=args.shard_manifest,
        worker_manifest_path=args.worker_manifest,
        output_path=args.output,
        stage=args.stage,
        run_config_path=args.run_config,
    )


__all__ = [
    "B300_GPU_TYPE",
    "GPU_STAGES",
    "GPU_WORKER_SCHEMA_VERSION",
    "_runtime_compatibility_preflight",
    "main",
    "run_gpu_worker",
]


if __name__ == "__main__":
    raise SystemExit(main())
