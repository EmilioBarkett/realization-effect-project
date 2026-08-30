#!/usr/bin/env python3
"""Stage a model-specific, train-only steering bundle for the preflight.

The model-side behavior/accessibility preflight intentionally does not log
representations.  It therefore consumes a *separately produced* readout and
steering bundle.  This command is the boundary between that model-side B/R
run and the no-representation-logging preflight: it only verifies and copies
already-produced artifacts.  It never creates a direction, chooses a layer,
or calls a model/API.

The source manifest is deliberately explicit and hash-bound.  A minimal
manifest looks like::

    {
      "schema_version": "0.1.0",
      "manifest_type": "model_steering_preflight_source_bundle",
      "source_root": "/workspace/steering_source",
      "model": {"alias": "qwen", "model_id": "...", "revision": "..."},
      "run_config": {"path": "run.json", "sha256": "...",
                     "run_config_hash": "..."},
      "prompt_inventory": {"path": "combined.csv", "sha256": "..."},
      "construct_specs": [
        {"construct_id": "...", "path": "spec.json", "sha256": "...",
         "construct_spec_hash": "..."}
      ],
      "plans": [
        {"construct_id": "...", "path": "plans/construct.json",
         "sha256": "..."}
      ]
    }

Plan direction/control paths may be absolute paths from the producing run or
relative to ``source_root``.  Every reference must resolve under that root,
be a regular non-symlink ``.npy`` file, and match the plan's frozen hashes.
Staged plan references are rewritten to absolute paths below the output root
so the existing steering runner can consume them without relying on its
current working directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from construct_benchmark.config import load_construct_spec, load_run_config  # noqa: E402
from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402


SCHEMA_VERSION = "0.1.0"
MANIFEST_TYPE = "model_steering_preflight_source_bundle"
STAGED_MANIFEST_TYPE = "model_steering_preflight_staged_bundle"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_PATH_KEYS = ("target", "shuffled", "random")


class BundleValidationError(ValueError):
    """Raised when a source bundle is incomplete or provenance-inconsistent."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BundleValidationError(f"{label} must contain a JSON object.")
    return payload


def _declared_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip()):
        raise BundleValidationError(f"{label} must be a 64-character SHA-256 digest.")
    return value.strip().lower()


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise BundleValidationError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve(strict=False)
    if not resolved.is_file():
        raise BundleValidationError(f"{label} must be an existing regular file: {path}")
    return resolved


def _resolve_declared_path(raw: Any, *, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise BundleValidationError(f"{label} must be a non-empty path.")
    value = Path(raw.strip())
    if "\x00" in raw:
        raise BundleValidationError(f"{label} contains a NUL byte.")
    return _regular_file(value if value.is_absolute() else base / value, label=label)


def _resolve_under_root(raw: Any, *, root: Path, label: str) -> Path:
    if isinstance(raw, str) and any(
        part == ".." for part in raw.replace("\\", "/").split("/")
    ):
        raise BundleValidationError(f"{label} contains path traversal: {raw!r}")
    candidate = _resolve_declared_path(raw, base=root, label=label)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BundleValidationError(
            f"{label} resolves outside source_root {root}: {candidate}"
        ) from exc
    return candidate


def _same_hash(path: Path, declared: Any, *, label: str) -> str:
    expected = _declared_hash(declared, label=f"{label}.sha256")
    actual = file_sha256(path)
    if actual != expected:
        raise BundleValidationError(
            f"{label} SHA-256 mismatch: declared {expected}, actual {actual}"
        )
    return actual


def _safe_component(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT_RE.fullmatch(value):
        raise BundleValidationError(f"{label} is not a path-safe identifier: {value!r}")
    return value


def _model_metadata(manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = manifest.get("model")
    if not isinstance(raw, Mapping):
        raise BundleValidationError("model must be an object with alias, model_id, and revision.")
    metadata = {
        "alias": _safe_component(raw.get("alias"), label="model.alias"),
        "model_id": str(raw.get("model_id") or ""),
        "revision": str(raw.get("revision") or ""),
    }
    if not metadata["model_id"] or not metadata["revision"]:
        raise BundleValidationError("model.model_id and model.revision must be non-empty.")
    return metadata


def _array(path: Path, *, label: str) -> tuple[str, tuple[int, ...]]:
    if path.suffix.lower() != ".npy":
        raise BundleValidationError(f"{label} must be a .npy direction/control artifact: {path}")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise BundleValidationError(f"{label} is not a readable NumPy array: {exc}") from exc
    if not isinstance(value, np.ndarray) or value.ndim != 1 or value.size < 1:
        raise BundleValidationError(f"{label} must be a non-empty one-dimensional array.")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise BundleValidationError(f"{label} must contain finite numeric values.")
    if not np.any(value != 0):
        raise BundleValidationError(f"{label} must not be an all-zero direction/control.")
    return file_sha256(path), tuple(int(dimension) for dimension in value.shape)


def _plan_artifact_paths(plan: Mapping[str, Any], *, plan_label: str) -> list[tuple[str, str]]:
    """Return each direction/control path exactly once, preserving role labels."""

    paths = plan.get("direction_paths")
    if not isinstance(paths, Mapping):
        raise BundleValidationError(f"{plan_label}.direction_paths is missing.")
    for key in _PATH_KEYS:
        if key not in paths:
            raise BundleValidationError(f"{plan_label}.direction_paths.{key} is missing.")
    result: list[tuple[str, str]] = []
    result.append(("direction_paths.target", str(paths["target"])))
    result.append(("direction_paths.shuffled", str(paths["shuffled"])))
    random = paths["random"]
    if not isinstance(random, list) or not random:
        raise BundleValidationError(f"{plan_label}.direction_paths.random must be a non-empty list.")
    result.extend((f"direction_paths.random[{index}]", str(value)) for index, value in enumerate(random))

    tracking = plan.get("tracking_directions")
    if not isinstance(tracking, Mapping) or not tracking:
        raise BundleValidationError(f"{plan_label}.tracking_directions is missing.")
    for layer, entry in sorted(tracking.items(), key=lambda item: str(item[0])):
        if not isinstance(entry, Mapping):
            raise BundleValidationError(f"{plan_label}.tracking_directions[{layer!r}] is malformed.")
        result.append((f"tracking_directions[{layer}].path", str(entry.get("path", ""))))
    return result


def _plan_provenance(
    plan: Mapping[str, Any],
    *,
    config_hash: str,
    inventory_hash: str,
    spec_hash: str,
    model: Mapping[str, str],
    construct_id: str,
    run_id: str,
    plan_label: str,
    source_root: Path,
    expected_random_count: int,
    registered_layers: set[int],
) -> tuple[dict[str, Path], dict[str, Any]]:
    if plan.get("plan_type") != "construct_steering_conditions":
        raise BundleValidationError(f"{plan_label} has an unexpected plan_type.")
    if plan.get("run_id") != run_id:
        raise BundleValidationError(f"{plan_label}.run_id does not match the run configuration.")
    if plan.get("construct_id") != construct_id:
        raise BundleValidationError(f"{plan_label}.construct_id does not match its manifest entry.")
    plan_model = plan.get("model")
    if (
        not isinstance(plan_model, Mapping)
        or plan_model.get("model_id") != model["model_id"]
        or plan_model.get("revision") != model["revision"]
    ):
        raise BundleValidationError(f"{plan_label}.model does not match the registered model/revision.")
    if plan.get("confirmatory") is not False:
        raise BundleValidationError(f"{plan_label} must be non-confirmatory for preflight.")
    if plan.get("position_mode") != "last" or plan.get("intervention_timing") != "prefill_only":
        raise BundleValidationError(f"{plan_label} is not a last-token prefill-only plan.")

    provenance = plan.get("provenance")
    if not isinstance(provenance, Mapping):
        raise BundleValidationError(f"{plan_label}.provenance is missing.")
    if provenance.get("run_config_hash") != config_hash:
        raise BundleValidationError(f"{plan_label} has a stale/non-normalized run_config_hash.")
    if provenance.get("prompt_inventory_sha256") != inventory_hash:
        raise BundleValidationError(f"{plan_label} has the wrong prompt inventory hash.")
    if provenance.get("construct_spec_hash") != spec_hash:
        raise BundleValidationError(f"{plan_label} has the wrong construct specification hash.")

    direction_paths = plan.get("direction_paths")
    random_paths = direction_paths.get("random") if isinstance(direction_paths, Mapping) else None
    if not isinstance(random_paths, list) or not random_paths:
        raise BundleValidationError(f"{plan_label} has no random control directions.")
    if len(random_paths) != expected_random_count:
        raise BundleValidationError(
            f"{plan_label} random controls={len(random_paths)} does not match the registered "
            f"count={expected_random_count}."
        )

    tracking = plan.get("tracking_directions")
    target_path = str(direction_paths.get("target")) if isinstance(direction_paths, Mapping) else ""
    target_entries = []
    for layer, entry in (tracking.items() if isinstance(tracking, Mapping) else []):
        try:
            layer_number = int(layer)
        except (TypeError, ValueError) as exc:
            raise BundleValidationError(f"{plan_label} has a non-integer tracking layer: {layer!r}") from exc
        if layer_number not in registered_layers:
            raise BundleValidationError(f"{plan_label} tracks unregistered layer {layer_number}.")
        if isinstance(entry, Mapping) and str(entry.get("path")) == target_path:
            target_entries.append((str(layer), entry))
    if not target_entries or any(
        entry.get("source_split") != "direction_train" or entry.get("role") != "injection_immediate"
        for _, entry in target_entries
    ):
        raise BundleValidationError(
            f"{plan_label} has no target direction explicitly bound to direction_train/injection_immediate."
        )

    declared_tracking_hashes = provenance.get("tracking_direction_hashes")
    declared_controls = provenance.get("control_direction_hashes")
    if not isinstance(declared_tracking_hashes, Mapping) or not declared_tracking_hashes:
        raise BundleValidationError(f"{plan_label} is missing tracking_direction_hashes.")
    if not isinstance(declared_controls, Mapping):
        raise BundleValidationError(f"{plan_label} is missing control_direction_hashes.")
    if not _SHA256_RE.fullmatch(str(declared_controls.get("shuffled", ""))):
        raise BundleValidationError(f"{plan_label} has no frozen shuffled control hash.")
    random_control_hashes = declared_controls.get("random")
    if not isinstance(random_control_hashes, list) or len(random_control_hashes) != len(random_paths):
        raise BundleValidationError(f"{plan_label} random control hashes do not match random paths.")
    if any(not _SHA256_RE.fullmatch(str(value)) for value in random_control_hashes):
        raise BundleValidationError(f"{plan_label} contains an invalid random control hash.")

    conditions = plan.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise BundleValidationError(f"{plan_label}.conditions must be non-empty.")
    direction_kinds = {str(item.get("direction_kind")) for item in conditions if isinstance(item, Mapping)}
    if not {"target", "shuffled", "random"}.issubset(direction_kinds):
        raise BundleValidationError(f"{plan_label} conditions do not cover target/shuffled/random controls.")

    refs: dict[str, Path] = {}
    for role, raw_path in _plan_artifact_paths(plan, plan_label=plan_label):
        if role in refs:
            continue
        path = _resolve_under_root(raw_path, root=source_root, label=f"{plan_label}.{role}")
        _array(path, label=f"{plan_label}.{role}")
        refs[role] = path

    # Ensure all direction/control arrays share a hidden dimension.  A mixed
    # layer/model bundle must fail before a paid run rather than during the
    # first steering condition.
    shapes: dict[str, tuple[int, ...]] = {}
    for role, path in refs.items():
        _, shape = _array(path, label=f"{plan_label}.{role}")
        shapes[role] = shape
    dimensions = {shape[0] for shape in shapes.values()}
    if len(dimensions) != 1:
        raise BundleValidationError(f"{plan_label} direction/control dimensions disagree: {sorted(dimensions)}")

    # Direction and controls are validated against the plan's own declared
    # hashes.  Hash declarations are path-independent, so path rebasing does
    # not change the scientific provenance.
    target = refs["direction_paths.target"]
    _same_hash(target, provenance.get("direction_sha256"), label=f"{plan_label}.target direction")
    shuffled = refs["direction_paths.shuffled"]
    _same_hash(shuffled, declared_controls.get("shuffled"), label=f"{plan_label}.shuffled control")
    for index, expected in enumerate(random_control_hashes):
        _same_hash(
            refs[f"direction_paths.random[{index}]"],
            expected,
            label=f"{plan_label}.random control {index}",
        )
    for layer, entry in tracking.items():
        if not isinstance(entry, Mapping):
            continue
        key = f"tracking_directions[{layer}].path"
        expected = declared_tracking_hashes.get(str(layer), entry.get("direction_sha256"))
        _same_hash(refs[key], expected, label=f"{plan_label}.tracking direction {layer}")
        if entry.get("source_split") != "direction_train":
            raise BundleValidationError(f"{plan_label} tracking layer {layer} is not direction_train sourced.")

    return refs, {"hidden_dimension": next(iter(dimensions)), "direction_reference_count": len(refs)}


def _copy_identical(source: Path, destination: Path) -> None:
    source_hash = file_sha256(source)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or file_sha256(destination) != source_hash:
            raise BundleValidationError(f"Refusing to overwrite a different staged artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        if file_sha256(temporary) != source_hash:
            raise BundleValidationError(f"Staged copy hash changed while copying: {source}")
        try:
            destination.hardlink_to(temporary)
        except FileExistsError as exc:
            raise BundleValidationError(f"Concurrent staged artifact creation: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _staged_path(output_root: Path, construct_id: str, source: Path, *, role_index: int) -> Path:
    suffix = source.suffix.lower() or ".npy"
    return output_root / "directions" / construct_id / f"artifact_{role_index:03d}{suffix}"


def stage_steering_bundle(manifest_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Validate and copy one four-construct model-specific steering bundle.

    The operation is idempotent for the same source hashes and fail-closed for
    any changed or incomplete output.  It returns the durable staged manifest.
    """

    source_manifest_path = Path(manifest_path).expanduser().resolve()
    source_manifest = _load_json(source_manifest_path, label="source manifest")
    if source_manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleValidationError("source manifest has an unsupported schema_version.")
    if source_manifest.get("manifest_type") != MANIFEST_TYPE:
        raise BundleValidationError(f"source manifest must have manifest_type={MANIFEST_TYPE!r}.")
    model = _model_metadata(source_manifest)
    source_root = Path(str(source_manifest.get("source_root") or "")).expanduser()
    source_root = (source_manifest_path.parent / source_root if not source_root.is_absolute() else source_root).resolve()
    if not source_root.is_dir():
        raise BundleValidationError(f"source_root is not an existing directory: {source_root}")

    run_entry = source_manifest.get("run_config")
    inventory_entry = source_manifest.get("prompt_inventory")
    if not isinstance(run_entry, Mapping) or not isinstance(inventory_entry, Mapping):
        raise BundleValidationError("run_config and prompt_inventory entries are required.")
    run_config_path = _resolve_declared_path(run_entry.get("path"), base=source_manifest_path.parent, label="run_config.path")
    inventory_path = _resolve_declared_path(
        inventory_entry.get("path"), base=source_manifest_path.parent, label="prompt_inventory.path"
    )
    _same_hash(run_config_path, run_entry.get("sha256"), label="run_config")
    _same_hash(inventory_path, inventory_entry.get("sha256"), label="prompt_inventory")
    try:
        run_config = load_run_config(run_config_path)
    except (OSError, ValueError) as exc:
        raise BundleValidationError(f"run_config is invalid: {exc}") from exc
    config_hash = canonical_hash(run_config.to_mapping())
    declared_config_hash = run_entry.get("run_config_hash", run_entry.get("canonical_hash"))
    if declared_config_hash != config_hash:
        raise BundleValidationError(
            f"run_config_hash must be canonical_hash(load_run_config(...).to_mapping()): "
            f"declared {declared_config_hash!r}, actual {config_hash}"
        )
    if run_config.model.get("model_id") != model["model_id"] or run_config.model.get("revision") != model["revision"]:
        raise BundleValidationError("model metadata does not match run_config.model.")
    if not run_config.execution.get("preflight_only", False):
        raise BundleValidationError("run_config must be a preflight-only configuration.")
    if run_config.steering.get("direction_source") != "direction_train_only":
        raise BundleValidationError("run_config does not require direction_train_only steering.")

    spec_entries = source_manifest.get("construct_specs")
    if not isinstance(spec_entries, list) or not spec_entries:
        raise BundleValidationError("construct_specs must be a non-empty list.")
    specs: dict[str, Any] = {}
    for index, entry in enumerate(spec_entries):
        if not isinstance(entry, Mapping):
            raise BundleValidationError(f"construct_specs[{index}] is malformed.")
        construct_id = _safe_component(entry.get("construct_id"), label=f"construct_specs[{index}].construct_id")
        if construct_id in specs:
            raise BundleValidationError(f"duplicate construct specification: {construct_id}")
        spec_path = _resolve_declared_path(entry.get("path"), base=source_manifest_path.parent, label=f"construct_specs[{index}].path")
        _same_hash(spec_path, entry.get("sha256"), label=f"construct_specs[{index}]")
        try:
            spec = load_construct_spec(spec_path)
        except (OSError, ValueError) as exc:
            raise BundleValidationError(f"construct_specs[{index}] is invalid: {exc}") from exc
        if spec.construct_id != construct_id:
            raise BundleValidationError(f"construct_specs[{index}] construct_id does not resolve to {construct_id!r}.")
        spec_hash = canonical_hash(spec.to_mapping())
        if entry.get("construct_spec_hash", entry.get("canonical_hash")) != spec_hash:
            raise BundleValidationError(f"construct_specs[{index}] has the wrong normalized spec hash.")
        specs[construct_id] = spec
    if set(specs) != set(run_config.construct_ids):
        raise BundleValidationError(
            f"construct scope mismatch: run config={sorted(run_config.construct_ids)}, specs={sorted(specs)}"
        )
    if len(specs) != 4:
        raise BundleValidationError(f"preflight steering bundles must contain exactly four constructs, got {len(specs)}")

    plan_entries = source_manifest.get("plans")
    if not isinstance(plan_entries, list) or len(plan_entries) != 4:
        raise BundleValidationError("plans must contain exactly four construct steering plans.")
    plans: dict[str, tuple[dict[str, Any], Path, dict[str, Path], dict[str, Any]]] = {}
    for index, entry in enumerate(plan_entries):
        if not isinstance(entry, Mapping):
            raise BundleValidationError(f"plans[{index}] is malformed.")
        construct_id = _safe_component(entry.get("construct_id"), label=f"plans[{index}].construct_id")
        if construct_id in plans:
            raise BundleValidationError(f"duplicate steering plan for construct {construct_id}")
        if construct_id not in specs:
            raise BundleValidationError(f"steering plan references unregistered construct {construct_id}")
        plan_path = _resolve_declared_path(entry.get("path"), base=source_manifest_path.parent, label=f"plans[{index}].path")
        _same_hash(plan_path, entry.get("sha256"), label=f"plans[{index}]")
        plan = _load_json(plan_path, label=f"plans[{index}]")
        refs, plan_summary = _plan_provenance(
            plan,
            config_hash=config_hash,
            inventory_hash=file_sha256(inventory_path),
            spec_hash=canonical_hash(specs[construct_id].to_mapping()),
            model=model,
            construct_id=construct_id,
            run_id=run_config.run_id,
            plan_label=f"plans[{index}]",
            source_root=source_root,
            expected_random_count=int(run_config.steering["random_direction_count"]),
            registered_layers={int(layer) for layer in run_config.activation["layers"]},
        )
        plans[construct_id] = (plan, plan_path, refs, plan_summary)
    if set(plans) != set(specs):
        raise BundleValidationError(f"plan scope mismatch: plans={sorted(plans)}, specs={sorted(specs)}")

    output = Path(output_root).expanduser().resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise BundleValidationError(
            f"output_root must be outside source_root to preserve the source bundle: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest_output = output / "staging_manifest.json"
    if manifest_output.exists():
        existing = validate_staged_bundle(output)
        if existing.get("source_manifest_path") != str(source_manifest_path):
            raise BundleValidationError(f"Refusing to reuse a staged bundle from another source manifest: {manifest_output}")
        expected_run = {
            "path": str(run_config_path),
            "sha256": file_sha256(run_config_path),
            "run_config_hash": config_hash,
        }
        expected_inventory = {"path": str(inventory_path), "sha256": file_sha256(inventory_path)}
        if (
            existing.get("model") != model
            or existing.get("run_id") != run_config.run_id
            or existing.get("run_config") != expected_run
            or existing.get("prompt_inventory") != expected_inventory
            or sorted(existing.get("construct_ids", [])) != sorted(specs)
        ):
            raise BundleValidationError(f"Staged bundle provenance changed: {manifest_output}")
        expected_sources = {
            construct_id: value[1]
            for construct_id, value in plans.items()
        }
        existing_sources = {
            item.get("construct_id"): item.get("source_sha256")
            for item in existing.get("plans", [])
            if isinstance(item, Mapping)
        }
        if any(file_sha256(path) != existing_sources.get(construct_id) for construct_id, path in expected_sources.items()):
            raise BundleValidationError(f"Source steering plans changed since staging: {manifest_output}")
        return existing
    if any(output.iterdir()):
        raise BundleValidationError(
            f"Refusing to stage into a non-empty directory without a valid manifest: {output}"
        )
    staged_plans: list[dict[str, Any]] = []
    staged_artifacts: list[dict[str, Any]] = []
    for construct_id in sorted(plans):
        plan, plan_path, refs, plan_summary = plans[construct_id]
        rewritten = copy.deepcopy(plan)
        path_map: dict[Path, Path] = {}
        role_counter = 0
        for role, _ in _plan_artifact_paths(plan, plan_label=f"plans/{construct_id}"):
            source_path = refs[role]
            if source_path not in path_map:
                destination = _staged_path(output, construct_id, source_path, role_index=role_counter)
                role_counter += 1
                _copy_identical(source_path, destination)
                path_map[source_path] = destination.resolve()
                staged_artifacts.append(
                    {
                        "construct_id": construct_id,
                        "role": role,
                        "source_path": str(source_path),
                        "staged_path": str(destination.resolve()),
                        "sha256": file_sha256(destination),
                        "bytes": destination.stat().st_size,
                    }
                )

        def rewrite(raw: Any) -> Any:
            if not isinstance(raw, str):
                return raw
            source = _resolve_under_root(raw, root=source_root, label=f"plans/{construct_id}.path")
            return str(path_map[source])

        direction_paths = rewritten.get("direction_paths")
        for key in ("target", "shuffled"):
            direction_paths[key] = rewrite(direction_paths[key])
        direction_paths["random"] = [rewrite(value) for value in direction_paths["random"]]
        for entry in rewritten["tracking_directions"].values():
            entry["path"] = rewrite(entry["path"])
        staged_plan_path = output / "plans" / f"{construct_id}.json"
        # This metadata is path/provenance bookkeeping only; all scientific
        # fields and hashes remain those emitted by plan_construct_steering.
        rewritten["staging"] = {
            "manifest_type": STAGED_MANIFEST_TYPE,
            "source_plan_sha256": file_sha256(plan_path),
            "artifact_root": str(output.resolve()),
            "references_rebased": True,
        }
        _write_json(staged_plan_path, rewritten)
        staged_plans.append(
            {
                "construct_id": construct_id,
                "source_path": str(plan_path),
                "source_sha256": file_sha256(plan_path),
                "staged_path": str(staged_plan_path.resolve()),
                "staged_sha256": file_sha256(staged_plan_path),
                "direction_reference_count": plan_summary["direction_reference_count"],
                "hidden_dimension": plan_summary["hidden_dimension"],
            }
        )

    staged = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": STAGED_MANIFEST_TYPE,
        "status": "ready",
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_root": str(source_root),
        "output_root": str(output),
        "model": model,
        "run_id": run_config.run_id,
        "run_config": {
            "path": str(run_config_path),
            "sha256": file_sha256(run_config_path),
            "run_config_hash": config_hash,
        },
        "prompt_inventory": {"path": str(inventory_path), "sha256": file_sha256(inventory_path)},
        "construct_ids": sorted(specs),
        "plans": staged_plans,
        "direction_artifacts": staged_artifacts,
        "candidate_count": len(staged_plans),
        "policy": {
            "external_calls": False,
            "model_weights_loaded": False,
            "synthetic_directions": False,
            "outcome_dependent_selection": False,
            "path_references": "absolute_under_output_root",
            "overwrite_policy": "refuse_different_content",
        },
    }
    _write_json(manifest_output, staged)
    return staged


def validate_staged_bundle(path: str | Path) -> dict[str, Any]:
    """Verify staged files, plan hashes, and every rebased direction path."""

    root = Path(path).expanduser().resolve()
    manifest_path = root / "staging_manifest.json"
    manifest = _load_json(manifest_path, label="staged manifest")
    if manifest.get("manifest_type") != STAGED_MANIFEST_TYPE or manifest.get("status") != "ready":
        raise BundleValidationError("staged manifest is not a ready model steering bundle.")
    if Path(str(manifest.get("output_root"))).resolve() != root:
        raise BundleValidationError("staged manifest output_root does not match its directory.")
    plans = manifest.get("plans")
    if not isinstance(plans, list) or manifest.get("candidate_count") != len(plans) or len(plans) != 4:
        raise BundleValidationError("staged manifest must contain exactly four plan candidates.")
    for entry in plans:
        plan_path = Path(str(entry.get("staged_path"))).resolve()
        try:
            plan_path.relative_to(root)
        except ValueError as exc:
            raise BundleValidationError(f"staged plan escapes output root: {plan_path}") from exc
        if file_sha256(plan_path) != entry.get("staged_sha256"):
            raise BundleValidationError(f"staged plan hash mismatch: {plan_path}")
        plan = _load_json(plan_path, label=f"staged plan {plan_path}")
        for _, raw_path in _plan_artifact_paths(plan, plan_label=str(plan_path)):
            artifact = Path(raw_path).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as exc:
                raise BundleValidationError(f"staged direction path escapes output root: {artifact}") from exc
            _array(artifact, label=f"staged direction {artifact}")
        # The execution runner consumes absolute references directly.  Verify
        # every one is absolute, not merely present.
        for _, raw_path in _plan_artifact_paths(plan, plan_label=str(plan_path)):
            if not Path(raw_path).is_absolute():
                raise BundleValidationError(f"staged plan contains a non-absolute path: {raw_path}")
    return manifest


# A short alias is convenient for callers that treat all preflight staging
# operations as bundles, while the descriptive name remains the documented
# CLI/API entry point.
stage_bundle = stage_steering_bundle


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Hash-bound model B/R source-bundle manifest.")
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh per-model/per-wave staging root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        staged = stage_steering_bundle(args.manifest, args.output_root)
        validate_staged_bundle(args.output_root)
    except BundleValidationError as exc:
        print(f"stage_model_steering_preflight: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(staged, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
