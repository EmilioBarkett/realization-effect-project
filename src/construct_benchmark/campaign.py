"""Wave-scoped prompt inventories and confirmatory release checks.

The model-side benchmark runs one balanced four-construct wave at a time.  A
wave inventory combines the frozen all-construct vector prompts with the
independent downstream prompts, while retaining source manifests and hashes.
The composed artifact is frozen for engineering use; a separate release
check prevents an engineering inventory from being used as confirmatory data
before the scientific prerequisites are satisfied.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import (
    load_analysis_spec,
    load_construct_specs,
    load_run_config,
    validate_analysis_spec,
    validate_run_constructs,
)
from .manifests import build_run_plan, file_sha256
from .prompts import PromptRecord, load_prompt_records, validate_prompt_records, write_prompt_records
from .registry import load_construct_registry
from .run_modes import select_prompt_records


VECTOR_SPLITS = frozenset({"direction_train", "direction_validation", "direction_heldout"})
DOWNSTREAM_SPLITS = frozenset({"behavior_eval", "steering_eval", "calibration", "collateral_eval"})
WAVE_VECTOR_COUNTS = {
    "direction_train": 200,
    "direction_validation": 80,
    "direction_heldout": 80,
}


def wave_construct_ids(registry_path: str | Path, wave: int) -> tuple[str, ...]:
    """Return registry-ordered construct IDs for a four-construct wave."""

    if wave not in {2, 3, 4}:
        raise ValueError("Confirmatory wave execution currently supports waves 2, 3, and 4 only.")
    registry = load_construct_registry(registry_path)
    entries = tuple(entry for entry in registry.entries if entry.wave == wave)
    if len(entries) != 4:
        raise ValueError(f"Wave {wave} must contain exactly four constructs.")
    return tuple(entry.construct_id for entry in entries)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _resolve_manifest_file(manifest_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing a file path.")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist: {candidate}")
    return candidate


def _validate_frozen_source_manifest(
    manifest_path: Path,
    *,
    source_path: Path,
    expected_construct_ids: set[str],
    label: str,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    source_path = source_path.resolve()
    manifest = _load_object(manifest_path, label=label)
    manifest_type = manifest.get("manifest_type")
    is_vector_inventory = manifest_type in {
        "final_vector_prompt_inventory",
        "vector_prompt_generation",
    }
    if is_vector_inventory:
        # The active vector generator emits ``vector_prompt_generation``;
        # older frozen artifacts use ``final_vector_prompt_inventory``.  Both
        # are valid only as complete, non-confirmatory vector sources.  Keep
        # this compatibility explicit so a review/partial or dry-run manifest
        # cannot be composed into a wave inventory.
        if (
            manifest.get("confirmatory") is not False
            or manifest.get("scope_partial") is not True
            or manifest.get("run_mode") != "full"
            or manifest.get("partial") is not False
            or manifest.get("dry_run") is not False
        ):
            raise ValueError(f"{label} must be the non-confirmatory full vector/probe inventory.")
        manifest_construct_ids = {
            str(item.get("construct_id"))
            for item in manifest.get("constructs", [])
            if isinstance(item, Mapping) and item.get("construct_id")
        }
    else:
        if manifest.get("status") != "frozen" or manifest.get("frozen") is not True:
            raise ValueError(f"{label} must be frozen before wave composition.")
        if manifest.get("run_mode") != "full" or manifest.get("partial") is True:
            raise ValueError(f"{label} must describe a complete full inventory.")
        manifest_construct_ids = set(manifest.get("construct_ids", []))
    if not expected_construct_ids.issubset(manifest_construct_ids):
        missing = sorted(expected_construct_ids - manifest_construct_ids)
        raise ValueError(f"{label} is missing construct(s): {missing}")

    declared_path = _resolve_manifest_file(manifest_path, manifest.get("combined_path"), label=f"{label}.combined_path")
    if declared_path != source_path:
        raise ValueError(f"{label} combined_path does not match the supplied source file.")
    declared_hash = manifest.get("combined_sha256")
    actual_hash = file_sha256(source_path)
    if declared_hash != actual_hash:
        raise ValueError(f"{label} combined_sha256 does not match the source file.")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "combined_path": str(source_path),
        "combined_sha256": actual_hash,
        "status": manifest.get("status"),
        "frozen": manifest.get("frozen"),
        "confirmatory": manifest.get("confirmatory"),
        "record_count": manifest.get("record_count", manifest.get("counts", {}).get("record_count")),
        "construct_ids": sorted(manifest_construct_ids),
    }


def _validate_quality_gate(path: Path, expected_construct_ids: set[str]) -> dict[str, Any]:
    path = path.resolve()
    gate = _load_object(path, label="downstream quality gate")
    if gate.get("status") != "approved" or gate.get("approved") is not True:
        raise ValueError("The downstream quality gate must be approved before wave composition.")
    component_ids = {
        str(construct_id)
        for component in gate.get("components", [])
        if isinstance(component, Mapping)
        for construct_id in component.get("construct_ids", [])
    }
    if not expected_construct_ids.issubset(component_ids):
        missing = sorted(expected_construct_ids - component_ids)
        raise ValueError(f"The downstream quality gate is missing construct(s): {missing}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": gate["status"],
        "approved": gate["approved"],
        "scope": gate.get("scope"),
        "reviewer": gate.get("reviewer"),
        "construct_ids": sorted(component_ids),
    }


def _validate_prompt_audit(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Require an embedded passing prompt audit before confirmatory use."""

    audit = payload.get("prompt_audit")
    if not isinstance(audit, Mapping):
        raise ValueError(f"{label} is missing an embedded prompt_audit record.")
    if audit.get("passed") is not True or int(audit.get("severe_flag_count", 1)) != 0:
        raise ValueError(
            f"{label} is blocked by its prompt audit: "
            f"passed={audit.get('passed')!r}, severe_flag_count={audit.get('severe_flag_count')!r}."
        )
    return dict(audit)


def _records_by_construct(records: Iterable[PromptRecord], construct_ids: set[str]) -> list[PromptRecord]:
    selected = [record for record in records if record.construct_id in construct_ids]
    if not selected:
        raise ValueError("No records matched the requested wave constructs.")
    return selected


def _split_counts(records: Iterable[PromptRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for record in records:
        counts.setdefault(record.construct_id, Counter())[record.split] += 1
    return {
        construct_id: dict(sorted(split_counts.items()))
        for construct_id, split_counts in sorted(counts.items())
    }


def compose_wave_prompt_inventory(
    *,
    wave: int,
    registry_path: str | Path,
    vector_prompt_path: str | Path,
    vector_manifest_path: str | Path,
    downstream_prompt_path: str | Path,
    downstream_manifest_path: str | Path,
    quality_gate_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compose and freeze one wave's vector-plus-downstream inventory.

    The result is deliberately marked ``confirmatory=false``.  Composition
    proves source integrity and split coverage; it does not satisfy the
    separate Wave 1 measurement and precision-simulation release gates.
    """

    registry_path = Path(registry_path).resolve()
    construct_ids = wave_construct_ids(registry_path, wave)
    expected_ids = set(construct_ids)
    registry = load_construct_registry(registry_path)
    specs = load_construct_specs(
        registry_path.parent / entry.spec_path
        for entry in registry.entries
        if entry.construct_id in expected_ids
    )

    vector_prompt_path = Path(vector_prompt_path).resolve()
    vector_manifest_path = Path(vector_manifest_path).resolve()
    downstream_prompt_path = Path(downstream_prompt_path).resolve()
    downstream_manifest_path = Path(downstream_manifest_path).resolve()
    quality_gate_path = Path(quality_gate_path).resolve()
    for path, label in (
        (vector_prompt_path, "vector prompt inventory"),
        (vector_manifest_path, "vector inventory manifest"),
        (downstream_prompt_path, "downstream prompt inventory"),
        (downstream_manifest_path, "downstream inventory manifest"),
        (quality_gate_path, "downstream quality gate"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")

    vector_source = _validate_frozen_source_manifest(
        vector_manifest_path,
        source_path=vector_prompt_path,
        expected_construct_ids=expected_ids,
        label="vector inventory manifest",
    )
    downstream_source = _validate_frozen_source_manifest(
        downstream_manifest_path,
        source_path=downstream_prompt_path,
        expected_construct_ids=expected_ids,
        label="downstream inventory manifest",
    )
    quality_gate = _validate_quality_gate(quality_gate_path, expected_ids)

    vector_records = _records_by_construct(load_prompt_records(vector_prompt_path), expected_ids)
    downstream_records = _records_by_construct(load_prompt_records(downstream_prompt_path), expected_ids)
    if any(record.split not in VECTOR_SPLITS for record in vector_records):
        raise ValueError("Vector source contains a non-vector split.")
    if any(record.split not in DOWNSTREAM_SPLITS for record in downstream_records):
        raise ValueError("Downstream source contains a non-downstream split.")
    vector_validation = validate_prompt_records(vector_records, specs, require_all_splits=False)
    downstream_validation = validate_prompt_records(downstream_records, specs, require_all_splits=False)

    vector_counts = _split_counts(vector_records)
    for construct_id in construct_ids:
        if vector_counts.get(construct_id) != WAVE_VECTOR_COUNTS:
            raise ValueError(
                f"{construct_id} has unexpected vector split counts: {vector_counts.get(construct_id)!r}; "
                f"expected {WAVE_VECTOR_COUNTS!r}."
            )

    combined_records = [*vector_records, *downstream_records]
    combined_validation = validate_prompt_records(combined_records, specs, require_all_splits=True)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined.csv"
    manifest_path = output_dir / "inventory_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing wave inventory: {output_dir}")
    write_prompt_records(combined_records, output_path)

    manifest = {
        "schema_version": "0.1.0",
        "manifest_type": "wave_execution_prompt_inventory",
        "status": "frozen",
        "frozen": True,
        "confirmatory": False,
        "scope": "vector_probe_plus_downstream",
        "wave": wave,
        "construct_ids": list(construct_ids),
        "record_count": len(combined_records),
        "vector_probe_record_count": len(vector_records),
        "downstream_record_count": len(downstream_records),
        "counts_by_construct_split": _split_counts(combined_records),
        "sources": {
            "vector": vector_source,
            "downstream": downstream_source,
            "downstream_quality_gate": quality_gate,
        },
        "validation": {
            "vector": vector_validation,
            "downstream": downstream_validation,
            "combined": combined_validation,
        },
        "output_path": "combined.csv",
        "output_sha256": file_sha256(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def release_wave_prompt_inventory(
    *,
    wave: int,
    registry_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    released_by: str,
    release_statement: str,
    release_date: str,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Promote one frozen wave inventory into an immutable prompt release.

    This copies the exact frozen CSV into a new directory and writes a new
    manifest with an explicit prompt-input release attestation. It does not
    assert that the model-side campaign is ready: Wave 1 measurement and
    precision-simulation gates remain separate execution prerequisites.
    """

    if not released_by.strip():
        raise ValueError("released_by must be non-empty.")
    if not release_statement.strip():
        raise ValueError("release_statement must be non-empty.")
    if not release_date.strip():
        raise ValueError("release_date must be non-empty.")

    registry_path = Path(registry_path).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    source_manifest = _load_object(source_manifest_path, label="source wave inventory manifest")
    expected_construct_ids = wave_construct_ids(registry_path, wave)
    expected_ids = set(expected_construct_ids)
    if source_manifest.get("manifest_type") != "wave_execution_prompt_inventory":
        raise ValueError("The source manifest is not a wave execution prompt inventory.")
    if source_manifest.get("status") != "frozen" or source_manifest.get("frozen") is not True:
        raise ValueError("The source wave inventory must be frozen before release.")
    if source_manifest.get("confirmatory") is not False:
        raise ValueError("The release source must be explicitly non-confirmatory engineering data.")
    prompt_audit = _validate_prompt_audit(source_manifest, label="The source wave inventory")
    if tuple(source_manifest.get("construct_ids", [])) != expected_construct_ids:
        raise ValueError(
            f"Source construct IDs do not match wave {wave}: "
            f"expected {expected_construct_ids!r}."
        )

    source_prompt_path = _resolve_manifest_file(
        source_manifest_path,
        source_manifest.get("output_path"),
        label="source wave inventory output",
    )
    source_hash = file_sha256(source_prompt_path)
    if source_hash != source_manifest.get("output_sha256"):
        raise ValueError("The source wave inventory hash does not match its manifest.")

    sources = source_manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("The source wave inventory is missing source provenance.")
    downstream_gate = sources.get("downstream_quality_gate")
    if not isinstance(downstream_gate, Mapping) or downstream_gate.get("approved") is not True:
        raise ValueError("The source wave inventory lacks an approved downstream quality gate.")

    registry = load_construct_registry(registry_path)
    specs = load_construct_specs(
        registry_path.parent / entry.spec_path
        for entry in registry.entries
        if entry.construct_id in expected_ids
    )
    records = load_prompt_records(source_prompt_path)
    if len(records) != source_manifest.get("record_count"):
        raise ValueError("The source record count does not match its manifest.")
    validation = validate_prompt_records(records, specs, require_all_splits=True)
    counts = _split_counts(records)
    for construct_id in expected_construct_ids:
        vector_counts = {
            split: counts.get(construct_id, {}).get(split, 0)
            for split in VECTOR_SPLITS
        }
        if vector_counts != WAVE_VECTOR_COUNTS:
            raise ValueError(
                f"{construct_id} has unexpected vector counts before release: "
                f"{vector_counts!r}."
            )
        if not all(counts.get(construct_id, {}).get(split, 0) > 0 for split in DOWNSTREAM_SPLITS):
            raise ValueError(f"{construct_id} is missing a downstream split before release.")

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing prompt release: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "combined.csv"
    manifest_path = output_dir / "inventory_manifest.json"
    shutil.copyfile(source_prompt_path, output_path)
    output_hash = file_sha256(output_path)
    if output_hash != source_hash:
        raise ValueError("The released prompt copy does not match the frozen source.")

    release_manifest = dict(source_manifest)
    release_manifest.update(
        {
            "status": "released",
            "confirmatory": True,
            "output_path": "combined.csv",
            "output_sha256": output_hash,
            "validation": {
                **dict(source_manifest.get("validation", {})),
                "release_validation": validation,
            },
            "prompt_audit": prompt_audit,
            "release": {
                "release_id": release_id or f"wave{wave}_four_construct_confirmatory_prompt_release_v1",
                "release_date": release_date,
                "released_by": released_by,
                "release_statement": release_statement,
                "scope": "confirmatory_prompt_inputs_only",
                "source_manifest_path": str(source_manifest_path),
                "source_manifest_sha256": file_sha256(source_manifest_path),
                "source_output_path": str(source_prompt_path),
                "source_output_sha256": source_hash,
                "model_execution_release": False,
                "execution_gate_note": (
                    "This release freezes prompt inputs only. Model-side confirmatory "
                    "execution remains subject to the campaign prerequisites."
                ),
            },
        }
    )
    manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_manifest


def release_wave_prompt_inventories(
    *,
    waves: Iterable[int],
    registry_path: str | Path,
    source_manifest_paths: Mapping[int, str | Path],
    output_root: str | Path,
    released_by: str,
    release_statement: str,
    release_date: str,
) -> list[dict[str, Any]]:
    """Release multiple wave inventories without overwriting existing artifacts."""

    summaries = []
    output_root = Path(output_root).resolve()
    for wave in dict.fromkeys(waves):
        if wave not in {2, 3, 4}:
            raise ValueError("Only waves 2, 3, and 4 may be released by this package.")
        try:
            source_manifest_path = source_manifest_paths[wave]
        except KeyError as exc:
            raise ValueError(f"No source manifest supplied for wave {wave}.") from exc
        summaries.append(
            release_wave_prompt_inventory(
                wave=wave,
                registry_path=registry_path,
                source_manifest_path=source_manifest_path,
                output_dir=output_root / f"wave{wave}_four_construct_confirmatory_v1",
                released_by=released_by,
                release_statement=release_statement,
                release_date=release_date,
            )
        )
    return summaries


def _resolve_campaign_path(campaign_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path.")
    path = Path(value)
    if not path.is_absolute():
        path = campaign_path.parent / path
    return path.resolve()


def _check(ok: bool, name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if ok else "fail", "detail": detail}


def _prerequisite_check(campaign_path: Path, item: Mapping[str, Any], *, mode: str) -> dict[str, str]:
    """Validate a campaign prerequisite and any evidence it claims to have."""

    name = str(item.get("id") or "unnamed_prerequisite")
    satisfied = item.get("status") == "satisfied"
    detail = str(item.get("detail") or item.get("requirement") or "No detail supplied.")
    if not satisfied:
        return {
            "name": name,
            "status": "pending" if mode == "test" else "fail",
            "detail": detail,
        }

    evidence_value = item.get("evidence_path")
    evidence_hash = item.get("evidence_sha256")
    if not isinstance(evidence_value, str) or not evidence_value.strip():
        return _check(False, name, "Satisfied prerequisite is missing evidence_path.")
    if not isinstance(evidence_hash, str) or not evidence_hash.strip():
        return _check(False, name, "Satisfied prerequisite is missing evidence_sha256.")
    evidence_path = _resolve_campaign_path(campaign_path, evidence_value, label=f"{name}.evidence_path")
    if not evidence_path.is_file():
        return _check(False, name, f"Evidence file does not exist: {evidence_path}")
    actual_hash = file_sha256(evidence_path)
    if actual_hash != evidence_hash:
        return _check(False, name, "Evidence SHA-256 does not match the campaign manifest.")
    return _check(True, name, f"{detail} Evidence: {evidence_path}")


def confirmatory_execution_report(
    campaign_path: str | Path,
    *,
    mode: str = "test",
    waves: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Validate a wave campaign without loading weights or contacting services.

    ``test`` is allowed against frozen engineering inventories.  ``full`` is
    intentionally fail-closed until every prerequisite is marked satisfied
    and each composed inventory has been explicitly released as confirmatory.
    """

    if mode not in {"test", "full"}:
        raise ValueError("mode must be 'test' or 'full'.")
    campaign_path = Path(campaign_path).resolve()
    campaign = _load_object(campaign_path, label="confirmatory campaign")
    campaign_waves = campaign.get("waves")
    if not isinstance(campaign_waves, list) or not campaign_waves:
        raise ValueError("Confirmatory campaign must list one or more waves.")
    requested_waves = set(waves) if waves is not None else None
    if requested_waves is not None and not requested_waves.issubset({2, 3, 4}):
        raise ValueError("Only waves 2, 3, and 4 may be selected.")

    registry_path = _resolve_campaign_path(campaign_path, campaign.get("registry_path"), label="registry_path")
    analysis_path = _resolve_campaign_path(
        campaign_path,
        campaign.get("analysis_spec_path"),
        label="analysis_spec_path",
    )
    registry = load_construct_registry(registry_path)
    analysis_spec = load_analysis_spec(analysis_path)
    prerequisites = campaign.get("required_confirmatory_prerequisites", [])
    if not isinstance(prerequisites, list):
        raise ValueError("required_confirmatory_prerequisites must be a list.")
    prerequisite_checks = []
    for item in prerequisites:
        if not isinstance(item, Mapping):
            raise ValueError("Every confirmatory prerequisite must be an object.")
        prerequisite_checks.append(_prerequisite_check(campaign_path, item, mode=mode))

    run_entries = []
    for raw_entry in campaign_waves:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Every campaign wave entry must be an object.")
        wave = int(raw_entry.get("wave", -1))
        if requested_waves is not None and wave not in requested_waves:
            continue
        construct_ids = tuple(str(value) for value in raw_entry.get("construct_ids", []))
        expected_ids = wave_construct_ids(registry_path, wave)
        checks = []
        checks.append(_check(construct_ids == expected_ids, "wave_construct_ids", f"expected {expected_ids!r}"))
        run_config_path = _resolve_campaign_path(
            campaign_path,
            raw_entry.get("run_config_path"),
            label=f"wave {wave} run_config_path",
        )
        inventory_manifest_path = _resolve_campaign_path(
            campaign_path,
            raw_entry.get("inventory_manifest_path"),
            label=f"wave {wave} inventory_manifest_path",
        )
        run_config = load_run_config(run_config_path)
        specs = load_construct_specs(
            registry_path.parent / entry.spec_path
            for entry in registry.entries
            if entry.construct_id in set(construct_ids)
        )
        validate_run_constructs(run_config, specs)
        validate_analysis_spec(run_config, analysis_spec)
        inventory_manifest = _load_object(inventory_manifest_path, label=f"wave {wave} inventory manifest")
        prompt_path = _resolve_campaign_path(
            campaign_path,
            raw_entry.get("prompt_inventory_path"),
            label=f"wave {wave} prompt_inventory_path",
        )
        checks.append(
            _check(
                inventory_manifest.get("manifest_type") == "wave_execution_prompt_inventory",
                "inventory_manifest_type",
                "wave execution inventory manifest",
            )
        )
        checks.append(
            _check(
                inventory_manifest.get("status") in {"frozen", "released"}
                and inventory_manifest.get("frozen") is True,
                "inventory_frozen",
                "inventory is frozen and may have a released lifecycle status",
            )
        )
        declared_output = _resolve_manifest_file(
            inventory_manifest_path,
            inventory_manifest.get("output_path"),
            label=f"wave {wave} inventory output",
        )
        checks.append(
            _check(
                declared_output == prompt_path and file_sha256(prompt_path) == inventory_manifest.get("output_sha256"),
                "inventory_hash",
                "inventory path and SHA-256 match",
            )
        )
        try:
            _validate_prompt_audit(inventory_manifest, label=f"Wave {wave} inventory manifest")
            checks.append(_check(True, "inventory_prompt_audit", "embedded prompt audit passed with zero severe flags"))
        except (TypeError, ValueError) as exc:
            checks.append(_check(False, "inventory_prompt_audit", str(exc)))
        records = load_prompt_records(prompt_path)
        plan = build_run_plan(
            run_config,
            specs,
            analysis_spec,
            prompt_inventory_path=prompt_path,
            prompt_records=records,
            run_mode=mode,
        )
        selected_count = None
        if mode == "test":
            selected, _selection_manifest = select_prompt_records(
                records,
                run_config=run_config,
                construct_specs=specs,
                mode="test",
            )
            selected_count = len(selected)
            checks.append(_check(True, "test_selection", f"{selected_count} deterministic test prompts selected"))
        else:
            released = inventory_manifest.get("confirmatory") is True
            checks.append(
                _check(
                    released,
                    "inventory_confirmatory_release",
                    "inventory is explicitly released as confirmatory"
                    if released
                    else "inventory remains confirmatory=false engineering data",
                )
            )
            checks.extend(
                _check(
                    prerequisite["status"] == "pass",
                    prerequisite["name"],
                    prerequisite["detail"],
                )
                for prerequisite in prerequisite_checks
            )
        run_entries.append(
            {
                "wave": wave,
                "run_id": run_config.run_id,
                "construct_ids": list(construct_ids),
                "run_plan_mode": plan["run_mode"],
                "prompt_count": len(records),
                "test_prompt_count": selected_count,
                "checks": checks,
            }
        )

    if not run_entries:
        raise ValueError("No campaign waves matched the requested selection.")
    all_checks = [check for entry in run_entries for check in entry["checks"]]
    blocking = []
    seen_blockers: set[tuple[str, str]] = set()
    for check in all_checks:
        if check["status"] != "fail":
            continue
        blocker_key = (check["name"], check["detail"])
        if blocker_key not in seen_blockers:
            blocking.append(check)
            seen_blockers.add(blocker_key)
    return {
        "schema_version": "0.1.0",
        "manifest_type": "confirmatory_execution_report",
        "campaign_id": campaign.get("campaign_id"),
        "mode": mode,
        "confirmatory": mode == "full" and not blocking,
        "ready": not blocking,
        "prerequisites": prerequisite_checks,
        "runs": run_entries,
        "blocking_checks": blocking,
    }


__all__ = [
    "DOWNSTREAM_SPLITS",
    "VECTOR_SPLITS",
    "confirmatory_execution_report",
    "compose_wave_prompt_inventory",
    "release_wave_prompt_inventories",
    "release_wave_prompt_inventory",
    "wave_construct_ids",
]
