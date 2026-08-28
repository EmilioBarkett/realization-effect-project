#!/usr/bin/env python3
"""Compose and audit the repaired Wave 1 prompt inventory.

This is a no-API composition step.  It accepts only complete, per-construct
vector files and a complete repaired downstream generation manifest, validates
their plan/spec identities, checks global prompt-text uniqueness, and writes a
new engineering inventory with explicit provenance.  It never overwrites an
existing composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import load_construct_specs  # noqa: E402
from construct_benchmark.generation import (  # noqa: E402
    PROBE_WRAPPER_NORMALIZATION_VERSION,
    dry_run_summary,
    load_generation_plan,
    normalize_probe_prompt_wrapper,
)
from construct_benchmark.manifests import file_sha256  # noqa: E402
from construct_benchmark.prompts import (  # noqa: E402
    PromptRecord,
    load_prompt_records,
    validate_prompt_records,
    write_prompt_records,
)
from construct_benchmark.registry import load_construct_registry  # noqa: E402

try:  # direct CLI execution has ``scripts/`` on sys.path
    from scripts.audit_wave_prompt_inventories import audit_wave_inventory  # type: ignore
    from scripts.generate_downstream_prompts import (  # type: ignore
        DEFAULT_MODEL,
        DEFAULT_PROVIDER,
        DEFAULT_REASONING_EFFORT,
        DOWNSTREAM_SPLITS,
        _effective_entries,
        _validate_downstream_records,
    )
except ModuleNotFoundError:  # pragma: no cover - defensive direct-import path
    from audit_wave_prompt_inventories import audit_wave_inventory  # type: ignore
    from generate_downstream_prompts import (  # type: ignore
        DEFAULT_MODEL,
        DEFAULT_PROVIDER,
        DEFAULT_REASONING_EFFORT,
        DOWNSTREAM_SPLITS,
        _effective_entries,
        _validate_downstream_records,
    )


VECTOR_SPLITS = frozenset({"direction_train", "direction_validation", "direction_heldout"})
WAVE1_IDS = (
    "realization_account_closure",
    "evidence_diagnosticity",
    "source_reliability",
    "persistence_continuation",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _normalise_prompt_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _assert_unique_prompt_text(records: Iterable[PromptRecord]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        normalized = _normalise_prompt_text(record.prompt_text)
        previous = seen.get(normalized)
        if previous is not None:
            raise ValueError(
                "Combined Wave 1 inventory reuses normalized prompt text across records: "
                f"{previous!r} and {record.prompt_id!r}."
            )
        seen[normalized] = record.prompt_id


def _manifest_hashes(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def compose_wave1_inventory(
    *,
    registry_path: str | Path,
    vector_root: str | Path,
    downstream_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    registry_path = Path(registry_path).resolve()
    vector_root = Path(vector_root).resolve()
    downstream_root = Path(downstream_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty composition output: {output_dir}")

    registry = load_construct_registry(registry_path)
    raw_registry = _load_object(registry_path, label="Wave 1 registry")
    raw_entries = {
        str(item["construct_id"]): item
        for item in raw_registry.get("entries", [])
        if isinstance(item, Mapping) and isinstance(item.get("construct_id"), str)
    }
    entries = tuple(entry for entry in registry.entries if entry.wave == 1)
    if tuple(entry.construct_id for entry in entries) != WAVE1_IDS:
        raise ValueError(
            "The repaired Wave 1 registry must contain the expected four constructs in registry order."
        )
    specs = load_construct_specs(
        registry_path.parent / entry.spec_path
        for entry in entries
    )

    vector_manifest_path = vector_root / "vector_prompt_manifest.json"
    vector_manifest = _load_object(vector_manifest_path, label="Wave 1 vector manifest")
    if (
        vector_manifest.get("manifest_type") != "vector_prompt_generation"
        or vector_manifest.get("run_mode") != "full"
        or vector_manifest.get("partial") is not False
        or vector_manifest.get("dry_run") is not False
        or vector_manifest.get("confirmatory") is not False
        or vector_manifest.get("scope_partial") is not True
        or vector_manifest.get("construct_ids") != list(WAVE1_IDS)
    ):
        raise ValueError("Wave 1 vector input must be a complete non-confirmatory full manifest.")
    vector_combined_path = vector_root / "combined.csv"
    if not vector_combined_path.is_file():
        raise ValueError(f"Missing complete Wave 1 vector output: {vector_combined_path}")
    declared_vector_path = vector_manifest.get("combined_path")
    if declared_vector_path and Path(declared_vector_path).resolve() != vector_combined_path:
        raise ValueError("Wave 1 vector manifest combined_path does not match the supplied directory.")
    if vector_manifest.get("combined_sha256") != file_sha256(vector_combined_path):
        raise ValueError("Wave 1 vector combined.csv hash does not match its manifest.")
    vector_manifest_constructs = {
        str(item["construct_id"]): item
        for item in vector_manifest.get("constructs", [])
        if isinstance(item, Mapping) and isinstance(item.get("construct_id"), str)
    }
    if set(vector_manifest_constructs) != set(WAVE1_IDS):
        raise ValueError("Wave 1 vector manifest does not contain exactly the repaired construct set.")

    vector_records_by_construct: dict[str, tuple[PromptRecord, ...]] = {}
    vector_sources: dict[str, dict[str, Any]] = {}
    for entry in entries:
        spec = specs[entry.construct_id]
        raw_entry = raw_entries.get(entry.construct_id, {})
        plan_reference = raw_entry.get("generation_plan_path") or (
            f"generation_plans/wave1_{entry.construct_id}_v1.json"
        )
        plan_path = (registry_path.parent / str(plan_reference)).resolve()
        plan = load_generation_plan(plan_path, spec)
        expected = dry_run_summary(plan, model_aliases={"luna"}, splits=VECTOR_SPLITS)
        vector_path = vector_root / f"{entry.construct_id}.csv"
        if not vector_path.is_file():
            raise ValueError(f"Missing complete Wave 1 vector file: {vector_path}")
        records = tuple(load_prompt_records(vector_path))
        validate_prompt_records(records, {entry.construct_id: spec}, require_all_splits=False)
        normalized_records: list[PromptRecord] = []
        normalized_count = 0
        for record in records:
            prompt_text, wrapper_normalized = normalize_probe_prompt_wrapper(
                record.prompt_text,
                probe_prompt_template=spec.probe_prompt_template,
            ) if record.prompt_role == "probe" else (record.prompt_text, False)
            if wrapper_normalized:
                metadata = dict(record.metadata)
                metadata["probe_wrapper_normalization_version"] = PROBE_WRAPPER_NORMALIZATION_VERSION
                metadata["probe_wrapper_normalization_applied"] = True
                metadata["probe_wrapper_source_sha256"] = hashlib.sha256(
                    record.prompt_text.encode("utf-8")
                ).hexdigest()
                record = replace(record, prompt_text=prompt_text, metadata=metadata)
                normalized_count += 1
            normalized_records.append(record)
        records = tuple(normalized_records)
        observed_counts = Counter(record.split for record in records)
        expected_counts = Counter(expected["records_by_split"])
        if set(observed_counts) != VECTOR_SPLITS or observed_counts != expected_counts:
            raise ValueError(
                f"{entry.construct_id} vector counts are {dict(sorted(observed_counts.items()))}; "
                f"expected {dict(sorted(expected_counts.items()))}."
            )
        if any(record.prompt_role != "probe" for record in records):
            raise ValueError(f"{entry.construct_id} vector file contains a non-probe row.")
        manifest_item = vector_manifest_constructs[entry.construct_id]
        expected_source_plan_hash = _canonical_sha256(plan)
        if manifest_item.get("source_plan_sha256") != expected_source_plan_hash:
            raise ValueError(f"{entry.construct_id} vector manifest has a stale source plan hash.")
        effective_plan_hash = manifest_item.get("plan_sha256")
        if not isinstance(effective_plan_hash, str) or len(effective_plan_hash) != 64:
            raise ValueError(f"{entry.construct_id} vector manifest lacks an effective plan hash.")
        plan_hashes = {record.metadata.get("generation_plan_sha256") for record in records}
        if plan_hashes != {effective_plan_hash}:
            raise ValueError(
                f"{entry.construct_id} vector provenance does not match its vector manifest."
            )
        source_models = {record.metadata.get("source_model") for record in records}
        source_aliases = {record.metadata.get("source_model_alias") for record in records}
        if source_models != {DEFAULT_MODEL} or source_aliases != {"luna"}:
            raise ValueError(f"{entry.construct_id} vector file is not a Luna-only artifact.")
        vector_records_by_construct[entry.construct_id] = records
        vector_sources[entry.construct_id] = {
            "path": str(vector_path),
            "sha256": file_sha256(vector_path),
            "record_count": len(records),
            "split_counts": dict(sorted(observed_counts.items())),
            "plan_path": str(plan_path),
            "source_plan_sha256": expected_source_plan_hash,
            "plan_sha256": effective_plan_hash,
            "source_model": DEFAULT_MODEL,
            "source_model_alias": "luna",
            "source_provider": DEFAULT_PROVIDER,
            "probe_wrapper_normalization_version": PROBE_WRAPPER_NORMALIZATION_VERSION,
            "probe_wrapper_normalized_record_count": normalized_count,
        }

    downstream_manifest_path = downstream_root / "final_inventory_manifest.json"
    downstream_combined_path = downstream_root / "combined.csv"
    downstream_manifest = _load_object(downstream_manifest_path, label="Wave 1 downstream manifest")
    if (
        downstream_manifest.get("manifest_type") != "downstream_prompt_generation"
        or downstream_manifest.get("status") != "frozen"
        or downstream_manifest.get("run_mode") != "full"
        or downstream_manifest.get("partial") is not False
        or downstream_manifest.get("frozen") is not True
        or downstream_manifest.get("confirmatory") is not False
        or downstream_manifest.get("dry_run") is not False
    ):
        raise ValueError("Wave 1 downstream input must be a complete non-confirmatory full manifest.")
    if downstream_manifest.get("provider") != DEFAULT_PROVIDER:
        raise ValueError("Wave 1 downstream input has unexpected provider provenance.")
    if downstream_manifest.get("requested_model") != DEFAULT_MODEL:
        raise ValueError("Wave 1 downstream input has unexpected model provenance.")
    if downstream_manifest.get("reasoning_effort") != DEFAULT_REASONING_EFFORT:
        raise ValueError("Wave 1 downstream input has unexpected reasoning provenance.")
    if downstream_manifest.get("construct_ids") != list(WAVE1_IDS):
        raise ValueError("Wave 1 downstream manifest does not contain the repaired construct set.")
    declared_downstream_path = downstream_manifest.get("combined_path")
    if declared_downstream_path and Path(declared_downstream_path).resolve() != downstream_combined_path:
        raise ValueError("Wave 1 downstream manifest combined_path does not match the supplied directory.")
    if not downstream_combined_path.is_file():
        raise ValueError(f"Missing complete Wave 1 downstream output: {downstream_combined_path}")
    if downstream_manifest.get("combined_sha256") != file_sha256(downstream_combined_path):
        raise ValueError("Wave 1 downstream combined.csv hash does not match its manifest.")
    quality_gate = downstream_manifest.get("quality_gate")
    if not isinstance(quality_gate, Mapping) or quality_gate.get("approved") is not True:
        raise ValueError("Wave 1 downstream full output lacks an approved review quality gate.")

    downstream_entries = _effective_entries(
        registry_path,
        waves=[1],
        construct_ids=WAVE1_IDS,
        batch_size=int(downstream_manifest.get("batch_size", 20)),
        max_output_tokens=int(
            downstream_manifest.get("runtime_settings", {}).get("max_output_tokens", 30_000)
        ),
        model=DEFAULT_MODEL,
    )
    downstream_records = tuple(load_prompt_records(downstream_combined_path))
    downstream_by_construct: dict[str, tuple[PromptRecord, ...]] = {}
    for entry in downstream_entries:
        construct_records = tuple(
            record for record in downstream_records if record.construct_id == entry.construct_id
        )
        _validate_downstream_records(
            entry,
            construct_records,
            mode="full",
            input_price=float(downstream_manifest["runtime_settings"]["input_usd_per_million_tokens"]),
            output_price=float(downstream_manifest["runtime_settings"]["output_usd_per_million_tokens"]),
        )
        downstream_by_construct[entry.construct_id] = construct_records

    combined_records: list[PromptRecord] = []
    construct_summaries: list[dict[str, Any]] = []
    for construct_id in WAVE1_IDS:
        vector_records = vector_records_by_construct[construct_id]
        downstream_records_for_construct = downstream_by_construct[construct_id]
        combined_records.extend(vector_records)
        combined_records.extend(downstream_records_for_construct)
        construct_summaries.append(
            {
                "construct_id": construct_id,
                "vector_record_count": len(vector_records),
                "downstream_record_count": len(downstream_records_for_construct),
                "record_count": len(vector_records) + len(downstream_records_for_construct),
                "vector_split_counts": dict(sorted(Counter(record.split for record in vector_records).items())),
                "downstream_split_counts": dict(
                    sorted(Counter(record.split for record in downstream_records_for_construct).items())
                ),
            }
        )

    validate_prompt_records(combined_records, specs, require_all_splits=True)
    _assert_unique_prompt_text(combined_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined.csv"
    manifest_path = output_dir / "inventory_manifest.json"
    write_prompt_records(combined_records, output_path)
    prompt_audit = audit_wave_inventory(output_path, registry_path=registry_path, wave=1)
    if not prompt_audit["passed"]:
        raise ValueError(
            "The composed Wave 1 inventory failed its prompt audit: "
            f"{prompt_audit['severe_flag_count']} severe flags."
        )
    manifest = {
        "schema_version": "0.1.0",
        "manifest_type": "wave_execution_prompt_inventory",
        "status": "frozen",
        "frozen": True,
        "confirmatory": False,
        "scope": "repaired_wave1_vector_probe_plus_downstream",
        "wave": 1,
        "construct_ids": list(WAVE1_IDS),
        "record_count": len(combined_records),
        "constructs": construct_summaries,
        "sources": {
            "registry_path": str(registry_path),
            "registry_sha256": file_sha256(registry_path),
            "vector": vector_sources,
            "downstream_manifest": _manifest_hashes(downstream_manifest_path),
            "downstream_combined": _manifest_hashes(downstream_combined_path),
            "downstream_manifest_declared_sha256": downstream_manifest.get("combined_sha256"),
        },
        "validation": {
            "construct_ids": list(WAVE1_IDS),
            "vector_split_counts": dict(
                sorted(Counter(record.split for record in combined_records if record.split in VECTOR_SPLITS).items())
            ),
            "downstream_split_counts": dict(
                sorted(Counter(record.split for record in combined_records if record.split in DOWNSTREAM_SPLITS).items())
            ),
            "global_normalized_prompt_text_unique": True,
            "prompt_record_validation": "passed",
            "downstream_validation": "passed",
            "probe_wrapper_normalization": {
                "version": PROBE_WRAPPER_NORMALIZATION_VERSION,
                "record_count": sum(
                    source["probe_wrapper_normalized_record_count"]
                    for source in vector_sources.values()
                ),
                "scope": "whitespace-only registered prefix/suffix repair",
            },
        },
        "prompt_audit": prompt_audit,
        "output_path": "combined.csv",
        "output_sha256": file_sha256(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose and audit the repaired Wave 1 inventory.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=_ROOT / "configs/construct_benchmark/construct_registry_wave1_repaired_v2.json",
    )
    parser.add_argument(
        "--vector-root",
        type=Path,
        required=True,
        help="Directory containing one complete <construct_id>.csv vector file per Wave 1 construct.",
    )
    parser.add_argument(
        "--downstream-root",
        type=Path,
        required=True,
        help="Directory containing combined.csv and final_inventory_manifest.json from full downstream generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; must not already contain files.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = compose_wave1_inventory(
        registry_path=args.registry,
        vector_root=args.vector_root,
        downstream_root=args.downstream_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
