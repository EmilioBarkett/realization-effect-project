#!/usr/bin/env python3
"""Build a compact, non-confirmatory Wave 1 diagnostic bundle.

This command is intended to run beside the data on a persistent analysis
volume.  It reads raw JSONL only to make aggregate tables and stratified
samples; it never copies raw generations, residual observations, activation
arrays, checkpoints, or model weights into the bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from construct_benchmark.behavior import (  # noqa: E402
    orient_primary_outcome,
    parse_behavior_output,
    primary_outcome,
)
from construct_benchmark.config import load_construct_spec  # noqa: E402
from construct_benchmark.manifests import file_sha256  # noqa: E402


CONSTRUCT_IDS = (
    "realization_account_closure",
    "evidence_diagnosticity",
    "source_reliability",
    "persistence_continuation",
)
MODEL_LABELS = ("qwen", "mistral")
MAX_SAMPLE_ROWS = 5
MAX_TEXT_CHARS = 1200
MAX_COPY_BYTES = 5 * 1024 * 1024


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}.")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _truncate(value: Any, limit: int = MAX_TEXT_CHARS) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _load_prompt_texts(root: Path) -> dict[str, str]:
    """Load prompt text from compact inventory files without copying them."""

    prompts: dict[str, str] = {}
    candidates = list(root.glob("results/benchmark/**/prompts.jsonl"))
    candidates += list(root.glob("experiments/activation_analysis/**/*.csv"))
    candidates += list(root.glob("experiments/**/*.jsonl"))
    for path in candidates:
        try:
            if path.suffix == ".csv":
                with path.open(newline="", encoding="utf-8") as handle:
                    rows: Iterable[Mapping[str, Any]] = csv.DictReader(handle)
                    for row in rows:
                        prompt_id = str(row.get("prompt_id") or "")
                        text = str(row.get("prompt_text") or "")
                        if prompt_id and text:
                            prompts.setdefault(prompt_id, text)
            else:
                for row in _jsonl(path):
                    prompt_id = str(row.get("prompt_id") or "")
                    text = str(row.get("prompt_text") or "")
                    if prompt_id and text:
                        prompts.setdefault(prompt_id, text)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return prompts


def _spec_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for construct_id in CONSTRUCT_IDS:
        for version in ("v3", "v2", "v1"):
            candidate = root / f"configs/construct_benchmark/constructs/{construct_id}_{version}.json"
            if candidate.is_file():
                paths[construct_id] = candidate
                break
        if construct_id not in paths:
            raise FileNotFoundError(f"No Wave 1 construct spec found for {construct_id}.")
    return paths


def _parse_row(row: Mapping[str, Any], spec: Any) -> dict[str, Any]:
    parser_id = str(row.get("parser_id") or spec.parsing_rules["parser_id"])
    task_id = str(row.get("task_id") or spec.independent_behavior_task["task_id"])
    metadata = dict(row.get("task_metadata") or {})
    output_text = row.get("output_text") or ""
    parsed = parse_behavior_output(
        output_text,
        parser_id=parser_id,
        item_metadata=metadata,
        task_id=task_id,
    )
    outcome: float | None = None
    directed: float | None = None
    error = parsed.error
    if parsed.valid:
        try:
            outcome = primary_outcome(
                parsed,
                str(spec.independent_behavior_task["primary_outcome"]),
            )
            directed = orient_primary_outcome(
                spec.construct_id,
                outcome,
                metadata,
            )
        except (TypeError, ValueError) as exc:
            error = str(exc)
        # Collateral rows use a separate task and therefore need not contain
        # the construct's primary outcome.  Their parser validity is still a
        # useful accessibility signal and is retained in the tables.
    return {
        "valid": bool(parsed.valid and directed is not None),
        "parser_valid": bool(parsed.valid),
        "outcome": outcome,
        "directed_outcome": directed,
        "error": error,
        "parser_id": parser_id,
    }


def _sample(
    row: Mapping[str, Any],
    *,
    model: str,
    stage: str,
    parse_result: Mapping[str, Any],
    prompt_texts: Mapping[str, str],
    source: Path,
) -> dict[str, Any]:
    return {
        "model": model,
        "construct_id": row.get("construct_id"),
        "stage": stage,
        "source": str(source),
        "record_id": row.get("record_id"),
        "prompt_id": row.get("prompt_id"),
        "prompt_text": _truncate(prompt_texts.get(str(row.get("prompt_id")))) ,
        "output_text": _truncate(row.get("output_text")),
        "condition_id": row.get("condition_id"),
        "direction_kind": row.get("direction_kind"),
        "dose": row.get("dose"),
        "tracking_layer": row.get("tracking_layer"),
        "intervention_timing": row.get("intervention_timing"),
        "injection_applied": row.get("injection_applied"),
        "parser_id": parse_result.get("parser_id"),
        "parser_valid": parse_result.get("parser_valid"),
        "primary_valid": parse_result.get("valid"),
        "parse_error": parse_result.get("error"),
    }


def _add_sample(bucket: list[dict[str, Any]], sample: dict[str, Any]) -> None:
    if len(bucket) < MAX_SAMPLE_ROWS:
        bucket.append(sample)


def _register_rows(
    *,
    model: str,
    stage: str,
    source: Path,
    rows: list[dict[str, Any]],
    specs: Mapping[str, Any],
    prompt_texts: Mapping[str, str],
    frequencies: Counter[tuple[str, str, str, str, str, str, str, str]],
    samples: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    behavior_outcomes: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]],
) -> None:
    for row in rows:
        construct_id = str(row.get("construct_id") or "")
        if construct_id not in specs:
            continue
        parsed = _parse_row(row, specs[construct_id])
        direction_kind = str(row.get("direction_kind") or "none")
        dose = row.get("dose")
        dose_text = "none" if dose is None else f"{float(dose):g}"
        layer_text = "none" if row.get("tracking_layer") is None else str(row.get("tracking_layer"))
        outcome = parsed.get("directed_outcome")
        outcome_text = "invalid" if outcome is None else f"{float(outcome):.6g}"
        frequencies[(model, construct_id, stage, direction_kind, dose_text, layer_text, "valid" if parsed["valid"] else "invalid", outcome_text)] += 1

        sample = _sample(
            row,
            model=model,
            stage=stage,
            parse_result=parsed,
            prompt_texts=prompt_texts,
            source=source,
        )
        bucket = samples[model][construct_id]
        if not parsed["parser_valid"]:
            _add_sample(bucket["parse_failures"], sample)
        if not parsed["parser_valid"] or not str(row.get("output_text") or "").strip():
            _add_sample(bucket["accessibility_failures"], sample)
        if parsed["valid"]:
            _add_sample(bucket["successful_cases"], sample)
            if stage == "behavior_baseline" and outcome is not None:
                behavior_outcomes[(model, construct_id)].append((float(outcome), row))


def _write_outcome_tables(
    path: Path,
    frequencies: Counter[tuple[str, str, str, str, str, str, str, str]],
) -> None:
    fields = (
        "model",
        "construct_id",
        "stage",
        "direction_kind",
        "dose",
        "tracking_layer",
        "status",
        "outcome",
        "count",
    )
    rows = [dict(zip(fields, key), count=count) for key, count in sorted(frequencies.items())]
    payload = {
        "schema_version": "0.1.0",
        "manifest_type": "wave1_outcome_frequency_table",
        "confirmatory": False,
        "row_count": len(rows),
        "fields": list(fields),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manipulation_tables(root: Path, output: Path) -> None:
    """Extract the registered manipulation checks without copying raw CSVs."""

    score_roots = {
        "qwen": root / "results/benchmark/wave1_four_construct_qwen38_27b_supplemental_v3/scores_steering_nothink",
        "mistral": root / "results/benchmark/wave1_four_construct_mistral_supplemental_v3/scores_steering_final",
    }
    for model, score_root in score_roots.items():
        for construct_id in CONSTRUCT_IDS:
            source = score_root / construct_id / "summary.json"
            if not source.is_file():
                continue
            summary = _json(source)
            payload = {
                "schema_version": "0.1.0",
                "manifest_type": "wave1_steering_manipulation_checks",
                "confirmatory": False,
                "model": model,
                "construct_id": construct_id,
                "source": {"path": str(source), "sha256": _sha256(source)},
                "manipulation_checks": summary.get("manipulation_checks", {}),
            }
            target = output / "tables/manipulation_checks" / f"{model}_{construct_id}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_samples(
    output: Path,
    samples: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    behavior_outcomes: Mapping[tuple[str, str], list[tuple[float, dict[str, Any]]]],
    specs: Mapping[str, Any],
    prompt_texts: Mapping[str, str],
    source_by_model: Mapping[str, Path],
) -> None:
    for model in MODEL_LABELS:
        for construct_id in CONSTRUCT_IDS:
            bucket = samples[model][construct_id]
            values = behavior_outcomes.get((model, construct_id), [])
            counts = Counter(value for value, _ in values)
            saturation: dict[str, Any] = {
                "detected": False,
                "valid_behavior_rows": len(values),
                "unique_valid_outcomes": len(counts),
                "dominant_outcome": None,
                "dominant_share": None,
                "reason": None,
                "samples": [],
            }
            if counts:
                dominant, dominant_count = counts.most_common(1)[0]
                share = dominant_count / len(values)
                detected = len(counts) <= 2 or share >= 0.8
                saturation.update(
                    {
                        "detected": detected,
                        "dominant_outcome": dominant,
                        "dominant_share": share,
                        "reason": (
                            "at_most_two_unique_valid_outcomes"
                            if len(counts) <= 2
                            else "dominant_valid_outcome_share_at_least_0.8"
                        )
                        if detected
                        else None,
                    }
                )
                if detected:
                    for value, row in values:
                        if value == dominant and len(saturation["samples"]) < MAX_SAMPLE_ROWS:
                            parsed = _parse_row(row, specs[construct_id])
                            saturation["samples"].append(
                                _sample(
                                    row,
                                    model=model,
                                    stage="behavior_baseline",
                                    parse_result=parsed,
                                    prompt_texts=prompt_texts,
                                    source=source_by_model[model],
                                )
                            )
            payload = {
                "schema_version": "0.1.0",
                "manifest_type": "wave1_stratified_prompt_response_samples",
                "confirmatory": False,
                "model": model,
                "construct_id": construct_id,
                "categories": bucket,
                "saturation": saturation,
                "category_definitions": {
                    "parse_failures": "Rows rejected by the registered strict parser.",
                    "accessibility_failures": "Empty or parser-invalid outputs; this may overlap parse_failures.",
                    "saturation": "Baseline valid outcomes with <=2 unique values or a dominant share >=0.8.",
                    "successful_cases": "Rows with a valid oriented primary outcome.",
                },
            }
            target = output / "samples" / model / f"{construct_id}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_small_sources(root: Path, output: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    source_roots = [root / "results/benchmark", root / "results/runpod"]
    for source_root in source_roots:
        if not source_root.is_dir():
            continue
        for source in source_root.rglob("*"):
            if source == output or output in source.parents:
                continue
            if not source.is_file() or source.stat().st_size > MAX_COPY_BYTES:
                continue
            if "wave1_mistral_engineering_gate_v1" in source.parts:
                continue
            name = source.name
            suffix = source.suffix.lower()
            is_manifest = name == "manifest.json" or name.endswith(".manifest.json")
            is_summary = name.endswith("summary.json") or name == "precision_simulation.json"
            is_table = name == "heldout_pair_margins.csv"
            is_review_json = suffix == ".json" and any(
                token in str(source).lower()
                for token in ("gate", "control", "preflight", "failure", "report", "plan", "precision")
            )
            if not (is_manifest or is_summary or is_table or is_review_json):
                continue
            relative = source.relative_to(root)
            target = output / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            shutil.copy2(source, target)
            copied.append(
                {
                    "source": str(source),
                    "bundle_path": str(target.relative_to(output)),
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )

    config_root = root / "configs"
    if config_root.is_dir():
        for source in config_root.rglob("*"):
            if not source.is_file() or source.stat().st_size > MAX_COPY_BYTES:
                continue
            if source.suffix.lower() not in {".json", ".yaml", ".yml", ".toml"}:
                continue
            relative = source.relative_to(root)
            relative_text = str(relative).lower()
            wave1_construct = any(
                f"constructs/{construct_id}_v3.json" in relative_text
                for construct_id in CONSTRUCT_IDS
            )
            if not (
                wave1_construct
                or "wave1" in relative_text
                or "model_behavior_accessibility" in relative_text
                or "construct_registry_repaired_v2" in relative_text
                or "rsc_benchmark_core" in relative_text
                or "runpod_b300" in relative_text
            ):
                continue
            target = output / "configs" / relative.relative_to("configs")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(
                {
                    "source": str(source),
                    "bundle_path": str(target.relative_to(output)),
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
    return copied


def _write_checksums(output: Path) -> Path:
    checksum_path = output / "bundle_checksums.sha256"
    entries: list[str] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        entries.append(f"{_sha256(path)}  {path.relative_to(output)}")
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return checksum_path


def _build_archive(output: Path) -> Path:
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    return archive


def build_bundle(
    *,
    root: Path,
    output: Path,
    base_snapshot_commit: str,
    pod_id: str,
    volume_id: str,
    legacy_volume_id: str,
    c1_output: Path,
    c1_input: Path,
    c1_summary: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    root = root.resolve()
    output = output.resolve()
    c1_output = c1_output.resolve()
    c1_input = c1_input.resolve()
    c1_summary = c1_summary.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
    output.mkdir(parents=True)
    for directory in ("files", "configs", "metadata", "tables", "samples"):
        (output / directory).mkdir()

    specs = {construct_id: load_construct_spec(path) for construct_id, path in _spec_paths(root).items()}
    prompt_texts = _load_prompt_texts(root)
    source_index = _copy_small_sources(root, output)

    model_paths: dict[str, dict[str, Path]] = {
        "qwen": {
            "behavior": root / "results/benchmark/wave1_four_construct_qwen38_27b_supplemental_v3/behavior_nothink/behavior_eval.jsonl",
            "collateral": root / "results/benchmark/wave1_four_construct_qwen38_27b_supplemental_v3/behavior_nothink/collateral_eval.jsonl",
            "steering": root / "results/benchmark/wave1_four_construct_qwen38_27b_supplemental_v3/steering",
        },
        "mistral": {
            "behavior": root / "results/benchmark/wave1_four_construct_mistral_supplemental_v3/behavior/behavior_eval.jsonl",
            "collateral": root / "results/benchmark/wave1_four_construct_mistral_supplemental_v3/behavior/collateral_eval.jsonl",
            "steering": root / "results/benchmark/wave1_four_construct_mistral_supplemental_v3/steering_localcache",
        },
    }
    frequencies: Counter[tuple[str, str, str, str, str, str, str, str]] = Counter()
    samples: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        model: {
            construct_id: {
                "parse_failures": [],
                "accessibility_failures": [],
                "successful_cases": [],
            }
            for construct_id in CONSTRUCT_IDS
        }
        for model in MODEL_LABELS
    }
    behavior_outcomes: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)

    for model, paths in model_paths.items():
        for stage, path in (("behavior_baseline", paths["behavior"]), ("collateral_baseline", paths["collateral"])):
            if path.is_file():
                _register_rows(
                    model=model,
                    stage=stage,
                    source=path,
                    rows=_jsonl(path),
                    specs=specs,
                    prompt_texts=prompt_texts,
                    frequencies=frequencies,
                    samples=samples,
                    behavior_outcomes=behavior_outcomes,
                )
        steering_root = paths["steering"]
        for construct_id in CONSTRUCT_IDS:
            path = steering_root / f"{construct_id}.jsonl"
            if path.is_file():
                _register_rows(
                    model=model,
                    stage="steering",
                    source=path,
                    rows=_jsonl(path),
                    specs=specs,
                    prompt_texts=prompt_texts,
                    frequencies=frequencies,
                    samples=samples,
                    behavior_outcomes=behavior_outcomes,
                )

    _write_outcome_tables(output / "tables/outcome_frequency.json", frequencies)
    _write_manipulation_tables(root, output)
    _write_samples(output, samples, behavior_outcomes, specs, prompt_texts, {model: paths["behavior"] for model, paths in model_paths.items()})

    c1_manifest_path = Path(str(c1_output) + ".manifest.json")
    c1_input_manifest_path = Path(str(c1_input) + ".manifest.json")
    c1_manifest = _json(c1_manifest_path)
    c1_input_manifest = _json(c1_input_manifest_path)
    c1_metadata = {
        "schema_version": "0.1.0",
        "manifest_type": "wave1_c1_continuation_metadata",
        "confirmatory": False,
        "resumable": bool(c1_manifest.get("complete") is False),
        "partial_output": {
            "path": str(c1_output),
            "manifest_path": str(c1_manifest_path),
            "rows_observed": sum(1 for _ in c1_output.open(encoding="utf-8")),
            "sha256": _sha256(c1_output),
            "complete": c1_manifest.get("complete"),
        },
        "input": {
            "path": str(c1_input),
            "manifest_path": str(c1_input_manifest_path),
            "sha256": c1_input_manifest.get("input_sha256") or c1_input_manifest.get("sha256"),
            "expected_request_count": c1_manifest.get("expected_request_count"),
        },
        "counts": {
            key: c1_manifest.get(key)
            for key in (
                "completed_request_count",
                "expected_request_count",
                "completed_observation_count",
                "expected_observation_count",
            )
        },
        "resume_command": (
            "PYTHONPATH=src /workspace/rsc-venv/bin/python scripts/run_residual_interchange.py "
            "--model-id Qwen/Qwen3.8-27B --tokenizer-id Qwen/Qwen3.8-27B "
            "--revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 "
            f"--requests {c1_input} --output {c1_output} --layers 16,32,48 "
            "--max-length 1024 --max-new-tokens 8 --min-new-tokens 1 "
            "--prompt-format chat --device cuda --dtype bf16 --local-files-only --disable-thinking"
        ),
        "raw_output_not_in_bundle": True,
    }
    (output / "metadata/c1_continuation.json").write_text(
        json.dumps(c1_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    locations = {
        "schema_version": "0.1.0",
        "manifest_type": "wave1_artifact_location_manifest",
        "confirmatory": False,
        "base_snapshot_commit": base_snapshot_commit,
        "pod": {"pod_id": pod_id, "volume_id": volume_id, "legacy_volume_id": legacy_volume_id},
        "local_sync_policy": {
            "raw_activations": False,
            "raw_generations": False,
            "residual_interchange_observations": False,
            "checkpoints_and_weights": False,
            "reviewed_diagnostics_only": True,
        },
        "raw_artifacts": [
            {"model": "qwen", "kind": "residuals", "path": str(root / "results/benchmark/wave1_four_construct_qwen38_27b_supplemental_v3/residuals_full"), "storage": "RUNPOD_2 persistent volume", "local_sync": False},
            {"model": "qwen", "kind": "behavior_and_steering_generations", "path": str(root / "results/benchmark/wave1_four_construct_qwen38_27b_supplemental_v3"), "storage": "RUNPOD_2 persistent volume", "local_sync": False},
            {"model": "qwen", "kind": "c1_partial_residual_interchange", "path": str(c1_output), "storage": "RUNPOD_2 persistent volume", "local_sync": False, "complete": False, "resumable": True, "sha256_status": "recorded_in_c1_continuation_metadata"},
            {"model": "mistral", "kind": "residuals_and_generations", "path": str(root / "results/benchmark/wave1_four_construct_mistral_supplemental_v3"), "storage": "RUNPOD_2 persistent volume", "local_sync": False},
            {"model": "mistral", "kind": "legacy_recovery_archive", "path": "/workspace/realization-effect-project/results/benchmark/wave1_four_construct_repaired_v2/", "volume_id": legacy_volume_id, "storage": "legacy RunPod persistent volume", "access": "read_only", "local_sync": False},
        ],
        "reviewed_bundle": {
            "path_on_volume": str(output),
            "archive_path_on_volume": str(output.with_suffix(".tar.gz")),
            "contains_raw_artifacts": False,
        },
    }
    (output / "metadata/artifact_locations.json").write_text(
        json.dumps(locations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": "0.1.0",
        "manifest_type": "wave1_diagnostic_bundle",
        "confirmatory": False,
        "base_snapshot_commit": base_snapshot_commit,
        "pod_id": pod_id,
        "volume_id": volume_id,
        "legacy_volume_id": legacy_volume_id,
        "source_index": source_index,
        "included": [
            "manifests",
            "configs",
            "scored summaries",
            "outcome-frequency tables",
            "steering manipulation-check data",
            "stratified prompt/response samples",
            "C1 continuation metadata",
            "artifact-location manifest",
        ],
        "excluded": [
            "model weights and checkpoints",
            "activation arrays and direction arrays",
            "full residual-interchange observations",
            "raw generation JSONL",
            "full prompt inventories",
        ],
        "raw_source_paths_are_pointers_only": True,
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_path = _write_checksums(output)
    archive = _build_archive(output)
    archive_hash = _sha256(archive)
    archive_hash_path = Path(str(archive) + ".sha256")
    archive_hash_path.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    return archive, checksum_path, {**manifest, "archive_sha256": archive_hash}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-snapshot-commit", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--legacy-volume-id", required=True)
    parser.add_argument("--c1-output", type=Path, required=True)
    parser.add_argument("--c1-input", type=Path, required=True)
    parser.add_argument("--c1-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.c1_summary.is_file():
        raise SystemExit(f"Missing C1 summary: {args.c1_summary}")
    archive, checksums, manifest = build_bundle(
        root=args.root,
        output=args.output,
        base_snapshot_commit=args.base_snapshot_commit,
        pod_id=args.pod_id,
        volume_id=args.volume_id,
        legacy_volume_id=args.legacy_volume_id,
        c1_output=args.c1_output,
        c1_input=args.c1_input,
        c1_summary=args.c1_summary,
    )
    print(json.dumps({"bundle": str(args.output), "archive": str(archive), "checksums": str(checksums), "archive_sha256": manifest["archive_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
