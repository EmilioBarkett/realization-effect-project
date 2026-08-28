from __future__ import annotations

import json
from pathlib import Path

import pytest

from construct_benchmark.staging import stage_bundle, validate_staging_bundle


def test_stage_bundle_is_hash_verified_and_idempotent(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.csv"
    config = tmp_path / "run.json"
    inventory.write_text("request_id,construct_id\nr1,c1\n", encoding="utf-8")
    config.write_text('{"model":"fixture"}\n', encoding="utf-8")

    first = stage_bundle(tmp_path / "bundle", {"inventory": inventory, "run_config": config})
    second = stage_bundle(tmp_path / "bundle", {"inventory": inventory, "run_config": config})

    assert first["manifest_type"] == "benchmark_staging_bundle"
    assert second["inputs"] == first["inputs"]
    assert validate_staging_bundle(tmp_path / "bundle")["input_count"] == 2
    assert (tmp_path / "bundle" / "inputs" / "inventory.csv").read_text(encoding="utf-8") == inventory.read_text(
        encoding="utf-8"
    )


def test_stage_bundle_refuses_changed_input_or_manifest(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text("{\"v\": 1}\n", encoding="utf-8")
    output = tmp_path / "bundle"
    stage_bundle(output, {"input": source})

    source.write_text("{\"v\": 2}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different staged file"):
        stage_bundle(output, {"input": source})

    source.write_text("{\"v\": 1}\n", encoding="utf-8")
    manifest_path = output / "staging_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {"unexpected": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different staging manifest"):
        stage_bundle(output, {"input": source})


def test_staging_rejects_unsafe_labels_and_tampered_files(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("safe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path-safe"):
        stage_bundle(tmp_path / "bundle", {"../bad": source})

    output = tmp_path / "bundle"
    stage_bundle(output, {"input": source})
    (output / "inputs" / "input.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_staging_bundle(output)
