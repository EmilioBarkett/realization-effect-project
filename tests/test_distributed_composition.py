from __future__ import annotations

import json
from pathlib import Path

import pytest

from construct_benchmark.composition import compose_worker_outputs, validate_worker_outputs
from construct_benchmark.distributed_contracts import file_sha256


def _manifest(
    request_ids: list[str],
    *,
    observation_ids: list[str] | None = None,
    complete: bool = True,
    version: str = "v1",
    **overrides: object,
) -> dict[str, object]:
    observations = observation_ids or [f"{request_id}__obs" for request_id in request_ids]
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "manifest_type": "benchmark_worker_output",
        "complete": complete,
        "expected_request_ids": request_ids,
        "expected_observation_ids": observations,
        "expected_request_count": len(request_ids),
        "expected_observation_count": len(observations),
        "completed_request_ids": request_ids if complete else [],
        "completed_observation_ids": observations if complete else [],
        "completed_request_count": len(request_ids) if complete else 0,
        "completed_observation_count": len(observations) if complete else 0,
        "parent_inventory_sha256": "parent-sha",
        "prompt_hash": "prompt-sha",
        "run_config_hash": "config-sha",
        "model_revision": "model-revision-1",
        "tokenizer_revision": "tokenizer-revision-1",
        "model": {"id": "fake-model", "revision": "model-revision-1"},
        "tokenizer": {"id": "fake-tokenizer", "revision": "tokenizer-revision-1"},
        "dtype": "float16",
        "layer": 12,
        "activation_site": "resid_post",
        "decoding": {"temperature": 0.0, "max_new_tokens": 8},
        "token_limit": 512,
        "causal_token_limit": 512,
        "prompt_format": "completion",
        "system_message": "",
        "run_mode": "test",
        "confirmatory": False,
        "artifact_version": version,
        "construct_ids": [request_ids[0].split("_")[0]],
    }
    payload.update(overrides)
    return payload


def _write_worker(
    root: Path,
    name: str,
    request_ids: list[str],
    *,
    manifest_overrides: dict[str, object] | None = None,
    rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    output = root / f"{name}.jsonl"
    output_rows = rows or [
        {"request_id": request_id, "observation_id": f"{request_id}__obs", "value": index}
        for index, request_id in enumerate(request_ids)
    ]
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows), encoding="utf-8")
    observations = [str(row["observation_id"]) for row in output_rows if "observation_id" in row]
    manifest = _manifest(request_ids, observation_ids=observations, **(manifest_overrides or {}))
    manifest["output_sha256"] = file_sha256(output)
    manifest_path = root / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return output, manifest_path


def test_complete_composition_is_sorted_hash_preserving_and_non_mutating(tmp_path: Path) -> None:
    first = _write_worker(tmp_path, "worker_a", ["a_2", "a_1"])
    second = _write_worker(tmp_path, "worker_b", ["b_1", "b_2"])
    source_bytes = [path.read_bytes() for path in (*first, *second)]

    output = tmp_path / "combined.jsonl"
    report = compose_worker_outputs(
        [first, second],
        output,
        expected_request_ids=["a_1", "a_2", "b_1", "b_2"],
        expected_observation_ids=[f"{request_id}__obs" for request_id in ("a_1", "a_2", "b_1", "b_2")],
    )
    assert report["complete"] is True
    assert report["request_count"] == 4
    assert report["source_manifest_sha256s"] == [file_sha256(first[1]), file_sha256(second[1])]
    output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["request_id"] for row in output_rows] == ["a_1", "a_2", "b_1", "b_2"]
    assert [path.read_bytes() for path in (*first, *second)] == source_bytes
    manifest = json.loads((tmp_path / "combined.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_manifest_sha256s"] == report["source_manifest_sha256s"]
    with pytest.raises(FileExistsError, match="overwrite"):
        compose_worker_outputs([first, second], output)


def test_composition_rejects_duplicate_missing_unknown_and_malformed_rows(tmp_path: Path) -> None:
    first = _write_worker(tmp_path, "worker_a", ["a_1"])
    second = _write_worker(tmp_path, "worker_b", ["b_1"])
    with pytest.raises(ValueError, match="overlap request"):
        duplicate = _write_worker(tmp_path, "worker_duplicate", ["a_1"])
        validate_worker_outputs([first, duplicate])
    with pytest.raises(ValueError, match="missing=.*b_1"):
        validate_worker_outputs([first], expected_request_ids=["a_1", "b_1"])
    with pytest.raises(ValueError, match="unknown=.*unknown"):
        validate_worker_outputs([first, second], expected_request_ids=["a_1", "unknown"])

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"request_id": "bad",\n', encoding="utf-8")
    malformed_manifest = tmp_path / "malformed.manifest.json"
    malformed_manifest.write_text(json.dumps(_manifest(["bad"], observation_ids=["bad__obs"])), encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_worker_outputs([(malformed, malformed_manifest)])

    missing_observation = _write_worker(
        tmp_path,
        "missing_observation",
        ["missing"],
        rows=[{"request_id": "missing", "value": 1}],
    )
    with pytest.raises(ValueError, match="missing observation"):
        validate_worker_outputs([missing_observation])


def test_composition_rejects_swapped_observation_owners(tmp_path: Path) -> None:
    worker = _write_worker(
        tmp_path,
        "swapped_owner",
        ["a_1"],
        manifest_overrides={
            "expected_observations": [
                {"observation_id": "a_1__obs", "request_id": "different_request"},
            ]
        },
    )
    with pytest.raises(ValueError, match="observation ownership"):
        validate_worker_outputs([worker])


def test_composition_rejects_incomplete_terminal_inputs_and_test_promotion(tmp_path: Path) -> None:
    incomplete = _write_worker(
        tmp_path,
        "incomplete",
        ["a_1"],
        manifest_overrides={"complete": False},
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_worker_outputs([incomplete])
    complete_test = _write_worker(tmp_path, "test", ["b_1"])
    with pytest.raises(ValueError, match="test-to-confirmatory"):
        validate_worker_outputs([complete_test], target_confirmatory=True)
    with pytest.raises(ValueError, match="test-to-confirmatory"):
        compose_worker_outputs([complete_test], tmp_path / "promotion.jsonl", target_run_mode="full")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_revision", "model-revision-2", "model_revision"),
        ("tokenizer_revision", "tokenizer-revision-2", "tokenizer_revision"),
        ("prompt_hash", "prompt-sha-2", "prompt_hash"),
        ("dtype", "float32", "dtype"),
        ("layer", 24, "layer"),
        ("activation_site", "resid_pre", "activation_site"),
        ("decoding", {"temperature": 0.7}, "decoding"),
        ("token_limit", 768, "token_limit"),
        ("prompt_format", "chat", "prompt_format"),
        ("system_message", "different system", "system_message"),
        ("schema_version", "2.0.0", "schema_version"),
    ],
)
def test_incompatible_worker_metadata_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    first = _write_worker(tmp_path, "identity_a", ["a_1"])
    second = _write_worker(
        tmp_path,
        f"identity_b_{field}",
        ["b_1"],
        manifest_overrides={field: value},
    )
    with pytest.raises(ValueError, match=message):
        validate_worker_outputs([first, second])


def test_mixed_causal_token_limit_and_v1_v2_are_rejected(tmp_path: Path) -> None:
    first = _write_worker(tmp_path, "causal_512", ["a_1"])
    second = _write_worker(
        tmp_path,
        "causal_768",
        ["b_1"],
        manifest_overrides={"token_limit": 768, "causal_token_limit": 768},
    )
    with pytest.raises(ValueError, match="causal token-limit"):
        validate_worker_outputs([first, second])

    v1 = _write_worker(tmp_path, "version_v1", ["c_1"], manifest_overrides={"artifact_version": "v1"})
    v2 = _write_worker(tmp_path, "version_v2", ["d_1"], manifest_overrides={"artifact_version": "v2"})
    with pytest.raises(ValueError, match="v1/v2"):
        validate_worker_outputs([v1, v2])
