from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from activation_analysis.activation_store import (
    ActivationVectorRecord,
    iter_activation_vectors,
    summarize_activation_dataset,
    validate_activation_run,
)
from activation_analysis.vector_analysis import collect_prompt_mean_activations


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _make_activation_run(tmp_path: Path, *, activation_site: str = "resid_post") -> Path:
    run_dir = tmp_path / "activation_run"
    layer_dir = run_dir / "activations" / "layer_12"
    layer_dir.mkdir(parents=True)
    np.save(layer_dir / "batch_000000.npy", np.arange(24, dtype=np.float32).reshape(1, 3, 8))
    _write_jsonl(
        layer_dir / "batch_000000.jsonl",
        [
            {
                "prompt_id": "paper_even",
                "activation_site": activation_site,
                "token_mode": "nonpad",
                "token_ids": [101, 102, 103],
                "token_positions": [0, 1, 2],
                "token_regions": ["scenario", "decision_question", "response_instruction"],
                "num_tokens": 3,
                "metadata": {"condition": "paper_even", "prompt_family": "realization_frame"},
            }
        ],
    )
    _write_jsonl(
        run_dir / "prompts.jsonl",
        [
            {
                "prompt_id": "paper_even",
                "prompt_text": "Prompt text",
                "metadata": {"condition": "paper_even", "split": "direction_train"},
            }
        ],
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "model": {"d_model": 8},
                "extraction": {
                    "layers": [12],
                    "activation_site": activation_site,
                    "token_mode": "nonpad",
                },
                "stats": {"total_prompts": 1, "total_shards": 1},
                "shards": [
                    {
                        "layer": 12,
                        "tensor_file": "activations/layer_12/batch_000000.npy",
                        "index_file": "activations/layer_12/batch_000000.jsonl",
                        "shape": [1, 3, 8],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_iter_activation_vectors_filters_by_layer_region_and_metadata(tmp_path: Path) -> None:
    run_dir = _make_activation_run(tmp_path)

    records = list(
        iter_activation_vectors(
            run_dir,
            layers={12},
            token_regions={"scenario", "decision_question"},
            prompt_metadata_filters={"prompt_family": {"realization_frame"}},
        )
    )

    assert all(isinstance(record, ActivationVectorRecord) for record in records)
    assert len(records) == 2
    assert records[0].vector.shape == (8,)
    assert records[0].metadata["prompt_id"] == "paper_even"
    assert records[0].metadata["token_region"] == "scenario"
    assert records[1].metadata["token_position"] == 1
    assert records[0].metadata["prompt_metadata"]["split"] == "direction_train"


def test_iter_activation_vectors_filters_activation_site_and_max_vectors(tmp_path: Path) -> None:
    run_dir = _make_activation_run(tmp_path, activation_site="mlp_out")

    assert list(iter_activation_vectors(run_dir, activation_site="resid_post")) == []
    records = list(iter_activation_vectors(run_dir, activation_site="mlp_out", max_vectors=2))

    assert len(records) == 2
    assert records[-1].metadata["activation_site"] == "mlp_out"


def test_summarize_activation_dataset_counts_vectors(tmp_path: Path) -> None:
    run_dir = _make_activation_run(tmp_path)

    summary = summarize_activation_dataset([run_dir], layers={12})

    assert summary == {
        "total_vectors": 3,
        "hidden_size": 8,
        "counts_by_layer": {"12": 3},
        "counts_by_region": {
            "decision_question": 1,
            "response_instruction": 1,
            "scenario": 1,
        },
    }


def test_collect_prompt_means_keeps_candidate_layers_separate(tmp_path: Path) -> None:
    run_dir = tmp_path / "multi_layer_run"
    prompts = [
        {
            "prompt_id": "prompt_1",
            "prompt_text": "Prompt text",
            "metadata": {"construct_id": "example", "split": "direction_train"},
        }
    ]
    _write_jsonl(run_dir / "prompts.jsonl", prompts)
    shards = []
    for layer, value in ((12, 1.0), (18, 3.0)):
        layer_dir = run_dir / "activations" / f"layer_{layer}"
        layer_dir.mkdir(parents=True)
        np.save(layer_dir / "batch_000000.npy", np.full((1, 2, 4), value, dtype=np.float32))
        _write_jsonl(
            layer_dir / "batch_000000.jsonl",
            [
                {
                    "prompt_id": "prompt_1",
                    "activation_site": "resid_post",
                    "token_mode": "nonpad",
                    "token_ids": [101, 102],
                    "token_positions": [0, 1],
                    "token_regions": ["scenario", "scenario"],
                    "num_tokens": 2,
                }
            ],
        )
        shards.append(
            {
                "layer": layer,
                "tensor_file": f"activations/layer_{layer}/batch_000000.npy",
                "index_file": f"activations/layer_{layer}/batch_000000.jsonl",
                "shape": [1, 2, 4],
            }
        )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "model": {"d_model": 4},
                "extraction": {"layers": [12, 18], "activation_site": "resid_post", "token_mode": "nonpad"},
                "stats": {"total_prompts": 1, "total_shards": 2},
                "shards": shards,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    activations = collect_prompt_mean_activations(
        run_dir,
        layers={12, 18},
        token_regions={"scenario"},
    )
    assert {(activation.prompt_id, activation.layer) for activation in activations} == {
        ("prompt_1", 12),
        ("prompt_1", 18),
    }
    by_layer = {activation.layer: activation for activation in activations}
    assert np.allclose(by_layer[12].vector, 1.0)
    assert np.allclose(by_layer[18].vector, 3.0)


def test_iter_activation_vectors_rejects_malformed_index_length(tmp_path: Path) -> None:
    run_dir = _make_activation_run(tmp_path)
    index_path = run_dir / "activations" / "layer_12" / "batch_000000.jsonl"
    row = json.loads(index_path.read_text(encoding="utf-8"))
    row["token_regions"] = ["scenario"]
    _write_jsonl(index_path, [row])

    with pytest.raises(ValueError, match="token_regions/token_ids length mismatch"):
        list(iter_activation_vectors(run_dir))


def test_iter_activation_vectors_rejects_malformed_batch_index(tmp_path: Path) -> None:
    run_dir = _make_activation_run(tmp_path)
    index_path = run_dir / "activations" / "layer_12" / "batch_000000.jsonl"
    _write_jsonl(index_path, [])

    with pytest.raises(ValueError, match="tensor batch size"):
        list(iter_activation_vectors(run_dir))


def test_validate_activation_run_rejects_invalid_tokenization_contract(tmp_path: Path) -> None:
    run_dir = _make_activation_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extraction"]["max_length"] = 128
    manifest["tokenization"] = {
        "truncation": True,
        "max_length": 128,
        "checked_prompt_count": 1,
        "max_observed_token_length": 129,
        "over_limit_count": 1,
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    errors = validate_activation_run(run_dir)

    assert any("truncation must be false" in error for error in errors)
    assert any("over_limit_count must be zero" in error for error in errors)
    assert any("exceeds max_length" in error for error in errors)
