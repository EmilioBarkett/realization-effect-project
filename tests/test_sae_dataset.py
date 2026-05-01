from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sae.config import SAEDatasetConfig
from sae.dataset import iter_activation_vectors, summarize_activation_dataset
from sae.features import SAEFeatureAnalysisNotImplementedError, top_activating_examples
from sae.metrics import planned_metric_names
from sae.training import SAETrainingNotImplementedError, train_sae


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _make_activation_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "activation_run"
    layer_dir = run_dir / "activations" / "layer_12"
    layer_dir.mkdir(parents=True)
    tensor_path = layer_dir / "batch_000000.npy"
    index_path = layer_dir / "batch_000000.jsonl"

    np.save(tensor_path, np.arange(24, dtype=np.float32).reshape(1, 3, 8))
    _write_jsonl(
        index_path,
        [
            {
                "prompt_id": "paper_even",
                "activation_site": "resid_post",
                "token_mode": "nonpad",
                "token_ids": [101, 102, 103],
                "token_positions": [0, 1, 2],
                "token_regions": ["scenario", "decision_question", "response_instruction"],
                "num_tokens": 3,
                "metadata": {"condition": "paper_even"},
            }
        ],
    )
    _write_jsonl(
        run_dir / "prompts.jsonl",
        [
            {
                "prompt_id": "paper_even",
                "prompt_text": "Prompt text",
                "metadata": {"condition": "paper_even"},
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
                    "activation_site": "resid_post",
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


def test_iter_activation_vectors_filters_by_layer_and_region(tmp_path: Path) -> None:
    run_dir = _make_activation_run(tmp_path)

    records = list(
        iter_activation_vectors(
            run_dir,
            layers={12},
            token_regions={"scenario", "decision_question"},
        )
    )

    assert len(records) == 2
    assert records[0].vector.shape == (8,)
    assert records[0].metadata["prompt_id"] == "paper_even"
    assert records[0].metadata["token_region"] == "scenario"
    assert records[1].metadata["token_position"] == 1


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


def test_sae_dataset_config_round_trips_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "sae_config.json"
    config_path.write_text(
        json.dumps(
            {
                "activation_runs": ["results/residual_streams/example"],
                "layers": [12, 18],
                "token_regions": ["scenario"],
                "activation_site": "resid_post",
                "max_vectors": 100,
            }
        ),
        encoding="utf-8",
    )

    config = SAEDatasetConfig.from_json(config_path)

    assert config.activation_runs == (Path("results/residual_streams/example"),)
    assert config.layers == (12, 18)
    assert config.token_regions == ("scenario",)
    assert config.to_json_dict()["max_vectors"] == 100


def test_train_sae_placeholder_is_explicit() -> None:
    try:
        train_sae()
    except SAETrainingNotImplementedError as exc:
        assert "not implemented yet" in str(exc)
    else:  # pragma: no cover - defensive.
        raise AssertionError("train_sae should be an explicit placeholder")


def test_sae_metric_plan_names_future_metrics() -> None:
    assert planned_metric_names() == [
        "reconstruction_mse",
        "mean_active_features",
        "fraction_active_features",
        "feature_activation_counts",
    ]


def test_feature_analysis_placeholder_is_explicit() -> None:
    try:
        top_activating_examples()
    except SAEFeatureAnalysisNotImplementedError as exc:
        assert "not implemented yet" in str(exc)
    else:  # pragma: no cover - defensive.
        raise AssertionError("top_activating_examples should be an explicit placeholder")
