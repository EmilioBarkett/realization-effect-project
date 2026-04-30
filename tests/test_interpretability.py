from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from interpretability.log_residuals import (
    _batched,
    _load_prompt_csv,
    _parse_layers,
    _write_batch,
    _write_manifest,
    PromptRecord,
)
from interpretability.residual_streams import BatchResiduals, ResidualStreamLogger


def test_parse_layers_sorts_deduplicates_and_rejects_invalid() -> None:
    assert _parse_layers("12, 3,12") == [3, 12]

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_layers("")

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_layers("0,2")


def test_load_prompt_csv_preserves_prompt_ids_and_metadata(tmp_path: Path) -> None:
    prompt_csv = tmp_path / "prompts.csv"
    prompt_csv.write_text(
        "prompt_id,prompt_text,condition\n"
        "paper_even,How much would you wager?,paper_even\n",
        encoding="utf-8",
    )

    records = _load_prompt_csv(prompt_csv, prompt_column="prompt_text", id_column=None)

    assert records == [
        PromptRecord(
            prompt_id="paper_even",
            prompt_text="How much would you wager?",
            metadata={"prompt_id": "paper_even", "condition": "paper_even"},
        )
    ]


def test_batched_returns_indexed_chunks() -> None:
    records = [
        PromptRecord(str(index), f"prompt {index}", {})
        for index in range(5)
    ]

    chunks = list(_batched(records, batch_size=2))

    assert [(index, len(chunk)) for index, chunk in chunks] == [(0, 2), (1, 2), (2, 1)]


def test_write_batch_creates_activation_and_index_files(tmp_path: Path) -> None:
    class FakeTensor:
        def __init__(self, array: np.ndarray) -> None:
            self._array = array
            self.shape = array.shape

        def numpy(self) -> np.ndarray:
            return self._array

    records = [
        PromptRecord("prompt_a", "Prompt A", {"condition": "paper_even"}),
        PromptRecord("prompt_b", "Prompt B", {"condition": "paper_loss_large"}),
    ]
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    batch = BatchResiduals(
        prompt_ids=["prompt_a", "prompt_b"],
        token_ids=[[101, 102], [201, 202, 203]],
        token_positions=[[0, 1], [0, 1, 2]],
        hidden_states_by_layer={2: FakeTensor(tensor)},
    )

    shards = _write_batch(tmp_path, 0, records, batch, layers=[2])

    assert shards == [
        {
            "layer": 2,
            "tensor_file": "activations/layer_02/batch_000000.npy",
            "index_file": "activations/layer_02/batch_000000.jsonl",
            "shape": [2, 3, 4],
        }
    ]
    saved_tensor = np.load(tmp_path / shards[0]["tensor_file"])
    assert saved_tensor.shape == (2, 3, 4)

    index_rows = [
        json.loads(line)
        for line in (tmp_path / shards[0]["index_file"]).read_text(encoding="utf-8").splitlines()
    ]
    assert index_rows[0]["prompt_id"] == "prompt_a"
    assert index_rows[1]["token_ids"] == [201, 202, 203]


def test_write_manifest_records_extraction_contract(tmp_path: Path) -> None:
    args = argparse.Namespace(
        max_length=128,
        batch_size=2,
        local_files_only=True,
        dtype="float32",
        device="cpu",
        conditions_csv="configs/realization_effect/conditions.csv",
        prompt_csv=None,
        prompt_version="absolute",
        prompt_column="prompt_text",
        id_column=None,
        limit=3,
    )
    logger = argparse.Namespace(
        model_id="local/model",
        tokenizer_id="local/tokenizer",
        num_transformer_layers=24,
        d_model=2048,
    )

    _write_manifest(
        tmp_path,
        args,
        logger,  # type: ignore[arg-type]
        layers=[1, 3],
        shards=[{"layer": 1}, {"layer": 3}],
        total_prompts=3,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "0.1.0"
    assert manifest["model"]["model_id"] == "local/model"
    assert manifest["extraction"]["layers"] == [1, 3]
    assert manifest["stats"] == {"total_prompts": 3, "total_shards": 2}


def test_residual_stream_logger_validates_batch_inputs_without_model_init() -> None:
    logger = object.__new__(ResidualStreamLogger)

    with pytest.raises(ValueError, match="same length"):
        logger.extract_batch(["prompt"], [], [1])

    with pytest.raises(ValueError, match="At least one layer"):
        logger.extract_batch(["prompt"], ["prompt"], [])
