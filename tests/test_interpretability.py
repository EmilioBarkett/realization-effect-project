from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from emotion_activation.log_residuals import (
    _batched,
    _build_run_name,
    _load_behavioral_summaries,
    _load_prompt_csv,
    _parse_layers,
    _with_token_regions,
    _write_batch,
    _write_manifest,
    PromptRecord,
)
from emotion_activation.activation_store import load_activation_run, validate_activation_run
from emotion_activation.emotion_probes import (
    load_emotion_probe_records,
    write_emotion_probe_csv,
)
from emotion_activation.openrouter_prompt_generation import (
    generate_prompt_csv,
    iter_generation_jobs,
    load_generation_plan,
    pilot_plan_one_job_per_cell,
    rows_for_job,
    validate_unique_prompt_ids,
)
from emotion_activation.residual_streams import BatchResiduals, ResidualStreamLogger
from realization_effect.runner import (
    DEFAULT_GENERATION_OUTPUT,
    GENERATION_PROMPT_VERSION,
    build_prompt,
)


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


def test_load_behavioral_summaries_groups_results_by_condition(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "condition,prompt_version,parsed_wager,log_wager,risk_profile,model\n"
        "paper_even,absolute,100,4.605,2,model-a\n"
        "paper_even,absolute,200,5.298,4,model-b\n"
        "paper_even,balance,900,6.802,5,model-c\n",
        encoding="utf-8",
    )

    summaries = _load_behavioral_summaries(results_csv, prompt_version="absolute", enabled=True)

    assert summaries["paper_even"]["rows"] == 2
    assert summaries["paper_even"]["mean_wager"] == 150
    assert summaries["paper_even"]["mean_risk_profile"] == 3
    assert summaries["paper_even"]["models"] == ["model-a", "model-b"]


def test_generation_prompt_version_is_reserved_for_prompt_generation() -> None:
    assert GENERATION_PROMPT_VERSION == "generation"
    assert DEFAULT_GENERATION_OUTPUT == Path(
        "experiments/emotion_activation/prompts/final/final_inference_prompts_v1.csv"
    )

    with pytest.raises(ValueError, match="Unsupported prompt_version"):
        build_prompt("paper", 0, prompt_version=GENERATION_PROMPT_VERSION)


def test_with_token_regions_labels_emotion_prompt_spans() -> None:
    record = PromptRecord(
        prompt_id="regret__positive",
        prompt_text=(
            "Read the following short scenario.\n\n"
            "Scenario:\n"
            "A visitor regrets a costly choice.\n\n"
            "Do not answer yet. Continue processing the scenario."
        ),
        metadata={"emotion": "regret"},
    )

    annotated = _with_token_regions(record, "auto")

    regions = annotated.metadata["prompt_regions"]
    assert [region["label"] for region in regions] == [
        "wrapper",
        "scenario",
        "processing_instruction",
    ]
    scenario = regions[1]
    assert annotated.prompt_text[scenario["start"] : scenario["end"]] == "A visitor regrets a costly choice."


def test_with_token_regions_labels_generated_scenario_without_emotion_metadata() -> None:
    record = PromptRecord(
        prompt_id="neutral_baseline",
        prompt_text=(
            "Read the following short scenario.\n\n"
            "Scenario:\n"
            "A city employee compares two vendor quotes before preparing a summary.\n\n"
            "Do not answer yet. Continue processing the scenario."
        ),
        metadata={"emotion": "", "concept_axis": "neutral"},
    )

    annotated = _with_token_regions(record, "auto")

    regions = annotated.metadata["prompt_regions"]
    assert [region["label"] for region in regions] == [
        "wrapper",
        "scenario",
        "processing_instruction",
    ]
    scenario = regions[1]
    assert (
        annotated.prompt_text[scenario["start"] : scenario["end"]]
        == "A city employee compares two vendor quotes before preparing a summary."
    )


def test_with_token_regions_labels_realization_prompt_spans() -> None:
    record = PromptRecord(
        prompt_id="paper_even",
        prompt_text=(
            "You are currently on a casino visit. "
            "How much do you want to wager in your next session?"
            "\nRespond with two integers on separate lines."
        ),
        metadata={"condition": "paper_even"},
    )

    annotated = _with_token_regions(record, "auto")

    assert [region["label"] for region in annotated.metadata["prompt_regions"]] == [
        "scenario",
        "decision_question",
        "response_instruction",
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
        token_regions=[["scenario", "decision_question"], ["scenario", "scenario", "response_instruction"]],
        hidden_states_by_layer={2: FakeTensor(tensor)},
        activation_site="resid_post",
    )

    shards = _write_batch(tmp_path, 0, records, batch, layers=[2])

    assert shards == [
        {
            "layer": 2,
            "tensor_file": "activations/layer_02/batch_000000.npy",
            "index_file": "activations/layer_02/batch_000000.jsonl",
            "shape": [2, 3, 4],
            "dtype": "float16",
        }
    ]
    saved_tensor = np.load(tmp_path / shards[0]["tensor_file"])
    assert saved_tensor.shape == (2, 3, 4)
    assert saved_tensor.dtype == np.float16

    index_rows = [
        json.loads(line)
        for line in (tmp_path / shards[0]["index_file"]).read_text(encoding="utf-8").splitlines()
    ]
    assert index_rows[0]["prompt_id"] == "prompt_a"
    assert index_rows[0]["activation_site"] == "resid_post"
    assert index_rows[0]["token_regions"] == ["scenario", "decision_question"]
    assert index_rows[1]["token_ids"] == [201, 202, 203]


def test_write_batch_filters_included_token_regions(tmp_path: Path) -> None:
    class FakeTensor:
        def __init__(self, array: np.ndarray) -> None:
            self._array = array
            self.shape = array.shape

        def numpy(self) -> np.ndarray:
            return self._array

    records = [
        PromptRecord("prompt_a", "Prompt A", {}),
        PromptRecord("prompt_b", "Prompt B", {}),
    ]
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    batch = BatchResiduals(
        prompt_ids=["prompt_a", "prompt_b"],
        token_ids=[[101, 102, 103], [201, 202, 203]],
        token_positions=[[0, 1, 2], [0, 1, 2]],
        token_regions=[["wrapper", "scenario", "processing_instruction"], ["scenario", "response_instruction", "scenario"]],
        hidden_states_by_layer={2: FakeTensor(tensor)},
        token_mode="nonpad",
        activation_site="resid_post",
    )

    shards = _write_batch(
        tmp_path,
        0,
        records,
        batch,
        layers=[2],
        storage_dtype="float32",
        include_token_regions={"scenario"},
    )

    saved_tensor = np.load(tmp_path / shards[0]["tensor_file"])
    assert saved_tensor.shape == (2, 2, 4)
    np.testing.assert_array_equal(saved_tensor[0, 0], tensor[0, 1])
    np.testing.assert_array_equal(saved_tensor[0, 1], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(saved_tensor[1, 0], tensor[1, 0])
    np.testing.assert_array_equal(saved_tensor[1, 1], tensor[1, 2])

    index_rows = [
        json.loads(line)
        for line in (tmp_path / shards[0]["index_file"]).read_text(encoding="utf-8").splitlines()
    ]
    assert index_rows[0]["token_ids"] == [102]
    assert index_rows[0]["token_regions"] == ["scenario"]
    assert index_rows[0]["num_tokens"] == 1
    assert index_rows[1]["token_ids"] == [201, 203]
    assert index_rows[1]["token_regions"] == ["scenario", "scenario"]


def test_write_manifest_records_extraction_contract(tmp_path: Path) -> None:
    args = argparse.Namespace(
        run_name="tiny-run",
        max_length=128,
        batch_size=2,
        activation_site="resid_post",
        token_mode="final",
        token_region_strategy="auto",
        include_token_regions=None,
        storage_dtype="float16",
        block_path="model.layers",
        local_files_only=True,
        dtype="float32",
        device="cpu",
        conditions_csv="configs/realization_effect/conditions.csv",
        emotion_config=None,
        prompt_csv=None,
        results_csv="results/results.csv",
        no_results_join=False,
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
        resolved_block_path="model.layers",
        device="cpu",
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
    assert manifest["extraction"]["activation_site"] == "resid_post"
    assert manifest["extraction"]["token_mode"] == "final"
    assert manifest["extraction"]["token_region_strategy"] == "auto"
    assert manifest["extraction"]["include_token_regions"] is None
    assert manifest["extraction"]["storage_dtype"] == "float16"
    assert manifest["extraction"]["block_path"] == "model.layers"
    assert manifest["input"]["results_csv"] == "results/results.csv"
    assert manifest["stats"] == {"total_prompts": 3, "total_shards": 2}


def test_activation_store_validates_written_run(tmp_path: Path) -> None:
    class FakeTensor:
        def __init__(self, array: np.ndarray) -> None:
            self._array = array
            self.shape = array.shape

        def numpy(self) -> np.ndarray:
            return self._array

    records = [PromptRecord("prompt_a", "Prompt A", {"condition": "paper_even"})]
    batch = BatchResiduals(
        prompt_ids=["prompt_a"],
        token_ids=[[101, 102]],
        token_positions=[[0, 1]],
        token_regions=[["scenario", "decision_question"]],
        hidden_states_by_layer={2: FakeTensor(np.ones((1, 2, 4), dtype=np.float32))},
        token_mode="nonpad",
        activation_site="resid_post",
    )
    shards = _write_batch(tmp_path, 0, records, batch, layers=[2])
    (tmp_path / "prompts.jsonl").write_text(
        "\n".join(json.dumps(record.__dict__, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        run_name="tiny-run",
        max_length=128,
        batch_size=1,
        activation_site="resid_post",
        token_mode="nonpad",
        token_region_strategy="auto",
        include_token_regions=None,
        storage_dtype="float16",
        block_path="model.layers",
        local_files_only=True,
        dtype="float32",
        device="cpu",
        conditions_csv="configs/realization_effect/conditions.csv",
        emotion_config=None,
        prompt_csv=None,
        results_csv="results/results.csv",
        no_results_join=False,
        prompt_version="absolute",
        prompt_column="prompt_text",
        id_column=None,
        limit=1,
    )
    logger = argparse.Namespace(
        model_id="local/model",
        tokenizer_id="local/tokenizer",
        num_transformer_layers=2,
        d_model=4,
        resolved_block_path="model.layers",
        device="cpu",
    )
    _write_manifest(
        tmp_path,
        args,
        logger,  # type: ignore[arg-type]
        layers=[2],
        shards=shards,
        total_prompts=1,
    )

    run = load_activation_run(tmp_path)

    assert validate_activation_run(tmp_path) == []
    assert run.shards[0].shape == (1, 2, 4)
    assert run.shards[0].dtype == "float16"
    assert next(run.iter_layer_arrays(2))[1].shape == (1, 2, 4)


def test_build_run_name_is_deterministic() -> None:
    args = argparse.Namespace(
        model_id="models/tiny-model",
        tokenizer_id=None,
        revision=None,
        layers=[1, 3],
        activation_site="resid_post",
        token_mode="nonpad",
        token_region_strategy="auto",
        include_token_regions=None,
        storage_dtype="float16",
        prompt_csv=None,
        emotion_config=None,
        conditions_csv="configs/realization_effect/conditions.csv",
        prompt_version="absolute",
    )
    records = [PromptRecord("paper_even", "Prompt text", {})]

    assert _build_run_name(args, records) == _build_run_name(args, records)
    assert _build_run_name(args, records).startswith(
        "tiny-model__prompt-absolute__layers-1-3__site-resid_post__tokens-nonpad__store-float16__"
    )


def test_build_run_name_uses_emotion_config_when_present() -> None:
    args = argparse.Namespace(
        model_id="models/tiny-model",
        tokenizer_id=None,
        revision=None,
        layers=[12],
        activation_site="resid_post",
        token_mode="final",
        token_region_strategy="auto",
        include_token_regions=None,
        storage_dtype="float16",
        prompt_csv=None,
        emotion_config="configs/emotion_activation/emotions_initial.json",
        conditions_csv="configs/realization_effect/conditions.csv",
        prompt_version="absolute",
    )
    records = [PromptRecord("regret__positive", "Prompt text", {})]

    run_name = _build_run_name(args, records)

    assert run_name.startswith(
        "tiny-model__prompt-emotions_initial__layers-12__site-resid_post__tokens-final__store-float16__"
    )


def test_emotion_probe_config_exports_positive_and_control_rows(tmp_path: Path) -> None:
    records = load_emotion_probe_records(Path("configs/emotion_activation/emotions_initial.json"))

    assert len(records) == 16
    assert {record.metadata["contrast_role"] for record in records} == {"positive", "control"}
    assert {record.metadata["emotion"] for record in records} == {
        "regret",
        "frustration",
        "desperation",
        "temptation",
        "anxiety",
        "caution",
        "relief",
        "calm",
    }
    assert records[0].prompt_id == "regret__positive"
    assert "Do not answer yet" in records[0].prompt_text

    output_path = tmp_path / "emotion_probes.csv"
    write_emotion_probe_csv(records, output_path)
    exported = output_path.read_text(encoding="utf-8")
    assert "prompt_id,prompt_text,emotion" in exported
    assert "calm__control" in exported


def test_empty_emotion_metadata_uses_realization_regions() -> None:
    prompt = (
        "You are currently on a casino visit. So far during this visit you have lost 60 CHF. "
        "How much do you want to wager in your next session?\n"
        "Respond with one integer."
    )
    record = _with_token_regions(
        PromptRecord("realization_prompt", prompt, {"emotion": "", "prompt_family": "realization"}),
        "auto",
    )

    assert [region["label"] for region in record.metadata["prompt_regions"]] == [
        "scenario",
        "decision_question",
        "response_instruction",
    ]


def test_general_emotion_probe_config_exports_variants(tmp_path: Path) -> None:
    records = load_emotion_probe_records(Path("configs/emotion_activation/emotions_general_v2.json"))

    assert len(records) == 48
    assert {record.metadata["contrast_role"] for record in records} == {"positive", "control"}
    assert len({record.metadata["variant_id"] for record in records}) == 24
    assert records[0].prompt_id == "regret__work_choice__positive"
    assert "casino" not in "\n".join(record.prompt_text.lower() for record in records)

    output_path = tmp_path / "general_emotion_probes.csv"
    write_emotion_probe_csv(records, output_path)
    exported = output_path.read_text(encoding="utf-8")
    assert "variant_id" in exported
    assert "anxiety__travel_delay__control" in exported


def test_final_prompt_generation_plan_expands_balanced_model_jobs() -> None:
    plan = load_generation_plan(Path("configs/emotion_activation/final_inference_prompt_generation_v1.json"))

    jobs = list(iter_generation_jobs(plan, limit_jobs=12))

    assert jobs
    assert {job.model_alias for job in jobs} == {"codex"}
    assert jobs[0].metadata["emotion"] == "regret"
    assert jobs[0].metadata["risk_orientation"] == "neutral"
    assert jobs[0].count == 10
    assert jobs[0].batch_id.startswith("final_inference_prompt_generation_v1__codex__")


def test_final_prompt_generation_plan_can_sample_all_cells() -> None:
    plan = load_generation_plan(Path("configs/emotion_activation/final_inference_prompt_generation_v1.json"))

    pilot = pilot_plan_one_job_per_cell(plan)
    jobs = list(iter_generation_jobs(pilot))

    assert len(jobs) == len(plan["models"]) * len(plan["cells"])
    assert {job.count for job in jobs} == {1}
    assert {job.cell["cell_id"] for job in jobs} == {cell["cell_id"] for cell in plan["cells"]}


def test_openrouter_prompt_generation_writes_csv_with_fake_client(tmp_path: Path) -> None:
    plan = {
        "name": "tiny_plan",
        "default_count_per_cell_per_model": 2,
        "models": [{"alias": "model_a", "model": "provider/model-a"}],
        "generation": {"temperature": 0.7, "seed": 11},
        "cells": [
            {
                "cell_id": "general_risk",
                "prompt_family": "tiny_family",
                "domain": "general",
                "concept_axis": "risk",
                "emotion": "none",
                "risk_orientation": "risk_seeking",
                "risk_intensity": "medium",
                "casino_context": "none",
                "control_type": "none",
                "contrast_role": "positive",
                "expected_feature": "risk_orientation",
            }
        ],
    }

    def fake_request(model_id, messages, options):
        assert model_id == "provider/model-a"
        assert options["api_key"] == "test-key"
        assert "general_risk" in messages[1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "prompts": [
                                    {
                                        "variant_id": "variance_choice",
                                        "prompt_text": (
                                            "Read the following short scenario.\n\n"
                                            "Scenario:\n"
                                            "A planner chooses the option with the wider range of outcomes.\n\n"
                                            "Do not answer yet. Continue processing the scenario."
                                        ),
                                        "notes": "risk-seeking without loss",
                                    },
                                    {
                                        "variant_id": "upside_focus",
                                        "prompt_text": (
                                            "Read the following short scenario.\n\n"
                                            "Scenario:\n"
                                            "A designer selects the prototype with the largest possible upside "
                                            "after comparing equal averages.\n\n"
                                            "Do not answer yet. Continue processing the scenario."
                                        ),
                                        "notes": "upside salience",
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }

    output = tmp_path / "generated.csv"
    written = generate_prompt_csv(plan, output, request_fn=fake_request, api_key="test-key")

    rows = list(csv.DictReader(output.open("r", newline="", encoding="utf-8")))
    assert written == 2
    assert len(rows) == 2
    assert rows[0]["source"] == "openrouter_generated"
    assert rows[0]["source_llm"] == "model_a"
    assert rows[0]["risk_orientation"] == "risk_seeking"
    assert rows[0]["realization_frame"] == "none"
    assert rows[0]["outcome_valence"] == "none"
    assert rows[0]["behavior_target"] == "none"
    assert rows[0]["prompt_family"] == "tiny_family"


def test_openrouter_prompt_generation_can_chunk_large_jobs(tmp_path: Path) -> None:
    plan = {
        "name": "tiny_plan",
        "default_count_per_cell_per_model": 5,
        "models": [{"alias": "model_a", "model": "provider/model-a"}],
        "generation": {"max_prompts_per_request": 2},
        "cells": [
            {
                "cell_id": "neutral_baseline",
                "prompt_family": "neutral_baseline",
                "domain": "general",
                "concept_axis": "neutral",
                "emotion": "none",
                "risk_orientation": "neutral",
                "contrast_role": "baseline",
            }
        ],
    }
    calls: list[int] = []

    def fake_request(_model_id, messages, _options):
        payload = json.loads(messages[1]["content"])
        count = int(payload["count"])
        calls.append(count)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "prompts": [
                                    {
                                        "variant_id": f"variant_{len(calls)}_{index}",
                                        "prompt_text": (
                                            "Read the following short scenario.\n\n"
                                            "Scenario:\n"
                                            f"A clerk files report {len(calls)}-{index} before the office closes.\n\n"
                                            "Do not answer yet. Continue processing the scenario."
                                        ),
                                        "notes": "neutral chunk",
                                    }
                                    for index in range(count)
                                ]
                            }
                        )
                    }
                }
            ]
        }

    output = tmp_path / "generated.csv"
    written = generate_prompt_csv(plan, output, request_fn=fake_request, api_key="test-key")

    rows = list(csv.DictReader(output.open("r", newline="", encoding="utf-8")))
    assert calls == [2, 2, 1]
    assert written == 5
    assert len(rows) == 5
    assert len({row["prompt_id"] for row in rows}) == 5
    assert rows[0]["generation_batch_id"].endswith("__part_001")
    assert rows[-1]["generation_batch_id"].endswith("__part_003")


def test_openrouter_prompt_generation_rejects_duplicate_prompt_ids() -> None:
    plan = {
        "name": "tiny_plan",
        "default_count_per_cell_per_model": 2,
        "models": [{"alias": "model_a", "model": "provider/model-a"}],
        "cells": [
            {
                "cell_id": "neutral_baseline",
                "domain": "general",
                "concept_axis": "neutral",
                "emotion": "none",
                "risk_orientation": "neutral",
                "contrast_role": "baseline",
            }
        ],
    }
    job = next(iter_generation_jobs(plan))

    def fake_request(_model_id, _messages, _options):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "prompts": [
                                    {
                                        "variant_id": "non_casino",
                                        "prompt_text": (
                                            "Read the following short scenario.\n\n"
                                            "Scenario:\n"
                                            "A clerk compares two filing schedules before the afternoon meeting.\n\n"
                                            "Do not answer yet. Continue processing the scenario."
                                        ),
                                        "notes": "neutral",
                                    },
                                    {
                                        "variant_id": "non_casino",
                                        "prompt_text": (
                                            "Read the following short scenario.\n\n"
                                            "Scenario:\n"
                                            "A manager reviews two supply lists before placing the order.\n\n"
                                            "Do not answer yet. Continue processing the scenario."
                                        ),
                                        "notes": "neutral",
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }

    with pytest.raises(ValueError, match="duplicate prompt_id"):
        rows_for_job(plan, job, request_fn=fake_request, options={"api_key": "test-key"})


def test_openrouter_prompt_generation_rejects_duplicate_resume_csv(tmp_path: Path) -> None:
    output = tmp_path / "generated.csv"
    output.write_text(
        "prompt_id,prompt_text\n"
        "duplicate,Prompt A\n"
        "duplicate,Prompt B\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate prompt_id"):
        validate_unique_prompt_ids(output)


def test_openrouter_prompt_generation_rejects_label_leakage() -> None:
    plan = {
        "name": "tiny_plan",
        "default_count_per_cell_per_model": 1,
        "models": [{"alias": "model_a", "model": "provider/model-a"}],
        "cells": [
            {
                "cell_id": "general_emotion",
                "domain": "general",
                "concept_axis": "emotion",
                "emotion": "regret",
                "risk_orientation": "neutral",
                "contrast_role": "positive",
            }
        ],
    }
    job = next(iter_generation_jobs(plan))

    def fake_request(_model_id, _messages, _options):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "prompts": [
                                    {
                                        "variant_id": "leaky",
                                        "prompt_text": (
                                            "Read the following short scenario.\n\n"
                                            "Scenario:\n"
                                            "A speaker regretted skipping the final rehearsal.\n\n"
                                            "Do not answer yet. Continue processing the scenario."
                                        ),
                                        "notes": "leaky",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }

    with pytest.raises(ValueError, match="explicit emotion label"):
        rows_for_job(plan, job, request_fn=fake_request, options={"api_key": "test-key"})


def test_openrouter_prompt_generation_rejects_plan_forbidden_terms() -> None:
    plan = {
        "name": "general_emotion_risk",
        "default_count_per_cell_per_model": 1,
        "forbidden_terms": ["casino", "wager"],
        "models": [{"alias": "model_a", "model": "provider/model-a"}],
        "cells": [
            {
                "cell_id": "neutral_baseline",
                "domain": "general",
                "concept_axis": "neutral",
                "emotion": "none",
                "risk_orientation": "neutral",
                "contrast_role": "baseline",
            }
        ],
    }
    job = next(iter_generation_jobs(plan))

    def fake_request(_model_id, _messages, _options):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "prompts": [
                                    {
                                        "variant_id": "bad_domain",
                                        "prompt_text": (
                                            "Read the following short scenario.\n\n"
                                            "Scenario:\n"
                                            "A planner compares two casino schedules before a meeting.\n\n"
                                            "Do not answer yet. Continue processing the scenario."
                                        ),
                                        "notes": "forbidden domain term",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }

    with pytest.raises(ValueError, match="forbidden term"):
        rows_for_job(plan, job, request_fn=fake_request, options={"api_key": "test-key"})


def test_residual_stream_logger_validates_batch_inputs_without_model_init() -> None:
    logger = object.__new__(ResidualStreamLogger)

    with pytest.raises(ValueError, match="same length"):
        logger.extract_batch(["prompt"], [], [1])

    with pytest.raises(ValueError, match="At least one layer"):
        logger.extract_batch(["prompt"], ["prompt"], [])


def test_residual_stream_logger_smoke_extracts_from_fake_torch_model() -> None:
    import torch

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, prompts, padding, truncation, max_length, return_tensors):
            del padding, truncation, max_length, return_tensors
            encoded = []
            for prompt in prompts:
                encoded.append([len(token) + 1 for token in prompt.split()])
            max_len = max(len(row) for row in encoded)
            input_ids = []
            attention_mask = []
            for row in encoded:
                padding_len = max_len - len(row)
                input_ids.append(row + [0] * padding_len)
                attention_mask.append([1] * len(row) + [0] * padding_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    class FakeBlock(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        def forward(self, hidden):
            return hidden + self.offset

    class FakeConfig:
        num_hidden_layers = 2
        hidden_size = 4

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = FakeConfig()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([FakeBlock(1.0), FakeBlock(10.0)])

        def forward(self, input_ids, attention_mask, output_hidden_states, use_cache, return_dict):
            del attention_mask, output_hidden_states, use_cache, return_dict
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
            for block in self.model.layers:
                hidden = block(hidden)
            return {"last_hidden_state": hidden}

    logger = object.__new__(ResidualStreamLogger)
    logger._torch = torch
    logger.tokenizer = FakeTokenizer()
    logger.model = FakeModel()
    logger.device = "cpu"
    logger.block_path = "model.layers"
    logger.resolved_block_path = None
    logger.stop_after_last_requested_layer = False

    batch = logger.extract_batch(
        ["one two", "three"],
        ["prompt_a", "prompt_b"],
        [1, 2],
        token_mode="nonpad",
    )

    assert logger.resolved_block_path == "model.layers"
    assert batch.activation_site == "resid_post"
    assert batch.token_ids == [[4, 4], [6]]
    assert batch.token_positions == [[0, 1], [0]]
    assert batch.hidden_states_by_layer[1].shape == (2, 2, 4)
    assert batch.hidden_states_by_layer[2].shape == (2, 2, 4)
    assert torch.allclose(batch.hidden_states_by_layer[2][0, 0], torch.full((4,), 15.0))


def test_residual_stream_logger_final_token_mode() -> None:
    import torch

    logger = object.__new__(ResidualStreamLogger)
    logger._torch = torch

    input_ids = torch.tensor([[9, 8, 0], [7, 6, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    token_ids, token_positions, token_regions = logger._token_metadata(input_ids, attention_mask, "final")

    assert token_ids == [[8], [5]]
    assert token_positions == [[1], [2]]
    assert token_regions == [["unknown"], ["unknown"]]
