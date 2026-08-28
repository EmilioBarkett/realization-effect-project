from __future__ import annotations

from pathlib import Path

import pytest

from construct_benchmark.behavior_baseline import (
    read_behavior_output,
    score_behavior_rows,
    validate_behavior_output_manifest,
)
from construct_benchmark.config import load_construct_spec, load_run_config
from construct_benchmark.prompts import PromptRecord, write_prompt_records
from scripts.run_prompt_only_behavior import execute_prompt_only_behavior


ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_PATH = ROOT / "configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json"
SPEC_PATH = ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v1.json"


def _inventory(path: Path) -> None:
    spec = load_construct_spec(SPEC_PATH)
    records: list[PromptRecord] = []
    for split in spec.paired_splits:
        for condition_id in spec.condition_ids:
            records.append(
                PromptRecord(
                    prompt_id=f"{split}__{condition_id}",
                    construct_id=spec.construct_id,
                    split=split,
                    prompt_role="probe",
                    prompt_text=f"{split} {condition_id} scenario.",
                    condition_id=condition_id,
                    pair_id=f"{split}__pair_000",
                    pair_role=condition_id,
                    prompt_family=f"{spec.construct_id}_probe",
                )
            )
    for split, prompt_role in (
        ("behavior_eval", "behavior"),
        ("steering_eval", "steering"),
        ("calibration", "calibration"),
    ):
        for index in range(8):
            outcome_valence = "gain" if index % 2 == 0 else "loss"
            records.append(
                PromptRecord(
                    prompt_id=f"{split}__item_{index:02d}",
                    construct_id=spec.construct_id,
                    split=split,
                    prompt_role=prompt_role,
                    prompt_text=f"Independent {split} decision item {index}.",
                    condition_id=None,
                    prompt_family=f"{spec.construct_id}_{prompt_role}",
                    task_id=spec.independent_behavior_task["task_id"],
                    expected_output_format=spec.independent_behavior_task["response_format"],
                    parser_id=spec.parsing_rules["parser_id"],
                    metadata={"task_metadata": {"outcome_valence": outcome_valence}},
                )
            )
    write_prompt_records(records, path)


class _FakeGenerator:
    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        type(self).instances += 1
        self.resolved_device = "cpu"
        self.last_tokenization_report = {"truncation": False, "max_length": 512}
        self.calls = 0

    def generate(self, prompt: str, **kwargs):
        del prompt, kwargs
        self.calls += 1
        return f"500\n{self.calls % 5 + 1}", None


def test_prompt_only_baseline_writes_complete_manifest_and_scores_rows(tmp_path: Path) -> None:
    run_config = load_run_config(RUN_CONFIG_PATH)
    spec = load_construct_spec(SPEC_PATH)
    inventory = tmp_path / "prompts.csv"
    output = tmp_path / "behavior.jsonl"
    _inventory(inventory)

    result = execute_prompt_only_behavior(
        run_config=run_config,
        construct_specs={spec.construct_id: spec},
        prompt_inventory=inventory,
        output=output,
        mode="full",
        generator_factory=_FakeGenerator,
    )

    assert result["complete"] is True
    rows = read_behavior_output(output)
    assert len(rows) == 8
    manifest, complete = validate_behavior_output_manifest(
        output,
        rows,
        run_config=run_config,
        construct_specs={spec.construct_id: spec},
    )
    assert complete is True
    assert manifest["expected_record_count"] == 8
    assert all(row["intervention"] == "none" for row in rows)

    parsed_rows, summary = score_behavior_rows(rows, {spec.construct_id: spec})
    assert len(parsed_rows) == 8
    assert summary["constructs"][spec.construct_id]["primary_valid_rate"] == 1.0
    assert summary["constructs"][spec.construct_id]["unique_outcome_count"] >= 2


def test_prompt_only_resume_requires_the_same_frozen_inventory(tmp_path: Path) -> None:
    run_config = load_run_config(RUN_CONFIG_PATH)
    spec = load_construct_spec(SPEC_PATH)
    inventory = tmp_path / "prompts.csv"
    output = tmp_path / "behavior.jsonl"
    _inventory(inventory)
    execute_prompt_only_behavior(
        run_config=run_config,
        construct_specs={spec.construct_id: spec},
        prompt_inventory=inventory,
        output=output,
        mode="full",
        generator_factory=_FakeGenerator,
    )

    rows = read_behavior_output(output)
    inventory.write_text(
        inventory.read_text(encoding="utf-8").replace(
            "Independent behavior_eval decision item 0.",
            "Changed item.",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt_inventory_sha256"):
        execute_prompt_only_behavior(
            run_config=run_config,
            construct_specs={spec.construct_id: spec},
            prompt_inventory=inventory,
            output=output,
            mode="full",
            resume=True,
            generator_factory=_FakeGenerator,
        )

    assert len(rows) == 8
