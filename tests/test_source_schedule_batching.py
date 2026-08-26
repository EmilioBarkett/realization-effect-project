from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from construct_benchmark.config import load_construct_spec
from construct_benchmark.generation import (
    _paired_metadata_assignment,
    _validate_paired_job_schedule,
    iter_generation_request_jobs,
    load_generation_plan,
)
from construct_benchmark.prompts import PromptRecord
from scripts.audit_vector_pairs import audit_vector_records


SOURCE_SPEC_PATH = ROOT / "configs/construct_benchmark/constructs/source_reliability_v1.json"
SOURCE_PLAN_PATH = ROOT / "configs/construct_benchmark/generation_plans/wave1_source_reliability_v1.json"


def _source_plan() -> dict[str, object]:
    spec = load_construct_spec(SOURCE_SPEC_PATH)
    return load_generation_plan(SOURCE_PLAN_PATH, spec)


def test_full_forty_pair_requests_are_balanced_per_request() -> None:
    plan = copy.deepcopy(_source_plan())
    plan["generation"]["max_items_per_request"] = 40  # type: ignore[index]
    jobs = list(iter_generation_request_jobs(plan, splits={"direction_train"}))

    assert [job.count for job in jobs] == [40, 40, 20]
    for job in jobs:
        _validate_paired_job_schedule(job)
        assignments = [
            _paired_metadata_assignment(job, index)["minority_report_position"]
            for index in range(job.count)
        ]
        expected = job.count // 5
        assert {position: assignments.count(position) for position in range(1, 6)} == {
            position: expected for position in range(1, 6)
        }


def test_review_one_pair_keeps_deterministic_assignment() -> None:
    plan = _source_plan()
    jobs_a = list(
        iter_generation_request_jobs(
            plan,
            count_per_model_override=1,
            splits={"direction_train"},
        )
    )
    jobs_b = list(
        iter_generation_request_jobs(
            plan,
            count_per_model_override=1,
            splits={"direction_train"},
        )
    )

    assert len(jobs_a) == len(jobs_b) == 1
    _validate_paired_job_schedule(jobs_a[0])
    assert _paired_metadata_assignment(jobs_a[0], 0) == _paired_metadata_assignment(jobs_b[0], 0)


def test_full_nonmultiple_request_size_is_rejected() -> None:
    plan = copy.deepcopy(_source_plan())
    plan["generation"]["max_items_per_request"] = 7  # type: ignore[index]
    job = next(iter(iter_generation_request_jobs(plan, splits={"direction_train"})))

    with pytest.raises(ValueError, match="positive multiple of 5"):
        _validate_paired_job_schedule(job)


def test_pair_audit_checks_equal_position_counts_for_forty_pairs() -> None:
    records: list[PromptRecord] = []
    for pair_index in range(40):
        position = pair_index % 5 + 1
        pair_id = f"source_job_part_001_pair_{pair_index}"
        for condition_id, ending in (("reliable", "remains open"), ("unreliable", "is settled")):
            records.append(
                PromptRecord(
                    prompt_id=f"{pair_id}__{condition_id}",
                    construct_id="source_reliability",
                    split="direction_train",
                    prompt_role="probe",
                    prompt_text=f"Report {pair_index} concerns the same invented event and {ending}.",
                    condition_id=condition_id,
                    pair_id=pair_id,
                    pair_role=condition_id,
                    prompt_family="source_probe_train",
                    metadata={
                        "generation_job_id": "source_job_part_001",
                        "minority_report_position": position,
                    },
                )
            )

    summary = audit_vector_records(records)

    assert summary["hard_failure_count"] == 0
    assert summary["metadata_balance"] == [
        {
            "construct_id": "source_reliability",
            "split": "direction_train",
            "generation_job_id": "source_job_part_001",
            "pair_count": 40,
            "counts_by_position": {"1": 8, "2": 8, "3": 8, "4": 8, "5": 8},
            "full_ten_pair_balance_checked": True,
        }
    ]
