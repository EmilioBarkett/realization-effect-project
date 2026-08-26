from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from construct_benchmark.prompts import PromptRecord, write_prompt_records  # noqa: E402
from scripts.audit_vector_pairs import audit_vector_records, main  # noqa: E402


def _record(
    prompt_id: str,
    text: str,
    *,
    split: str = "direction_train",
    pair_id: str = "pair_1",
    condition_id: str = "condition_a",
    pair_role: str | None = None,
    metadata: dict[str, object] | None = None,
) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        construct_id="demo_construct",
        split=split,
        prompt_role="probe",
        prompt_text=text,
        condition_id=condition_id,
        pair_id=pair_id,
        pair_role=pair_role if pair_role is not None else condition_id,
        prompt_family=f"demo_{split}",
        metadata=dict(metadata or {}),
    )


def test_good_minimal_pair_has_clean_metrics() -> None:
    records = [
        _record("good_a", "Alice reviews 12 samples during the weekly audit. The report remains open.", condition_id="a"),
        _record("good_b", "Alice reviews 12 samples during the weekly audit. The report is now settled.", condition_id="b"),
    ]
    summary = audit_vector_records(records)
    assert summary["hard_failure_count"] == 0
    assert summary["warning_count"] == 0
    assert summary["severe_count"] == 0
    assert summary["valid_pair_count"] == 1
    metrics = summary["pair_metrics"][0]
    assert metrics["token_jaccard"] > 0.65
    assert metrics["length_ratio"] > 0.8
    assert metrics["numeric_token_symmetric_difference"] == ()
    assert metrics["entity_like_symmetric_difference"] == ()


def test_low_overlap_pair_is_quality_flagged_not_structural_failure() -> None:
    records = [
        _record("low_a", "Alice reviews the coastal survey in March.", condition_id="a"),
        _record("low_b", "The unrelated warehouse closes after a late inspection.", condition_id="b"),
    ]
    summary = audit_vector_records(records)
    assert summary["hard_failure_count"] == 0
    assert summary["severe_count"] >= 1
    assert any(flag["flag_type"] == "low_token_jaccard" for flag in summary["flags"])


def test_changed_numbers_and_entities_are_explicitly_flagged() -> None:
    records = [
        _record("changed_a", "Alice reviews 10 samples in Boston.", split="direction_validation", condition_id="a"),
        _record("changed_b", "Bob reviews 20 samples in Boston.", split="direction_validation", condition_id="b"),
    ]
    summary = audit_vector_records(records)
    flag_types = {flag["flag_type"] for flag in summary["flags"]}
    assert "numeric_token_difference" in flag_types
    assert "entity_like_difference" in flag_types
    metrics = summary["pair_metrics"][0]
    assert set(metrics["numeric_token_symmetric_difference"]) == {"10", "20"}
    assert set(metrics["entity_like_symmetric_difference"]) >= {"alice", "bob"}


def test_malformed_pair_is_hard_structural_failure() -> None:
    records = [
        _record("malformed_a", "Alice reviews the sample.", pair_id="malformed", condition_id="a"),
    ]
    summary = audit_vector_records(records)
    assert summary["hard_failure_count"] == 1
    assert summary["valid_pair_count"] == 0
    assert summary["flags"][0]["flag_type"] == "wrong_pair_size"


def test_cross_split_duplicate_is_severe_and_reported() -> None:
    train = [
        _record("train_a", "Alice reviews the same coastal report.", condition_id="a"),
        _record("train_b", "Alice reviews a different coastal report.", condition_id="b"),
    ]
    validation = [
        _record(
            "validation_a",
            "Alice reviews the same coastal report.",
            split="direction_validation",
            pair_id="validation_pair",
            condition_id="a",
        ),
        _record(
            "validation_b",
            "Alice reviews another coastal report.",
            split="direction_validation",
            pair_id="validation_pair",
            condition_id="b",
        ),
    ]
    summary = audit_vector_records(train + validation)
    assert summary["hard_failure_count"] == 0
    assert any(flag["flag_type"] == "cross_split_exact_duplicate" for flag in summary["flags"])


def test_cross_split_audit_excludes_members_of_same_pair() -> None:
    records = [
        _record("shared_a", "Alice reviews the same report.", pair_id="shared", condition_id="a"),
        _record("shared_b", "Alice reviews a different report.", pair_id="shared", condition_id="b"),
        _record(
            "shared_a_validation",
            "Alice reviews the same report.",
            split="direction_validation",
            pair_id="shared",
            condition_id="a",
        ),
        _record(
            "shared_b_validation",
            "Alice reviews a different report.",
            split="direction_validation",
            pair_id="shared",
            condition_id="b",
        ),
    ]
    summary = audit_vector_records(records)
    assert not any(flag["scope"] == "cross_split" for flag in summary["flags"])


def test_paired_metadata_audit_checks_within_pair_and_ten_pair_balance() -> None:
    records: list[PromptRecord] = []
    for pair_index in range(10):
        position = pair_index % 5 + 1
        for condition_id in ("a", "b"):
            records.append(
                _record(
                    f"position_{pair_index}_{condition_id}",
                    f"Alice reviews report {pair_index} while the record remains open.",
                    pair_id=f"job_part_001_pair_{pair_index}",
                    condition_id=condition_id,
                    metadata={
                        "generation_job_id": "job_part_001",
                        "minority_report_position": position,
                    },
                )
            )
    summary = audit_vector_records(records)
    assert summary["hard_failure_count"] == 0
    assert summary["metadata_balance"] == [
        {
            "construct_id": "demo_construct",
            "split": "direction_train",
            "generation_job_id": "job_part_001",
            "pair_count": 10,
            "counts_by_position": {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2},
            "full_ten_pair_balance_checked": True,
        }
    ]

    records[-1] = _record(
        "position_9_b",
        "Alice reviews report 9 while the record remains open.",
        pair_id="job_part_001_pair_9",
        condition_id="b",
        metadata={
            "generation_job_id": "job_part_001",
            "minority_report_position": 4,
        },
    )
    imbalanced = audit_vector_records(records)
    assert any(flag["flag_type"] == "paired_metadata_mismatch" for flag in imbalanced["flags"])


def test_cli_loads_csv_and_jsonl_writes_summary_and_optional_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _record("csv_a", "Alice reviews 12 samples.", condition_id="a"),
        _record("csv_b", "Alice reviews 12 reports.", condition_id="b"),
    ]
    csv_path = tmp_path / "vector.csv"
    jsonl_path = tmp_path / "vector.jsonl"
    write_prompt_records(records, csv_path)
    write_prompt_records(records, jsonl_path)
    summary_path = tmp_path / "summary.json"
    flags_path = tmp_path / "flags.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_vector_pairs.py",
            "--input",
            str(csv_path),
            "--summary-output",
            str(summary_path),
            "--flags-output",
            str(flags_path),
        ],
    )
    main()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_record_count"] == 2
    assert flags_path.is_file()
