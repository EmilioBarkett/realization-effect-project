#!/usr/bin/env python3
"""Audit generated vector prompt pairs for structural and lexical quality.

The auditor intentionally does not call a model.  Structural problems (for
example, a missing member of a pair) are hard failures.  Metric threshold
violations and cross-split duplicates are reported as warnings or severe
flags, so prompt review can proceed without silently treating a weak pair as
valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.prompts import PromptRecord, load_prompt_records  # noqa: E402


VECTOR_SPLITS = frozenset({"direction_train", "direction_validation", "direction_heldout"})
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:%|[A-Za-z]+)?\b")
CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
SENTENCE_RE = re.compile(r"[^.!?]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)
CAPITALIZED_COMMON = frozenset(
    {
        "A",
        "An",
        "After",
        "Before",
        "For",
        "From",
        "If",
        "In",
        "On",
        "The",
        "That",
        "Then",
        "This",
        "While",
        "When",
        "Where",
        "Which",
    }
)


@dataclass(frozen=True)
class AuditThresholds:
    """Warning/severe thresholds for pair and leakage checks."""

    min_token_jaccard_warning: float = 0.65
    min_token_jaccard_severe: float = 0.45
    min_length_ratio_warning: float = 0.80
    min_length_ratio_severe: float = 0.60
    max_sentence_difference_warning: int = 1
    max_sentence_difference_severe: int = 2
    max_numeric_difference_warning: int = 0
    max_numeric_difference_severe: int = 1
    max_entity_difference_warning: int = 0
    max_entity_difference_severe: int = 1
    near_duplicate_token_jaccard: float = 0.90

    def validate(self) -> None:
        if not 0 <= self.min_token_jaccard_severe <= self.min_token_jaccard_warning <= 1:
            raise ValueError("Token Jaccard thresholds must satisfy 0 <= severe <= warning <= 1.")
        if not 0 <= self.min_length_ratio_severe <= self.min_length_ratio_warning <= 1:
            raise ValueError("Length-ratio thresholds must satisfy 0 <= severe <= warning <= 1.")
        if self.max_sentence_difference_warning < 0 or self.max_sentence_difference_severe < self.max_sentence_difference_warning:
            raise ValueError("Sentence-difference thresholds must be non-negative and ordered.")
        if self.max_numeric_difference_warning < 0 or self.max_numeric_difference_severe < self.max_numeric_difference_warning:
            raise ValueError("Numeric-difference thresholds must be non-negative and ordered.")
        if self.max_entity_difference_warning < 0 or self.max_entity_difference_severe < self.max_entity_difference_warning:
            raise ValueError("Entity-difference thresholds must be non-negative and ordered.")
        if not 0 <= self.near_duplicate_token_jaccard <= 1:
            raise ValueError("near_duplicate_token_jaccard must be between 0 and 1.")


@dataclass(frozen=True)
class PairMetrics:
    construct_id: str
    split: str
    pair_id: str
    prompt_ids: tuple[str, str]
    condition_ids: tuple[str, str]
    token_jaccard: float
    length_ratio: float
    sentence_count_difference: int
    numeric_tokens_left: tuple[str, ...]
    numeric_tokens_right: tuple[str, ...]
    numeric_token_symmetric_difference: tuple[str, ...]
    entity_tokens_left: tuple[str, ...]
    entity_tokens_right: tuple[str, ...]
    entity_like_symmetric_difference: tuple[str, ...]


@dataclass
class AuditResult:
    input_record_count: int
    vector_record_count: int
    vector_split_counts: dict[str, int]
    pair_group_count: int
    valid_pair_count: int
    flags: list[dict[str, Any]] = field(default_factory=list)
    pair_metrics: list[dict[str, Any]] = field(default_factory=list)
    metadata_balance: list[dict[str, Any]] = field(default_factory=list)

    @property
    def hard_failure_count(self) -> int:
        return sum(flag["severity"] == "hard" for flag in self.flags)

    @property
    def warning_count(self) -> int:
        return sum(flag["severity"] == "warning" for flag in self.flags)

    @property
    def severe_count(self) -> int:
        return sum(flag["severity"] == "severe" for flag in self.flags)

    def summary(self, *, thresholds: AuditThresholds, input_paths: Iterable[str] = ()) -> dict[str, Any]:
        return {
            "summary_version": "0.1.0",
            "input_paths": list(input_paths),
            "vector_splits": sorted(VECTOR_SPLITS),
            "input_record_count": self.input_record_count,
            "vector_record_count": self.vector_record_count,
            "vector_split_counts": dict(sorted(self.vector_split_counts.items())),
            "pair_group_count": self.pair_group_count,
            "valid_pair_count": self.valid_pair_count,
            "hard_failure_count": self.hard_failure_count,
            "warning_count": self.warning_count,
            "severe_count": self.severe_count,
            "flag_count": len(self.flags),
            "thresholds": asdict(thresholds),
            "pair_metrics": list(self.pair_metrics),
            "metadata_balance": list(self.metadata_balance),
            "flags": list(self.flags),
        }


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def _content_token_set(text: str) -> frozenset[str]:
    return frozenset(token for token in _tokens(text) if token not in STOPWORDS and len(token) > 1)


def _numeric_tokens(text: str) -> frozenset[str]:
    return frozenset(match.group(0).casefold() for match in NUMBER_RE.finditer(text))


def _entity_like_tokens(text: str) -> frozenset[str]:
    values: set[str] = set()
    for match in CAPITALIZED_RE.finditer(text):
        token = match.group(0)
        if token in CAPITALIZED_COMMON:
            continue
        values.add(token.casefold())
    return frozenset(values)


def _sentence_count(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, len(SENTENCE_RE.findall(text)))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _length_ratio(left: str, right: str) -> float:
    left_count = len(_tokens(left))
    right_count = len(_tokens(right))
    if left_count == 0 and right_count == 0:
        return 1.0
    if left_count == 0 or right_count == 0:
        return 0.0
    return min(left_count, right_count) / max(left_count, right_count)


def _normalized(text: str) -> str:
    return " ".join(_tokens(text))


def _metrics(left: PromptRecord, right: PromptRecord) -> PairMetrics:
    left_numbers = _numeric_tokens(left.prompt_text)
    right_numbers = _numeric_tokens(right.prompt_text)
    left_entities = _entity_like_tokens(left.prompt_text)
    right_entities = _entity_like_tokens(right.prompt_text)
    return PairMetrics(
        construct_id=left.construct_id,
        split=left.split,
        pair_id=str(left.pair_id),
        prompt_ids=(left.prompt_id, right.prompt_id),
        condition_ids=(str(left.condition_id), str(right.condition_id)),
        token_jaccard=_jaccard(_content_token_set(left.prompt_text), _content_token_set(right.prompt_text)),
        length_ratio=_length_ratio(left.prompt_text, right.prompt_text),
        sentence_count_difference=abs(_sentence_count(left.prompt_text) - _sentence_count(right.prompt_text)),
        numeric_tokens_left=tuple(sorted(left_numbers)),
        numeric_tokens_right=tuple(sorted(right_numbers)),
        numeric_token_symmetric_difference=tuple(sorted(left_numbers ^ right_numbers)),
        entity_tokens_left=tuple(sorted(left_entities)),
        entity_tokens_right=tuple(sorted(right_entities)),
        entity_like_symmetric_difference=tuple(sorted(left_entities ^ right_entities)),
    )


def _flag(
    *,
    severity: str,
    flag_type: str,
    message: str,
    left: PromptRecord | None = None,
    right: PromptRecord | None = None,
    metrics: PairMetrics | None = None,
    scope: str,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "scope": scope,
        "flag_type": flag_type,
        "message": message,
        "construct_id": (left.construct_id if left is not None else (right.construct_id if right else "")),
        "split": (left.split if left is not None else (right.split if right else "")),
        "pair_id": (left.pair_id if left is not None else (right.pair_id if right else "")) or "",
        "left_prompt_id": left.prompt_id if left is not None else "",
        "right_prompt_id": right.prompt_id if right is not None else "",
        "left_split": left.split if left is not None else "",
        "right_split": right.split if right is not None else "",
        "token_jaccard": metrics.token_jaccard if metrics is not None else None,
        "length_ratio": metrics.length_ratio if metrics is not None else None,
        "sentence_count_difference": metrics.sentence_count_difference if metrics is not None else None,
        "numeric_token_symmetric_difference": list(metrics.numeric_token_symmetric_difference) if metrics is not None else [],
        "entity_like_symmetric_difference": list(metrics.entity_like_symmetric_difference) if metrics is not None else [],
    }


def _quality_flags(left: PromptRecord, right: PromptRecord, metrics: PairMetrics, thresholds: AuditThresholds) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    def add_metric_flag(
        *,
        value: float,
        warning_violation: bool,
        severe_violation: bool,
        flag_type: str,
        message: str,
    ) -> None:
        if severe_violation:
            flags.append(_flag(severity="severe", flag_type=flag_type, message=message, left=left, right=right, metrics=metrics, scope="within_pair"))
        elif warning_violation:
            flags.append(_flag(severity="warning", flag_type=flag_type, message=message, left=left, right=right, metrics=metrics, scope="within_pair"))

    add_metric_flag(
        value=metrics.token_jaccard,
        warning_violation=metrics.token_jaccard < thresholds.min_token_jaccard_warning,
        severe_violation=metrics.token_jaccard < thresholds.min_token_jaccard_severe,
        flag_type="low_token_jaccard",
        message=f"Within-pair content-token Jaccard is {metrics.token_jaccard:.3f}.",
    )
    add_metric_flag(
        value=metrics.length_ratio,
        warning_violation=metrics.length_ratio < thresholds.min_length_ratio_warning,
        severe_violation=metrics.length_ratio < thresholds.min_length_ratio_severe,
        flag_type="length_ratio",
        message=f"Within-pair token length ratio is {metrics.length_ratio:.3f}.",
    )
    add_metric_flag(
        value=float(metrics.sentence_count_difference),
        warning_violation=metrics.sentence_count_difference > thresholds.max_sentence_difference_warning,
        severe_violation=metrics.sentence_count_difference > thresholds.max_sentence_difference_severe,
        flag_type="sentence_count_difference",
        message=f"Within-pair sentence-count difference is {metrics.sentence_count_difference}.",
    )
    numeric_difference = len(metrics.numeric_token_symmetric_difference)
    add_metric_flag(
        value=float(numeric_difference),
        warning_violation=numeric_difference > thresholds.max_numeric_difference_warning,
        severe_violation=numeric_difference > thresholds.max_numeric_difference_severe,
        flag_type="numeric_token_difference",
        message=f"Numeric-token symmetric difference is {list(metrics.numeric_token_symmetric_difference)!r}.",
    )
    entity_difference = len(metrics.entity_like_symmetric_difference)
    add_metric_flag(
        value=float(entity_difference),
        warning_violation=entity_difference > thresholds.max_entity_difference_warning,
        severe_violation=entity_difference > thresholds.max_entity_difference_severe,
        flag_type="entity_like_difference",
        message=f"Capitalized/entity-like symmetric difference is {list(metrics.entity_like_symmetric_difference)!r}.",
    )
    return flags


def _pair_key(record: PromptRecord) -> tuple[str, str, str]:
    return record.construct_id, record.split, str(record.pair_id)


def _structural_flags(records: list[PromptRecord]) -> tuple[dict[tuple[str, str, str], list[PromptRecord]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[PromptRecord]] = defaultdict(list)
    flags: list[dict[str, Any]] = []
    seen_ids: dict[str, PromptRecord] = {}
    vector_records = [record for record in records if record.split in VECTOR_SPLITS]
    for record in vector_records:
        if record.prompt_id in seen_ids:
            flags.append(
                _flag(
                    severity="hard",
                    flag_type="duplicate_prompt_id",
                    message=f"Prompt ID {record.prompt_id!r} is repeated.",
                    left=seen_ids[record.prompt_id],
                    right=record,
                    scope="structure",
                )
            )
        else:
            seen_ids[record.prompt_id] = record
        if not record.construct_id:
            flags.append(_flag(severity="hard", flag_type="missing_construct_id", message="Vector prompt has no construct_id.", right=record, scope="structure"))
        if not record.prompt_text.strip():
            flags.append(_flag(severity="hard", flag_type="empty_prompt_text", message="Vector prompt has empty prompt_text.", right=record, scope="structure"))
        if record.prompt_role != "probe":
            flags.append(_flag(severity="hard", flag_type="wrong_prompt_role", message="Vector split prompt must have prompt_role=probe.", right=record, scope="structure"))
        if not record.pair_id:
            flags.append(_flag(severity="hard", flag_type="missing_pair_id", message="Vector prompt has no pair_id.", right=record, scope="structure"))
            continue
        groups[_pair_key(record)].append(record)

    for key, pair_records in groups.items():
        if len(pair_records) != 2:
            flags.append(
                _flag(
                    severity="hard",
                    flag_type="wrong_pair_size",
                    message=f"Pair has {len(pair_records)} records; exactly two are required.",
                    left=pair_records[0] if pair_records else None,
                    right=pair_records[1] if len(pair_records) > 1 else None,
                    scope="structure",
                )
            )
            continue
        left, right = pair_records
        condition_ids = [left.condition_id, right.condition_id]
        if any(not condition_id for condition_id in condition_ids):
            flags.append(_flag(severity="hard", flag_type="missing_condition_id", message="Pair member has no condition_id.", left=left, right=right, scope="structure"))
        if len(set(condition_ids)) != 2:
            flags.append(_flag(severity="hard", flag_type="wrong_condition_count", message="Pair must contain exactly two distinct condition IDs.", left=left, right=right, scope="structure"))
        for record in pair_records:
            if record.pair_role != record.condition_id:
                flags.append(_flag(severity="hard", flag_type="pair_role_mismatch", message="pair_role must equal condition_id.", right=record, scope="structure"))
    return groups, flags


def _paired_metadata_value(record: PromptRecord, field_name: str) -> Any:
    value = record.metadata.get(field_name)
    if value is not None:
        return value
    nested = record.metadata.get("task_metadata")
    if isinstance(nested, Mapping):
        return nested.get(field_name)
    return None


def _paired_metadata_balance(
    groups: Mapping[tuple[str, str, str], list[PromptRecord]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit opt-in paired metadata and exact per-request position balance.

    Constructs that do not emit ``minority_report_position`` are ignored. A
    partial review job may contain one pair and is checked for presence and
    within-pair agreement. A full job must contain a positive multiple of five
    pairs and exactly the same number of each position 1 through 5 (for
    example, forty pairs must contain eight of each position).
    """

    flags: list[dict[str, Any]] = []
    job_groups: dict[tuple[str, str, str], dict[str, list[PromptRecord]]] = defaultdict(dict)
    for (construct_id, split, pair_id), pair_records in groups.items():
        values = [_paired_metadata_value(record, "minority_report_position") for record in pair_records]
        if not any(value is not None for value in values):
            continue
        if len(pair_records) != 2:
            continue
        if any(value is None for value in values):
            flags.append(
                _flag(
                    severity="hard",
                    flag_type="missing_paired_metadata",
                    message="Pair metadata is missing minority_report_position on one or more members.",
                    left=pair_records[0],
                    right=pair_records[1],
                    scope="metadata",
                )
            )
            continue
        if values[0] != values[1]:
            flags.append(
                _flag(
                    severity="hard",
                    flag_type="paired_metadata_mismatch",
                    message="Pair members disagree on minority_report_position.",
                    left=pair_records[0],
                    right=pair_records[1],
                    scope="metadata",
                )
            )
            continue
        if not isinstance(values[0], int) or isinstance(values[0], bool) or values[0] not in {1, 2, 3, 4, 5}:
            flags.append(
                _flag(
                    severity="hard",
                    flag_type="invalid_paired_metadata",
                    message=f"minority_report_position={values[0]!r} is outside the registered positions 1 through 5.",
                    left=pair_records[0],
                    right=pair_records[1],
                    scope="metadata",
                )
            )
            continue
        generation_job_id = str(pair_records[0].metadata.get("generation_job_id", ""))
        job_groups[(construct_id, split, generation_job_id)][pair_id] = pair_records

    summaries: list[dict[str, Any]] = []
    for (construct_id, split, generation_job_id), pairs in sorted(job_groups.items()):
        positions = [
            _paired_metadata_value(pair_records[0], "minority_report_position")
            for pair_records in pairs.values()
        ]
        counts = {str(position): positions.count(position) for position in sorted(set(positions))}
        pair_count = len(pairs)
        expected = (
            {str(position): pair_count // 5 for position in range(1, 6)}
            if pair_count > 1 and pair_count % 5 == 0
            else None
        )
        summaries.append(
            {
                "construct_id": construct_id,
                "split": split,
                "generation_job_id": generation_job_id,
                "pair_count": pair_count,
                "counts_by_position": counts,
                # Keep the historical field for consumers of the original
                # ten-pair auditor; it now means that a full balanced request
                # (including 40-pair requests) was checked.
                "full_ten_pair_balance_checked": pair_count > 1 and expected is not None,
            }
        )
        if pair_count > 1 and expected is None:
            first_pair = next(iter(pairs.values()))
            flags.append(
                _flag(
                    severity="hard",
                    flag_type="paired_metadata_invalid_request_size",
                    message=(
                        "Full paired-metadata job must contain a positive multiple of five pairs; "
                        f"received {pair_count}."
                    ),
                    left=first_pair[0],
                    right=first_pair[1],
                    scope="metadata",
                )
            )
        elif expected is not None and counts != expected:
            first_pair = next(iter(pairs.values()))
            flags.append(
                _flag(
                    severity="hard",
                    flag_type="paired_metadata_imbalance",
                    message=(
                        "Full paired-metadata job has minority_report_position "
                        f"counts={counts}; expected={expected}."
                    ),
                    left=first_pair[0],
                    right=first_pair[1],
                    scope="metadata",
                )
            )
    return summaries, flags


def _cross_split_flags(records: list[PromptRecord], thresholds: AuditThresholds) -> list[dict[str, Any]]:
    vector_records = [record for record in records if record.split in VECTOR_SPLITS and record.prompt_text.strip()]
    exact_index: dict[str, list[PromptRecord]] = defaultdict(list)
    token_index: dict[str, list[PromptRecord]] = defaultdict(list)
    for record in vector_records:
        exact_index[_normalized(record.prompt_text)].append(record)
        for token in _content_token_set(record.prompt_text):
            token_index[token].append(record)

    flags: list[dict[str, Any]] = []
    compared: set[tuple[str, str]] = set()
    for left in vector_records:
        candidate_records: dict[str, PromptRecord] = {
            record.prompt_id: record for record in exact_index[_normalized(left.prompt_text)]
        }
        for token in _content_token_set(left.prompt_text):
            candidate_records.update({record.prompt_id: record for record in token_index[token]})
        for right in candidate_records.values():
            if left.prompt_id == right.prompt_id or left.split == right.split:
                continue
            if left.construct_id == right.construct_id and left.pair_id and left.pair_id == right.pair_id:
                continue
            pair_ids = tuple(sorted((left.prompt_id, right.prompt_id)))
            if pair_ids in compared:
                continue
            compared.add(pair_ids)
            normalized_equal = bool(_normalized(left.prompt_text)) and _normalized(left.prompt_text) == _normalized(right.prompt_text)
            token_score = _jaccard(_content_token_set(left.prompt_text), _content_token_set(right.prompt_text))
            if not normalized_equal and token_score < thresholds.near_duplicate_token_jaccard:
                continue
            metrics = PairMetrics(
                construct_id=left.construct_id,
                split=left.split,
                pair_id=str(left.pair_id or ""),
                prompt_ids=(left.prompt_id, right.prompt_id),
                condition_ids=(str(left.condition_id), str(right.condition_id)),
                token_jaccard=token_score,
                length_ratio=_length_ratio(left.prompt_text, right.prompt_text),
                sentence_count_difference=abs(_sentence_count(left.prompt_text) - _sentence_count(right.prompt_text)),
                numeric_tokens_left=tuple(sorted(_numeric_tokens(left.prompt_text))),
                numeric_tokens_right=tuple(sorted(_numeric_tokens(right.prompt_text))),
                numeric_token_symmetric_difference=(),
                entity_tokens_left=(),
                entity_tokens_right=(),
                entity_like_symmetric_difference=(),
            )
            flags.append(
                _flag(
                    severity="severe",
                    flag_type="cross_split_exact_duplicate" if normalized_equal else "cross_split_near_duplicate",
                    message="Prompt text is duplicated or near-duplicated across vector splits.",
                    left=left,
                    right=right,
                    metrics=metrics,
                    scope="cross_split",
                )
            )
    return flags


def audit_vector_records(
    records: Iterable[PromptRecord],
    *,
    thresholds: AuditThresholds | None = None,
    input_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Audit canonical records and return a JSON-serializable summary."""

    thresholds = thresholds or AuditThresholds()
    thresholds.validate()
    materialized = list(records)
    vector_records = [record for record in materialized if record.split in VECTOR_SPLITS]
    groups, flags = _structural_flags(materialized)
    metadata_balance, metadata_flags = _paired_metadata_balance(groups)
    flags.extend(metadata_flags)
    metrics_rows: list[dict[str, Any]] = []
    valid_pairs = 0
    for key in sorted(groups):
        pair_records = groups[key]
        if len(pair_records) != 2 or len({record.condition_id for record in pair_records}) != 2:
            continue
        left, right = pair_records
        if any(not record.prompt_text.strip() or record.pair_role != record.condition_id for record in pair_records):
            continue
        metrics = _metrics(left, right)
        metrics_rows.append(asdict(metrics))
        valid_pairs += 1
        flags.extend(_quality_flags(left, right, metrics, thresholds))
    flags.extend(_cross_split_flags(vector_records, thresholds))
    split_counts: dict[str, int] = {}
    for record in vector_records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
    result = AuditResult(
        input_record_count=len(materialized),
        vector_record_count=len(vector_records),
        vector_split_counts=split_counts,
        pair_group_count=len(groups),
        valid_pair_count=valid_pairs,
        flags=flags,
        pair_metrics=metrics_rows,
        metadata_balance=metadata_balance,
    )
    return result.summary(thresholds=thresholds, input_paths=input_paths)


FLAG_FIELDS = [
    "severity",
    "scope",
    "flag_type",
    "message",
    "construct_id",
    "split",
    "pair_id",
    "left_prompt_id",
    "right_prompt_id",
    "left_split",
    "right_split",
    "token_jaccard",
    "length_ratio",
    "sentence_count_difference",
    "numeric_token_symmetric_difference",
    "entity_like_symmetric_difference",
]


def write_flags(path: str | Path, flags: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAG_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for flag in flags:
            row = dict(flag)
            for field_name in ("numeric_token_symmetric_difference", "entity_like_symmetric_difference"):
                if isinstance(row.get(field_name), list):
                    row[field_name] = "|".join(str(value) for value in row[field_name])
            writer.writerow(row)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit canonical vector prompt pairs and cross-split leakage.")
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Canonical CSV or JSONL inventories.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--flags-output", type=Path, default=None, help="Optional CSV flag path.")
    parser.add_argument("--min-token-jaccard-warning", type=float, default=AuditThresholds.min_token_jaccard_warning)
    parser.add_argument("--min-token-jaccard-severe", type=float, default=AuditThresholds.min_token_jaccard_severe)
    parser.add_argument("--min-length-ratio-warning", type=float, default=AuditThresholds.min_length_ratio_warning)
    parser.add_argument("--min-length-ratio-severe", type=float, default=AuditThresholds.min_length_ratio_severe)
    parser.add_argument("--max-sentence-difference-warning", type=int, default=AuditThresholds.max_sentence_difference_warning)
    parser.add_argument("--max-sentence-difference-severe", type=int, default=AuditThresholds.max_sentence_difference_severe)
    parser.add_argument("--max-numeric-difference-warning", type=int, default=AuditThresholds.max_numeric_difference_warning)
    parser.add_argument("--max-numeric-difference-severe", type=int, default=AuditThresholds.max_numeric_difference_severe)
    parser.add_argument("--max-entity-difference-warning", type=int, default=AuditThresholds.max_entity_difference_warning)
    parser.add_argument("--max-entity-difference-severe", type=int, default=AuditThresholds.max_entity_difference_severe)
    parser.add_argument("--near-duplicate-token-jaccard", type=float, default=AuditThresholds.near_duplicate_token_jaccard)
    parser.add_argument("--fail-on-severe", action="store_true", help="Exit non-zero when severe quality flags are found.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    thresholds = AuditThresholds(
        min_token_jaccard_warning=args.min_token_jaccard_warning,
        min_token_jaccard_severe=args.min_token_jaccard_severe,
        min_length_ratio_warning=args.min_length_ratio_warning,
        min_length_ratio_severe=args.min_length_ratio_severe,
        max_sentence_difference_warning=args.max_sentence_difference_warning,
        max_sentence_difference_severe=args.max_sentence_difference_severe,
        max_numeric_difference_warning=args.max_numeric_difference_warning,
        max_numeric_difference_severe=args.max_numeric_difference_severe,
        max_entity_difference_warning=args.max_entity_difference_warning,
        max_entity_difference_severe=args.max_entity_difference_severe,
        near_duplicate_token_jaccard=args.near_duplicate_token_jaccard,
    )
    records: list[PromptRecord] = []
    for path in args.input:
        records.extend(load_prompt_records(path))
    summary = audit_vector_records(records, thresholds=thresholds, input_paths=(str(path) for path in args.input))
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.flags_output is not None:
        write_flags(args.flags_output, summary["flags"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["hard_failure_count"]:
        raise SystemExit(2)
    if args.fail_on_severe and summary["severe_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
