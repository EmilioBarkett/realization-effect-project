#!/usr/bin/env python3
"""Audit candidate prompt CSVs for overlap with a reference prompt set."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


DEFAULT_REFERENCE = Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_v1.csv")
DEFAULT_CANDIDATE = Path("experiments/activation_analysis/prompts/activation_vectors/realization_vector_heldout_v1.csv")
DEFAULT_OUTPUT = Path("results/audits/heldout_prompt_overlap.csv")

FIELDNAMES = [
    "comparison_scope",
    "candidate_prompt_id",
    "reference_prompt_id",
    "candidate_split",
    "reference_split",
    "candidate_source_llm",
    "reference_source_llm",
    "overlap_type",
    "char_similarity",
    "token_jaccard",
    "ngram_jaccard",
    "candidate_text",
    "reference_text",
]

STOPWORDS = {
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
    "to",
    "with",
}


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    pair_id: str
    split: str
    source_llm: str
    text: str
    core_text: str
    normalized: str
    tokens: frozenset[str]
    ngrams: frozenset[tuple[str, ...]]


def extract_core_text(prompt_text: str) -> str:
    """Remove shared prompt wrappers so boilerplate does not dominate overlap."""
    text = prompt_text.strip()
    marker = "Scenario:"
    if marker in text:
        text = text.split(marker, maxsplit=1)[1].strip()
    for suffix in (
        "\n\nDo not answer yet.",
        "\n\nAnswer now.",
        "\n\nReturn only",
    ):
        if suffix in text:
            text = text.split(suffix, maxsplit=1)[0].strip()
    return text


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(normalized: str) -> frozenset[str]:
    return frozenset(token for token in normalized.split() if len(token) > 2 and token not in STOPWORDS)


def token_ngrams(normalized: str, n: int) -> frozenset[tuple[str, ...]]:
    tokens = [token for token in normalized.split() if token not in STOPWORDS]
    if len(tokens) < n:
        return frozenset()
    return frozenset(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def jaccard(left: frozenset, right: frozenset) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def load_prompt_records(path: Path, *, ngram_size: int) -> list[PromptRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row.")
        if "prompt_text" not in reader.fieldnames:
            raise ValueError(f"{path} is missing required column prompt_text.")
        records = []
        for index, row in enumerate(reader, start=1):
            text = row.get("prompt_text", "")
            core = extract_core_text(text)
            normalized = normalize_text(core)
            records.append(
                PromptRecord(
                    prompt_id=row.get("prompt_id") or f"{path.name}:row{index}",
                    pair_id=row.get("pair_id", ""),
                    split=row.get("split", ""),
                    source_llm=row.get("source_llm", ""),
                    text=text,
                    core_text=core,
                    normalized=normalized,
                    tokens=content_tokens(normalized),
                    ngrams=token_ngrams(normalized, ngram_size),
                )
            )
    return records


def overlap_type(
    *,
    exact: bool,
    char_similarity: float,
    token_jaccard_score: float,
    ngram_jaccard_score: float,
    char_threshold: float,
    token_threshold: float,
    ngram_threshold: float,
) -> str:
    if exact:
        return "exact_normalized"
    labels = []
    if char_similarity >= char_threshold:
        labels.append("high_char_similarity")
    if token_jaccard_score >= token_threshold:
        labels.append("high_token_overlap")
    if ngram_jaccard_score >= ngram_threshold:
        labels.append("high_ngram_overlap")
    return "+".join(labels)


def audit_overlaps(
    reference_records: list[PromptRecord],
    candidate_records: list[PromptRecord],
    *,
    char_threshold: float,
    token_threshold: float,
    ngram_threshold: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in candidate_records:
        best_row: dict[str, str] | None = None
        best_score = -1.0
        for reference in reference_records:
            exact = bool(candidate.normalized and candidate.normalized == reference.normalized)
            char_similarity = SequenceMatcher(None, candidate.normalized, reference.normalized).ratio()
            token_score = jaccard(candidate.tokens, reference.tokens)
            ngram_score = jaccard(candidate.ngrams, reference.ngrams)
            label = overlap_type(
                exact=exact,
                char_similarity=char_similarity,
                token_jaccard_score=token_score,
                ngram_jaccard_score=ngram_score,
                char_threshold=char_threshold,
                token_threshold=token_threshold,
                ngram_threshold=ngram_threshold,
            )
            score = max(char_similarity, token_score, ngram_score)
            if label and score > best_score:
                best_score = score
                best_row = {
                    "comparison_scope": "candidate_vs_reference",
                    "candidate_prompt_id": candidate.prompt_id,
                    "reference_prompt_id": reference.prompt_id,
                    "candidate_split": candidate.split,
                    "reference_split": reference.split,
                    "candidate_source_llm": candidate.source_llm,
                    "reference_source_llm": reference.source_llm,
                    "overlap_type": label,
                    "char_similarity": f"{char_similarity:.4f}",
                    "token_jaccard": f"{token_score:.4f}",
                    "ngram_jaccard": f"{ngram_score:.4f}",
                    "candidate_text": candidate.core_text,
                    "reference_text": reference.core_text,
                }
        if best_row is not None:
            rows.append(best_row)
    return rows


def audit_internal_overlaps(
    candidate_records: list[PromptRecord],
    *,
    char_threshold: float,
    token_threshold: float,
    ngram_threshold: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, candidate in enumerate(candidate_records):
        for reference in candidate_records[index + 1 :]:
            if candidate.pair_id and candidate.pair_id == reference.pair_id:
                continue
            exact = bool(candidate.normalized and candidate.normalized == reference.normalized)
            char_similarity = SequenceMatcher(None, candidate.normalized, reference.normalized).ratio()
            token_score = jaccard(candidate.tokens, reference.tokens)
            ngram_score = jaccard(candidate.ngrams, reference.ngrams)
            label = overlap_type(
                exact=exact,
                char_similarity=char_similarity,
                token_jaccard_score=token_score,
                ngram_jaccard_score=ngram_score,
                char_threshold=char_threshold,
                token_threshold=token_threshold,
                ngram_threshold=ngram_threshold,
            )
            if not label:
                continue
            rows.append(
                {
                    "comparison_scope": "candidate_internal",
                    "candidate_prompt_id": candidate.prompt_id,
                    "reference_prompt_id": reference.prompt_id,
                    "candidate_split": candidate.split,
                    "reference_split": reference.split,
                    "candidate_source_llm": candidate.source_llm,
                    "reference_source_llm": reference.source_llm,
                    "overlap_type": label,
                    "char_similarity": f"{char_similarity:.4f}",
                    "token_jaccard": f"{token_score:.4f}",
                    "ngram_jaccard": f"{ngram_score:.4f}",
                    "candidate_text": candidate.core_text,
                    "reference_text": reference.core_text,
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit candidate prompts for overlap with reference prompts.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--char-threshold", type=float, default=0.82)
    parser.add_argument("--token-threshold", type=float, default=0.55)
    parser.add_argument("--ngram-threshold", type=float, default=0.30)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--skip-internal", action="store_true", help="Skip candidate-vs-candidate overlap checks.")
    parser.add_argument("--fail-on-overlap", action="store_true")
    args = parser.parse_args()

    reference_records = load_prompt_records(args.reference, ngram_size=args.ngram_size)
    candidate_records = load_prompt_records(args.candidate, ngram_size=args.ngram_size)
    rows = audit_overlaps(
        reference_records,
        candidate_records,
        char_threshold=args.char_threshold,
        token_threshold=args.token_threshold,
        ngram_threshold=args.ngram_threshold,
    )
    if not args.skip_internal:
        rows.extend(
            audit_internal_overlaps(
                candidate_records,
                char_threshold=args.char_threshold,
                token_threshold=args.token_threshold,
                ngram_threshold=args.ngram_threshold,
            )
        )
    write_rows(args.output, rows)
    print(
        f"audited {len(candidate_records)} candidate prompts against {len(reference_records)} reference prompts; "
        f"flagged {len(rows)} candidate overlaps -> {args.output}"
    )
    if rows and args.fail_on_overlap:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
