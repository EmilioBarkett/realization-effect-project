from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_prompt_overlap.py"
    spec = importlib.util.spec_from_file_location("audit_prompt_overlap", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_core_text_removes_shared_wrapper():
    audit = _load_module()
    prompt = (
        "Read the following short scenario.\n\n"
        "Scenario:\n"
        "In a Basel casino, Mara has CHF 200 still on her card.\n\n"
        "Do not answer yet. Continue processing the scenario."
    )

    assert audit.extract_core_text(prompt) == "In a Basel casino, Mara has CHF 200 still on her card."


def test_audit_flags_near_duplicate_prompt():
    audit = _load_module()
    reference = audit.PromptRecord(
        prompt_id="old",
        pair_id="old_pair",
        split="direction_train",
        source_llm="gpt54",
        text="",
        core_text="A student sees a provisional grade before the official posting.",
        normalized=audit.normalize_text("A student sees a provisional grade before the official posting."),
        tokens=audit.content_tokens(audit.normalize_text("A student sees a provisional grade before the official posting.")),
        ngrams=audit.token_ngrams(audit.normalize_text("A student sees a provisional grade before the official posting."), 3),
    )
    candidate = audit.PromptRecord(
        prompt_id="new",
        pair_id="new_pair",
        split="heldout_readout",
        source_llm="gemini_pro",
        text="",
        core_text="A student sees a provisional grade before the official posting.",
        normalized=audit.normalize_text("A student sees a provisional grade before the official posting."),
        tokens=audit.content_tokens(audit.normalize_text("A student sees a provisional grade before the official posting.")),
        ngrams=audit.token_ngrams(audit.normalize_text("A student sees a provisional grade before the official posting."), 3),
    )

    rows = audit.audit_overlaps(
        [reference],
        [candidate],
        char_threshold=0.82,
        token_threshold=0.55,
        ngram_threshold=0.30,
    )

    assert len(rows) == 1
    assert rows[0]["overlap_type"] == "exact_normalized"


def test_internal_audit_skips_matched_pair_but_flags_other_candidates():
    audit = _load_module()
    normalized = audit.normalize_text("A manager sees a pending bonus before payroll closes.")
    first = audit.PromptRecord(
        prompt_id="candidate_1",
        pair_id="pair_1",
        split="heldout_readout",
        source_llm="deepseek_v32",
        text="",
        core_text="A manager sees a pending bonus before payroll closes.",
        normalized=normalized,
        tokens=audit.content_tokens(normalized),
        ngrams=audit.token_ngrams(normalized, 3),
    )
    same_pair = audit.PromptRecord(
        prompt_id="candidate_2",
        pair_id="pair_1",
        split="heldout_readout",
        source_llm="deepseek_v32",
        text="",
        core_text="A manager sees a pending bonus before payroll closes.",
        normalized=normalized,
        tokens=audit.content_tokens(normalized),
        ngrams=audit.token_ngrams(normalized, 3),
    )
    other_pair = audit.PromptRecord(
        prompt_id="candidate_3",
        pair_id="pair_2",
        split="heldout_readout",
        source_llm="llama_maverick",
        text="",
        core_text="A manager sees a pending bonus before payroll closes.",
        normalized=normalized,
        tokens=audit.content_tokens(normalized),
        ngrams=audit.token_ngrams(normalized, 3),
    )

    rows = audit.audit_internal_overlaps(
        [first, same_pair, other_pair],
        char_threshold=0.82,
        token_threshold=0.55,
        ngram_threshold=0.30,
    )

    assert len(rows) == 2
    assert {row["comparison_scope"] for row in rows} == {"candidate_internal"}
