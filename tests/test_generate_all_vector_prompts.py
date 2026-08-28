from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from construct_benchmark.prompts import load_prompt_records  # noqa: E402
from construct_benchmark.manifests import file_sha256  # noqa: E402
from scripts.generate_all_vector_prompts import (  # noqa: E402
    RuntimeBudget,
    RuntimeBudgetExceeded,
    VECTOR_SPLITS,
    _checkpoint_checksum,
    _request_with_runtime_budget,
    discover_vector_plans,
    orchestrate_vector_prompts,
    validate_quality_gate,
)


REGISTRY = ROOT / "configs/construct_benchmark/construct_registry_v1.json"


def _write_test_quality_gate(tmp_path: Path, *, waves: list[int] | None = None) -> Path:
    """Create a hash-complete, structurally valid review artifact."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    review = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=waves or [3],
        output_dir=tmp_path / "review",
        mode="review",
        api_key="test-only",
        provider="openai",
        max_estimated_cost_usd=100.0,
        request_fn=_deterministic_response,
    )
    review_path = Path(review["manifest_path"])
    gate_path = tmp_path / "quality_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "quality_gate_version": "1",
                "status": "approved",
                "approved": True,
                "reviewer": "test-reviewer",
                "review_manifest_path": str(review_path),
                "review_manifest_sha256": file_sha256(review_path),
            }
        ),
        encoding="utf-8",
    )
    return gate_path


def _deterministic_response(model_id, messages, options):
    del model_id
    payload = json.loads(messages[1]["content"])
    count = int(payload["count"])
    domains = payload["assigned_content_domains"]
    nonce = hashlib.sha256(options["generation_job_id"].encode("utf-8")).hexdigest()[:10]
    if payload["generation_mode"] == "paired":
        condition_ids = [item["condition_id"] for item in payload["condition_definitions"]]
        paired_schema = payload.get("paired_item_metadata_schema")
        required_category_assignments = payload.get("required_category_assignments", [])
        data = {
            "pairs": [
                {
                    "pair_id": f"pair_{index}",
                    "content_domain": domains[index],
                    "prompts": [
                        {
                            "condition_id": condition_id,
                            "prompt_text": f"Matched scenario {nonce}_{index}_{condition_index}.",
                            **(
                                {
                                    "task_metadata": {
                                        **{
                                            field_name: (
                                                field_schema["enum"][0]
                                                if field_schema.get("enum")
                                                else int(field_schema.get("minimum", 0))
                                                if field_schema["type"] == "integer"
                                                else float(field_schema.get("minimum", 0))
                                                if field_schema["type"] == "number"
                                                else True
                                                if field_schema["type"] == "boolean"
                                                else "mock_value"
                                            )
                                            for field_name, field_schema in paired_schema["properties"].items()
                                        },
                                        **(
                                            required_category_assignments[index]
                                            if index < len(required_category_assignments)
                                            else {}
                                        ),
                                    }
                                }
                                if paired_schema
                                else {}
                            ),
                        }
                        for condition_index, condition_id in enumerate(condition_ids)
                    ],
                    "notes": "deterministic vector fixture",
                }
                for index in range(count)
            ]
        }
    else:
        raise AssertionError("Vector orchestration must not submit downstream single-prompt jobs.")
    return {
        "choices": [{"message": {"content": json.dumps(data)}}],
        "_generation_metadata": {
            "actual_cost_usd": 0.0001,
            "input_tokens": 100,
            "output_tokens": 100,
        },
    }


def test_discovery_selects_exact_registry_plan_paths() -> None:
    entries = discover_vector_plans(REGISTRY, waves=["3"])
    assert [entry.construct_id for entry in entries] == [
        "ambiguity_orientation",
        "causal_interpretation",
        "consensus_conformity",
        "plan_replanning",
    ]
    assert all(entry.plan_path.name == f"wave3_{entry.construct_id}_v1.json" for entry in entries)
    assert all(entry.plan["construct_spec_path"] == f"../constructs/{entry.construct_id}_v1.json" for entry in entries)


def test_dry_run_never_calls_request_function_or_writes_outputs(tmp_path: Path) -> None:
    calls: list[str] = []

    def fail_request(*args, **kwargs):
        calls.append("called")
        raise AssertionError("dry-run must not call a request function")

    manifest = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[3],
        output_dir=tmp_path,
        mode="review",
        workers=2,
        dry_run=True,
        request_fn=fail_request,
    )
    assert calls == []
    assert manifest["dry_run"] is True
    assert manifest["confirmatory"] is False
    assert manifest["counts"] == {
        "split_counts": {
            "direction_heldout": 8,
            "direction_train": 8,
            "direction_validation": 8,
        },
        "pair_count": 12,
        "record_count": 24,
        "request_count": 12,
        "estimated_input_tokens": 16800,
            "estimated_output_tokens": 7200,
            "estimated_total_tokens": 24000,
                "estimated_cost_usd": 0.012,
                "budget_estimate_usd": 0.015,
    }
    assert all(item["request_count"] == 3 for item in manifest["constructs"])
    assert all(item["estimated_total_tokens"] == 6000 for item in manifest["constructs"])
    assert not list(tmp_path.iterdir())


def test_dry_run_can_select_construct_subset_for_targeted_pilot(tmp_path: Path) -> None:
    manifest = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[2, 3],
        construct_ids=["exploration_exploitation", "ambiguity_orientation"],
        output_dir=tmp_path,
        mode="review",
        dry_run=True,
        provider="openai",
        model="gpt-5.6-luna",
    )
    assert manifest["construct_ids"] == ["exploration_exploitation", "ambiguity_orientation"]
    assert manifest["counts"]["pair_count"] == 6
    assert manifest["counts"]["record_count"] == 12
    assert manifest["counts"]["request_count"] == 6
    assert not list(tmp_path.iterdir())


def test_generation_writes_per_construct_and_combined_manifest_then_resumes(tmp_path: Path) -> None:
    calls: list[str] = []

    def request_fn(model_id, messages, options):
        calls.append(options["generation_job_id"])
        return _deterministic_response(model_id, messages, options)

    first = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[3],
        output_dir=tmp_path,
        mode="review",
        workers=2,
        api_key="test-only",
        request_fn=request_fn,
        provider="openai",
        max_estimated_cost_usd=100.0,
    )
    assert len(calls) == 12
    assert first["dry_run"] is False
    assert first["partial"] is True
    assert first["confirmatory"] is False
    assert first["counts"]["record_count"] == 24
    assert first["counts"]["pair_count"] == 12
    assert first["counts"]["request_count"] == 12
    assert first["counts"]["estimated_input_tokens"] == 16800
    assert first["counts"]["estimated_output_tokens"] == 7200
    assert first["counts"]["estimated_total_tokens"] == 24000
    assert Path(first["combined_path"]).is_file()
    assert Path(first["manifest_path"]).is_file()
    assert len(list(tmp_path.glob("*.csv"))) == 5
    assert all(item["output_sha256"] for item in first["constructs"])
    assert set(load_prompt_records(tmp_path / "combined.csv")[0].metadata) >= {
        "generation_plan_sha256",
        "generation_plan_id",
    }
    assert {record.split for record in load_prompt_records(tmp_path / "combined.csv")} == VECTOR_SPLITS

    calls.clear()

    def no_request(*args, **kwargs):
        raise AssertionError("valid --resume outputs should be skipped")

    resumed = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[3],
        output_dir=tmp_path,
        mode="review",
        workers=4,
        resume=True,
        api_key="test-only",
        request_fn=no_request,
        provider="openai",
        max_estimated_cost_usd=100.0,
    )
    assert calls == []
    assert all(item["resumed"] is True for item in resumed["constructs"])
    assert resumed["counts"]["request_count"] == first["counts"]["request_count"]
    assert resumed["counts"]["estimated_total_tokens"] == first["counts"]["estimated_total_tokens"]
    assert resumed["combined_sha256"] == first["combined_sha256"]


def test_resume_rejects_stale_plan_hash_before_generation(tmp_path: Path) -> None:
    orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[3],
        output_dir=tmp_path,
        mode="review",
        api_key="test-only",
        request_fn=_deterministic_response,
        provider="openai",
        max_estimated_cost_usd=100.0,
    )
    target = tmp_path / "ambiguity_orientation.csv"
    target.write_text(
        target.read_text(encoding="utf-8").replace("generation_plan_sha256", "stale_plan_hash"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale or missing generation plan hash"):
        orchestrate_vector_prompts(
            registry_path=REGISTRY,
            waves=[3],
            output_dir=tmp_path,
            mode="review",
            resume=True,
            api_key="test-only",
            request_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate")),
            provider="openai",
            max_estimated_cost_usd=100.0,
        )


def test_non_dry_generation_requires_an_explicit_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max-estimated-cost-usd"):
        orchestrate_vector_prompts(
            registry_path=REGISTRY,
            waves=[3],
            output_dir=tmp_path,
            mode="review",
            api_key="test-only",
            provider="openai",
            request_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate")),
        )


def test_full_generation_requires_an_approved_quality_gate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="quality-gate-file"):
        orchestrate_vector_prompts(
            registry_path=REGISTRY,
            waves=[3],
            output_dir=tmp_path,
            mode="full",
            api_key="test-only",
            provider="openai",
            max_estimated_cost_usd=100.0,
            request_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate")),
        )


def test_full_preflight_rejects_budget_before_requests(tmp_path: Path) -> None:
    gate_path = _write_test_quality_gate(tmp_path / "gate")
    with pytest.raises(ValueError, match="exceeds the cap"):
        orchestrate_vector_prompts(
            registry_path=REGISTRY,
            waves=[3],
            output_dir=tmp_path / "full",
            mode="full",
            api_key="test-only",
            provider="openai",
            max_estimated_cost_usd=0.0,
            quality_gate_file=gate_path,
            request_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate")),
        )


def test_quality_gate_hashes_and_plan_ids_are_checked(tmp_path: Path) -> None:
    gate_path = _write_test_quality_gate(tmp_path)
    entries = discover_vector_plans(REGISTRY, waves=[3])
    validated = validate_quality_gate(gate_path, entries=entries)
    assert validated["approved"] is True
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    review_path = Path(gate["review_manifest_path"])
    review_path.write_text(review_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_quality_gate(gate_path, entries=entries)


def _budget_options() -> dict[str, int]:
    return {
        "max_tokens": 1_000,
        "estimated_input_tokens_per_request": 0,
    }


def test_runtime_budget_stops_requests_before_cap() -> None:
    budget = RuntimeBudget(
        max_budget_usd=0.0015,
        input_usd_per_million_tokens=0.0,
        output_usd_per_million_tokens=1.0,
    )
    calls: list[str] = []

    def request_fn(model_id, messages, options):
        del model_id, messages, options
        calls.append("called")
        return {"_generation_metadata": {"actual_cost_usd": 0.0005}}

    guarded = _request_with_runtime_budget(request_fn, budget)
    guarded("luna", [], _budget_options())
    guarded("luna", [], _budget_options())
    with pytest.raises(RuntimeBudgetExceeded, match="would be exceeded"):
        guarded("luna", [], _budget_options())

    assert calls == ["called", "called"]
    snapshot = budget.snapshot()
    assert snapshot["actual_spent_usd"] == pytest.approx(0.001)
    assert snapshot["outstanding_reserved_usd"] == pytest.approx(0.0)


def test_runtime_budget_reservations_cannot_oversubscribe_concurrently() -> None:
    budget = RuntimeBudget(
        max_budget_usd=0.001,
        input_usd_per_million_tokens=0.0,
        output_usd_per_million_tokens=1.0,
    )
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def request_fn(model_id, messages, options):
        del model_id, messages, options
        calls.append("called")
        started.set()
        assert release.wait(timeout=2.0)
        return {"_generation_metadata": {"actual_cost_usd": 0.0001}}

    guarded = _request_with_runtime_budget(request_fn, budget)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(guarded, "luna", [], _budget_options())
        assert started.wait(timeout=2.0)
        second = executor.submit(guarded, "luna", [], _budget_options())
        time.sleep(0.02)
        with pytest.raises(RuntimeBudgetExceeded, match="would be exceeded"):
            second.result()
        release.set()
        first.result()

    assert calls == ["called"]
    assert budget.snapshot()["outstanding_reserved_usd"] == pytest.approx(0.0)


def test_full_generation_checkpoints_jobs_and_resumes_without_duplicates(tmp_path: Path) -> None:
    gate_path = _write_test_quality_gate(tmp_path / "gate", waves=[1])
    output_dir = tmp_path / "full"
    calls: list[str] = []
    failure_job: dict[str, str | None] = {"id": None}

    def fail_later(model_id, messages, options):
        calls.append(options["generation_job_id"])
        if len(calls) > 4 and failure_job["id"] is None:
            failure_job["id"] = options["generation_job_id"]
        if failure_job["id"] == options["generation_job_id"]:
            raise ValueError("synthetic later-job failure")
        return _deterministic_response(model_id, messages, options)

    with pytest.raises(ValueError, match="synthetic later-job failure"):
        orchestrate_vector_prompts(
            registry_path=REGISTRY,
            waves=[1],
            construct_ids=["realization_account_closure"],
            output_dir=output_dir,
            mode="full",
            workers=1,
            api_key="test-only",
            request_fn=fail_later,
            provider="openai",
            max_estimated_cost_usd=100.0,
            quality_gate_file=gate_path,
        )

    checkpoint_path = output_dir / "checkpoints" / "realization_account_closure.json"
    assert checkpoint_path.is_file()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["jobs"]
    assert checkpoint["checksum_sha256"] == _checkpoint_checksum(checkpoint)
    assert not (output_dir / "realization_account_closure.csv").exists()
    completed_job_ids = {job["job_id"] for job in checkpoint["jobs"]}

    resumed_calls: list[str] = []

    def resume_request(model_id, messages, options):
        resumed_calls.append(options["generation_job_id"])
        return _deterministic_response(model_id, messages, options)

    manifest = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[1],
        construct_ids=["realization_account_closure"],
        output_dir=output_dir,
        mode="full",
        workers=1,
        resume=True,
        api_key="test-only",
        request_fn=resume_request,
        provider="openai",
        max_estimated_cost_usd=100.0,
        quality_gate_file=gate_path,
    )
    assert not completed_job_ids.intersection(resumed_calls)
    assert len(resumed_calls) < manifest["counts"]["request_count"]
    records = load_prompt_records(output_dir / "combined.csv")
    assert len({record.prompt_id for record in records}) == len(records)
    assert manifest["counts"]["record_count"] == len(records)
    assert manifest["constructs"][0]["checkpoint_job_count"] == manifest["counts"]["request_count"]


def test_resume_rejects_a_stale_job_checkpoint_identity(tmp_path: Path) -> None:
    gate_path = _write_test_quality_gate(tmp_path / "gate", waves=[1])
    output_dir = tmp_path / "full"

    call_count = {"value": 0}

    def fail_after_first(model_id, messages, options):
        call_count["value"] += 1
        if call_count["value"] > 2:
            raise ValueError("stop after checkpoint")
        return _deterministic_response(model_id, messages, options)

    with pytest.raises(ValueError, match="stop after checkpoint"):
        orchestrate_vector_prompts(
            registry_path=REGISTRY,
            waves=[1],
            construct_ids=["realization_account_closure"],
            output_dir=output_dir,
            mode="full",
            workers=1,
            api_key="test-only",
            request_fn=fail_after_first,
            provider="openai",
            max_estimated_cost_usd=100.0,
            quality_gate_file=gate_path,
        )

    checkpoint_path = output_dir / "checkpoints" / "realization_account_closure.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["identity"]["runtime_settings"]["max_output_tokens"] = 123
    checkpoint["checksum_sha256"] = _checkpoint_checksum(checkpoint)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint identity is stale"):
        orchestrate_vector_prompts(
            registry_path=REGISTRY,
            waves=[1],
            construct_ids=["realization_account_closure"],
            output_dir=output_dir,
            mode="full",
            workers=1,
            resume=True,
            api_key="test-only",
            request_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate")),
            provider="openai",
            max_estimated_cost_usd=100.0,
            quality_gate_file=gate_path,
        )


def test_full_generation_staged_limit_pauses_and_resumes_without_final_outputs(tmp_path: Path) -> None:
    gate_path = _write_test_quality_gate(tmp_path / "gate", waves=[1])
    output_dir = tmp_path / "staged"
    first_calls: list[str] = []

    def first_request(model_id, messages, options):
        first_calls.append(options["generation_job_id"])
        return _deterministic_response(model_id, messages, options)

    paused = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[1],
        construct_ids=["realization_account_closure", "evidence_diagnosticity"],
        output_dir=output_dir,
        mode="full",
        workers=4,
        api_key="test-only",
        request_fn=first_request,
        provider="openai",
        max_estimated_cost_usd=100.0,
        max_new_jobs=2,
        quality_gate_file=gate_path,
    )
    assert paused["status"] == "paused"
    assert paused["pause_reason"] == "max_new_jobs"
    assert len(first_calls) == 2
    assert not (output_dir / "combined.csv").exists()
    assert not (output_dir / "vector_prompt_manifest.json").exists()
    assert not list(output_dir.glob("*.csv"))
    state_path = output_dir / "vector_prompt_run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "paused"
    assert state["new_jobs_completed_this_invocation"] == 2
    assert state["new_spend_usd_this_invocation"] == pytest.approx(0.0002)
    assert sum(state["checkpoint_job_counts"].values()) == 2
    identity_before_resume = state["run_identity"]

    resumed_calls: list[str] = []

    def resumed_request(model_id, messages, options):
        resumed_calls.append(options["generation_job_id"])
        return _deterministic_response(model_id, messages, options)

    complete = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[1],
        construct_ids=["realization_account_closure", "evidence_diagnosticity"],
        output_dir=output_dir,
        mode="full",
        workers=2,
        resume=True,
        api_key="test-only",
        request_fn=resumed_request,
        provider="openai",
        max_estimated_cost_usd=100.0,
        quality_gate_file=gate_path,
    )
    assert complete["dry_run"] is False
    final_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert final_state["status"] == "complete"
    assert final_state["run_identity"] == identity_before_resume
    assert (output_dir / "combined.csv").exists()
    assert (output_dir / "vector_prompt_manifest.json").exists()
    records = load_prompt_records(output_dir / "combined.csv")
    assert len({record.prompt_id for record in records}) == len(records)
    assert not set(first_calls).intersection(resumed_calls)

def test_runtime_overrides_are_hashed_and_passed_to_request_options(tmp_path: Path) -> None:
    seen_options: list[dict] = []

    def request_fn(model_id, messages, options):
        seen_options.append(dict(options))
        return _deterministic_response(model_id, messages, options)

    manifest = orchestrate_vector_prompts(
        registry_path=REGISTRY,
        waves=[3],
        construct_ids=["ambiguity_orientation"],
        output_dir=tmp_path,
        mode="review",
        workers=1,
        api_key="test-only",
        request_fn=request_fn,
        provider="openai",
        model="gpt-5.6-luna",
        max_items_per_request=1,
        max_output_tokens=321,
        max_estimated_cost_usd=100.0,
    )
    assert seen_options
    assert all(item["max_items_per_request"] == 1 for item in seen_options)
    assert all(item["max_output_tokens"] == 321 for item in seen_options)
    assert manifest["runtime_settings"]["max_items_per_request"] == 1
    assert manifest["runtime_settings"]["max_output_tokens"] == 321
    construct = manifest["constructs"][0]
    source_entry = discover_vector_plans(REGISTRY, waves=[3])[0]
    assert construct["source_plan_sha256"] == source_entry.source_plan_sha256
    assert construct["plan_sha256"] != construct["source_plan_sha256"]
