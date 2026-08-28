from __future__ import annotations

from construct_benchmark.scoring_pipeline import build_scoring_report, evaluate_expansion_gates


def _zero_adapter(stage, context):
    return {
        "status": "passed",
        "complete": True,
        "confirmatory": False,
        "summary": {
            "stage_effect": 0.0,
            "uncertainty": {"estimate": 0.0, "lower": 0.0, "upper": 0.0},
            "invalid_or_unparseable_exclusions": [{"record_id": "bad", "reason": "invalid"}],
        },
        "exclusions": [{"record_id": "bad", "reason": "invalid"}],
        "uncertainty": {"estimate": 0.0, "lower": 0.0, "upper": 0.0},
    }


def test_stage_namespaces_keep_null_effects_exclusions_and_uncertainty_separate() -> None:
    report = build_scoring_report(
        {code: f"{code.lower()}-fixture" for code in ("B", "R", "C", "S")},
        adapters={code: _zero_adapter for code in ("B", "R", "C", "S")},
    )

    assert report["complete"] is True
    assert report["ready"] is True
    assert report["confirmatory"] is False
    assert set(report["stages"]) == {"B", "R", "C", "S"}
    assert all(report["stage_summaries"][code]["stage_effect"] == 0.0 for code in report["stage_summaries"])
    assert report["stage_exclusions"]["B"][0]["reason"] == "invalid"
    assert report["stage_uncertainty"]["S"]["estimate"] == 0.0
    assert report["stage_summaries"]["B"] is not report["stage_summaries"]["R"]


def test_diagnostic_and_complete_modes_do_not_promote_incomplete_inputs() -> None:
    def incomplete(stage, context):
        return {"status": "passed", "complete": False, "confirmatory": True, "summary": {"effect": 0.0}}

    diagnostic = build_scoring_report({"B": "fixture"}, mode="diagnostic", adapters={"B": incomplete})
    assert diagnostic["report_mode"] == "diagnostic"
    assert diagnostic["stages"]["B"]["status"] == "diagnostic_incomplete"
    assert diagnostic["ready"] is True
    assert diagnostic["confirmatory"] is False

    complete = build_scoring_report({"B": "fixture"}, mode="complete", adapters={"B": incomplete})
    assert complete["stages"]["B"]["status"] == "incomplete"
    assert complete["ready"] is False
    assert complete["confirmatory"] is False


def test_existing_gate_status_is_preserved_and_holds_expansion() -> None:
    campaign_report = {
        "ready": True,
        "confirmatory": False,
        "prerequisites": [
            {"id": "wave1_measurement_gate", "status": "pending", "detail": "measurement pending"},
            {"id": "precision_simulation", "status": "fail", "detail": "precision pending"},
        ],
        "blocking_checks": [],
    }
    gates = evaluate_expansion_gates(campaign_report=campaign_report)
    assert gates["evaluated"] is True
    assert gates["wave1_measurement_gate"]["status"] == "pending"
    assert gates["precision_simulation"]["status"] == "fail"
    assert gates["expansion_decision"] == "hold"

    report = build_scoring_report(
        {"B": "fixture"},
        adapters={"B": _zero_adapter},
        campaign_report=campaign_report,
    )
    assert report["gates"]["expansion_decision"] == "hold"
    assert report["confirmatory"] is False
    assert any("wave1_measurement_gate" in reason for reason in report["gates"]["reasons"])


def test_no_campaign_is_explicitly_not_evaluated() -> None:
    report = build_scoring_report({"R": "fixture"}, adapters={"R": _zero_adapter})
    assert report["gates"]["evaluated"] is False
    assert report["gates"]["expansion_decision"] == "hold"
    assert report["stages"]["B"]["status"] == "not_available"
