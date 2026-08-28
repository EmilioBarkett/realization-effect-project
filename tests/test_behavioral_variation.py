from __future__ import annotations

from pathlib import Path

from construct_benchmark.behavioral_variation import audit_zero_dose_variation
from construct_benchmark.config import load_construct_spec


ROOT = Path(__file__).resolve().parents[1]
V2_SPEC = load_construct_spec(
    ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v2.json"
)


def _row(prompt_id: str, output: str, *, dose: float = 0.0, tracking_layer: int = 10) -> dict:
    return {
        "prompt_id": prompt_id,
        "direction_kind": "target",
        "dose": dose,
        "tracking_role": "injection_immediate",
        "tracking_layer": tracking_layer,
        "injection_layer": 10,
        "parser_id": "single_integer_allocation_0_to_100_v1",
        "task_id": "goal_renewal_allocation_v2",
        "task_metadata": {},
        "output_text": output,
    }


def test_behavioral_variation_gate_passes_on_varied_zero_dose_rows() -> None:
    rows = [_row(f"item-{index}", str(20 + index * 10)) for index in range(8)]
    report = audit_zero_dose_variation(
        rows,
        V2_SPEC,
        thresholds={
            "minimum_zero_dose_valid": 8,
            "minimum_zero_dose_distinct": 3,
            "minimum_zero_dose_sample_sd": 2.0,
        },
    )
    assert report["pass"] is True
    assert report["valid_zero_dose_rows"] == 8
    assert report["unique_zero_dose_outcomes"] == [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    assert report["invalid_zero_dose_rows"] == 0


def test_behavioral_variation_gate_fails_closed_on_constant_or_invalid_rows() -> None:
    rows = [_row(f"item-{index}", "90" if index < 7 else "not an integer") for index in range(8)]
    report = audit_zero_dose_variation(rows, V2_SPEC)
    assert report["pass"] is False
    assert report["valid_zero_dose_rows"] == 7
    assert report["invalid_zero_dose_rows"] == 1
    assert any("distinct zero-dose outcomes" in failure for failure in report["failures"])
    assert any("invalid zero-dose rows" in failure for failure in report["failures"])


def test_behavioral_variation_gate_ignores_non_target_and_downstream_tracking_rows() -> None:
    rows = [
        _row("target-zero", "20"),
        _row("target-nonzero", "80", dose=1.0),
        {**_row("target-downstream", "90", tracking_layer=20), "tracking_role": "downstream_construct_state"},
        {**_row("shuffled-zero", "10"), "direction_kind": "shuffled"},
    ]
    report = audit_zero_dose_variation(
        rows,
        V2_SPEC,
        thresholds={
            "minimum_zero_dose_valid": 1,
            "minimum_zero_dose_distinct": 2,
            "minimum_zero_dose_sample_sd": 1.0,
        },
    )
    assert report["zero_dose_target_injection_rows"] == 1
    assert report["valid_zero_dose_rows"] == 1
    assert report["pass"] is False
