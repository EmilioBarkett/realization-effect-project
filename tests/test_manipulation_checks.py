from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from activation_analysis.steering import (
    ResidualSteeringGenerator,
    SteeringConfig,
    SteeringTrace,
    SteeringVectorInfo,
)
from construct_benchmark.manipulation import (
    downstream_persistence_ratio,
    raw_downstream_projection_transfer,
    score_expected_observed_shift,
    summarize_manipulation_records,
)
from scripts.plan_construct_steering import _build_tracking_directions
from scripts.run_construct_steering import (
    _build_output_manifest,
    _output_manifest_path,
    _trace_rows,
    _validate_output_manifest,
)
from scripts.score_construct_steering import _load_and_validate_output_manifest
from construct_benchmark.manifests import file_sha256
from construct_benchmark.config import load_run_config


def test_expected_observed_scoring_handles_positive_negative_and_zero_doses() -> None:
    positive = score_expected_observed_shift(1.0, 3.0, 2.0)
    negative = score_expected_observed_shift(4.0, 1.0, -3.0)
    zero = score_expected_observed_shift(2.5, 2.5, 0.0)

    assert positive["observed_shift"] == pytest.approx(2.0)
    assert positive["expected_observed_difference"] == pytest.approx(0.0)
    assert positive["sign_agreement"] == 1.0
    assert negative["observed_shift"] == pytest.approx(-3.0)
    assert negative["relative_error"] == pytest.approx(0.0)
    assert zero["observed_shift"] == pytest.approx(0.0)
    assert zero["relative_error"] is None
    assert zero["sign_agreement"] == 1.0


def test_downstream_persistence_uses_observed_injection_shift() -> None:
    assert downstream_persistence_ratio(
        2.0,
        1.0,
        0.5,
        downstream_calibration_scale=2.0,
        injection_calibration_scale=1.0,
    ) == pytest.approx(1.0)
    assert raw_downstream_projection_transfer(2.0, 1.0, 0.5) == pytest.approx(2.0)
    assert downstream_persistence_ratio(
        2.0,
        1.0,
        0.0,
        downstream_calibration_scale=2.0,
        injection_calibration_scale=1.0,
    ) is None


def _immediate(
    condition_id: str,
    prompt_id: str,
    direction_kind: str,
    dose: float,
    *,
    pre: float,
    expected: float,
) -> dict:
    return {
        "record_id": f"{condition_id}__layer_10",
        "condition_id": condition_id,
        "prompt_id": prompt_id,
        "direction_kind": direction_kind,
        "direction_index": 0,
        "dose": dose,
        "physical_scale": expected,
        "injection_layer": 10,
        "tracking_layer": 10,
        "tracking_direction_id": f"{direction_kind}__layer_10",
        "tracking_role": "injection_immediate",
        "injection_calibration_projection_scale": 1.0,
        "tracking_calibration_projection_scale": 1.0,
        "pre_projection": pre,
        "post_projection": pre + expected,
        "expected_shift": expected,
        "projection": pre + expected,
    }


def _downstream(
    condition_id: str,
    prompt_id: str,
    direction_kind: str,
    dose: float,
    projection: float,
) -> dict:
    return {
        "record_id": f"{condition_id}__layer_20",
        "condition_id": condition_id,
        "prompt_id": prompt_id,
        "direction_kind": direction_kind,
        "direction_index": 0,
        "dose": dose,
        "physical_scale": dose,
        "injection_layer": 10,
        "tracking_layer": 20,
        "tracking_direction_id": "construct_state__layer_20",
        "tracking_role": "downstream_construct_state",
        "injection_calibration_projection_scale": 1.0,
        "tracking_calibration_projection_scale": 1.0,
        "projection": projection,
        "downstream_projection": projection,
    }


def test_manipulation_summary_separates_immediate_and_downstream_records() -> None:
    records = [
        _immediate("zero", "item", "target", 0.0, pre=1.0, expected=0.0),
        _downstream("zero", "item", "target", 0.0, projection=4.0),
        _immediate("positive", "item", "target", 1.0, pre=1.0, expected=1.0),
        _downstream("positive", "item", "target", 1.0, projection=4.5),
        _immediate("negative", "item", "target", -1.0, pre=1.0, expected=-1.0),
        _downstream("negative", "item", "target", -1.0, projection=3.5),
        _immediate("shuffled", "item", "shuffled", 1.0, pre=1.0, expected=1.0),
        _downstream("shuffled", "item", "shuffled", 1.0, projection=4.2),
    ]

    summary = summarize_manipulation_records(records)

    assert summary["injection_record_count"] == 4
    assert summary["injection_by_direction_and_dose"]["target:1"]["observed_shift_mean"] == pytest.approx(1.0)
    assert summary["dose_response_slopes"]["target"] == pytest.approx(1.0)
    persistence = summary["downstream_persistence"][
        "target:1:layer_20:downstream_construct_state"
    ]
    assert persistence["persistence_ratio_mean"] == pytest.approx(0.5)
    assert summary["downstream_persistence"][
        "shuffled:1:layer_20:downstream_construct_state"
    ]["persistence_ratio_mean"] == pytest.approx(0.2)


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def detach(self) -> "_FakeScalar":
        return self

    def float(self) -> "_FakeScalar":
        return self

    def cpu(self) -> "_FakeScalar":
        return self

    def item(self) -> float:
        return self.value


class _FakeTorch:
    @staticmethod
    def sum(value: np.ndarray) -> _FakeScalar:
        return _FakeScalar(float(np.sum(value)))


class _FakeDirection:
    def __init__(self, value: list[float]) -> None:
        self.value = np.asarray(value, dtype=np.float32)

    def to(self, *, device: str, dtype: object) -> "_FakeDirection":
        del device, dtype
        return self

    def __mul__(self, scale: float) -> np.ndarray:
        return self.value * float(scale)

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        return self.value.astype(dtype) if dtype is not None else self.value


class _FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value, dtype=np.float32)
        self.device = "cpu"
        self.dtype = np.float32

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self.value.copy())

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value) -> None:
        self.value[key] = value


def _fake_generator() -> ResidualSteeringGenerator:
    generator = object.__new__(ResidualSteeringGenerator)
    generator._torch = _FakeTorch()
    return generator


def test_hook_records_prefill_arithmetic_and_preserves_tuple_output() -> None:
    generator = _fake_generator()
    direction = _FakeDirection([1.0, 0.0])
    info = SteeringVectorInfo("target.npy", 2, 1.0, True)
    trace = SteeringTrace(
        injection_layer=10,
        intervention_timing="prefill_only",
        position_mode="last",
        direction_id="target",
        direction_source="injection_direction_train_only",
        direction_role="injection_immediate",
        requested_dose=1.0,
        physical_scale=2.0,
        calibration_projection_scale=2.0,
    )
    config = SteeringConfig(
        direction_path=Path("target.npy"),
        layer=10,
        scale=2.0,
        position_mode="last",
        intervention_timing="prefill_only",
        direction_id="target",
        requested_dose=1.0,
        calibration_projection_scale=2.0,
    )
    hook = generator._make_injection_tracking_hook(
        config=config,
        direction=direction,
        info=info,
        trace=trace,
        prefill_token_position=4,
    )
    original = _FakeTensor(np.asarray([[[1.0, 5.0], [3.0, 4.0]]], dtype=np.float32))
    returned = hook(None, None, (original, "cache"))
    hook(None, None, (returned[0], "cache"))

    assert isinstance(returned, tuple)
    assert returned[1] == "cache"
    assert returned[0].value[0, -1, 0] == pytest.approx(5.0)
    assert trace.injection_observations[0].phase == "prefill"
    assert trace.injection_observations[0].token_position == 4
    assert trace.injection_observations[0].pre_projection == pytest.approx(3.0)
    assert trace.injection_observations[0].post_projection == pytest.approx(5.0)
    assert trace.injection_observations[0].observed_shift == pytest.approx(2.0)
    assert trace.injection_observations[0].expected_observed_difference == pytest.approx(0.0)
    assert trace.injection_observations[1].phase == "generation"
    assert trace.injection_observations[1].injection_applied is False
    assert trace.injection_observations[1].expected_shift == pytest.approx(0.0)
    serialized = trace.to_mapping()
    assert serialized["injection_direction"] == {
        "path": "target.npy",
        "hidden_size": 2,
        "raw_norm": 1.0,
        "normalized": True,
    }
    assert all(isinstance(item["projection"], float) for item in serialized["projection_observations"])


def test_steering_hooks_remove_all_handles_when_body_raises() -> None:
    class Handle:
        def __init__(self, hooks: list[object], hook: object) -> None:
            self.hooks = hooks
            self.hook = hook

        def remove(self) -> None:
            self.hooks.remove(self.hook)

    class Block:
        def __init__(self) -> None:
            self.hooks: list[object] = []

        def register_forward_hook(self, hook: object) -> Handle:
            self.hooks.append(hook)
            return Handle(self.hooks, hook)

    generator = _fake_generator()
    blocks = [Block(), Block()]
    generator._resolve_transformer_blocks = lambda: blocks
    info = SteeringVectorInfo("target.npy", 2, 1.0, True)
    direction = _FakeDirection([1.0, 0.0])
    generator.load_direction = lambda config: (direction, info)
    config = SteeringConfig(Path("target.npy"), layer=1, scale=1.0)

    with pytest.raises(RuntimeError, match="boom"):
        with generator.steering_hooks(config, tracking_directions={2: Path("later.npy")}):
            raise RuntimeError("boom")

    assert all(not block.hooks for block in blocks)


def test_integrated_fake_forward_writes_tracking_rows_and_rejects_truncation(tmp_path: Path) -> None:
    class Handle:
        def __init__(self, hooks: list[object], hook: object) -> None:
            self.hooks = hooks
            self.hook = hook

        def remove(self) -> None:
            self.hooks.remove(self.hook)

    class Block:
        def __init__(self) -> None:
            self.hooks: list[object] = []

        def register_forward_hook(self, hook: object) -> Handle:
            self.hooks.append(hook)
            return Handle(self.hooks, hook)

        def run(self, tensor: _FakeTensor) -> _FakeTensor:
            output: object = (tensor, "cache")
            for hook in list(self.hooks):
                output = hook(self, None, output)
            return output[0]

    generator = _fake_generator()
    blocks = [Block(), Block()]
    generator._resolve_transformer_blocks = lambda: blocks
    target_direction = _FakeDirection([1.0, 0.0])
    direction_info = {
        "target.npy": (target_direction, SteeringVectorInfo("target.npy", 2, 1.0, True)),
        "downstream.npy": (target_direction, SteeringVectorInfo("downstream.npy", 2, 1.0, True)),
    }
    generator.load_direction = lambda config: direction_info[str(config.direction_path)]
    config = SteeringConfig(
        Path("target.npy"),
        layer=1,
        scale=2.0,
        position_mode="last",
        intervention_timing="prefill_only",
        direction_id="target",
        requested_dose=1.0,
        calibration_projection_scale=2.0,
    )
    trace = SteeringTrace(
        injection_layer=1,
        intervention_timing="prefill_only",
        position_mode="last",
        direction_id="target",
        direction_source="injection_direction_train_only",
        direction_role="injection_immediate",
        requested_dose=1.0,
        physical_scale=2.0,
        calibration_projection_scale=2.0,
    )
    tracking = {
        1: {
            "layer": 1,
            "direction_id": "target",
            "path": "target.npy",
            "source": "injection_direction_train_only",
            "role": "injection_immediate",
            "calibration": {"projection_scale": 2.0},
        },
        2: {
            "layer": 2,
            "direction_id": "construct_state__layer_02",
            "path": "downstream.npy",
            "source": "independent_train_only",
            "role": "downstream_construct_state",
            "calibration": {"projection_scale": 3.0},
        },
    }
    with generator.steering_hooks(
        config,
        tracking_directions={2: tracking[2]},
        trace=trace,
        prefill_token_position=4,
    ):
        original = _FakeTensor(np.asarray([[[1.0, 5.0], [3.0, 4.0]]], dtype=np.float32))
        after_injection = blocks[0].run(original)
        after_downstream = blocks[1].run(after_injection)

    assert after_injection.value[0, -1, 0] == pytest.approx(5.0)
    assert after_downstream.value[0, -1, 0] == pytest.approx(5.0)
    downstream_projection = [
        observation
        for observation in trace.projection_observations
        if observation.layer == 2
    ]
    assert len(downstream_projection) == 1
    assert downstream_projection[0].projection == pytest.approx(5.0)
    assert all(not block.hooks for block in blocks)

    condition = {
        "condition_id": "prompt__target_00__dose_+1",
        "prompt_id": "prompt",
        "direction_kind": "target",
        "direction_index": 0,
        "dose": 1.0,
        "physical_scale": 2.0,
        "intervention_timing": "prefill_only",
        "order": 0,
        "seed": 1,
    }
    prompt = Namespace(
        parser_id="single_integer_choice_1_or_2_v1",
        expected_output_format="single_integer_1_or_2",
        metadata={"task_metadata": {}},
    )
    plan = {
        "construct_id": "integration_construct",
        "layer": 1,
        "position_mode": "last",
        "intervention_timing": "prefill_only",
        "activation_site": "resid_post",
        "model": {"model_id": "fake/model", "revision": "rev"},
        "provenance": {"run_config_hash": "run-hash", "construct_spec_hash": "spec-hash"},
    }
    rows = _trace_rows(
        condition=condition,
        prompt=prompt,
        output_text="1",
        trace=trace.to_mapping(),
        tracking={1: tracking[1], 2: tracking[2]},
        plan=plan,
        plan_sha256="plan-hash",
        prompt_inventory_sha256="prompt-hash",
        model=plan["model"],
        dtype="fp32",
        device="cpu",
        resolved_block_path="model.layers",
    )
    assert [row["record_id"] for row in rows] == [
        "prompt__target_00__dose_+1__tracking_layer_01",
        "prompt__target_00__dose_+1__tracking_layer_02",
    ]
    assert rows[0]["observed_shift"] == pytest.approx(2.0)
    assert rows[1]["downstream_projection"] == pytest.approx(5.0)

    raw_path = tmp_path / "steering.jsonl"
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = _build_output_manifest(
        plan=plan | {"conditions": [condition]},
        plan_sha256="plan-hash",
        prompt_inventory_sha256="prompt-hash",
        output=raw_path,
        tracking={1: tracking[1], 2: tracking[2]},
        args=Namespace(
            prompt_format="completion",
            system_prompt="",
            max_new_tokens=4,
            min_new_tokens=1,
            max_length=32,
            dtype="fp32",
            device="cpu",
            device_map=None,
            block_path="model.layers",
        ),
    )
    manifest["complete"] = True
    manifest["completed_record_count"] = 2
    manifest["raw_generations_sha256"] = file_sha256(raw_path)
    _output_manifest_path(raw_path).write_text(json.dumps(manifest), encoding="utf-8")
    validated_manifest, manifest_complete = _load_and_validate_output_manifest(
        raw_path,
        rows,
        construct_id="integration_construct",
        construct_spec_hash="spec-hash",
    )
    assert validated_manifest["expected_record_count"] == 2
    assert manifest_complete is True

    raw_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    manifest["complete"] = False
    manifest["completed_record_count"] = 0
    manifest["raw_generations_sha256"] = file_sha256(raw_path)
    _output_manifest_path(raw_path).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        _load_and_validate_output_manifest(
            raw_path,
            rows[:1],
            construct_id="integration_construct",
            construct_spec_hash="spec-hash",
        )
    _, diagnostic_complete = _load_and_validate_output_manifest(
        raw_path,
        rows[:1],
        construct_id="integration_construct",
        construct_spec_hash="spec-hash",
        allow_incomplete_diagnostic=True,
    )
    assert diagnostic_complete is False

    manifest["complete"] = True
    manifest["completed_record_count"] = 2
    _output_manifest_path(raw_path).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="has 1 rows"):
        _load_and_validate_output_manifest(
            raw_path,
            rows[:1],
            construct_id="integration_construct",
            construct_spec_hash="spec-hash",
        )


def test_tracking_plan_uses_independent_candidates_and_labels_fallbacks(tmp_path: Path) -> None:
    run_config = load_run_config(
        Path("configs/construct_benchmark/run_configs/wave1_realization_single_construct_smoke_v1.json")
    )
    injection_path = tmp_path / "injection.npy"
    candidate_path = tmp_path / "candidate_20.npy"
    np.save(injection_path, np.asarray([1.0, 0.0], dtype=np.float32))
    np.save(candidate_path, np.asarray([0.0, 1.0], dtype=np.float32))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    layers, tracking = _build_tracking_directions(
        summary={
            "candidate_directions": {
                "20": {
                    "path": str(candidate_path),
                    "source_split": "direction_train",
                    "calibration": {"projection_scale": 1.0},
                }
            }
        },
        summary_path=summary_path,
        run_config=run_config,
        construct_id="realization_account_closure",
        selected_layer=10,
        injection_direction=injection_path,
    )

    assert layers == [10, 20, 30]
    assert tracking["20"]["source"] == "independent_train_only"
    assert tracking["30"]["source"] == "same_vector_persistence_diagnostic"
    assert tracking["30"]["path"] == str(injection_path)


def test_output_manifest_round_trips_stringified_tracking_layer_keys(tmp_path: Path) -> None:
    plan = {
        "schema_version": "0.1.0",
        "run_id": "run",
        "construct_id": "construct",
        "model": {"model_id": "model", "revision": "rev"},
        "layer": 10,
        "position_mode": "last",
        "intervention_timing": "prefill_only",
        "activation_site": "resid_post",
        "conditions": [{"condition_id": "condition"}],
        "provenance": {"run_config_hash": "run-hash", "construct_spec_hash": "spec-hash"},
    }
    tracking = {
        10: {
            "layer": 10,
            "direction_id": "injection",
            "path": "injection.npy",
            "source": "injection_direction_train_only",
            "role": "injection_immediate",
        }
    }
    args = Namespace(
        prompt_format="chat",
        system_prompt="",
        max_new_tokens=4,
        min_new_tokens=1,
        max_length=32,
        dtype="fp32",
        device="cpu",
        device_map=None,
        block_path="model.layers",
    )
    manifest = _build_output_manifest(
        plan=plan,
        plan_sha256="plan-hash",
        prompt_inventory_sha256="prompt-hash",
        output=tmp_path / "steering.jsonl",
        tracking=tracking,
        args=args,
    )
    manifest_path = _output_manifest_path(tmp_path / "steering.jsonl")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _validate_output_manifest(manifest_path, manifest)

    incompatible = dict(manifest)
    incompatible["dtype"] = "bf16"
    with pytest.raises(ValueError, match="incompatible"):
        _validate_output_manifest(manifest_path, incompatible)
