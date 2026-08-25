from __future__ import annotations

import numpy as np
import pytest

from activation_analysis.vector_analysis import PromptActivation
from activation_analysis.steering import should_inject_at_forward
from construct_benchmark.behavior import (
    BehaviorObservation,
    directed_mean_state_transfer,
    orient_primary_outcome,
    parse_behavior_output,
    primary_outcome,
)
from construct_benchmark.calibration import estimate_projection_scale, intervention_scale
from construct_benchmark.readout import estimate_train_direction, evaluate_heldout_readout
from construct_benchmark.steering import (
    build_steering_conditions,
    random_control_direction,
    shuffled_label_direction,
)
from construct_benchmark.uncertainty import bootstrap_readout_margin_ci, bootstrap_state_transfer_ci


def _activation(
    prompt_id: str,
    vector: list[float],
    *,
    split: str,
    condition_id: str = "neutral",
    pair_id: str = "",
) -> PromptActivation:
    return PromptActivation(
        prompt_id=prompt_id,
        metadata={
            "construct_id": "example_construct",
            "split": split,
            "condition_id": condition_id,
            "pair_role": condition_id,
            "pair_id": pair_id,
        },
        vector=np.asarray(vector, dtype=np.float32),
        token_count=4,
    )


def _activation_fixture() -> list[PromptActivation]:
    return [
        _activation("train_p1", [3, 0], split="direction_train", condition_id="positive", pair_id="train_1"),
        _activation("train_n1", [1, 0], split="direction_train", condition_id="negative", pair_id="train_1"),
        _activation("train_p2", [4, 1], split="direction_train", condition_id="positive", pair_id="train_2"),
        _activation("train_n2", [1, 1], split="direction_train", condition_id="negative", pair_id="train_2"),
        _activation("held_p1", [5, 0], split="direction_heldout", condition_id="positive", pair_id="held_1"),
        _activation("held_n1", [1, 0], split="direction_heldout", condition_id="negative", pair_id="held_1"),
        _activation("held_p2", [2, 1], split="direction_heldout", condition_id="positive", pair_id="held_2"),
        _activation("held_n2", [3, 1], split="direction_heldout", condition_id="negative", pair_id="held_2"),
        _activation("cal_1", [0, 0], split="calibration"),
        _activation("cal_2", [2, 0], split="calibration"),
        _activation("cal_3", [4, 0], split="calibration"),
    ]


def test_train_only_direction_calibration_and_heldout_margin() -> None:
    activations = _activation_fixture()
    estimate = estimate_train_direction(
        activations,
        construct_id="example_construct",
        positive_condition_id="positive",
        negative_condition_id="negative",
    )
    assert estimate.pair_count == 2
    assert np.allclose(estimate.direction, [2.5, 0.0])

    calibration = estimate_projection_scale(
        activations,
        estimate.direction,
        construct_id="example_construct",
        method="neutral",
    )
    assert calibration.sample_count == 3
    assert calibration.projection_scale == pytest.approx(2.0)
    assert intervention_scale(calibration, -1.5) == pytest.approx(-3.0)

    readout = evaluate_heldout_readout(
        activations,
        estimate,
        projection_scale=calibration.projection_scale,
    )
    assert readout.pair_count == 2
    assert readout.mean_standardized_margin == pytest.approx(0.75)
    assert readout.pair_accuracy == pytest.approx(0.5)


def test_bootstrap_intervals_resample_complete_units() -> None:
    activations = _activation_fixture()
    estimate = estimate_train_direction(
        activations,
        construct_id="example_construct",
        positive_condition_id="positive",
        negative_condition_id="negative",
    )
    calibration = estimate_projection_scale(
        activations,
        estimate.direction,
        construct_id="example_construct",
        method="neutral",
    )
    readout = evaluate_heldout_readout(
        activations,
        estimate,
        projection_scale=calibration.projection_scale,
    )
    readout_ci = bootstrap_readout_margin_ci(readout.margins, resamples=100, seed=3)
    assert readout_ci.estimate == pytest.approx(readout.mean_standardized_margin)
    assert readout_ci.valid_resamples == 100
    observations = [
        BehaviorObservation(f"item_{index}", scale, 5.0 + 0.5 * scale + index * 0.1, True)
        for index in range(5)
        for scale in (-1.0, 0.0, 1.0)
    ]
    effect, effect_ci = bootstrap_state_transfer_ci(
        observations,
        positive_scale=1.0,
        negative_scale=-1.0,
        resamples=100,
        seed=3,
    )
    assert effect.directed_standardized_effect > 0
    assert effect_ci.estimate == pytest.approx(effect.directed_standardized_effect)
    assert 0 < effect_ci.valid_resamples <= 100


def test_within_condition_calibration_removes_between_condition_separation() -> None:
    activations = _activation_fixture()
    estimate = estimate_train_direction(
        activations,
        construct_id="example_construct",
        positive_condition_id="positive",
        negative_condition_id="negative",
    )
    result = estimate_projection_scale(
        activations,
        estimate.direction,
        construct_id="example_construct",
        method="within_condition",
    )
    assert result.group_count == 2
    assert result.projection_scale == pytest.approx(0.5)


def test_readout_rejects_nontraining_direction_input_and_incomplete_pairs() -> None:
    activations = _activation_fixture()[4:]
    with pytest.raises(ValueError, match="No complete direction_train pairs"):
        estimate_train_direction(
            activations,
            construct_id="example_construct",
            positive_condition_id="positive",
            negative_condition_id="negative",
        )


@pytest.mark.parametrize(
    ("parser_id", "text", "expected"),
    [
        ("two_integers_risk_choice_v1", "500\n4", {"allocation_amount": 500.0, "risk_preference": 4.0}),
        ("two_integers_sum_100_v1", "65 35", {"continue_allocation": 65.0, "alternative_allocation": 35.0}),
        ("single_integer_probability_v1", "72", {"probability": 72.0, "posterior_probability": 72.0}),
    ],
)
def test_wave_one_parsers_are_strict(parser_id: str, text: str, expected: dict[str, float]) -> None:
    parsed = parse_behavior_output(text, parser_id=parser_id)
    assert parsed.valid
    assert parsed.values == expected
    assert not parse_behavior_output(f"Answer: {text}", parser_id=parser_id).valid


def test_probability_outcome_requires_structured_prior_metadata() -> None:
    without_prior = parse_behavior_output("72", parser_id="single_integer_probability_v1")
    with pytest.raises(ValueError, match="structured item metadata"):
        primary_outcome(without_prior, "absolute_posterior_update")

    with_prior = parse_behavior_output(
        "72",
        parser_id="single_integer_probability_v1",
        item_metadata={"prior_probability": 40},
    )
    assert primary_outcome(with_prior, "absolute_posterior_update") == pytest.approx(32.0)

    testimony = parse_behavior_output(
        "25",
        parser_id="single_integer_probability_v1",
        item_metadata={"prior_probability": 40, "testimony_valence": "contradicting"},
    )
    assert primary_outcome(testimony, "testimony_weight") == pytest.approx(15.0)


def test_realization_outcome_orientation_respects_registered_valence() -> None:
    assert orient_primary_outcome("realization_account_closure", 4.0, {"outcome_valence": "gain"}) == 4.0
    assert orient_primary_outcome("realization_account_closure", 4.0, {"outcome_valence": "loss"}) == -4.0
    assert orient_primary_outcome("realization_account_closure", 4.0, {"outcome_valence": "neutral"}) is None


def test_directed_state_transfer_excludes_invalid_rows_and_uses_zero_sd() -> None:
    observations = [
        BehaviorObservation("p1", 1.0, 8.0, True),
        BehaviorObservation("p2", 1.0, None, False),
        BehaviorObservation("n1", -1.0, 4.0, True),
        BehaviorObservation("z1", 0.0, 4.0, True),
        BehaviorObservation("z2", 0.0, 6.0, True),
    ]
    result = directed_mean_state_transfer(
        observations,
        positive_scale=1.0,
        negative_scale=-1.0,
    )
    assert result.directed_standardized_effect == pytest.approx(np.sqrt(2.0))
    assert result.valid_counts == {"positive": 1, "negative": 1, "zero": 2}
    assert result.total_counts == {"positive": 2, "negative": 1, "zero": 2}


def test_registered_steering_timings_map_to_forward_passes() -> None:
    assert [should_inject_at_forward("prefill_only", index) for index in range(3)] == [True, False, False]
    assert [should_inject_at_forward("generation_only", index) for index in range(3)] == [False, True, True]
    assert [should_inject_at_forward("every_step", index) for index in range(3)] == [True, True, True]
    assert [
        should_inject_at_forward("fixed_window", index, fixed_window_start=1, fixed_window_end=3)
        for index in range(4)
    ] == [False, True, True, False]


def test_steering_plan_has_target_shuffled_and_multiple_random_controls() -> None:
    calibration = estimate_projection_scale(
        _activation_fixture(),
        np.asarray([1.0, 0.0], dtype=np.float32),
        construct_id="example_construct",
    )
    first = build_steering_conditions(
        ["item_1"],
        calibration,
        doses=[-1, 0, 1],
        intervention_timing="prefill_only",
        seed=17,
    )
    second = build_steering_conditions(
        ["item_1"],
        calibration,
        doses=[-1, 0, 1],
        intervention_timing="prefill_only",
        seed=17,
    )
    assert first == second
    assert len(first) == 9
    assert {condition.direction_kind for condition in first} == {"target", "shuffled", "random"}
    assert len({condition.seed for condition in first}) == len(first)
    assert next(condition for condition in first if condition.dose == 1).physical_scale == pytest.approx(2.0)


def test_control_directions_are_reproducible_and_random_is_orthogonal() -> None:
    differences = np.asarray([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 1.0]])
    shuffled = shuffled_label_direction(differences, seed=3)
    assert np.allclose(shuffled, shuffled_label_direction(differences, seed=3))
    target = np.asarray([1.0, 0.0, 0.0])
    random_direction = random_control_direction(3, seed=9, orthogonal_to=target)
    assert float(np.dot(random_direction, target)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.linalg.norm(random_direction)) == pytest.approx(1.0)
