"""Deterministic, no-model fixtures for the benchmark vertical slice.

The fake runner is deliberately synthetic. It exercises prompt validation,
train-only direction construction, validation layer selection, neutral
calibration, held-out readout, steering controls, and item-level uncertainty
without calling an API or loading model weights.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np

from activation_analysis.vector_analysis import PromptActivation

from .behavior import BehaviorObservation
from .calibration import CalibrationResult, estimate_projection_scale
from .readout import (
    DirectionEstimate,
    ReadoutResult,
    estimate_train_direction,
    evaluate_heldout_readout,
    evaluate_validation_readout,
)
from .prompts import PromptRecord
from .schemas import ConstructSpec, RunConfig
from .steering import build_steering_conditions
from .uncertainty import (
    BootstrapInterval,
    bootstrap_readout_margin_ci,
    bootstrap_state_transfer_ci,
)


FAKE_HIDDEN_SIZE = 16
PAIRED_COUNTS = {
    "direction_train": 8,
    "direction_validation": 4,
    "direction_heldout": 6,
}
DOWNSTREAM_COUNT = 6


def _seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % (2**32)


def _nonce(*parts: object) -> str:
    text = "::".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _task_metadata(spec: ConstructSpec, index: int) -> dict[str, Any]:
    properties = spec.independent_behavior_task["item_metadata_schema"]["properties"]
    metadata: dict[str, Any] = {}
    for offset, (field_name, field_schema) in enumerate(properties.items()):
        enum = field_schema.get("enum")
        if isinstance(enum, list) and enum:
            metadata[field_name] = enum[(index + offset) % len(enum)]
            continue
        property_type = field_schema["type"]
        if property_type == "boolean":
            metadata[field_name] = (index + offset) % 2 == 0
        elif property_type == "integer":
            minimum = int(field_schema.get("minimum", 0))
            maximum = int(field_schema.get("maximum", minimum + 10))
            metadata[field_name] = minimum + ((index + offset) * 17) % (maximum - minimum + 1)
        elif property_type == "number":
            minimum = float(field_schema.get("minimum", 0.0))
            maximum = float(field_schema.get("maximum", minimum + 1.0))
            metadata[field_name] = minimum + ((index + offset) % 5) * (maximum - minimum) / 4.0
        else:
            metadata[field_name] = f"fake_{field_name}_{index}_{offset}"
    return metadata


def build_fake_prompt_inventory(spec: ConstructSpec) -> list[PromptRecord]:
    """Create independent, balanced prompt records for one construct."""

    records: list[PromptRecord] = []
    for split, pair_count in PAIRED_COUNTS.items():
        prompt_family = f"{spec.construct_id}_probe_{split}"
        for pair_index in range(pair_count):
            pair_id = f"fake_{spec.construct_id}_{split}_pair_{pair_index:03d}"
            for condition_index, condition_id in enumerate(spec.condition_ids):
                records.append(
                    PromptRecord(
                        prompt_id=f"{pair_id}__{condition_id}",
                        construct_id=spec.construct_id,
                        split=split,
                        prompt_role="probe",
                        prompt_text=(
                            f"Synthetic scenario {_nonce(spec.construct_id, split, pair_index, condition_index)} "
                            "describes an unfamiliar event for memory."
                        ),
                        condition_id=condition_id,
                        pair_id=pair_id,
                        pair_role=condition_id,
                        prompt_family=prompt_family,
                        metadata={"fake_fixture": True},
                    )
                )

    for split, prompt_role in (
        ("behavior_eval", "behavior"),
        ("steering_eval", "steering"),
        ("calibration", "calibration"),
    ):
        prompt_family = f"{spec.construct_id}_{prompt_role}_{split}"
        for index in range(DOWNSTREAM_COUNT):
            task_metadata = _task_metadata(spec, index)
            records.append(
                PromptRecord(
                    prompt_id=f"fake_{spec.construct_id}_{split}_{index:03d}",
                    construct_id=spec.construct_id,
                    split=split,
                    prompt_role=prompt_role,
                    prompt_text=(
                        f"Synthetic independent task {_nonce(spec.construct_id, split, index)} "
                        "asks for a structured response."
                    ),
                    condition_id="neutral",
                    prompt_family=prompt_family,
                    task_id=spec.independent_behavior_task["task_id"],
                    expected_output_format=spec.independent_behavior_task["response_format"],
                    parser_id=spec.parsing_rules["parser_id"],
                    metadata={"fake_fixture": True, "task_metadata": task_metadata, **task_metadata},
                )
            )
    return records


def build_fake_activations(
    records: Iterable[PromptRecord],
    spec: ConstructSpec,
    layers: Iterable[int],
) -> list[PromptActivation]:
    """Create layer-separated activations with a known construct signal."""

    signal_rng = np.random.default_rng(_seed(spec.construct_id, "signal"))
    signal = signal_rng.normal(size=FAKE_HIDDEN_SIZE).astype(np.float32)
    signal /= np.linalg.norm(signal)
    ordered_layers = sorted(layers)
    base_strengths = (0.8, 1.5, 0.9)
    layer_strengths = {
        layer: base_strengths[index] if index < len(base_strengths) else 0.8 + 0.1 * (index % 3)
        for index, layer in enumerate(ordered_layers)
    }
    activations: list[PromptActivation] = []
    for record in records:
        for layer in ordered_layers:
            rng = np.random.default_rng(_seed(spec.construct_id, record.prompt_id, layer))
            base_key = record.pair_id or record.prompt_id
            base_rng = np.random.default_rng(_seed(spec.construct_id, base_key, layer, "base"))
            vector = base_rng.normal(0.0, 0.15, size=FAKE_HIDDEN_SIZE).astype(np.float32)
            vector += rng.normal(0.0, 0.04, size=FAKE_HIDDEN_SIZE).astype(np.float32)
            if record.split in spec.paired_splits:
                state_sign = 1.0 if record.condition_id == spec.positive_condition_id else -1.0
                vector += (state_sign * 0.5 * layer_strengths[layer] * signal).astype(np.float32)
            elif record.split == "calibration":
                vector += (rng.normal(0.0, 0.35) * signal).astype(np.float32)
            activations.append(
                PromptActivation(
                    prompt_id=record.prompt_id,
                    metadata={
                        "construct_id": spec.construct_id,
                        "split": record.split,
                        "condition_id": record.condition_id or "",
                        "pair_role": record.pair_role or "",
                        "pair_id": record.pair_id or "",
                        "prompt_metadata": record.to_mapping(),
                        "layer": layer,
                        "activation_site": "resid_post",
                    },
                    vector=vector,
                    token_count=8,
                    layer=layer,
                )
            )
    return activations


def _readout_mapping(result: ReadoutResult) -> dict[str, Any]:
    return {
        "split": result.split,
        "pair_count": result.pair_count,
        "mean_standardized_margin": result.mean_standardized_margin,
        "pair_accuracy": result.pair_accuracy,
    }


def _direction_and_readout(
    activations: list[PromptActivation],
    spec: ConstructSpec,
    layers: list[int],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[int, DirectionEstimate, dict[str, Any], dict[str, Any], BootstrapInterval]:
    candidates: list[dict[str, Any]] = []
    estimates: dict[int, DirectionEstimate] = {}
    calibrations = {}
    selected_by_layer: dict[int, list[PromptActivation]] = {}
    for layer in layers:
        layer_activations = [activation for activation in activations if activation.layer == layer]
        estimate = estimate_train_direction(
            layer_activations,
            construct_id=spec.construct_id,
            positive_condition_id=spec.positive_condition_id,
            negative_condition_id=spec.negative_condition_id,
        )
        calibration = estimate_projection_scale(
            layer_activations,
            estimate.direction,
            construct_id=spec.construct_id,
            method="neutral",
        )
        validation = evaluate_validation_readout(
            layer_activations,
            estimate,
            projection_scale=calibration.projection_scale,
        )
        estimates[layer] = estimate
        calibrations[layer] = calibration
        selected_by_layer[layer] = layer_activations
        candidates.append({"layer": layer, **_readout_mapping(validation)})

    selected_layer = min(
        layers,
        key=lambda layer: (
            -next(item["mean_standardized_margin"] for item in candidates if item["layer"] == layer),
            layer,
        ),
    )
    estimate = estimates[selected_layer]
    calibration = calibrations[selected_layer]
    heldout = evaluate_heldout_readout(
        selected_by_layer[selected_layer],
        estimate,
        projection_scale=calibration.projection_scale,
    )
    heldout_ci = bootstrap_readout_margin_ci(
        heldout.margins,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    layer_selection = {
        "rule": "validation_max_margin",
        "selection_split": "direction_validation",
        "selection_metric": "mean_standardized_margin",
        "candidate_layers": candidates,
        "selected_layer": selected_layer,
    }
    readout = {
        "direction": {
            "pair_count": estimate.pair_count,
            "norm": float(np.linalg.norm(estimate.direction)),
            "source_split": "direction_train",
        },
        "validation": next(item for item in candidates if item["layer"] == selected_layer),
        "heldout": _readout_mapping(heldout),
        "heldout_uncertainty": heldout_ci.to_mapping(),
        "calibration": asdict(calibration),
    }
    return selected_layer, estimate, layer_selection, readout, heldout_ci


def run_fake_construct(
    spec: ConstructSpec,
    run_config: RunConfig,
    *,
    bootstrap_resamples: int = 250,
    bootstrap_seed: int = 17,
) -> tuple[list[PromptRecord], dict[str, Any]]:
    """Run the synthetic vertical slice for one registered construct."""

    records = build_fake_prompt_inventory(spec)
    layers = [int(layer) for layer in run_config.activation["layers"]]
    activations = build_fake_activations(records, spec, layers)
    selected_layer, estimate, layer_selection, readout, _ = _direction_and_readout(
        activations,
        spec,
        layers,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    steering_records = [record for record in records if record.split == "steering_eval"]
    calibration = readout["calibration"]
    steering_calibration = CalibrationResult(**calibration)
    conditions = build_steering_conditions(
        [record.prompt_id for record in steering_records],
        steering_calibration,
        doses=run_config.steering["scales"],
        intervention_timing=run_config.steering["intervention_timing"],
        seed=run_config.seed,
        include_shuffled=True,
        random_direction_count=int(run_config.steering["random_direction_count"]),
    )
    target_conditions = [condition for condition in conditions if condition.direction_kind == "target"]
    observations: list[BehaviorObservation] = []
    for condition in target_conditions:
        baseline = 5.0 + (int(_seed(spec.construct_id, condition.prompt_id)) % 7) * 0.2
        noise = ((int(_seed(condition.prompt_id, "outcome")) % 11) - 5) * 0.01
        observations.append(
            BehaviorObservation(
                item_id=condition.prompt_id,
                scale=condition.dose,
                outcome=baseline + 0.75 * condition.dose + noise,
                valid=True,
            )
        )
    positive_dose = max(float(dose) for dose in run_config.steering["scales"])
    negative_dose = min(float(dose) for dose in run_config.steering["scales"])
    effect, effect_ci = bootstrap_state_transfer_ci(
        observations,
        positive_scale=positive_dose,
        negative_scale=negative_dose,
        zero_scale=0.0,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 1,
    )
    control_counts = dict(sorted(Counter(condition.direction_kind for condition in conditions).items()))
    summary = {
        "construct_id": spec.construct_id,
        "prompt_count": len(records),
        "activation_count": len(activations),
        "candidate_layers": layers,
        "selected_layer": selected_layer,
        "layer_selection": layer_selection,
        "readout": readout,
        "steering": {
            "intervention_timing": run_config.steering["intervention_timing"],
            "doses": [float(dose) for dose in run_config.steering["scales"]],
            "calibration": calibration,
            "condition_counts": control_counts,
            "target_direction_effect": asdict(effect),
            "uncertainty": effect_ci.to_mapping(),
        },
        "fake_fixture": {
            "hidden_size": FAKE_HIDDEN_SIZE,
            "model_loaded": False,
            "api_called": False,
            "empirical_result": False,
        },
    }
    del estimate
    return records, summary


__all__ = [
    "DOWNSTREAM_COUNT",
    "FAKE_HIDDEN_SIZE",
    "PAIRED_COUNTS",
    "build_fake_activations",
    "build_fake_prompt_inventory",
    "run_fake_construct",
]
