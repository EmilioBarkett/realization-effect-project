"""Pure numerical manipulation checks for residual-stream interventions.

The model runner records scalar projections; this module turns those records
into auditable immediate-shift and downstream-persistence summaries.  It does
not import torch, transformers, or a model runtime, so the same checks can be
tested with deterministic fixtures and applied to RunPod JSONL outputs later.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def score_expected_observed_shift(
    pre_projection: float,
    post_projection: float,
    expected_shift: float,
) -> dict[str, float | None]:
    """Score the arithmetic of one pre/post injection observation."""

    values = (float(pre_projection), float(post_projection), float(expected_shift))
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Injection projections and expected_shift must be finite.")
    observed_shift = values[1] - values[0]
    error = values[2] - observed_shift
    relative_error = None if values[2] == 0 else error / abs(values[2])
    sign_agreement = (
        1.0
        if (values[2] == 0 and abs(observed_shift) == 0) or values[2] * observed_shift > 0
        else 0.0
    )
    return {
        "pre_projection": values[0],
        "post_projection": values[1],
        "observed_shift": observed_shift,
        "expected_shift": values[2],
        "expected_observed_difference": error,
        "absolute_error": abs(error),
        "relative_error": relative_error,
        "sign_agreement": sign_agreement,
    }


def raw_downstream_projection_transfer(
    downstream_projection: float,
    baseline_projection: float,
    injection_observed_shift: float,
) -> float | None:
    """Return the uncalibrated downstream change divided by injection shift.

    This is retained as a raw diagnostic only.  It is not comparable across
    layers because the numerator and denominator can use different direction
    spaces and activation scales.
    """

    values = (float(downstream_projection), float(baseline_projection), float(injection_observed_shift))
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Downstream and baseline projections must be finite.")
    if values[2] == 0:
        return None
    return (values[0] - values[1]) / values[2]


def downstream_persistence_ratio(
    downstream_projection: float,
    baseline_projection: float,
    injection_observed_shift: float,
    *,
    downstream_calibration_scale: float,
    injection_calibration_scale: float,
) -> float | None:
    """Return a layer-comparable, calibration-standardized persistence ratio.

    The downstream and injection shifts are first expressed in their own
    frozen training-calibration units.  The baseline is normally the same
    prompt's zero-dose record at the same tracking layer and direction.  A
    zero injection shift is reported as missing rather than converted to zero.
    """

    values = (
        float(downstream_projection),
        float(baseline_projection),
        float(injection_observed_shift),
        float(downstream_calibration_scale),
        float(injection_calibration_scale),
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Persistence projections and calibration scales must be finite.")
    if values[3] <= 0 or values[4] <= 0:
        raise ValueError("Persistence calibration scales must be greater than zero.")
    standardized_downstream_shift = (values[0] - values[1]) / values[3]
    standardized_injection_shift = values[2] / values[4]
    if standardized_injection_shift == 0:
        return None
    return standardized_downstream_shift / standardized_injection_shift


def _mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values if np.isfinite(float(value))]
    return None if not materialized else float(np.mean(materialized))


def _linear_slope(points: Iterable[tuple[float, float]]) -> float | None:
    materialized = [(float(x), float(y)) for x, y in points]
    if len(materialized) < 2 or len({x for x, _ in materialized}) < 2:
        return None
    x_values = np.asarray([x for x, _ in materialized], dtype=np.float64)
    y_values = np.asarray([y for _, y in materialized], dtype=np.float64)
    return float(np.polyfit(x_values, y_values, 1)[0])


def _group_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    expected = [float(row["expected_shift"]) for row in rows if row.get("expected_shift") is not None]
    observed = [float(row["observed_shift"]) for row in rows if row.get("observed_shift") is not None]
    errors = [
        float(row["expected_observed_difference"])
        for row in rows
        if row.get("expected_observed_difference") is not None
    ]
    return {
        "row_count": len(rows),
        "valid_count": len(observed),
        "requested_shift_mean": _mean(expected),
        "observed_shift_mean": _mean(observed),
        "expected_observed_difference_mean": _mean(errors),
        "absolute_error_mean": _mean(abs(value) for value in errors),
        "sign_agreement_rate": _mean(
            float(row["sign_agreement"])
            for row in rows
            if row.get("sign_agreement") is not None
        ),
    }


def summarize_manipulation_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize immediate manipulation and downstream persistence records.

    The input is intentionally mapping-based: it accepts the runner's JSONL
    rows and simple fake records without coupling the analysis core to a model
    class.  Rows should contain one injection-layer record and zero or more
    downstream-layer records per condition.
    """

    materialized = [dict(record) for record in records]
    injection_rows: list[dict[str, Any]] = []
    injection_candidates = 0
    zero_projection_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    injection_by_condition: dict[str, dict[str, Any]] = {}
    for record in materialized:
        tracking_role = str(record.get("tracking_role", ""))
        if tracking_role == "injection_immediate" or record.get("pre_projection") is not None:
            injection_candidates += 1
            if record.get("pre_projection") is not None and record.get("post_projection") is not None:
                expected_shift = record.get("expected_shift")
                if expected_shift is None:
                    expected_shift = record.get("physical_scale", 0.0)
                scored = score_expected_observed_shift(
                    float(record["pre_projection"]),
                    float(record["post_projection"]),
                    float(expected_shift),
                )
                injection_record = {**record, **scored}
                if injection_record.get("injection_calibration_projection_scale") is None:
                    injection_record["injection_calibration_projection_scale"] = record.get(
                        "calibrated_projection_scale"
                    )
                injection_rows.append(injection_record)
                injection_by_condition[str(record["condition_id"])] = injection_record
        if record.get("projection") is not None and record.get("tracking_layer") is not None:
            key = (
                str(record["prompt_id"]),
                int(record["tracking_layer"]),
                str(record.get("tracking_direction_id", "")),
            )
            if float(record.get("dose", 1.0)) == 0:
                zero_projection_by_key[key] = record

    grouped_injection: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in injection_rows:
        grouped_injection[(str(record.get("direction_kind", "")), float(record["dose"]))].append(record)
    injection_by_condition_and_dose = {
        f"{direction_kind}:{dose:g}": _group_summary(rows)
        for (direction_kind, dose), rows in sorted(grouped_injection.items())
    }
    slope_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (direction_kind, dose), rows in grouped_injection.items():
        mean_observed = _mean(float(row["observed_shift"]) for row in rows)
        if mean_observed is not None:
            slope_points[direction_kind].append((dose, mean_observed))
    dose_response_slopes = {
        direction_kind: _linear_slope(points)
        for direction_kind, points in sorted(slope_points.items())
    }

    persistence_groups: dict[tuple[str, float, int, str], list[float]] = defaultdict(list)
    persistence_missing: dict[tuple[str, float, int, str], int] = defaultdict(int)
    raw_transfer_groups: dict[tuple[str, float, int, str], list[float]] = defaultdict(list)
    standardized_downstream_groups: dict[tuple[str, float, int, str], list[float]] = defaultdict(list)
    standardized_injection_groups: dict[tuple[str, float, int, str], list[float]] = defaultdict(list)
    for record in materialized:
        if record.get("projection") is None or record.get("tracking_layer") is None:
            continue
        if str(record.get("tracking_role", "")) == "injection_immediate":
            continue
        dose = float(record["dose"])
        layer = int(record["tracking_layer"])
        direction_kind = str(record.get("direction_kind", ""))
        role = str(record.get("tracking_role", ""))
        group_key = (direction_kind, dose, layer, role)
        if dose == 0:
            continue
        baseline_key = (
            str(record["prompt_id"]),
            layer,
            str(record.get("tracking_direction_id", "")),
        )
        baseline = zero_projection_by_key.get(baseline_key)
        injection = injection_by_condition.get(str(record["condition_id"]))
        if baseline is None or injection is None:
            persistence_missing[group_key] += 1
            continue
        raw_transfer = raw_downstream_projection_transfer(
            float(record["projection"]),
            float(baseline["projection"]),
            float(injection["observed_shift"]),
        )
        if raw_transfer is not None:
            raw_transfer_groups[group_key].append(raw_transfer)
        downstream_calibration_scale = record.get("tracking_calibration_projection_scale")
        injection_calibration_scale = injection.get("injection_calibration_projection_scale")
        if downstream_calibration_scale is None or injection_calibration_scale is None:
            persistence_missing[group_key] += 1
            continue
        try:
            downstream_scale = float(downstream_calibration_scale)
            injection_scale = float(injection_calibration_scale)
            standardized_downstream_shift = (
                float(record["projection"]) - float(baseline["projection"])
            ) / downstream_scale
            standardized_injection_shift = float(injection["observed_shift"]) / injection_scale
            ratio = downstream_persistence_ratio(
                float(record["projection"]),
                float(baseline["projection"]),
                float(injection["observed_shift"]),
                downstream_calibration_scale=downstream_scale,
                injection_calibration_scale=injection_scale,
            )
        except (TypeError, ValueError, ZeroDivisionError):
            persistence_missing[group_key] += 1
            continue
        standardized_downstream_groups[group_key].append(standardized_downstream_shift)
        standardized_injection_groups[group_key].append(standardized_injection_shift)
        if ratio is not None:
            persistence_groups[group_key].append(ratio)
        else:
            persistence_missing[group_key] += 1

    persistence_summary = {}
    for key in sorted(
        set(persistence_groups)
        | set(persistence_missing)
        | set(raw_transfer_groups)
        | set(standardized_downstream_groups)
    ):
        direction_kind, dose, layer, role = key
        ratios = persistence_groups.get(key, [])
        persistence_summary[
            f"{direction_kind}:{dose:g}:layer_{layer:02d}:{role}"
        ] = {
            "direction_kind": direction_kind,
            "dose": dose,
            "tracking_layer": layer,
            "tracking_role": role,
            "valid_count": len(ratios),
            "missing_count": persistence_missing.get(key, 0),
            "raw_projection_transfer_mean": _mean(raw_transfer_groups.get(key, [])),
            "standardized_downstream_shift_mean": _mean(
                standardized_downstream_groups.get(key, [])
            ),
            "standardized_injection_shift_mean": _mean(
                standardized_injection_groups.get(key, [])
            ),
            "persistence_ratio_mean": _mean(ratios),
            "persistence_ratio_definition": (
                "((downstream_projection - zero_dose_projection) / "
                "downstream_training_calibration) / "
                "(injection_observed_shift / injection_training_calibration)"
            ),
        }

    return {
        "record_count": len(materialized),
        "injection_candidate_count": injection_candidates,
        "injection_record_count": len(injection_rows),
        "missing_injection_records": injection_candidates - len(injection_rows),
        "injection_by_direction_and_dose": injection_by_condition_and_dose,
        "dose_response_slopes": dose_response_slopes,
        "downstream_persistence": persistence_summary,
        "missing_or_unscorable_records": (
            injection_candidates - len(injection_rows) + sum(persistence_missing.values())
        ),
    }


__all__ = [
    "downstream_persistence_ratio",
    "raw_downstream_projection_transfer",
    "score_expected_observed_shift",
    "summarize_manipulation_records",
]
