#!/usr/bin/env python3
"""Run the preregistered CPU-only item-precision simulation.

This simulation is a planning gate, not an analysis of the observed steering
effects.  It asks how many independent downstream items are needed to estimate
the registered directed state-transfer estimand with useful power and a useful
confidence interval.  The effect grid, targets, and item-count grid are read
from a versioned JSON configuration so that they cannot be selected after
looking at the steering results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCRIPT_SCHEMA_VERSION = "0.1.0"
DEFAULT_ALPHA = 0.05


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _resolve(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _normal_quantile_975() -> float:
    # Fixed constant avoids making the planning gate depend on SciPy versions.
    return 1.959963984540054


def _validate_config(config: Mapping[str, Any]) -> None:
    required = (
        "simulation_id",
        "seed",
        "resamples",
        "alpha",
        "target_standardized_effect",
        "target_power",
        "target_ci_half_width",
        "effect_grid",
        "item_counts",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Precision configuration is missing: {missing}.")
    if float(config["alpha"]) <= 0 or float(config["alpha"]) >= 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if int(config["resamples"]) < 100:
        raise ValueError("resamples must be at least 100.")
    effects = [float(value) for value in config["effect_grid"]]
    counts = [int(value) for value in config["item_counts"]]
    if not effects or any(value <= 0 for value in effects):
        raise ValueError("effect_grid must contain positive standardized effects.")
    if not counts or any(value < 2 for value in counts) or counts != sorted(set(counts)):
        raise ValueError("item_counts must be sorted, unique integers >= 2.")
    for key in ("target_standardized_effect", "target_power", "target_ci_half_width"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if float(config["target_power"]) >= 1:
        raise ValueError("target_power must be below 1.")


def _extract_baseline_summary(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _load_object(path)
    behavior = payload.get("behavior", payload)
    if not isinstance(behavior, Mapping):
        raise ValueError(f"Baseline summary has no behavior mapping: {path}")
    constructs = behavior.get("constructs")
    if not isinstance(constructs, Mapping) or not constructs:
        raise ValueError(f"Baseline summary has no behavior.constructs mapping: {path}")
    extracted: dict[str, dict[str, Any]] = {}
    for construct_id, raw in constructs.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Baseline entry for {construct_id} is not an object.")
        count = int(raw.get("valid_primary_rows", raw.get("total_rows", 0)))
        if count < 2:
            raise ValueError(f"Baseline entry for {construct_id} has fewer than two valid items.")
        observed_sd = raw.get("outcome_sample_sd")
        scale = float(observed_sd) if observed_sd is not None else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        extracted[str(construct_id)] = {
            "current_item_count": count,
            "observed_outcome_sample_sd": observed_sd,
            "simulation_scale": scale,
        }
    return extracted, payload


def _simulate_cell(
    *,
    rng: np.random.Generator,
    item_count: int,
    standardized_effect: float,
    scale: float,
    resamples: int,
    alpha: float,
) -> dict[str, float]:
    """Simulate paired positive/negative item outcomes.

    The estimator is ``mean(positive - negative) / (2 * reference_sd)``.  The
    same reference scale is used for the data-generating process and the
    standardized estimator; this mirrors the preregistered use of a baseline
    or externally defined standard deviation and keeps the gate independent of
    the observed steering effect.
    """

    positive = standardized_effect * scale + rng.normal(0.0, scale, size=(resamples, item_count))
    negative = -standardized_effect * scale + rng.normal(0.0, scale, size=(resamples, item_count))
    standardized_differences = (positive - negative) / (2.0 * scale)
    estimates = standardized_differences.mean(axis=1)
    standard_errors = standardized_differences.std(axis=1, ddof=1) / np.sqrt(item_count)
    critical = _normal_quantile_975()
    lower = estimates - critical * standard_errors
    upper = estimates + critical * standard_errors
    significant = (lower > 0.0) | (upper < 0.0)
    return {
        "standardized_effect": float(standardized_effect),
        "item_count": int(item_count),
        "resamples": int(resamples),
        "power": float(significant.mean()),
        "mean_estimate": float(estimates.mean()),
        "mean_ci_half_width": float(((upper - lower) / 2.0).mean()),
        "median_ci_half_width": float(np.median((upper - lower) / 2.0)),
        "estimate_sd": float(estimates.std(ddof=1)),
        "lower_quantile": float(np.quantile(estimates, 0.025)),
        "upper_quantile": float(np.quantile(estimates, 0.975)),
    }


def run_simulation(
    *,
    config: Mapping[str, Any],
    baseline_path: Path,
) -> dict[str, Any]:
    _validate_config(config)
    baseline, baseline_payload = _extract_baseline_summary(baseline_path)
    seed = int(config["seed"])
    resamples = int(config["resamples"])
    alpha = float(config["alpha"])
    effect_grid = [float(value) for value in config["effect_grid"]]
    item_counts = [int(value) for value in config["item_counts"]]
    target_effect = float(config["target_standardized_effect"])
    target_power = float(config["target_power"])
    target_half_width = float(config["target_ci_half_width"])
    if target_effect not in effect_grid:
        raise ValueError("target_standardized_effect must be present in effect_grid.")

    construct_results: dict[str, Any] = {}
    for construct_index, (construct_id, baseline_info) in enumerate(sorted(baseline.items())):
        construct_seed = np.random.SeedSequence([seed, construct_index]).generate_state(1)[0]
        rng = np.random.default_rng(int(construct_seed))
        cells: list[dict[str, Any]] = []
        for item_count in item_counts:
            by_effect = {
                str(effect): _simulate_cell(
                    rng=rng,
                    item_count=item_count,
                    standardized_effect=effect,
                    scale=float(baseline_info["simulation_scale"]),
                    resamples=resamples,
                    alpha=alpha,
                )
                for effect in effect_grid
            }
            target_cell = by_effect[str(target_effect)]
            meets = bool(
                target_cell["power"] >= target_power
                and target_cell["mean_ci_half_width"] <= target_half_width
            )
            cells.append(
                {
                    "item_count": item_count,
                    "by_standardized_effect": by_effect,
                    "target_cell_meets_rule": meets,
                }
            )
        passing = [cell["item_count"] for cell in cells if cell["target_cell_meets_rule"]]
        current_count = int(baseline_info["current_item_count"])
        current_cells = [cell for cell in cells if cell["item_count"] == current_count]
        return_current = current_cells[0] if current_cells else None
        construct_results[construct_id] = {
            **baseline_info,
            "seed": int(construct_seed),
            "item_count_grid": cells,
            "minimum_item_count_meeting_rule": min(passing) if passing else None,
            "current_item_count_evaluated": current_count in item_counts,
            "current_item_count_meets_rule": bool(return_current and return_current["target_cell_meets_rule"]),
            "current_item_count_result": return_current,
        }

    all_current_pass = all(item["current_item_count_meets_rule"] for item in construct_results.values())
    recommended = [
        item["minimum_item_count_meeting_rule"]
        for item in construct_results.values()
        if item["minimum_item_count_meeting_rule"] is not None
    ]
    recommended_count = max(recommended) if recommended else None
    return {
        "schema_version": SCRIPT_SCHEMA_VERSION,
        "manifest_type": "precision_simulation_report",
        "simulation_id": str(config["simulation_id"]),
        "status": "complete",
        "confirmatory": False,
        "planning_only": True,
        "decision_rule": {
            "estimand": "paired_directed_state_transfer_standardized_by_baseline_sd",
            "target_standardized_effect": target_effect,
            "target_power": target_power,
            "target_ci_half_width": target_half_width,
            "alpha": alpha,
            "criterion": "two-sided 95% interval excludes zero and mean interval half-width is at most the registered target",
        },
        "simulation_design": {
            "seed": seed,
            "resamples": resamples,
            "effect_grid": effect_grid,
            "item_counts": item_counts,
            "pairing": "positive and negative outcomes are measured on the same downstream item set",
            "data_generating_process": "independent Gaussian residuals around opposite standardized effects; no observed steering effect is used",
        },
        "baseline_provenance": {
            "path": str(baseline_path),
            "construct_ids": sorted(baseline),
            "source_summary_manifest_type": baseline_payload.get("manifest_type"),
        },
        "constructs": construct_results,
        "decision": {
            "current_plan_meets_rule_for_all_constructs": all_current_pass,
            "recommended_minimum_item_count_across_constructs": recommended_count,
            "release_decision": "continue_current_plan" if all_current_pass else "expand_downstream_items_before_confirmatory_release",
            "wave2_4_confirmatory_release": bool(all_current_pass),
            "reason": (
                "Every current construct has the registered precision at the target effect."
                if all_current_pass
                else "At least one current construct is below the preregistered item-level precision rule."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_object(config_path)
    baseline_path = args.baseline_summary.resolve()
    report = run_simulation(config=config, baseline_path=baseline_path)
    report["config_path"] = str(config_path)
    report["config"] = dict(config)
    report["report_sha256"] = _canonical_hash(report)
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    import hashlib

    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
