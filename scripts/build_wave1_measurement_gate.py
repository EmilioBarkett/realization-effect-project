#!/usr/bin/env python3
"""Build a frozen, non-confirmatory Wave 1 B/R/C/S measurement-gate report.

This command only consumes completed CPU-derived summaries, manifests, and
control audits.  It does not load a model, inspect raw activations into the
local repository, or promote an older run's ``confirmatory`` flag.  The
historical Wave 1 Mistral run predates the current gate contract, so all
results assembled here are explicitly reclassified as engineering evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from construct_benchmark.manifests import canonical_hash, file_sha256


SCHEMA_VERSION = "0.1.0"
CONSTRUCT_IDS = (
    "realization_account_closure",
    "evidence_diagnosticity",
    "source_reliability",
    "persistence_continuation",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _resolve(value: str | Path, *, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _path_map(value: Any, *, root: Path) -> dict[str, Path]:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a construct-to-path mapping.")
    return {str(key): _resolve(item, root=root) for key, item in value.items()}


def _manifest_status(path: Path, *, kind: str) -> dict[str, Any]:
    payload = _load_object(path)
    complete = payload.get("complete")
    if kind == "activation":
        execution = payload.get("execution")
        if isinstance(execution, Mapping):
            complete = execution.get("complete", complete)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "complete": bool(complete),
        "source_confirmatory": bool(payload.get("confirmatory", False)),
        "schema_version": payload.get("schema_version"),
        "manifest_type": payload.get("manifest_type", payload.get("type")),
        "expected_record_count": payload.get("expected_record_count"),
        "completed_record_count": payload.get("completed_record_count"),
        "expected_observation_count": payload.get("expected_observation_count"),
        "completed_observation_count": payload.get("completed_observation_count"),
    }


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    return next((value for value in values if isinstance(value, Mapping)), None)


def _ci_fields(value: Any) -> tuple[float | None, float | None, float | None]:
    if not isinstance(value, Mapping):
        return None, None, None
    estimate = value.get("estimate")
    lower = value.get("lower")
    upper = value.get("upper")
    return (
        None if estimate is None else float(estimate),
        None if lower is None else float(lower),
        None if upper is None else float(upper),
    )


def _stage_result(status: str, *, detail: str = "", **fields: Any) -> dict[str, Any]:
    return {"status": status, "pass": status == "pass", "detail": detail, **fields}


def build_gate(*, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    construct_ids = tuple(config.get("construct_ids", CONSTRUCT_IDS))
    if construct_ids != CONSTRUCT_IDS:
        raise ValueError("The Wave 1 gate requires the frozen four-construct registry.")
    stages = config.get("stages")
    manifests = config.get("manifests")
    controls = _path_map(config.get("controls"), root=root)
    if not isinstance(stages, Mapping) or not isinstance(manifests, Mapping):
        raise ValueError("Gate config must contain stages and manifests mappings.")
    behavior_path = _resolve(stages["B"], root=root)
    causal_path = _resolve(stages["C"], root=root)
    readout_paths = _path_map(stages["R"], root=root)
    steering_paths = _path_map(stages["S"], root=root)
    behavior = _load_object(behavior_path)
    causal = _load_object(causal_path)
    readouts = {construct_id: _load_object(readout_paths[construct_id]) for construct_id in construct_ids}
    steerings = {construct_id: _load_object(steering_paths[construct_id]) for construct_id in construct_ids}
    control_reports = {construct_id: _load_object(controls[construct_id]) for construct_id in construct_ids}

    manifest_paths = {
        "activation": _resolve(manifests["activation"], root=root),
        "behavior": _resolve(manifests["behavior"], root=root),
        "causal": _resolve(manifests["causal"], root=root),
        "steering": _path_map(manifests["steering"], root=root),
    }
    manifest_status = {
        "activation": _manifest_status(manifest_paths["activation"], kind="activation"),
        "behavior": _manifest_status(manifest_paths["behavior"], kind="output"),
        "causal": _manifest_status(manifest_paths["causal"], kind="output"),
        "steering": {
            construct_id: _manifest_status(manifest_paths["steering"][construct_id], kind="output")
            for construct_id in construct_ids
        },
    }

    behavior_namespace = behavior.get("behavior", behavior)
    behavior_constructs = behavior_namespace.get("constructs", {}) if isinstance(behavior_namespace, Mapping) else {}
    b_ok = bool(
        behavior.get("manifest_complete") is True
        and behavior.get("pass") is True
        and isinstance(behavior_constructs, Mapping)
        and all(
            isinstance(behavior_constructs.get(construct_id), Mapping)
            and behavior_constructs[construct_id].get("valid_primary_rows", 0) > 0
            for construct_id in construct_ids
        )
        and manifest_status["behavior"]["complete"]
    )
    b = _stage_result(
        "pass" if b_ok else "fail",
        detail="Completed manifest-backed prompt-only baseline with usable primary outcomes."
        if b_ok
        else "Behavior baseline summary, variation gate, or manifest is incomplete.",
        source_summary=str(behavior_path),
        manifest=manifest_status["behavior"],
    )

    r_entries: dict[str, Any] = {}
    r_ok = True
    for construct_id in construct_ids:
        summary = readouts[construct_id]
        readout = summary.get("readout", {})
        direction = summary.get("direction", {})
        calibration = summary.get("calibration", {})
        layer_selection = summary.get("layer_selection", {})
        entry_ok = bool(
            isinstance(readout, Mapping)
            and readout.get("mean_standardized_margin") is not None
            and isinstance(direction, Mapping)
            and direction.get("source_split") == "direction_train"
            and isinstance(calibration, Mapping)
            and float(calibration.get("projection_scale", 0)) > 0
            and isinstance(layer_selection, Mapping)
            and layer_selection.get("rule") == "validation_max_margin"
            and layer_selection.get("selection_split") in {"direction_validation", "validation"}
        )
        r_ok = r_ok and entry_ok
        r_entries[construct_id] = {
            "status": "pass" if entry_ok else "fail",
            "mean_standardized_margin": readout.get("mean_standardized_margin"),
            "pair_accuracy": readout.get("pair_accuracy"),
            "pair_count": readout.get("pair_count"),
            "heldout_split": readout.get("split"),
            "direction_source_split": direction.get("source_split"),
            "calibration_projection_scale": calibration.get("projection_scale"),
            "selected_layer": summary.get("selected_layer"),
            "layer_selection": layer_selection,
            "source_summary": str(readout_paths[construct_id]),
        }
    r = _stage_result(
        "pass" if r_ok and manifest_status["activation"]["complete"] else "fail",
        detail="Train-only directions, validation-only layer selection, held-out readouts, and positive calibration scales are present."
        if r_ok and manifest_status["activation"]["complete"]
        else "Representation summary or activation manifest does not satisfy the frozen readout contract.",
        activation_manifest=manifest_status["activation"],
        constructs=r_entries,
    )

    c_observed = int(causal.get("observation_count", -1))
    c_expected = manifest_status["causal"].get("expected_observation_count")
    if c_expected is None:
        c_expected = c_observed
    c_ok = bool(
        causal.get("complete") is True
        and c_observed == int(c_expected)
        and manifest_status["causal"]["complete"]
    )
    c = _stage_result(
        "pass" if c_ok else "fail",
        detail="Complete matched residual-interchange diagnostic is available."
        if c_ok
        else "Causal residual-interchange output is incomplete or mismatched.",
        source_summary=str(causal_path),
        manifest=manifest_status["causal"],
        observation_count=c_observed,
        expected_observation_count=c_expected,
    )

    s_entries: dict[str, Any] = {}
    s_ok = True
    for construct_id in construct_ids:
        summary = steerings[construct_id]
        effect = summary.get("target_direction_effect")
        manipulation = summary.get("manipulation_checks")
        controls_summary = summary.get("control_rows")
        control = control_reports[construct_id]
        entry_ok = bool(
            isinstance(effect, Mapping)
            and effect.get("directed_standardized_effect") is not None
            and isinstance(controls_summary, Mapping)
            and int(controls_summary.get("random", 0)) > 0
            and int(controls_summary.get("shuffled", 0)) > 0
            and isinstance(manipulation, Mapping)
            and int(manipulation.get("missing_or_unscorable_records", 0)) == 0
            and control.get("accessibility", {}).get("pass") is True
            and control.get("control_coverage", {}).get("pass") is True
            and manifest_status["steering"][construct_id]["complete"]
        )
        s_ok = s_ok and entry_ok
        estimate, lower, upper = _ci_fields(summary.get("uncertainty"))
        s_entries[construct_id] = {
            "status": "pass" if entry_ok else "fail",
            "directed_standardized_effect": effect.get("directed_standardized_effect") if isinstance(effect, Mapping) else None,
            "uncertainty": {"estimate": estimate, "lower": lower, "upper": upper},
            "control_rows": controls_summary,
            "missing_or_unscorable_records": manipulation.get("missing_or_unscorable_records") if isinstance(manipulation, Mapping) else None,
            "accessibility_pass": control.get("accessibility", {}).get("pass"),
            "collateral_status": control.get("collateral", {}).get("status"),
            "source_summary": str(steering_paths[construct_id]),
            "manifest": manifest_status["steering"][construct_id],
        }
    s = _stage_result(
        "pass" if s_ok else "fail",
        detail="Complete steering outputs include target, shuffled, and random controls, manipulation checks, and accessible independent-task outputs."
        if s_ok
        else "At least one steering output is incomplete or failed an accessibility/control check.",
        constructs=s_entries,
    )

    precision_path = _resolve(config["precision_report"], root=root)
    precision = _load_object(precision_path)
    precision_decision = precision.get("decision", {})
    precision_complete = precision.get("status") == "complete" and precision.get("planning_only") is True
    precision_gate = {
        "status": "pass" if precision_complete else "fail",
        "planning_only": precision.get("planning_only"),
        "report": str(precision_path),
        "report_sha256": file_sha256(precision_path),
        "release_decision": precision_decision.get("release_decision"),
        "recommended_minimum_item_count_across_constructs": precision_decision.get("recommended_minimum_item_count_across_constructs"),
        "current_plan_meets_rule_for_all_constructs": precision_decision.get("current_plan_meets_rule_for_all_constructs"),
    }

    collateral_missing = [
        construct_id
        for construct_id in construct_ids
        if control_reports[construct_id].get("collateral", {}).get("status") == "not_collected"
    ]
    collateral_unusable = [
        construct_id
        for construct_id in construct_ids
        if control_reports[construct_id].get("collateral", {}).get("status")
        not in {"collected", "not_collected"}
    ]
    stage_pass = bool(b["pass"] and r["pass"] and c["pass"] and s["pass"])
    blockers: list[dict[str, Any]] = []
    if not stage_pass:
        blockers.append({"id": "stage_failure", "detail": "At least one B/R/C/S stage failed its completed-output gate."})
    if collateral_missing:
        blockers.append({"id": "collateral_behavior_not_collected", "construct_ids": collateral_missing, "detail": "The existing Wave 1 steering inventory has no independent unrelated-behavior task; same-task compliance is not a collateral test."})
    if collateral_unusable:
        blockers.append({"id": "collateral_behavior_unusable", "construct_ids": collateral_unusable, "detail": "An independent collateral task was collected, but its valid correctness denominator is below the registered audit threshold."})
    if precision_decision.get("release_decision") != "continue_current_plan":
        blockers.append({"id": "precision_item_count", "detail": precision_decision.get("reason", "The precision simulation did not authorize the current item counts."), "recommended_minimum_item_count": precision_decision.get("recommended_minimum_item_count_across_constructs")})

    correspondence_rows: list[dict[str, Any]] = []
    construct_families = config.get("construct_families", {})
    construct_spec_paths = config.get("construct_specs", {})
    for construct_id in construct_ids:
        if isinstance(construct_spec_paths, Mapping) and construct_id in construct_spec_paths:
            spec_path = _resolve(construct_spec_paths[construct_id], root=root)
            spec = _load_object(spec_path)
            family = spec.get("family")
        else:
            family = construct_families.get(construct_id) if isinstance(construct_families, Mapping) else None
        b_item = behavior_constructs[construct_id]
        r_item = r_entries[construct_id]
        s_item = s_entries[construct_id]
        correspondence_rows.append(
            {
                "construct_id": construct_id,
                "family": family,
                "behavior_n": b_item.get("valid_primary_rows"),
                "behavior_mean": b_item.get("outcome_mean"),
                "behavior_sd": b_item.get("outcome_sample_sd"),
                "readout_margin": r_item.get("mean_standardized_margin"),
                "readout_pair_accuracy": r_item.get("pair_accuracy"),
                "steering_effect": s_item.get("directed_standardized_effect"),
                "steering_ci_lower": s_item["uncertainty"].get("lower"),
                "steering_ci_upper": s_item["uncertainty"].get("upper"),
                "causal_cross_condition_changed": c.get("cross_condition_output_changed_count"),
                "collateral_status": s_item.get("collateral_status"),
                "confirmatory": False,
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "wave1_non_confirmatory_measurement_gate",
        "status": "frozen",
        "frozen": True,
        "confirmatory": False,
        "gate_status": "pass_non_confirmatory_with_expansion_hold" if stage_pass else "blocked",
        "source_run": {
            "run_id": config.get("source_run_id"),
            "model": config.get("model"),
            "base_snapshot_commit": config.get("base_snapshot_commit"),
            "reclassification": "Historical source manifests may carry confirmatory=true, but this report treats the entire run as non-confirmatory engineering evidence because it predates the current Wave 1 gate contract.",
        },
        "storage": config.get("storage", {}),
        "stages": {"B": b, "R": r, "C": c, "S": s},
        "precision_gate": precision_gate,
        "collateral": {
            "status": (
                "not_collected"
                if collateral_missing
                else ("collected_with_failures" if collateral_unusable else "collected")
            ),
            "construct_ids_without_unrelated_task": collateral_missing,
            "construct_ids_with_unusable_outputs": collateral_unusable,
            "interpretation": "Output accessibility, compliance, response length, and target/control comparisons are reported as proxies only; no claim about unrelated behavior is made.",
        },
        "blockers": blockers,
        "expansion_decision": "hold_wave2_4_confirmatory_release" if blockers else "eligible_for_confirmatory_expansion_review",
        "next_authorized_step": "Run one-model Qwen Wave 1 engineering replication on the persistent volume; do not call Waves 2-4 confirmatory until the listed blockers are resolved and re-reviewed.",
        "manifest_status": manifest_status,
        "correspondence_input": correspondence_rows,
    }
    report["report_sha256"] = canonical_hash(report)
    return report


def write_artifacts(*, report: Mapping[str, Any], output_dir: Path, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "measurement_gate_report.json": report,
        "exclusions.json": {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "wave1_gate_exclusions",
            "confirmatory": False,
            "exclusions": report.get("blockers", []),
        },
        "continuation_manifest.json": {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "wave1_continuation_manifest",
            "status": "frozen",
            "confirmatory": False,
            "base_snapshot_commit": report["source_run"].get("base_snapshot_commit"),
            "measurement_gate_report": "measurement_gate_report.json",
            "precision_report": report["precision_gate"].get("report"),
            "raw_data_policy": "Raw activations, generations, causal observations, and checkpoints remain on the RunPod persistent volume; none are synchronized to the laptop repository.",
            "archive_status": report.get("storage", {}).get("archive_status", "not_configured"),
            "qwen_wave1_status": "pending_b300",
            "waves2_4_confirmatory_status": "held",
            "blockers": report.get("blockers", []),
        },
    }
    for filename, payload in paths.items():
        path = output_dir / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_dir / "correspondence_input.csv"
    rows = list(report.get("correspondence_input", []))
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {csv_path}")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Root used to resolve relative artifact paths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_object(config_path)
    report = build_gate(config=config, root=args.root.resolve())
    write_artifacts(report=report, output_dir=args.output_dir.resolve(), overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
