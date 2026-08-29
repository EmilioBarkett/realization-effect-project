#!/usr/bin/env python3
"""Assemble a small, checksummed record of Wave 1 engineering failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from construct_benchmark.manifests import canonical_hash, file_sha256  # noqa: E402


CONSTRUCT_IDS = (
    "realization_account_closure",
    "evidence_diagnosticity",
    "source_reliability",
    "persistence_continuation",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_sha256(path) if path.is_file() else None,
    }


def _behavior(path: Path) -> dict[str, Any]:
    payload = _load(path)
    behavior = payload.get("behavior", {})
    constructs = behavior.get("constructs", {}) if isinstance(behavior, dict) else {}
    variation = payload.get("variation_gate", {})
    compact: dict[str, Any] = {}
    for construct_id in CONSTRUCT_IDS:
        entry = dict(constructs.get(construct_id, {}))
        gate = dict(variation.get(construct_id, {})) if isinstance(variation, dict) else {}
        compact[construct_id] = {
            "total_rows": entry.get("total_rows"),
            "valid_parser_rows": entry.get("valid_parser_rows"),
            "valid_primary_rows": entry.get("valid_primary_rows"),
            "invalid_rows": entry.get("invalid_rows"),
            "primary_valid_rate": entry.get("primary_valid_rate"),
            "outcome_mean": entry.get("outcome_mean"),
            "outcome_sample_sd": entry.get("outcome_sample_sd"),
            "unique_outcome_count": entry.get("unique_outcome_count"),
            "variation_gate": {
                key: gate.get(key)
                for key in (
                    "pass",
                    "prompt_only_rows",
                    "valid_prompt_only_rows",
                    "invalid_prompt_only_rows",
                    "prompt_only_sample_sd",
                    "unique_prompt_only_outcomes",
                    "thresholds",
                    "failures",
                    "invalid_rows",
                )
                if key in gate
            },
        }
    return {
        "source": _ref(path),
        "manifest_complete": payload.get("manifest_complete"),
        "pass": payload.get("pass"),
        "total_rows": behavior.get("total_rows") if isinstance(behavior, dict) else None,
        "constructs": compact,
    }


def _collateral(path: Path) -> dict[str, Any]:
    payload = _load(path)
    behavior = payload.get("behavior", {})
    constructs = behavior.get("constructs", {}) if isinstance(behavior, dict) else {}
    return {
        "source": _ref(path),
        "manifest_complete": payload.get("manifest_complete"),
        "pass": payload.get("pass"),
        "constructs": {
            construct_id: {
                key: constructs.get(construct_id, {}).get(key)
                for key in (
                    "total_rows",
                    "valid_parser_rows",
                    "valid_primary_rows",
                    "invalid_rows",
                    "primary_valid_rate",
                    "outcome_mean",
                )
            }
            for construct_id in CONSTRUCT_IDS
        },
    }


def _steering(score_dir: Path, controls_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for construct_id in CONSTRUCT_IDS:
        score_path = score_dir / construct_id / "summary.json"
        control_path = controls_dir / f"{construct_id}.json"
        score = _load(score_path)
        control = _load(control_path)
        manipulation = score.get("manipulation_checks", {})
        result[construct_id] = {
            "score": {
                "source": _ref(score_path),
                "confirmatory": score.get("confirmatory"),
                "score_status": score.get("score_status", "complete"),
                "score_error": score.get("score_error"),
                "raw_record_count": score.get("raw_record_count"),
                "behavior_record_count": score.get("behavior_record_count"),
                "target_direction_effect": score.get("target_direction_effect"),
                "uncertainty": score.get("uncertainty"),
                "missing_injection_records": manipulation.get("missing_injection_records"),
                "missing_or_unscorable_records": manipulation.get("missing_or_unscorable_records"),
            },
            "control_audit": {
                "source": _ref(control_path),
                "row_count": control.get("row_count"),
                "accessibility": control.get("accessibility"),
                "control_coverage": control.get("control_coverage"),
                "collateral": control.get("collateral"),
                "source_manifest": control.get("source_manifest"),
            },
        }
    return result


def _blockers(model: str, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    behavior = bundle["behavior"]["constructs"]
    for construct_id, entry in behavior.items():
        gate = entry["variation_gate"]
        if gate.get("pass") is False:
            blockers.append(
                {
                    "code": f"{model}_prompt_only_behavior_variation",
                    "model": model,
                    "construct_id": construct_id,
                    "evidence": gate,
                }
            )
    for construct_id, entry in bundle["steering"].items():
        accessibility = entry["control_audit"].get("accessibility") or {}
        if accessibility.get("pass") is False:
            blockers.append(
                {
                    "code": f"{model}_steering_accessibility",
                    "model": model,
                    "construct_id": construct_id,
                    "evidence": accessibility,
                }
            )
        collateral = entry["control_audit"].get("collateral") or {}
        if collateral.get("status") not in {None, "collected"}:
            blockers.append(
                {
                    "code": f"{model}_collateral_quality",
                    "model": model,
                    "construct_id": construct_id,
                    "evidence": collateral,
                }
            )
        missing = entry["score"].get("missing_or_unscorable_records")
        if isinstance(missing, int) and missing > 0:
            blockers.append(
                {
                    "code": f"{model}_steering_unscorable_records",
                    "model": model,
                    "construct_id": construct_id,
                    "evidence": {"missing_or_unscorable_records": missing},
                }
            )
    return blockers


def build_report(
    *,
    qwen_behavior: Path,
    qwen_collateral: Path,
    qwen_score_dir: Path,
    qwen_controls_dir: Path,
    mistral_behavior: Path,
    mistral_collateral: Path,
    mistral_score_dir: Path,
    mistral_controls_dir: Path,
    qwen_causal_summary: Path,
    precision_report: Path,
    base_snapshot_commit: str,
    pod_id: str,
    volume_id: str,
    legacy_volume_id: str,
) -> dict[str, Any]:
    qwen = {
        "behavior": _behavior(qwen_behavior),
        "collateral": _collateral(qwen_collateral),
        "steering": _steering(qwen_score_dir, qwen_controls_dir),
    }
    mistral = {
        "behavior": _behavior(mistral_behavior),
        "collateral": _collateral(mistral_collateral),
        "steering": _steering(mistral_score_dir, mistral_controls_dir),
    }
    report: dict[str, Any] = {
        "schema_version": "0.1.0",
        "manifest_type": "wave1_failure_and_continuation_report",
        "confirmatory": False,
        "status": "complete",
        "base_snapshot_commit": base_snapshot_commit,
        "pod": {"pod_id": pod_id, "volume_id": volume_id, "legacy_volume_id": legacy_volume_id},
        "models": {"qwen": qwen, "mistral": mistral},
        "causal": {"source": _ref(qwen_causal_summary), "summary": _load(qwen_causal_summary)},
        "precision": {"source": _ref(precision_report), "report": _load(precision_report)},
        "hard_blockers": _blockers("qwen", qwen) + _blockers("mistral", mistral),
        "waves2_4_execution": "not_started_by_user_correction",
        "continuation_state": {
            "large_execution": "hold",
            "model_behavior_accessibility_preflight_required": True,
            "preflight_scope": "8_to_16_real_items_per_construct_and_model",
            "next_allowed_action": "run_and_pass_the_frozen_model_behavior_accessibility_preflight",
        },
        "provenance": {
            "legacy_raw_mistral_recovery": "read_only_archive_not_transferred",
            "legacy_derived_bundle_required": False,
            "new_account_used_for_new_storage_and_compute": True,
        },
    }
    report["report_sha256"] = canonical_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "qwen-behavior",
        "qwen-collateral",
        "qwen-score-dir",
        "qwen-controls-dir",
        "mistral-behavior",
        "mistral-collateral",
        "mistral-score-dir",
        "mistral-controls-dir",
        "qwen-causal-summary",
        "precision-report",
        "output",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--base-snapshot-commit", required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--legacy-volume-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        qwen_behavior=args.qwen_behavior,
        qwen_collateral=args.qwen_collateral,
        qwen_score_dir=args.qwen_score_dir,
        qwen_controls_dir=args.qwen_controls_dir,
        mistral_behavior=args.mistral_behavior,
        mistral_collateral=args.mistral_collateral,
        mistral_score_dir=args.mistral_score_dir,
        mistral_controls_dir=args.mistral_controls_dir,
        qwen_causal_summary=args.qwen_causal_summary,
        precision_report=args.precision_report,
        base_snapshot_commit=args.base_snapshot_commit,
        pod_id=args.pod_id,
        volume_id=args.volume_id,
        legacy_volume_id=args.legacy_volume_id,
    )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
