from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_train_only_br_preflight.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_only_br_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _audit(module) -> dict:
    constructs_by_wave = {
        1: [
            "evidence_diagnosticity",
            "persistence_continuation",
            "realization_account_closure",
            "source_reliability",
        ],
        2: ["reference_frame", "prior_weighting", "authority_deference", "exploration_exploitation"],
        3: ["ambiguity_orientation", "causal_interpretation", "consensus_conformity", "plan_replanning"],
        4: ["temporal_orientation", "epistemic_uncertainty", "reciprocity_obligation", "goal_shielding"],
    }
    entries = []
    for alias in ("qwen", "mistral"):
        for wave in range(1, 5):
            constructs = constructs_by_wave[wave]
            entries.append(
                {
                    "model_alias": alias,
                    "model_id": f"example/{alias}",
                    "revision": _digest(f"{alias}-revision"),
                    "wave": wave,
                    "construct_specs": [{"construct_id": construct_id} for construct_id in constructs],
                    "inventory": {
                        "sha256": _digest(f"{alias}-{wave}-csv"),
                        "manifest_sha256": _digest(f"{alias}-{wave}-manifest"),
                        "direction_train_content_sha256": _digest(f"{alias}-{wave}-train"),
                        "direction_train_record_count": 800,
                        "direction_train_pair_count": 400,
                        "direction_train_pair_counts_by_construct": {
                            construct_id: 100 for construct_id in constructs
                        },
                        "vector_split_counts_by_construct": {
                            construct_id: {"direction_train": 200} for construct_id in constructs
                        },
                        "selection_filter": {
                            "downstream_task_id_required": "empty",
                            "pair_conditions_from_construct_spec": True,
                            "prompt_role": "probe",
                            "split": "direction_train",
                        },
                    },
                }
            )
    return {
        "audit_type": module.AUDIT_TYPE,
        "audit_revision": 2,
        "canonical_index": {
            "status": "ready",
            "execution_allowed": False,
            "entry_count": 8,
            "artifact_ready_count": 32,
            "sha256": _digest("canonical-index"),
        },
        "entries": entries,
        "aggregate_by_model": {
            alias: {"direction_train_rows": 3200, "direction_train_pairs": 1600, "waves": [1, 2, 3, 4]}
            for alias in ("qwen", "mistral")
        },
    }


def _rows(module, *, omit: tuple[int, str, int] | None = None) -> list[dict]:
    rows = []
    constructs_by_wave = {
        1: [
            "evidence_diagnosticity",
            "persistence_continuation",
            "realization_account_closure",
            "source_reliability",
        ],
        2: ["reference_frame", "prior_weighting", "authority_deference", "exploration_exploitation"],
        3: ["ambiguity_orientation", "causal_interpretation", "consensus_conformity", "plan_replanning"],
        4: ["temporal_orientation", "epistemic_uncertainty", "reciprocity_obligation", "goal_shielding"],
    }
    for wave in range(1, 5):
        constructs = constructs_by_wave[wave]
        for construct_id in constructs:
            for pair_index in range(100):
                for pair_role in ("negative", "positive"):
                    if omit == (wave, construct_id, pair_index) and pair_role == "positive":
                        continue
                    rows.append(
                        {
                            "prompt_id": f"w{wave}-{construct_id}-p{pair_index}-{pair_role}",
                            "construct_id": construct_id,
                            "split": "direction_train",
                            "prompt_role": "probe",
                            "pair_id": f"w{wave}-{construct_id}-p{pair_index}",
                            "pair_role": pair_role,
                            "task_id": "",
                            "wave": wave,
                        }
                    )
    rows.extend(
        {
            "prompt_id": "downstream",
            "construct_id": constructs[0],
            "split": "steering_eval",
            "prompt_role": "steering",
            "pair_id": "downstream-pair",
            "task_id": "task-1",
            "wave": 1,
        }
        for _ in range(3)
    )
    return rows


def test_registered_audit_and_train_only_selection_are_exact() -> None:
    module = _load_module()
    audit = _audit(module)
    summary = module.validate_audit(audit, "qwen")
    assert summary["direction_train_rows"] == 3200
    assert summary["direction_train_pairs"] == 1600
    selected = module.select_train_rows(_rows(module))
    assert len(selected) == 3200
    assert {row["split"] for row in selected} == {"direction_train"}
    assert {row["prompt_role"] for row in selected} == {"probe"}
    assert {row["task_id"] for row in selected} == {""}


def test_selection_count_failure_is_fail_closed() -> None:
    module = _load_module()
    with pytest.raises(module.PreflightValidationError, match="selected rows=3199"):
        module.select_train_rows(_rows(module, omit=(2, "reference_frame", 99)))


def test_full_mode_is_refused_before_manifest_creation(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(module.PreflightValidationError, match="refuses full"):
        module.build_manifest(
            _audit(module),
            model_alias="qwen",
            audit_sha256="a" * 64,
            repo_sha="b" * 40,
            run_mode="full",
        )


def test_resume_requires_matching_audit_and_identity(tmp_path: Path) -> None:
    module = _load_module()
    manifest = module.build_manifest(
        _audit(module),
        model_alias="qwen",
        audit_sha256="a" * 64,
        repo_sha="b" * 40,
        audit_path=tmp_path / "audit.json",
    )
    first = module.write_manifest(tmp_path / "output", manifest)
    resumed = module.write_manifest(tmp_path / "output", manifest, resume=True)
    assert first["resume_count"] == 0
    assert resumed["resume_count"] == 1
    assert resumed["previous_manifest_sha256"]

    changed = dict(manifest)
    changed["resume_identity"] = "c" * 64
    with pytest.raises(module.PreflightValidationError, match="resume identity"):
        module.write_manifest(tmp_path / "output", changed, resume=True)


def test_hash_shape_failure_and_no_fake_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    audit = _audit(module)
    audit["canonical_index"]["sha256"] = "not-a-hash"
    with pytest.raises(module.PreflightValidationError, match="canonical_index.sha256"):
        module.validate_audit(audit, "mistral")

    manifest = module.build_manifest(
        _audit(module),
        model_alias="mistral",
        audit_sha256="a" * 64,
        repo_sha="b" * 40,
    )
    written = module.write_manifest(tmp_path / "output", manifest)
    assert written["semantic_runner"] == "not_executed"
    assert written["artifacts"] == []
    assert written["policy"]["directions_created"] is False
    assert list((tmp_path / "output").iterdir()) == [tmp_path / "output" / "manifest.json"]


def test_cli_rejects_full_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(_audit(module)), encoding="utf-8")
    result = module.main(
        [
            "--audit",
            str(audit_path),
            "--model-alias",
            "qwen",
            "--output-root",
            str(tmp_path / "output"),
            "--repo-sha",
            "b" * 40,
            "--run-mode",
            "full",
        ]
    )
    assert result == 2
    assert not (tmp_path / "output").exists()
