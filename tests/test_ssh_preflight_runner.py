from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/_ssh_preflight_runner.py"


def _load_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    alias: str = "qwen",
    model_id: str = "Qwen/Qwen3.8-27B",
):
    values = {
        "RSC_RUN_ID": "fixture-run",
        "RSC_MODEL_ALIAS": alias,
        "RSC_MODEL_ID": model_id,
        "RSC_MODEL_REVISION": "fixture-revision",
        "RSC_EXPECTED_REPO_SHA": "fixture-sha",
        "RSC_REPO_URL": "https://example.invalid/repo.git",
        "RSC_WORK_ROOT": str(tmp_path),
        "RSC_STORAGE_KIND": "ephemeral_container_disk",
        "RSC_EXPECTED_STORAGE_GB": "160",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    module_name = "ssh_preflight_runner_fixture"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_output_fixture(
    path: Path,
    rows: list[dict[str, object]],
    *,
    model: dict[str, object],
    inventory_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    record_ids = [str(row["record_id"]) for row in rows]
    manifest = {
        "manifest_type": "construct_behavior_output",
        "complete": True,
        "model": model,
        "prompt_inventory_sha256": inventory_sha256,
        "expected_record_count": len(record_ids),
        "completed_record_count": len(record_ids),
        "expected_record_ids": record_ids,
        "raw_generations_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _wave1_gate_rows(selection: dict[str, object], specs: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = dict(selection["model"])
    behavior_rows: list[dict[str, object]] = []
    collateral_rows: list[dict[str, object]] = []
    selected = selection["selected"]
    from construct_benchmark.prompts import load_prompt_records

    inventory = ROOT / "results/benchmark/prompt_inventories/wave1_preflight_v4_luna_v2/combined.csv"
    metadata_by_prompt = {
        record.prompt_id: dict(record.metadata.get("task_metadata") or {})
        for record in load_prompt_records(inventory)
    }
    for construct_id in selection["construct_ids"]:
        spec = specs[construct_id]
        behavior_task = spec.independent_behavior_task
        collateral_task = spec.collateral_behavior_task
        for index, prompt_id in enumerate(selected[construct_id]["behavior_eval"]["prompt_ids"]):
            metadata = dict(metadata_by_prompt[prompt_id])
            output = "50" if construct_id == "evidence_diagnosticity" else str(20 + index * 3)
            behavior_rows.append(
                {
                    "record_id": f"{prompt_id}__prompt_only",
                    "prompt_id": prompt_id,
                    "construct_id": construct_id,
                    "split": "behavior_eval",
                    "model": model,
                    "parser_id": spec.parsing_rules["parser_id"],
                    "task_id": behavior_task["task_id"],
                    "task_metadata": metadata,
                    "output_text": output,
                }
            )
        for index, prompt_id in enumerate(selected[construct_id]["collateral_eval"]["prompt_ids"]):
            metadata = dict(metadata_by_prompt[prompt_id])
            correct = int(metadata["correct_option"])
            # Deliberately make only source reliability's factual control
            # wrong; the other constructs remain useful control fixtures.
            answer = (3 - correct) if construct_id == "source_reliability" else correct
            collateral_rows.append(
                {
                    "record_id": f"{prompt_id}__prompt_only",
                    "prompt_id": prompt_id,
                    "construct_id": construct_id,
                    "split": "collateral_eval",
                    "model": model,
                    "parser_id": "single_integer_choice_1_or_2_v1",
                    "task_id": collateral_task["task_id"],
                    "task_metadata": metadata,
                    "output_text": str(answer),
                }
            )
    return behavior_rows, collateral_rows


def _probe(
    module,
    interpreter: str,
    *,
    transformers: str | None,
    accelerate: str | None = None,
    returncode: int,
) -> dict[str, object]:
    errors = []
    if transformers is None:
        errors.append("transformers: ModuleNotFoundError: No module named 'transformers'")
    if accelerate is None:
        errors.append("accelerate: ModuleNotFoundError: No module named 'accelerate'")
    identity = {
        "python": "3.12.1 (fixture)",
        "python_executable": interpreter,
        "python_version": "3.12.1",
        "torch": "2.8.0+cu128",
        "transformers": transformers,
        "accelerate": accelerate,
        "cuda_available": True,
        "cuda_version": "12.8",
        "devices": ["NVIDIA A100-SXM4-80GB"],
        "errors": errors,
        "ok": returncode == 0,
    }
    return {
        "interpreter": interpreter,
        "command": [interpreter, "-c", module.RUNTIME_PROBE_CODE],
        "returncode": returncode,
        "stdout": json.dumps(identity, sort_keys=True) + "\n",
        "stderr": "fixture stderr\n" if returncode else "",
        "identity": identity,
        "torch_compatible": True,
        "transformers_compatible": transformers == module.EXPECTED_TRANSFORMERS_VERSION,
        "accelerate_compatible": accelerate == module.EXPECTED_ACCELERATE_VERSION,
        "python_compatible": True,
    }


def test_runtime_candidates_prefer_path_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    path_python = tmp_path / "python"
    path_python.write_text("", encoding="utf-8")
    path_python.chmod(0o755)
    path_python3 = tmp_path / "python3"
    path_python3.write_text("", encoding="utf-8")
    path_python3.chmod(0o755)

    monkeypatch.setattr(module.shutil, "which", lambda name: str(path_python if name == "python" else path_python3))
    candidates = module._candidate_interpreters()

    assert candidates[:2] == [str(path_python), str(path_python3)]


def test_runtime_resolution_repairs_missing_transformers_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    interpreter = str(tmp_path / "python")
    runtime_interpreter = str(module.RUNTIME_VENV / "bin" / "python")
    failed = _probe(module, interpreter, transformers=None, returncode=1)
    venv_failed = _probe(module, runtime_interpreter, transformers=None, returncode=1)
    repaired = _probe(
        module,
        runtime_interpreter,
        transformers=module.EXPECTED_TRANSFORMERS_VERSION,
        accelerate=module.EXPECTED_ACCELERATE_VERSION,
        returncode=0,
    )
    probes = iter([failed, venv_failed, repaired])
    installs: list[list[str]] = []
    creates: list[str] = []

    monkeypatch.setattr(module, "_candidate_interpreters", lambda: [interpreter])
    monkeypatch.setattr(module, "_probe_runtime", lambda _interpreter: next(probes))
    venv_lookups = iter([None, runtime_interpreter])
    monkeypatch.setattr(module, "_runtime_venv_interpreter", lambda: next(venv_lookups))

    def fake_create(base_interpreter: str) -> dict[str, object]:
        creates.append(base_interpreter)
        return {
            "base_interpreter": base_interpreter,
            "runtime_venv": str(module.RUNTIME_VENV),
            "command": [base_interpreter, "-m", "venv", "--system-site-packages", str(module.RUNTIME_VENV)],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(module, "_create_runtime_venv", fake_create)

    def fake_install(_interpreter: str) -> dict[str, object]:
        installs.append([_interpreter, "transformers==" + module.EXPECTED_TRANSFORMERS_VERSION])
        return {"interpreter": _interpreter, "command": installs[-1], "returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(module, "_install_transformers", fake_install)
    selected, probe = module._runtime_resolution()

    assert selected == runtime_interpreter
    assert probe is repaired
    assert creates == [interpreter]
    assert installs == [[runtime_interpreter, "transformers==" + module.EXPECTED_TRANSFORMERS_VERSION]]
    resolution = json.loads((module.STATE / "runtime_resolution.json").read_text(encoding="utf-8"))
    assert resolution["attempts"][0]["returncode"] == 1
    assert resolution["attempts"][0]["stderr"] == "fixture stderr\n"
    assert resolution["attempts"][1]["kind"] == "create_runtime_venv"
    assert resolution["attempts"][2]["kind"] == "probe_runtime_venv"
    assert resolution["attempts"][3]["kind"] == "install_transformers"


def test_runtime_probe_requires_exact_accelerate_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    interpreter = str(tmp_path / "python")

    missing = _probe(
        module,
        interpreter,
        transformers=module.EXPECTED_TRANSFORMERS_VERSION,
        accelerate=None,
        returncode=1,
    )
    wrong = _probe(
        module,
        interpreter,
        transformers=module.EXPECTED_TRANSFORMERS_VERSION,
        accelerate="1.10.0",
        returncode=0,
    )
    installed = _probe(
        module,
        interpreter,
        transformers=module.EXPECTED_TRANSFORMERS_VERSION,
        accelerate=module.EXPECTED_ACCELERATE_VERSION,
        returncode=0,
    )

    assert missing["accelerate_compatible"] is False
    assert module._probe_is_ready(missing) is False
    assert wrong["accelerate_compatible"] is False
    assert module._probe_is_ready(wrong) is False
    assert installed["accelerate_compatible"] is True
    assert module._probe_is_ready(installed) is True


def test_installed_accelerate_allows_runtime_without_model_loader_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    interpreter = str(module.RUNTIME_VENV / "bin" / "python")
    installed = _probe(
        module,
        interpreter,
        transformers=module.EXPECTED_TRANSFORMERS_VERSION,
        accelerate=module.EXPECTED_ACCELERATE_VERSION,
        returncode=0,
    )
    module.RUNTIME_PROBE = installed

    # The runner's model-construction boundary is reached only after this
    # dependency gate. A Mistral load therefore cannot emit the misleading
    # multimodal fallback cascade caused by a missing Accelerate import.
    module._require_model_runtime()


def test_missing_accelerate_blocks_mistral_model_loader_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(
        monkeypatch,
        tmp_path,
        alias="mistral",
        model_id="mistralai/Mistral-Small-24B-Instruct-2501",
    )
    interpreter = str(module.RUNTIME_VENV / "bin" / "python")
    missing = _probe(
        module,
        interpreter,
        transformers=module.EXPECTED_TRANSFORMERS_VERSION,
        accelerate=None,
        returncode=1,
    )
    module.RUNTIME_PROBE = missing

    with pytest.raises(module.RunnerFailure, match="model loading prerequisites"):
        module._require_model_runtime()


def test_install_transformers_pins_accelerate_in_isolated_venv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    interpreter = str(module.RUNTIME_VENV / "bin" / "python")
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return module.subprocess.CompletedProcess(command, 0, "pip stdout\n", "pip stderr\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module._install_transformers(interpreter)

    assert result["returncode"] == 0
    assert result["stdout"] == "pip stdout\n"
    assert result["stderr"] == "pip stderr\n"
    assert calls[0]["command"][-2:] == [
        "transformers==" + module.EXPECTED_TRANSFORMERS_VERSION,
        "accelerate==" + module.EXPECTED_ACCELERATE_VERSION,
    ]
    environment = calls[0]["env"]
    assert isinstance(environment, dict)
    assert environment["PIP_NO_INPUT"] == "1"
    assert environment["PIP_ROOT_USER_ACTION"] == "ignore"


def test_pep668_base_never_receives_package_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    base_interpreter = str(tmp_path / "python")
    runtime_interpreter = str(module.RUNTIME_VENV / "bin" / "python")
    base_probe = _probe(module, base_interpreter, transformers=None, returncode=1)
    base_probe["stderr"] = "error: externally-managed-environment\n"
    venv_probe = _probe(module, runtime_interpreter, transformers=None, returncode=1)
    repaired_probe = _probe(
        module,
        runtime_interpreter,
        transformers=module.EXPECTED_TRANSFORMERS_VERSION,
        accelerate=module.EXPECTED_ACCELERATE_VERSION,
        returncode=0,
    )
    probes = iter([base_probe, venv_probe, repaired_probe])
    installs: list[str] = []

    monkeypatch.setattr(module, "_candidate_interpreters", lambda: [base_interpreter])
    monkeypatch.setattr(module, "_probe_runtime", lambda _interpreter: next(probes))
    venv_lookups = iter([None, runtime_interpreter])
    monkeypatch.setattr(module, "_runtime_venv_interpreter", lambda: next(venv_lookups))
    monkeypatch.setattr(
        module,
        "_create_runtime_venv",
        lambda base: {
            "base_interpreter": base,
            "runtime_venv": str(module.RUNTIME_VENV),
            "command": [base, "-m", "venv", "--system-site-packages", str(module.RUNTIME_VENV)],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        module,
        "_install_transformers",
        lambda interpreter: installs.append(interpreter)
        or {
            "interpreter": interpreter,
            "command": [interpreter, "-m", "pip", "install", "transformers==5.16.1"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    selected, _ = module._runtime_resolution()

    assert selected == runtime_interpreter
    assert installs == [runtime_interpreter]
    resolution = json.loads((module.STATE / "runtime_resolution.json").read_text(encoding="utf-8"))
    assert resolution["attempts"][0]["stderr"] == "error: externally-managed-environment\n"
    assert resolution["attempts"][1]["kind"] == "create_runtime_venv"
    assert resolution["attempts"][2]["kind"] == "probe_runtime_venv"


def test_reexec_candidates_prefer_existing_runtime_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    runtime_interpreter = module.RUNTIME_VENV / "bin" / "python"
    runtime_interpreter.parent.mkdir(parents=True)
    runtime_interpreter.write_text("", encoding="utf-8")
    runtime_interpreter.chmod(0o755)
    path_python = tmp_path / "path-python"
    path_python.write_text("", encoding="utf-8")
    path_python.chmod(0o755)
    monkeypatch.setenv(module.RUNTIME_REEXEC_ENV, "1")
    monkeypatch.setattr(module.shutil, "which", lambda _name: str(path_python))

    candidates = module._candidate_interpreters()

    assert candidates[0] == str(runtime_interpreter)
    assert candidates.count(str(runtime_interpreter)) == 1


def test_install_transformers_rejects_non_venv_interpreter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_runner(monkeypatch, tmp_path)

    with pytest.raises(module.RunnerFailure, match="outside the isolated runtime venv"):
        module._install_transformers(str(tmp_path / "python"))


def test_create_runtime_venv_isolated_and_preserves_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return module.subprocess.CompletedProcess(command, 0, "venv stdout\n", "venv stderr\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module._create_runtime_venv("/usr/local/bin/python")

    assert result["returncode"] == 0
    assert result["stdout"] == "venv stdout\n"
    assert result["stderr"] == "venv stderr\n"
    assert calls[0]["command"] == [
        "/usr/local/bin/python",
        "-m",
        "venv",
        "--system-site-packages",
        str(module.RUNTIME_VENV),
    ]


def test_ensure_runtime_reexec_propagates_selected_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    monkeypatch.delenv(module.RUNTIME_REEXEC_ENV, raising=False)
    selected = str(module.RUNTIME_VENV / "bin" / "python")
    probe = _probe(
        module,
        selected,
        transformers=module.EXPECTED_TRANSFORMERS_VERSION,
        accelerate=module.EXPECTED_ACCELERATE_VERSION,
        returncode=0,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "_runtime_resolution", lambda: (selected, probe))

    def fake_execvpe(interpreter: str, argv: list[str], env: dict[str, str]) -> None:
        captured.update({"interpreter": interpreter, "argv": argv, "env": env})

    monkeypatch.setattr(module.os, "execvpe", fake_execvpe)

    with pytest.raises(AssertionError, match="execvpe returned unexpectedly"):
        module._ensure_runtime()

    child_env = captured["env"]
    assert captured["interpreter"] == selected
    assert isinstance(child_env, dict)
    assert child_env[module.RUNTIME_REEXEC_ENV] == "1"
    assert child_env[module.RUNTIME_PYTHON_ENV] == selected
    assert child_env["VIRTUAL_ENV"] == str(module.RUNTIME_VENV)
    assert child_env["PATH"].split(":", 1)[0] == str(module.RUNTIME_VENV / "bin")
    assert module.RUNTIME_PYTHON == selected


def test_baseline_gate_preserves_per_construct_failure_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    module.CHECKOUT = ROOT
    module.STATE = tmp_path / "state"
    module.STATE.mkdir()
    selection_path = ROOT / "results/benchmark/model_preflight_v4/wave1_preflight_v4_luna_v2_normalized_v1_qwen_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    spec_paths = [
        ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v4.json",
        ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v5.json",
        ROOT / "configs/construct_benchmark/constructs/source_reliability_v3.json",
        ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v4.json",
    ]
    from construct_benchmark.config import load_construct_specs

    specs = load_construct_specs(spec_paths)
    behavior_rows, collateral_rows = _wave1_gate_rows(selection, specs)
    behavior_path = tmp_path / "behavior.jsonl"
    collateral_path = tmp_path / "collateral.jsonl"
    _write_output_fixture(
        behavior_path,
        behavior_rows,
        model=selection["model"],
        inventory_sha256=selection["source_inventory_sha256"],
    )
    _write_output_fixture(
        collateral_path,
        collateral_rows,
        model=selection["model"],
        inventory_sha256=selection["source_inventory_sha256"],
    )
    report_path = tmp_path / "baseline_collateral_gate.json"

    report = module.baseline_gate(
        ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_qwen_model_preflight_repaired_v4.json",
        selection_path,
        ROOT / "configs/construct_benchmark/gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json",
        spec_paths,
        behavior_path,
        collateral_path,
        report_path,
    )

    assert report["pass"] is False
    assert report_path.is_file()
    matrix = {
        (entry["construct_id"], entry["stage"]): entry
        for entry in report["failure_matrix"]
        if entry.get("construct_id") is not None
    }
    assert {
        stage
        for construct_id, stage in matrix
        if construct_id in selection["construct_ids"]
    } == {
        "behavior_coverage",
        "behavior_parser",
        "behavior_variation",
        "collateral_coverage",
        "collateral_parser",
        "collateral_correctness",
    }
    assert matrix[("evidence_diagnosticity", "behavior_variation")]["pass"] is False
    assert matrix[("source_reliability", "collateral_correctness")]["pass"] is False
    assert matrix[("realization_account_closure", "behavior_parser")]["pass"] is True
    assert matrix[("evidence_diagnosticity", "behavior_parser")]["pass"] is True
    assert matrix[("persistence_continuation", "behavior_parser")]["pass"] is True
    assert report["constructs"]["evidence_diagnosticity"]["behavior"]["selected_item_count"] == 16
    assert report["constructs"]["source_reliability"]["collateral"]["selected_item_count"] == 16

    module._stream_gate_report(report)
    streamed = capsys.readouterr().out
    assert streamed.startswith("[RSC_GATE_REPORT] {")
    assert '"diagnostic_version":"wave1_baseline_collateral_gate_v1"' in streamed


def test_discover_plans_resolves_inherited_wrapper_specs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Discovery accepts the registered schema from the dedicated staging root."""

    module = _load_runner(monkeypatch, tmp_path)
    module.CHECKOUT = ROOT
    plan_root = module.RUN_ROOT / "steering_plans"
    plan_root.mkdir(parents=True)
    monkeypatch.delenv(module.STEERING_PLAN_ROOT_ENV, raising=False)
    config = ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_qwen_model_preflight_repaired_v4.json"
    spec_paths = [
        ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v4.json",
        ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v5.json",
        ROOT / "configs/construct_benchmark/constructs/source_reliability_v3.json",
        ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v4.json",
    ]
    from construct_benchmark.config import load_construct_specs, load_run_config
    from construct_benchmark.manifests import canonical_hash

    loaded_specs = load_construct_specs(spec_paths)
    config_hash = canonical_hash(load_run_config(config).to_mapping())
    for construct_id in loaded_specs:
        plan = {
            # This is the actual plan_construct_steering schema.  In
            # particular, it has no top-level ``prefill_only`` key; the
            # prefill-only invariant is expressed by these timing fields.
            "schema_version": "0.1.0",
            "plan_type": "construct_steering_conditions",
            "run_id": "registered-source-run",
            "mode": "full",
            "purpose": "model_behavior_accessibility",
            "confirmatory": False,
            "construct_id": construct_id,
            "model": {
                "model_id": module.MODEL_ID,
                "revision": module.REVISION,
                "tokenizer_id": module.MODEL_ID,
            },
            "candidate_layers": [16, 32, 48],
            "layer": 16,
            "tracking_layers": [16],
            "tracking_directions": {
                "16": {
                    "layer": 16,
                    "direction_id": f"{construct_id}__injected_direction__layer_16",
                    "path": str(plan_root / "directions" / f"{construct_id}_target.npy"),
                    "source": "injection_direction_train_only",
                    "role": "injection_immediate",
                    "source_split": "direction_train",
                    "direction_sha256": "fixture-direction-sha",
                    "calibration": {"projection_scale": 1.0},
                }
            },
            "layer_selection": {"selection": "validation_max_margin"},
            "activation_site": "resid_post",
            "position_mode": "last",
            "intervention_timing": "prefill_only",
            "fixed_window": None,
            "calibration": {"projection_scale": 1.0},
            "direction_storage_dtype": "float16",
            "direction_paths": {
                "target": str(plan_root / "directions" / f"{construct_id}_target.npy"),
                "shuffled": str(plan_root / "directions" / f"{construct_id}_shuffled.npy"),
                "random": [
                    str(plan_root / "directions" / f"{construct_id}_random_{index:02d}.npy")
                    for index in range(3)
                ],
            },
            "condition_count": 1,
            "conditions": [
                {
                    "condition_id": f"{construct_id}__target__dose_0",
                    "prompt_id": f"{construct_id}__steering_000",
                    "direction_kind": "target",
                    "direction_index": 0,
                    "dose": 0.0,
                    "physical_scale": 0.0,
                    "intervention_timing": "prefill_only",
                    "order": 0,
                    "seed": 1729,
                }
            ],
            "provenance": {"run_config_hash": config_hash},
        }
        (plan_root / f"{construct_id}.json").write_text(
            json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8"
        )

    report = module.discover_plans(config, spec_paths, tmp_path / "plan_discovery.json")

    assert report["pass"] is True
    assert report["search_root"] == str(plan_root)
    assert report["search_root_source"] == "default"
    assert report["run_config_hash_method"].startswith("canonical_hash(load_run_config")
    assert report["rejected_by_reason"] == {}
    assert report["expected_construct_ids"] == list(loaded_specs)
    assert set(report["selected"]) == set(loaded_specs)


def test_discover_plans_rejects_explicit_non_prefill_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit false timing marker cannot bypass the prefill-only gate."""

    module = _load_runner(monkeypatch, tmp_path)
    module.CHECKOUT = ROOT
    plan_root = tmp_path / "registered_steering_plans"
    plan_root.mkdir()
    monkeypatch.setenv(module.STEERING_PLAN_ROOT_ENV, str(plan_root))
    config = ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_qwen_model_preflight_repaired_v4.json"
    spec_paths = [
        ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v4.json",
        ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v5.json",
        ROOT / "configs/construct_benchmark/constructs/source_reliability_v3.json",
        ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v4.json",
    ]
    from construct_benchmark.config import load_construct_specs, load_run_config
    from construct_benchmark.manifests import canonical_hash

    construct_id = next(iter(load_construct_specs(spec_paths)))
    config_hash = canonical_hash(load_run_config(config).to_mapping())
    plan = {
        "plan_type": "construct_steering_conditions",
        "model": {"model_id": module.MODEL_ID, "revision": module.REVISION},
        "prefill_only": False,
        "position_mode": "last",
        "intervention_timing": "prefill_only",
        "provenance": {"run_config_hash": config_hash},
        "construct_id": construct_id,
    }
    (plan_root / "not_prefill.json").write_text(
        json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(module.RunnerFailure, match="registered steering plans are missing"):
        module.discover_plans(config, spec_paths, tmp_path / "plan_discovery.json")

    report = json.loads((tmp_path / "plan_discovery.json").read_text(encoding="utf-8"))
    assert report["pass"] is False
    assert report["rejected_by_reason"] == {"not_prefill_only": 1}


@pytest.mark.parametrize(
    ("task_id", "primary_outcome"),
    [
        ("realization_risk_allocation_v4", "risky_allocation"),
        ("diagnostic_test_allocation_v5", "high_information_test_allocation"),
        ("goal_renewal_allocation_v4", "established_goal_allocation"),
    ],
)
def test_repaired_wave_allocation_aliases_expose_registered_primary_outcome(
    task_id: str, primary_outcome: str
) -> None:
    from construct_benchmark.behavior import parse_behavior_output

    parsed = parse_behavior_output(
        "42",
        parser_id="single_integer_allocation_0_to_100_v1",
        task_id=task_id,
    )
    assert parsed.valid
    assert parsed.values[primary_outcome] == pytest.approx(42.0)


def test_wave1_release_alias_gap_reproduces_zero_primary_rows_and_repair() -> None:
    """Keep the observed release failure as a deterministic scoring fixture.

    The failed Qwen run returned syntactically valid integer responses for all
    16 items in each affected construct, but the checkout at the released SHA
    did not register the versioned task IDs.  That made the strict parser
    valid while leaving the spec's named primary outcome absent.  This fixture
    exercises that exact distinction, including the unaffected source
    reliability task, and then verifies the alias repair without changing any
    gate threshold or prompt.
    """

    from construct_benchmark.behavior import _ALLOCATION_TASK_OUTCOMES
    from construct_benchmark.behavior_baseline import score_behavior_rows
    from construct_benchmark.config import load_construct_specs

    spec_paths = [
        ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v4.json",
        ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v5.json",
        ROOT / "configs/construct_benchmark/constructs/source_reliability_v3.json",
        ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v4.json",
    ]
    specs = load_construct_specs(spec_paths)
    expected_task_ids = {
        "realization_account_closure": "realization_risk_allocation_v4",
        "evidence_diagnosticity": "diagnostic_test_allocation_v5",
        "source_reliability": "source_evidence_allocation_v2",
        "persistence_continuation": "goal_renewal_allocation_v4",
    }
    metadata = {
        "realization_account_closure": {"outcome_valence": "gain"},
        "evidence_diagnosticity": {"high_information_option": "option_a"},
        "source_reliability": {},
        "persistence_continuation": {},
    }
    rows = [
        {
            "record_id": f"{construct_id}__{index:02d}__prompt_only",
            "prompt_id": f"{construct_id}__{index:02d}",
            "construct_id": construct_id,
            "split": "behavior_eval",
            "prompt_role": "behavior",
            "parser_id": "single_integer_allocation_0_to_100_v1",
            "task_id": expected_task_ids[construct_id],
            "task_metadata": metadata[construct_id],
            "output_text": str(20 + (index % 8) * 10),
        }
        for construct_id in expected_task_ids
        for index in range(16)
    ]

    with pytest.MonkeyPatch.context() as patch:
        for task_id in (
            "realization_risk_allocation_v4",
            "diagnostic_test_allocation_v5",
            "goal_renewal_allocation_v4",
        ):
            patch.delitem(_ALLOCATION_TASK_OUTCOMES, task_id)
        _, failed_summary = score_behavior_rows(rows, specs)

    failed_constructs = failed_summary["constructs"]
    for construct_id in (
        "realization_account_closure",
        "evidence_diagnosticity",
        "persistence_continuation",
    ):
        assert failed_constructs[construct_id]["valid_parser_rows"] == 16
        assert failed_constructs[construct_id]["valid_primary_rows"] == 0
        assert failed_constructs[construct_id]["primary_valid_rate"] == 0.0
    assert failed_constructs["source_reliability"]["valid_parser_rows"] == 16
    assert failed_constructs["source_reliability"]["valid_primary_rows"] == 16

    _, repaired_summary = score_behavior_rows(rows, specs)
    assert all(
        repaired_summary["constructs"][construct_id]["valid_primary_rows"] == 16
        for construct_id in expected_task_ids
    )


def test_baseline_gate_writes_report_for_missing_output_and_upstream_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    module.CHECKOUT = ROOT
    module.STATE = tmp_path / "state"
    selection_path = ROOT / "results/benchmark/model_preflight_v4/wave1_preflight_v4_luna_v2_normalized_v1_qwen_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    spec_paths = [
        ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v4.json",
        ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v5.json",
        ROOT / "configs/construct_benchmark/constructs/source_reliability_v3.json",
        ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v4.json",
    ]
    report_path = tmp_path / "failed_gate.json"
    report = module.baseline_gate(
        ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_qwen_model_preflight_repaired_v4.json",
        selection_path,
        ROOT / "configs/construct_benchmark/gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json",
        spec_paths,
        tmp_path / "missing_behavior.jsonl",
        tmp_path / "missing_collateral.jsonl",
        report_path,
        upstream_error="RuntimeError: model generation interrupted",
    )

    assert report_path.is_file()
    assert report["pass"] is False
    assert report["upstream_error"] == "RuntimeError: model generation interrupted"
    assert any(item.get("code") == "generation_error" for item in report["failure_matrix"])
    assert report["inputs"]["behavior"]["validation"]["pass"] is False
    assert set(report["constructs"]) == set(selection["construct_ids"])


def test_preserve_gate_artifacts_copies_jsonl_and_manifests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    module.STATE = tmp_path / "state"
    source = tmp_path / "wave1" / "behavior.jsonl"
    source.parent.mkdir()
    source.write_text('{"record_id":"r1"}\n', encoding="utf-8")
    manifest = source.with_suffix(source.suffix + ".manifest.json")
    manifest.write_text('{"complete":false}\n', encoding="utf-8")

    result = module._preserve_gate_artifacts(tmp_path / "wave1", [source, manifest], wave=1)

    destination = Path(result["directory"])
    assert (destination / source.name).read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert (destination / manifest.name).read_text(encoding="utf-8") == manifest.read_text(encoding="utf-8")
    assert all(entry["destination_sha256"] for entry in result["files"])


def test_run_wave_streams_and_preserves_gate_failure_before_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The paid-runner failure path must leave a compact report and raw inputs."""

    module = _load_runner(monkeypatch, tmp_path)
    module.CHECKOUT = ROOT
    module.STATE = tmp_path / "state"
    module.STATE.mkdir()
    config = ROOT / "configs/construct_benchmark/run_configs/wave1_four_construct_qwen_model_preflight_repaired_v4.json"
    inventory = ROOT / "results/benchmark/prompt_inventories/wave1_preflight_v4_luna_v2/combined.csv"
    selection = ROOT / "results/benchmark/model_preflight_v4/wave1_preflight_v4_luna_v2_normalized_v1_qwen_selection.json"
    gate = ROOT / "configs/construct_benchmark/gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json"
    specs = [
        ROOT / "configs/construct_benchmark/constructs/realization_account_closure_v4.json",
        ROOT / "configs/construct_benchmark/constructs/evidence_diagnosticity_v5.json",
        ROOT / "configs/construct_benchmark/constructs/source_reliability_v3.json",
        ROOT / "configs/construct_benchmark/constructs/persistence_continuation_v4.json",
    ]
    selection_data = json.loads(selection.read_text(encoding="utf-8"))
    from construct_benchmark.config import load_construct_specs

    behavior_rows, collateral_rows = _wave1_gate_rows(
        selection_data,
        load_construct_specs(specs),
    )

    def fake_run(command, output, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    def fake_execute_prompt_only_behavior(**kwargs):
        rows = behavior_rows if kwargs["prompt_split"] == "behavior_eval" else collateral_rows
        _write_output_fixture(
            Path(kwargs["output"]),
            rows,
            model=selection_data["model"],
            inventory_sha256=selection_data["source_inventory_sha256"],
        )
        return {"completed_records": len(rows)}

    fake_behavior_module = types.ModuleType("run_prompt_only_behavior")
    fake_behavior_module.ResidualSteeringGenerator = object
    fake_behavior_module.execute_prompt_only_behavior = fake_execute_prompt_only_behavior
    monkeypatch.setitem(sys.modules, "run_prompt_only_behavior", fake_behavior_module)
    monkeypatch.setattr(module, "_require_model_runtime", lambda: None)
    monkeypatch.setattr(module, "run", fake_run)

    wave_root = tmp_path / "wave1"
    with pytest.raises(module.RunnerFailure, match="diagnostics="):
        module.run_wave(1, config, inventory, selection, gate, specs, wave_root)

    report_path = wave_root / "baseline_collateral_gate.json"
    state_dir = module.STATE / "wave1" / "baseline_collateral"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pass"] is False
    assert report["preserved_artifacts"]["directory"] == str(state_dir)
    assert (state_dir / report_path.name).is_file()
    assert (state_dir / "behavior.jsonl").read_text(encoding="utf-8") == (wave_root / "behavior.jsonl").read_text(encoding="utf-8")
    assert (state_dir / "collateral.jsonl.manifest.json").is_file()
    streamed = capsys.readouterr().out
    assert "[RSC_GATE_REPORT] {" in streamed
    assert "behavior_variation" in streamed


def test_remote_launcher_payload_is_syntax_checked_before_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    launcher = tmp_path / "remote" / "remote_br_launcher.py"

    metadata = module.validate_remote_br_launcher_payload()
    staged = module.write_remote_launcher(launcher)

    assert metadata["syntax_checked"] is True
    assert metadata["sha256"] == module.REMOTE_BR_LAUNCHER_PAYLOAD_SHA256
    assert staged["sha256"] == metadata["sha256"]
    assert launcher.read_text(encoding="utf-8") == module.REMOTE_BR_LAUNCHER_PAYLOAD
    assert launcher.stat().st_mode & 0o111

    invalid_launcher = tmp_path / "remote" / "invalid_launcher.py"
    with pytest.raises(module.RunnerFailure, match="syntax check"):
        module.write_remote_launcher(invalid_launcher, payload=module.REMOTE_BR_LAUNCHER_PAYLOAD + "\nif (")
    assert not invalid_launcher.exists()


def test_remote_launcher_rejects_existing_and_dangling_symlinks_before_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    real_launcher = tmp_path / "remote" / "real_launcher.py"
    real_launcher.parent.mkdir(parents=True)
    real_launcher.write_text("sentinel\n", encoding="utf-8")

    linked_launcher = tmp_path / "remote" / "linked_launcher.py"
    linked_launcher.symlink_to(real_launcher)
    with pytest.raises(module.RunnerFailure, match="symlink"):
        module.write_remote_launcher(linked_launcher)
    assert real_launcher.read_text(encoding="utf-8") == "sentinel\n"

    dangling_launcher = tmp_path / "remote" / "dangling_launcher.py"
    dangling_launcher.symlink_to(tmp_path / "remote" / "missing_launcher.py")
    with pytest.raises(module.RunnerFailure, match="symlink"):
        module.write_remote_launcher(dangling_launcher)

    dangling_manifest = tmp_path / "remote" / "dangling_manifest.json"
    dangling_manifest.symlink_to(tmp_path / "remote" / "missing_manifest.json")
    with pytest.raises(module.RunnerFailure, match="symlink"):
        module.prepare_remote_br_launch_handoff(
            [sys.executable, "-c", "pass"],
            manifest_path=dangling_manifest,
            input_hashes={"inventory": "a" * 64},
        )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_remote_br_handoff_requires_finite_positive_heartbeat_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: float
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    with pytest.raises(module.RunnerFailure, match="finite and positive"):
        module.build_remote_br_launch_manifest(
            [sys.executable, "-c", "pass"],
            manifest_path=tmp_path / "handoff" / "manifest.json",
            launcher_path=tmp_path / "handoff" / "launcher.py",
            input_hashes={"inventory": "b" * 64},
            heartbeat_interval_seconds=value,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("first_heartbeat_timeout_seconds", 0.0),
        ("first_heartbeat_timeout_seconds", -1.0),
        ("first_heartbeat_timeout_seconds", float("nan")),
        ("first_heartbeat_timeout_seconds", float("inf")),
        ("first_heartbeat_timeout_seconds", float("-inf")),
        ("poll_interval_seconds", 0.0),
        ("poll_interval_seconds", -1.0),
        ("poll_interval_seconds", float("nan")),
        ("poll_interval_seconds", float("inf")),
        ("poll_interval_seconds", float("-inf")),
    ],
)
def test_remote_br_handoff_rejects_nonfinite_or_nonpositive_wait_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: float
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    kwargs = {field: value}
    with pytest.raises(module.RunnerFailure, match="finite and positive"):
        module.launch_remote_br_handoff(
            [sys.executable, "-c", "pass"],
            manifest_path=tmp_path / "handoff" / "manifest.json",
            input_hashes={"inventory": "c" * 64},
            **kwargs,
        )
    assert not (tmp_path / "handoff" / "manifest.json").exists()


def test_remote_br_handoff_persists_manifest_before_detached_spawn_and_verifies_first_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    manifest_path = tmp_path / "handoff" / "br_launch_manifest.json"
    command = [sys.executable, "-c", "import time; time.sleep(3)"]
    input_hashes = {"direction_train": "a" * 64, "prompt_inventory": "b" * 64}
    observed: dict[str, object] = {}
    real_popen = module.subprocess.Popen

    def recording_popen(argv, **kwargs):
        observed["manifest_before_spawn"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed["argv"] = list(argv)
        observed["kwargs"] = kwargs
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", recording_popen)
    result: dict[str, object] | None = None
    try:
        result = module.launch_remote_br_handoff(
            command,
            manifest_path=manifest_path,
            input_hashes=input_hashes,
            first_heartbeat_timeout_seconds=5.0,
            poll_interval_seconds=0.05,
            heartbeat_interval_seconds=0.05,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = json.loads(Path(manifest["status_path"]).read_text(encoding="utf-8"))
        heartbeat = json.loads(Path(manifest["heartbeat_path"]).read_text(encoding="utf-8"))
    finally:
        if result is not None:
            for key in ("pid", "launcher_pid"):
                try:
                    os.kill(int(result[key]), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        time.sleep(0.1)

    before_spawn = observed["manifest_before_spawn"]
    assert isinstance(before_spawn, dict)
    assert before_spawn["status"] == "PLANNED"
    assert before_spawn["model_id"] == module.MODEL_ID
    assert before_spawn["model_revision"] == module.REVISION
    assert before_spawn["repo_sha"] == module.REPO_SHA
    assert before_spawn["input_hashes"] == input_hashes
    assert before_spawn["command_digest"] == module._command_digest(command)
    assert before_spawn["preflight_only"] is True
    assert before_spawn["resumable"] is True
    assert before_spawn["protocol_only"] is True
    assert observed["argv"][-2:] == [str(before_spawn["launcher_path"]), str(manifest_path)]
    assert observed["kwargs"]["start_new_session"] is True

    assert result is not None
    assert result["pid_verified"] is True
    assert result["first_heartbeat_verified"] is True
    assert result["protocol_only"] is True
    assert result["semantic_runner"] == "caller_supplied_reviewed_command"
    assert manifest["status"] == "RUNNING"
    assert manifest["pid"] == result["pid"]
    assert manifest["launcher_pid"] == result["launcher_pid"]
    assert status["status"] == "RUNNING"
    assert heartbeat["status"] == "RUNNING"
    assert status["pid"] == result["pid"]
    assert heartbeat["pid"] == result["pid"]
    assert heartbeat["heartbeat_sequence"] >= 1


def test_remote_br_handoff_preserves_preflight_only_guard_before_any_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_runner(monkeypatch, tmp_path)
    manifest_path = tmp_path / "handoff" / "br_launch_manifest.json"
    command = [sys.executable, "-c", "import time; time.sleep(1)"]

    with pytest.raises(module.RunnerFailure, match="preflight-only"):
        module.launch_remote_br_handoff(
            command,
            manifest_path=manifest_path,
            input_hashes={"inventory": "c" * 64},
            run_mode="full",
        )

    assert not manifest_path.exists()
    assert not (manifest_path.parent / "remote_br_launcher.py").exists()
