#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping


# The official RunPod image used by the A100 preflight installs this exact
# PyTorch build. Qwen3.8's ``qwen3_5`` architecture is supported by the
# pinned Transformers release below. These are deliberately explicit rather
# than inferred from whichever interpreter happens to start this script.
EXPECTED_TORCH_VERSION = "2.8.0"
EXPECTED_TRANSFORMERS_VERSION = "5.16.1"
# Transformers' ``device_map`` loading path requires Accelerate. Keep this
# explicit and pinned so a fresh image cannot silently resolve a newer
# package with different loader behavior. Accelerate 1.10.1 supports Python
# 3.12 and torch >=2.0, which covers the image's torch 2.8 build.
EXPECTED_ACCELERATE_VERSION = "1.10.1"
EXPECTED_CUDA_VERSION_PREFIX = "12.8"
RUNTIME_REEXEC_ENV = "RSC_RUNTIME_RESOLVED"
RUNTIME_PYTHON_ENV = "RSC_RUNTIME_PYTHON"
RUNTIME_PROBE_CODE = r"""
import json
import sys

identity = {
    "python": sys.version,
    "python_executable": sys.executable,
    "python_version": ".".join(str(value) for value in sys.version_info[:3]),
    "torch": None,
    "transformers": None,
    "accelerate": None,
    "cuda_available": False,
    "cuda_version": None,
    "devices": [],
    "errors": [],
}
try:
    import torch

    identity["torch"] = str(torch.__version__)
    identity["cuda_version"] = str(getattr(getattr(torch, "version", None), "cuda", None))
    identity["cuda_available"] = bool(torch.cuda.is_available())
    if identity["cuda_available"]:
        identity["devices"] = [str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())]
except Exception as exc:
    identity["errors"].append(f"torch: {type(exc).__name__}: {exc}")
try:
    import transformers

    identity["transformers"] = str(transformers.__version__)
except Exception as exc:
    identity["errors"].append(f"transformers: {type(exc).__name__}: {exc}")
try:
    import accelerate

    identity["accelerate"] = str(accelerate.__version__)
except Exception as exc:
    identity["errors"].append(f"accelerate: {type(exc).__name__}: {exc}")

identity["ok"] = not identity["errors"] and bool(identity["cuda_available"])
print(json.dumps(identity, sort_keys=True))
sys.exit(0 if identity["ok"] else 1)
"""

RUN_ID = os.environ["RSC_RUN_ID"]
ALIAS = os.environ["RSC_MODEL_ALIAS"]
MODEL_ID = os.environ["RSC_MODEL_ID"]
REVISION = os.environ["RSC_MODEL_REVISION"]
REPO_SHA = os.environ["RSC_EXPECTED_REPO_SHA"]
REPO_URL = os.environ["RSC_REPO_URL"]
VOLUME = Path(os.environ.get("RSC_WORK_ROOT", "/workspace"))
STORAGE_KIND = os.environ.get("RSC_STORAGE_KIND", "network_volume")
EXPECTED_STORAGE_GB = int(os.environ.get("RSC_EXPECTED_STORAGE_GB", "120"))
RUN_ROOT = VOLUME / RUN_ID / ALIAS
STATE = RUN_ROOT / "state"
CHECKOUT = RUN_ROOT / "repo"
MODEL_CACHE = VOLUME / RUN_ID / "model-cache" / ALIAS
RUN_ROOT.mkdir(parents=True, exist_ok=True)
STATE.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
RUNTIME_VENV = RUN_ROOT / "runtime-venv"
STEERING_PLAN_ROOT_ENV = "RSC_STEERING_PLAN_ROOT"
STEERING_ARTIFACT_MANIFEST_ENV = "RSC_STEERING_ARTIFACT_MANIFEST"
ENV = os.environ.copy()
ENV.update(
    {
        "HF_HOME": str(MODEL_CACHE),
        "TRANSFORMERS_CACHE": str(MODEL_CACHE / "transformers"),
        "HUGGINGFACE_HUB_CACHE": str(MODEL_CACHE / "hub"),
        "RSC_BENCH_WORKSPACE_ROOT": str(RUN_ROOT),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
)
START = time.time()
WATCHDOG_DEADLINE = START + 4 * 60 * 60
CURRENT = {"wave": 0, "construct": "none", "phase": "boot", "completed_rows": 0, "total_rows": 0}
GATE_DIAGNOSTIC_VERSION = "wave1_baseline_collateral_gate_v1"
RUNTIME_PYTHON = sys.executable
RUNTIME_PROBE: dict[str, object] | None = None


class RunnerFailure(RuntimeError):
    pass


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _version_matches(value: object, expected: str) -> bool:
    """Match a package release while allowing a local build suffix."""

    return str(value or "").split("+", 1)[0] == expected


def _candidate_interpreters() -> list[str]:
    """Return deterministic candidates, preferring the image's PATH resolver."""

    candidates: list[str] = []
    venv_python = _runtime_venv_interpreter()
    # A re-exec must resolve the already prepared isolated runtime first. If
    # the PATH interpreters ran first here, the resolver would retry global
    # pip installs before reaching the venv on every child invocation.
    if venv_python and os.environ.get(RUNTIME_REEXEC_ENV) == "1":
        candidates.append(venv_python)
    for name in ("python", "python3"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)
    candidates.append(sys.executable)
    configured = os.environ.get("RSC_PYTHON_EXECUTABLE")
    if configured:
        candidates.append(configured)
    # RunPod's official image exposes Python 3.12 through /usr/local/bin;
    # retain common supported image layouts only as fallback candidates.
    candidates.extend(
        [
            "/usr/local/bin/python",
            "/usr/local/bin/python3",
            "/opt/conda/bin/python",
            "/opt/pyvenv/bin/python",
            "/opt/venv/bin/python",
        ]
    )
    if venv_python:
        candidates.append(venv_python)
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = str(candidate)
        if path in seen or not Path(path).is_file() or not os.access(path, os.X_OK):
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _runtime_venv_interpreter() -> str | None:
    """Return the prepared venv interpreter when its executable exists."""

    candidate = RUNTIME_VENV / "bin" / "python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _is_runtime_venv_interpreter(interpreter: str) -> bool:
    """Keep package installation strictly inside the prepared venv."""

    candidate = Path(interpreter)
    venv_bin = RUNTIME_VENV / "bin"
    return candidate.parent == venv_bin and candidate.name.startswith("python")


def _parse_probe_stdout(stdout: str) -> dict[str, object]:
    """Parse the final JSON line without discarding any raw probe output."""

    for line in reversed((stdout or "").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _probe_runtime(interpreter: str) -> dict[str, object]:
    command = [interpreter, "-c", RUNTIME_PROBE_CODE]
    completed = subprocess.run(
        command,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    identity = _parse_probe_stdout(stdout)
    python_version = str(identity.get("python_version", "0.0")).split(".")
    return {
        "interpreter": interpreter,
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "identity": identity,
        "torch_compatible": _version_matches(identity.get("torch"), EXPECTED_TORCH_VERSION),
        "transformers_compatible": _version_matches(identity.get("transformers"), EXPECTED_TRANSFORMERS_VERSION),
        "accelerate_compatible": _version_matches(identity.get("accelerate"), EXPECTED_ACCELERATE_VERSION),
        "python_compatible": len(python_version) >= 2
        and python_version[0].isdigit()
        and python_version[1].isdigit()
        and (int(python_version[0]), int(python_version[1])) >= (3, 11),
    }


def _probe_is_ready(probe: dict[str, object]) -> bool:
    """Return whether every model-loading prerequisite has passed.

    In particular, Accelerate is checked before any ``device_map`` model
    loader runs. This keeps a missing dependency from being reported as a
    misleading cascade of causal/multimodal loader errors.
    """

    identity = probe.get("identity")
    if not isinstance(identity, dict):
        return False
    cuda_version = str(identity.get("cuda_version") or "")
    return bool(
        probe.get("returncode") == 0
        and probe.get("torch_compatible")
        and probe.get("transformers_compatible")
        and probe.get("accelerate_compatible")
        and probe.get("python_compatible")
        and identity.get("cuda_available") is True
        and cuda_version.startswith(EXPECTED_CUDA_VERSION_PREFIX)
    )


def _require_model_runtime() -> None:
    """Fail before model construction when the pinned runtime is incomplete."""

    if RUNTIME_PROBE is None or not _probe_is_ready(RUNTIME_PROBE):
        raise RunnerFailure(
            "model loading prerequisites are not satisfied; pinned torch, "
            "transformers, accelerate, Python, and CUDA checks must pass first"
        )


def _create_runtime_venv(base_interpreter: str) -> dict[str, object]:
    """Create the isolated runtime while reusing image-provided Torch/CUDA."""

    command = [
        base_interpreter,
        "-m",
        "venv",
        "--system-site-packages",
        str(RUNTIME_VENV),
    ]
    RUNTIME_VENV.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "base_interpreter": base_interpreter,
        "runtime_venv": str(RUNTIME_VENV),
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def _install_transformers(interpreter: str) -> dict[str, object]:
    """Install the reviewed Transformers and Accelerate pins into the venv."""

    if not _is_runtime_venv_interpreter(interpreter):
        raise RunnerFailure("refusing to install Transformers outside the isolated runtime venv")

    command = [
        interpreter,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--upgrade",
        f"transformers=={EXPECTED_TRANSFORMERS_VERSION}",
        f"accelerate=={EXPECTED_ACCELERATE_VERSION}",
    ]
    env = os.environ.copy()
    env.update({"PIP_NO_INPUT": "1", "PIP_ROOT_USER_ACTION": "ignore"})
    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "interpreter": interpreter,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def _runtime_resolution() -> tuple[str, dict[str, object]]:
    """Resolve and, if needed, install pinned Python runtime dependencies."""

    attempts: list[dict[str, object]] = []
    runtime_attempted = False
    for interpreter in _candidate_interpreters():
        probe = _probe_runtime(interpreter)
        attempts.append({"kind": "probe", **probe})
        if _probe_is_ready(probe):
            report = {
                "selected_interpreter": interpreter,
                "runtime_venv": str(RUNTIME_VENV) if _is_runtime_venv_interpreter(interpreter) else None,
                "expected": {
                    "python_minimum": "3.11",
                    "torch": EXPECTED_TORCH_VERSION,
                    "transformers": EXPECTED_TRANSFORMERS_VERSION,
                    "accelerate": EXPECTED_ACCELERATE_VERSION,
                    "cuda_prefix": EXPECTED_CUDA_VERSION_PREFIX,
                },
                "attempts": attempts,
            }
            _write_json(STATE / "runtime_resolution.json", report)
            return interpreter, probe
        if not probe.get("torch_compatible") or runtime_attempted:
            continue

        runtime_attempted = True
        runtime_interpreter = interpreter if _is_runtime_venv_interpreter(interpreter) else _runtime_venv_interpreter()
        if runtime_interpreter is None:
            create = _create_runtime_venv(interpreter)
            attempts.append({"kind": "create_runtime_venv", **create})
            if create["returncode"] != 0:
                continue
            runtime_interpreter = _runtime_venv_interpreter()
        if runtime_interpreter is None:
            attempts.append(
                {
                    "kind": "runtime_venv_check",
                    "runtime_venv": str(RUNTIME_VENV),
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "venv creation completed without bin/python",
                }
            )
            continue

        runtime_probe = _probe_runtime(runtime_interpreter)
        attempts.append({"kind": "probe_runtime_venv", **runtime_probe})
        if _probe_is_ready(runtime_probe):
            report = {
                "selected_interpreter": runtime_interpreter,
                "runtime_venv": str(RUNTIME_VENV),
                "expected": {
                    "python_minimum": "3.11",
                    "torch": EXPECTED_TORCH_VERSION,
                    "transformers": EXPECTED_TRANSFORMERS_VERSION,
                    "accelerate": EXPECTED_ACCELERATE_VERSION,
                    "cuda_prefix": EXPECTED_CUDA_VERSION_PREFIX,
                },
                "attempts": attempts,
            }
            _write_json(STATE / "runtime_resolution.json", report)
            return runtime_interpreter, runtime_probe

        if not runtime_probe.get("torch_compatible"):
            continue
        install = _install_transformers(runtime_interpreter)
        attempts.append({"kind": "install_transformers", **install})
        if install["returncode"] != 0:
            continue
        after = _probe_runtime(runtime_interpreter)
        attempts.append({"kind": "probe_after_install", **after})
        if _probe_is_ready(after):
            report = {
                "selected_interpreter": runtime_interpreter,
                "runtime_venv": str(RUNTIME_VENV),
                "expected": {
                    "python_minimum": "3.11",
                    "torch": EXPECTED_TORCH_VERSION,
                    "transformers": EXPECTED_TRANSFORMERS_VERSION,
                    "accelerate": EXPECTED_ACCELERATE_VERSION,
                    "cuda_prefix": EXPECTED_CUDA_VERSION_PREFIX,
                },
                "attempts": attempts,
            }
            _write_json(STATE / "runtime_resolution.json", report)
            return runtime_interpreter, after
    report = {
        "selected_interpreter": None,
        "runtime_venv": str(RUNTIME_VENV),
        "expected": {
            "python_minimum": "3.11",
            "torch": EXPECTED_TORCH_VERSION,
            "transformers": EXPECTED_TRANSFORMERS_VERSION,
            "accelerate": EXPECTED_ACCELERATE_VERSION,
            "cuda_prefix": EXPECTED_CUDA_VERSION_PREFIX,
        },
        "attempts": attempts,
    }
    _write_json(STATE / "runtime_resolution.json", report)
    if attempts:
        _write_json(STATE / "runtime_identity.json", {"status": "FAILED", **attempts[-1]})
    raise RunnerFailure("no compatible official-image Python runtime; see " + str(STATE / "runtime_resolution.json"))


def _ensure_runtime() -> None:
    global RUNTIME_PYTHON, RUNTIME_PROBE
    selected, probe = _runtime_resolution()
    # Keep the selected runtime visible even when a test harness replaces the
    # process re-exec call; in production the child process imports these
    # values again after execvpe.
    RUNTIME_PYTHON = selected
    RUNTIME_PROBE = probe
    # Re-exec through the selected PATH entry even when it is a symlink to the
    # same binary. This matters when system and package-bearing aliases expose
    # different startup paths.
    current = os.path.abspath(sys.executable)
    resolved = os.path.abspath(selected)
    if current != resolved:
        if os.environ.get(RUNTIME_REEXEC_ENV) == "1":
            raise RunnerFailure(
                "runtime resolver selected a different interpreter after re-exec: "
                + selected
            )
        child_env = os.environ.copy()
        child_env[RUNTIME_REEXEC_ENV] = "1"
        child_env[RUNTIME_PYTHON_ENV] = selected
        if _is_runtime_venv_interpreter(selected):
            child_env["VIRTUAL_ENV"] = str(RUNTIME_VENV)
            child_env["PATH"] = str(Path(selected).parent) + os.pathsep + child_env.get("PATH", "")
        os.execvpe(selected, [selected, *sys.argv], child_env)
        raise AssertionError("os.execvpe returned unexpectedly")
    ENV[RUNTIME_PYTHON_ENV] = selected
    if _is_runtime_venv_interpreter(selected):
        ENV["VIRTUAL_ENV"] = str(RUNTIME_VENV)
        ENV["PATH"] = str(Path(selected).parent) + os.pathsep + ENV.get("PATH", "")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def status_file(status: str = "RUNNING", error: str | None = None) -> None:
    payload = {
        "run_id": RUN_ID,
        "model_alias": ALIAS,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "status": status,
        "wave": CURRENT["wave"],
        "construct": CURRENT["construct"],
        "phase": CURRENT["phase"],
        "completed_rows": int(CURRENT["completed_rows"]),
        "total_rows": int(CURRENT["total_rows"]),
        "last_progress_timestamp": now_iso(),
        "started_epoch": int(START),
        "watchdog_deadline_epoch": int(WATCHDOG_DEADLINE),
        "watchdog_deadline": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(WATCHDOG_DEADLINE)),
        "pid": os.getpid(),
    }
    if error:
        payload["error"] = error
    (STATE / "status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (STATE / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    print(
        "[RSC_STATE] "
        + " ".join(
            [
                "alias=" + ALIAS,
                "wave=" + str(CURRENT["wave"]),
                "construct=" + str(CURRENT["construct"]),
                "phase=" + str(CURRENT["phase"]),
                "completed=" + str(CURRENT["completed_rows"]),
                "total=" + str(CURRENT["total_rows"]),
                "status=" + status,
            ]
        ),
        flush=True,
    )


def write_hashes() -> None:
    entries = []
    if RUN_ROOT.exists():
        for path in sorted(RUN_ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(RUN_ROOT)
            if any(part in {"repo", "model-cache", "runtime-venv", ".git"} for part in relative.parts):
                continue
            if relative.as_posix() == "state/output_hashes.json":
                continue
            entries.append(
                {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
    (STATE / "output_hashes.json").write_text(
        json.dumps({"run_root": str(RUN_ROOT), "files": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(command, output: Path, *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=str(cwd or CHECKOUT) if (cwd or CHECKOUT).exists() else None,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output.write_text(completed.stdout or "", encoding="utf-8")
    if check and completed.returncode != 0:
        raise RunnerFailure(
            "command failed with "
            + str(completed.returncode)
            + ": "
            + " ".join(str(value) for value in command)
            + "; see "
            + str(output)
        )
    return completed


def run_captured(
    command, output: Path, *, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one identity command and retain exact stdout/stderr separately."""

    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = [str(value) for value in command]
    completed = subprocess.run(
        normalized,
        cwd=str(cwd or CHECKOUT) if (cwd or CHECKOUT).exists() else None,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    _write_json(
        output,
        {
            "command": normalized,
            "returncode": completed.returncode,
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
        },
    )
    if check and completed.returncode != 0:
        raise RunnerFailure(
            "command failed with "
            + str(completed.returncode)
            + ": "
            + " ".join(normalized)
            + "; see "
            + str(output)
        )
    return completed


def wave_definition(wave: int):
    base = CHECKOUT / "configs/construct_benchmark"
    if wave == 1:
        return (
            base / f"run_configs/wave1_four_construct_{ALIAS}_model_preflight_repaired_v4.json",
            CHECKOUT / "results/benchmark/prompt_inventories/wave1_preflight_v4_luna_v2/combined.csv",
            CHECKOUT / f"results/benchmark/model_preflight_v4/wave1_preflight_v4_luna_v2_normalized_v1_{ALIAS}_selection.json",
            base / "gates/model_behavior_accessibility_preflight_v2_diagnostic_repair.json",
            [
                base / "constructs/realization_account_closure_v4.json",
                base / "constructs/evidence_diagnosticity_v5.json",
                base / "constructs/source_reliability_v3.json",
                base / "constructs/persistence_continuation_v4.json",
            ],
        )
    if wave == 2:
        return (
            base / f"run_configs/wave2_four_construct_{ALIAS}_model_preflight_repaired_v3.json",
            CHECKOUT / "results/benchmark/prompt_inventories/wave2_four_construct_repaired_v3_luna_label_v4_v2vector_r1/combined.csv",
            CHECKOUT / f"results/benchmark/model_preflight_v4/wave2_repaired_v3_luna_label_v4_v2vector_normalized_v1_{ALIAS}_selection.json",
            base / "gates/model_behavior_accessibility_wave2_repaired_v3.json",
            [
                base / "constructs/reference_frame_repaired_v3_collateral_v3_disjoint.json",
                base / "constructs/prior_weighting_repaired_v3_collateral_v3_disjoint.json",
                base / "constructs/authority_deference_repaired_v3_collateral_v3_disjoint.json",
                base / "constructs/exploration_exploitation_repaired_v3_collateral_v3_disjoint.json",
            ],
        )
    if wave == 3:
        return (
            base / f"run_configs/wave3_four_construct_{ALIAS}_model_preflight_repaired_v3.json",
            CHECKOUT / "results/benchmark/prompt_inventories/wave3_four_construct_repaired_v3_luna_label_v4_v2vector_r1/combined.csv",
            CHECKOUT / f"results/benchmark/model_preflight_v4/wave3_repaired_v3_luna_label_v4_v2vector_normalized_v1_{ALIAS}_selection.json",
            base / "gates/model_behavior_accessibility_wave3_repaired_v3.json",
            [
                base / "constructs/ambiguity_orientation_repaired_v3_collateral_v3_disjoint.json",
                base / "constructs/causal_interpretation_repaired_v3_collateral_v3_disjoint_label_v3.json",
                base / "constructs/consensus_conformity_repaired_v3_collateral_v3_disjoint_label_v4.json",
                base / "constructs/plan_replanning_repaired_v3_collateral_v3_disjoint.json",
            ],
        )
    if wave == 4:
        return (
            base / f"run_configs/wave4_four_construct_{ALIAS}_model_preflight_repaired_v3.json",
            CHECKOUT / "results/benchmark/prompt_inventories/wave4_four_construct_repaired_v3_luna_label_v4_v2vector_r1/combined.csv",
            CHECKOUT / f"results/benchmark/model_preflight_v4/wave4_repaired_v3_luna_label_v4_v2vector_normalized_v1_{ALIAS}_selection.json",
            base / "gates/model_behavior_accessibility_wave4_repaired_v3.json",
            [
                base / "constructs/temporal_orientation_repaired_v3_collateral_v3_disjoint.json",
                base / "constructs/epistemic_uncertainty_repaired_v3_collateral_v3_disjoint.json",
                base / "constructs/reciprocity_obligation_repaired_v3_collateral_v3_disjoint_label_v3.json",
                base / "constructs/goal_shielding_repaired_v3_collateral_v3_disjoint.json",
            ],
        )
    raise ValueError(wave)


def verify_release() -> None:
    CURRENT.update({"wave": 0, "construct": "release", "phase": "identity_storage_revision"})
    status_file()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (STATE / "gpu_identity.txt").write_text(gpu.stdout or "", encoding="utf-8")
    gpu_lines = [line.strip() for line in (gpu.stdout or "").splitlines() if line.strip()]
    if gpu.returncode != 0 or len(gpu_lines) != 1 or "A100" not in gpu_lines[0]:
        raise RunnerFailure("in-pod GPU identity check failed")
    stat = os.statvfs("/")
    total = stat.f_frsize * stat.f_blocks
    free = stat.f_frsize * stat.f_bavail
    minimum_bytes = EXPECTED_STORAGE_GB * 1_000_000_000
    if STORAGE_KIND == "network_volume":
        storage_ok = os.path.ismount(VOLUME) and total >= minimum_bytes
    else:
        storage_ok = VOLUME.is_dir() and total >= minimum_bytes
    if not storage_ok or free < 20 * 1024**3:
        raise RunnerFailure(
            f"{STORAGE_KIND} storage check failed: total={total} free={free} expected_gb={EXPECTED_STORAGE_GB}"
        )
    (STATE / "identity_storage.json").write_text(
        json.dumps(
            {
                "storage_kind": STORAGE_KIND,
                "volume_id": os.environ.get("RSC_VOLUME_ID"),
                "region": os.environ.get("RSC_VOLUME_REGION"),
                "mount_target": str(VOLUME),
                "filesystem_check_path": "/",
                "total_bytes": total,
                "free_bytes": free,
                "gpu": gpu_lines[0],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if RUNTIME_PROBE is None:
        raise RunnerFailure("runtime resolver did not produce an identity record")
    runtime_identity = {
        "status": "PASS",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "expected": {
            "python_minimum": "3.11",
            "torch": EXPECTED_TORCH_VERSION,
            "transformers": EXPECTED_TRANSFORMERS_VERSION,
            "accelerate": EXPECTED_ACCELERATE_VERSION,
            "cuda_prefix": EXPECTED_CUDA_VERSION_PREFIX,
        },
        **RUNTIME_PROBE,
    }
    _write_json(STATE / "runtime_identity.json", runtime_identity)
    if not _probe_is_ready(RUNTIME_PROBE):
        raise RunnerFailure(
            "required torch/transformers/accelerate runtime check failed; see "
            + str(STATE / "runtime_identity.json")
        )
    hf = run_captured(
        ["git", "ls-remote", f"https://huggingface.co/{MODEL_ID}", "refs/heads/main"],
        STATE / "model_revision_identity.json",
        cwd=RUN_ROOT,
    )
    observed = (hf.stdout or "").split()[0] if hf.stdout else ""
    (STATE / "model_revision_remote.txt").write_text(hf.stdout or "", encoding="utf-8")
    runtime_identity["model_revision_observed"] = observed
    runtime_identity["model_revision_match"] = observed == REVISION
    _write_json(STATE / "runtime_identity.json", runtime_identity)
    if observed != REVISION:
        raise RunnerFailure(f"registered Hugging Face revision mismatch: {observed}")
    CURRENT.update({"phase": "checkout_and_release_hashes", "construct": "release"})
    status_file()
    if CHECKOUT.exists():
        raise RunnerFailure("refusing to reuse an existing checkout")
    run(["git", "clone", "--quiet", "--no-tags", REPO_URL, CHECKOUT], STATE / "git_clone.log", cwd=RUN_ROOT)
    fetched = run(["git", "fetch", "--quiet", "origin", REPO_SHA], STATE / "git_fetch.log", check=False)
    if fetched.returncode != 0:
        run(["git", "fetch", "--quiet", "origin", "main"], STATE / "git_fetch_main.log")
    run(["git", "checkout", "--quiet", "--detach", REPO_SHA], STATE / "git_checkout.log")
    actual = run(["git", "rev-parse", "HEAD"], STATE / "git_head.txt").stdout.strip()
    dirty = run(["git", "status", "--porcelain"], STATE / "git_status.txt").stdout.strip()
    if actual != REPO_SHA or dirty:
        raise RunnerFailure("clean pinned checkout check failed")
    ENV["PYTHONPATH"] = str(CHECKOUT / "src")
    index = CHECKOUT / "configs/construct_benchmark/preflight_campaigns/waves1_4_preflight_artifact_index_v1.json"
    run(
        [
            RUNTIME_PYTHON,
            str(CHECKOUT / "scripts/validate_preflight_artifact_index.py"),
            "--index",
            str(index),
            "--repo-root",
            str(CHECKOUT),
            "--json",
        ],
        STATE / "release_index_validation.json",
    )
    index_data = json.loads(index.read_text(encoding="utf-8"))
    hash_rows = []
    for entry in index_data["entries"]:
        row = {"wave": entry["wave"], "model": entry["model"]}
        for key, artifact in entry["artifacts"].items():
            if not isinstance(artifact, dict) or "path" not in artifact:
                continue
            path = CHECKOUT / artifact["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact.get("sha256"):
                raise RunnerFailure(f"release hash mismatch: {artifact['path']}")
            row[key] = {"path": artifact["path"], "sha256": digest}
            if "selection_sha256" in artifact:
                row[key]["selection_sha256"] = artifact["selection_sha256"]
            manifest = artifact.get("manifest")
            if isinstance(manifest, dict):
                manifest_path = CHECKOUT / manifest["path"]
                manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                if manifest_digest != manifest.get("sha256"):
                    raise RunnerFailure(f"release manifest hash mismatch: {manifest['path']}")
                row[key + "_manifest"] = {"path": manifest["path"], "sha256": manifest_digest}
        hash_rows.append(row)
    (STATE / "release_hashes.json").write_text(
        json.dumps({"repository_sha": actual, "entries": hash_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_record(path: Path) -> dict[str, object]:
    """Return non-secret provenance for one output or manifest path."""

    path = Path(path)
    record: dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        try:
            record.update(
                {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        except OSError as exc:
            record["read_error"] = f"{type(exc).__name__}: {exc}"
    else:
        record.update({"bytes": None, "sha256": None})
    return record


def _read_rows_for_diagnostics(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Read usable JSONL rows while retaining bounded syntax diagnostics."""

    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return rows, [{"line": None, "error": f"{type(exc).__name__}: {exc}"}]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if len(errors) < 8:
                errors.append({"line": line_number, "error": f"JSONDecodeError: {exc.msg}"})
            continue
        if not isinstance(value, dict):
            if len(errors) < 8:
                errors.append({"line": line_number, "error": "row is not a JSON object"})
            continue
        rows.append(value)
    return rows, errors


def _preserve_gate_artifacts(
    wave_root: Path,
    paths: list[Path],
    *,
    wave: int,
) -> dict[str, object]:
    """Copy bounded gate inputs/reports into the state tree before failure.

    The state tree is still on the worker's workspace, so the supervisor can
    copy one stable directory as soon as the streamed gate event is observed.
    This deliberately copies only the two baseline JSONL files, their
    manifests, and compact reports; model caches and activation artifacts are
    never copied here.
    """

    destination = STATE / f"wave{wave}" / "baseline_collateral"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, object]] = []
    for source in paths:
        source = Path(source)
        entry = _artifact_record(source)
        entry["destination"] = None
        if source.is_file():
            target = destination / source.name
            if source.resolve() != target.resolve():
                try:
                    shutil.copy2(source, target)
                except OSError as exc:
                    entry["copy_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    entry["destination"] = str(target)
            else:
                entry["destination"] = str(target)
            if target.is_file():
                entry["destination_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        copied.append(entry)
    return {"directory": str(destination), "files": copied}


def _stream_gate_report(report: Mapping[str, object]) -> None:
    """Emit a single bounded, machine-readable gate event for SSH capture."""

    print(
        "[RSC_GATE_REPORT] "
        + json.dumps(dict(report), ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _add_matrix_stage(
    report: dict[str, object],
    row: dict[str, object],
    *,
    construct_id: str,
    stage: str,
    failures: list[str],
) -> None:
    """Record pass/fail for one explicit parser/variation/collateral stage."""

    entry: dict[str, object] = {
        "construct_id": construct_id,
        "stage": stage,
        "pass": not failures,
        "failures": list(failures),
    }
    row.setdefault("failure_matrix", []).append(entry)
    report.setdefault("failure_matrix", []).append(entry)


def _safe_selected_rows(
    rows: list[dict[str, object]],
    *,
    construct_id: str,
    split: str,
    prompt_ids: list[str],
    expected_model: Mapping[str, object],
    selected_rows_fn,
) -> tuple[list[dict[str, object]], list[str], str | None]:
    """Select rows without allowing a duplicate/mismatch to erase diagnostics."""

    try:
        selected, missing = selected_rows_fn(
            rows,
            construct_id=construct_id,
            split=split,
            prompt_ids=prompt_ids,
            expected_model=expected_model,
        )
        return selected, missing, None
    except Exception as exc:
        wanted = {str(value) for value in prompt_ids}
        fallback = [
            dict(row)
            for row in rows
            if row.get("construct_id") == construct_id
            and row.get("split") == split
            and str(row.get("prompt_id")) in wanted
        ]
        observed = {str(row.get("prompt_id")) for row in fallback}
        return fallback, sorted(wanted - observed), f"{type(exc).__name__}: {exc}"


def _fallback_stats(rows: list[dict[str, object]], error: str) -> dict[str, object]:
    """Produce a conservative stats shape when scoring cannot run."""

    return {
        "selected_item_count": len(rows),
        "valid_parser_rows": None,
        "valid_primary_rows": 0,
        "invalid_rows": len(rows),
        "valid_primary_rate": 0.0,
        "invalid_or_unscorable_items": len(rows),
        "sample_sd": None,
        "distinct_outcomes": 0,
        "mean_correctness": None,
        "outcome_frequency": {},
        "floor_outcome": None,
        "ceiling_outcome": None,
        "floor_share": None,
        "ceiling_share": None,
        "parser_failure_count": len(rows),
        "parser_failure_examples": [
            {
                "record_id": row.get("record_id"),
                "prompt_id": row.get("prompt_id"),
                "error": error,
            }
            for row in rows[:8]
        ],
        "diagnostic_error": error,
    }


def _safe_stats(rows: list[dict[str, object]], spec, *, collateral: bool, stats_fn) -> tuple[dict[str, object], str | None]:
    try:
        return dict(stats_fn(rows, spec, collateral=collateral)), None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return _fallback_stats(rows, error), error


def _failure_buckets(failures: list[str], *, collateral: bool = False) -> dict[str, list[str]]:
    """Split gate strings into stable diagnostic categories."""

    buckets = {"coverage": [], "parser": [], "variation": [], "correctness": []}
    for failure in failures:
        text = str(failure)
        lower = text.casefold()
        if (
            "item_count" in lower
            or "missing selected" in lower
            or "missing behavior" in lower
            or "missing collateral" in lower
            or "selection_error" in lower
        ):
            buckets["coverage"].append(text)
        elif "correctness" in lower:
            buckets["correctness"].append(text)
        elif "distinct_outcomes" in lower or "sample_sd" in lower or "ceiling_share" in lower or "floor_share" in lower:
            buckets["variation"].append(text)
        elif "valid_primary" in lower or "invalid_or_unscorable" in lower or "scoring_error" in lower:
            buckets["parser"].append(text)
        elif collateral:
            buckets["parser"].append(text)
        else:
            buckets["variation"].append(text)
    return buckets


def baseline_gate(
    config_path,
    selection_path,
    gate_path,
    specs,
    behavior,
    collateral,
    report_path,
    *,
    upstream_error: str | None = None,
):
    """Evaluate baseline/collateral outputs and always persist diagnostics.

    A gate failure is a normal engineering outcome, not an exception that may
    discard evidence.  Structural validation errors remain fail-closed, while
    usable rows are scored independently so the report identifies which
    construct and which parser/variation/collateral criterion failed.
    """

    sys.path.insert(0, str(CHECKOUT / "src"))
    from construct_benchmark.config import load_construct_specs
    from construct_benchmark.manifests import file_sha256
    from construct_benchmark.model_preflight import (
        _behavior_stats,
        _gate_behavior,
        _selected_rows,
        _validate_model_output_manifest,
        load_preflight_gate_config,
    )

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selection: dict[str, object] = {}
    loaded_specs: dict[str, object] = {}
    gate: dict[str, object] = {}
    setup_failures: list[dict[str, object]] = []
    try:
        raw_selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        if not isinstance(raw_selection, dict):
            raise ValueError("selection manifest must be a JSON object")
        selection = raw_selection
    except Exception as exc:
        setup_failures.append(
            {
                "construct_id": None,
                "stage": "selection_manifest",
                "code": "selection_manifest_error",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
    try:
        loaded_specs = load_construct_specs(specs)
    except Exception as exc:
        setup_failures.append(
            {
                "construct_id": None,
                "stage": "construct_specs",
                "code": "construct_spec_error",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
    try:
        gate = load_preflight_gate_config(gate_path)
        gate["gate_config_sha256"] = file_sha256(gate_path)
    except Exception as exc:
        setup_failures.append(
            {
                "construct_id": None,
                "stage": "gate_config",
                "code": "gate_config_error",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    model = selection.get("model", {})
    if not isinstance(model, dict):
        model = {}
    construct_ids = [str(value) for value in selection.get("construct_ids", []) if value is not None]
    if not construct_ids:
        construct_ids = list(loaded_specs)
    thresholds = {
        **dict(gate.get("execution_contract", {})),
        **dict(gate.get("release_thresholds", {})),
    }
    item_bounds = dict(gate.get("item_bounds", {}))
    minimum_items = int(item_bounds.get("minimum", 8))
    maximum_items = int(item_bounds.get("maximum", 16))
    report: dict[str, object] = {
        "schema_version": "0.1.0",
        "diagnostic_version": GATE_DIAGNOSTIC_VERSION,
        "stage": "baseline_collateral",
        "created_at": now_iso(),
        "model": model,
        "config": _artifact_record(Path(config_path)),
        "selection": _artifact_record(Path(selection_path)),
        "gate": _artifact_record(Path(gate_path)),
        "inputs": {
            "behavior": _artifact_record(Path(behavior)),
            "behavior_manifest": _artifact_record(Path(str(behavior) + ".manifest.json")),
            "collateral": _artifact_record(Path(collateral)),
            "collateral_manifest": _artifact_record(Path(str(collateral) + ".manifest.json")),
        },
        "thresholds": {
            "item_bounds": {"minimum": minimum_items, "maximum": maximum_items},
            **{str(key): value for key, value in thresholds.items()},
        },
        "constructs": {},
        "failure_matrix": [],
        "failures": [],
        "pass": not setup_failures,
    }
    if upstream_error:
        report["upstream_error"] = str(upstream_error)
        report["pass"] = False
        setup_failures.append(
            {
                "construct_id": None,
                "stage": "generation",
                "code": "generation_error",
                "detail": str(upstream_error),
            }
        )
    report["failures"].extend(setup_failures)
    report["failure_matrix"].extend(setup_failures)

    common = {
        "expected_manifest_type": "construct_behavior_output",
        "expected_model": model,
        "expected_inventory_sha256": selection.get("source_inventory_sha256"),
        "required_prompt_format": thresholds.get("required_prompt_format", thresholds.get("prompt_format")),
        "require_constrained_numeric_generation": bool(
            thresholds.get("require_constrained_numeric_generation", thresholds.get("constrained_numeric_generation", False))
        ),
        "require_manifest_record_count": True,
        "require_thinking_disabled": bool(thresholds.get("disable_thinking_when_supported", False)),
    }
    validated_rows: dict[str, list[dict[str, object]]] = {}
    for stage, path in (("behavior", Path(behavior)), ("collateral", Path(collateral))):
        try:
            rows, manifest = _validate_model_output_manifest(path, **common)
            validated_rows[stage] = [dict(row) for row in rows]
            report["inputs"][stage]["validation"] = {"pass": True, "manifest": manifest}
        except Exception as exc:
            rows, parse_errors = _read_rows_for_diagnostics(path)
            validated_rows[stage] = rows
            detail = f"{type(exc).__name__}: {exc}"
            report["inputs"][stage]["validation"] = {
                "pass": False,
                "error": detail,
                "jsonl_parse_errors": parse_errors,
            }
            failure = {
                "construct_id": None,
                "stage": f"{stage}_manifest",
                "code": "output_manifest_error",
                "detail": detail,
            }
            report["failures"].append(failure)
            report["failure_matrix"].append(failure)
            report["pass"] = False

    selected_by_construct = selection.get("selected", {})
    if not isinstance(selected_by_construct, dict):
        selected_by_construct = {}
    for construct_id in construct_ids:
        selected = selected_by_construct.get(construct_id, {})
        if not isinstance(selected, dict):
            selected = {}
        spec = loaded_specs.get(construct_id)
        row: dict[str, object] = {"failure_matrix": [], "failures": []}
        if spec is None:
            row["failures"].append("construct specification is unavailable")
            _add_matrix_stage(
                report,
                row,
                construct_id=construct_id,
                stage="construct_spec",
                failures=["construct specification is unavailable"],
            )
            report["constructs"][construct_id] = row
            report["failures"].append({"construct_id": construct_id, "failures": row["failures"]})
            report["pass"] = False
            continue
        behavior_selection = selected.get("behavior_eval", {})
        collateral_selection = selected.get("collateral_eval", {})
        behavior_ids = [str(value) for value in behavior_selection.get("prompt_ids", [])] if isinstance(behavior_selection, dict) else []
        collateral_ids = [str(value) for value in collateral_selection.get("prompt_ids", [])] if isinstance(collateral_selection, dict) else []
        behavior_selected, missing_behavior, behavior_selection_error = _safe_selected_rows(
            validated_rows.get("behavior", []),
            construct_id=construct_id,
            split="behavior_eval",
            prompt_ids=behavior_ids,
            expected_model=model,
            selected_rows_fn=_selected_rows,
        )
        collateral_selected, missing_collateral, collateral_selection_error = _safe_selected_rows(
            validated_rows.get("collateral", []),
            construct_id=construct_id,
            split="collateral_eval",
            prompt_ids=collateral_ids,
            expected_model=model,
            selected_rows_fn=_selected_rows,
        )
        behavior_stats, behavior_stats_error = _safe_stats(
            behavior_selected,
            spec,
            collateral=False,
            stats_fn=_behavior_stats,
        )
        collateral_stats, collateral_stats_error = _safe_stats(
            collateral_selected,
            spec,
            collateral=True,
            stats_fn=_behavior_stats,
        )
        behavior_failures: list[str] = []
        if missing_behavior:
            behavior_failures.append(f"missing selected behavior prompts: {missing_behavior[:3]}")
        if behavior_selection_error:
            behavior_failures.append(f"selection_error={behavior_selection_error}")
        if behavior_stats_error:
            behavior_failures.append(f"scoring_error={behavior_stats_error}")
        try:
            behavior_pass, gate_failures = _gate_behavior(
                behavior_stats,
                minimum_items=minimum_items,
                maximum_items=maximum_items,
                minimum_valid_rate=float(thresholds.get("behavior_minimum_valid_rate", 1.0)),
                maximum_invalid_items=int(thresholds.get("behavior_maximum_invalid_items", 0)),
                minimum_distinct_outcomes=int(thresholds.get("behavior_minimum_distinct_outcomes", 3)),
                minimum_sample_sd=float(thresholds.get("behavior_minimum_sample_sd", 2.0)),
                maximum_ceiling_share=thresholds.get("behavior_maximum_ceiling_share"),
                maximum_floor_share=thresholds.get("behavior_maximum_floor_share"),
            )
            behavior_failures.extend(gate_failures)
        except Exception as exc:
            behavior_pass = False
            behavior_failures.append(f"scoring_error={type(exc).__name__}: {exc}")

        collateral_failures: list[str] = []
        if missing_collateral:
            collateral_failures.append(f"missing selected collateral prompts: {missing_collateral[:3]}")
        if collateral_selection_error:
            collateral_failures.append(f"selection_error={collateral_selection_error}")
        if collateral_stats_error:
            collateral_failures.append(f"scoring_error={collateral_stats_error}")
        collateral_valid_rate = collateral_stats.get("valid_primary_rate", 0.0)
        if collateral_valid_rate is None or float(collateral_valid_rate) < float(thresholds.get("collateral_minimum_valid_rate", 0.95)):
            collateral_failures.append(
                f"valid_primary_rate={collateral_valid_rate!r} < {float(thresholds.get('collateral_minimum_valid_rate', 0.95)):.4f}"
            )
        correctness = collateral_stats.get("mean_correctness")
        if correctness is None or float(correctness) < float(thresholds.get("collateral_minimum_correctness_rate", 0.75)):
            collateral_failures.append(
                f"correctness_rate={correctness!r} < {float(thresholds.get('collateral_minimum_correctness_rate', 0.75)):.4f}"
            )
        collateral_count_failures: list[str] = []
        if not minimum_items <= len(collateral_selected) <= maximum_items:
            collateral_count_failures.append(
                f"item_count={len(collateral_selected)} outside [{minimum_items}, {maximum_items}]"
            )
        collateral_failures.extend(collateral_count_failures)
        collateral_pass = not collateral_failures

        behavior_buckets = _failure_buckets(behavior_failures)
        collateral_buckets = _failure_buckets(collateral_failures, collateral=True)
        _add_matrix_stage(report, row, construct_id=construct_id, stage="behavior_coverage", failures=behavior_buckets["coverage"])
        _add_matrix_stage(report, row, construct_id=construct_id, stage="behavior_parser", failures=behavior_buckets["parser"])
        _add_matrix_stage(report, row, construct_id=construct_id, stage="behavior_variation", failures=behavior_buckets["variation"])
        _add_matrix_stage(report, row, construct_id=construct_id, stage="collateral_coverage", failures=collateral_buckets["coverage"])
        _add_matrix_stage(report, row, construct_id=construct_id, stage="collateral_parser", failures=collateral_buckets["parser"])
        _add_matrix_stage(report, row, construct_id=construct_id, stage="collateral_correctness", failures=collateral_buckets["correctness"])
        row.update(
            {
                "behavior": behavior_stats,
                "collateral": collateral_stats,
                "behavior_pass": behavior_pass and not behavior_failures,
                "collateral_pass": collateral_pass,
                "failures": behavior_failures + collateral_failures,
                "selected_counts": {
                    "behavior_eval": len(behavior_selected),
                    "collateral_eval": len(collateral_selected),
                    "expected_behavior_eval": len(behavior_ids),
                    "expected_collateral_eval": len(collateral_ids),
                },
            }
        )
        report["constructs"][construct_id] = row
        if behavior_failures or collateral_failures:
            report["pass"] = False
            report["failures"].append({"construct_id": construct_id, "failures": row["failures"]})

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def discover_plans(config_path, specs, report_path):
    sys.path.insert(0, str(CHECKOUT / "src"))
    from construct_benchmark.config import load_construct_specs, load_run_config
    from construct_benchmark.manifests import canonical_hash

    # ``plan_construct_steering.py`` records the canonical hash of the parsed
    # RunConfig mapping.  Hash the same normalized representation here rather
    # than the raw JSON, whose optional/default fields and ordering can differ.
    loaded_config = load_run_config(config_path)
    config_hash = canonical_hash(loaded_config.to_mapping())
    raw_config_hash = canonical_hash(json.loads(config_path.read_text(encoding="utf-8")))
    configured_root = os.environ.get(STEERING_PLAN_ROOT_ENV)
    if configured_root:
        search_root = Path(configured_root)
        if not search_root.is_absolute():
            search_root = VOLUME / search_root
        search_root_source = "environment"
    else:
        # Plans and their direction artifacts are a per-run staging input, not
        # repository files.  Keep the default deterministic and isolated from
        # old runs, caches, and the checkout itself.  The launcher may override
        # this with an absolute RSC_STEERING_PLAN_ROOT when using a shared
        # persistent workspace.
        search_root = RUN_ROOT / "steering_plans"
        search_root_source = "default"

    candidates = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    if search_root.is_dir():
        plan_paths = sorted(path for path in search_root.rglob("*.json") if path.is_file())
    else:
        plan_paths = []
    for path in plan_paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            reject("invalid_json")
            continue
        if not isinstance(raw, dict):
            reject("non_object")
            continue
        if raw.get("plan_type") != "construct_steering_conditions":
            reject("plan_type")
            continue
        model = raw.get("model")
        provenance = raw.get("provenance")
        if not isinstance(model, dict) or model.get("model_id") != MODEL_ID or model.get("revision") != REVISION:
            reject("model_or_revision")
            continue
        # The registered plan schema expresses this invariant through
        # intervention_timing and does not include a top-level prefill_only
        # field.  Accept that schema, while rejecting an explicit false (or
        # malformed) value so a broad staging directory cannot weaken timing.
        if "prefill_only" in raw and raw.get("prefill_only") is not True:
            reject("not_prefill_only")
            continue
        if raw.get("position_mode") != "last" or raw.get("intervention_timing") != "prefill_only":
            reject("steering_timing")
            continue
        if not isinstance(provenance, dict) or provenance.get("run_config_hash") != config_hash:
            reject("run_config_hash")
            continue
        construct_id = raw.get("construct_id")
        if not isinstance(construct_id, str) or not construct_id:
            reject("construct_id")
            continue
        candidates.append((construct_id, str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
    selected = {}
    duplicates: dict[str, list[str]] = {}
    for construct_id, path, digest in sorted(candidates):
        if construct_id in selected:
            duplicates.setdefault(construct_id, [selected[construct_id]["path"]]).append(path)
            continue
        selected[construct_id] = {"path": path, "sha256": digest}
    # Wave specs may be versioned overlays.  Their raw JSON intentionally has
    # ``base_spec_path`` rather than a duplicated top-level construct_id; use
    # the same inheritance-aware loader as the rest of the runner.
    expected = list(load_construct_specs(specs))
    report = {
        "pass": set(expected).issubset(selected) and not duplicates,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "run_config_hash": config_hash,
        "run_config_hash_method": "canonical_hash(load_run_config(config).to_mapping())",
        "raw_run_config_hash": raw_config_hash,
        "search_root": str(search_root),
        "search_root_source": search_root_source,
        "search_root_exists": search_root.is_dir(),
        "expected_construct_ids": expected,
        "selected": selected,
        "missing": sorted(set(expected) - set(selected)),
        "candidate_count": len(candidates),
        "rejected_by_reason": rejected,
        "duplicates": duplicates,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["pass"]:
        if not report["search_root_exists"]:
            raise RunnerFailure(
                "registered steering plan staging root is missing: "
                + str(search_root)
                + "; stage model-matched plans there before starting steering"
            )
        if duplicates:
            raise RunnerFailure("registered steering plans are ambiguous: " + str(sorted(duplicates)))
        raise RunnerFailure("registered steering plans are missing: " + str(report["missing"]))
    return report


def stage_steering_artifacts(wave: int, wave_root: Path) -> Path | None:
    """Stage one model/wave B/R bundle before steering-plan discovery.

    The preflight deliberately does not log representations or invent a
    direction.  If the executor provides a hash-bound source bundle through
    ``RSC_STEERING_ARTIFACT_MANIFEST``, this hook invokes the offline staging
    validator into an isolated per-wave root and points discovery there.  A
    manifest may use ``{wave}`` and ``{alias}`` placeholders, or the variable
    may name a directory containing ``wave{wave}_{alias}.json``.  Without a
    manifest, an explicitly configured ``RSC_STEERING_PLAN_ROOT`` remains
    supported for an already-staged bundle.
    """

    raw_manifest = os.environ.get(STEERING_ARTIFACT_MANIFEST_ENV)
    if not raw_manifest:
        return None
    manifest_value = raw_manifest.format(wave=wave, alias=ALIAS, model_alias=ALIAS)
    manifest_path = Path(manifest_value).expanduser()
    if manifest_path.is_dir():
        manifest_path = manifest_path / f"wave{wave}_{ALIAS}.json"
    if not manifest_path.is_file():
        raise RunnerFailure(
            "registered steering artifact manifest is missing for "
            f"wave={wave}, alias={ALIAS}: {manifest_path}"
        )
    staged_root = (wave_root / "steering_plans").resolve()
    CURRENT.update({"phase": "steering_artifact_staging", "construct": "all"})
    status_file()
    run(
        [
            RUNTIME_PYTHON,
            str(CHECKOUT / "scripts/stage_model_steering_preflight.py"),
            "--manifest",
            manifest_path,
            "--output-root",
            staged_root,
        ],
        wave_root / "steering_artifact_staging.log",
        cwd=CHECKOUT,
    )
    # Discovery reads the process environment, while child commands inherit
    # the explicit ENV mapping used by ``run``.  Set both so this remains
    # deterministic when the runner is exercised from a test harness.
    os.environ[STEERING_PLAN_ROOT_ENV] = str(staged_root)
    ENV[STEERING_PLAN_ROOT_ENV] = str(staged_root)
    return staged_root


def run_wave(wave: int, config, inventory, selection, gate, specs, wave_root):
    _require_model_runtime()
    sys.path.insert(0, str(CHECKOUT / "scripts"))
    sys.path.insert(0, str(CHECKOUT / "src"))
    from construct_benchmark.config import load_construct_specs, load_run_config
    from run_prompt_only_behavior import ResidualSteeringGenerator, execute_prompt_only_behavior

    loaded_config = load_run_config(config)
    loaded_specs = load_construct_specs(specs)
    selection_data = json.loads(selection.read_text(encoding="utf-8"))
    behavior_total = sum(len(selection_data["selected"][cid]["behavior_eval"]["prompt_ids"]) for cid in selection_data["construct_ids"])
    collateral_total = sum(len(selection_data["selected"][cid]["collateral_eval"]["prompt_ids"]) for cid in selection_data["construct_ids"])
    baseline_total = behavior_total + collateral_total
    CURRENT.update({"wave": wave, "construct": "all", "phase": "configuration_and_tokenizer", "completed_rows": 0, "total_rows": baseline_total})
    status_file()
    for path in [config, inventory, selection, gate, CHECKOUT / "configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json", *specs]:
        if not path.is_file():
            raise RunnerFailure("missing registered artifact: " + str(path))
    spec_args = []
    for spec in specs:
        spec_args.extend(["--construct-spec", str(spec)])
    analysis = CHECKOUT / "configs/construct_benchmark/analysis_specs/rsc_benchmark_core_v1.json"
    run(
        [
            RUNTIME_PYTHON,
            str(CHECKOUT / "scripts/preflight_benchmark_run.py"),
            "--run-config",
            config,
            *spec_args,
            "--analysis-spec",
            analysis,
            "--prompts",
            inventory,
            "--run-mode",
            "test",
            "--require-model",
            "--require-gpu",
            "--require-persistent-workspace",
        ],
        wave_root / "configuration_preflight.json",
    )
    run(
        [
            RUNTIME_PYTHON,
            str(CHECKOUT / "scripts/preflight_tokenizer.py"),
            "--run-config",
            config,
            *spec_args,
            "--prompts",
            inventory,
            "--output",
            wave_root / "tokenizer_preflight.json",
            "--prompt-format",
            "chat",
            "--max-length",
            "1024",
        ],
        wave_root / "tokenizer_stdout.json",
    )
    cache = {}

    def factory(*args, **kwargs):
        nonlocal cache
        if "generator" not in cache:
            cache["generator"] = ResidualSteeringGenerator(*args, **kwargs)
        return cache["generator"]

    completed = 0
    gate_outputs = [
        wave_root / "behavior.jsonl",
        wave_root / "behavior.jsonl.manifest.json",
        wave_root / "collateral.jsonl",
        wave_root / "collateral.jsonl.manifest.json",
    ]
    for split, output in (
        ("behavior_eval", wave_root / "behavior.jsonl"),
        ("collateral_eval", wave_root / "collateral.jsonl"),
    ):
        CURRENT.update({"phase": split + "_generation", "construct": "all", "completed_rows": completed, "total_rows": baseline_total})
        status_file()
        try:
            result = execute_prompt_only_behavior(
                run_config=loaded_config,
                construct_specs=loaded_specs,
                prompt_inventory=inventory,
                output=output,
                mode="test",
                prompt_format="chat",
                enable_thinking=False,
                max_new_tokens=32,
                min_new_tokens=1,
                max_length=1024,
                device="auto",
                dtype="auto",
                device_map="auto",
                local_files_only=False,
                trust_remote_code=False,
                prompt_split=split,
                constrained_numeric_generation=True,
                preflight_selection=selection_data,
                generator_factory=factory,
            )
        except Exception as exc:
            # Preserve an incomplete manifest/raw prefix too.  This makes a
            # model-load or generation failure recoverable and distinguishes it
            # from a completed output that fails a scientific gate.
            diagnostic = baseline_gate(
                config,
                selection,
                gate,
                specs,
                wave_root / "behavior.jsonl",
                wave_root / "collateral.jsonl",
                wave_root / "baseline_collateral_gate.json",
                upstream_error=f"{type(exc).__name__}: {exc}",
            )
            preserved = _preserve_gate_artifacts(wave_root, gate_outputs, wave=wave)
            diagnostic["preserved_artifacts"] = preserved
            Path(wave_root / "baseline_collateral_gate.json").write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _preserve_gate_artifacts(
                wave_root,
                [wave_root / "baseline_collateral_gate.json"],
                wave=wave,
            )
            _stream_gate_report(diagnostic)
            raise
        completed += result["completed_records"]
        CURRENT.update({"completed_rows": completed})
        status_file()
    del cache
    gc.collect()
    CURRENT.update({"phase": "baseline_collateral_gate", "construct": "all", "completed_rows": completed, "total_rows": baseline_total})
    status_file()
    report = baseline_gate(
        config,
        selection,
        gate,
        specs,
        wave_root / "behavior.jsonl",
        wave_root / "collateral.jsonl",
        wave_root / "baseline_collateral_gate.json",
    )
    preserved = _preserve_gate_artifacts(
        wave_root,
        gate_outputs,
        wave=wave,
    )
    report["preserved_artifacts"] = preserved
    Path(wave_root / "baseline_collateral_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _preserve_gate_artifacts(
        wave_root,
        [wave_root / "baseline_collateral_gate.json"],
        wave=wave,
    )
    # Emit only after the report and its copied state artifacts are durable.
    _stream_gate_report(report)
    if not report.get("pass"):
        failed = [
            f"{entry.get('construct_id')}:{entry.get('stage')}"
            for entry in report.get("failure_matrix", [])
            if isinstance(entry, dict) and entry.get("pass") is False
        ]
        raise RunnerFailure(
            "baseline/collateral gate failed for wave "
            + str(wave)
            + "; diagnostics="
            + str(wave_root / "baseline_collateral_gate.json")
            + "; failed_stages="
            + ",".join(failed[:12])
        )
    CURRENT.update({"phase": "steering_plan_discovery", "construct": "all"})
    status_file()
    stage_steering_artifacts(wave, wave_root)
    discovery = discover_plans(config, specs, wave_root / "plan_discovery.json")
    outputs = []
    for spec_path, loaded_spec in zip(specs, loaded_specs.values()):
        construct_id = loaded_spec.construct_id
        CURRENT.update({"phase": "steering_preparation", "construct": construct_id, "completed_rows": completed, "total_rows": baseline_total})
        status_file()
        source = Path(discovery["selected"][construct_id]["path"])
        derived = wave_root / "plans" / f"{construct_id}_preflight.json"
        raw_output = wave_root / "steering" / f"{construct_id}.jsonl"
        score_dir = wave_root / "steering" / f"{construct_id}_score"
        derived.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                RUNTIME_PYTHON,
                str(CHECKOUT / "scripts/prepare_model_steering_preflight.py"),
                "--steering-plan",
                source,
                "--selection-manifest",
                selection,
                "--prompt-inventory",
                inventory,
                "--output",
                derived,
            ],
            wave_root / "steering" / f"{construct_id}_prepare.log",
        )
        CURRENT.update({"phase": "steering_generation", "construct": construct_id})
        status_file()
        run(
            [
                RUNTIME_PYTHON,
                str(CHECKOUT / "scripts/run_construct_steering.py"),
                "--steering-plan",
                derived,
                "--prompt-inventory",
                inventory,
                "--output",
                raw_output,
                "--device",
                "auto",
                "--device-map",
                "auto",
                "--dtype",
                "auto",
                "--prompt-format",
                "chat",
                "--disable-thinking",
                "--max-new-tokens",
                "32",
                "--min-new-tokens",
                "1",
                "--max-length",
                "1024",
            ],
            wave_root / "steering" / f"{construct_id}_generation.log",
        )
        CURRENT.update({"phase": "steering_scoring", "construct": construct_id})
        status_file()
        run(
            [
                RUNTIME_PYTHON,
                str(CHECKOUT / "scripts/score_construct_steering.py"),
                "--raw-generations",
                raw_output,
                "--construct-spec",
                spec_path,
                "--output-dir",
                score_dir,
                "--bootstrap-resamples",
                "200",
            ],
            wave_root / "steering" / f"{construct_id}_scoring.log",
        )
        outputs.append(f"{construct_id}={raw_output}")
    CURRENT.update({"phase": "final_preflight_validation", "construct": "all"})
    status_file()
    validator = [
        RUNTIME_PYTHON,
        str(CHECKOUT / "scripts/validate_model_behavior_accessibility_preflight.py"),
        "--selection-manifest",
        selection,
    ]
    for spec in specs:
        validator.extend(["--construct-spec", spec])
    validator.extend(["--behavior-output", wave_root / "behavior.jsonl", "--collateral-output", wave_root / "collateral.jsonl"])
    for value in outputs:
        validator.extend(["--steering-output", value])
    validator.extend(["--gate-config", gate, "--output", wave_root / "preflight_validation.json", "--overwrite"])
    run(validator, wave_root / "final_validation_stdout.json")
    CURRENT.update({"phase": "wave_complete", "construct": "all", "completed_rows": completed, "total_rows": baseline_total})
    status_file()


def main():
    code = 0
    try:
        _ensure_runtime()
        verify_release()
        for wave in (1, 2, 3, 4):
            config, inventory, selection, gate, specs = wave_definition(wave)
            run_wave(wave, config, inventory, selection, gate, specs, RUN_ROOT / f"wave{wave}")
    except Exception as exc:
        code = 1
        error = f"{type(exc).__name__}: {exc}"
        (STATE / "error.json").write_text(
            json.dumps({"error": error, "wave": CURRENT["wave"], "phase": CURRENT["phase"]}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status_file("FAILED", error)
    else:
        status_file("COMPLETED")
    write_hashes()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
