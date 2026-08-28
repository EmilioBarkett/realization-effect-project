#!/usr/bin/env python3
"""Validate benchmark configuration and runtime prerequisites without side effects.

This command never contacts OpenRouter, Hugging Face, RunPod, or an archive
provider. It only validates local configuration files and reports whether the
requested environment variables, Python packages, GPU, and archive tooling are
available. Use the ``--require-*`` flags to turn an advisory check into a
blocking check for a particular stage of the run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import (  # noqa: E402
    load_analysis_spec,
    load_construct_specs,
    load_run_config,
    validate_analysis_spec,
    validate_run_constructs,
)
from construct_benchmark.manifests import build_run_plan  # noqa: E402
from construct_benchmark.prompts import load_prompt_records  # noqa: E402
from construct_benchmark.storage import resolve_archive_uri  # noqa: E402


MODEL_PLACEHOLDERS = frozenset({"REPLACE_WITH_LOCAL_MODEL", "REPLACE_WITH_MODEL_ID"})


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _check(name: str, condition: bool, detail: str, *, required: bool) -> Check:
    if condition:
        return Check(name, "pass", detail)
    return Check(name, "fail" if required else "warn", detail)


def _configured_model(run_config) -> tuple[bool, str]:
    model_id = str(run_config.model.get("model_id", "")).strip()
    tokenizer_id = str(run_config.model.get("tokenizer_id", "")).strip()
    if not model_id or model_id in MODEL_PLACEHOLDERS:
        return False, "run config still contains a model placeholder"
    if tokenizer_id in MODEL_PLACEHOLDERS:
        return False, "run config still contains a tokenizer placeholder"
    if run_config.model.get("revision") is None:
        return True, "model and tokenizer are configured; revision is not pinned"
    return True, "model, tokenizer, and revision are configured"


def _package_available(name: str) -> tuple[bool, str]:
    available = importlib.util.find_spec(name) is not None
    return available, f"{name} {'is installed' if available else 'is not installed'}"


def _gpu_available() -> tuple[bool, str]:
    if importlib.util.find_spec("torch") is None:
        return False, "torch is not installed"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        return False, f"torch could not be imported ({type(exc).__name__})"
    if not torch.cuda.is_available():
        return False, "torch is installed but CUDA is unavailable"
    count = int(torch.cuda.device_count())
    names = [str(torch.cuda.get_device_name(index)) for index in range(count)]
    return True, f"CUDA is available on {count} device(s): {', '.join(names)}"


def _workspace_check(run_config, env: Mapping[str, str], *, required: bool) -> Check:
    env_name = str(run_config.storage["workspace_root_env"])
    value = env.get(env_name, "").strip()
    if not value:
        return _check(
            "persistent_workspace",
            False,
            f"{env_name} is not set; storage will fall back to the current directory",
            required=required,
        )
    path = Path(value).expanduser()
    if not path.exists():
        return _check(
            "persistent_workspace",
            False,
            f"{env_name} points to a missing directory",
            required=required,
        )
    if not path.is_dir() or not os.access(path, os.W_OK):
        return _check(
            "persistent_workspace",
            False,
            f"{env_name} does not point to a writable directory",
            required=required,
        )
    return Check("persistent_workspace", "pass", f"{env_name} points to a writable directory")


def build_preflight_report(
    *,
    run_config_path: Path,
    construct_spec_paths: list[Path] | None = None,
    analysis_spec_path: Path | None = None,
    prompt_inventory_path: Path | None = None,
    run_mode: str | None = None,
    env: Mapping[str, str] | None = None,
    require_model: bool = False,
    require_openai: bool = False,
    require_openrouter: bool = False,
    require_gpu: bool = False,
    require_archive: bool = False,
    require_runpod_api: bool = False,
    require_legacy_runpod_api: bool = False,
    require_persistent_workspace: bool = False,
) -> dict[str, object]:
    """Build a credential-safe preflight report without making network calls."""

    environment = dict(os.environ if env is None else env)
    run_config = load_run_config(run_config_path)
    checks: list[Check] = []

    model_ok, model_detail = _configured_model(run_config)
    checks.append(_check("model_configuration", model_ok, model_detail, required=require_model))

    if construct_spec_paths or analysis_spec_path or prompt_inventory_path:
        if not construct_spec_paths or analysis_spec_path is None:
            raise ValueError(
                "--construct-spec and --analysis-spec must be supplied together when validating the run plan."
            )
        construct_specs = load_construct_specs(construct_spec_paths)
        validate_run_constructs(run_config, construct_specs)
        analysis_spec = load_analysis_spec(analysis_spec_path)
        validate_analysis_spec(run_config, analysis_spec)
        prompt_count = None
        if prompt_inventory_path is not None:
            prompt_records = load_prompt_records(prompt_inventory_path)
            build_run_plan(
                run_config,
                construct_specs,
                analysis_spec,
                prompt_inventory_path=prompt_inventory_path,
                prompt_records=prompt_records,
                run_mode=run_mode,
            )
            prompt_count = len(prompt_records)
        detail = f"{len(construct_specs)} construct spec(s) and analysis spec agree with the run config"
        if prompt_count is not None:
            detail += f"; {prompt_count} prompt record(s) validated"
        checks.append(Check("configuration_cross_validation", "pass", detail))
    else:
        checks.append(
            Check(
                "configuration_cross_validation",
                "warn",
                "construct and analysis specs were not supplied; only the run config was checked",
            )
        )

    checks.append(_workspace_check(run_config, environment, required=require_persistent_workspace))

    openai_env = environment.get("OPENAI_API_KEY", "").strip()
    checks.append(
        _check(
            "openai_api_key",
            bool(openai_env),
            "OPENAI_API_KEY is present" if openai_env else "OPENAI_API_KEY is not set",
            required=require_openai,
        )
    )
    # Retain this check for the legacy activation-analysis generator and for
    # historical reproduction. The active construct-benchmark workflow does
    # not require it.
    openrouter_env = environment.get("OPENROUTER_API_KEY", "").strip()
    checks.append(
        _check(
            "openrouter_api_key",
            bool(openrouter_env),
            "OPENROUTER_API_KEY is present" if openrouter_env else "OPENROUTER_API_KEY is not set",
            required=require_openrouter,
        )
    )
    runpod_env = environment.get("RUNPOD_2_API_KEY", "").strip()
    checks.append(
        _check(
            "runpod_2_api_key",
            bool(runpod_env),
            "RUNPOD_2_API_KEY is present" if runpod_env else "RUNPOD_2_API_KEY is not set",
            required=require_runpod_api,
        )
    )
    if require_legacy_runpod_api:
        legacy_env = environment.get("RUNPOD_API_KEY", "").strip()
        checks.append(
            _check(
                "legacy_runpod_api_key",
                bool(legacy_env),
                "legacy RUNPOD_API_KEY is present" if legacy_env else "legacy RUNPOD_API_KEY is not set",
                required=True,
            )
        )

    for package_name in ("torch", "transformers"):
        available, detail = _package_available(package_name)
        checks.append(
            _check(
                f"python_{package_name}",
                available,
                detail,
                required=require_gpu,
            )
        )
    gpu_ok, gpu_detail = _gpu_available()
    checks.append(_check("cuda_gpu", gpu_ok, gpu_detail, required=require_gpu))

    archive_uri = resolve_archive_uri(run_config, environment.get(str(run_config.storage["archive_uri_env"])))
    checks.append(
        _check(
            "archive_configuration",
            archive_uri is not None,
            "archive URI is configured" if archive_uri else "no archive URI is configured",
            required=require_archive,
        )
    )
    sync_tool = str(run_config.storage["sync_tool"])
    sync_available = shutil.which(sync_tool) is not None
    checks.append(
        _check(
            "archive_sync_tool",
            sync_available,
            f"{sync_tool} is available" if sync_available else f"{sync_tool} is not installed or not on PATH",
            required=require_archive,
        )
    )

    report = {
        "run_id": run_config.run_id,
        "construct_ids": list(run_config.construct_ids),
        "model_id": run_config.model.get("model_id"),
        "checks": [asdict(item) for item in checks],
    }
    report["ready"] = not any(item.status == "fail" for item in checks)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check benchmark configuration and runtime prerequisites without making network calls."
    )
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--construct-spec", action="append", type=Path, default=None)
    parser.add_argument("--analysis-spec", type=Path, default=None)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--run-mode", choices=("test", "full"), default=None)
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--require-openai", action="store_true")
    parser.add_argument(
        "--require-openrouter",
        action="store_true",
        help="Require the legacy OpenRouter key for historical activation-analysis generation.",
    )
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-archive", action="store_true")
    parser.add_argument("--require-runpod-api", action="store_true")
    parser.add_argument(
        "--require-legacy-runpod-api",
        action="store_true",
        help="Require the legacy RUNPOD_API_KEY only for an explicitly historical check.",
    )
    parser.add_argument("--require-persistent-workspace", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = build_preflight_report(
            run_config_path=args.run_config,
            construct_spec_paths=args.construct_spec,
            analysis_spec_path=args.analysis_spec,
            prompt_inventory_path=args.prompts,
            run_mode=args.run_mode,
            require_model=args.require_model,
            require_openai=args.require_openai,
            require_openrouter=args.require_openrouter,
            require_gpu=args.require_gpu,
            require_archive=args.require_archive,
            require_runpod_api=args.require_runpod_api,
            require_legacy_runpod_api=args.require_legacy_runpod_api,
            require_persistent_workspace=args.require_persistent_workspace,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
