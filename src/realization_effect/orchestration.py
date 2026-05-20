"""Experiment orchestration for realization-effect behavioral runs."""

from __future__ import annotations

import csv
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realization_effect.api import call_model, create_openrouter_client
from realization_effect.blocks import (
    RESULT_FIELDS,
    _normalize_temperature,
    build_block_output_path,
    default_grouped_output_path,
    load_resume_state,
    reconcile_merged_outputs,
)
from realization_effect.conditions import load_conditions
from realization_effect.parsing import parse_response
from realization_effect.prompts import build_prompt


def run_experiment(
    conditions_path: Path,
    output_path: Path,
    n_trials: int,
    model: str,
    temperature: float,
    target_model: str | None = None,
    target_temperature: float | None = None,
    target_prompt_version: str | None = None,
    sleep_seconds: float = 0.0,
    prompt_version: str = "absolute",
    shuffle: bool = False,
    max_retries: int = 5,
    seed: int | None = None,
) -> None:
    """Execute one experiment block and append each row immediately to disk."""
    if n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    if max_retries < 0:
        raise ValueError("max_retries must be zero or a positive integer")
    if not math.isfinite(temperature) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a finite number between 0 and 2")
    if target_temperature is not None and (
        not math.isfinite(target_temperature) or target_temperature < 0 or target_temperature > 2
    ):
        raise ValueError("target_temperature must be a finite number between 0 and 2")

    conditions = load_conditions(conditions_path)
    rng = random.Random(seed) if seed is not None else random.Random()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_runs, max_trial_id = load_resume_state(output_path)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    temp_key = _normalize_temperature(temperature)
    target_model_value = target_model.strip() if target_model else model
    target_temperature_value = temperature if target_temperature is None else target_temperature
    target_temp_key = _normalize_temperature(target_temperature_value)
    target_prompt_version_value = target_prompt_version.strip() if target_prompt_version else prompt_version

    if (
        target_model is not None
        or target_temperature is not None
        or target_prompt_version is not None
    ) and (
        model != target_model_value
        or temp_key != target_temp_key
        or prompt_version != target_prompt_version_value
    ):
        raise ValueError(
            "Target filters must match --model, --temperature, and --prompt-version "
            "for safe resume of that experiment block."
        )

    completed_by_condition: dict[str, set[int]] = {
        str(condition["condition"]): set() for condition in conditions
    }
    for condition_name, run_number, row_model, row_temp_key, row_prompt_version in completed_runs:
        if (
            row_model == target_model_value
            and row_temp_key == target_temp_key
            and row_prompt_version == target_prompt_version_value
            and condition_name in completed_by_condition
            and 1 <= run_number <= n_trials
        ):
            completed_by_condition[condition_name].add(run_number)

    missing_by_condition: dict[str, list[int]] = {}
    print(
        "Resume summary for block: "
        f"model={target_model_value}, temperature={target_temperature_value}, "
        f"prompt_version={target_prompt_version_value}"
    )
    for condition in conditions:
        condition_name = str(condition["condition"])
        completed_numbers = completed_by_condition[condition_name]
        missing_run_numbers = [
            run_number for run_number in range(1, n_trials + 1) if run_number not in completed_numbers
        ]
        missing_by_condition[condition_name] = missing_run_numbers

        missing_display = "none" if not missing_run_numbers else str(missing_run_numbers)
        print(
            f"  {condition_name}: {len(completed_numbers)} unique completed; "
            f"missing run_numbers={missing_display}"
        )

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_api_key:
        raise EnvironmentError(
            "Missing OPENROUTER_API_KEY in environment. "
            "Set it in your shell before running this script."
        )

    client = create_openrouter_client(openrouter_api_key)
    trial_id = max_trial_id
    total_new = 0

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

        trial_plan: list[tuple[dict[str, Any], int]] = []
        for condition in conditions:
            condition_name = str(condition["condition"])
            for run_number in missing_by_condition[condition_name]:
                trial_plan.append((condition, run_number))

        if shuffle:
            rng.shuffle(trial_plan)

        for condition, run_number in trial_plan:
            condition_name = str(condition["condition"])
            outcome_type = str(condition["outcome_type"])
            amount = int(condition["amount"])

            run_key = (condition_name, run_number, model, temp_key, prompt_version)
            if run_key in completed_runs:
                continue

            prompt_text = build_prompt(
                outcome_type=outcome_type,
                amount=amount,
                prompt_version=prompt_version,
            )
            response_text, request_id = call_model(
                client=client,
                prompt=prompt_text,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
            )
            parsed_wager, parsed_risk_profile, valid, valid_risk_profile, refusal_flag, parse_error_type = (
                parse_response(response_text)
            )
            log_wager = math.log(parsed_wager) if valid and parsed_wager is not None else ""

            trial_id += 1
            row = {
                "trial_id": trial_id,
                "condition": condition_name,
                "outcome_type": outcome_type,
                "amount": amount,
                "run_number": run_number,
                "model": model,
                "temperature": temperature,
                "prompt_version": prompt_version,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "parsed_wager": parsed_wager if parsed_wager is not None else "",
                "log_wager": log_wager,
                "valid": valid,
                "risk_profile": parsed_risk_profile if parsed_risk_profile is not None else "",
                "valid_risk_profile": valid_risk_profile,
                "refusal_flag": refusal_flag,
                "parse_error_type": parse_error_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
            }

            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

            completed_runs.add(run_key)
            total_new += 1

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"Completed. Added {total_new} new trial rows to {output_path}.")


def run_experiment_grid(
    conditions_path: Path,
    output_path: Path,
    n_trials: int,
    models: list[str],
    temperatures: list[float],
    target_model: str | None = None,
    target_temperature: float | None = None,
    target_prompt_version: str | None = None,
    sleep_seconds: float = 0.0,
    prompt_version: str = "absolute",
    prompt_versions: list[str] | None = None,
    shuffle: bool = False,
    max_retries: int = 5,
    seed: int | None = None,
    max_workers: int = 1,
    round_trials_step: int = 1,
) -> None:
    """Run all model/temperature/prompt_version blocks, optionally in parallel."""
    if not models:
        raise ValueError("models must contain at least one model")
    if not temperatures:
        raise ValueError("temperatures must contain at least one temperature")
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    if round_trials_step <= 0:
        raise ValueError("round_trials_step must be a positive integer")

    versions = prompt_versions if prompt_versions is not None else [prompt_version]
    cleaned_versions = [version.strip() for version in versions if version and version.strip()]
    if not cleaned_versions:
        raise ValueError("prompt_versions must contain at least one non-empty value")

    target_model_value = target_model.strip() if target_model else None
    target_prompt_version_value = target_prompt_version.strip() if target_prompt_version else None
    target_temp_key = _normalize_temperature(target_temperature)

    block_specs: list[tuple[str, float, str, Path]] = []
    for model in models:
        for temperature in temperatures:
            for block_prompt_version in cleaned_versions:
                if target_model_value is not None and model != target_model_value:
                    continue
                if target_temperature is not None and _normalize_temperature(temperature) != target_temp_key:
                    continue
                if target_prompt_version_value is not None and block_prompt_version != target_prompt_version_value:
                    continue

                block_output_path = build_block_output_path(
                    output_path=output_path,
                    model=model,
                    temperature=temperature,
                    prompt_version=block_prompt_version,
                )
                block_specs.append((model, temperature, block_prompt_version, block_output_path))

    deduped_block_specs: list[tuple[str, float, str, Path]] = []
    seen_block_paths: set[str] = set()
    for block_spec in block_specs:
        block_path_key = str(block_spec[3])
        if block_path_key in seen_block_paths:
            continue
        seen_block_paths.add(block_path_key)
        deduped_block_specs.append(block_spec)
    block_specs = deduped_block_specs

    block_specs.sort(
        key=lambda block: (block[0], _normalize_temperature(block[1]), block[2], block[3].name)
    )
    if not block_specs:
        print("No experiment blocks matched the requested filters.")
        return

    print(f"Running {len(block_specs)} block(s) with max_workers={min(max_workers, len(block_specs))}.")
    print(
        "Write process: each block appends to its own CSV under "
        f"{output_path.parent / 'blocks'}, then all blocks are reconciled once "
        f"into {output_path} at the end."
    )

    def _run_single_block(block_spec: tuple[str, float, str, Path]) -> None:
        block_model, block_temperature, block_prompt_version, block_output_path = block_spec
        print(
            "Starting block: "
            f"model={block_model}, temperature={block_temperature}, "
            f"prompt_version={block_prompt_version}, output={block_output_path}"
        )
        run_experiment(
            conditions_path=conditions_path,
            output_path=block_output_path,
            n_trials=n_trials,
            model=block_model,
            temperature=block_temperature,
            target_model=target_model,
            target_temperature=target_temperature,
            target_prompt_version=target_prompt_version,
            sleep_seconds=sleep_seconds,
            prompt_version=block_prompt_version,
            shuffle=shuffle,
            max_retries=max_retries,
            seed=seed,
        )

    failed_blocks: list[tuple[tuple[str, float, str, Path], Exception]] = []
    if len(block_specs) == 1 or max_workers == 1:
        for block_spec in block_specs:
            try:
                _run_single_block(block_spec)
            except Exception as error:
                failed_blocks.append((block_spec, error))
                break
    else:
        worker_count = min(max_workers, len(block_specs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_block = {
                executor.submit(_run_single_block, block_spec): block_spec
                for block_spec in block_specs
            }
            for future in as_completed(future_to_block):
                block_spec = future_to_block[future]
                try:
                    future.result()
                except Exception as error:
                    failed_blocks.append((block_spec, error))

    grouped_output_path = default_grouped_output_path(output_path)
    merged_block_count, merged_rows = reconcile_merged_outputs(
        output_path=output_path,
        grouped_output_path=grouped_output_path,
        round_trials_step=round_trials_step,
    )
    if merged_block_count == 0:
        blocks_dir = output_path.parent / "blocks"
        print(f"No block CSV files found in {blocks_dir}; skipping merge.")
    else:
        print(
            "Final reconciliation complete: "
            f"merged {merged_block_count} block file(s) into {output_path} "
            f"with {merged_rows} total row(s), and refreshed {grouped_output_path}."
        )

    if failed_blocks:
        summary_lines = []
        for block_spec, error in failed_blocks:
            block_model, block_temperature, block_prompt_version, _ = block_spec
            summary_lines.append(
                "  - "
                f"model={block_model}, temperature={block_temperature}, "
                f"prompt_version={block_prompt_version}: {error}"
            )
        raise RuntimeError(
            "One or more experiment blocks failed:\n" + "\n".join(summary_lines)
        ) from failed_blocks[0][1]
