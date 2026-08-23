from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_steering_classification_prompt_does_not_wrap_already_classified_prompt() -> None:
    build_module = _load_script_module("build_realization_classification_prompts.py")
    steer_module = _load_script_module("steer_realization_classification.py")

    base_prompt = (
        "Read the following short scenario.\n\n"
        "Scenario:\n"
        "The position is still open and the gain is unrealized.\n\n"
        "Answer now.\n"
        "Return only the two requested integers on separate lines, with no labels and no explanation.\n"
    )

    classification_prompt = build_module._classification_prompt(base_prompt)
    wrapped_prompt = steer_module._classification_prompt(classification_prompt)

    assert wrapped_prompt == classification_prompt.strip()
    assert wrapped_prompt.count("Task: Determine whether") == 1
