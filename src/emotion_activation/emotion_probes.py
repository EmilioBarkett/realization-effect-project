from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EMOTION_CONFIG = Path("configs/emotion_activation/emotions_initial.json")


@dataclass(frozen=True)
class EmotionProbeRecord:
    prompt_id: str
    prompt_text: str
    metadata: dict[str, Any]


def _require_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Emotion config entry is missing non-empty string field '{key}'.")
    return value.strip()


def load_emotion_probe_records(path: Path = DEFAULT_EMOTION_CONFIG) -> list[EmotionProbeRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    template = data.get("template")
    if not isinstance(template, str) or "{scenario}" not in template:
        raise ValueError(f"{path} must define a template containing '{{scenario}}'.")

    emotions = data.get("emotions")
    if not isinstance(emotions, list) or not emotions:
        raise ValueError(f"{path} must define a non-empty emotions list.")

    records: list[EmotionProbeRecord] = []
    for row in emotions:
        if not isinstance(row, dict):
            raise ValueError("Each emotion config entry must be an object.")
        emotion = _require_string(row, "emotion")
        cluster = _require_string(row, "cluster")
        expected_behavior_effect = _require_string(row, "expected_behavior_effect")
        for contrast_role, scenario_key in (
            ("positive", "positive_prompt"),
            ("control", "control_prompt"),
        ):
            scenario = _require_string(row, scenario_key)
            prompt_id = f"{emotion}__{contrast_role}"
            records.append(
                EmotionProbeRecord(
                    prompt_id=prompt_id,
                    prompt_text=template.format(scenario=scenario),
                    metadata={
                        "prompt_family": data.get("name", "emotion_probe"),
                        "emotion": emotion,
                        "emotion_cluster": cluster,
                        "contrast_role": contrast_role,
                        "expected_behavior_effect": expected_behavior_effect,
                        "source_config": str(path),
                    },
                )
            )

    return records


def write_emotion_probe_csv(records: Iterable[EmotionProbeRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "prompt_text",
        "emotion",
        "emotion_cluster",
        "contrast_role",
        "expected_behavior_effect",
        "prompt_family",
        "source_config",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "prompt_id": record.prompt_id,
                    "prompt_text": record.prompt_text,
                    "emotion": record.metadata["emotion"],
                    "emotion_cluster": record.metadata["emotion_cluster"],
                    "contrast_role": record.metadata["contrast_role"],
                    "expected_behavior_effect": record.metadata["expected_behavior_effect"],
                    "prompt_family": record.metadata["prompt_family"],
                    "source_config": record.metadata["source_config"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export emotion contrast prompts for activation extraction.")
    parser.add_argument("--config", default=str(DEFAULT_EMOTION_CONFIG))
    parser.add_argument("--output", default="experiments/emotion_activation/prompts/initial_emotion_contrasts.csv")
    args = parser.parse_args()

    records = load_emotion_probe_records(Path(args.config))
    write_emotion_probe_csv(records, Path(args.output))
    print(f"wrote {len(records)} emotion probe prompts to {args.output}")


if __name__ == "__main__":
    main()
