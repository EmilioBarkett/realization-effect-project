#!/usr/bin/env python3
"""Compare project-trained and external SAE feature associations on one dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sae.config import SAEDatasetConfig
from sae.dataset import iter_activation_vectors
from sae.external import ExternalSAEConfig, load_gemma_scope_jumprelu_model
from sae.inspection import (
    DEFAULT_FIELDS,
    association_rows,
    load_normalization,
    load_prompt_texts,
    normalize_batch,
    resolve_device,
    top_examples,
    write_csv,
)
from sae.model import load_sae_model


def collect_prompt_mean_vectors(
    dataset_config: SAEDatasetConfig,
) -> tuple[list[str], list[dict[str, Any]], np.ndarray, list[int]]:
    prompt_id_to_index: dict[str, int] = {}
    prompt_ids: list[str] = []
    prompt_metadata: list[dict[str, Any]] = []
    sums: list[np.ndarray] = []
    counts: list[int] = []

    for run_dir in dataset_config.activation_runs:
        for record in iter_activation_vectors(
            run_dir,
            layers=set(dataset_config.layers),
            token_regions=set(dataset_config.token_regions) if dataset_config.token_regions else None,
            prompt_metadata_filters=(
                {
                    key: set(value)
                    for key, value in dataset_config.prompt_metadata_filters.items()
                }
                if dataset_config.prompt_metadata_filters
                else None
            ),
            activation_site=dataset_config.activation_site,
            max_vectors=dataset_config.max_vectors,
        ):
            prompt_id = str(record.metadata.get("prompt_id") or "")
            if not prompt_id:
                continue
            prompt_index = prompt_id_to_index.get(prompt_id)
            if prompt_index is None:
                prompt_index = len(prompt_ids)
                prompt_id_to_index[prompt_id] = prompt_index
                prompt_ids.append(prompt_id)
                prompt_metadata.append(dict(record.metadata.get("prompt_metadata", {})))
                sums.append(np.zeros_like(record.vector, dtype=np.float32))
                counts.append(0)
            sums[prompt_index] += np.asarray(record.vector, dtype=np.float32)
            counts[prompt_index] += 1

    if not sums:
        raise ValueError("No activation vectors were collected.")

    vectors = np.stack([total / max(count, 1) for total, count in zip(sums, counts, strict=True)])
    return prompt_ids, prompt_metadata, vectors.astype(np.float32, copy=False), counts


def encode_project_sae(
    *,
    checkpoint_path: Path,
    normalization_stats_path: Path,
    vectors: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    resolved_device = resolve_device(torch, device)
    model = load_sae_model(checkpoint_path, map_location=resolved_device).to(resolved_device)
    model.eval()
    method, mean, scale = load_normalization(normalization_stats_path)

    rows: list[np.ndarray] = []
    for start in range(0, len(vectors), batch_size):
        batch = normalize_batch(vectors[start : start + batch_size], method=method, mean=mean, scale=scale)
        with torch.inference_mode():
            inputs = torch.as_tensor(batch, dtype=torch.float32, device=resolved_device)
            rows.append(model.encode(inputs).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(rows, axis=0)


def encode_external_sae(
    *,
    config_path: Path,
    vectors: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[ExternalSAEConfig, np.ndarray]:
    import torch

    resolved_device = resolve_device(torch, device)
    config = ExternalSAEConfig.from_json(config_path)
    model = load_gemma_scope_jumprelu_model(config, device=resolved_device).to(resolved_device)
    model.eval()

    rows: list[np.ndarray] = []
    for start in range(0, len(vectors), batch_size):
        with torch.inference_mode():
            inputs = torch.as_tensor(vectors[start : start + batch_size], dtype=torch.float32, device=resolved_device)
            rows.append(model.encode(inputs).detach().cpu().numpy().astype(np.float32, copy=False))
    return config, np.concatenate(rows, axis=0)


def top_association_by_group(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["field"]), str(row["value"]))
        if key not in best or abs(float(row["effect_size"])) > abs(float(best[key]["effect_size"])):
            best[key] = row
    return best


def write_comparison_csv(
    path: Path,
    *,
    project_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    project_best = top_association_by_group(project_rows)
    external_best = top_association_by_group(external_rows)
    keys = sorted(set(project_best) | set(external_best))
    rows: list[dict[str, Any]] = []
    for field, value in keys:
        project = project_best.get((field, value), {})
        external = external_best.get((field, value), {})
        rows.append(
            {
                "field": field,
                "value": value,
                "project_feature_id": project.get("feature_id", ""),
                "project_effect_size": project.get("effect_size", ""),
                "project_mean_difference": project.get("mean_difference", ""),
                "external_feature_id": external.get("feature_id", ""),
                "external_effect_size": external.get("effect_size", ""),
                "external_mean_difference": external.get("mean_difference", ""),
            }
        )
    write_csv(
        path,
        rows,
        [
            "field",
            "value",
            "project_feature_id",
            "project_effect_size",
            "project_mean_difference",
            "external_feature_id",
            "external_effect_size",
            "external_mean_difference",
        ],
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare project and external SAE baselines.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--project-checkpoint", required=True)
    parser.add_argument("--project-normalization-stats", required=True)
    parser.add_argument("--external-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--examples-per-feature", type=int, default=8)
    args = parser.parse_args()

    dataset_config = SAEDatasetConfig.from_json(args.dataset_config)
    prompt_ids, prompt_metadata, vectors, token_counts = collect_prompt_mean_vectors(dataset_config)
    prompt_texts = load_prompt_texts(dataset_config)

    project_features = encode_project_sae(
        checkpoint_path=Path(args.project_checkpoint),
        normalization_stats_path=Path(args.project_normalization_stats),
        vectors=vectors,
        batch_size=args.batch_size,
        device=args.device,
    )
    external_config, external_features = encode_external_sae(
        config_path=Path(args.external_config),
        vectors=vectors,
        batch_size=args.batch_size,
        device=args.device,
    )

    project_rows = association_rows(
        project_features,
        prompt_metadata,
        fields=DEFAULT_FIELDS,
        min_group_size=args.min_group_size,
        top_n=args.top_n,
    )
    external_rows = association_rows(
        external_features,
        prompt_metadata,
        fields=DEFAULT_FIELDS,
        min_group_size=args.min_group_size,
        top_n=args.top_n,
    )
    project_examples = top_examples(
        project_features,
        prompt_ids,
        prompt_metadata,
        prompt_texts,
        project_rows,
        top_features=25,
        examples_per_feature=args.examples_per_feature,
    )
    external_examples = top_examples(
        external_features,
        prompt_ids,
        prompt_metadata,
        prompt_texts,
        external_rows,
        top_features=25,
        examples_per_feature=args.examples_per_feature,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "project_feature_associations.csv",
        project_rows,
        [
            "field",
            "value",
            "rank",
            "feature_id",
            "group_count",
            "rest_count",
            "mean_in_group",
            "mean_out_group",
            "mean_difference",
            "effect_size",
        ],
    )
    write_csv(
        output_dir / "external_feature_associations.csv",
        external_rows,
        [
            "field",
            "value",
            "rank",
            "feature_id",
            "group_count",
            "rest_count",
            "mean_in_group",
            "mean_out_group",
            "mean_difference",
            "effect_size",
        ],
    )
    example_fields = [
        "feature_id",
        "rank",
        "activation",
        "prompt_id",
        "concept_axis",
        "expected_feature",
        "emotion",
        "risk_orientation",
        "domain",
        "realization_frame",
        "outcome_valence",
        "control_type",
        "prompt_text",
    ]
    write_csv(output_dir / "project_top_feature_examples.csv", project_examples, example_fields)
    write_csv(output_dir / "external_top_feature_examples.csv", external_examples, example_fields)
    comparison_rows = write_comparison_csv(
        output_dir / "association_comparison.csv",
        project_rows=project_rows,
        external_rows=external_rows,
    )

    summary = {
        "dataset_config": str(args.dataset_config),
        "aggregation": "prompt_mean_over_selected_token_vectors",
        "prompt_count": len(prompt_ids),
        "vector_count": int(sum(token_counts)),
        "mean_tokens_per_prompt": float(np.mean(token_counts)),
        "project_checkpoint": str(args.project_checkpoint),
        "project_normalization_stats": str(args.project_normalization_stats),
        "project_d_sae": int(project_features.shape[1]),
        "external_config": str(args.external_config),
        "external_repo_id": external_config.repo_id,
        "external_hook_point": external_config.hook_point,
        "external_project_logger_layer": external_config.project_logger_layer,
        "external_d_sae": int(external_features.shape[1]),
        "project_top_associations": project_rows[:25],
        "external_top_associations": external_rows[:25],
        "comparison_preview": comparison_rows[:25],
        "outputs": {
            "project_feature_associations": str(output_dir / "project_feature_associations.csv"),
            "external_feature_associations": str(output_dir / "external_feature_associations.csv"),
            "association_comparison": str(output_dir / "association_comparison.csv"),
            "project_top_feature_examples": str(output_dir / "project_top_feature_examples.csv"),
            "external_top_feature_examples": str(output_dir / "external_top_feature_examples.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
