#!/usr/bin/env python3
"""Inspect trained SAE feature activations over an activation dataset."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sae.config import SAEDatasetConfig
from sae.dataset import iter_activation_vectors
from sae.model import load_sae_model


DEFAULT_FIELDS = (
    "concept_axis",
    "expected_feature",
    "emotion",
    "risk_orientation",
    "domain",
    "realization_frame",
    "outcome_valence",
    "behavior_target",
    "casino_context",
    "control_type",
    "source_llm",
)


def _resolve_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_normalization(path: Path) -> tuple[str, np.ndarray, float]:
    stats = np.load(path)
    method = str(stats["method"].item())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    scale = float(stats["scale"].item())
    return method, mean, scale


def _normalize(batch: np.ndarray, *, method: str, mean: np.ndarray, scale: float) -> np.ndarray:
    if method == "none":
        return batch.astype(np.float32, copy=False)
    if method != "mean_center_global_norm":
        raise ValueError(f"Unsupported normalization method: {method}")
    return ((batch - mean) / scale).astype(np.float32, copy=False)


def _metadata_value(metadata: dict[str, Any], field: str) -> str:
    value = metadata.get(field, "")
    if value is None:
        return ""
    return str(value).strip()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_prompt_texts(dataset_config: SAEDatasetConfig) -> dict[str, str]:
    prompt_texts: dict[str, str] = {}
    for run_dir in dataset_config.activation_runs:
        prompt_path = Path(run_dir) / "prompts.jsonl"
        if not prompt_path.exists():
            continue
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt_id = str(row.get("prompt_id") or "")
                if prompt_id:
                    prompt_texts[prompt_id] = str(row.get("prompt_text") or "")
    return prompt_texts


def _association_rows(
    prompt_features: np.ndarray,
    prompt_metadata: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    min_group_size: int,
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt_count = prompt_features.shape[0]
    for field in fields:
        values = sorted({_metadata_value(metadata, field) for metadata in prompt_metadata})
        for value in values:
            if not value:
                continue
            mask = np.array(
                [_metadata_value(metadata, field) == value for metadata in prompt_metadata],
                dtype=bool,
            )
            group_count = int(mask.sum())
            rest_count = prompt_count - group_count
            if group_count < min_group_size or rest_count < min_group_size:
                continue

            group = prompt_features[mask]
            rest = prompt_features[~mask]
            mean_in = group.mean(axis=0)
            mean_out = rest.mean(axis=0)
            diff = mean_in - mean_out
            pooled = np.sqrt((group.var(axis=0) + rest.var(axis=0)) / 2.0) + 1e-12
            effect = diff / pooled
            top_features = np.argsort(effect)[-top_n:][::-1]
            for rank, feature_id in enumerate(top_features, start=1):
                rows.append(
                    {
                        "field": field,
                        "value": value,
                        "rank": rank,
                        "feature_id": int(feature_id),
                        "group_count": group_count,
                        "rest_count": rest_count,
                        "mean_in_group": float(mean_in[feature_id]),
                        "mean_out_group": float(mean_out[feature_id]),
                        "mean_difference": float(diff[feature_id]),
                        "effect_size": float(effect[feature_id]),
                    }
                )
    rows.sort(key=lambda row: abs(float(row["effect_size"])), reverse=True)
    return rows


def _top_examples(
    prompt_features: np.ndarray,
    prompt_ids: list[str],
    prompt_metadata: list[dict[str, Any]],
    prompt_texts: dict[str, str],
    associations: list[dict[str, Any]],
    *,
    top_features: int,
    examples_per_feature: int,
) -> list[dict[str, Any]]:
    feature_ids: list[int] = []
    for row in associations:
        feature_id = int(row["feature_id"])
        if feature_id not in feature_ids:
            feature_ids.append(feature_id)
        if len(feature_ids) >= top_features:
            break

    rows: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        scores = prompt_features[:, feature_id]
        top_prompt_indices = np.argsort(scores)[-examples_per_feature:][::-1]
        for rank, prompt_index in enumerate(top_prompt_indices, start=1):
            metadata = prompt_metadata[int(prompt_index)]
            rows.append(
                {
                    "feature_id": feature_id,
                    "rank": rank,
                    "activation": float(scores[int(prompt_index)]),
                    "prompt_id": prompt_ids[int(prompt_index)],
                    "concept_axis": _metadata_value(metadata, "concept_axis"),
                    "expected_feature": _metadata_value(metadata, "expected_feature"),
                    "emotion": _metadata_value(metadata, "emotion"),
                    "risk_orientation": _metadata_value(metadata, "risk_orientation"),
                    "domain": _metadata_value(metadata, "domain"),
                    "realization_frame": _metadata_value(metadata, "realization_frame"),
                    "outcome_valence": _metadata_value(metadata, "outcome_valence"),
                    "control_type": _metadata_value(metadata, "control_type"),
                    "prompt_text": prompt_texts.get(prompt_ids[int(prompt_index)], ""),
                }
            )
    return rows


def inspect_sae_features(
    *,
    dataset_config: SAEDatasetConfig,
    checkpoint_path: Path,
    normalization_stats_path: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
    min_group_size: int,
    top_n: int,
    examples_per_feature: int,
) -> dict[str, Any]:
    import torch

    resolved_device = _resolve_device(torch, device)
    model = load_sae_model(checkpoint_path, map_location=resolved_device).to(resolved_device)
    model.eval()
    method, mean, scale = _load_normalization(normalization_stats_path)
    prompt_texts = _load_prompt_texts(dataset_config)

    prompt_id_to_index: dict[str, int] = {}
    prompt_ids: list[str] = []
    prompt_metadata: list[dict[str, Any]] = []
    prompt_features: list[np.ndarray] = []
    prompt_token_counts: list[int] = []
    vector_count = 0

    batch_vectors: list[np.ndarray] = []
    batch_metadata: list[dict[str, Any]] = []
    d_sae = int(model.config.d_sae)

    def flush_batch() -> None:
        nonlocal vector_count
        if not batch_vectors:
            return
        raw_batch = np.stack(batch_vectors, axis=0).astype(np.float32, copy=False)
        normalized = _normalize(raw_batch, method=method, mean=mean, scale=scale)
        with torch.inference_mode():
            inputs = torch.as_tensor(normalized, dtype=torch.float32, device=resolved_device)
            activations = model.encode(inputs).detach().cpu().numpy()
        for activation_row, metadata in zip(activations, batch_metadata, strict=True):
            prompt_id = str(metadata.get("prompt_id") or "")
            if not prompt_id:
                continue
            prompt_index = prompt_id_to_index.get(prompt_id)
            if prompt_index is None:
                prompt_index = len(prompt_ids)
                prompt_id_to_index[prompt_id] = prompt_index
                prompt_ids.append(prompt_id)
                prompt_metadata.append(dict(metadata.get("prompt_metadata", {})))
                prompt_features.append(np.zeros(d_sae, dtype=np.float32))
                prompt_token_counts.append(0)
            np.maximum(prompt_features[prompt_index], activation_row.astype(np.float32, copy=False), out=prompt_features[prompt_index])
            prompt_token_counts[prompt_index] += 1
            vector_count += 1
        batch_vectors.clear()
        batch_metadata.clear()

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
            batch_vectors.append(np.asarray(record.vector, dtype=np.float32))
            batch_metadata.append(record.metadata)
            if len(batch_vectors) >= batch_size:
                flush_batch()
    flush_batch()

    if not prompt_features:
        raise ValueError("No feature activations were collected.")

    feature_matrix = np.stack(prompt_features, axis=0)
    active_prompt_features = (feature_matrix > 0).sum(axis=1)
    associations = _association_rows(
        feature_matrix,
        prompt_metadata,
        fields=DEFAULT_FIELDS,
        min_group_size=min_group_size,
        top_n=top_n,
    )
    examples = _top_examples(
        feature_matrix,
        prompt_ids,
        prompt_metadata,
        prompt_texts,
        associations,
        top_features=25,
        examples_per_feature=examples_per_feature,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    association_path = output_dir / "feature_associations.csv"
    examples_path = output_dir / "top_feature_examples.csv"
    summary_path = output_dir / "summary.json"
    _write_csv(
        association_path,
        associations,
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
    _write_csv(
        examples_path,
        examples,
        [
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
        ],
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "normalization_stats_path": str(normalization_stats_path),
        "prompt_count": len(prompt_ids),
        "vector_count": vector_count,
        "d_sae": d_sae,
        "mean_tokens_per_prompt": float(np.mean(prompt_token_counts)),
        "mean_active_prompt_features": float(np.mean(active_prompt_features)),
        "median_active_prompt_features": float(np.median(active_prompt_features)),
        "max_active_prompt_features": int(np.max(active_prompt_features)),
        "top_associations": associations[:25],
        "association_csv": str(association_path),
        "top_examples_csv": str(examples_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect trained SAE feature associations.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--examples-per-feature", type=int, default=8)
    args = parser.parse_args()

    summary = inspect_sae_features(
        dataset_config=SAEDatasetConfig.from_json(args.dataset_config),
        checkpoint_path=Path(args.checkpoint),
        normalization_stats_path=Path(args.normalization_stats),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        device=args.device,
        min_group_size=args.min_group_size,
        top_n=args.top_n,
        examples_per_feature=args.examples_per_feature,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
