from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from sae.dataset import iter_activation_vectors


@dataclass(frozen=True)
class PromptActivation:
    prompt_id: str
    metadata: dict[str, Any]
    vector: np.ndarray
    token_count: int


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def collect_prompt_mean_activations(
    run_dir: str | Path,
    *,
    layers: set[int] | None,
    token_regions: set[str] | None,
    activation_site: str = "resid_post",
) -> list[PromptActivation]:
    """Mean-pool selected token activations into one vector per prompt."""

    prompt_id_to_index: dict[str, int] = {}
    prompt_ids: list[str] = []
    prompt_metadata: list[dict[str, Any]] = []
    sums: list[np.ndarray] = []
    counts: list[int] = []

    for record in iter_activation_vectors(
        run_dir,
        layers=layers,
        token_regions=token_regions,
        activation_site=activation_site,
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

    return [
        PromptActivation(
            prompt_id=prompt_id,
            metadata=metadata,
            vector=(vector_sum / max(count, 1)).astype(np.float32, copy=False),
            token_count=count,
        )
        for prompt_id, metadata, vector_sum, count in zip(
            prompt_ids,
            prompt_metadata,
            sums,
            counts,
            strict=True,
        )
    ]


def build_pair_directions(
    activations: Iterable[PromptActivation],
    *,
    positive_role: str = "realized_closed",
    negative_role: str = "paper_open",
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """Build per-pair contrast vectors and their mean direction."""

    by_pair: dict[str, dict[str, PromptActivation]] = {}
    for activation in activations:
        pair_id = _metadata_value(activation.metadata, "pair_id")
        pair_role = _metadata_value(activation.metadata, "pair_role")
        if not pair_id or not pair_role:
            continue
        by_pair.setdefault(pair_id, {})[pair_role] = activation

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for pair_id in sorted(by_pair):
        pair = by_pair[pair_id]
        positive = pair.get(positive_role)
        negative = pair.get(negative_role)
        if positive is None or negative is None:
            continue
        direction = positive.vector - negative.vector
        vectors.append(direction.astype(np.float32, copy=False))
        metadata = positive.metadata
        rows.append(
            {
                "pair_id": pair_id,
                "positive_prompt_id": positive.prompt_id,
                "negative_prompt_id": negative.prompt_id,
                "positive_role": positive_role,
                "negative_role": negative_role,
                "domain": _metadata_value(metadata, "domain"),
                "split": _metadata_value(metadata, "split"),
                "outcome_valence": _metadata_value(metadata, "outcome_valence"),
                "amount_bucket": _metadata_value(metadata, "amount_bucket"),
                "risk_context": _metadata_value(metadata, "risk_context"),
                "behavior_target": _metadata_value(metadata, "behavior_target"),
                "direction_norm": float(np.linalg.norm(direction)),
            }
        )

    if not vectors:
        return rows, None
    return rows, np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32, copy=False)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
