from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from sae.config import SAEDatasetConfig, SAEModelConfig, SAETrainingConfig
from sae.dataset import iter_activation_vectors
from sae.model import build_sae_model


@dataclass(frozen=True)
class SAETrainingResult:
    """Serializable summary of one SAE training run."""

    steps: int
    model_config: SAEModelConfig
    training_config: SAETrainingConfig
    final_metrics: dict[str, float]
    checkpoint_path: Path
    manifest_path: Path
    normalization_stats_path: Path


@dataclass(frozen=True)
class ActivationNormalizationStats:
    """Dataset statistics used to transform raw activations before SAE training."""

    method: str
    vector_count: int
    d_in: int
    mean: np.ndarray
    scale: float

    def transform(self, batch: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return batch.astype(np.float32, copy=False)
        return ((batch - self.mean) / self.scale).astype(np.float32, copy=False)

    def to_json_summary(self, stats_file: Path) -> dict[str, object]:
        return {
            "method": self.method,
            "vector_count": self.vector_count,
            "d_in": self.d_in,
            "scale": self.scale,
            "mean_file": str(stats_file),
            "mean_l2_norm": float(np.linalg.norm(self.mean)),
        }


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "SAE training requires torch. Install the project with the 'interp' extra."
        ) from exc
    return torch


def _resolve_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _iter_numpy_batches(
    dataset_config: SAEDatasetConfig,
    *,
    batch_size: int,
) -> Iterator[np.ndarray]:
    batch: list[np.ndarray] = []
    yielded = 0
    for run_dir in dataset_config.activation_runs:
        remaining = None
        if dataset_config.max_vectors is not None:
            remaining = dataset_config.max_vectors - yielded
            if remaining <= 0:
                break

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
            max_vectors=remaining,
        ):
            batch.append(np.asarray(record.vector, dtype=np.float32))
            yielded += 1
            if len(batch) == batch_size:
                yield np.stack(batch, axis=0)
                batch = []
            if dataset_config.max_vectors is not None and yielded >= dataset_config.max_vectors:
                break
    if batch:
        yield np.stack(batch, axis=0)


def _first_batch(dataset_config: SAEDatasetConfig, batch_size: int) -> np.ndarray:
    try:
        return next(_iter_numpy_batches(dataset_config, batch_size=batch_size))
    except StopIteration as exc:
        raise ValueError("SAE training dataset is empty.") from exc


def compute_activation_normalization_stats(
    dataset_config: SAEDatasetConfig,
    *,
    batch_size: int,
    method: str = "mean_center_global_norm",
    eps: float = 1e-8,
) -> ActivationNormalizationStats:
    """Compute centering/scaling stats from the exact SAE training slice."""

    if method not in {"none", "mean_center_global_norm"}:
        raise ValueError("normalization must be 'none' or 'mean_center_global_norm'.")

    vector_count = 0
    vector_sum: np.ndarray | None = None
    d_in: int | None = None
    for batch in _iter_numpy_batches(dataset_config, batch_size=batch_size):
        if d_in is None:
            d_in = int(batch.shape[1])
            vector_sum = np.zeros(d_in, dtype=np.float64)
        vector_sum += batch.astype(np.float64, copy=False).sum(axis=0)
        vector_count += int(batch.shape[0])

    if vector_count == 0 or d_in is None or vector_sum is None:
        raise ValueError("SAE training dataset is empty.")

    if method == "none":
        return ActivationNormalizationStats(
            method=method,
            vector_count=vector_count,
            d_in=d_in,
            mean=np.zeros(d_in, dtype=np.float32),
            scale=1.0,
        )

    mean = (vector_sum / vector_count).astype(np.float32)
    norm_sum = 0.0
    for batch in _iter_numpy_batches(dataset_config, batch_size=batch_size):
        centered = batch.astype(np.float32, copy=False) - mean
        norm_sum += float(np.linalg.norm(centered, axis=1).sum())
    scale = norm_sum / vector_count
    if scale <= eps:
        scale = 1.0

    return ActivationNormalizationStats(
        method=method,
        vector_count=vector_count,
        d_in=d_in,
        mean=mean,
        scale=float(scale),
    )


def _write_manifest(
    result: SAETrainingResult,
    dataset_config: SAEDatasetConfig,
    metrics_history: list[dict[str, float]],
    normalization_stats: ActivationNormalizationStats,
) -> None:
    manifest = {
        "schema_version": "0.1.0",
        "model_config": result.model_config.to_json_dict(),
        "training_config": result.training_config.to_json_dict(),
        "dataset_config": dataset_config.to_json_dict(),
        "steps": result.steps,
        "final_metrics": result.final_metrics,
        "metrics_history": metrics_history,
        "checkpoint_file": result.checkpoint_path.name,
        "normalization": normalization_stats.to_json_summary(result.normalization_stats_path),
    }
    result.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def train_sae(
    dataset_config: SAEDatasetConfig,
    training_config: SAETrainingConfig | None = None,
) -> SAETrainingResult:
    """Train a small local SAE over activation vectors.

    This is intentionally a first scaffold: it exercises the full path from
    activation-run vectors to a saved SAE checkpoint, while keeping the backend
    simple enough to replace once the project commits to a larger SAE library.
    """

    torch = _import_torch()
    training_config = training_config or SAETrainingConfig()
    if training_config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if training_config.max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    torch.manual_seed(training_config.seed)
    normalization_stats = compute_activation_normalization_stats(
        dataset_config,
        batch_size=training_config.batch_size,
        method=training_config.normalization,
        eps=training_config.normalization_eps,
    )
    model_config = training_config.model_config_for_input(normalization_stats.d_in)
    device = _resolve_device(torch, training_config.device)
    model = build_sae_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    output_dir = training_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "sae_checkpoint.pt"
    manifest_path = output_dir / "manifest.json"
    normalization_stats_path = output_dir / "normalization_stats.npz"
    np.savez_compressed(
        normalization_stats_path,
        method=np.array(normalization_stats.method),
        vector_count=np.array(normalization_stats.vector_count),
        d_in=np.array(normalization_stats.d_in),
        mean=normalization_stats.mean.astype(np.float32, copy=False),
        scale=np.array(normalization_stats.scale, dtype=np.float32),
    )

    step = 0
    final_metrics: dict[str, float] = {}
    metrics_history: list[dict[str, float]] = []
    while step < training_config.max_steps:
        made_progress = False
        for numpy_batch in _iter_numpy_batches(dataset_config, batch_size=training_config.batch_size):
            normalized_batch = normalization_stats.transform(numpy_batch)
            inputs = torch.as_tensor(normalized_batch, dtype=torch.float32, device=device)
            outputs = model(inputs)
            reconstruction_mse = torch.nn.functional.mse_loss(outputs.reconstruction, inputs)
            l1_loss = outputs.feature_activations.abs().mean()
            loss = reconstruction_mse + training_config.l1_coefficient * l1_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            active_mask = outputs.feature_activations.detach() > 0
            final_metrics = {
                "loss": float(loss.detach().cpu().item()),
                "reconstruction_mse": float(reconstruction_mse.detach().cpu().item()),
                "l1_loss": float(l1_loss.detach().cpu().item()),
                "mean_active_features": float(active_mask.sum(dim=1).float().mean().cpu().item()),
                "fraction_active_features": float(active_mask.float().mean().cpu().item()),
            }
            metrics_history.append({"step": float(step + 1), **final_metrics})
            step += 1
            made_progress = True
            if step >= training_config.max_steps:
                break
        if not made_progress:
            raise ValueError("SAE training dataset is empty.")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config.to_json_dict(),
        "training_config": training_config.to_json_dict(),
        "normalization": normalization_stats.to_json_summary(normalization_stats_path),
        "steps": step,
        "final_metrics": final_metrics,
    }
    torch.save(checkpoint, checkpoint_path)

    result = SAETrainingResult(
        steps=step,
        model_config=model_config,
        training_config=training_config,
        final_metrics=final_metrics,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        normalization_stats_path=normalization_stats_path,
    )
    _write_manifest(result, dataset_config, metrics_history, normalization_stats)
    return result
