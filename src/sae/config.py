from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SAEDatasetConfig:
    """Configuration for turning residual-stream runs into SAE inputs."""

    activation_runs: tuple[Path, ...]
    layers: tuple[int, ...]
    token_regions: tuple[str, ...] | None = None
    activation_site: str | None = "resid_post"
    prompt_metadata_filters: dict[str, tuple[str, ...]] | None = None
    max_vectors: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "SAEDatasetConfig":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(
            activation_runs=tuple(Path(value) for value in data["activation_runs"]),
            layers=tuple(int(value) for value in data["layers"]),
            token_regions=tuple(data["token_regions"]) if data.get("token_regions") else None,
            activation_site=data.get("activation_site", "resid_post"),
            prompt_metadata_filters=(
                {
                    str(key): tuple(str(item) for item in value)
                    for key, value in data["prompt_metadata_filters"].items()
                }
                if data.get("prompt_metadata_filters")
                else None
            ),
            max_vectors=int(data["max_vectors"]) if data.get("max_vectors") is not None else None,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "activation_runs": [str(path) for path in self.activation_runs],
            "layers": list(self.layers),
            "token_regions": list(self.token_regions) if self.token_regions is not None else None,
            "activation_site": self.activation_site,
            "prompt_metadata_filters": (
                {key: list(value) for key, value in self.prompt_metadata_filters.items()}
                if self.prompt_metadata_filters is not None
                else None
            ),
            "max_vectors": self.max_vectors,
        }


@dataclass(frozen=True)
class SAEModelConfig:
    """Architecture settings for a single-layer sparse autoencoder."""

    d_in: int
    d_sae: int
    activation: str = "topk"
    top_k: int | None = None
    decoder_bias: bool = True

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "SAEModelConfig":
        return cls(
            d_in=int(data["d_in"]),
            d_sae=int(data["d_sae"]),
            activation=str(data.get("activation", "topk")),
            top_k=int(data["top_k"]) if data.get("top_k") is not None else None,
            decoder_bias=bool(data.get("decoder_bias", True)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation": self.activation,
            "top_k": self.top_k,
            "decoder_bias": self.decoder_bias,
        }


@dataclass(frozen=True)
class SAETrainingConfig:
    """Training-loop settings that are independent of activation extraction."""

    expansion_factor: int = 8
    d_sae: int | None = None
    activation: str = "topk"
    top_k: int | None = 32
    l1_coefficient: float = 0.0
    learning_rate: float = 3e-4
    batch_size: int = 256
    max_steps: int = 1_000
    seed: int = 0
    device: str = "auto"
    normalization: str = "mean_center_global_norm"
    normalization_eps: float = 1e-8
    output_dir: Path = Path("results/final/sae")

    @classmethod
    def from_json(cls, path: str | Path) -> "SAETrainingConfig":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.from_json_dict(data)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "SAETrainingConfig":
        return cls(
            expansion_factor=int(data.get("expansion_factor", 8)),
            d_sae=int(data["d_sae"]) if data.get("d_sae") is not None else None,
            activation=str(data.get("activation", "topk")),
            top_k=int(data["top_k"]) if data.get("top_k") is not None else None,
            l1_coefficient=float(data.get("l1_coefficient", 0.0)),
            learning_rate=float(data.get("learning_rate", 3e-4)),
            batch_size=int(data.get("batch_size", 256)),
            max_steps=int(data.get("max_steps", 1_000)),
            seed=int(data.get("seed", 0)),
            device=str(data.get("device", "auto")),
            normalization=str(data.get("normalization", "mean_center_global_norm")),
            normalization_eps=float(data.get("normalization_eps", 1e-8)),
            output_dir=Path(data.get("output_dir", "results/final/sae")),
        )

    def model_config_for_input(self, d_in: int) -> SAEModelConfig:
        d_sae = self.d_sae if self.d_sae is not None else d_in * self.expansion_factor
        return SAEModelConfig(
            d_in=d_in,
            d_sae=d_sae,
            activation=self.activation,
            top_k=self.top_k,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "expansion_factor": self.expansion_factor,
            "d_sae": self.d_sae,
            "activation": self.activation,
            "top_k": self.top_k,
            "l1_coefficient": self.l1_coefficient,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "device": self.device,
            "normalization": self.normalization,
            "normalization_eps": self.normalization_eps,
            "output_dir": str(self.output_dir),
        }
