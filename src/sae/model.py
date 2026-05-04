from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sae.config import SAEModelConfig


@dataclass(frozen=True)
class SAEForwardOutput:
    """Forward-pass outputs needed by training and feature inspection."""

    reconstruction: object
    feature_activations: object


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "SAE model code requires torch. Install the project with the 'interp' extra."
        ) from exc
    return torch


def build_sae_model(config: SAEModelConfig):
    """Construct a sparse autoencoder without importing torch at package import time."""

    torch = _import_torch()

    class SparseAutoencoder(torch.nn.Module):
        def __init__(self, model_config: SAEModelConfig) -> None:
            super().__init__()
            if model_config.d_in <= 0 or model_config.d_sae <= 0:
                raise ValueError("SAE dimensions must be positive.")
            if model_config.activation not in {"relu", "topk"}:
                raise ValueError("SAE activation must be 'relu' or 'topk'.")
            if model_config.activation == "topk":
                if model_config.top_k is None:
                    raise ValueError("top_k is required when activation='topk'.")
                if model_config.top_k <= 0 or model_config.top_k > model_config.d_sae:
                    raise ValueError("top_k must be between 1 and d_sae.")

            self.config = model_config
            self.encoder = torch.nn.Linear(model_config.d_in, model_config.d_sae)
            self.decoder = torch.nn.Linear(
                model_config.d_sae,
                model_config.d_in,
                bias=model_config.decoder_bias,
            )
            self._reset_parameters()

        def _reset_parameters(self) -> None:
            torch.nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
            torch.nn.init.zeros_(self.encoder.bias)
            torch.nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
            if self.decoder.bias is not None:
                torch.nn.init.zeros_(self.decoder.bias)

        def encode(self, x):
            pre_activations = self.encoder(x)
            if self.config.activation == "relu":
                return torch.nn.functional.relu(pre_activations)

            assert self.config.top_k is not None
            values, indices = torch.topk(pre_activations, k=self.config.top_k, dim=-1)
            values = torch.nn.functional.relu(values)
            activations = torch.zeros_like(pre_activations)
            return activations.scatter(dim=-1, index=indices, src=values)

        def decode(self, feature_activations):
            return self.decoder(feature_activations)

        def forward(self, x) -> SAEForwardOutput:
            feature_activations = self.encode(x)
            reconstruction = self.decode(feature_activations)
            return SAEForwardOutput(
                reconstruction=reconstruction,
                feature_activations=feature_activations,
            )

    return SparseAutoencoder(config)


def load_sae_model(checkpoint_path: str | Path, *, map_location: str = "cpu"):
    """Load a checkpoint written by `sae.training.train_sae`."""

    torch = _import_torch()
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_config = SAEModelConfig.from_json_dict(checkpoint["model_config"])
    model = build_sae_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
