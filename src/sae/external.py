from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalSAEConfig:
    """Metadata needed to load a pretrained external SAE baseline."""

    repo_id: str
    repo_path: str
    local_dir: Path
    config_file: Path
    params_file: Path
    target_model: str
    hook_point: str
    activation_site: str
    layer: int
    project_logger_layer: int | None
    architecture: str
    d_in: int
    d_sae: int
    comparison_activation_run: Path | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ExternalSAEConfig":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(
            repo_id=str(data["repo_id"]),
            repo_path=str(data["repo_path"]),
            local_dir=Path(data["local_dir"]),
            config_file=Path(data["config_file"]),
            params_file=Path(data["params_file"]),
            target_model=str(data["target_model"]),
            hook_point=str(data["hook_point"]),
            activation_site=str(data["activation_site"]),
            layer=int(data["layer"]),
            project_logger_layer=(
                int(data["project_logger_layer"])
                if data.get("project_logger_layer") is not None
                else None
            ),
            architecture=str(data["architecture"]),
            d_in=int(data["d_in"]),
            d_sae=int(data["d_sae"]),
            comparison_activation_run=(
                Path(data["comparison_activation_run"])
                if data.get("comparison_activation_run")
                else None
            ),
        )


def load_gemma_scope_jumprelu_model(config: ExternalSAEConfig, *, device: str = "cpu"):
    """Load a Gemma Scope JumpReLU SAE with a minimal PyTorch wrapper.

    The wrapper exposes `encode(x)`, where `x` is a `[batch, d_in]` residual
    activation tensor aligned to the SAE's hook point.
    """

    if config.architecture != "jump_relu":
        raise ValueError(f"Unsupported external SAE architecture: {config.architecture}")
    if not config.params_file.exists():
        raise FileNotFoundError(
            f"Missing external SAE weights: {config.params_file}. "
            "Download them with the command in external/archive/20260506_sae_first_pass/saes/README.md."
        )

    import torch
    from safetensors.torch import load_file

    params = load_file(str(config.params_file), device=device)

    class GemmaScopeJumpReLUSAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.register_buffer("w_enc", params["w_enc"])
            self.register_buffer("b_enc", params["b_enc"])
            self.register_buffer("threshold", params["threshold"])
            self.register_buffer("w_dec", params["w_dec"])
            self.register_buffer("b_dec", params["b_dec"])

        def encode(self, x):
            pre_acts = x @ self.w_enc + self.b_enc
            return torch.where(pre_acts > self.threshold, pre_acts, torch.zeros_like(pre_acts))

        def decode(self, features):
            return features @ self.w_dec + self.b_dec

        def forward(self, x):
            features = self.encode(x)
            return self.decode(features)

    return GemmaScopeJumpReLUSAE()
