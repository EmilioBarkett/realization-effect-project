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
            max_vectors=int(data["max_vectors"]) if data.get("max_vectors") is not None else None,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "activation_runs": [str(path) for path in self.activation_runs],
            "layers": list(self.layers),
            "token_regions": list(self.token_regions) if self.token_regions is not None else None,
            "activation_site": self.activation_site,
            "max_vectors": self.max_vectors,
        }

