#!/usr/bin/env python3
"""Report whether a local or RunPod environment can execute model-side work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.config import load_run_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check activation/steering runtime dependencies.")
    parser.add_argument("--run-config", type=Path, required=True)
    args = parser.parse_args()
    run_config = load_run_config(args.run_config)
    report: dict[str, object] = {
        "model_id": run_config.model["model_id"],
        "tokenizer_id": run_config.model.get("tokenizer_id"),
        "model_configured": run_config.model["model_id"] != "REPLACE_WITH_LOCAL_MODEL",
    }
    try:
        import torch

        report.update(
            {
                "torch": True,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
    except Exception:
        report.update({"torch": False, "cuda_available": False, "cuda_device_count": 0})
    try:
        import transformers

        report.update({"transformers": True, "transformers_version": transformers.__version__})
    except Exception:
        report["transformers"] = False
    report["ready"] = bool(
        report["model_configured"]
        and report["torch"]
        and report["transformers"]
        and report["cuda_available"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
