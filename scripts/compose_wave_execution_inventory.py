#!/usr/bin/env python3
"""Compose frozen vector-plus-downstream inventories for waves 2--4.

This command performs no API or model calls.  It prepares the input artifact
for a four-construct wave run and keeps the result explicitly non-confirmatory
until the campaign release gate is satisfied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct_benchmark.campaign import compose_wave_prompt_inventory  # noqa: E402

try:  # direct CLI execution has ``scripts/`` on sys.path
    from scripts.audit_wave_prompt_inventories import audit_wave_inventory  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - defensive direct-import path
    from audit_wave_prompt_inventories import audit_wave_inventory  # type: ignore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose frozen vector-plus-downstream prompt inventories for waves 2-4."
    )
    parser.add_argument("--waves", nargs="+", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=_ROOT / "configs/construct_benchmark/construct_registry_v1.json",
    )
    parser.add_argument(
        "--vector-prompts",
        type=Path,
        default=_ROOT / "results/benchmark/vector_prompts_v2_luna/full_final_all16/combined.csv",
    )
    parser.add_argument(
        "--vector-manifest",
        type=Path,
        default=_ROOT / "results/benchmark/vector_prompts_v2_luna/full_final_all16/final_inventory_manifest.json",
    )
    parser.add_argument(
        "--downstream-prompts",
        type=Path,
        default=_ROOT / "results/benchmark/downstream_prompts_v1_waves2_4_full_luna_b20_o30k_v1/combined.csv",
    )
    parser.add_argument(
        "--downstream-manifest",
        type=Path,
        default=_ROOT / "results/benchmark/downstream_prompts_v1_waves2_4_full_luna_b20_o30k_v1/final_inventory_manifest.json",
    )
    parser.add_argument(
        "--quality-gate",
        type=Path,
        default=_ROOT / "configs/construct_benchmark/quality_gates/waves2_4_downstream_luna_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_ROOT / "results/benchmark/prompt_inventories",
    )
    parser.add_argument(
        "--output-suffix",
        default="four_construct_full_luna_v1",
        help=(
            "Directory suffix under --output-root for each wave. Use a new "
            "versioned suffix for repaired prompt artifacts."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.output_suffix.strip():
        raise SystemExit("--output-suffix must be non-empty.")
    summaries = []
    for wave in dict.fromkeys(args.waves):
        output_dir = args.output_root / f"wave{wave}_{args.output_suffix.strip()}"
        summary = compose_wave_prompt_inventory(
                wave=wave,
                registry_path=args.registry,
                vector_prompt_path=args.vector_prompts,
                vector_manifest_path=args.vector_manifest,
                downstream_prompt_path=args.downstream_prompts,
                downstream_manifest_path=args.downstream_manifest,
                quality_gate_path=args.quality_gate,
                output_dir=output_dir,
            )
        audit = audit_wave_inventory(
            output_dir / "combined.csv",
            registry_path=args.registry,
            wave=wave,
        )
        summary["prompt_audit"] = {
            key: value
            for key, value in audit.items()
            if key not in {"severe_flags", "warnings"}
        }
        manifest_path = output_dir / "inventory_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["prompt_audit"] = summary["prompt_audit"]
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(summary)
    print(
        json.dumps(
            {
                "status": "prepared",
                "confirmatory": False,
                "waves": summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
