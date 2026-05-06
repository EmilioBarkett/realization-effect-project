#!/usr/bin/env python3
"""Score prompt activations against a saved activation direction."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from activation_analysis.vector_analysis import collect_prompt_mean_activations, write_csv, write_json


def _parse_csv_set(value: str | None, *, as_int: bool = False):
    if not value:
        return None
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if as_int:
        return {int(part) for part in parts}
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an activation direction on prompt activations.")
    parser.add_argument("--activation-run", required=True)
    parser.add_argument("--direction", required=True, help="Path to .npy direction vector.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--token-regions", default="scenario")
    parser.add_argument("--activation-site", default="resid_post")
    args = parser.parse_args()

    direction = np.load(args.direction).astype(np.float32, copy=False)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0:
        raise SystemExit("Direction vector has zero norm.")
    unit_direction = direction / direction_norm

    activations = collect_prompt_mean_activations(
        args.activation_run,
        layers=_parse_csv_set(args.layers, as_int=True),
        token_regions=_parse_csv_set(args.token_regions),
        activation_site=args.activation_site,
    )
    rows = []
    for activation in activations:
        metadata = activation.metadata
        rows.append(
            {
                "prompt_id": activation.prompt_id,
                "projection": float(np.dot(activation.vector, unit_direction)),
                "token_count": activation.token_count,
                "split": metadata.get("split", ""),
                "pair_id": metadata.get("pair_id", ""),
                "pair_role": metadata.get("pair_role", ""),
                "domain": metadata.get("domain", ""),
                "realization_frame": metadata.get("realization_frame", ""),
                "outcome_valence": metadata.get("outcome_valence", ""),
                "amount_bucket": metadata.get("amount_bucket", ""),
                "risk_context": metadata.get("risk_context", ""),
                "behavior_target": metadata.get("behavior_target", ""),
                "source_llm": metadata.get("source_llm", ""),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "prompt_projections.csv",
        rows,
        [
            "prompt_id",
            "projection",
            "token_count",
            "split",
            "pair_id",
            "pair_role",
            "domain",
            "realization_frame",
            "outcome_valence",
            "amount_bucket",
            "risk_context",
            "behavior_target",
            "source_llm",
        ],
    )

    group_summary = []
    for field in ("split", "domain", "realization_frame", "pair_role", "outcome_valence"):
        values = sorted({str(row[field]) for row in rows if row[field]})
        for value in values:
            scores = [float(row["projection"]) for row in rows if row[field] == value]
            group_summary.append(
                {
                    "field": field,
                    "value": value,
                    "count": len(scores),
                    "mean_projection": float(np.mean(scores)),
                    "std_projection": float(np.std(scores)),
                }
            )
    write_csv(
        output_dir / "projection_group_summary.csv",
        group_summary,
        ["field", "value", "count", "mean_projection", "std_projection"],
    )
    write_json(
        output_dir / "summary.json",
        {
            "activation_run": args.activation_run,
            "direction": args.direction,
            "prompt_count": len(rows),
            "direction_norm": direction_norm,
        },
    )
    print(f"scored {len(rows)} prompts in {output_dir}")


if __name__ == "__main__":
    main()
