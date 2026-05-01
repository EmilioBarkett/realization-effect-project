#!/usr/bin/env python3
"""Validate a residual-stream activation run directory."""

from pathlib import Path
import argparse
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from emotion_activation.activation_store import load_activation_run, validate_activation_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate residual-stream activation run files.")
    parser.add_argument("run_dir", help="Path to a results/residual_streams/<run_name> directory.")
    args = parser.parse_args()

    run = load_activation_run(args.run_dir)
    errors = validate_activation_run(args.run_dir)
    if errors:
        print(f"invalid activation run: {run.path}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    layers = sorted({shard.layer for shard in run.shards})
    print(f"valid activation run: {run.path}")
    print(f"prompts: {len(run.prompts)}")
    print(f"shards: {len(run.shards)}")
    print(f"layers: {','.join(str(layer) for layer in layers)}")


if __name__ == "__main__":
    main()
