#!/usr/bin/env python3
"""Placeholder entrypoint for realization-direction steering experiments."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Steer model inference with a saved realization direction. "
            "This entrypoint is reserved until the intervention hook is implemented."
        )
    )
    parser.add_argument("--direction", required=True, help="Path to a saved .npy direction vector.")
    parser.add_argument("--layer", type=int, required=True, help="Layer where the direction should be injected.")
    parser.add_argument("--scale", type=float, default=1.0, help="Steering coefficient.")
    parser.parse_args()
    raise SystemExit(
        "Steering is not implemented yet. Next step: add a generation-time forward hook "
        "that injects scale * direction at the selected residual stream layer."
    )


if __name__ == "__main__":
    main()
