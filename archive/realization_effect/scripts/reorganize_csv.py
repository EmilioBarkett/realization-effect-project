#!/usr/bin/env python3
"""Reorganize realization-effect CSV outputs."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from realization_effect.reorganize_csv import main


if __name__ == "__main__":
    main()
