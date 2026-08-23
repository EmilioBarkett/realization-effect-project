#!/usr/bin/env python3
"""Audit or repair parsed realization-effect result CSV columns."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from realization_effect.reparse_results import main


if __name__ == "__main__":
    main()
