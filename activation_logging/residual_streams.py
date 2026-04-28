"""Compatibility wrapper for `interpretability.residual_streams`."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from interpretability.residual_streams import *  # noqa: F401,F403
