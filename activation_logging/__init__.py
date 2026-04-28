"""Compatibility package for the moved interpretability modules."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from interpretability.residual_streams import BatchResiduals, ResidualStreamLogger

__all__ = ["BatchResiduals", "ResidualStreamLogger"]
