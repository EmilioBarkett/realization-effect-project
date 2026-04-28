"""Compatibility wrapper for `interpretability.log_residuals`."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from interpretability.log_residuals import *  # noqa: F401,F403
from interpretability.log_residuals import main


if __name__ == "__main__":
    main()
