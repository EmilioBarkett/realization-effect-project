#!/usr/bin/env python3
"""Export emotion contrast prompts for emotion-activation experiments."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from emotion_activation.emotion_probes import main


if __name__ == "__main__":
    main()
