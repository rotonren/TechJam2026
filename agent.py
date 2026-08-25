import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from compasscart.agent import CompassCartAgent as Agent

__all__ = ["Agent"]
