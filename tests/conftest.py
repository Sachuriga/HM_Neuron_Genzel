import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
for sub in ("tracker", "tools", "nwb"):
    p = str(SRC / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
