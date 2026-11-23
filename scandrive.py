#!/usr/bin/env python3
"""`scandrive` command — opens the Drive Coverage Checker GUI.

The GUI itself lives in src/tools/scan_drive_gui.py. That folder is a plain
script folder (not a Python package), so it cannot be named directly as a
command in pyproject.toml. This small top-level module bridges the gap:

    pip install -e .      # once, creates the `scandrive` command
    scandrive             # opens the GUI

It also works without installing:  python scandrive.py
"""

import sys
from pathlib import Path

# Folder that holds scan_drive_gui.py and the helper modules it imports.
TOOLS_DIR = Path(__file__).resolve().parent / "src" / "tools"


def main():
    # Make `import scan_drive_gui` work no matter which folder we are started from.
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))

    # Imported here (not at the top) so the path above is set first.
    import scan_drive_gui

    scan_drive_gui.main()


if __name__ == "__main__":
    main()
