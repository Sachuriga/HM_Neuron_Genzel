#!/usr/bin/env bash
# ============================================================
#   HM Tracker — macOS / Linux launcher.
#
#   Thin wrapper: all pipeline logic lives in runner.py (the single
#   cross-platform source of truth, shared with runner_windows.bat).
#   Edit the menu / steps THERE, not here.
#
#   Usage:  ./scripts/runner_unix.sh /path/to/data_root
# ============================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
    PY="${PYTHON:-python3}"
elif command -v python >/dev/null 2>&1; then
    PY="${PYTHON:-python}"
else
    echo "[ERROR] No python interpreter found on PATH."
    exit 1
fi

exec "$PY" "$SCRIPT_DIR/../runner.py" "$@"
