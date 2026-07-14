"""Rat / session-date filename prefixing, shared across the pipeline.

Every generated file is prefixed with ``<rat>_<YYYYMMDD_HHMMSS>_`` derived from the
Trodes recording name, so outputs from different rats/sessions never collide and
are self-describing. E.g. ``Rat6_HM_Neurons_20260707_091045_pre`` ->
``Rat6_20260707_091045_``.

Readers use :func:`find_output` so they locate a file whether or not it carries a
prefix (backward compatible with old unprefixed outputs).
"""

from __future__ import annotations

import re
from pathlib import Path

# rat token (letters+digits, e.g. Rat6 / rat10) ... Trodes datetime YYYYMMDD_HHMMSS
_SESSION_RE = re.compile(r"(?P<rat>[A-Za-z]+\d+).*?(?P<dt>\d{8}_\d{6})")


def session_prefix(name) -> str:
    """Return ``'Rat6_20260707_091045_'`` from a recording name, or ``''`` if the
    rat/datetime can't be parsed (callers then fall back to no prefix)."""
    if not name:
        return ""
    m = _SESSION_RE.search(str(name))
    if not m:
        return ""
    return f"{m.group('rat')}_{m.group('dt')}_"


def find_output(folder, suffix):
    """Find ``<folder>/<prefix?><suffix>`` — prefixed or not. Returns a Path or None.

    Prefers an exact ``<folder>/<suffix>`` match, else the (newest) ``*<suffix>``.
    """
    folder = Path(folder)
    exact = folder / suffix
    if exact.is_file():
        return exact
    matches = sorted(folder.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None
