"""rat_sessiondate_ filename prefixing for the NWB plot/video steps.

Every generated plot/video/data file is prefixed with ``<rat>_<YYYYMMDD_HHMMSS>_``
so outputs from different rats/sessions never collide. The prefix is derived from
the session identity already present in the output folder (the ``.nwb`` file, the
``*_sorting_output`` folder, or the folder name).
"""

from __future__ import annotations

import re
from pathlib import Path

_SESSION_RE = re.compile(r"(?P<rat>[A-Za-z]+\d+).*?(?P<dt>\d{8}_\d{6})")


def _from_name(name: str) -> str:
    m = _SESSION_RE.search(str(name))
    return f"{m.group('rat')}_{m.group('dt')}_" if m else ""


def file_prefix(output_folder) -> str:
    """Derive ``'Rat6_20260707_091045_'`` from the session identity in
    ``output_folder`` (``.nwb`` name > ``*_sorting_output`` folder > folder name).
    Returns ``''`` if nothing parseable is found (callers fall back to no prefix)."""
    op = Path(output_folder)
    for nwb in sorted(op.glob("*.nwb")) + sorted(op.glob("**/*.nwb")):
        p = _from_name(nwb.stem)
        if p:
            return p
    for d in sorted(op.glob("*_sorting_output")):
        p = _from_name(d.name)
        if p:
            return p
    return _from_name(op.name)


def prefixed(output_folder, name: str) -> str:
    """Return ``name`` with the folder's session prefix prepended (idempotent)."""
    pfx = file_prefix(output_folder)
    return name if (not pfx or name.startswith(pfx)) else f"{pfx}{name}"
