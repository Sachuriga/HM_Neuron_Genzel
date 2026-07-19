"""Check how far each session has been through the preprocessing pipeline, by
looking at what the pipeline leaves behind in the ``HM_neuron_preprocess`` tree.

The runner (see runner.py / tracker_gui presets) runs a fixed set of steps per
animal type:

    implanted      1 e 2 3 4 5 6 7 8 d
    non-implanted        2 3 4 5 6   d

A step is judged **done** by the presence of the output file(s) it leaves in the
session's preprocess folder (``HM_neuron_preprocess/Rat<N>/<YYYYMMDD>/``), per the
signature table below. Intermediates that the pipeline cleans up (the stitched
video, the framewise-sync CSVs) don't survive, so steps 2 and 3 are taken as done
once the tracker output (step 4) exists — the tracker cannot run without them.
Compression (6) leaves no distinct filename and is reported as not-checked.

The signature table is the one thing worth tuning if a step's output name changes
— everything else is derived from it.
"""

from __future__ import annotations

import re
from pathlib import Path

# Steps each animal type is expected to run (from tracker_gui PRESETS).
PRESET = {True: "1e2345678d", False: "23456d"}

STEP_LABEL = {
    "1": "Trodes export", "e": "LFP export", "2": "Sync", "3": "Stitch",
    "4": "Tracker", "5": "Plotting", "6": "Compression", "7": "Sorting",
    "8": "LFP+Motion", "d": "DeepLabCut",
}

# glob(s) whose presence marks a step done. A folder pattern counts only when the
# folder exists AND holds at least one file (an empty LFP_Output is not "done").
STEP_SIGNATURES = {
    "1": ["*.DIO", "*.raw"],                 # Trodes DIO/raw/analog export
    "e": ["*.LFP"],                          # Trodes LFP export
    "4": ["*_Coordinates_Full.csv"],         # Tracker
    "5": ["*_analysis_final.pdf"],           # Plotting
    "7": ["*_sorting_output"],               # Spike sorting
    "8": ["LFP_Output"],                     # LFP + motion/EMG
    "d": ["collected_framesDLC_*.h5", "collected_framesDLC_*.csv"],  # DeepLabCut
}
# Intermediates cleaned up: done once the step that consumes them is done.
IMPLIED_BY = {"2": "4", "3": "4"}
# No distinct on-disk marker.
NOT_DETECTABLE = {"6"}

# status values
DONE = "done"
IMPLIED = "implied"        # inferred done from a downstream step's output
MISSING = "missing"
UNCHECKED = "unchecked"    # step has no detectable marker (compression)

_DATE_RE = re.compile(r"^\d{8}$")
_RAT_RE = re.compile(r"[Rr]at\s*_?(\d+)")


def _rat_of(name: str):
    m = _RAT_RE.search(name)
    return int(m.group(1)) if m else None


def _present(folder: Path, pattern: str) -> bool:
    """Does `folder` contain a match for `pattern`? A matched directory counts
    only if it holds at least one file (a created-but-empty folder is not done)."""
    try:
        for m in folder.glob(pattern):
            if m.is_file():
                return True
            if m.is_dir():
                try:
                    if any(f.is_file() for f in m.rglob("*")):
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


def _step_done(folder: Path, step: str) -> bool:
    return any(_present(folder, pat) for pat in STEP_SIGNATURES.get(step, []))


def check_session_steps(folder, implanted: bool) -> dict:
    """For one preprocess session folder, the status of every expected step:
    {step: done|implied|missing|unchecked}. Empty if `folder` is None."""
    if folder is None:
        return {}
    folder = Path(folder)
    steps = list(PRESET[implanted])
    # cache the detectable steps once
    done = {s: _step_done(folder, s) for s in steps if s in STEP_SIGNATURES}
    out = {}
    for s in steps:
        if s in STEP_SIGNATURES:
            out[s] = DONE if done[s] else MISSING
        elif s in IMPLIED_BY:
            out[s] = IMPLIED if done.get(IMPLIED_BY[s]) else MISSING
        elif s in NOT_DETECTABLE:
            out[s] = UNCHECKED
        else:
            out[s] = MISSING
    return out


def is_preprocess_tree(name: str) -> bool:
    n = name.lower()
    return "preprocess" in n


def find_preprocess_sessions(roots, max_depth: int = 7,
                             on_dir=None, should_stop=None) -> dict:
    """Map (rat_no, YYYYMMDD) -> [preprocess session folder] across the given
    roots, looking only inside ``*preprocess*`` trees. Mirrors the raw session
    walk but for the derived tree, which the raw walk deliberately skips."""
    idx: dict = {}
    for root in roots:
        base = Path(root)
        try:
            if not base.exists():
                continue
        except OSError:
            continue
        # only descend into preprocess trees: either root already is one, or find
        # the preprocess subfolders under it.
        starts = [base] if is_preprocess_tree(base.name) else []
        if not starts:
            try:
                starts = [c for c in base.iterdir()
                          if c.is_dir() and is_preprocess_tree(c.name)]
            except OSError:
                starts = []
        for start in starts:
            stack = [(start, 0)]
            while stack:
                if should_stop is not None and should_stop():
                    return idx
                d, depth = stack.pop()
                if on_dir is not None:
                    on_dir(d)
                try:
                    children = [c for c in d.iterdir() if c.is_dir()]
                except OSError:
                    continue
                for c in children:
                    if _DATE_RE.match(c.name):
                        rat_no = _rat_of(d.name)
                        if rat_no is not None:
                            idx.setdefault((rat_no, c.name), []).append(c)
                        continue
                    if depth < max_depth and not c.name.startswith("."):
                        stack.append((c, depth + 1))
    return idx


def summarize_steps(status: dict) -> str:
    """One-line 'done / missing' summary of a step-status dict, in preset order."""
    if not status:
        return "no preprocess folder"
    done = [s for s, v in status.items() if v in (DONE, IMPLIED)]
    missing = [s for s, v in status.items() if v == MISSING]
    parts = []
    if done:
        parts.append("done " + "".join(done))
    if missing:
        parts.append("missing " + "".join(missing))
    return " · ".join(parts) if parts else "—"


def build_preprocess(roster: list[dict], pre_idx: dict) -> list[dict]:
    """Per roster session: which preprocessing steps are done, from the preprocess
    index. One row per session."""
    rows = []
    for e in sorted(roster, key=lambda x: (x["rat_no"], x["day"], x["session"])):
        if not e["date8"]:
            continue
        folders = pre_idx.get((e["rat_no"], e["date8"]), [])
        # merge status across copies: a step is done if done in ANY copy
        merged: dict = {}
        for f in folders:
            st = check_session_steps(f, e["implanted"])
            for s, v in st.items():
                cur = merged.get(s)
                order = {DONE: 3, IMPLIED: 2, UNCHECKED: 1, MISSING: 0, None: -1}
                if order[v] > order.get(cur):
                    merged[s] = v
        if not merged:
            merged = {s: MISSING for s in PRESET[e["implanted"]]}
        expected = list(PRESET[e["implanted"]])
        n_done = sum(1 for s in expected if merged.get(s) in (DONE, IMPLIED))
        n_check = sum(1 for s in expected if merged.get(s) != UNCHECKED)
        rows.append(dict(rat=e["rat"], rat_no=e["rat_no"], date8=e["date8"],
                         day=e["day"], session=e["session"], repeat=e.get("repeat"),
                         implanted=e["implanted"], expected=expected, status=merged,
                         n_done=n_done, n_expected=n_check,
                         folders=[str(f) for f in folders],
                         summary=summarize_steps(merged)))
    return rows
