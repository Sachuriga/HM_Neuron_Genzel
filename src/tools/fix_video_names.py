"""Detect camera videos whose name has the parts in the wrong order and rename
them to the canonical ``eye<NN>_<date>_<time>.mp4``.

The rig writes ``eye01_2019-05-27_13-45-32.mp4``. Some files came out with the
``eye`` part shoved to the end — ``2019-05-27_13-45-32_eye01.mp4`` — or without a
time. Those don't match ``_CAM_RE`` / ``_EYE_RE``, so scan/organize/match all miss
the whole folder. This tool finds them, works out the correct name (eye number
from the file, date+time from the file or, if the file lacks a time, from the
timestamp folder they sit in), and renames — never overwriting, always by plan
first so the old→new list can be checked before anything changes on disk.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import scan_drive as sd

# Already-correct name — left alone.
CANON_RE = re.compile(r"^eye\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mp4$", re.I)

# Pieces we pull out of a name wherever they sit, so any ordering / extra junk
# (eye at the end, a trailing '.mp4_fixed.mp4', a bare 'eye01.mp4', …) rebuilds
# to the canonical name. Date+time are matched as one block so the time is never
# a stray match inside the date (e.g. '20-05-13' out of '2020-05-13').
_EYE = re.compile(r"eye(\d+)", re.I)
_DATETIME = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# The timestamp a camera folder is named with, a date/time fallback for files
# that carry no time of their own.
_FOLDER_TS = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")

# A re-encoded repair carries a 'fixed' marker (e.g. ..._fixed.mp4). When one of
# these is present its broken original is set aside rather than overwritten.
_IS_FIXED = re.compile(r"fixed", re.I)
BROKEN_SUFFIX = ".broken"          # appended AFTER .mp4 so the file is no longer a
#                                    camera video and is never re-detected/renamed.

# Plan actions.
RENAME = "rename"
BROKEN = "broken"         # a broken original set aside so its fixed re-encode can take the name
CONFLICT = "conflict"     # target name already exists / two files want the same name
UNKNOWN = "unknown"       # looks like a camera video but no rule fits — left alone


def canonical_name(fname: str, folder: str) -> str | None:
    """The correct ``eye<NN>_<date>_<time>.mp4`` for a file, or None if it is
    already correct or cannot be worked out. The eye number and the date/time are
    extracted from wherever they appear in the name; a missing time (or date) is
    filled from the timestamp `folder` the file sits in."""
    if CANON_RE.match(fname):
        return None                                  # already correct
    if not fname.lower().endswith(".mp4"):
        return None
    m_eye = _EYE.search(fname)
    if not m_eye:
        return None                                  # no camera number — not ours to touch
    num = int(m_eye.group(1))
    ft = _FOLDER_TS.search(Path(folder).name)
    m_dt = _DATETIME.search(fname)
    if m_dt:
        date, time = m_dt.group(1), m_dt.group(2)
    else:
        m_d = _DATE.search(fname)
        if m_d and ft:                               # date in name, time from folder
            date, time = m_d.group(1), ft.group(2)
        elif ft:                                     # neither in name — both from folder
            date, time = ft.group(1), ft.group(2)
        else:
            return None                              # nothing to anchor a date/time on
    target = f"eye{num:02d}_{date}_{time}.mp4"
    return None if target == fname else target


def plan_folder(folder: Path, files) -> list[dict]:
    """Rename rows for one folder's mp4 files. Detects target-name collisions
    within the folder as well as against files already on disk."""
    rows = []
    planned: dict = {}                               # target name -> source that claimed it
    existing = {f.name for f in files}
    for f in sorted(files):
        if CANON_RE.match(f.name):
            continue                                 # correct — nothing to do
        target = canonical_name(f.name, str(folder))
        if target is None:
            # a camera-numbered mp4 we still couldn't name (no date/time anywhere)
            if _EYE.search(f.name) and f.name.lower().endswith(".mp4") and not CANON_RE.match(f.name):
                rows.append(dict(action=UNKNOWN, folder=str(folder), old=f.name,
                                 new="", src=str(f), dst="",
                                 reason="looks like a camera video but no date/time found"))
            continue                                 # non-camera mp4 (e.g. merged) — ignore
        clash_disk = target in existing and target != f.name
        clash_plan = target in planned
        # A fixed re-encode whose canonical name is taken by the (broken) original:
        # stash the original as ...mp4.broken, then let the fixed file take the name.
        if clash_disk and not clash_plan and _IS_FIXED.search(f.name):
            broken = target + BROKEN_SUFFIX
            if broken in existing or broken in planned:
                rows.append(dict(action=CONFLICT, folder=str(folder), old=f.name, new=target,
                                 src=str(f), dst=str(folder / target),
                                 reason=f"'{broken}' already exists — resolve by hand"))
                continue
            rows.append(dict(action=BROKEN, folder=str(folder), old=target, new=broken,
                             src=str(folder / target), dst=str(folder / broken),
                             reason="broken original set aside for its fixed re-encode"))
            rows.append(dict(action=RENAME, folder=str(folder), old=f.name, new=target,
                             src=str(f), dst=str(folder / target),
                             reason="fixed re-encode takes the original name"))
            planned[broken] = target
            planned[target] = f.name
            existing.discard(target)
            existing.add(broken)
            continue
        if clash_disk or clash_plan:
            rows.append(dict(action=CONFLICT, folder=str(folder), old=f.name, new=target,
                             src=str(f), dst=str(folder / target),
                             reason=f"'{target}' already exists here — not overwritten"))
            continue
        planned[target] = f.name
        rows.append(dict(action=RENAME, folder=str(folder), old=f.name, new=target,
                         src=str(f), dst=str(folder / target),
                         reason="reorder to eye<NN>_<date>_<time>"))
    return rows


def find_mp4_folders(roots, max_depth: int = 6, on_dir=None, should_stop=None):
    """Every folder under `roots` that directly holds ``*.mp4`` files, with those
    files. Walks by name only (does not need the files to be correctly named), so
    a folder where *every* file is misnamed is still found."""
    out = []
    for root in roots:
        base = Path(root)
        try:
            if not base.exists():
                continue
        except OSError:
            continue
        stack = [(base, 0)]
        while stack:
            if should_stop is not None and should_stop():
                return out
            d, depth = stack.pop()
            if on_dir is not None:
                on_dir(d)
            try:
                entries = list(d.iterdir())
            except OSError:
                continue
            mp4s = [e for e in entries if e.is_file() and e.suffix.lower() == ".mp4"]
            if mp4s:
                out.append((d, mp4s))
            for c in entries:
                if c.is_dir() and not sd.skip_dir(c.name) and depth < max_depth:
                    stack.append((c, depth + 1))
    return out


def plan_renames(folder_files) -> list[dict]:
    """`folder_files` is the list of (folder, [mp4 files]) from find_mp4_folders."""
    plan = []
    for folder, files in folder_files:
        plan += plan_folder(Path(folder), [Path(f) for f in files])
    return plan


def scan_and_plan(roots, depth: int = 6, on_dir=None, should_stop=None) -> list[dict]:
    return plan_renames(find_mp4_folders(roots, max_depth=depth, on_dir=on_dir,
                                         should_stop=should_stop))


def execute_renames(plan, on_progress=None, should_stop=None) -> list[dict]:
    """Rename the RENAME rows (in-folder rename — atomic). Never overwrites: a
    target that has appeared since planning is skipped."""
    results = []
    for i, item in enumerate(plan):
        if should_stop is not None and should_stop():
            break
        # RENAME and BROKEN are both plain in-folder renames; BROKEN rows are
        # ordered before the RENAME that reuses the freed name, so the original is
        # always moved out of the way first.
        if item["action"] not in (RENAME, BROKEN):
            results.append(dict(item, result="skipped"))
            continue
        src, dst = Path(item["src"]), Path(item["dst"])
        if on_progress is not None:
            on_progress(i, len(plan), item["old"])
        try:
            if dst.exists():
                results.append(dict(item, result="skipped", reason="target appeared — not overwritten"))
                continue
            os.rename(src, dst)
            results.append(dict(item, result="renamed" if item["action"] == RENAME else "set-aside"))
        except OSError as exc:
            results.append(dict(item, result="error", reason=str(exc)))
    return results


def totals(plan) -> dict:
    out = {a: 0 for a in (RENAME, BROKEN, CONFLICT, UNKNOWN)}
    for p in plan:
        out[p["action"]] = out.get(p["action"], 0) + 1
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Rename misordered camera videos to eye<NN>_<date>_<time>.mp4.")
    ap.add_argument("roots", nargs="*", help="folders/drives to sweep (default: all mounted volumes)")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    roots = a.roots or [str(r) for r in sd.list_drive_roots()]
    print("Scanning:", ", ".join(roots))
    plan = scan_and_plan(roots, depth=a.depth)
    t = totals(plan)
    print(f"\n{t[RENAME]} rename · {t[BROKEN]} broken-set-aside · "
          f"{t[CONFLICT]} conflict · {t[UNKNOWN]} unknown\n")
    for p in plan:
        print(f"  [{p['action']:8}] {p['old']}  ->  {p['new'] or '?'}   ({p['folder']})")
    if not t[RENAME]:
        print("nothing to rename."); return
    if not a.yes and input(f"\nRename {t[RENAME]} file(s)? [y/N] ").strip().lower() != "y":
        print("aborted."); return
    res = execute_renames(plan)
    print(f"renamed {sum(1 for r in res if r.get('result') == 'renamed')} file(s).")


if __name__ == "__main__":
    main()
