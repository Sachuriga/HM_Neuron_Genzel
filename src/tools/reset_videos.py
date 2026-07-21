"""Reset: move every camera folder back to a flat per-drive ``raw/`` folder,
undoing any ``Rat<N>/<date>`` filing so the whole match/organize step can be redone
from a clean slate.

Per drive: a camera folder on drive ``D:`` goes to ``D:\\raw\\<foldername>``, so
every move is a **same-drive rename** — instant, and never a cross-drive
copy+delete. Both already-filed folders (under ``Rat<N>/<date>``) and loose ones
are collected. A folder already under a ``raw/`` folder is left alone; a name
collision in ``raw`` is reported, never overwritten.

Planning is separate from doing: ``plan_reset`` reads only and returns a
reviewable list; nothing moves until ``execute_reset`` is called on that plan.
After a move the now-empty ``Rat<N>/<date>`` skeleton is removed only while it is
genuinely empty (``os.rmdir`` refuses otherwise), so no data is ever deleted.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import scan_drive as sd
import find_videos as fv

RAW_DIRNAME = "raw"

# Plan actions.
MOVE = "move"                 # same-drive rename into <drive>/raw/
SKIP_IN_RAW = "skip-in-raw"   # already under a raw/ folder — nothing to do
DUPLICATE = "duplicate"       # an identical folder is already in raw — left in place
CONFLICT = "conflict"         # a different folder of that name is in raw — needs a human
BLOCKED = "blocked"           # the drive's raw/ location is not writable (permissions)

WRITING = (MOVE,)


def _drive_anchor(p: Path) -> Path:
    """The drive/mount root a path lives on — ``D:\\`` on Windows, the mount point
    (e.g. ``/media/you/EXT``) on Linux/mac. This is where that drive's ``raw/``
    folder is created, so it must be the writable mount root, NOT the filesystem
    root ``/`` (writing there needs sudo and is the classic Errno 13 on Linux)."""
    d = Path(p).resolve()
    if not d.is_dir():
        d = d.parent
    while True:
        try:
            if os.path.ismount(d):
                return d
        except OSError:
            pass
        if d.parent == d:                 # reached the filesystem root
            return d
        d = d.parent


def _in_raw(p: Path) -> bool:
    return any(part.lower() == RAW_DIRNAME for part in p.parts)


def _is_ancestor(a: Path, b: Path) -> bool:
    try:
        b.relative_to(a)
        return True
    except ValueError:
        return False


def _base_for(src: Path, roots) -> Path:
    """The folder under which this drive's ``raw/`` goes. Prefer the deepest scan
    root the user actually picked that contains ``src`` — that root is a location
    they can read/write. Fall back to the auto-detected mount root only when the
    folder sits under no given root."""
    if roots:
        src_r = src.resolve()
        owning = [Path(r).resolve() for r in roots if r
                  and _is_ancestor(Path(r).resolve(), src_r)]
        if owning:
            return max(owning, key=lambda r: len(r.parts))
    return _drive_anchor(src)


def _signature(path: Path):
    """(file count, total bytes) under path, or None if unreadable — a cheap
    same-name comparison to tell a duplicate from a genuine conflict."""
    try:
        n = tot = 0
        for f in Path(path).rglob("*"):
            if f.is_file():
                n += 1
                tot += f.stat().st_size
        return n, tot
    except OSError:
        return None


def plan_reset(camera_folders: list[dict], roots=None) -> list[dict]:
    """Work out where every camera folder would move. Reads only.

    `camera_folders` is the list from
    ``find_videos.find_loose_camera_folders(roots, include_filed=True)`` — each
    dict has a ``path`` (and ``is_filed``). `roots` are the drive roots being
    swept; when given, a folder's ``raw/`` goes under the root that contains it
    (a writable place the user picked), else under its mount root. One plan row
    per folder."""
    plan = []
    claimed: dict = {}                       # target path -> source already headed there
    for v in camera_folders:
        src = Path(v["path"])
        name = src.name
        if _in_raw(src):
            plan.append(dict(action=SKIP_IN_RAW, src=str(src), dst="", name=name,
                             is_filed=v.get("is_filed", False),
                             total=v.get("total", 0), reason="already under a raw/ folder"))
            continue
        base = _base_for(src, roots)
        dst = base / RAW_DIRNAME / name
        row = dict(src=str(src), dst=str(dst), name=name,
                   is_filed=v.get("is_filed", False), total=v.get("total", 0))
        # the drive's raw/ location must be writable — otherwise the move would die
        # with Errno 13. Flag it here, up front, instead of failing mid-run.
        raw_dir = base / RAW_DIRNAME
        probe = raw_dir if raw_dir.exists() else base
        if not os.access(probe, os.W_OK):
            plan.append(dict(row, action=BLOCKED,
                             reason=f"{probe} is not writable — mount the drive or fix permissions"))
            continue
        # collide with something already at the destination, or with another
        # source we already planned to move there
        rival = claimed.get(str(dst))
        if dst.exists() or rival is not None:
            other = rival if rival is not None else str(dst)
            sig_src, sig_other = _signature(src), _signature(Path(other))
            same = sig_src is not None and sig_src == sig_other
            plan.append(dict(row, action=DUPLICATE if same else CONFLICT,
                             reason=("identical copy already in raw — left in place" if same
                                     else f"a different folder named {name} is already in raw")))
            continue
        claimed[str(dst)] = str(src)
        plan.append(dict(row, action=MOVE,
                         reason="move to raw (same-drive rename)"
                                if _drive_anchor(src) == _drive_anchor(dst) else
                                "move to raw (cross-drive copy)"))
    return plan


def _prune_empty(start: Path, stop: Path) -> None:
    """Remove now-empty parent dirs from `start` up toward (not including) the drive
    root `stop`. os.rmdir refuses a non-empty dir, so real data is never removed."""
    d = start
    while d != stop and d.parent != d:
        try:
            os.rmdir(d)
        except OSError:
            break                            # not empty (or gone) — stop climbing
        d = d.parent


def execute_reset(plan: list[dict], on_progress=None, should_stop=None) -> list[dict]:
    """Perform the MOVE rows of a plan. Same-drive moves are renames. Never
    overwrites (a target that appears is skipped); empty Rat/date parents left
    behind are pruned. Returns a result row per plan row."""
    results = []
    todo = [p for p in plan if p["action"] == MOVE]
    for i, item in enumerate(plan):
        if should_stop is not None and should_stop():
            break
        if item["action"] != MOVE:
            results.append(dict(item, result="skipped"))
            continue
        src, dst = Path(item["src"]), Path(item["dst"])
        if on_progress is not None:
            on_progress(i, len(plan), item["name"])
        try:
            if dst.exists():
                results.append(dict(item, result="skipped", reason="target appeared — not overwritten"))
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            _prune_empty(src.parent, _drive_anchor(src))
            results.append(dict(item, result="moved"))
        except OSError as exc:
            results.append(dict(item, result="error", reason=str(exc)))
    return results


def totals(plan: list[dict]) -> dict:
    out = {a: 0 for a in (MOVE, SKIP_IN_RAW, DUPLICATE, CONFLICT, BLOCKED)}
    for p in plan:
        out[p["action"]] = out.get(p["action"], 0) + 1
    out["bytes_move"] = sum(p.get("total", 0) for p in plan if p["action"] == MOVE)
    return out


def scan_and_plan(roots, depth: int = 6, on_dir=None, should_stop=None) -> list[dict]:
    """Convenience: find every camera folder (loose + filed) under `roots` and plan
    the reset in one call."""
    folders = fv.find_loose_camera_folders(roots, max_depth=depth, on_dir=on_dir,
                                           should_stop=should_stop, include_filed=True)
    return plan_reset(folders, roots=roots)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Move all camera folders into per-drive raw/ folders.")
    ap.add_argument("roots", nargs="*", help="drive roots to sweep (default: all mounted volumes)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args(argv)
    roots = a.roots or [str(r) for r in sd.list_drive_roots()]
    print("Scanning:", ", ".join(roots))
    plan = scan_and_plan(roots, depth=a.depth, on_dir=lambda d: None)
    t = totals(plan)
    print(f"\nplan: {t[MOVE]} move  ·  {t[SKIP_IN_RAW]} already in raw  ·  "
          f"{t[DUPLICATE]} duplicate  ·  {t[CONFLICT]} conflict")
    for p in plan:
        if p["action"] in (MOVE, CONFLICT):
            print(f"  [{p['action']:8}] {p['src']}  ->  {p['dst'] or '(n/a)'}")
    if not t[MOVE]:
        print("nothing to move."); return
    if not a.yes:
        if input(f"\nMove {t[MOVE]} folder(s) into per-drive raw\\ ? [y/N] ").strip().lower() != "y":
            print("aborted."); return
    res = execute_reset(plan, on_progress=lambda i, n, nm: None)
    moved = sum(1 for r in res if r.get("result") == "moved")
    print(f"moved {moved} folder(s).")


if __name__ == "__main__":
    main()
