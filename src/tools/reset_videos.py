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

WRITING = (MOVE,)


def _drive_anchor(p: Path) -> Path:
    """The drive root a path lives on: ``D:\\`` on Windows, the mount root on posix
    (best effort). Used as the base for that drive's ``raw/`` folder."""
    if p.anchor:
        return Path(p.anchor)
    return Path(p.parts[0]) if p.parts else p


def _in_raw(p: Path) -> bool:
    return any(part.lower() == RAW_DIRNAME for part in p.parts)


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


def plan_reset(camera_folders: list[dict]) -> list[dict]:
    """Work out where every camera folder would move. Reads only.

    `camera_folders` is the list from
    ``find_videos.find_loose_camera_folders(roots, include_filed=True)`` — each
    dict has a ``path`` (and ``is_filed``). One plan row per folder."""
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
        dst = _drive_anchor(src) / RAW_DIRNAME / name
        row = dict(src=str(src), dst=str(dst), name=name,
                   is_filed=v.get("is_filed", False), total=v.get("total", 0))
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
    out = {a: 0 for a in (MOVE, SKIP_IN_RAW, DUPLICATE, CONFLICT)}
    for p in plan:
        out[p["action"]] = out.get(p["action"], 0) + 1
    out["bytes_move"] = sum(p.get("total", 0) for p in plan if p["action"] == MOVE)
    return out


def scan_and_plan(roots, depth: int = 6, on_dir=None, should_stop=None) -> list[dict]:
    """Convenience: find every camera folder (loose + filed) under `roots` and plan
    the reset in one call."""
    folders = fv.find_loose_camera_folders(roots, max_depth=depth, on_dir=on_dir,
                                           should_stop=should_stop, include_filed=True)
    return plan_reset(folders)


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
