"""Plan and execute a consolidation of scattered session data into one tree.

Raw HexMaze data ends up spread across several drives, and not always in whole
sessions: one drive may hold a session's ``pre`` and ``post`` recordings while
the ``task`` recording sits on another, with a third drive holding a redundant
copy of that task folder. Nothing is wrong with any single folder — the session
is simply not assembled anywhere.

This module assembles it. Given the roster (what the spreadsheet expects) and the
index produced by ``scan_drive_gui.search_all_drives`` (what is actually on the
drives), ``plan_organize`` works out what would have to be copied where, and
``execute_plan`` performs the copy.

Two properties matter more than anything else here, because the inputs are the
only copies of months of irreplaceable recordings:

  * **It only ever copies.** Nothing is deleted, moved, or overwritten — not
    sources, not existing files at the destination. Freeing up the redundant
    copies afterwards is a decision left to a human with the plan in hand.
  * **Planning is separate from doing.** ``plan_organize`` touches nothing and
    returns a reviewable list of actions. Nothing is written until someone looks
    at that plan and calls ``execute_plan``.

The merge happens at the granularity of a session folder's top-level entries
(each recording folder, each camera folder, each loose video), because that is
the level at which the data is actually split across drives.
"""

from __future__ import annotations

import os
import re
import stat
import shutil
from pathlib import Path

# A rat folder that names its implant, e.g. Rat5_491390 — preferred over a bare
# "rat5" when choosing what to call the folder in the destination tree.
_RAT_IMPLANT_RE = re.compile(r"^[Rr]at\s*_?\d+_\d+$")

# A camera video: eye01_2026-05-11_09-57-36.mp4
_EYE_RE = re.compile(r"^eye\d+.*\.mp4$", re.I)

# Plan actions.
MOVE = "move"                  # rename within one drive — instant, no extra space
COPY = "copy"                  # cross-drive transfer; source is left in place
SKIP_PRESENT = "skip-present"  # already at the destination, same size
SKIP_DERIVED = "skip-derived"  # DLC/LFP output, not raw acquisition data
SKIP_VIDEO = "skip-video"      # a loose video not ours to file (aborted / other batch)
SKIP_ARCHIVED = "skip-archived"  # already filed under an HM_neuron archive — leave it
DUPLICATE = "duplicate"        # identical to a chosen source on another drive
CONFLICT = "conflict"          # same name, different content — needs a human
UNREADABLE = "unreadable"      # could not be read — drive disconnected?


ARCHIVE_NAME = "HM_neurons"     # the canonical archive folder to consolidate into


def _is_archive_part(name: str) -> bool:
    n = name.lower()
    return n.startswith("hm_neuron") and "preprocess" not in n


def in_archive(path) -> bool:
    """True if `path` already lives under an ``HM_neuron``/``HM_neurons`` archive
    folder — i.e. it is already correctly filed as Rat<N>/<date>, just possibly on
    another drive. Such data is left exactly where it is; only misplaced data (a
    loose dump, a GL37/GL38 folder, some other tree) is ever moved into the
    archive. The preprocessing tree is not an archive and is excluded."""
    return any(_is_archive_part(part) for part in Path(path).parts)


def archive_dest(dest) -> str:
    """Resolve the chosen destination to an HM_neuron archive folder, so organizing
    always consolidates into ``…/HM_neurons/Rat<N>/<date>/``.

    If the pick already is (or sits under) an HM_neuron archive, it is used as-is;
    otherwise an ``HM_neurons`` folder is placed inside it. This is why a drive
    root like ``E:\\`` becomes ``E:\\HM_neurons`` while ``F:\\HM_neurons`` is kept
    unchanged — never nested into ``F:\\HM_neurons\\HM_neurons``."""
    p = Path(dest)
    if _is_archive_part(p.name) or in_archive(p):
        return str(p)
    return str(p / ARCHIVE_NAME)

# Actions that write. Everything else in a plan is informational.
WRITING = (MOVE, COPY)


def is_raw_session(sess: Path) -> bool:
    """Is this folder raw acquisition data, rather than derived output?

    The drives carry both, filed under the same ``Rat<N>/<YYYYMMDD>`` names: the
    raw tree and, beside it, a preprocessing tree of DLC coordinates, LFP
    exports, logs and labelled videos for the very same sessions. Consolidating
    without telling them apart would merge analysis output into the raw archive.

    Raw data is identified by what only raw data has: camera videos named
    ``eye<NN>*.mp4``, or Trodes ``.rec`` recordings (as a folder or a file).
    A derived session has neither — its videos are ``collected_frames.mp4`` and
    friends. Probes two levels deep, which is where both always appear.

    Raises OSError if the folder cannot be read. That must not be softened into
    a False: these drives are externals that can drop off mid-run, and a
    disconnected drive answering "not raw" would quietly drop real recordings
    out of the plan, reported as derived output that was never there."""
    for c in sess.iterdir():                       # OSError propagates by design
        if _EYE_RE.match(c.name) or c.name.lower().endswith(".rec"):
            return True
        try:
            if c.is_dir():
                for g in c.iterdir():
                    if _EYE_RE.match(g.name) or g.name.lower().endswith(".rec"):
                        return True
        except OSError:
            # One unreadable subfolder is not evidence either way; keep looking,
            # and let a wholly unreadable session fail on the outer iterdir.
            continue
    return False


def canonical_rat_dirname(rat_no: int, candidates: list[str]) -> str:
    """Pick the folder name to use for a rat in the destination tree.

    The same animal is filed under different names on different drives
    (``Rat5_491390`` on one, ``rat5`` on another). Prefer a name that carries the
    implant id, since that is the lab's canonical form and the one the rest of
    the pipeline expects; fall back to the commonest name, then to ``Rat<N>``."""
    named = [c for c in candidates if _RAT_IMPLANT_RE.match(c)]
    pool = named or candidates
    if not pool:
        return f"Rat{rat_no}"
    # commonest, ties broken deterministically by name
    return max(sorted(set(pool)), key=lambda n: (pool.count(n), n))


def entry_signature(p: Path) -> tuple[int, int] | None:
    """(file count, total bytes) for a file or a folder tree, or None if it
    cannot be read.

    Cheap stand-in for "is this the same data?" — it reads no file contents, only
    metadata. Two entries with the same name and the same signature are treated
    as the same thing; a mismatch is never guessed at, it becomes a CONFLICT for
    a human to resolve.

    None, never a zero signature, is the answer for an unreadable entry. A (0, 0)
    fallback would be actively unsafe: two entries on two disconnected drives
    would both signature as (0, 0), compare equal, and be declared identical
    duplicates — so one would be silently dropped on the strength of having
    failed to read either.

    Getting that right takes more care than it looks. Path.is_file() answers
    False for a missing path instead of raising, and rglob() over a missing
    directory yields nothing at all rather than failing — so the obvious
    implementation walks a disconnected drive, finds no files, and reports a
    confident (0, 0). Hence the explicit stat() up front, and os.walk's onerror
    hook, which is the only way its errors surface."""
    try:
        st = p.stat()
    except OSError:
        return None                      # missing / disconnected / no permission
    if stat.S_ISREG(st.st_mode):
        return (1, st.st_size)

    n = tot = 0
    failed = False

    def _onerr(_exc):
        nonlocal failed
        failed = True                    # os.walk swallows these unless asked

    for root, _dirs, files in os.walk(p, onerror=_onerr):
        for f in files:
            try:
                tot += os.stat(os.path.join(root, f)).st_size
                n += 1
            except OSError:
                failed = True
    return None if failed else (n, tot)


def _session_entries(sess: Path) -> dict[str, Path]:
    """Top-level entries of a session folder, by name — the units we merge."""
    out = {}
    try:
        for c in sorted(sess.iterdir()):
            out[c.name] = c
    except OSError:
        pass
    return out


def volume_of(p: Path) -> str:
    """Identify the volume a path lives on, so same-volume work can be a rename.

    Uses the drive letter on Windows and the device id on POSIX. Walks up to the
    nearest existing parent, because the destination usually does not exist yet
    and a non-existent path has no device."""
    q = Path(p)
    while True:
        try:
            if q.exists():
                if os.name == "nt":
                    return str(q.drive or q.anchor).upper()
                return str(q.stat().st_dev)
        except OSError:
            pass
        if q.parent == q:
            return str(Path(p).drive or Path(p).anchor).upper()
        q = q.parent


def plan_organize(roster: list[dict], found: dict, dest: str,
                  videos: list[dict] | None = None,
                  on_progress=None, should_stop=None) -> list[dict]:
    """Work out what it would take to assemble every roster session under `dest`.

    `found` is the (rat_no, date8) -> [hit] index from search_all_drives(), and
    `videos` the loose camera folders assign_videos() matched to a rat/date.
    Reads only metadata; writes nothing. Returns one row per (session, entry,
    source) with an action of MOVE / COPY / SKIP_PRESENT / SKIP_DERIVED /
    DUPLICATE / CONFLICT / UNREADABLE.

    Whether an entry moves or copies is decided by the volume it is already on:
    within one drive a move is a rename — instant, needing no extra space, and
    leaving the bytes exactly where they are. Across drives that is impossible,
    so the data is copied and the source left untouched."""
    dest_root = Path(dest)
    dest_vol = volume_of(dest_root)
    plan: list[dict] = []

    # What to call each rat's folder, decided once from every name seen on disk.
    names: dict[int, list[str]] = {}
    for (rat_no, _d), hits in found.items():
        for h in hits:
            names.setdefault(rat_no, []).append(Path(h["path"]).parent.name)
    rat_dir = {r: canonical_rat_dirname(r, n) for r, n in names.items()}

    for e in roster:
        if should_stop is not None and should_stop():
            break
        if not e["date8"]:
            continue
        key = (e["rat_no"], e["date8"])
        hits = found.get(key, [])
        if not hits:
            continue                       # nothing on any drive; nothing to plan
        if on_progress is not None:
            on_progress(f"{e['rat']} {e['date8']}")

        rdir = rat_dir.get(e["rat_no"], f"Rat{e['rat_no']}")
        target_sess = dest_root / rdir / e["date8"]
        base_sess = dict(rat=e["rat"], rat_no=e["rat_no"], date8=e["date8"],
                         day=e["day"], session=e["session"], repeat=e["repeat"])

        # name -> [(source path, signature, hit)] across every drive. An entry
        # that is already the destination is not a source for itself — the
        # destination is usually one of the drives being searched, so without
        # this every in-place file would be planned against its own path.
        by_name: dict[str, list[tuple[Path, tuple, dict]]] = {}
        for h in hits:
            src_sess = Path(h["path"])
            try:
                raw = is_raw_session(src_sess)
            except OSError as exc:
                plan.append(dict(base_sess, entry="(whole session)", src=str(src_sess),
                                 dst="", action=UNREADABLE, n_files=0, bytes=0,
                                 reason=f"cannot read source — drive disconnected? ({exc})"))
                continue
            if not raw:
                # Derived output (DLC/LFP) for a session the roster does expect.
                # Reported, not merged: it belongs beside the raw archive, not in it.
                plan.append(dict(base_sess, entry="(whole session)", src=str(src_sess),
                                 dst="", action=SKIP_DERIVED, n_files=0, bytes=0,
                                 reason="no eye*.mp4 or .rec — derived output, "
                                        "not raw acquisition data"))
                continue
            for name, path in _session_entries(src_sess).items():
                if path == target_sess / name:
                    continue
                sig = entry_signature(path)
                if sig is None:
                    plan.append(dict(base_sess, entry=name, src=str(path), dst="",
                                     action=UNREADABLE, n_files=0, bytes=0,
                                     reason="cannot read source — drive disconnected?"))
                    continue
                by_name.setdefault(name, []).append((path, sig, h))

        for name, cands in sorted(by_name.items()):
            dst = target_sess / name
            base = dict(base_sess, entry=name, dst=str(dst))

            # Already filed under an HM_neuron archive (this or another drive)?
            # Then it is correctly placed — leave it, whatever the destination.
            archived = next((c for c in cands if in_archive(c[0])), None)
            if archived:
                s = archived[1]
                plan.append(dict(base, src=str(archived[0]), action=SKIP_ARCHIVED,
                                 n_files=s[0], bytes=s[1],
                                 reason="already under an HM_neuron archive — left in place"))
                continue

            sigs = {c[1] for c in cands}
            if len(sigs) > 1:
                # Same name, different content, on different drives. Could be a
                # partial copy, could be two different recordings. Not ours to
                # guess — surface every version and copy none of them.
                for path, sig, h in cands:
                    plan.append(dict(base, src=str(path), action=CONFLICT,
                                     n_files=sig[0], bytes=sig[1],
                                     reason=f"{len(sigs)} differing versions of "
                                            f"'{name}' across drives — resolve by hand"))
                continue

            sig = sigs.pop()
            if dst.exists():
                dsig = entry_signature(dst)
                if dsig is None:
                    plan.append(dict(base, src=str(cands[0][0]), action=UNREADABLE,
                                     n_files=sig[0], bytes=sig[1],
                                     reason="destination exists but cannot be read"))
                elif dsig == sig:
                    # Already assembled here; this entry is done and every source
                    # of it is redundant.
                    plan.append(dict(base, src=str(cands[0][0]), action=SKIP_PRESENT,
                                     n_files=sig[0], bytes=sig[1],
                                     reason="already at destination, same size"))
                else:
                    plan.append(dict(base, src=str(cands[0][0]), action=CONFLICT,
                                     n_files=sig[0], bytes=sig[1],
                                     reason="destination exists with different size — "
                                            "not overwriting"))
                continue

            # Identical copies on several drives: take one, name the rest. Prefer
            # a source already on the destination volume — that turns the whole
            # entry into a rename instead of a cross-drive transfer of 60+ GB.
            chosen = _prefer(cands, dest_vol)
            same_vol = volume_of(chosen[0]) == dest_vol
            plan.append(dict(base, src=str(chosen[0]),
                             action=MOVE if same_vol else COPY,
                             n_files=sig[0], bytes=sig[1],
                             reason="rename within the same drive" if same_vol else ""))
            for path, s, h in cands:
                if path == chosen[0]:
                    continue
                plan.append(dict(base, src=str(path), action=DUPLICATE,
                                 n_files=s[0], bytes=s[1],
                                 reason=f"identical copy of '{name}' — "
                                        f"using {chosen[0]}, leaving this one"))

    _plan_videos(videos or [], roster, found, rat_dir, dest_root, dest_vol, plan)
    return plan


def _plan_videos(videos, roster, found, rat_dir, dest_root, dest_vol, plan):
    """Add filing actions for the loose camera folders assign_videos() matched.

    Each matched folder is filed as a timestamp subfolder of its session —
    ``<dest>/Rat<N>/<date>/<2026-07-08_11-00-27>/`` — which is exactly the shape
    the rig already uses for implanted sessions and which the scanner reads for
    every rat. Same volume → move (rename); across drives → copy, source kept.
    Aborted false-starts and folders that only duplicate video the session
    already has are reported, never filed over."""
    roster_by = {(e["rat_no"], e["date8"]): e for e in roster if e["date8"]}
    for v in videos:
        rat_no, date8 = v.get("rat_no"), v.get("date8")
        refile = v.get("refile")
        # File a video when it FILLS an empty roster session (the high-confidence
        # case), or when it is `refile` — already sorted, but to the WRONG rat, so
        # it must be moved to the correct one (this is the deliberate exception to
        # "leave archived data alone"). Everything else is shown but never moved:
        # an aborted start, a folder from another batch, an in-window date no rat
        # ran, or video the session already has.
        if in_archive(v["path"]) and not refile:
            plan.append(dict(rat=v.get("rat") or "?", rat_no=rat_no, date8=date8 or "?",
                             day="", session="", repeat=None,
                             entry=Path(v["path"]).name, src=v["path"], dst="",
                             action=SKIP_ARCHIVED, n_files=0, bytes=v.get("total", 0),
                             reason="already under an HM_neuron archive — left in place"))
            continue
        if not (v.get("fills_missing") or refile) or rat_no is None or not date8:
            plan.append(dict(rat=v.get("rat") or "?", rat_no=rat_no, date8=date8 or "?",
                             day="", session="", repeat=None,
                             entry=Path(v["path"]).name, src=v["path"], dst="",
                             action=SKIP_VIDEO, n_files=0, bytes=v.get("total", 0),
                             reason=v.get("status", "unassigned video")))
            continue
        e = roster_by.get((rat_no, date8))
        rdir = rat_dir.get(rat_no, f"Rat{rat_no}")
        src = Path(v["path"])
        dst = dest_root / rdir / date8 / src.name
        # Prefer the per-folder day/session assign_videos resolved (correct even
        # when a rat has several sessions in a day); fall back to the roster entry.
        base = dict(rat=v.get("rat") or f"Rat{rat_no}", rat_no=rat_no, date8=date8,
                    day=v.get("day") or (e["day"] if e else ""),
                    session=v.get("session") if v.get("session") != "" else (e["session"] if e else ""),
                    repeat=e["repeat"] if e else None, entry=src.name, dst=str(dst))
        sig = entry_signature(src)
        if sig is None:
            plan.append(dict(base, src=str(src), action=UNREADABLE, n_files=0,
                             bytes=v.get("total", 0),
                             reason="cannot read video source — drive disconnected?"))
            continue
        if dst.exists():
            dsig = entry_signature(dst)
            done = dsig == sig if dsig is not None else False
            plan.append(dict(base, src=str(src),
                             action=SKIP_PRESENT if done else CONFLICT,
                             n_files=sig[0], bytes=sig[1],
                             reason="video already filed here" if done
                             else "a different folder of this name is already there"))
            continue
        same_vol = volume_of(src) == dest_vol
        if refile:
            reason = ("re-file MISFILED video → correct rat (rename, same drive)"
                      if same_vol else
                      "re-file MISFILED video → correct rat (copy; wrong-place original kept)")
        else:
            reason = ("file unfiled video → session (rename, same drive)"
                      if same_vol else "file unfiled video → session")
        plan.append(dict(base, src=str(src), action=MOVE if same_vol else COPY,
                         n_files=sig[0], bytes=sig[1], is_video=True, reason=reason))


def _prefer(cands: list[tuple[Path, tuple, dict]], dest_vol: str):
    """Of several identical copies, pick the one to take: prefer one already on
    the destination volume (a rename beats a transfer), then a drive the user
    selected, then a folder named with the implant id, then the shortest path.
    Deterministic, so a re-plan makes the same choice."""
    def rank(c):
        path, _sig, h = c
        return (0 if volume_of(path) == dest_vol else 1,
                0 if h.get("in_selected") else 1,
                0 if _RAT_IMPLANT_RE.match(path.parent.name) else 1,
                len(str(path)), str(path))
    return min(cands, key=rank)


def plan_totals(plan: list[dict]) -> dict:
    """Headline numbers for a plan: what it would write, and what needs eyes.

    `bytes_needed` counts only the cross-drive copies, because that is the only
    part that consumes space at the destination — a same-drive move is a rename
    and costs nothing, however many terabytes it relocates."""
    mv = [p for p in plan if p["action"] == MOVE]
    cp = [p for p in plan if p["action"] == COPY]
    return dict(
        n_move=len(mv), bytes_move=sum(p["bytes"] for p in mv),
        n_copy=len(cp), bytes_copy=sum(p["bytes"] for p in cp),
        bytes_needed=sum(p["bytes"] for p in cp),
        n_files=sum(p["n_files"] for p in mv + cp),
        n_present=sum(1 for p in plan if p["action"] == SKIP_PRESENT),
        n_derived=sum(1 for p in plan if p["action"] == SKIP_DERIVED),
        n_archived=sum(1 for p in plan if p["action"] == SKIP_ARCHIVED),
        n_duplicate=sum(1 for p in plan if p["action"] == DUPLICATE),
        n_conflict=sum(1 for p in plan if p["action"] == CONFLICT),
        n_unreadable=sum(1 for p in plan if p["action"] == UNREADABLE),
        sessions=len({(p["rat"], p["date8"]) for p in mv + cp}),
    )


def free_bytes(dest: str) -> int:
    """Free space at `dest`, walking up to the nearest existing parent."""
    q = Path(dest)
    while True:
        try:
            return shutil.disk_usage(q).free
        except OSError:
            if q.parent == q:
                return -1
            q = q.parent


def space_check(plan: list[dict], dest: str) -> tuple[bool, str]:
    """(ok, message) — can `dest` hold the copies this plan wants to make?

    Blocks rather than starting a transfer that cannot finish: a half-copied
    archive is harder to reason about than one that was never started. Keeps a
    small headroom margin so the destination is not driven to literally zero."""
    need = plan_totals(plan)["bytes_needed"]
    free = free_bytes(dest)
    if free < 0:
        return False, f"Cannot read free space at {dest}."
    margin = 1 * 1024 ** 3
    if need + margin > free:
        return False, (f"Not enough space at {dest}.\n\n"
                       f"Needs {hsize(need)} of cross-drive copies (plus {hsize(margin)} "
                       f"headroom), but only {hsize(free)} is free — "
                       f"short by {hsize(need + margin - free)}.\n\n"
                       f"Pick a destination with more room, or narrow the scope.")
    return True, (f"{hsize(need)} to copy, {hsize(free)} free at {dest}.")


def _copy_file(src: Path, dst: Path, on_bytes=None, should_stop=None) -> None:
    """Copy one file via a .partial temp, then rename.

    The rename is what makes an interrupted copy safe: a half-written file never
    occupies the real name, so a later scan sees the entry as absent rather than
    as a complete-looking file that is silently truncated."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            if should_stop is not None and should_stop():
                fo.close()
                tmp.unlink(missing_ok=True)
                raise InterruptedError("cancelled")
            buf = fi.read(8 * 1024 * 1024)
            if not buf:
                break
            fo.write(buf)
            if on_bytes is not None:
                on_bytes(len(buf))
    want = src.stat().st_size
    got = tmp.stat().st_size
    if got != want:
        tmp.unlink(missing_ok=True)
        raise OSError(f"size mismatch after copy: {got} != {want}")
    shutil.copystat(src, tmp, follow_symlinks=False)
    os.replace(tmp, dst)


def _move_entry(src: Path, dst: Path) -> None:
    """Relocate `src` to `dst` within one volume, by rename.

    os.rename only ever succeeds within a filesystem — it will not fall back to
    a copy-then-delete. That is exactly the property wanted here: if the volumes
    turn out to differ, this raises instead of quietly deleting the source after
    a long transfer. shutil.move() would do the opposite, which is why it is not
    used. The rename itself is atomic, so an interrupted move leaves the entry
    whole at one path or the other, never half at both."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise OSError(f"destination already exists: {dst}")
    os.rename(src, dst)


def execute_plan(plan: list[dict], on_progress=None, on_bytes=None,
                 should_stop=None) -> list[dict]:
    """Perform the MOVE and COPY rows of `plan`. Every other action is a no-op.

    A MOVE is a same-volume rename: instant, and it consumes no space. A COPY
    goes file by file to a temp name, is size-verified, then renamed into place,
    and never deletes the source. Nothing existing is overwritten either way.
    Returns one result row per attempted entry with ok/skipped/error/cancelled."""
    results = []
    todo = [p for p in plan if p["action"] in WRITING]
    for i, item in enumerate(todo):
        if should_stop is not None and should_stop():
            results.append(dict(item, result="cancelled", detail="stopped before start"))
            continue
        src, dst = Path(item["src"]), Path(item["dst"])
        if on_progress is not None:
            on_progress(i + 1, len(todo), f"{item['rat']} {item['date8']} / {item['entry']}")
        try:
            if dst.exists():
                # Something appeared since planning — never clobber it.
                results.append(dict(item, result="skipped",
                                    detail="destination appeared since planning"))
                continue
            if not src.exists():
                results.append(dict(item, result="error",
                                    detail="source vanished since planning"))
                continue
            if item["action"] == MOVE:
                _move_entry(src, dst)
                if on_bytes is not None:
                    on_bytes(item["bytes"])       # instant, but keep the bar honest
            elif src.is_file():
                _copy_file(src, dst, on_bytes=on_bytes, should_stop=should_stop)
            else:
                for f in sorted(src.rglob("*")):
                    if not f.is_file():
                        continue
                    _copy_file(f, dst / f.relative_to(src),
                               on_bytes=on_bytes, should_stop=should_stop)
            results.append(dict(item, result="ok", detail=""))
        except InterruptedError:
            results.append(dict(item, result="cancelled", detail="cancelled mid-copy"))
            break
        except (OSError, shutil.Error) as exc:
            results.append(dict(item, result="error", detail=str(exc)))
    return results


def hsize(n: int) -> str:
    """Human-readable byte size."""
    if n is None or n < 0:
        return "?"
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if v < 1024 or unit == "PB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024.0
