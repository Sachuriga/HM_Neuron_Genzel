"""Locate camera videos that were never filed under a rat, and work out where
they belong from how the rig records.

The acquisition PC writes each camera recording to a raw dump folder named only
by timestamp — ``<dump>/2026-07-08_11-00-27/eye01..eye12.mp4`` — and the videos
are only later sorted into ``Rat<N>/<YYYYMMDD>/``. When that sorting step is
skipped (or a session is missing video), the recording sits unattached in the
dump, and the coverage scan reports the session as video-less even though the
video exists.

This module finds those loose camera folders and assigns each to a rat + date
using the fixed way the rig runs, given by the experimenter:

  * Two rats are recorded per block, smaller id first.
  * Non-implanted animals start around 09:00 and record ~20 min  -> small files.
  * Implanted animals start around 11:00 and record  >1 hour     -> large files.

So a folder's average file size says implanted-or-not (duration), and its start
time orders the pair (earliest = smaller id). A near-empty folder is an aborted
false-start, not a recording, and is never assigned.

Nothing here writes or moves anything: it reads sizes and names and returns an
assignment for review. Acting on it is the organizer's job.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

import scan_drive as sd

# eye03_2026-07-08_11-00-27.mp4  ->  cam idx, date, HH, MM, SS
_CAM_RE = re.compile(r"^eye(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.mp4$", re.I)
_DATE_RE = re.compile(r"^\d{8}$")

# Average bytes-per-camera thresholds. These separate the three cases by the
# only thing that varies — recording length — and are deliberately wide: a real
# non-implant clip is ~70 MB/cam, a real implant clip ~210-230 MB/cam, and an
# aborted start is a few MB at most. Calibrated against known-good sessions.
ABORTED_MB = 5.0        # below this per camera: a false start, not a recording
IMPLANT_MB = 130.0      # at/above this per camera: implanted (>1 h); else ~20 min

ABORTED = "aborted"
NON_IMPLANT = "non-implant"
IMPLANT = "implant"


def scan_camera_folder(folder: Path) -> dict | None:
    """Summarise a folder of ``eye<NN>_*.mp4`` files, or None if it holds none.

    Returns the camera count, the shared date and start time parsed from the file
    names, the total bytes and the average bytes per camera (the size signal used
    to tell a 20-minute clip from an hour-plus one)."""
    n = 0
    total = 0
    date8 = start = None
    try:
        for p in folder.iterdir():
            m = _CAM_RE.match(p.name)
            if not m:
                continue
            n += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
            date8 = m.group(2).replace("-", "")
            start = f"{m.group(3)}:{m.group(4)}:{m.group(5)}"
    except OSError:
        return None
    if not n:
        return None
    return dict(path=str(folder), name=folder.name, n_cam=n, date8=date8,
                start=start, hour=int(start[:2]) if start else None,
                total=total, avg_mb=total / n / 1e6)


def classify_video(v: dict) -> str:
    """ABORTED / NON_IMPLANT / IMPLANT from a folder's average file size."""
    if v["avg_mb"] < ABORTED_MB:
        return ABORTED
    return IMPLANT if v["avg_mb"] >= IMPLANT_MB else NON_IMPLANT


def _filed_rat_date(p) -> tuple:
    """The (rat_no, date8) a path is currently filed under, or (None, None) if it
    is loose. Read from the ``Rat<N>/<YYYYMMDD>/`` segment of the path itself —
    where the folder *currently* sits, which the re-sort compares against where it
    *should* sit."""
    parts = Path(p).parts
    for i, seg in enumerate(parts):
        if _DATE_RE.match(seg) and i >= 1:
            rn = sd._rat_of(parts[i - 1])
            if rn is not None:
                return rn, seg
    return None, None


def _is_filed(p: Path) -> bool:
    """True if p already sits under a ``Rat<N>/<YYYYMMDD>/`` path — i.e. it has
    been sorted to a rat and is not loose."""
    return _filed_rat_date(p)[0] is not None


def find_loose_camera_folders(roots, max_depth: int = 6, on_dir=None,
                              should_stop=None, include_filed: bool = False) -> list[dict]:
    """Camera-video folders under `roots`, each tagged with ``is_filed``.

    Walks each root to `max_depth`, skipping the system/junk folders scan_drive
    already knows to avoid. A directory counts as a camera folder when it holds
    at least one ``eye<NN>_*.mp4``.

    By default only *loose* folders are returned (not already under
    ``Rat<N>/<date>/``) — those are the lost ones. Pass ``include_filed=True`` to
    also return the already-filed folders: assign_videos needs them so a loose
    folder gets the right session number when an earlier session that day is
    already sorted (the filed one holds its slot in the by-time ordering)."""
    out = []
    seen = set()
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
                children = [c for c in d.iterdir() if c.is_dir()]
            except OSError:
                continue
            info = scan_camera_folder(d)
            if info:
                filed = _is_filed(d)
                if (include_filed or not filed) and info["path"] not in seen:
                    seen.add(info["path"])
                    info["is_filed"] = filed
                    info["kind"] = classify_video(info)
                    out.append(info)
                continue                     # a camera folder has no camera subfolders
            for c in children:
                if sd.skip_dir(c.name):
                    continue
                if depth < max_depth:
                    stack.append((c, depth + 1))
    return out


def assign_videos(loose: list[dict], roster: list[dict],
                  have_video=None) -> list[dict]:
    """Attach each loose camera folder to a rat + session from the sheet.

    Matching is by date then time, using the sheet's *training order*. For one
    date, every camera folder is ordered by start time and zipped 1:1 onto that
    date's ``(rat, session)`` slots taken in ``train_order`` — the Raw-sheet row
    order, which the experimenter confirms equals the recording order that day. So
    the n-th recording of a day goes to the n-th session in the sheet; a rat with
    several sessions in a day therefore gets several folders, one per session.

    Implant status is NOT inferred from file size (a non-implant session can still
    run 40+ minutes); it comes from the sheet's Implant column, carried on the
    roster. The only size test kept is the tiny ``ABORTED`` one that drops a
    few-MB false start so it never consumes a session slot.

    `loose` may include already-filed folders (tagged ``is_filed``, from
    ``find_loose_camera_folders(include_filed=True)``). No folder is trusted as
    correctly placed — an earlier sort may have filed them to the wrong rat, so
    *every* folder is re-derived from the by-time / training-order rule. A filed
    folder already under its re-derived ``Rat<N>/<date>`` is reported "already
    correctly filed"; one under a different rat/date is flagged ``refile`` so it
    can be re-sorted.

    `have_video` is accepted for signature compatibility; positional slotting
    already keeps a loose folder from colliding with another, and the organizer's
    own destination checks prevent any overwrite."""
    # Every (rat, session) slot per date, in training order — NOT deduped, so a
    # rat with several sessions in a day gets one slot per session. Fall back to
    # rat id / session for any roster lacking train_order.
    date_slots: dict = defaultdict(list)
    for e in sorted(roster, key=lambda x: (x.get("train_order", x["rat_no"]),
                                           x["rat_no"], x.get("session", 0))):
        if e["date8"]:
            date_slots[e["date8"]].append(e)

    # Only videos dated within this experiment's window are ours to assign. The
    # drives also hold earlier batches whose loose folders would otherwise be
    # zipped onto this batch's rats by date. Pad a year each way; out-of-window
    # folders are returned, clearly, but never assigned.
    dates = sorted(e["date8"] for e in roster if e["date8"])
    if dates:
        lo = str(int(dates[0][:4]) - 1) + dates[0][4:]
        hi = str(int(dates[-1][:4]) + 1) + dates[-1][4:]
    else:
        lo = hi = ""

    rows = []
    groups: dict = defaultdict(list)          # date8 -> [folder, ...] (loose + filed)
    for v in loose:
        if v["kind"] == ABORTED:
            if not v.get("is_filed"):
                rows.append(dict(v, rat=None, rat_no=None, day="", session="", refile=False,
                                 fills_missing=False, status="aborted false-start (skip)"))
            continue
        if lo and not (lo <= (v["date8"] or "") <= hi):
            if not v.get("is_filed"):
                rows.append(dict(v, rat=None, rat_no=None, day="", session="", refile=False,
                                 fills_missing=False,
                                 status="outside this experiment's dates — different batch"))
            continue
        groups[v["date8"]].append(v)

    # Re-sort EVERY date's folders (loose + filed) by start time and re-derive each
    # one's (rat, session) from the training-order slots — the earlier filing is
    # not trusted. A date is only reported if it has something to act on (a loose
    # folder to file, a misfiled folder to move, or an extra); a date whose videos
    # are all already correctly filed produces no rows.
    for date8 in sorted(groups):
        vids = sorted(groups[date8], key=lambda x: x["start"] or "")
        slots = date_slots.get(date8, [])
        date_rows = []
        actionable = False
        for i, v in enumerate(vids):
            e = slots[i] if i < len(slots) else None
            rat_no = e["rat_no"] if e else None
            day = e["day"] if e else ""
            session = e["session"] if e else ""
            refile = False
            if e is None:
                fills = False
                status = "extra recording — more folders than sessions this date"
                actionable = True
            elif v.get("is_filed"):
                fills = False
                cur_rat, cur_date = _filed_rat_date(v["path"])
                if cur_rat == rat_no and cur_date == date8:
                    status = "already correctly filed"
                else:
                    refile = True
                    actionable = True
                    status = (f"misfiled under Rat{cur_rat}/{cur_date} — belongs to "
                              f"Rat{rat_no}/{date8} s{session}")
            else:
                fills = True
                actionable = True
                status = "fills a session with no video"
            date_rows.append(dict(v, rat=f"Rat{rat_no}" if rat_no else None, rat_no=rat_no,
                                  day=day, session=session, refile=refile,
                                  fills_missing=fills, status=status))
        if actionable:
            rows.extend(date_rows)
    rows.sort(key=lambda r: (r["date8"] or "", r["start"] or ""))
    return rows
