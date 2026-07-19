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


def _is_filed(p: Path) -> bool:
    """True if p already sits under a ``Rat<N>/<YYYYMMDD>/`` path — i.e. it has
    been sorted to a rat and is not loose."""
    parts = p.parts
    for i, seg in enumerate(parts):
        if _DATE_RE.match(seg) and i >= 1 and sd._rat_of(parts[i - 1]) is not None:
            return True
    return False


def find_loose_camera_folders(roots, max_depth: int = 6,
                              on_dir=None, should_stop=None) -> list[dict]:
    """Camera-video folders under `roots` that are NOT already filed to a rat.

    Walks each root to `max_depth`, skipping the system/junk folders scan_drive
    already knows to avoid. A directory counts as a camera folder when it holds
    at least one ``eye<NN>_*.mp4``; filed ones (already under Rat<N>/<date>) are
    left out, since those are exactly the videos that are not lost."""
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
            if info and not _is_filed(d):
                if info["path"] not in seen:
                    seen.add(info["path"])
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
    """Attach each real loose camera folder to a rat + date from the rig rules.

    Within one date, the videos of a given type (implanted vs not) are ordered by
    start time and matched to that type's rats in ascending id order — smaller id
    first, exactly as the rig records. Aborted folders are returned but never
    assigned.

    `have_video`, if given, is the set of (rat_no, date8) that already have video
    on disk (from the coverage scan). It lets each row say whether the video fills
    a genuine gap or merely duplicates video the session already has — so acting
    on the finds never overwrites good data with a stray copy."""
    have_video = have_video or set()
    impl_rats = sorted({e["rat_no"] for e in roster if e["implanted"]})
    noni_rats = sorted({e["rat_no"] for e in roster if not e["implanted"]})
    roster_by = {(e["rat_no"], e["date8"]): e for e in roster if e["date8"]}

    # Only videos dated within this experiment's window are ours to assign. The
    # drives also hold earlier batches (Rat1/Rat2 from 2020) whose loose folders
    # would otherwise be zipped onto this batch's rats by date — a 2020 clip
    # landing under Rat5. Pad a year each way so a session just outside the logged
    # span still counts; a stray sheet date can only widen this, never hide a real
    # video. Out-of-window folders are returned, clearly, but never assigned.
    dates = sorted(e["date8"] for e in roster if e["date8"])
    if dates:
        lo = str(int(dates[0][:4]) - 1) + dates[0][4:]
        hi = str(int(dates[-1][:4]) + 1) + dates[-1][4:]
    else:
        lo = hi = ""

    rows = []
    groups: dict = defaultdict(list)
    for v in loose:
        if v["kind"] == ABORTED:
            rows.append(dict(v, rat=None, rat_no=None, day="", session="",
                             fills_missing=False, status="aborted false-start (skip)"))
            continue
        if lo and not (lo <= (v["date8"] or "") <= hi):
            rows.append(dict(v, rat=None, rat_no=None, day="", session="",
                             fills_missing=False,
                             status="outside this experiment's dates — different batch"))
            continue
        groups[(v["date8"], v["kind"])].append(v)

    for (date8, kind), vids in sorted(groups.items()):
        vids.sort(key=lambda x: x["start"] or "")
        rats = noni_rats if kind == NON_IMPLANT else impl_rats
        for i, v in enumerate(vids):
            rat_no = rats[i] if i < len(rats) else None
            e = roster_by.get((rat_no, date8)) if rat_no else None
            fills = False
            if rat_no is None:
                status = "extra recording — more videos than rats this date"
            elif e is None:
                status = "no roster session for this rat/date"
            elif (rat_no, date8) in have_video:
                status = "session already has video (possible duplicate)"
            else:
                status = "fills a session with no video"
                fills = True
            rows.append(dict(v, rat=f"Rat{rat_no}" if rat_no else None, rat_no=rat_no,
                             day=e["day"] if e else "", session=e["session"] if e else "",
                             fills_missing=fills, status=status))
    rows.sort(key=lambda r: (r["date8"] or "", r["start"] or ""))
    return rows
