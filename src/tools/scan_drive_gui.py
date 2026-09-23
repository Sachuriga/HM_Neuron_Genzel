#!/usr/bin/env python3
"""Drive-coverage checker — a Qt front-end that cross-checks a HexMaze
experiment spreadsheet against what is actually stored on the acquisition
drives.

Workflow:

  1. Browse to the experiment Excel (the ``Raw`` sheet, one row per trial).
     Every unique (subject, day, session, Date) becomes an *expected*
     rat-session, and the ``Implant`` column decides what data it should have:

         Implant == 0  ->  camera video only          (e.g. Rat3 / Rat4)
         Implant == 1  ->  video + ephys pre/task/post (e.g. Rat5 / Rat6)

  2. Browse to up to 4 drive-root folders (data is spread across several
     drives). Each root is laid out as ``Rat<N>_*/<YYYYMMDD>/...`` — the same
     layout ``scan_drive.py`` already understands, whose scanning helpers this
     GUI reuses.

  3. Press *Scan drives*. For every expected rat-session the tool looks for a
     matching ``Rat<N>/<YYYYMMDD>`` folder across all 4 roots and reports which
     data is present and which is MISSING.

  4. Press *Find scattered data* when something comes back MISSING. Step 3 only
     looks directly under the chosen drive folders; this walks the WHOLE drive
     each of them lives on, to a chosen depth, so it finds sessions filed
     somewhere odd anywhere on that drive (a backup folder, a per-experimenter
     folder, a nested copy). It reports data found *nowhere* on those drives,
     folders on disk the sheet never asked for (*orphans*), and unfiled camera
     videos.

Launch with:  python scan_drive_gui.py
"""

from __future__ import annotations

import os
import sys
import datetime
import subprocess
from pathlib import Path

import pandas as pd

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QPixmap, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QComboBox, QDialog, QCheckBox,
    QSpinBox, QProgressBar, QScrollArea, QListWidget, QWidgetAction, QInputDialog,
)

# Reuse the drive-scanning helpers from the sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_drive as sd  # noqa: E402
import organize_data as od  # noqa: E402
import reset_videos as rv  # noqa: E402
import fix_video_names as fx  # noqa: E402
import find_videos as fvid  # noqa: E402
import summarize as smz  # noqa: E402
import prepare_meta as pmeta  # noqa: E402
import preprocess_check as ppc  # noqa: E402

EXPECTED_PHASES = ("pre", "task", "post")
# The pre/task/post columns show each phase's ephys size in GB (✗ if absent),
# so a phase that is present but too small is visible, not just presence.
_PHASE_COL = {"pre": "pre GB", "task": "task GB", "post": "post GB"}
COLUMNS = ["Rat", "Date", "day", "session", "repeat", "Expected",
           "Video", "pre GB", "task GB", "post GB", "Split", "Found in", "Status"]
# Columns rendered centred, and the one that carries the status text.
_CENTER_COLS = {"day", "session", "repeat", "Video",
                "pre GB", "task GB", "post GB", "Status"}
_STATUS_COL = COLUMNS.index("Status")

# All the row tints below are light, so every cell that gets one also needs a
# dark foreground: on a dark OS theme Qt's default text colour is near-white,
# which is unreadable on a pale background. Paired via _paint() at each site.
_INK = QColor(26, 26, 26)

# Status -> row background tint (light, works on default palette).
_STATUS_COLOR = {
    "OK": QColor(210, 244, 214),
    "PARTIAL": QColor(255, 244, 205),
    "MISSING": QColor(250, 214, 214),
    "?": QColor(238, 238, 238),
}


def _paint(item, bg):
    """Give a table cell a light background and a dark, legible foreground."""
    item.setBackground(QBrush(bg))
    item.setForeground(QBrush(_INK))


def _xlsx(path: str) -> Path:
    """Force an .xlsx suffix on whatever the save dialog returned.

    The dialog only suggests a name — a user who types "coverage" or leaves an
    old ".csv" on it would otherwise get a workbook Excel refuses to open by
    double-click.
    """
    p = Path(path)
    return p if p.suffix.lower() == ".xlsx" else p.with_suffix(".xlsx")


def _rep(v) -> str:
    """Render a repeat number for a table / CSV cell ('' when the sheet has none)."""
    return "" if v is None else str(v)


# ------------------------------------------------------------------
#                       spreadsheet -> roster
# ------------------------------------------------------------------
def parse_date8(val) -> str | None:
    """Normalise a spreadsheet Date cell to a ``YYYYMMDD`` drive-folder name.

    The column mixes ``DD.MM.YYYY`` strings and real Timestamps; handle both."""
    try:
        # pd.NaT (a blank cell in a date-typed column) is a datetime subclass with
        # no strftime — it must be caught here, before the Timestamp branch.
        if val is None or pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return val.strftime("%Y%m%d")
    s = str(val).strip()
    if not s:
        return None
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
                "%Y%m%d", "%d.%m.%y"):
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y%m%d")
    except Exception:
        return None


def _find_col(df, name: str):
    """Case-insensitive column lookup; None if the sheet doesn't have it."""
    for c in df.columns:
        if str(c).strip().lower() == name:
            return c
    return None


def build_roster(xlsx_path: str, sheet: str = "Raw") -> list[dict]:
    """Read the spreadsheet and return one dict per expected rat-session."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    required = ["subject", "day", "session", "Date", "Implant"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sheet '{sheet}' is missing column(s): {', '.join(missing)}")

    # a row without subject/day/session can't name a session — skip it rather
    # than letting int(NaN) abort the whole roster
    sub = df.dropna(subset=["subject"]).copy()
    incomplete = sub["day"].isna() | sub["session"].isna()
    if incomplete.any():
        print(f"  roster: skipped {int(incomplete.sum())} row(s) with a blank day/session")
        sub = sub[~incomplete]
    sub["date8"] = sub["Date"].apply(parse_date8)
    # Row order in the sheet IS the training order: on a given date the animals
    # appear in the sequence they were run. dropna() keeps that order, so a
    # monotonic counter over the retained rows records each row's sheet position;
    # a group's smallest position is where that session first appears — its rank
    # in the training order. Used to assign loose videos to rats by time.
    sub["_rowpos"] = range(len(sub))
    # 'repeat' is the repetition block a session belongs to. It is constant
    # within a (subject, day, session), so it rides along as an aggregate rather
    # than a grouping key — that way a stray value can never split one real
    # session into two roster rows. Optional: older sheets may not have it.
    rep_col = _find_col(sub, "repeat")
    sub["_repeat"] = sub[rep_col] if rep_col else pd.NA
    grouped = (sub.groupby(["subject", "day", "session", "date8", "Implant"], dropna=False)
                  .agg(trials=("subject", "size"), repeat=("_repeat", "first"),
                       first_row=("_rowpos", "min"))
                  .reset_index())

    roster = []
    for _, r in grouped.iterrows():
        # A blank Implant cell reads as NaN; treat it as "not implanted" instead of
        # letting int(NaN) abort the whole roster.
        implanted = (not pd.isna(r["Implant"])) and int(r["Implant"]) == 1
        rep = r["repeat"]
        # parse_date8 says None for a date it cannot read, but grouping turns
        # that None into NaN — a float, and a *truthy* one. Left as-is it defeats
        # every `date8 or "?"` fallback and every `if date8 is None` guard
        # downstream, and reaches the table as a float Qt refuses. Put None back.
        d8 = r["date8"]
        roster.append(dict(
            rat_no=int(r["subject"]),
            rat=f"Rat{int(r['subject'])}",
            date8=None if pd.isna(d8) else str(d8),
            day=int(r["day"]),
            session=int(r["session"]),
            repeat=None if pd.isna(rep) else int(rep),
            implanted=implanted,
            trials=int(r["trials"]),
            train_order=int(r["first_row"]),
            expected="video + ephys (pre/task/post)" if implanted else "video",
        ))
    roster.sort(key=lambda x: (x["rat_no"], x["day"], x["session"]))
    return roster


# ------------------------------------------------------------------
#                       drives -> session index
# ------------------------------------------------------------------
def index_drives(roots: list[str]) -> dict:
    """Map (rat_no, YYYYMMDD) -> list of (drive_label, session_path) across all
    selected roots."""
    idx: dict = {}
    for i, root in enumerate(roots):
        if not root:
            continue
        p = Path(root)
        if not p.exists():
            continue
        label = f"Drive {i + 1}"
        for sess in sd.find_sessions(p):
            rat_no = sd._rat_of(sess.parent.name)
            if rat_no is None:
                continue
            idx.setdefault((rat_no, sess.name), []).append((label, sess))
    return idx


def _split_text(splits: dict) -> str:
    """Render a {phase: parts} split map as e.g. 'post×2, task×2' (phase order)."""
    return ", ".join(f"{p}×{splits[p]}" for p in EXPECTED_PHASES if p in splits)


def inspect_session(sess: Path) -> dict:
    """Inventory a single located session folder. Returns n_video, the set of
    ephys phases present, and any within-session recording splits, by reusing
    scan_drive's scan_session + split detector."""
    issues, inv_rows, file_rows = [], [], []
    try:
        sd.scan_session(sess, issues, inv_rows, file_rows)
    except OSError:
        return dict(n_video=0, phases=set(), splits={})
    splits = sd.detect_session_splits(sess)
    if not inv_rows:
        return dict(n_video=0, phases=set(), splits=splits, phase_bytes={})
    iv = inv_rows[0]
    phases = set(p for p in EXPECTED_PHASES if p in (iv.get("phases") or "").split("+"))
    return dict(n_video=int(iv.get("n_video", 0)), phases=phases, splits=splits,
                phase_bytes=iv.get("phase_bytes", {}),
                n_rec=int(iv.get("n_rec", 0)), n_merged=int(iv.get("n_merged", 0)),
                n_logger=int(iv.get("n_logger", 0)))


def evaluate(entry: dict, idx: dict) -> dict:
    """Compare one roster entry against the drive index; fill result fields."""
    res = dict(entry)
    res.update(n_video=0, phases=set(), found_in="", status="MISSING",
               n_rec=0, n_merged=0, n_logger=0, paths=[], splits={}, split_text="",
               phase_gb={p: 0.0 for p in EXPECTED_PHASES}, short_phases=[], reasons=[])

    if entry["date8"] is None:
        res["status"] = "MISSING"
        res["found_in"] = "(bad date in sheet)"
        return res

    matches = idx.get((entry["rat_no"], entry["date8"]), [])
    if not matches:
        return res

    labels, n_video, phases = [], 0, set()
    n_rec = n_merged = n_logger = 0
    paths = []
    splits: dict = {}
    phase_bytes = {p: 0 for p in EXPECTED_PHASES}
    for label, sess in matches:
        info = inspect_session(sess)
        n_video += info["n_video"]
        phases |= info["phases"]
        n_rec += info.get("n_rec", 0)
        n_merged += info.get("n_merged", 0)
        n_logger += info.get("n_logger", 0)
        for ph, k in info.get("splits", {}).items():   # keep the largest part-count seen
            splits[ph] = max(splits.get(ph, 0), k)
        for ph, b in (info.get("phase_bytes") or {}).items():
            # keep the largest per phase across copies — duplicates hold the same
            # data, so summing them would double-count a phase that sits on 2 drives
            if ph in phase_bytes:
                phase_bytes[ph] = max(phase_bytes[ph], b)
        labels.append(label)
        paths.append(str(sess))
    res.update(n_video=n_video, phases=phases, found_in=", ".join(sorted(set(labels))),
               n_rec=n_rec, n_merged=n_merged, n_logger=n_logger, paths=paths,
               splits=splits, split_text=_split_text(splits),
               phase_bytes=dict(phase_bytes),
               phase_gb={p: round(phase_bytes[p] / 1e9, 1) for p in EXPECTED_PHASES})

    # Preliminary completeness by presence only. Whether each present phase is the
    # right SIZE is judged later against the rat's own median (needs every session
    # first), in flag_short_phases(); that pass may downgrade OK -> PARTIAL.
    have_video = n_video > 0
    if entry["implanted"]:
        have_all_phases = all(p in phases for p in EXPECTED_PHASES)
        res["status"] = "OK" if (have_video and have_all_phases) else "PARTIAL"
        if not have_video:
            res["reasons"].append("no video")
        for p in EXPECTED_PHASES:
            if p not in phases:
                res["reasons"].append(f"no {p}")
    else:
        res["status"] = "OK" if have_video else "PARTIAL"
        if not have_video:
            res["reasons"].append("no video")
    return res


# A phase is flagged "short" when its size falls below this fraction of that
# rat's own median for the same phase — self-calibrating, so it adapts to each
# animal's channel count and the rig, with no absolute GB baked in.
SHORT_FRACTION = 0.6
# Below this many sessions for a rat, its median is too shaky to judge against.
_MIN_SESSIONS_FOR_MEDIAN = 3


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def _phase_gb_raw(r: dict, p: str) -> float:
    """Unrounded GB of phase `p` — from raw bytes when present. The rounded
    phase_gb (0.1 GB) turns a 40 MB truncated phase into 0.0, which would read as
    'absent' and escape the short-phase check."""
    pb = r.get("phase_bytes")
    if pb:
        return pb.get(p, 0) / 1e9
    return (r.get("phase_gb") or {}).get(p, 0.0)


def phase_medians(results: list) -> dict:
    """Per-rat median GB for each phase, over that rat's implanted sessions that
    actually have the phase. Only non-zero sizes count, so a missing phase never
    drags the median down. Returns {rat_no: {phase: median_gb}}.

    Each (rat, date) folder counts once: several sessions run on one day share one
    folder, and counting it per session would weight that day twice."""
    by_rat: dict = {}
    seen = set()
    for r in results:
        if not r or not r.get("implanted"):
            continue
        key = (r["rat_no"], r.get("date8"))
        if r.get("date8") and key in seen:
            continue
        seen.add(key)
        for p in EXPECTED_PHASES:
            gb = _phase_gb_raw(r, p)
            if gb > 0:
                by_rat.setdefault(r["rat_no"], {}).setdefault(p, []).append(gb)
    # Only trust a median built from enough sessions; too few and one short
    # recording would set a baseline that flags the healthy ones.
    return {rat: {p: _median(v) for p, v in phs.items()
                  if len(v) >= _MIN_SESSIONS_FOR_MEDIAN}
            for rat, phs in by_rat.items()}


def flag_short_phases(res: dict, medians: dict) -> None:
    """Mark phases whose size is well below the rat's median, and downgrade the
    session's status accordingly. A short phase is present but too small — a pre
    that ran minutes not hours, a truncated post — which the presence-only check
    cannot see. Mutates `res` in place (short_phases, reasons, status)."""
    if not res or not res.get("implanted"):
        return
    med = medians.get(res["rat_no"], {})
    for p in EXPECTED_PHASES:
        gb = _phase_gb_raw(res, p)
        m = med.get(p, 0.0)
        # only judge a phase that is present (by presence, not by a rounded size —
        # a near-empty phase is the most truncated of all), against a trusted median
        if p in (res.get("phases") or set()) and m > 0 and gb < SHORT_FRACTION * m:
            res["short_phases"].append(p)
            res["reasons"].append(f"{p} short ({gb:.1f} GB vs ~{m:.0f})")
    if res["short_phases"] and res["status"] == "OK":
        res["status"] = "PARTIAL"


# ------------------------------------------------------------------
#              search every drive for scattered session data
# ------------------------------------------------------------------
def search_all_drives(roots_selected: list[str], max_depth: int = 6,
                      on_dir=None, should_stop=None) -> dict:
    """Hunt for Rat<N>/<YYYYMMDD> session folders WITHIN the given roots, at any
    depth (not just the structured 2-level layout the main scan reads). The GUI
    passes the whole drive each listed folder lives on (see _search_roots); drives
    with no listed folder on them are NOT walked.

    Returns (rat_no, date8) -> list of dicts with the path, the root it lives
    under, and what it contains. Use it to surface sessions filed somewhere odd
    on the drives, plus on-disk folders the sheet never asked for (orphans)."""
    selected = []
    for r in roots_selected:
        if r:
            try:
                selected.append(Path(r).resolve())
            except OSError:
                pass

    idx: dict = {}
    for vol in selected:                       # only the roots passed in
        if should_stop is not None and should_stop():
            break
        for sess in sd.find_sessions_deep(vol, max_depth=max_depth,
                                          on_dir=on_dir, should_stop=should_stop):
            rat_no = sd._rat_of(sess.parent.name)
            if rat_no is None:
                continue
            info = inspect_session(sess)
            idx.setdefault((rat_no, sess.name), []).append(dict(
                path=str(sess), volume=str(vol),
                n_video=info["n_video"], phases=info["phases"], splits=info["splits"],
                phase_bytes=info.get("phase_bytes", {}),
                n_rec=info.get("n_rec", 0), n_merged=info.get("n_merged", 0),
                n_logger=info.get("n_logger", 0)))
    return idx


def scatter_report(roster: list[dict], found: dict, videos=None) -> list[dict]:
    """Turn a search_all_drives() index + the roster into reviewable rows.

    Two kinds of row, the two ways the drives and the sheet can disagree
    (the search covers only the drives the selected folders live on, so every
    hit is on one of them):
      not-found  — an expected session found on none of the searched drives
      orphan     — a session folder on disk that the sheet never asked for
                   (a typo'd folder, a stray copy, an unlogged recording)

    `videos` is the assign_videos() result — the loose camera folders matched to
    a rat/date. It folds the video search into the session search so that, for an
    IMPLANTED rat, one row shows the whole session (pre/task/post ephys AND its
    video) instead of splitting the ephys and the video onto two rows. It also
    lets a session that has no ephys folder but whose video was found in the dump
    count as found rather than missing.

    Orphans are limited to this experiment: a rat the roster mentions AND a date
    inside the roster's own date span. The drives also carry earlier batches, and
    those reuse the same rat numbers (2020-era `Rat4` is a different animal from
    this batch's `Rat4_491389`) — so the rat number alone can't tell them apart
    and the date window has to. Without it the real anomalies drown in hundreds
    of rows about experiments you're not checking."""
    rows = []
    videos = videos or []
    # A session's video, located loose in the dump, keyed by (rat_no, date8).
    dump_video = {(v["rat_no"], v["date8"]): v for v in videos if v.get("rat_no")}
    video_fills = {(v["rat_no"], v["date8"]) for v in videos if v.get("fills_missing")}
    expected = {(e["rat_no"], e["date8"]) for e in roster if e["date8"]}
    roster_rats = {e["rat_no"] for e in roster}
    # Date window, padded a year each way so a session recorded slightly outside
    # the logged span still counts. A typo'd date in the sheet can only widen
    # this, which at worst shows a few extra rows — it can never hide a real one.
    dates = sorted(e["date8"] for e in roster if e["date8"])
    if dates:
        lo = str(int(dates[0][:4]) - 1) + dates[0][4:]
        hi = str(int(dates[-1][:4]) + 1) + dates[-1][4:]
    else:
        lo = hi = ""

    def _merge_splits(hits):
        """Combine the split maps of every copy of a session into one, keeping the
        largest part-count seen for each phase."""
        merged: dict = {}
        for h in hits:
            for ph, k in (h.get("splits") or {}).items():
                merged[ph] = max(merged.get(ph, 0), k)
        return merged

    def _agg_phase_bytes(hits):
        """Per-phase bytes for a session, max per phase across its copies
        (duplicates hold the same data, so max not sum)."""
        pb = {p: 0 for p in EXPECTED_PHASES}
        for h in hits:
            for ph, b in (h.get("phase_bytes") or {}).items():
                if ph in pb:
                    pb[ph] = max(pb[ph], b)
        return pb

    def _agg_phase_gb(hits):
        pb = _agg_phase_bytes(hits)
        return {p: round(pb[p] / 1e9, 1) for p in EXPECTED_PHASES}

    # Per-rat phase medians, from every session found on any drive (not just the
    # scattered ones), so an implanted session shown here can be flagged when a
    # phase is short against that rat's own norm — same rule as the coverage table.
    implanted_rats = {e["rat_no"] for e in roster if e["implanted"]}
    allsess = [dict(rat_no=rn, implanted=(rn in implanted_rats),
                    phase_bytes=_agg_phase_bytes(hits))
               for (rn, _d), hits in found.items()]
    medians = phase_medians(allsess)

    def _video_where(rat_no, date8, n_video):
        """(label, path) for a session's video: in the session folder, in the dump,
        or absent. Lets an implanted session's video ride on its own row instead of
        a separate one."""
        if n_video > 0:
            return f"{n_video}", ""            # video is with the session
        v = dump_video.get((rat_no, date8))
        if v:
            return "dump", v["path"]           # located loose in the acquisition dump
        return "✗", ""

    def _missing(rat_no, phases, n_video, date8):
        """Which of an implanted session's four expected files are absent: pre,
        task, post, and video. Video counts as present if it is in the session
        OR was located loose in the dump. Empty for a video-only animal."""
        if rat_no not in implanted_rats:
            return []
        m = [p for p in EXPECTED_PHASES if p not in phases]
        if not (n_video > 0 or (rat_no, date8) in video_fills):
            m.append("video")
        return m

    def _session_row(kind, rat, date8, day, session, repeat, hits, tail, rat_no=None):
        """One row per (rat, date) session — never one per split piece. Lists
        every volume the session's pieces live on, marks a PC-error recording
        split, and (for implanted rats) shows each phase's GB, flags any short
        against the rat's median, and flags which of the 4 files are missing."""
        vols = sorted({h["volume"] for h in hits})
        paths = [h["path"] for h in hits]
        splits = _merge_splits(hits)
        agg_phases = set().union(*(h.get("phases", set()) for h in hits)) if hits else set()
        n_video = sum(h.get("n_video", 0) for h in hits)
        n_rec = sum(h.get("n_rec", 0) for h in hits)
        phase_gb = _agg_phase_gb(hits)
        raw_gb = {p: b / 1e9 for p, b in _agg_phase_bytes(hits).items()}
        short = []
        if rat_no in implanted_rats:
            med = medians.get(rat_no, {})
            for p in EXPECTED_PHASES:
                # presence, not rounded size: a near-empty phase is the most short
                if p in agg_phases and med.get(p, 0) > 0 and raw_gb[p] < SHORT_FRACTION * med[p]:
                    short.append(p)
        missing = _missing(rat_no, agg_phases, n_video, date8)
        vlabel, vpath = _video_where(rat_no, date8, n_video)
        if rat_no not in implanted_rats:
            vlabel, vpath = "", ""             # video-only rats: no ephys/video split to show
        parts = f" · split: {_split_text(splits)}" if splits else ""
        # More than one folder for one session is itself a form of splitting —
        # the same session recorded at different times / kept on different drives.
        across = f" · in {len(paths)} folders" if len(paths) > 1 else ""
        shorts = f" · short: {'+'.join(short)}" if short else ""
        miss = f" · MISSING: {'+'.join(missing)}" if missing else ""
        vid = f" · video: {vlabel}" if vlabel and vlabel != "✗" else ""
        return dict(kind=kind, rat=rat, date8=date8, day=day, session=session,
                    repeat=repeat, volume=", ".join(vols), volumes=vols,
                    path=paths[0] if paths else "", paths=paths,
                    split=bool(splits), split_text=_split_text(splits),
                    phase_gb=phase_gb, short_phases=short, missing=missing, phases=agg_phases,
                    video=vlabel, video_path=vpath,
                    detail=f"{n_video} video, {n_rec} rec — {tail}{parts}{across}{shorts}{miss}{vid}")
    # One row per (rat, date): a rat's several sessions on one day share a folder,
    # so a per-session loop would repeat the same not-found row for each of them.
    by_key: dict = {}
    for ent in roster:
        if ent["date8"]:
            by_key.setdefault((ent["rat_no"], ent["date8"]), []).append(ent)
    for ents in by_key.values():
        e = dict(ents[0], session="+".join(str(x["session"]) for x in ents))
        hits = found.get((e["rat_no"], e["date8"]), [])
        if not hits:
            if (e["rat_no"], e["date8"]) in video_fills:
                # No session folder anywhere, but the video WAS found loose in the
                # dump. For a video-only animal that video is the whole session, so
                # it is not missing — the 'video' row represents it. For an
                # implanted animal the ephys is still absent, so keep a row but say
                # so plainly instead of the misleading 'in no drive folder'.
                if not e["implanted"]:
                    continue
                vlabel, vpath = _video_where(e["rat_no"], e["date8"], 0)
                miss = _missing(e["rat_no"], set(), 0, e["date8"])
                rows.append(dict(kind="not-found", rat=e["rat"], date8=e["date8"],
                                 day=e["day"], session=e["session"], repeat=e["repeat"],
                                 volume="", volumes=[], path="", paths=[],
                                 split=False, split_text="", phase_gb={}, short_phases=[],
                                 missing=miss, phases=set(), video=vlabel, video_path=vpath,
                                 detail="ephys in no selected drive folder — video found in dump"
                                        + (f" · MISSING: {'+'.join(miss)}" if miss else "")))
                continue
            miss = _missing(e["rat_no"], set(), 0, e["date8"])
            rows.append(dict(kind="not-found", rat=e["rat"], date8=e["date8"],
                             day=e["day"], session=e["session"], repeat=e["repeat"],
                             volume="", volumes=[], path="", paths=[],
                             split=False, split_text="", phase_gb={}, short_phases=[],
                             missing=miss, phases=set(),
                             video="✗" if e["implanted"] else "", video_path="",
                             detail="in no selected drive folder"
                                    + (f" · MISSING: {'+'.join(miss)}" if miss else "")))
            continue

    for (rat_no, date8), hits in sorted(found.items()):
        if (rat_no, date8) in expected or rat_no not in roster_rats:
            continue
        if lo and not (lo <= date8 <= hi):      # an earlier batch, not this one
            continue
        # One orphan row per session, not per split piece: a session the PC error
        # broke into several folders is still a single unlogged session, not a
        # crowd of anomalies.
        rows.append(_session_row("orphan", f"Rat{rat_no}", date8, "", "", "",
                                 hits, "not in the spreadsheet", rat_no=rat_no))
    order = {"not-found": 0, "orphan": 1}
    rows.sort(key=lambda r: (order[r["kind"]], r["rat"], str(r["date8"])))
    return rows


def video_scatter_rows(videos: list[dict], implanted_rats=None) -> list[dict]:
    """Render assign_videos() results as scatter rows for VIDEO-ONLY animals, so
    an unfiled Rat3/Rat4 recording (which *is* the whole session) shows up in the
    'find all the data' list. Implanted rats are skipped here — their video is
    folded into the session row instead, so the ephys and video are never split
    onto two rows. Aborted / other-batch folders carry no rat and are skipped."""
    implanted_rats = implanted_rats or set()
    out = []
    for v in videos:
        if not v.get("rat_no") or v["rat_no"] in implanted_rats:
            continue
        if v.get("is_filed"):        # already under Rat/date — not lost, don't list it
            continue
        vol = str(Path(v["path"]).drive or Path(v["path"]).anchor).upper()
        out.append(dict(kind="video", rat=v.get("rat") or "?", date8=v.get("date8") or "?",
                        day=v.get("day", ""), session=v.get("session", ""),
                        repeat=v.get("repeat", ""), volume=vol, volumes=[vol],
                        path=v["path"], paths=[v["path"]], split=False, split_text="",
                        phase_gb={}, short_phases=[], missing=[], phases=set(),
                        detail=f"{v['n_cam']} cam, {v['avg_mb']:.0f} MB/cam "
                               f"({v['kind']}) — {v['status']}"))
    out.sort(key=lambda r: (str(r["date8"]), r["rat"]))
    return out


class SearchWorker(QObject):
    """Runs search_all_drives() off the GUI thread.

    Emits both shapes of the result: `done` carries the digested scatter rows for
    the report dialog, `raw_index` the underlying (rat, date) -> hits index that
    the organize planner needs. One walk of the drives serves both."""
    progress = pyqtSignal(str)               # current directory
    done = pyqtSignal(list, int)             # scatter rows, sessions seen
    # 'object', not 'dict': the index is keyed by (rat_no, date8) tuples and its
    # hits hold sets, none of which survive Qt's QVariantMap conversion.
    raw_index = pyqtSignal(object)           # (rat_no, date8) -> [hit]
    failed = pyqtSignal(str)

    def __init__(self, roster, roots, depth, want_rows=True):
        super().__init__()
        self.roster, self.roots = roster, roots
        # Organize / summary / meta only need raw_index; skip the scatter digest
        # for them so the thread ends as soon as the index is out.
        self.depth, self.want_rows = depth, want_rows
        self._stop = False
        self._n = 0

    def stop(self):
        self._stop = True

    def _tick(self, d: Path):
        self._n += 1
        if self._n % 400 == 0:               # cheap throttle — don't flood the UI
            self.progress.emit(str(d))

    def run(self):
        try:
            found = search_all_drives(self.roots, max_depth=self.depth,
                                      on_dir=self._tick,
                                      should_stop=lambda: self._stop)
            # Also find camera videos that were never sorted into a rat folder
            # (the raw acquisition dump), and match each to a rat/session. This is
            # the other half of "find all the data".
            have_video = {k for k, hits in found.items()
                          if any(h.get("n_video", 0) for h in hits)}
            vols = [r for r in self.roots if r]    # the same roots as the session search
            # include_filed=True: already-sorted camera folders are not re-filed,
            # but assign_videos needs them to hold their slot in the by-time
            # ordering so a still-loose folder gets the right session number.
            loose = fvid.find_loose_camera_folders(
                vols, max_depth=self.depth, on_dir=self._tick,
                should_stop=lambda: self._stop, include_filed=True)
            videos = fvid.assign_videos(loose, self.roster, have_video)

            self.raw_index.emit({"sessions": found, "videos": videos})
            if not self.want_rows:
                return
            # Fold each implanted session's video into its own row (pre/task/post
            # + video together); video-only rats still get a standalone video row.
            impl = {e["rat_no"] for e in self.roster if e["implanted"]}
            rows = scatter_report(self.roster, found, videos) \
                + video_scatter_rows(videos, impl)
            self.done.emit(rows, len(found))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class PreprocessWorker(QObject):
    """Walks the HM_neuron_preprocess trees and reports per-session step status."""
    progress = pyqtSignal(str)
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, roster, roots):
        super().__init__()
        self.roster, self.roots = roster, roots
        self._stop = False
        self._n = 0

    def stop(self):
        self._stop = True

    def _tick(self, d):
        self._n += 1
        if self._n % 200 == 0:
            self.progress.emit(str(d))

    def run(self):
        try:
            idx = ppc.find_preprocess_sessions(self.roots, on_dir=self._tick,
                                               should_stop=lambda: self._stop)
            self.done.emit(ppc.build_preprocess(self.roster, idx))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class MetaWorker(QObject):
    """Plans RecordingMeta.xlsx files off the GUI thread (plan_meta walks every
    session's folders on the externals). In `auto` mode (after a scan, no review)
    it also writes — but only rows whose session<->camera-folder pairing is
    unambiguous; anything with mismatched counts is left for the reviewed dialog."""
    done = pyqtSignal(list, list, bool)       # plan, write results, auto
    failed = pyqtSignal(str, bool)            # message, auto

    def __init__(self, roster, found, raw, auto):
        super().__init__()
        self.roster, self.found, self.raw, self.auto = roster, found, raw, auto

    def run(self):
        try:
            plan = pmeta.plan_meta(self.roster, self.found, self.raw)
            results = []
            if self.auto:
                safe = [p for p in plan
                        if p["action"] == pmeta.WRITE and not p.get("ambiguous")]
                results = pmeta.write_plan(safe) if safe else []
            self.done.emit(plan, results, self.auto)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc), self.auto)


# ------------------------------------------------------------------
#                       background scan worker
# ------------------------------------------------------------------
class ScanWorker(QObject):
    progress = pyqtSignal(int, int)          # done, total
    row_done = pyqtSignal(int, dict)         # row index, result
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, roster: list[dict], roots: list[str]):
        super().__init__()
        self.roster = roster
        self.roots = roots
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            idx = index_drives(self.roots)
            total = len(self.roster)
            # Pass 1: measure every session (presence + per-phase sizes).
            results = []
            for i, entry in enumerate(self.roster):
                if self._stop:                 # window closing — abandon quietly
                    return
                results.append(evaluate(entry, idx))
                self.progress.emit(i + 1, total)
            # Pass 2: now that every session is measured, flag phases that are
            # short against the rat's own median, then emit the finished rows.
            medians = phase_medians(results)
            for i, res in enumerate(results):
                flag_short_phases(res, medians)
                self.row_done.emit(i, res)
            self.finished.emit()
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


# ------------------------------------------------------------------
#                    scattered-data results dialog
# ------------------------------------------------------------------
_SCATTER_COLS = ["Kind", "Rat", "Date", "day", "session", "repeat",
                 "pre GB", "task GB", "post GB", "Video", "Missing", "Split",
                 "Volume(s)", "Path", "Detail"]
_KIND_COLOR = {
    "not-found": QColor(250, 214, 214),      # on none of the searched drives
    "orphan":    QColor(222, 233, 250),      # on disk, absent from the sheet
    "video":     QColor(214, 234, 250),      # unfiled camera video, matched to a rat
}


class ScatterDialog(QDialog):
    """Shows where expected-but-missing sessions actually live, plus any session
    folders on the drives that the spreadsheet never asked for."""

    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        self.rows = rows
        self.setWindowTitle("Scattered data — where the missing sessions are")
        self.resize(1150, 620)
        v = QVBoxLayout(self)

        n = {k: sum(1 for r in rows if r["kind"] == k)
             for k in ("not-found", "orphan", "video")}
        n_split = sum(1 for r in rows if r.get("split"))
        head = QLabel(
            f"<b>{n['not-found']}</b> found in no selected drive folder · "
            f"<b>{n['orphan']}</b> folder(s) on disk not in the spreadsheet · "
            f"<b>{n['video']}</b> unfiled camera video(s) · "
            f"<b>{n_split}</b> with PC-error split recordings")
        v.addWidget(head)

        self.table = QTableWidget(len(rows), len(_SCATTER_COLS))
        self.table.setHorizontalHeaderLabels(_SCATTER_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._open)
        for i, r in enumerate(rows):
            split_cell = f"⚠ {r['split_text']}" if r.get("split") else ""
            pg = r.get("phase_gb") or {}
            short = set(r.get("short_phases", []))
            missing = set(r.get("missing", []))
            phases = r.get("phases", set())
            # per-phase GB: value if present, ✗ if a missing phase, blank otherwise
            phase_vals = []
            for p in EXPECTED_PHASES:
                if p in missing:
                    phase_vals.append("✗")
                elif p in phases or pg.get(p, 0):
                    phase_vals.append(f"{pg.get(p, 0):.0f}⚠" if p in short else f"{pg.get(p, 0):.0f}")
                else:
                    phase_vals.append("")
            miss_cell = "+".join(r.get("missing", []))
            video_cell = r.get("video", "")
            vals = [r["kind"], r["rat"], str(r["date8"]), str(r["day"]), str(r["session"]),
                    _rep(r["repeat"] or None), *phase_vals, video_cell, miss_cell, split_cell,
                    r.get("volume", ""), r.get("path", ""), r["detail"]]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(val)
                _paint(it, _KIND_COLOR.get(r["kind"], Qt.GlobalColor.white))
                self.table.setItem(i, c, it)
            # red-flag short phase cells, missing phase cells, and the Missing cell
            for p in short:
                cell = self.table.item(i, _SCATTER_COLS.index(_PHASE_COL[p]))
                if cell is not None:
                    cell.setBackground(QBrush(_STATUS_COLOR["MISSING"]))
            for p in (missing & set(EXPECTED_PHASES)):
                cell = self.table.item(i, _SCATTER_COLS.index(_PHASE_COL[p]))
                if cell is not None:
                    cell.setBackground(QBrush(_STATUS_COLOR["MISSING"]))
            vc = self.table.item(i, _SCATTER_COLS.index("Video"))
            if video_cell == "✗":
                vc.setBackground(QBrush(_STATUS_COLOR["MISSING"]))
            elif video_cell == "dump" and r.get("video_path"):
                vc.setToolTip(f"video located in the acquisition dump:\n{r['video_path']}")
            if miss_cell:
                mc = self.table.item(i, _SCATTER_COLS.index("Missing"))
                mc.setBackground(QBrush(_STATUS_COLOR["MISSING"]))
                f = QFont(); f.setBold(True); mc.setFont(f)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_SCATTER_COLS.index("Path"), QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)

        v.addWidget(QLabel(
            "red = on none of the searched drives · blue = on disk but not in the "
            "sheet.  For an implanted rat one "
            "row holds the whole session: pre/task/post GB (✗ = absent, red ⚠ = short vs the "
            "rat's median) and Video (count, 'dump' = in the acquisition dump, ✗ = none); "
            "Missing lists which of the 4 files are absent.  ⚠ Split = recorded in parts.  "
            "Double-click a row to open the folder."))
        row = QHBoxLayout()
        exp = QPushButton("Export CSV…")
        exp.clicked.connect(self._export)
        row.addWidget(exp)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

    def _open(self, r: int, _c: int):
        _reveal(self.rows[r].get("path"))

    def _export(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export scattered data",
                                           str(Path.home() / "drive_scatter.xlsx"),
                                           "Excel (*.xlsx)")
        if not f:
            return
        cols = ["kind", "rat", "date8", "day", "session", "repeat", "pre_gb",
                "task_gb", "post_gb", "video", "missing", "split", "split_text",
                "volume", "path", "detail"]
        out = []
        for r in self.rows:
            row = {c: r.get(c, "") for c in cols}
            pg = r.get("phase_gb") or {}
            row["pre_gb"] = pg.get("pre", "")
            row["task_gb"] = pg.get("task", "")
            row["post_gb"] = pg.get("post", "")
            row["video"] = r.get("video", "")
            row["missing"] = "+".join(r.get("missing", []))
            row["split"] = int(bool(r.get("split")))
            # every volume the (possibly split) session lives on
            row["volume"] = "; ".join(r.get("volumes", [])) or r.get("volume", "")
            row["path"] = "; ".join(r.get("paths", [])) or r.get("path", "")
            out.append(row)
        ok, msg = sd.write_xlsx(_xlsx(f), [("scattered", cols, out)])
        if not ok:
            QMessageBox.critical(self, "Export failed", msg)


# ------------------------------------------------------------------
#                    organize: plan + run, and the dialog
# ------------------------------------------------------------------
_PLAN_COLS = ["Action", "Rat", "Date", "d/s/r", "Entry", "Size", "Source",
              "Destination", "Why"]
_ACTION_COLOR = {
    od.MOVE:         QColor(214, 234, 250),   # rename within a drive
    od.COPY:         QColor(210, 244, 214),   # cross-drive transfer
    od.SKIP_PRESENT: QColor(238, 238, 238),   # nothing to do
    od.SKIP_DERIVED: QColor(238, 238, 238),
    od.SKIP_VIDEO:   QColor(238, 238, 238),   # loose video not ours to file
    od.SKIP_ARCHIVED: QColor(210, 244, 214),  # already archived — left in place (fine)
    od.DUPLICATE:    QColor(255, 244, 205),   # redundant copy, left alone
    od.SKIP_OTHER_DRIVE: QColor(238, 238, 238),  # other drive — not touched
    od.CONFLICT:     QColor(250, 214, 214),   # needs a human
    od.UNREADABLE:   QColor(250, 214, 214),
}


def _guarded_reject(self):
    """Esc / the window's X on a run dialog. While its worker is moving or copying,
    refuse: closing would hide a transfer that keeps going, and would let a second
    run start over the same plan. Stop first, then close."""
    if self.run_thread is not None and self.run_thread.isRunning():
        self.run_lbl.setText("Still running — press Stop and let it finish before closing.")
        return
    QDialog.reject(self)


class PlanWorker(QObject):
    """Builds the organize plan off the GUI thread. Reads only; writes nothing."""
    progress = pyqtSignal(str)
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, roster, found, dest, videos=None):
        super().__init__()
        self.roster, self.found, self.dest = roster, found, dest
        self.videos = videos or []
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # move_only: Organize files data within the chosen drive by rename
            # only — nothing is ever copied to or from another drive.
            self.done.emit(od.plan_organize(
                self.roster, self.found, self.dest, videos=self.videos,
                on_progress=lambda s: self.progress.emit(s),
                should_stop=lambda: self._stop, move_only=True))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class RunWorker(QObject):
    """Executes an organize plan off the GUI thread."""
    progress = pyqtSignal(int, int, str)      # done, total, label
    bytes_done = pyqtSignal(int)              # delta
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.done.emit(od.execute_plan(
                self.plan,
                on_progress=lambda i, n, s: self.progress.emit(i, n, s),
                on_bytes=lambda b: self.bytes_done.emit(b),
                should_stop=lambda: self._stop))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class OrganizeDialog(QDialog):
    """Review an organize plan, then run it.

    Nothing here writes until the plan on screen has been looked at and the run
    explicitly confirmed. The plan is the product; executing it is the optional
    second step."""

    reject = _guarded_reject

    def __init__(self, plan: list[dict], dest: str, parent=None):
        super().__init__(parent)
        self.plan, self.dest = plan, dest
        self.run_thread = None
        self.run_worker = None
        self.setWindowTitle(f"Organize into {dest}")
        self.resize(1250, 700)
        v = QVBoxLayout(self)

        t = od.plan_totals(plan)
        self.totals = t
        v.addWidget(QLabel(
            f"<b>{t['sessions']}</b> session(s) into <b>{dest}</b> &nbsp;·&nbsp; "
            f"<b>{t['n_move']}</b> move ({od.hsize(t['bytes_move'])}, rename — free) "
            f"&nbsp;·&nbsp; <b>{t['n_copy']}</b> copy ({od.hsize(t['bytes_copy'])}, "
            f"cross-drive) &nbsp;·&nbsp; needs <b>{od.hsize(t['bytes_needed'])}</b> free"))
        v.addWidget(QLabel(
            f"already in an HM_neuron archive (left in place): {t['n_archived']} "
            f"&nbsp;·&nbsp; already there: {t['n_present']} &nbsp;·&nbsp; derived (skipped): "
            f"{t['n_derived']} &nbsp;·&nbsp; duplicates left alone: {t['n_duplicate']} "
            f"&nbsp;·&nbsp; on other drives (not touched): {t['n_other_drive']} "
            f"&nbsp;·&nbsp; <b>conflicts: {t['n_conflict']}</b> &nbsp;·&nbsp; "
            f"<b>unreadable: {t['n_unreadable']}</b>"))

        self.table = QTableWidget(len(plan), len(_PLAN_COLS))
        self.table.setHorizontalHeaderLabels(_PLAN_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for i, p in enumerate(plan):
            vals = [p["action"], p["rat"], p["date8"],
                    f"{p['day']}/{p['session']}/{_rep(p['repeat'])}",
                    p["entry"], od.hsize(p["bytes"]), p["src"], p["dst"], p["reason"]]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                _paint(it, _ACTION_COLOR.get(p["action"], Qt.GlobalColor.white))
                self.table.setItem(i, c, it)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_PLAN_COLS.index("Source"), QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)

        v.addWidget(QLabel(
            "blue = moved by rename within this drive (instant, frees nothing, data "
            "leaves its current path) · grey = nothing to do, or on another drive "
            "(never touched) · yellow = redundant duplicate, left where it is · red = "
            "conflict or unreadable, skipped for you to resolve."))

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.run_lbl = QLabel("")
        v.addWidget(self.run_lbl)

        row = QHBoxLayout()
        exp = QPushButton("Export plan CSV…")
        exp.clicked.connect(self._export)
        row.addWidget(exp)
        row.addStretch(1)
        self.run_btn = QPushButton(f"Run — {t['n_move']} move, {t['n_copy']} copy")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(bool(t["n_move"] or t["n_copy"]))
        row.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        row.addWidget(self.cancel_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        row.addWidget(self.close_btn)
        v.addLayout(row)

    def _export(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export organize plan",
                                           str(Path.home() / "organize_plan.xlsx"),
                                           "Excel (*.xlsx)")
        if not f:
            return
        cols = ["action", "rat", "date8", "day", "session", "repeat", "entry",
                "n_files", "bytes", "src", "dst", "reason"]
        rows = [{c: p.get(c, "") for c in cols} for p in self.plan]
        ok, msg = sd.write_xlsx(_xlsx(f), [("plan", cols, rows)])
        if ok:
            self.run_lbl.setText(f"Plan exported → {msg}")
        else:
            QMessageBox.critical(self, "Export failed", msg)

    def _run(self):
        ok, msg = od.space_check(self.plan, self.dest)
        if not ok:
            QMessageBox.critical(self, "Not enough space", msg)
            return
        t = self.totals
        warn = ""
        if t["n_conflict"] or t["n_unreadable"]:
            warn = (f"\n\n{t['n_conflict']} conflict(s) and {t['n_unreadable']} "
                    f"unreadable entr(ies) will be SKIPPED, not resolved. If a drive "
                    f"is disconnected, reconnect it and re-scan first.")
        if QMessageBox.question(
                self, "Run organize?",
                f"Into {self.dest}:\n\n"
                f"  • {t['n_move']} entr(ies) MOVED by rename ({od.hsize(t['bytes_move'])}) "
                f"— same drive, instant, but the data leaves its current location.\n"
                f"  • {t['n_copy']} entr(ies) COPIED ({od.hsize(t['bytes_copy'])}) "
                f"— cross-drive; sources are kept.\n\n"
                f"{msg}\n"
                f"Nothing existing is overwritten and nothing is deleted.{warn}\n\n"
                f"Proceed?") != QMessageBox.StandardButton.Yes:
            return

        self.run_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.bar.setVisible(True)
        total = max(1, t["bytes_move"] + t["bytes_copy"])
        self.bar.setRange(0, 1000)
        self._total, self._seen = total, 0

        self.run_thread = QThread()
        self.run_worker = RunWorker(self.plan)
        self.run_worker.moveToThread(self.run_thread)
        self.run_thread.started.connect(self.run_worker.run)
        self.run_worker.progress.connect(
            lambda i, n, s: self.run_lbl.setText(f"{i}/{n} — {s}"))
        self.run_worker.bytes_done.connect(self._bytes)
        self.run_worker.done.connect(self._done)
        self.run_worker.failed.connect(self._failed)
        self.run_worker.done.connect(self.run_thread.quit)
        self.run_worker.failed.connect(self.run_thread.quit)
        self.run_thread.start()

    def _bytes(self, delta: int):
        self._seen += delta
        self.bar.setValue(min(1000, int(1000 * self._seen / self._total)))

    def _cancel(self):
        if self.run_worker:
            self.run_worker.stop()
            self.run_lbl.setText("Stopping after the current file…")

    def _done(self, results: list):
        self.cancel_btn.setVisible(False)
        self.close_btn.setEnabled(True)
        n_ok = sum(1 for r in results if r["result"] == "ok")
        errs = [r for r in results if r["result"] == "error"]
        canc = sum(1 for r in results if r["result"] == "cancelled")
        skip = sum(1 for r in results if r["result"] == "skipped")
        self.run_lbl.setText(f"Done — {n_ok} ok, {len(errs)} error, {skip} skipped, "
                             f"{canc} cancelled.")
        detail = "\n".join(f"{r['entry']}: {r['detail']}" for r in errs[:12])
        QMessageBox.information(
            self, "Organize finished",
            f"{n_ok} entr(ies) organized into {self.dest}.\n"
            f"{len(errs)} error(s), {skip} skipped, {canc} cancelled."
            + (f"\n\nErrors:\n{detail}" if detail else ""))

    def _failed(self, msg: str):
        self.cancel_btn.setVisible(False)
        self.close_btn.setEnabled(True)
        QMessageBox.critical(self, "Organize failed", msg)


# ------------------------------------------------------------------
#         reset: un-file every camera folder back to per-drive raw/
# ------------------------------------------------------------------
_RESET_COLOR = {
    rv.MOVE:        QColor(214, 234, 250),   # rename into raw/ (same drive)
    rv.SKIP_IN_RAW: QColor(238, 238, 238),   # already in raw — nothing to do
    rv.DUPLICATE:   QColor(255, 244, 205),   # identical copy already in raw — left
    rv.CONFLICT:    QColor(250, 214, 214),   # name clash, different content — skipped
    rv.BLOCKED:     QColor(250, 214, 214),   # raw/ not writable (permissions)
}
_RESET_COLS = ["action", "folder", "source", "raw destination", "reason"]


class ResetScanWorker(QObject):
    """Sweep the drives for every camera folder and build the reset plan. Reads
    only; writes nothing."""
    progress = pyqtSignal(str)
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, roots, depth):
        super().__init__()
        self.roots, self.depth = roots, depth
        self._stop = False
        self._n = 0

    def stop(self):
        self._stop = True

    def _tick(self, d):
        self._n += 1
        if self._n % 400 == 0:
            self.progress.emit(str(d))

    def run(self):
        try:
            self.done.emit(rv.scan_and_plan(self.roots, depth=self.depth,
                                            on_dir=self._tick,
                                            should_stop=lambda: self._stop))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class ResetRunWorker(QObject):
    """Execute a reset plan (the same-drive moves) off the GUI thread."""
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.done.emit(rv.execute_reset(
                self.plan, on_progress=lambda i, n, s: self.progress.emit(i, n, s),
                should_stop=lambda: self._stop))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class ResetDialog(QDialog):
    """Review the reset plan, then run it. Nothing moves until the plan on screen
    is confirmed."""

    reject = _guarded_reject

    def __init__(self, plan: list[dict], parent=None):
        super().__init__(parent)
        self.plan = plan
        self.run_thread = None
        self.run_worker = None
        self.setWindowTitle("Reset videos → raw")
        self.resize(1150, 650)
        v = QVBoxLayout(self)

        t = rv.totals(plan)
        self.totals = t
        v.addWidget(QLabel(
            f"<b>{t[rv.MOVE]}</b> folder(s) move to per-drive <b>raw/</b> (same-drive rename) "
            f"&nbsp;·&nbsp; already in raw: {t[rv.SKIP_IN_RAW]} &nbsp;·&nbsp; "
            f"identical duplicates left alone: {t[rv.DUPLICATE]} &nbsp;·&nbsp; "
            f"<b>name conflicts: {t[rv.CONFLICT]}</b> &nbsp;·&nbsp; "
            f"<b>blocked (not writable): {t[rv.BLOCKED]}</b>"))

        self.table = QTableWidget(len(plan), len(_RESET_COLS))
        self.table.setHorizontalHeaderLabels(_RESET_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for i, p in enumerate(plan):
            vals = [p["action"], p["name"], p["src"], p.get("dst") or "", p["reason"]]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                _paint(it, _RESET_COLOR.get(p["action"], Qt.GlobalColor.white))
                self.table.setItem(i, c, it)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_RESET_COLS.index("source"), QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)

        v.addWidget(QLabel(
            "blue = moved into raw/ by rename (same drive, instant) · grey = already in "
            "raw · yellow = identical copy already in raw, left in place · red = name "
            "conflict, or the drive's raw/ is not writable (permissions) — both skipped. "
            "Emptied Rat/date folders are removed only while genuinely empty."))

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.run_lbl = QLabel("")
        v.addWidget(self.run_lbl)

        row = QHBoxLayout()
        exp = QPushButton("Export plan CSV…")
        exp.clicked.connect(self._export)
        row.addWidget(exp)
        row.addStretch(1)
        self.run_btn = QPushButton(f"Run — move {t[rv.MOVE]}")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(bool(t[rv.MOVE]))
        row.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        row.addWidget(self.cancel_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        row.addWidget(self.close_btn)
        v.addLayout(row)

    def _export(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export reset plan",
                                           str(Path.home() / "reset_plan.xlsx"),
                                           "Excel (*.xlsx)")
        if not f:
            return
        cols = ["action", "name", "src", "dst", "reason"]
        rows = [{c: p.get(c, "") for c in cols} for p in self.plan]
        ok, msg = sd.write_xlsx(_xlsx(f), [("reset_plan", cols, rows)])
        self.run_lbl.setText(f"Plan exported → {msg}" if ok else "")
        if not ok:
            QMessageBox.critical(self, "Export failed", msg)

    def _run(self):
        t = self.totals
        warn = ""
        if t[rv.CONFLICT]:
            warn = (f"\n\n{t[rv.CONFLICT]} name conflict(s) will be SKIPPED (never "
                    f"overwritten). Resolve those by hand.")
        if QMessageBox.question(
                self, "Run reset?",
                f"MOVE {t[rv.MOVE]} camera folder(s) into their drive's raw\\ folder "
                f"by rename.\n\n"
                f"This undoes the Rat/date filing: the folders leave their current "
                f"location (originals do not stay). Same-drive renames are instant and "
                f"nothing is overwritten; emptied Rat/date folders are removed only "
                f"while empty.{warn}\n\nProceed?"
                ) != QMessageBox.StandardButton.Yes:
            return

        self.run_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.bar.setVisible(True)
        self.bar.setRange(0, max(1, len(self.plan)))

        self.run_thread = QThread()
        self.run_worker = ResetRunWorker(self.plan)
        self.run_worker.moveToThread(self.run_thread)
        self.run_thread.started.connect(self.run_worker.run)
        self.run_worker.progress.connect(self._progress)
        self.run_worker.done.connect(self._done)
        self.run_worker.failed.connect(self._failed)
        self.run_worker.done.connect(self.run_thread.quit)
        self.run_worker.failed.connect(self.run_thread.quit)
        self.run_thread.start()

    def _progress(self, i, n, name):
        self.bar.setValue(i)
        self.run_lbl.setText(f"{i}/{n} — {name}")

    def _cancel(self):
        if self.run_worker:
            self.run_worker.stop()
            self.run_lbl.setText("Stopping after the current folder…")

    def _done(self, results: list):
        self.cancel_btn.setVisible(False)
        self.close_btn.setEnabled(True)
        self.bar.setValue(self.bar.maximum())
        moved = sum(1 for r in results if r.get("result") == "moved")
        errs = [r for r in results if r.get("result") == "error"]
        skip = sum(1 for r in results if r.get("result") == "skipped")
        self.run_lbl.setText(f"Done — {moved} moved, {len(errs)} error, {skip} skipped.")
        detail = "\n".join(f"{r['name']}: {r.get('reason','')}" for r in errs[:12])
        QMessageBox.information(
            self, "Reset finished",
            f"{moved} folder(s) moved into raw\\.\n{len(errs)} error(s), {skip} skipped."
            + (f"\n\nErrors:\n{detail}" if detail else ""))

    def _failed(self, msg: str):
        self.cancel_btn.setVisible(False)
        self.close_btn.setEnabled(True)
        QMessageBox.critical(self, "Reset failed", msg)


# ------------------------------------------------------------------
#      fix video names: eye moved to the end -> eye<NN>_<date>_<time>
# ------------------------------------------------------------------
_FIX_COLOR = {
    fx.RENAME:   QColor(214, 234, 250),   # will be renamed
    fx.BROKEN:   QColor(255, 224, 178),   # broken original set aside (.broken)
    fx.CONFLICT: QColor(250, 214, 214),   # target already exists — skipped
    fx.UNKNOWN:  QColor(255, 244, 205),   # camera-looking but unparsed — left alone
}
_FIX_COLS = ["action", "folder", "current name", "corrected name", "reason"]


class FixNamesScanWorker(QObject):
    """Sweep the drives for misnamed camera videos and build the rename plan."""
    progress = pyqtSignal(str)
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, roots, depth):
        super().__init__()
        self.roots, self.depth = roots, depth
        self._stop = False
        self._n = 0

    def stop(self):
        self._stop = True

    def _tick(self, d):
        self._n += 1
        if self._n % 400 == 0:
            self.progress.emit(str(d))

    def run(self):
        try:
            self.done.emit(fx.scan_and_plan(self.roots, depth=self.depth,
                                            on_dir=self._tick,
                                            should_stop=lambda: self._stop))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class FixNamesRunWorker(QObject):
    """Execute a rename plan off the GUI thread."""
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.done.emit(fx.execute_renames(
                self.plan, on_progress=lambda i, n, s: self.progress.emit(i, n, s),
                should_stop=lambda: self._stop))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class FixNamesDialog(QDialog):
    """Review the rename plan, then run it. Nothing is renamed until confirmed."""

    reject = _guarded_reject

    def __init__(self, plan: list[dict], parent=None):
        super().__init__(parent)
        self.plan = plan
        self.run_thread = None
        self.run_worker = None
        self.setWindowTitle("Fix video names")
        self.resize(1150, 650)
        v = QVBoxLayout(self)

        t = fx.totals(plan)
        self.totals = t
        v.addWidget(QLabel(
            f"<b>{t[fx.RENAME]}</b> file(s) to rename to eye&lt;NN&gt;_&lt;date&gt;_&lt;time&gt;.mp4 "
            f"&nbsp;·&nbsp; broken originals set aside: {t[fx.BROKEN]} &nbsp;·&nbsp; "
            f"<b>conflicts: {t[fx.CONFLICT]}</b> &nbsp;·&nbsp; "
            f"unparsed camera-looking: {t[fx.UNKNOWN]}"))

        self.table = QTableWidget(len(plan), len(_FIX_COLS))
        self.table.setHorizontalHeaderLabels(_FIX_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for i, p in enumerate(plan):
            vals = [p["action"], p["folder"], p["old"], p.get("new") or "", p["reason"]]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                _paint(it, _FIX_COLOR.get(p["action"], Qt.GlobalColor.white))
                self.table.setItem(i, c, it)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_FIX_COLS.index("folder"), QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)

        v.addWidget(QLabel(
            "blue = renamed in place · orange = broken original set aside as ....mp4.broken "
            "so its _fixed re-encode can take the real name · red = corrected name already "
            "exists, skipped for you to resolve · yellow = camera-looking but unparsed, left "
            "untouched. Nothing is ever overwritten or deleted."))

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.run_lbl = QLabel("")
        v.addWidget(self.run_lbl)

        row = QHBoxLayout()
        exp = QPushButton("Export plan CSV…")
        exp.clicked.connect(self._export)
        row.addWidget(exp)
        row.addStretch(1)
        self.run_btn = QPushButton(f"Run — rename {t[fx.RENAME]}")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(bool(t[fx.RENAME]))
        row.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        row.addWidget(self.cancel_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        row.addWidget(self.close_btn)
        v.addLayout(row)

    def _export(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export rename plan",
                                           str(Path.home() / "rename_plan.xlsx"),
                                           "Excel (*.xlsx)")
        if not f:
            return
        cols = ["action", "folder", "old", "new", "reason"]
        rows = [{c: p.get(c, "") for c in cols} for p in self.plan]
        ok, msg = sd.write_xlsx(_xlsx(f), [("rename_plan", cols, rows)])
        self.run_lbl.setText(f"Plan exported → {msg}" if ok else "")
        if not ok:
            QMessageBox.critical(self, "Export failed", msg)

    def _run(self):
        t = self.totals
        warn = (f"\n\n{t[fx.CONFLICT]} conflict(s) will be SKIPPED, never overwritten."
                if t[fx.CONFLICT] else "")
        if QMessageBox.question(
                self, "Run rename?",
                f"Rename {t[fx.RENAME]} file(s) in place to eye<NN>_<date>_<time>.mp4.\n\n"
                f"Only the file names change; nothing is moved or overwritten.{warn}\n\n"
                f"Proceed?") != QMessageBox.StandardButton.Yes:
            return
        self.run_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.bar.setVisible(True)
        self.bar.setRange(0, max(1, len(self.plan)))
        self.run_thread = QThread()
        self.run_worker = FixNamesRunWorker(self.plan)
        self.run_worker.moveToThread(self.run_thread)
        self.run_thread.started.connect(self.run_worker.run)
        self.run_worker.progress.connect(self._progress)
        self.run_worker.done.connect(self._done)
        self.run_worker.failed.connect(self._failed)
        self.run_worker.done.connect(self.run_thread.quit)
        self.run_worker.failed.connect(self.run_thread.quit)
        self.run_thread.start()

    def _progress(self, i, n, name):
        self.bar.setValue(i)
        self.run_lbl.setText(f"{i}/{n} — {name}")

    def _cancel(self):
        if self.run_worker:
            self.run_worker.stop()
            self.run_lbl.setText("Stopping after the current file…")

    def _done(self, results: list):
        self.cancel_btn.setVisible(False)
        self.close_btn.setEnabled(True)
        self.bar.setValue(self.bar.maximum())
        ren = sum(1 for r in results if r.get("result") == "renamed")
        errs = [r for r in results if r.get("result") == "error"]
        skip = sum(1 for r in results if r.get("result") == "skipped")
        self.run_lbl.setText(f"Done — {ren} renamed, {len(errs)} error, {skip} skipped.")
        detail = "\n".join(f"{r['old']}: {r.get('reason','')}" for r in errs[:12])
        QMessageBox.information(
            self, "Rename finished",
            f"{ren} file(s) renamed.\n{len(errs)} error(s), {skip} skipped."
            + (f"\n\nErrors:\n{detail}" if detail else ""))

    def _failed(self, msg: str):
        self.cancel_btn.setVisible(False)
        self.close_btn.setEnabled(True)
        QMessageBox.critical(self, "Rename failed", msg)


_META_COLS = ["Action", "Rat", "Date", "d/s", "trials", "Folder", "Note"]
_META_COLOR = {
    pmeta.WRITE:     QColor(210, 244, 214),   # will create a new file
    pmeta.EXISTS:    QColor(238, 238, 238),   # left untouched
    pmeta.NO_VIDEO:  QColor(255, 244, 205),   # no folder to write into
    pmeta.NO_TRIALS: QColor(255, 244, 205),   # no trials in the sheet
}


class MetaDialog(QDialog):
    """Review where each RecordingMeta.xlsx would be written, then write them.

    Nothing is written until the plan is looked at and the write confirmed.
    Existing RecordingMeta.xlsx files are shown as 'exists' and never overwritten."""

    def __init__(self, plan: list[dict], parent=None):
        super().__init__(parent)
        self.plan = plan
        self.setWindowTitle("Prepare RecordingMeta.xlsx")
        self.resize(1150, 680)
        v = QVBoxLayout(self)

        n = {a: sum(1 for p in plan if p["action"] == a)
             for a in (pmeta.WRITE, pmeta.EXISTS, pmeta.NO_VIDEO, pmeta.NO_TRIALS)}
        v.addWidget(QLabel(
            f"<b>{n[pmeta.WRITE]}</b> RecordingMeta.xlsx to write &nbsp;·&nbsp; "
            f"{n[pmeta.EXISTS]} already present (kept) &nbsp;·&nbsp; "
            f"{n[pmeta.NO_VIDEO]} session(s) with no video folder &nbsp;·&nbsp; "
            f"{n[pmeta.NO_TRIALS]} with no trials in the sheet"))

        self.table = QTableWidget(len(plan), len(_META_COLS))
        self.table.setHorizontalHeaderLabels(_META_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for i, p in enumerate(plan):
            vals = [p["action"], p["rat"], p["date8"], f"{p['day']}/{p['session']}",
                    str(p["n_trials"]), p["folder"], p.get("detail", "")]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                _paint(it, _META_COLOR.get(p["action"], Qt.GlobalColor.white))
                self.table.setItem(i, c, it)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_META_COLS.index("Folder"), QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)

        v.addWidget(QLabel(
            "green = a new RecordingMeta.xlsx will be written into that video folder · "
            "grey = one is already there and is left untouched · yellow = nothing to write "
            "(no video folder, or no trials logged for that session)."))
        row = QHBoxLayout()
        self.write_btn = QPushButton(f"Write {n[pmeta.WRITE]} file(s)")
        self.write_btn.clicked.connect(self._write)
        self.write_btn.setEnabled(bool(n[pmeta.WRITE]))
        row.addWidget(self.write_btn)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

    def _write(self):
        n = sum(1 for p in self.plan if p["action"] == pmeta.WRITE)
        if QMessageBox.question(
                self, "Write RecordingMeta?",
                f"Write {n} new RecordingMeta.xlsx file(s) into their video folders?\n\n"
                "Existing files are left untouched; nothing else is changed.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.write_btn.setEnabled(False)
        results = pmeta.write_plan(self.plan)
        ok = sum(1 for r in results if r["result"] == "written")
        errs = [r for r in results if r["result"] == "error"]
        QMessageBox.information(
            self, "RecordingMeta written",
            f"{ok} RecordingMeta.xlsx written."
            + (f"\n\n{len(errs)} error(s):\n" + "\n".join(r["detail"] for r in errs[:10])
               if errs else ""))


_PP_STEPS = list("1e2345678d")
_PP_COLS = ["Rat", "Date", "d/s", "type"] + _PP_STEPS + ["progress", "folder"]
_PP_STATUS_COLOR = {
    ppc.DONE:      QColor(210, 244, 214),   # output present
    ppc.IMPLIED:   QColor(226, 240, 226),   # inferred from a downstream step
    ppc.MISSING:   QColor(250, 214, 214),   # expected but not there
    ppc.UNCHECKED: QColor(238, 238, 238),   # no detectable marker (compression)
}
_PP_GLYPH = {ppc.DONE: "✓", ppc.IMPLIED: "✓", ppc.MISSING: "✗", ppc.UNCHECKED: "?"}


class PreprocessDialog(QDialog):
    """Per-session preprocessing progress: one column per pipeline step, coloured
    by whether that step's output is present in the HM_neuron_preprocess tree."""

    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        self.rows = rows
        self.setWindowTitle("Preprocessing progress — HM_neuron_preprocess")
        self.resize(1200, 720)
        v = QVBoxLayout(self)

        complete = sum(1 for r in rows if r["n_done"] >= r["n_expected"] and r["n_expected"])
        none = sum(1 for r in rows if r["n_done"] == 0)
        v.addWidget(QLabel(
            f"<b>{len(rows)}</b> session(s) · <b>{complete}</b> fully preprocessed · "
            f"<b>{none}</b> not started.  Steps: implanted 1 e 2 3 4 5 6 7 8 d · "
            f"non-implanted 2 3 4 5 6 d"))

        self.table = QTableWidget(len(rows), len(_PP_COLS))
        self.table.setHorizontalHeaderLabels(_PP_COLS)
        for c, name in enumerate(_PP_COLS):      # tooltip step columns with labels
            if name in ppc.STEP_LABEL:
                self.table.horizontalHeaderItem(c).setToolTip(
                    f"step {name}: {ppc.STEP_LABEL[name]}")
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellDoubleClicked.connect(self._open)
        for i, r in enumerate(rows):
            base = [r["rat"], r["date8"], f"{r['day']}/{r['session']}",
                    "impl" if r["implanted"] else "video"]
            step_cells = []
            for s in _PP_STEPS:
                st = r["status"].get(s)
                step_cells.append(_PP_GLYPH.get(st, "") if s in r["expected"] else "")
            folder = r["folders"][0] if r["folders"] else "—"
            vals = base + step_cells + [f"{r['n_done']}/{r['n_expected']}", folder]
            for c, val in enumerate(vals):
                it = QTableWidgetItem(str(val))
                name = _PP_COLS[c]
                if name in _PP_STEPS:
                    st = r["status"].get(name)
                    if name in r["expected"] and st in _PP_STATUS_COLOR:
                        _paint(it, _PP_STATUS_COLOR[st])
                        it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    else:
                        _paint(it, Qt.GlobalColor.white)
                else:
                    _paint(it, Qt.GlobalColor.white)
                self.table.setItem(i, c, it)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_PP_COLS.index("folder"), QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)

        v.addWidget(QLabel(
            "green ✓ = step output present · pale green ✓ = implied done (sync/stitch, "
            "inferred from the tracker output) · red ✗ = expected but missing · grey ? = "
            "not separately detectable (compression).  Double-click to open the folder."))
        row = QHBoxLayout()
        exp = QPushButton("Export CSV…")
        exp.clicked.connect(self._export)
        row.addWidget(exp)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

    def _open(self, r: int, _c: int):
        fs = self.rows[r].get("folders") or []
        if fs:
            _reveal(fs[0])

    def _export(self):
        f, _ = QFileDialog.getSaveFileName(self, "Export preprocessing progress",
                                           str(Path.home() / "preprocess_progress.xlsx"),
                                           "Excel (*.xlsx)")
        if not f:
            return
        cols = ["rat", "date8", "day", "session", "repeat", "implanted"] + \
            [f"step_{s}" for s in _PP_STEPS] + ["n_done", "n_expected", "summary", "folder"]
        out = []
        for r in self.rows:
            row = dict(rat=r["rat"], date8=r["date8"], day=r["day"],
                       session=r["session"], repeat=_rep(r["repeat"]),
                       implanted=int(r["implanted"]), n_done=r["n_done"],
                       n_expected=r["n_expected"], summary=r["summary"],
                       folder="; ".join(r["folders"]))
            for s in _PP_STEPS:
                row[f"step_{s}"] = r["status"].get(s, "") if s in r["expected"] else ""
            out.append(row)
        ok, msg = sd.write_xlsx(_xlsx(f), [("progress", cols, out)])
        if not ok:
            QMessageBox.critical(self, "Export failed", msg)


class SummaryDialog(QDialog):
    """Show the rendered summary figure in a scroll area, with Save-as / open."""

    def __init__(self, png_path: str, parent=None):
        super().__init__(parent)
        self.png_path = png_path
        self.setWindowTitle("Dataset summary")
        self.resize(1200, 860)
        v = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        lbl = QLabel()
        lbl.setPixmap(QPixmap(png_path))
        lbl.adjustSize()
        scroll.setWidget(lbl)
        v.addWidget(scroll, 1)

        row = QHBoxLayout()
        save = QPushButton("Save as…")
        save.clicked.connect(self._save)
        row.addWidget(save)
        opn = QPushButton("Open in image viewer")
        opn.clicked.connect(lambda: _reveal(self.png_path))
        row.addWidget(opn)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

    def _save(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save summary figure",
                                           str(Path.home() / "hexmaze_summary.png"),
                                           "PNG (*.png)")
        if not f:
            return
        try:
            import shutil
            shutil.copyfile(self.png_path, f)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))


def _reveal(target: str | None):
    """Open a folder in the OS file browser."""
    if not target:
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", target])
        elif sys.platform.startswith("win"):
            subprocess.run(["explorer", target])
        else:
            subprocess.run(["xdg-open", target])
    except Exception:
        pass


# ------------------------------------------------------------------
#                              GUI
# ------------------------------------------------------------------
class ScanDriveGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HexMaze — Drive Coverage Checker")
        self.resize(1100, 720)
        self.roster: list[dict] = []
        self.results: list[dict] = []
        self.thread: QThread | None = None
        self.worker: ScanWorker | None = None
        self.search_thread: QThread | None = None
        self.search_worker: SearchWorker | None = None
        self.plan_thread: QThread | None = None
        self.plan_worker: PlanWorker | None = None
        self.preproc_thread: QThread | None = None
        self.preproc_worker: PreprocessWorker | None = None
        self.meta_thread: QThread | None = None
        self.meta_worker: MetaWorker | None = None
        self.reset_thread: QThread | None = None
        self.reset_worker: ResetScanWorker | None = None
        self.fix_thread: QThread | None = None
        self.fix_worker: FixNamesScanWorker | None = None
        self._meta_raw = None
        self._closing = False

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(10)

        title = QLabel("Drive Coverage Checker")
        title.setStyleSheet("font-size:18px; font-weight:700; color:#1f6f43;")
        v.addWidget(title)

        # --- Excel picker -------------------------------------------------
        self.xlsx_edit = QLineEdit()
        self.xlsx_edit.setPlaceholderText("experiment spreadsheet (.xlsx) — the 'Raw' sheet")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setEditable(True)
        self.sheet_combo.addItem("Raw")
        self.sheet_combo.setFixedWidth(140)
        row = QHBoxLayout()
        row.addWidget(QLabel("Excel:"))
        row.addWidget(self.xlsx_edit, 1)
        row.addWidget(QLabel("Sheet:"))
        row.addWidget(self.sheet_combo)
        b = QPushButton("Browse…")
        b.clicked.connect(self._pick_excel)
        row.addWidget(b)
        load = QPushButton("Load roster")
        load.clicked.connect(self._load_roster)
        row.addWidget(load)
        v.addLayout(row)

        # --- drive folders: an add/remove list (0..N) ---------------------
        box = QGroupBox("Drive folders — each directly contains the Rat<N>_* folders "
                        "(e.g. F:\\HM_neurons)")
        bl = QVBoxLayout(box)
        self.drive_list = QListWidget()
        self.drive_list.setMaximumHeight(90)
        self.drive_list.setToolTip("Add every drive/folder the data spans. Each should be one "
                                   "level above the Rat<N>_* folders.")
        bl.addWidget(self.drive_list)
        drow = QHBoxLayout()
        add_btn = QPushButton("Add folder…")
        add_btn.clicked.connect(self._add_drive)
        rem_btn = QPushButton("Remove")
        rem_btn.clicked.connect(self._remove_drive)
        drow.addWidget(add_btn)
        drow.addWidget(rem_btn)
        drow.addStretch(1)
        bl.addLayout(drow)
        v.addWidget(box)

        # --- menu bar: the occasional / maintenance actions tucked away ---
        mb = self.menuBar()
        m_file = mb.addMenu("&File")
        a_open = QAction("Open spreadsheet…", self)
        a_open.triggered.connect(self._pick_excel)
        m_file.addAction(a_open)
        m_file.addSeparator()
        a_quit = QAction("Quit", self)
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_fix = mb.addMenu("Fix / &Prepare")
        self.organize_btn = QAction("Organize drive…", self)
        self.organize_btn.setToolTip("File the roster sessions found anywhere on the drive of "
                                     "ONE of your listed folders into that drive's HM_neurons "
                                     "archive — moves (renames) within that drive only; data "
                                     "on other drives is listed but never copied or touched. "
                                     "Shows the full plan before writing anything.")
        self.organize_btn.triggered.connect(self._organize)
        self.organize_btn.setEnabled(False)
        m_fix.addAction(self.organize_btn)
        self.reset_btn = QAction("Reset → raw…", self)
        self.reset_btn.setToolTip("Undo all filing: move every camera folder into a flat raw\\ "
                                  "folder on its own drive; shows the plan before moving.")
        self.reset_btn.triggered.connect(self._reset)
        m_fix.addAction(self.reset_btn)
        self.fixnames_btn = QAction("Fix video names…", self)
        self.fixnames_btn.setToolTip("Rename camera videos whose name parts are out of order "
                                     "to eye<NN>_<date>_<time>.mp4; shows the plan first.")
        self.fixnames_btn.triggered.connect(self._fix_names)
        m_fix.addAction(self.fixnames_btn)
        m_fix.addSeparator()
        self.meta_btn = QAction("Prepare RecordingMeta…", self)
        self.meta_btn.setToolTip("Build a RecordingMeta.xlsx for every maze session from the "
                                 "sheet; existing files are never overwritten.")
        self.meta_btn.triggered.connect(self._prepare_meta)
        self.meta_btn.setEnabled(False)
        m_fix.addAction(self.meta_btn)

        m_rep = mb.addMenu("&Reports")
        self.summary_btn = QAction("Summary figure…", self)
        self.summary_btn.setToolTip("One-look status figure per rat/repeat: pre/task/post/video "
                                    "presence, size, and location. Searches the whole drive(s) "
                                    "the listed folders are on first.")
        self.summary_btn.triggered.connect(self._summary)
        self.summary_btn.setEnabled(False)
        m_rep.addAction(self.summary_btn)
        self.preproc_btn = QAction("Preprocess progress…", self)
        self.preproc_btn.setToolTip("Per-session pipeline-step status from the "
                                    "HM_neuron_preprocess tree.")
        self.preproc_btn.triggered.connect(self._preprocess)
        self.preproc_btn.setEnabled(False)
        m_rep.addAction(self.preproc_btn)

        m_opt = mb.addMenu("&Options")
        self.act_meta_after = QAction("Prepare RecordingMeta after scan", self)
        self.act_meta_after.setCheckable(True)
        # Off by default: it writes into the data folders with no review step.
        self.act_meta_after.setChecked(False)
        self.act_meta_after.setToolTip("After each scan, write RecordingMeta.xlsx into any maze "
                                       "session that lacks one — only where that day's sessions "
                                       "pair one-to-one with its camera folders; everything else "
                                       "is left for Fix / Prepare ▸ Prepare RecordingMeta (never "
                                       "overwrites existing files).")
        m_opt.addAction(self.act_meta_after)
        self.act_sysdrive = QAction("Include system drive (C:)", self)
        self.act_sysdrive.setCheckable(True)
        self.act_sysdrive.setToolTip("Also sweep C:\\ in Preprocess progress, and in Fix video "
                                     "names when no drive folder is selected. Find scattered / "
                                     "Organize / Summary search the whole drive each listed "
                                     "folder is on — a folder on C:\\ is searched whole only "
                                     "with this on. RecordingMeta searches just the listed "
                                     "folders.")
        m_opt.addAction(self.act_sysdrive)
        m_opt.addSeparator()
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 12)
        self.depth_spin.setValue(6)
        dw = QWidget()
        dwl = QHBoxLayout(dw)
        dwl.setContentsMargins(20, 2, 12, 2)
        dwl.addWidget(QLabel("Search depth:"))
        dwl.addWidget(self.depth_spin)
        dwa = QWidgetAction(m_opt)
        dwa.setDefaultWidget(dw)
        m_opt.addAction(dwa)

        # --- toolbar: just the daily actions ------------------------------
        act = QHBoxLayout()
        self.scan_btn = QPushButton("Scan drives")
        self.scan_btn.clicked.connect(self._scan)
        self.scan_btn.setEnabled(False)
        act.addWidget(self.scan_btn)
        self.search_btn = QPushButton("Find scattered data…")
        self.search_btn.setToolTip(
            "Deep-search the WHOLE drive each folder above lives on (from the drive root, "
            "up to Options ▸ Search depth) for Rat<N>/<YYYYMMDD> folders filed somewhere "
            "odd on it.")
        self.search_btn.clicked.connect(self._search_all)
        self.search_btn.setEnabled(False)
        act.addWidget(self.search_btn)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        act.addWidget(self.export_btn)
        act.addStretch(1)
        self.status_lbl = QLabel("Load a spreadsheet to begin.")
        act.addWidget(self.status_lbl)
        v.addLayout(act)

        # --- results table ------------------------------------------------
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._open_folder)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(COLUMNS.index("Found in"), QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)

        hint = QLabel("Rows: green = complete · yellow = partial · red = missing. "
                      "pre/task/post columns show ephys size in GB (✗ = absent); a red "
                      "⚠ cell is present but short vs that rat's median. "
                      "Double-click a found row to open the folder.")
        hint.setStyleSheet("color:#666;")
        v.addWidget(hint)

    # -- pickers -----------------------------------------------------------
    def _pick_excel(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select experiment spreadsheet",
                                           str(Path.home()), "Excel (*.xlsx *.xlsm *.xls)")
        if f:
            self.xlsx_edit.setText(f)
            self._refresh_sheets(f)

    def _refresh_sheets(self, path: str):
        try:
            names = pd.ExcelFile(path).sheet_names
        except Exception:
            return
        cur = self.sheet_combo.currentText()
        self.sheet_combo.clear()
        self.sheet_combo.addItems(names)
        if "Raw" in names:
            self.sheet_combo.setCurrentText("Raw")
        elif cur in names:
            self.sheet_combo.setCurrentText(cur)

    @staticmethod
    def _norm_path(p: str) -> str:
        """Comparable form of a folder path: resolved, and case-folded on Windows,
        so F:/HM_neurons, f:\\hm_neurons and a mapped-drive alias compare equal."""
        try:
            p = str(Path(p).resolve())
        except OSError:
            pass
        return os.path.normcase(os.path.normpath(p))

    @staticmethod
    def _is_under(child: str, parent: str) -> bool:
        """Both normalised. True if `child` is strictly inside `parent`."""
        return child != parent and child.startswith(parent.rstrip(os.sep) + os.sep)

    def _add_drive(self):
        d = QFileDialog.getExistingDirectory(
            self, "Add a drive folder (one level above the Rat<N>_* folders)", str(Path.home()))
        if not d:
            return
        # One folder must never be listed twice (or inside another listed folder):
        # the scan would index its sessions twice and double their counts.
        nd = self._norm_path(d)
        existing = [(i, self.drive_list.item(i).text().strip())
                    for i in range(self.drive_list.count())]
        for _i, ex in existing:
            ne = self._norm_path(ex)
            if ne == nd:
                self.status_lbl.setText(f"{d} is already in the list.")
                return
            if self._is_under(nd, ne):
                QMessageBox.information(self, "Already covered",
                                        f"{d}\n\nis inside\n\n{ex}\n\nwhich is already "
                                        f"in the list — not added.")
                return
        inner = [i for i, ex in existing if self._is_under(self._norm_path(ex), nd)]
        for i in reversed(inner):                 # the new folder covers these
            self.drive_list.takeItem(i)
        self.drive_list.addItem(d)
        if inner:
            self.status_lbl.setText(f"Added {d}; removed {len(inner)} folder(s) inside it.")

    def _remove_drive(self):
        for it in self.drive_list.selectedItems():
            self.drive_list.takeItem(self.drive_list.row(it))

    def _drive_paths(self) -> list:
        return [self.drive_list.item(i).text().strip()
                for i in range(self.drive_list.count())
                if self.drive_list.item(i).text().strip()]

    def _search_roots(self) -> list:
        """What Find scattered / Organize / Summary walk: the WHOLE drive each listed
        folder lives on, not just the folder — scattered data sits anywhere on the
        drive (a backup folder, a stray copy at the root). Two folders on one drive
        give one root; drives with no listed folder are never walked.

        A folder on the system drive stays as listed unless Options ▸ Include system
        drive is on — and always on POSIX, where / also holds every other mounted
        volume (/Volumes, /media), so walking it would sweep them all."""
        out = []
        for d in self._drive_paths():
            vr = sd.volume_root(d)
            if sd.is_system_volume(vr) and not (os.name == "nt"
                                                and self.act_sysdrive.isChecked()):
                vr = Path(d)
            out.append(str(vr))
        keep: list = []                            # (normalised, as shown)
        for r in out:
            nr = self._norm_path(r)
            if any(nk == nr or self._is_under(nr, nk) for nk, _ in keep):
                continue
            keep = [(nk, k) for nk, k in keep if not self._is_under(nk, nr)]
            keep.append((nr, r))
        return [k for _, k in keep]

    # -- roster ------------------------------------------------------------
    def _load_roster(self):
        if self._running(self.thread) or self._search_busy():
            # the running worker writes into self.results / the table by row index
            QMessageBox.information(self, "Busy", "Wait for the running scan/search to finish.")
            return
        path = self.xlsx_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "No spreadsheet", "Pick a valid .xlsx file first.")
            return
        try:
            self.roster = build_roster(path, self.sheet_combo.currentText().strip() or "Raw")
        except Exception as exc:
            QMessageBox.critical(self, "Could not read spreadsheet", str(exc))
            return
        self.results = []
        self._populate_expected()
        n_imp = sum(1 for e in self.roster if e["implanted"])
        rats = sorted({e["rat"] for e in self.roster}, key=lambda s: int(s[3:]))
        self.status_lbl.setText(
            f"{len(self.roster)} rat-sessions · {', '.join(rats)} · "
            f"{n_imp} with ephys, {len(self.roster) - n_imp} video-only. "
            f"Now pick drives and press Scan.")
        self._refresh_actions()
        self.preproc_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

    def _populate_expected(self):
        self.table.setRowCount(len(self.roster))
        for i, e in enumerate(self.roster):
            vals = [e["rat"], e["date8"] or "?", str(e["day"]), str(e["session"]),
                    _rep(e["repeat"]), e["expected"], "", "", "", "", "", "", "?"]
            for c, val in enumerate(vals):
                self._set(i, c, val, status="?")

    # -- scanning ----------------------------------------------------------
    # -- background-task bookkeeping ----------------------------------------
    @staticmethod
    def _running(t) -> bool:
        return t is not None and t.isRunning()

    def _search_busy(self) -> bool:
        """True while a drive search, organize plan or meta plan is in flight. They
        share self.search_thread & co., and replacing a running QThread destroys it
        — Qt then aborts the whole app."""
        return (self._running(self.search_thread) or self._running(self.plan_thread)
                or self._running(self.meta_thread))

    def _busy_note(self) -> bool:
        if self._search_busy():
            self.status_lbl.setText("A drive search is still running — try again when it finishes.")
            return True
        return False

    def _refresh_actions(self):
        """Enable the search-triggering actions only when nothing is in flight."""
        have = bool(self.roster)
        busy = self._search_busy()
        for a in (self.search_btn, self.organize_btn, self.summary_btn, self.meta_btn):
            a.setEnabled(have and not busy)
        self.scan_btn.setEnabled(have and not busy and not self._running(self.thread))

    def _thread_ended(self):
        # QThread.finished fires just BEFORE the thread is marked not-running; wait
        # the last instant out so the refresh below sees it as idle.
        t = self.sender()
        if isinstance(t, QThread):
            t.wait(2000)
        self._refresh_actions()

    def _start_search(self, status: str, on_failed, on_index=None, on_rows=None,
                      roots=None) -> bool:
        """Start the one shared SearchWorker; refuse if one is already running.
        `roots` defaults to the listed drive folders themselves."""
        if self._busy_note() or self._closing:
            return False
        if self.search_thread is not None:
            self.search_thread.wait(2000)        # previous one has finished; let it exit
        self.search_thread = QThread()
        self.search_worker = SearchWorker(self.roster,
                                          self._drive_paths() if roots is None else roots,
                                          self.depth_spin.value(),
                                          want_rows=on_rows is not None)
        self.search_worker.moveToThread(self.search_thread)
        self.search_thread.started.connect(self.search_worker.run)
        self.search_worker.progress.connect(lambda d: self.status_lbl.setText(f"{status} {d}"))
        if on_index is not None:
            self.search_worker.raw_index.connect(on_index)
        if on_rows is not None:
            self.search_worker.done.connect(on_rows)
            self.search_worker.done.connect(self.search_thread.quit)
        else:
            self.search_worker.raw_index.connect(self.search_thread.quit)
        self.search_worker.failed.connect(on_failed)
        self.search_worker.failed.connect(self.search_thread.quit)
        self.search_thread.finished.connect(self._thread_ended)
        self.search_thread.start()
        self._refresh_actions()
        return True

    def closeEvent(self, event):
        """Never let Qt destroy a running QThread (it aborts the process — mid-scan,
        or mid-plan). Offer to stop the background work and wait for it."""
        pairs = [(self.thread, self.worker), (self.search_thread, self.search_worker),
                 (self.plan_thread, self.plan_worker), (self.meta_thread, self.meta_worker),
                 (self.preproc_thread, self.preproc_worker),
                 (self.reset_thread, self.reset_worker), (self.fix_thread, self.fix_worker)]
        live = [(t, w) for t, w in pairs if self._running(t)]
        if not live:
            event.accept()
            return
        if QMessageBox.question(
                self, "Quit?",
                f"{len(live)} background task(s) are still running. Stop them and quit?") \
                != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self._closing = True
        for t, w in live:
            if w is not None:
                w.blockSignals(True)             # no follow-up steps from a half-done run
                if hasattr(w, "stop"):
                    w.stop()
            t.quit()
        for t, _w in live:
            if not t.wait(20000):
                self._closing = False
                QMessageBox.warning(self, "Still busy",
                                    "A background task did not stop in time. "
                                    "Try closing again in a moment.")
                event.ignore()
                return
        event.accept()

    def _scan(self):
        if not self.roster or self._running(self.thread) or self._busy_note():
            return
        roots = self._drive_paths()
        if not any(roots):
            QMessageBox.warning(self, "No drives", "Select at least one drive folder.")
            return
        self.scan_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.results = [None] * len(self.roster)
        self.status_lbl.setText("Scanning…")

        self.thread = QThread()
        self.worker = ScanWorker(self.roster, roots)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.row_done.connect(self._row_done)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._scan_finished)
        self.worker.failed.connect(self._scan_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_ended)
        self.thread.start()

    def _progress(self, done: int, total: int):
        self.status_lbl.setText(f"Scanning… {done}/{total}")

    def _row_done(self, i: int, res: dict):
        self.results[i] = res
        st = res["status"]
        video_cell = str(res["n_video"]) if res["n_video"] else "✗"
        short = set(res.get("short_phases", []))
        pg = res.get("phase_gb", {})
        if res["implanted"]:
            # each phase cell shows its GB (✗ if absent); short ones get a ⚠
            phase_cells = []
            for p in EXPECTED_PHASES:
                if p not in res["phases"]:
                    phase_cells.append("✗")
                else:
                    phase_cells.append(f"{pg.get(p, 0):.0f}⚠" if p in short
                                       else f"{pg.get(p, 0):.0f}")
        else:
            phase_cells = ["—", "—", "—"]
        split_cell = f"⚠ {res['split_text']}" if res.get("split_text") else ""
        # Show the actual folder path(s) the data was found in, not "Drive 1".
        found_cell = "; ".join(res.get("paths", [])) or res.get("found_in", "")
        vals = [res["rat"], res["date8"] or "?", str(res["day"]), str(res["session"]),
                _rep(res["repeat"]), res["expected"], video_cell, *phase_cells,
                split_cell, found_cell, st]
        for c, val in enumerate(vals):
            self._set(i, c, val, status=st)
        # Repaint short-phase cells red on top of the row tint, and tooltip why.
        if res.get("reasons"):
            self.table.item(i, _STATUS_COL).setToolTip("; ".join(res["reasons"]))
        for p in short:
            c = COLUMNS.index(_PHASE_COL[p])
            cell = self.table.item(i, c)
            if cell is not None:
                cell.setBackground(QBrush(_STATUS_COLOR["MISSING"]))
                cell.setToolTip(f"{p}: {pg.get(p, 0):.0f} GB — short vs this rat's median")

    def _scan_finished(self):
        ok = sum(1 for r in self.results if r and r["status"] == "OK")
        part = sum(1 for r in self.results if r and r["status"] == "PARTIAL")
        miss = sum(1 for r in self.results if r and r["status"] == "MISSING")
        self.status_lbl.setText(f"Done — {ok} complete · {part} partial · {miss} missing "
                                f"(of {len(self.results)}).")
        self.export_btn.setEnabled(True)
        self._refresh_actions()
        # Optionally chain into writing missing RecordingMeta.xlsx (never overwrites;
        # only unambiguous session<->folder pairings — see MetaWorker). Off by
        # default: Options ▸ "Prepare RecordingMeta after scan".
        if self.roster and self.act_meta_after.isChecked() and not self._closing:
            self._prepare_meta(auto=True)

    def _scan_failed(self, msg: str):
        QMessageBox.critical(self, "Scan failed", msg)
        self.status_lbl.setText("Scan failed.")
        self._refresh_actions()

    # -- search every drive for scattered data ------------------------------
    def _search_all(self):
        if not self.roster or self._busy_note():
            return
        if not self._drive_paths():
            QMessageBox.warning(self, "No drive folders",
                                "Add at least one drive folder above first.")
            return
        roots = self._search_roots()
        if QMessageBox.question(
                self, "Find scattered data",
                f"Deep-search the whole drive of each listed folder — {len(roots)} root(s), "
                f"{self.depth_spin.value()} levels down — for Rat<N>/<YYYYMMDD> folders:\n\n  "
                + "\n  ".join(roots) + "\n\nThis may take a few minutes.") \
                != QMessageBox.StandardButton.Yes:
            return
        self.status_lbl.setText("Searching the drives…")
        self._start_search("Searching…", self._search_failed, on_rows=self._search_done,
                           roots=roots)

    def _search_done(self, rows: list, n_found: int):
        self.status_lbl.setText(f"Search done — {n_found} session folder(s) on the drives.")
        if not rows:
            QMessageBox.information(
                self, "Nothing scattered",
                f"Found {n_found} session folder(s) on the searched drives.\n\n"
                "Every expected session was found, and every folder on disk is "
                "accounted for in the spreadsheet.")
            return
        ScatterDialog(rows, self).exec()

    def _search_failed(self, msg: str):
        QMessageBox.critical(self, "Search failed", msg)
        self.status_lbl.setText("Search failed.")

    # -- organize ----------------------------------------------------------
    def _organize(self):
        if not self.roster or self._busy_note():
            return
        # Organize files into ONE of the listed drive folders and only moves data
        # already on that drive — never a free-picked folder, never a copy. The
        # search covers the whole drive, so data anywhere on it is gathered in.
        roots = self._drive_paths()
        if not roots:
            QMessageBox.information(
                self, "Organize",
                "Add the drive folder you want to organize to the list above first.")
            return
        if len(roots) == 1:
            picked = roots[0]
        else:
            picked, ok = QInputDialog.getItem(
                self, "Organize drive",
                "Which drive folder should be organized?\n"
                "(data from anywhere on that drive is moved into it; other drives are "
                "not touched)",
                roots, 0, False)
            if not ok or not picked:
                return
        # All consolidated data goes into an HM_neurons archive folder — a drive
        # root becomes <drive>\HM_neurons, an existing archive is used as-is.
        dest = od.archive_dest(picked)
        if dest != picked:
            QMessageBox.information(
                self, "Destination",
                f"Data will be consolidated into the HM_neurons archive:\n\n{dest}")
        # Plan against a fresh look at the drives, never a cached index: the
        # externals here drop off mid-session, and planning terabytes of renames
        # from a stale picture is how data goes missing.
        self.status_lbl.setText("Looking at the drives…")
        self._start_search("Looking…", self._organize_failed,
                           on_index=lambda combined: self._plan_organize(combined, dest),
                           roots=self._search_roots())

    def _organize_failed(self, msg: str):
        QMessageBox.critical(self, "Organize: search failed", msg)
        self.status_lbl.setText("Organize search failed.")

    def _plan_organize(self, combined: dict, dest: str):
        if self._closing:
            return
        self.status_lbl.setText("Planning…")
        found = combined.get("sessions", {})
        videos = combined.get("videos", [])
        if self.plan_thread is not None:
            self.plan_thread.wait(2000)
        self.plan_thread = QThread()
        self.plan_worker = PlanWorker(self.roster, found, dest, videos)
        self.plan_worker.moveToThread(self.plan_thread)
        self.plan_thread.started.connect(self.plan_worker.run)
        self.plan_worker.progress.connect(
            lambda s: self.status_lbl.setText(f"Planning… {s}"))
        self.plan_worker.done.connect(lambda plan: self._plan_done(plan, dest))
        self.plan_worker.failed.connect(self._plan_failed)
        self.plan_worker.done.connect(self.plan_thread.quit)
        self.plan_worker.failed.connect(self.plan_thread.quit)
        self.plan_thread.finished.connect(self._thread_ended)
        self.plan_thread.start()
        self._refresh_actions()

    def _plan_done(self, plan: list, dest: str):
        t = od.plan_totals(plan)
        self.status_lbl.setText(
            f"Plan: {t['n_move']} move, {t['n_copy']} copy, "
            f"{t['n_conflict']} conflict, {t['n_unreadable']} unreadable.")
        if not plan:
            QMessageBox.information(self, "Nothing to organize",
                                    "No roster session has data to assemble.")
            return
        OrganizeDialog(plan, dest, self).exec()

    def _plan_failed(self, msg: str):
        QMessageBox.critical(self, "Planning failed", msg)
        self.status_lbl.setText("Planning failed.")

    # -- reset: un-file all videos to per-drive raw/ -----------------------
    def _reset(self):
        if self._running(self.reset_thread):
            return
        roots = self._drive_paths()
        if not roots:
            # No silent every-volume fallback here: a reset un-files EVERY camera
            # folder it finds, and other experiments' drives are mounted too.
            QMessageBox.information(
                self, "Reset",
                "Add the drive folder(s) to reset above first — reset only ever "
                "touches the folders you list.")
            return
        if QMessageBox.question(
                self, "Reset videos → raw?",
                "Look at:\n  " + "\n  ".join(roots) + "\n\nand plan moving EVERY camera "
                "folder — both already filed under Rat/date AND loose — into a flat raw\\ "
                "folder on its own drive, undoing all filing so matching can be redone.\n\n"
                "You'll see the full plan before anything moves. Continue?"
                ) != QMessageBox.StandardButton.Yes:
            return
        self.status_lbl.setText("Scanning for camera folders…")
        self.reset_btn.setEnabled(False)
        self.reset_thread = QThread()
        self.reset_worker = ResetScanWorker(roots, self.depth_spin.value())
        self.reset_worker.moveToThread(self.reset_thread)
        self.reset_thread.started.connect(self.reset_worker.run)
        self.reset_worker.progress.connect(
            lambda d: self.status_lbl.setText(f"Scanning… {d}"))
        self.reset_worker.done.connect(self._reset_review)
        self.reset_worker.failed.connect(self._reset_failed)
        self.reset_worker.done.connect(self.reset_thread.quit)
        self.reset_worker.failed.connect(self.reset_thread.quit)
        self.reset_thread.finished.connect(self._thread_ended)
        self.reset_thread.start()

    def _reset_review(self, plan: list):
        self.reset_btn.setEnabled(True)
        t = rv.totals(plan)
        self.status_lbl.setText(
            f"Reset plan: {t[rv.MOVE]} to move, {t[rv.CONFLICT]} conflict, "
            f"{t[rv.SKIP_IN_RAW]} already in raw.")
        if not plan:
            QMessageBox.information(self, "Reset", "No camera folders found on those drives.")
            return
        ResetDialog(plan, self).exec()

    def _reset_failed(self, msg: str):
        self.reset_btn.setEnabled(True)
        self.status_lbl.setText("Reset scan failed.")
        QMessageBox.critical(self, "Reset failed", msg)

    # -- fix misordered video file names -----------------------------------
    def _fix_names(self):
        if self._running(self.fix_thread):
            return
        roots = self._drive_paths()
        if not roots:
            roots = [str(r) for r in sd.list_drive_roots(
                include_system=self.act_sysdrive.isChecked())]
            if not roots:
                QMessageBox.information(self, "Fix video names", "No drives to scan.")
                return
            # The scope widened from "the listed folders" to every mounted volume —
            # say so before sweeping them (the rename plan is still reviewed after).
            if QMessageBox.question(
                    self, "Fix video names — every volume?",
                    "No drive folder is listed, so this will sweep EVERY mounted volume:\n\n  "
                    + "\n  ".join(roots) + "\n\nfor misnamed camera videos (you review the "
                    "rename plan before anything changes). Continue?") \
                    != QMessageBox.StandardButton.Yes:
                return
        self.status_lbl.setText("Scanning for misnamed videos…")
        self.fixnames_btn.setEnabled(False)
        self.fix_thread = QThread()
        self.fix_worker = FixNamesScanWorker(roots, self.depth_spin.value())
        self.fix_worker.moveToThread(self.fix_thread)
        self.fix_thread.started.connect(self.fix_worker.run)
        self.fix_worker.progress.connect(
            lambda d: self.status_lbl.setText(f"Scanning… {d}"))
        self.fix_worker.done.connect(self._fix_review)
        self.fix_worker.failed.connect(self._fix_failed)
        self.fix_worker.done.connect(self.fix_thread.quit)
        self.fix_worker.failed.connect(self.fix_thread.quit)
        self.fix_thread.finished.connect(self._thread_ended)
        self.fix_thread.start()

    def _fix_review(self, plan: list):
        self.fixnames_btn.setEnabled(True)
        t = fx.totals(plan)
        self.status_lbl.setText(
            f"Rename plan: {t[fx.RENAME]} to fix, {t[fx.CONFLICT]} conflict, "
            f"{t[fx.UNKNOWN]} unparsed.")
        if not any(p["action"] in (fx.RENAME, fx.CONFLICT, fx.UNKNOWN) for p in plan):
            QMessageBox.information(self, "Fix video names",
                                    "No misnamed camera videos found — all names look correct.")
            return
        FixNamesDialog(plan, self).exec()

    def _fix_failed(self, msg: str):
        self.fixnames_btn.setEnabled(True)
        self.status_lbl.setText("Rename scan failed.")
        QMessageBox.critical(self, "Fix video names failed", msg)

    # -- summary figure ----------------------------------------------------
    def _summary(self):
        if not self.roster or self._busy_note():
            return
        self.status_lbl.setText("Searching the drives for the summary…")
        self._start_search("Looking…", self._summary_failed, on_index=self._render_summary,
                           roots=self._search_roots())

    def _render_summary(self, combined: dict):
        if self._closing:
            return
        try:
            summary = smz.build_summary(self.roster, combined.get("sessions", {}),
                                        combined.get("videos", []))
            scratch = Path.home() / "hexmaze_summary.png"
            smz.render_summary(summary, str(scratch))
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "Summary failed", str(exc))
            self.status_lbl.setText("Summary failed.")
            return
        self.status_lbl.setText("Summary ready.")
        SummaryDialog(str(scratch), self).exec()

    def _summary_failed(self, msg: str):
        QMessageBox.critical(self, "Summary failed", msg)
        self.status_lbl.setText("Summary failed.")

    # -- preprocessing progress --------------------------------------------
    def _preprocess(self):
        if not self.roster:
            return
        vols = sd.list_drive_roots(include_system=self.act_sysdrive.isChecked())
        if not vols:
            QMessageBox.warning(self, "No drives", "No mounted volumes to search.")
            return
        self.preproc_btn.setEnabled(False)
        self.status_lbl.setText("Reading HM_neuron_preprocess…")
        self.preproc_thread = QThread()
        self.preproc_worker = PreprocessWorker(self.roster, [str(v) for v in vols])
        self.preproc_worker.moveToThread(self.preproc_thread)
        self.preproc_thread.started.connect(self.preproc_worker.run)
        self.preproc_worker.progress.connect(
            lambda d: self.status_lbl.setText(f"Reading preprocess… {d}"))
        self.preproc_worker.done.connect(self._preprocess_done)
        self.preproc_worker.failed.connect(self._preprocess_failed)
        self.preproc_worker.done.connect(self.preproc_thread.quit)
        self.preproc_worker.failed.connect(self.preproc_thread.quit)
        self.preproc_thread.start()

    def _preprocess_done(self, rows: list):
        self.preproc_btn.setEnabled(True)
        done = sum(r["n_done"] for r in rows)
        exp = sum(r["n_expected"] for r in rows)
        self.status_lbl.setText(f"Preprocess: {done}/{exp} expected step-outputs present.")
        PreprocessDialog(rows, self).exec()

    def _preprocess_failed(self, msg: str):
        self.preproc_btn.setEnabled(True)
        QMessageBox.critical(self, "Preprocess check failed", msg)
        self.status_lbl.setText("Preprocess check failed.")

    # -- prepare RecordingMeta ---------------------------------------------
    def _prepare_meta(self, auto: bool = False):
        if not self.roster or self._closing:
            return
        if self._search_busy():
            self.status_lbl.setText(
                "Scan done · auto RecordingMeta skipped (a drive search is running)." if auto
                else "A drive search is still running — try again when it finishes.")
            return
        path = self.xlsx_edit.text().strip()
        try:
            raw = pd.read_excel(path, sheet_name=self.sheet_combo.currentText().strip() or "Raw")
            raw = raw.dropna(subset=["subject"])
        except Exception as exc:
            if auto:
                self.status_lbl.setText(f"Scan done · auto RecordingMeta skipped ({exc}).")
                return
            QMessageBox.critical(self, "Could not read sheet", str(exc))
            return
        self._meta_raw = raw
        self.status_lbl.setText("Searching the drive folders for video folders…")
        self._start_search("Looking…", lambda msg: self._meta_failed(msg, auto),
                           on_index=lambda combined: self._meta_plan(combined, auto))

    def _meta_plan(self, combined: dict, auto: bool = False):
        """Plan (and, in auto mode, write the unambiguous rows) off the GUI thread."""
        if self._closing:
            return
        if self.meta_thread is not None:
            self.meta_thread.wait(2000)
        self.status_lbl.setText("Planning RecordingMeta…")
        self.meta_thread = QThread()
        self.meta_worker = MetaWorker(self.roster, combined.get("sessions", {}),
                                      self._meta_raw, auto)
        self.meta_worker.moveToThread(self.meta_thread)
        self.meta_thread.started.connect(self.meta_worker.run)
        self.meta_worker.done.connect(self._meta_done)
        self.meta_worker.failed.connect(self._meta_failed)
        self.meta_worker.done.connect(self.meta_thread.quit)
        self.meta_worker.failed.connect(self.meta_thread.quit)
        self.meta_thread.finished.connect(self._thread_ended)
        self.meta_thread.start()
        self._refresh_actions()

    def _meta_done(self, plan: list, results: list, auto: bool):
        n_write = sum(1 for p in plan if p["action"] == pmeta.WRITE)
        if auto:
            # Auto mode (after scan): MetaWorker already wrote the WRITE rows whose
            # pairing is unambiguous; the rest wait for the reviewed dialog.
            nok = sum(1 for r in results if r["result"] == "written")
            nerr = sum(1 for r in results if r["result"] == "error")
            held = sum(1 for p in plan if p["action"] == pmeta.WRITE and p.get("ambiguous"))
            msg = f"Scan done · auto-wrote {nok} RecordingMeta.xlsx"
            if nerr:
                msg += f" ({nerr} error)"
            if held:
                msg += (f" · {held} held back (session/folder counts differ — use "
                        f"Fix / Prepare ▸ Prepare RecordingMeta to review)")
            self.status_lbl.setText(msg + ".")
            return
        self.status_lbl.setText(f"RecordingMeta: {n_write} to write.")
        MetaDialog(plan, self).exec()

    def _meta_failed(self, msg: str, auto: bool = False):
        if auto:
            self.status_lbl.setText(f"Scan done · auto RecordingMeta step failed ({msg}).")
            return
        QMessageBox.critical(self, "Prepare failed", msg)
        self.status_lbl.setText("Prepare failed.")

    # -- helpers -----------------------------------------------------------
    def _set(self, r: int, c: int, val, status: str = "?"):
        # str() rather than trusting the caller: Qt raises on a non-str cell, and
        # a single odd value from a spreadsheet should show up as one strange
        # cell, not abort the whole table with a TypeError.
        item = QTableWidgetItem("" if val is None else str(val))
        _paint(item, _STATUS_COLOR.get(status, Qt.GlobalColor.white))
        if COLUMNS[c] in _CENTER_COLS:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if c == _STATUS_COL and status in ("MISSING", "PARTIAL"):
            f = QFont()
            f.setBold(True)
            item.setFont(f)
        self.table.setItem(r, c, item)

    def _open_folder(self, r: int, _c: int):
        if r >= len(self.results) or not self.results[r]:
            return
        paths = self.results[r].get("paths") or []
        if paths:
            _reveal(paths[0])
        elif self.results[r].get("status") in ("MISSING", "PARTIAL"):
            # nothing found by the scan — offer the deep search of the whole drives
            self._search_all()

    def _export(self):
        if not any(self.results):
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export coverage",
                                           str(Path.home() / "drive_coverage.xlsx"),
                                           "Excel (*.xlsx)")
        if not f:
            return
        cols = ["rat", "date8", "day", "session", "repeat", "expected", "implanted",
                "n_video", "pre_gb", "task_gb", "post_gb", "short_phases", "split",
                "n_rec", "n_merged", "n_logger", "found_in", "status"]
        out = []
        for r in self.results:
            if not r:
                continue
            pg = r.get("phase_gb", {})
            imp = r["implanted"]
            out.append(dict(
                rat=r["rat"], date8=r["date8"], day=r["day"], session=r["session"],
                repeat=_rep(r["repeat"]),
                expected=r["expected"], implanted=int(imp),
                n_video=r["n_video"],
                pre_gb=pg.get("pre", 0) if imp else "",
                task_gb=pg.get("task", 0) if imp else "",
                post_gb=pg.get("post", 0) if imp else "",
                short_phases="+".join(r.get("short_phases", [])),
                split=r.get("split_text", ""),
                n_rec=r.get("n_rec", 0), n_merged=r.get("n_merged", 0),
                n_logger=r.get("n_logger", 0),
                found_in="; ".join(r.get("paths", [])) or r["found_in"],
                status=r["status"]))
        ok, msg = sd.write_xlsx(_xlsx(f), [("coverage", cols, out)])
        if ok:
            self.status_lbl.setText(f"Exported → {msg}")
        else:
            QMessageBox.critical(self, "Export failed", msg)


def main():
    app = QApplication(sys.argv)
    gui = ScanDriveGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
