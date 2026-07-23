"""Prepare a per-session ``RecordingMeta.xlsx`` (the tracker's input config) from
the experiment behavioural log, and drop it next to each session's videos.

The experiment ``Raw`` sheet holds one row per trial. For each maze session the
tracker needs, this pulls that session's trials and writes them in the layout the
tracker reads (see TrackerYolov11):

  * session-level fields on the FIRST row only — Rat_ID, Date, Repeat, Day,
    Session, Num_Trials, and the optional crop / resume fields;
  * one row PER TRIAL carrying ``Start_Nodes``, ``Goal_Node`` and ``Trial_Type`` —
    the three lists the tracker consumes by trial index, so each must be exactly
    ``Num_Trials`` long. Trial_Type genuinely varies within a session, so it is
    written per trial, not broadcast.

Researcher-only fields the log cannot supply (Special_Trials, Unnormal_Intervals,
the start/stop crop) are left blank for the researcher to fill.

``Date`` is written as the ``YYYYMMDD`` the tracker matches against the folder
name. An existing RecordingMeta.xlsx is never overwritten — a researcher-filled
file is left untouched; only missing ones are created.
"""

from __future__ import annotations

import re
import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd

# A camera folder's start, from its timestamp name (2019-05-27_11-51-08) as a
# sortable YYYYMMDDHHMMSS key; falls back to the name so ordering stays stable.
_TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})")


def _folder_start(folder_str: str) -> str:
    m = _TS_RE.search(Path(folder_str).name)
    return "".join(m.groups()) if m else Path(folder_str).name

# The tracker's expected column order (from examples/RecordingMeta.xlsx).
META_COLS = ["Rat_ID", "Date", "Repeat", "Day", "Session", "Goal_Node",
             "Prev_Goal_Node", "Num_Trials", "Trial_Type", "Start_Nodes",
             "Special_Trials", "Special_Trials_End", "Unnormal_Intervals",
             "Start_Min", "Start_Sec", "Stop_Min", "Stop_Sec", "Start_At_Trial_Num"]

_EYE_RE = re.compile(r"^eye\d+.*\.mp4$", re.I)
_META_NAME = "RecordingMeta.xlsx"


def _date8(val) -> str | None:
    """Normalise a Raw ``Date`` cell to YYYYMMDD (mirrors the GUI's parser)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return val.strftime("%Y%m%d")
    s = str(val).strip()
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y%m%d", "%d.%m.%y"):
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y%m%d")
    except Exception:
        return None


def _int_or(v, default=None):
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _session_goal(sub: pd.DataFrame) -> int | None:
    """A session's representative goal node (the most common), for Prev_Goal_Node."""
    g = sub["goal_node_n"].dropna()
    if g.empty:
        return None
    return _int_or(g.mode().iloc[0])


def build_meta_df(raw: pd.DataFrame, subject: int, day: int, session: int,
                  prev_goal: int | None = None) -> pd.DataFrame | None:
    """RecordingMeta rows for one (subject, day, session), or None if it has no
    usable trials. One row per trial; session fields on the first row only."""
    sub = raw[(raw["subject"] == subject) & (raw["day"] == day)
              & (raw["session"] == session)].copy()
    if sub.empty:
        return None
    if "trial" in sub.columns:
        sub = sub.sort_values("trial")
    # a trial is usable only if it has a start node to trigger on
    sub = sub[sub["start_node_n"].notna()]
    n = len(sub)
    if n == 0:
        return None

    date8 = _date8(sub["Date"].dropna().iloc[0]) if sub["Date"].notna().any() else None
    repeat = _int_or(sub["repeat"].dropna().iloc[0]) if "repeat" in sub and sub["repeat"].notna().any() else None
    starts = [_int_or(v) for v in sub["start_node_n"].tolist()]
    goals = [_int_or(v) for v in sub["goal_node_n"].tolist()]
    types = [_int_or(v, 1) for v in sub["trial_type"].tolist()] if "trial_type" in sub else [1] * n

    blank = [pd.NA] * n

    def first_only(v):
        return [v] + [pd.NA] * (n - 1)

    data = {
        "Rat_ID": first_only(int(subject)),
        "Date": first_only(int(date8) if date8 else pd.NA),
        "Repeat": first_only(repeat if repeat is not None else pd.NA),
        "Day": first_only(int(day)),
        "Session": first_only(int(session)),
        "Goal_Node": goals,
        "Prev_Goal_Node": first_only(prev_goal if prev_goal is not None else pd.NA),
        "Num_Trials": first_only(n),
        "Trial_Type": types,
        "Start_Nodes": starts,
        "Special_Trials": blank,
        "Special_Trials_End": blank,
        "Unnormal_Intervals": blank,
        "Start_Min": first_only(0),
        "Start_Sec": first_only(0),
        "Stop_Min": first_only(0),
        "Stop_Sec": first_only(0),
        "Start_At_Trial_Num": first_only(1),
    }
    return pd.DataFrame(data, columns=META_COLS)


def session_prev_goals(raw: pd.DataFrame) -> dict:
    """Map (subject, day, session) -> the previous session's goal node, in the
    chronological order the rat ran them, so Prev_Goal_Node can be filled."""
    keys = (raw.dropna(subset=["subject", "day", "session"])
            .groupby(["subject", "day", "session"]).size().reset_index())
    prev: dict = {}
    last_goal_by_rat: dict = {}
    for _, r in keys.sort_values(["subject", "day", "session"]).iterrows():
        subj, day, sess = int(r["subject"]), int(r["day"]), int(r["session"])
        prev[(subj, day, sess)] = last_goal_by_rat.get(subj)
        sub = raw[(raw["subject"] == subj) & (raw["day"] == day) & (raw["session"] == sess)]
        g = _session_goal(sub)
        if g is not None:
            last_goal_by_rat[subj] = g
    return prev


def find_video_folders(session_path) -> list[Path]:
    """Folders that actually hold this session's ``eye<NN>*.mp4`` — the loose date
    folder itself, and/or any timestamp subfolder. Empty if the session has no
    camera videos (e.g. an ephys-only recording)."""
    p = Path(session_path)
    out = []
    try:
        if any(_EYE_RE.match(c.name) for c in p.iterdir() if c.is_file()):
            out.append(p)
        for sub in sorted(c for c in p.iterdir() if c.is_dir()):
            try:
                if any(_EYE_RE.match(g.name) for g in sub.iterdir() if g.is_file()):
                    out.append(sub)
            except OSError:
                continue
    except OSError:
        pass
    return out


# Plan actions
WRITE = "write"            # will create a new RecordingMeta.xlsx here
EXISTS = "exists"          # a RecordingMeta.xlsx is already here — left untouched
NO_VIDEO = "no-video"      # session found, but no camera folder to write into
NO_TRIALS = "no-trials"    # no behavioural trials in the sheet for this session


def plan_meta(roster: list[dict], found: dict, raw: pd.DataFrame) -> list[dict]:
    """Work out where each session's RecordingMeta.xlsx would be written. Reads
    only; writes nothing. One row per session (mapped to its own video folder).

    A rat can run several sessions in a day, each its own timestamp camera folder.
    Those folders are matched to sessions by time: within one (rat, date) the
    folders are ordered by start time and zipped onto that rat's sessions taken in
    training order, so session N's meta lands in session N's folder — not smeared
    across every folder of the day. Run this after organizing, when each folder is
    filed under its correct ``Rat<N>/<date>``."""
    prev_goals = session_prev_goals(raw)
    by_rd: dict = defaultdict(list)
    for e in roster:
        if e["date8"]:
            by_rd[(e["rat_no"], e["date8"])].append(e)

    plan = []
    for (rat_no, date8), sessions in sorted(by_rd.items()):
        # Sessions in training order (= recording order that day); folders in time
        # order. Zip position i to position i.
        sessions = sorted(sessions, key=lambda x: (x.get("train_order", x["session"]),
                                                   x["session"]))
        folders = []
        for _label, sess in _iter_session_paths(found, rat_no, date8):
            folders += find_video_folders(sess)
        folders = sorted({str(f) for f in folders}, key=_folder_start)

        for i, e in enumerate(sessions):
            df = build_meta_df(raw, e["rat_no"], e["day"], e["session"],
                               prev_goals.get((e["rat_no"], e["day"], e["session"])))
            base = dict(rat=e["rat"], rat_no=e["rat_no"], date8=e["date8"],
                        day=e["day"], session=e["session"], repeat=e.get("repeat"),
                        n_trials=0 if df is None else len(df))
            if df is None:
                plan.append(dict(base, action=NO_TRIALS, folder="", detail="no trials in sheet"))
                continue
            # i keeps session<->folder aligned even when a no-trials session is
            # skipped above; a session past the last folder simply has no video.
            if i >= len(folders):
                plan.append(dict(base, action=NO_VIDEO, folder="",
                                 detail="no video folder for this session"))
                continue
            f = folders[i]
            exists = (Path(f) / _META_NAME).exists()
            plan.append(dict(base, action=EXISTS if exists else WRITE, folder=f,
                             df=df, detail="already present" if exists
                             else f"{len(df)} trials"))
    return plan


def _iter_session_paths(found: dict, rat_no: int, date8: str):
    for h in found.get((rat_no, date8), []):
        yield h.get("volume", ""), h["path"]


def write_plan(plan: list[dict], overwrite: bool = False) -> list[dict]:
    """Write the RecordingMeta.xlsx for every WRITE row (and, if `overwrite`, the
    EXISTS rows too). Never touches anything else. Returns result rows."""
    results = []
    for item in plan:
        if item["action"] == WRITE or (overwrite and item["action"] == EXISTS):
            dst = Path(item["folder"]) / _META_NAME
            try:
                item["df"].to_excel(dst, index=False)
                results.append(dict(item, result="written", detail=str(dst)))
            except OSError as exc:
                results.append(dict(item, result="error", detail=str(exc)))
        else:
            results.append(dict(item, result="skipped", detail=item.get("detail", "")))
    return results
