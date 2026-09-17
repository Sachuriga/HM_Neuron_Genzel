"""Shared NWB layout for sleep scoring — the contract between the tracker and
the scoring GUI.

ONE FILE PER SESSION: ``<op>/<Rat>_<YYYYMMDD>.nwb``. The tracker's **step 8**
creates it and writes everything the scorer needs; **step w** (behaviour /
trials) and **step u** (units) then append to that same file in ``r+`` mode.
Nothing rewrites it from scratch, so scorings stored inside survive the rest
of the pipeline.

Layout
------
``acquisition/``
    ``lfp``            TimeSeries ``(n_samples, n_channels)`` in uV, per-sample
                       timestamps (seconds). Written by step 8; step w skips it
                       when already present.
    ``emg_from_lfp``   TimeSeries ``(n,)`` normalised EMG-from-LFP (~5 Hz).
    ``motion``         TimeSeries ``(n,)`` accelerometer movement magnitude.

``processing/sleep/``
    ``sleep_channels``       TimeSeries carrying the per-rat cortex / sr / pyr
                             tetrode numbers as JSON in its ``description``.
    ``states_<scorer>``      TimeIntervals — one row per contiguous scored epoch
                             (``start_time``, ``stop_time``, ``state``,
                             ``state_code``); JSON metadata in ``description``.
    ``events_<scorer>``      TimeSeries of numbered event marks (optional).

A *scoring* is identified by its ``scorer`` name; re-saving under the same name
replaces that scorer's tables, so re-opening and continuing a scoring never
spawns a second copy. ``strip_scorings`` writes a copy with every scoring
removed — the version to hand to students, with no ground truth in it.

This file is kept byte-identical in both repos:
  * HM_Tracker_2025/src/nwb/sleep_nwb.py
  * HM_rat_sleep_score/python/sleep_nwb.py
Edit one, copy to the other.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- container names (the contract) ---------------------------------------- #
LFP_NAME = "lfp"
EMG_NAME = "emg_from_lfp"
MOTION_NAME = "motion"
SLEEP_MODULE = "sleep"
SLEEP_CHANNELS_NAME = "sleep_channels"
SESSION_INFO_NAME = "session_info"
# per-sample signals derived from the LFP by the tracker's LFP export
DERIVED_NAMES = ("awakeness", "emg_rms", "theta_delta_ratio")
STATES_PREFIX = "states_"
EVENTS_PREFIX = "events_"

# HM state codes, shared with the editor (1 WAKE / 3 NREM / 5 REM; 0 unscored).
STATE_NAMES = {0: "none", 1: "awake", 3: "NREM", 4: "intermediate",
               5: "REM"}   # 4 = the NREM->REM transition, scored by hand


def _slug(name: str) -> str:
    """Filesystem/NWB-safe token for a scorer name ('Sachu Riga' -> 'Sachu_Riga')."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "unknown")).strip("_")
    return s or "unknown"


# ---------------------------------------------------------------------------- #
#  Locating the session NWB
# ---------------------------------------------------------------------------- #
def folder_postfix(name_or_path) -> str:
    """The phase token trailing a **recording name**, e.g. ``'post'``.

    Recordings are named ``<Rat>_<label>_<YYYYMMDD>_<HHMMSS>`` with an optional
    phase suffix, so ``Rat5_HM_Neurons_20260807_123703_post`` gives ``'post'``
    and ``Rat1_HM_Neurons_20260211_104846`` gives ``''``.

    Take this from the recording (the ``.LFP`` folder / session name found in
    the **ip** folder), never from the op folder — op folders are named ``op1``,
    ``op6`` … and carry no phase. Sessions recorded on the same day are told
    apart by this token, which is why it belongs in the NWB's name.
    """
    if not name_or_path:
        return ""
    m = re.search(r"_\d{8}_\d{6}[_-]*(.*)$", Path(str(name_or_path)).name)
    if not m:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "_", m.group(1)).strip("_")


def session_nwb_name(prefix: str, session_name=None) -> str:
    """The session NWB's filename: ``Rat6_20260629.nwb``, or
    ``Rat5_20260807_post.nwb`` when the recording carries a phase postfix.

    ``prefix`` is a step-8 file prefix (``Rat6_20260629_143022_``) and
    ``session_name`` the recording's name (from the ip folder), whose postfix
    is appended.

    Step w does not recompute this: it looks for the file step 8 already wrote
    (:func:`find_session_nwb`) and only falls back to a name of its own when
    there is none — so the two can never disagree.
    """
    m = re.match(r"^([A-Za-z]+\d+)_(\d{8})", str(prefix or ""))
    base = f"{m.group(1)}_{m.group(2)}" if m else "session"
    post = folder_postfix(session_name)
    return f"{base}_{post}.nwb" if post else f"{base}.nwb"


def find_session_nwb(folder):
    """The session NWB for an op folder, or None.

    Accepts the op folder itself or its ``LFP_Output`` subfolder (the scoring
    GUI is usually pointed at the latter), and ignores ``*.tmp.nwb`` leftovers.
    """
    folder = Path(folder)
    roots = [folder, folder.parent] if folder.name == "LFP_Output" else [folder]
    for root in roots:
        cands = [p for p in sorted(root.glob("*.nwb"))
                 if not p.name.endswith(".tmp.nwb")]
        if cands:
            return cands[0]
    return None


# ---------------------------------------------------------------------------- #
#  Reading the scoring inputs
# ---------------------------------------------------------------------------- #
def read_sleep_inputs(nwb_path, lazy=True):
    """Read everything the scorer needs from ``nwb_path``.

    Returns a dict with ``lfp`` (an h5py-backed ``(n_samples, n_channels)``
    array when ``lazy``, so channels are sliced without loading the session),
    ``lfp_timestamps``, ``fs``, ``emg``/``emg_timestamps``,
    ``motion``/``motion_timestamps``, ``sleep_channels`` and the open ``io``
    handle (close it via ``close_inputs``). Missing pieces come back as None.
    """
    from pynwb import NWBHDF5IO

    io = NWBHDF5IO(str(nwb_path), mode="r")
    nwbfile = io.read()
    out = {"io": io, "nwbfile": nwbfile, "path": str(nwb_path)}

    def _acq(name):
        try:
            return nwbfile.get_acquisition(name)
        except Exception:
            return None

    lfp = _acq(LFP_NAME)
    if lfp is not None:
        out["lfp"] = lfp.data if lazy else np.asarray(lfp.data)
        out["fs"] = _series_rate(lfp)
        out["lfp_timestamps"] = _series_times(lfp, lfp.data.shape[0])
    else:
        out["lfp"] = out["lfp_timestamps"] = out["fs"] = None

    for key, name in (("emg", EMG_NAME), ("motion", MOTION_NAME)):
        series = _acq(name)
        if series is None:
            out[key] = out[f"{key}_timestamps"] = None
            continue
        values = np.asarray(series.data[:]).ravel()
        out[key] = values
        out[f"{key}_timestamps"] = _series_times(series, values.size)

    mod = _sleep_module(nwbfile)
    out["sleep_channels"] = _read_json(mod, SLEEP_CHANNELS_NAME)
    out["session_info"] = _read_json(mod, SESSION_INFO_NAME)
    out["derived"] = {}
    if mod is not None:
        for name in DERIVED_NAMES:
            if name in mod.data_interfaces:
                series = mod[name]
                out["derived"][name] = (series.data if lazy
                                        else np.asarray(series.data[:]).ravel())
    if not lazy:
        io.close()
        out["io"] = None
    return out


def _series_rate(series):
    """Sampling rate of a TimeSeries, whether it stores a rate or timestamps."""
    if getattr(series, "rate", None):
        return float(series.rate)
    ts = getattr(series, "timestamps", None)
    if ts is not None and len(ts) > 1:
        head = np.asarray(ts[:1000])
        dt = float(np.median(np.diff(head))) if head.size > 1 else 0.0
        return 1.0 / dt if dt > 0 else None
    return None


def _series_times(series, n):
    """The series' time axis, synthesised from its rate when it has no explicit
    timestamps array (that is how uniformly-sampled signals are stored)."""
    ts = getattr(series, "timestamps", None)
    if ts is not None:
        return np.asarray(ts[:]).ravel()
    rate = getattr(series, "rate", None)
    if not rate:
        return None
    t0 = float(getattr(series, "starting_time", 0.0) or 0.0)
    return t0 + np.arange(int(n), dtype=np.float64) / float(rate)


def close_inputs(inputs):
    """Close the handle returned by ``read_sleep_inputs(lazy=True)``."""
    io = (inputs or {}).get("io")
    if io is not None:
        try:
            io.close()
        except Exception:
            pass


def _sleep_module(nwbfile, create=False):
    mod = nwbfile.processing.get(SLEEP_MODULE)
    if mod is None and create:
        mod = nwbfile.create_processing_module(
            name=SLEEP_MODULE,
            description="Sleep scoring: per-rat channel choices and one "
                        "TimeIntervals table per scorer.")
    return mod


def _read_json(mod, name):
    """A JSON record carried in a placeholder series' description, or None."""
    if mod is None or name not in mod.data_interfaces:
        return None
    try:
        return json.loads(mod[name].description)
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
#  Writing the scoring inputs (tracker step 8)
# ---------------------------------------------------------------------------- #
def add_sleep_inputs(nwbfile, lfp=None, lfp_timestamps=None, lfp_rate=None,
                     emg=None, emg_timestamps=None, emg_rate=None,
                     motion=None, motion_timestamps=None, motion_rate=None,
                     sleep_channels=None, derived=None, metadata=None,
                     lfp_unit="uV"):
    """Add the scorer's inputs to ``nwbfile``. Existing containers are left
    alone, so this is safe to re-run against a session that step w already
    populated. Returns the list of names actually added.

    A signal on a uniform clock should be given a ``*_rate`` rather than an
    explicit timestamps array: NWB then stores only the rate, which spares the
    file an 8-byte-per-sample time axis (hundreds of MB over a long session).

    ``derived`` holds per-sample signals computed from the LFP
    (``awakeness``, ``emg_rms``, ``theta_delta_ratio``), each on the LFP clock.
    ``metadata`` holds the small session records (channel map, session
    boundaries, cleanest channels, SNR scores) as JSON.
    """
    from pynwb import TimeSeries

    added = []

    def _series(name, data, ts, rate, unit, description):
        ts = None if ts is None else np.asarray(ts).ravel()
        # A chunk iterator / H5DataIO streams straight to HDF5 (a multi-GB LFP
        # never lands in RAM); only a plain array can be length-reconciled here.
        if isinstance(data, (np.ndarray, list, tuple)):
            data = np.asarray(data)
            if ts is not None and data.shape[0] != ts.shape[0]:
                n = min(data.shape[0], ts.shape[0])
                data, ts = data[:n], ts[:n]
        if ts is not None:
            kw = {"timestamps": ts}
        else:
            kw = {"rate": float(rate or 1.0), "starting_time": 0.0}
        return TimeSeries(name=name, data=data, unit=unit,
                          description=description, **kw)

    def _add_acq(name, data, ts, rate, unit, description):
        if data is None:
            return
        try:
            nwbfile.get_acquisition(name)
            return                      # already there — never clobber
        except Exception:
            pass
        nwbfile.add_acquisition(_series(name, data, ts, rate, unit, description))
        added.append(name)

    _add_acq(LFP_NAME, lfp, lfp_timestamps, lfp_rate, lfp_unit,
             "LFP voltage, one column per channel (see sleep/session_info "
             "for the channel map).")
    _add_acq(EMG_NAME, emg, emg_timestamps, emg_rate, "normalized",
             "EMG-from-LFP (Buzsaki cross-channel correlation), 0-1 normalised.")
    _add_acq(MOTION_NAME, motion, motion_timestamps, motion_rate, "g",
             "Accelerometer (IMU) movement magnitude.")

    if sleep_channels or derived or metadata:
        mod = _sleep_module(nwbfile, create=True)

    if sleep_channels and SLEEP_CHANNELS_NAME not in mod.data_interfaces:
        mod.add(TimeSeries(
            name=SLEEP_CHANNELS_NAME, data=[0], unit="n/a", rate=1.0,
            description=json.dumps(_jsonable(dict(sleep_channels)))))
        added.append(SLEEP_CHANNELS_NAME)

    for name, spec in (derived or {}).items():
        if spec is None or name in mod.data_interfaces:
            continue
        data, rate, unit, desc = (spec if isinstance(spec, tuple)
                                  else (spec, lfp_rate, "a.u.", f"{name} (per LFP sample)."))
        if data is None:
            continue
        mod.add(_series(name, data, None, rate, unit, desc))
        added.append(name)

    if metadata and SESSION_INFO_NAME not in mod.data_interfaces:
        mod.add(TimeSeries(name=SESSION_INFO_NAME, data=[0], unit="n/a", rate=1.0,
                           description=json.dumps(_jsonable(dict(metadata)))))
        added.append(SESSION_INFO_NAME)
    return added


def _jsonable(v):
    """Make numpy scalars / arrays / nested containers JSON-serialisable."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


# ---------------------------------------------------------------------------- #
#  Scorings
# ---------------------------------------------------------------------------- #
def list_scorings(nwb_path):
    """Every scoring in the file, newest first.

    Each entry is a dict with ``scorer``, ``date``, ``n_bins``, ``name`` (the
    NWB table name) and ``label`` (what the "State scored by" dropdown shows).
    """
    from pynwb import NWBHDF5IO

    if not Path(nwb_path).is_file():
        return []
    with NWBHDF5IO(str(nwb_path), mode="r") as io:
        mod = _sleep_module(io.read())
        if mod is None:
            return []
        out = []
        for name, obj in mod.data_interfaces.items():
            if not name.startswith(STATES_PREFIX):
                continue
            meta = _meta_of(obj)
            meta["name"] = name
            meta.setdefault("scorer", name[len(STATES_PREFIX):])
            out.append(meta)
    out.sort(key=lambda m: (m.get("date") or "", m.get("scorer") or ""), reverse=True)
    for m in out:
        date = m.get("date") or "?"
        n = m.get("n_bins")
        m["label"] = f"{m.get('scorer')} · {date}" + (f" · {n} bins" if n else "")
    return out


def _meta_of(table):
    try:
        return dict(json.loads(table.description))
    except Exception:
        return {}


def read_scoring(nwb_path, scorer=None, name=None, timestamps=None):
    """Read one scoring back as per-bin state codes.

    Picks the scoring by NWB table ``name`` or by ``scorer``; with neither it
    takes the newest. Returns ``(states, bin_timestamps, meta)`` — ``states``
    aligned to ``timestamps`` when given, else to the bin grid the scoring was
    saved on. Returns ``(None, None, {})`` when there is nothing to read.
    """
    from pynwb import NWBHDF5IO

    entries = list_scorings(nwb_path)
    if not entries:
        return None, None, {}
    if name:
        entries = [e for e in entries if e["name"] == name] or entries
    elif scorer:
        entries = [e for e in entries if e.get("scorer") == scorer] or entries
    target = entries[0]["name"]

    with NWBHDF5IO(str(nwb_path), mode="r") as io:
        nwbfile = io.read()
        mod = _sleep_module(nwbfile)
        table = mod[target]
        meta = _meta_of(table)
        starts = np.asarray(table["start_time"][:], dtype=float)
        stops = np.asarray(table["stop_time"][:], dtype=float)
        codes = np.asarray(table["state_code"][:], dtype=int)
        events = _read_events(mod, meta.get("scorer", ""))

    if timestamps is None:
        dt = float(meta.get("dt") or 1.0)
        t0 = float(meta.get("t0", starts[0] if starts.size else 0.0))
        n = int(meta.get("n_bins") or (round((stops[-1] - t0) / dt) if stops.size else 0))
        timestamps = t0 + np.arange(n) * dt
    timestamps = np.asarray(timestamps, dtype=float)

    states = np.zeros(timestamps.size, dtype=int)
    for a, b, c in zip(starts, stops, codes):
        states[(timestamps >= a) & (timestamps <= b)] = c
    meta["events"] = events
    meta["name"] = target
    return states, timestamps, meta


def _read_events(mod, scorer):
    name = EVENTS_PREFIX + _slug(scorer)
    if name not in mod.data_interfaces:
        return np.zeros((0, 2))
    ev = mod[name]
    nums = np.asarray(ev.data[:]).ravel()
    times = np.asarray(ev.timestamps[:]).ravel()
    return np.column_stack([nums, times]) if nums.size else np.zeros((0, 2))


def write_scoring(nwb_path, states, timestamps, scorer, events=None,
                  date=None, extra=None):
    """Save one scorer's result into the session NWB (in place, ``r+``).

    Stored as a ``TimeIntervals`` of contiguous epochs under
    ``processing/sleep/states_<scorer>``. Re-saving the same scorer REPLACES
    that scorer's tables rather than adding a second copy, so re-opening and
    continuing a scoring keeps one row per scorer. Returns the table name.
    """
    from pynwb import NWBHDF5IO, TimeSeries
    from pynwb.epoch import TimeIntervals

    states = np.asarray(states, dtype=int).ravel()
    timestamps = np.asarray(timestamps, dtype=float).ravel()
    n = min(states.size, timestamps.size)
    states, timestamps = states[:n], timestamps[:n]
    slug = _slug(scorer)
    table_name = STATES_PREFIX + slug
    events_name = EVENTS_PREFIX + slug
    dt = float(np.median(np.diff(timestamps))) if n > 1 else 1.0
    date = date or datetime.now(timezone.utc).astimezone().date().isoformat()

    meta = {"scorer": str(scorer), "date": date, "n_bins": int(n), "dt": dt,
            "t0": float(timestamps[0]) if n else 0.0,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "state_names": {str(k): v for k, v in STATE_NAMES.items()}}
    meta.update(extra or {})

    # drop any previous tables for this scorer first, at the HDF5 level: pynwb
    # can only add to an r+ file, so replacing means deleting before reopening.
    _h5_delete(nwb_path, [table_name, events_name])

    io = NWBHDF5IO(str(nwb_path), mode="r+")
    try:
        nwbfile = io.read()
        mod = _sleep_module(nwbfile, create=True)

        table = TimeIntervals(name=table_name, description=json.dumps(meta))
        table.add_column(name="state", description="Sleep state name.")
        table.add_column(name="state_code",
                         description="HM state code: 1 awake, 3 NREM, 5 REM.")
        for s, e, code in _runs(states):
            if code == 0:
                continue                       # unscored bins leave a gap
            stop = timestamps[e] if e < n else timestamps[-1] + dt
            table.add_row(start_time=float(timestamps[s]), stop_time=float(stop),
                          state=STATE_NAMES.get(int(code), str(code)),
                          state_code=int(code))
        mod.add(table)

        events = np.asarray(events if events is not None else
                            np.zeros((0, 2)), dtype=float).reshape(-1, 2)
        if events.size:
            mod.add(TimeSeries(name=events_name, data=events[:, 0],
                               timestamps=events[:, 1], unit="n/a",
                               description=f"Event marks placed by {scorer}."))
        io.write(nwbfile)
    finally:
        io.close()
    return table_name


def _h5_delete(nwb_path, names):
    """Delete containers from ``processing/sleep`` at the HDF5 level.

    pynwb can only add to a file opened ``r+`` — detaching an already-written
    container from its ProcessingModule corrupts the build on the next write.
    Removing the group with h5py instead is reliable: a ProcessingModule lists
    its children by group membership, so the container is simply gone when
    pynwb next reads the file. Missing names are ignored. Returns the names
    actually deleted.

    (HDF5 does not reclaim the freed bytes; the file keeps its size until it is
    rewritten. That is fine here — scoring tables are a few KB.)
    """
    import h5py

    names = [n for n in names if n]
    if not names or not Path(nwb_path).is_file():
        return []
    deleted = []
    with h5py.File(str(nwb_path), "r+") as f:
        mod = f.get(f"/processing/{SLEEP_MODULE}")
        if mod is None:
            return []
        for name in names:
            if name in mod:
                del mod[name]
                deleted.append(name)
    return deleted


def _runs(states):
    """Yield ``(start, end_exclusive, value)`` for each contiguous run."""
    n = len(states)
    if n == 0:
        return
    start = 0
    for i in range(1, n + 1):
        if i == n or states[i] != states[start]:
            yield start, i, int(states[start])
            start = i


# ---------------------------------------------------------------------------- #
#  Ground-truth isolation
# ---------------------------------------------------------------------------- #
def strip_scorings(nwb_path, out_path, keep=()):
    """Write a copy of ``nwb_path`` to ``out_path`` with scorings removed.

    This is the version to hand out: identical recording, no ground truth.
    ``keep`` optionally names scorers to retain (e.g. a demo scoring). Returns
    the list of scorer names that were removed.
    """
    from pynwb import NWBHDF5IO

    out_path = Path(out_path)
    if out_path.resolve() == Path(nwb_path).resolve():
        raise ValueError("out_path must differ from nwb_path")
    keep_slugs = {_slug(k) for k in keep}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nwb_path, out_path)

    doomed, removed = [], []
    for entry in list_scorings(out_path):
        slug = entry["name"][len(STATES_PREFIX):]
        if slug in keep_slugs:
            continue
        doomed += [entry["name"], EVENTS_PREFIX + slug]
        removed.append(entry.get("scorer") or slug)
    _h5_delete(out_path, doomed)
    return removed
