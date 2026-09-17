"""Writing the session NWB that the sleep scorer reads. The tracker's half.

ONE FILE PER SESSION: ``<op>/<Rat>_<YYYYMMDD>[_<phase>].nwb``. **Step 8**
creates it here and writes the per-sample data; **step w** (behaviour) and
**step u** (units) then append to that same file in ``r+`` mode. Nothing
rewrites it from scratch, so a scoring stored in it survives the pipeline.

What step 8 writes::

    acquisition/lfp             (n_samples, n_channels) uV, on a rate
    acquisition/emg_from_lfp    normalised EMG-from-LFP (~5 Hz)
    acquisition/motion          accelerometer movement magnitude
    processing/sleep/
        layout                  the container names used here, recorded so a
                                reader never has to agree with us in advance
        sleep_channels          per-rat cortex / sr / pyr tetrodes (JSON)
        session_info            channel map, session boundaries, SNR (JSON)
        awakeness, emg_rms, theta_delta_ratio

**This module is NOT shared with the scoring GUI.** It used to be kept
byte-identical with ``HM_rat_sleep_score/python/sleep_nwb_store.py``, which was a
standing trap: rename a container on one side and the other silently reads
nothing. Instead, every file now carries the ``layout`` record above, and the
scorer adopts those names per file. So the two modules are free to diverge —
only keep :func:`layout_contract` honest about what this writer actually does,
and files stay readable.

The scorings themselves (``processing/sleep/states_<scorer>``) belong to the
scorer; nothing here reads or writes them.
"""

from __future__ import annotations

import json
import re
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
LAYOUT_NAME = "layout"          # where the contract below is recorded in the file
LAYOUT_VERSION = 1

# HM state codes, shared with the editor (1 WAKE / 3 NREM / 5 REM; 0 unscored).
STATE_NAMES = {0: "none", 1: "awake", 3: "NREM", 4: "intermediate",
               5: "REM"}   # 4 = the NREM->REM transition, scored by hand


# ---------------------------------------------------------------------------- #
#  The layout record
# ---------------------------------------------------------------------------- #
# Stamped into every file this module writes. Readers take the names from here
# rather than hardcoding their own, which is what lets the scorer's module and
# this one evolve independently. Keep it honest: if a container above is
# renamed, rename it here too and bump LAYOUT_VERSION.


def layout_contract() -> dict:
    """The names this copy of the module reads and writes."""
    return {
        "layout_version": LAYOUT_VERSION,
        "module": SLEEP_MODULE,
        "lfp": LFP_NAME,
        "emg": EMG_NAME,
        "motion": MOTION_NAME,
        "sleep_channels": SLEEP_CHANNELS_NAME,
        "session_info": SESSION_INFO_NAME,
        "derived": list(DERIVED_NAMES),
        "states_prefix": STATES_PREFIX,
        "events_prefix": EVENTS_PREFIX,
        # informational only — a scoring carries its own copy, and adding a
        # state is additive, so a difference here is never fatal
        "state_names": {str(k): v for k, v in STATE_NAMES.items()},
    }


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

    # Everything below can raise (a layout mismatch, a malformed series). The
    # handle is only handed back to the caller on success, so close it on the
    # way out of a failure — otherwise the file stays open read-only and the
    # next writer cannot reopen it.
    try:
        # This module both writes and reads these files, so the names below are
        # by definition the ones they were written with.
        mod = _sleep_module(nwbfile)

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

        out["sleep_channels"] = _read_json(mod, SLEEP_CHANNELS_NAME)
        out["session_info"] = _read_json(mod, SESSION_INFO_NAME)
        out["derived"] = {}
        if mod is not None:
            for name in DERIVED_NAMES:
                if name in mod.data_interfaces:
                    series = mod[name]
                    out["derived"][name] = (series.data if lazy
                                            else np.asarray(series.data[:]).ravel())
    except Exception:
        io.close()
        raise

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


def _stamp_layout(mod):
    """Record this module's container names in the file. Returns True if added."""
    from pynwb import TimeSeries

    if LAYOUT_NAME in mod.data_interfaces:
        return False
    mod.add(TimeSeries(name=LAYOUT_NAME, data=[LAYOUT_VERSION], unit="n/a", rate=1.0,
                       description=json.dumps(layout_contract())))
    return True


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

    # Anything below lives under processing/sleep, so the module is created here
    # and stamped with the layout contract every reader checks. Everything that
    # follows is guarded by the same conditions, so `mod` is set whenever used.
    mod = None
    if sleep_channels or derived or metadata or added:
        mod = _sleep_module(nwbfile, create=True)
        if _stamp_layout(mod):
            added.append(LAYOUT_NAME)

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


