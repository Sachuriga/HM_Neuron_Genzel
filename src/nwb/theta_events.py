"""
Step [v] add-on: per-unit event-locked and theta analyses.

For every good unit three families of figures are appended to its Unit_<cid>.pdf,
each computed in three splits — GOAL-RUNNING (type-1 trial windows), FREE-ROAMING
(type-4/5 windows) and ALL (whole session):

  1. Peri-event rasters + histograms (PETH) around three event types:
       start        — every trial's start time
       goal arrival — the animal entering the goal-node region (first entry per
                      goal-run trial; every debounced entry during free-roaming)
       bridge entry — the animal stepping onto one of the LONG INTER-ISLAND
                      bridges (~0.7 m corridors; short honeycomb edges don't count)
  2. Spike-theta phase coupling: phase histogram (double-plotted), mean resultant
     length R, Rayleigh p, preferred phase.
  3. CLASSIC phase precession: place fields detected in 2-D (>=35% of peak, sweep-calibrated);
     every APPROACH pass through a field (speed-gated, moving toward the goal,
     net progress >= 0.10 m on the 1-D axis = graph distance to goal) is
     normalised to position-in-field 0..1 (0 = entry, 1 = exit); ALL passes of
     ALL fields pool into ONE phase-vs-position plot per split with a single
     circular-linear fit — the traditional precession plot.

  PETH panels share one rate scale per unit; phase-coupling panels share one
  y scale per unit.

Theta comes from the session LFP — the session NWB's acquisition/lfp (step 8),
or the legacy <op>/LFP_Output/*.npy export for older sessions. The theta
channel is the rat's pyr tetrode (SLEEP_CHANNELS_<RAT> in the tracker config,
its lowest channel not in BAD_CHANNELS) — the same channel step 8 computes
theta/delta on. Without a pyr tetrode it is the channel with the highest
theta/delta power ratio (6-12 / 1-4 Hz, the tracker's one definition in
sleep_nwb_writer) during a movement-rich block, never a BAD_CHANNELS one. That
channel is band-passed 6-10 Hz and Hilbert-transformed. Phase convention: 0 deg = peak of
filtered theta, +-180 deg = trough. LFP sample 0 of the MAZE session aligns with
spike time 0 (verified: pooled spike-phase locking collapses when the LFP is
artificially shifted). If the LFP is missing the phase/precession pages are
skipped with a note; the PETH pages need no LFP.

All heavy inputs are computed ONCE per session in prepare(); the per-unit page
renderers only slice spikes.
"""
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_nwb as V      # noqa: E402
import neural_umap as NU       # noqa: E402  _hex_assign / _bridges (no umap import)
import sleep_nwb_writer as snwb  # noqa: E402  NWB names + the theta/delta ratio

# Band-pass for the theta PHASE only. Choosing the theta channel uses the
# tracker-wide theta/delta bands instead (snwb.theta_delta_ratio).
THETA_BAND = (6.0, 10.0)
PETH_WIN = 3.0                 # s each side of the event
PETH_BIN = 0.1                 # s
SPLITS = ("goal running", "free roaming", "all")
_SPLIT_COL = {"goal running": "#1f77b4", "free roaming": "#2ca02c", "all": "#555555"}


# ------------------------------------------------------------
#                 splits (time windows)
# ------------------------------------------------------------
def split_windows(trials, t_min, t_max):
    """{split: [(a, b), ...]} — goal-running = type-1 trials, free-roaming =
    type-4/5 trials, all = the whole session."""
    return {
        "goal running": [(a, b) for (tt, _g, _s, a, b) in trials if tt == 1],
        "free roaming": [(a, b) for (tt, _g, _s, a, b) in trials if tt in (4, 5)],
        "all": [(float(t_min), float(t_max))],
    }


def _in_windows(times, wins):
    """Boolean mask: which of `times` fall inside any (a, b) window."""
    times = np.asarray(times, dtype=float)
    m = np.zeros(times.shape, dtype=bool)
    for a, b in wins:
        m |= (times >= a) & (times <= b)
    return m


# ------------------------------------------------------------
#                 events
# ------------------------------------------------------------
def _entries(inside, t, min_gap_s=2.0, max_step_s=0.5):
    """Times where a boolean 'inside the region' trace turns True, debounced:
    a new entry only counts if the previous one was > min_gap_s earlier, and a
    rising edge that jumps across a tracking gap (> max_step_s between samples)
    is refused — the animal 'appearing' inside after a gap is not an entry."""
    rise = inside[1:] & ~inside[:-1] & (np.diff(t) <= max_step_s)
    idx = np.where(rise)[0] + 1
    out = []
    for i in idx:
        if not out or t[i] - out[-1] >= min_gap_s:
            out.append(float(t[i]))
    return out


def build_events(x, y, t, trials, nodes):
    """{event_type: {split: [times...]}} for start / goal arrival / bridge entry.

    goal arrival: entries into the goal node's region (radius = the node-assignment
    radius used everywhere: 0.45 x median node spacing). In goal-running trials only
    the FIRST arrival of each trial counts (the run's end); in free-roaming every
    debounced entry counts. 'all' pools both.
    bridge entry: stepping onto one of the LONG INTER-ISLAND bridges (not the
    short honeycomb edges); split by trial window ('all' = the entire session)."""
    ev = {k: {s: [] for s in SPLITS} for k in ("start", "goal arrival", "bridge entry")}
    wins = split_windows(trials, t.min(), t.max())

    # --- starts ---
    for (tt, _g, _s, a, _b) in trials:
        if tt == 1:
            ev["start"]["goal running"].append(a)
        elif tt in (4, 5):
            ev["start"]["free roaming"].append(a)
        ev["start"]["all"].append(a)

    # --- goal arrivals ---
    # radius: the task controller ends a goal run when the rat is within ~25 px
    # (~0.19 m) of the goal, so the trial window can close BEFORE the animal gets
    # nearer than the node-assignment radius — use 0.20 m and search slightly past
    # the trial end; if still no entry, the trial END is the arrival (a type-1
    # trial ends BY reaching the goal, so t1 is the arrival by task definition).
    node_rad = 0.20
    ok_pos = np.isfinite(x) & np.isfinite(y)          # NaN = untracked, never "inside"
    for (tt, g, _s, a, b) in trials:
        ent = []
        if g in nodes:
            gx, gy = nodes[g]
            m = ok_pos & (t >= a) & (t <= b + (5.0 if tt == 1 else 0.0))
            if m.any():
                inside = np.hypot(x[m] - gx, y[m] - gy) <= node_rad
                ent = _entries(inside, t[m])
        # the fallback must run for EVERY type-1 trial (also when the goal node is
        # unknown or the window has no tracked samples): the trial ends BY reaching
        # the goal, so t1 is the arrival by task definition.
        if tt == 1:
            arr = ent[0] if ent else b
            ev["goal arrival"]["goal running"].append(arr)      # first arrival = run end
            ev["goal arrival"]["all"].append(arr)
        elif tt in (4, 5):
            ev["goal arrival"]["free roaming"] += ent
            ev["goal arrival"]["all"] += ent

    # --- bridge entries ---
    # "Bridge" = the LONG INTER-ISLAND corridors only (this maze: 4 of ~0.7 m),
    # NOT the short honeycomb edges within an island — counting every edge gave
    # ~10 events per goal run (~300/session), which is edge-crossing, not
    # bridge-crossing. NaN tracking gaps must be dropped BEFORE the hexmaze
    # assignment (a NaN sample lands on "bridge 0" and fabricates transitions),
    # and a transition must be contiguous in TIME (no tracking gap) and SPACE
    # (no single-frame teleport — the tracker holds a stale position while the
    # rat is carried out between trials).
    ok = np.isfinite(x) & np.isfinite(y)
    xf, yf, tf = x[ok], y[ok], t[ok]
    loc_node, loc_bridge = NU._hex_assign(xf, yf, nodes)
    segs = NU._bridges(nodes)
    seg_len = np.array([np.hypot(p1[0] - p2[0], p1[1] - p2[1]) for p1, p2 in segs])
    is_long = seg_len > 1.5 * float(np.median(seg_len))   # inter-island corridors
    on_bridge = (loc_bridge >= 0) & np.where(loc_bridge >= 0,
                                             is_long[np.clip(loc_bridge, 0, None)], False)
    was_off = np.zeros_like(on_bridge)
    was_off[1:] = ~on_bridge[:-1]                  # entry = first sample ON the bridge
    contig = np.ones_like(on_bridge)
    contig[1:] = (np.diff(tf) <= 0.5) & (np.hypot(np.diff(xf), np.diff(yf)) <= 0.25)
    idx = np.where(on_bridge & was_off & contig)[0]
    bt = []
    for i in idx:                                   # debounce: >=1 s apart
        ti = float(tf[i])
        if not bt or ti - bt[-1] >= 1.0:
            bt.append(ti)
    bt = np.asarray(bt)
    for s in ("goal running", "free roaming"):
        ev["bridge entry"][s] = bt[_in_windows(bt, wins[s])].tolist()
    ev["bridge entry"]["all"] = bt.tolist()

    for k in ev:
        for s in ev[k]:
            ev[k][s] = np.sort(np.asarray(ev[k][s], dtype=float))
    return ev


# ------------------------------------------------------------
#                 theta (LFP -> phase)
# ------------------------------------------------------------
# The LFP can come from two places:
#   1. the session NWB (acquisition/lfp) — where step 8 writes it now;
#   2. LFP_Output/*.npy — older sessions, or step 8 run with KEEP_NPY=1.
# Both are opened into the same small "LFP source" dict, so load_theta() does
# not need to care which one it got:
#   data         (n_samples, n_channels), sliced like data[rows, cols]; never
#                loaded whole (an h5py dataset or a numpy memmap)
#   fs           sampling rate in Hz
#   boundaries   [{'name': recording name, 'start': first sample}, ...], one per
#                concatenated recording, or None when unknown
#   channel_map  [{'ntrode': .., 'channel': ..}, ...] one per column, or None
#   name         session name starting with the rat ('Rat5_...'), to find the
#                rat's settings in the tracker config
#   read_column  function(col) -> that whole column as float64
#   close        function() that releases the file
#   origin       short text for the log
def _lfp_source_from_nwb(op_folder):
    """The LFP stored in the session NWB as an LFP source, or None if absent."""
    from pynwb import NWBHDF5IO

    nwb_path = snwb.find_session_nwb(op_folder)
    if nwb_path is None:
        return None

    io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True)
    try:
        nwbfile = io.read()
        lfp_series = nwbfile.acquisition.get(snwb.LFP_NAME)
        if lfp_series is None:
            io.close()
            return None

        # Works whether the series stores a sampling rate or a timestamps array.
        fs = snwb._series_rate(lfp_series)
        if not fs:
            io.close()
            return None

        # session_info is a small JSON record next to the LFP. It holds the
        # channel map and the sample where each concatenated recording starts.
        sleep_module = snwb._sleep_module(nwbfile)
        session_info = snwb._read_json(sleep_module, snwb.SESSION_INFO_NAME) or {}
    except Exception:
        io.close()
        raise

    boundaries = None
    if session_info.get("session_boundaries"):
        boundaries = []
        for b in session_info["session_boundaries"]:
            boundaries.append({"name": b.get("name", ""),
                               "start": int(b.get("start_sample", 0))})

    data = lfp_series.data      # h5py dataset: only the slices we ask for are read

    def read_column(col):
        # The LFP is stored in chunks of 50 s x 4 channels, so reading one
        # column decompresses only ~1/32 of the file.
        return np.asarray(data[:, col], dtype=np.float64)

    return {"data": data, "fs": float(fs), "boundaries": boundaries,
            "channel_map": session_info.get("channel_map"),
            "name": nwb_path.stem,
            "read_column": read_column, "close": io.close,
            "origin": nwb_path.name}


STANDARD_LFP_RATES = (1000.0, 1250.0, 1500.0, 2000.0, 2500.0)


def _lfp_rate_problem(source, t):
    """Why this LFP source cannot be the task session's LFP, or None if it can.

    Some exports stamp a ~1 kHz LFP with 30 kHz timestamps (the trace then lasts
    1/30 of the session), and old NWBs hold a stub paired with the 5 Hz EMG
    timestamps. A maze-only export lasts about as long as the tracking (``t``, on
    the spike clock); a multi-session export is longer by design and only its
    rate is checked."""
    fs, n = source["fs"], source["data"].shape[0]
    if not fs or fs < 100:
        return f"{fs or 0:.0f} Hz is not an LFP rate"
    expected = float(np.nanmax(t)) if t is not None and len(t) else 0.0
    if source["boundaries"] or expected <= 0:
        return None
    # the duration matches, or a standard rate and the ephys merely ran longer or
    # shorter than the camera (both are started by hand)
    ratio = n / fs / expected
    standard = any(abs(fs - r) <= 0.01 * r for r in STANDARD_LFP_RATES)
    if 0.8 <= ratio <= 1.25 or (standard and 0.5 <= ratio <= 2.0):
        return None
    return f"{n} samples at {fs:.0f} Hz last {n / fs:.0f} s, the session ~{expected:.0f} s"


def _lfp_source_from_npy(op_folder):
    """The legacy LFP_Output/*.npy export as an LFP source, or None if absent."""
    lfp_dir, pfx = _find_npy_lfp(op_folder)
    if lfp_dir is None:
        return None

    data = np.load(lfp_dir / f"{pfx}lfp_data.npy", mmap_mode="r")
    timestamps = np.load(lfp_dir / f"{pfx}lfp_timestamps.npy", mmap_mode="r")
    fs = 1.0 / float(np.median(np.diff(np.asarray(timestamps[:10000]))))

    boundaries = None
    boundaries_file = lfp_dir / f"{pfx}session_boundaries.npy"
    if boundaries_file.exists():
        boundaries = list(np.load(boundaries_file, allow_pickle=True))

    channel_map = None
    channel_map_file = lfp_dir / f"{pfx}channel_map.npy"
    if channel_map_file.exists():
        channel_map = list(np.load(channel_map_file, allow_pickle=True))

    def read_column(col):
        # Fast path: KEEP_NPY also wrote one small file per channel.
        try:
            nt = int(channel_map[col]["ntrode"])
            ch = int(channel_map[col]["channel"])
            one_channel_file = lfp_dir / "channels_npy" / f"{pfx}lfp_nt{nt:02d}_ch{ch:02d}.npy"
            if one_channel_file.exists():
                return np.load(one_channel_file).astype(np.float64)
        except (TypeError, KeyError, ValueError, IndexError):
            pass                     # no usable channel map: use the slow path
        # Slow path: one column of the big memmap (slow on a network drive).
        return np.asarray(data[:, col], dtype=np.float64)

    return {"data": data, "fs": fs, "boundaries": boundaries,
            "channel_map": channel_map, "name": pfx, "read_column": read_column,
            "close": lambda: None, "origin": "LFP_Output"}


def _rat_config(source, config_path):
    """This session's rat in the tracker config (hm_tracker_paths.txt):
    ``(bad hardware channel ids, {"cortex"/"sr"/"pyr": tetrode})``.

    Both are empty when there is no config, no entry for the rat, or the
    config cannot be read.
    """
    if not config_path:
        return [], {}
    try:
        # The config reader lives with the sorter. Append (never insert at 0):
        # src/nwb's own modules, e.g. session_prefix, must keep priority.
        sorter_dir = str(Path(__file__).resolve().parent.parent / "sorter")
        if sorter_dir not in sys.path:
            sys.path.append(sorter_dir)
        from sorting import (load_sorting_config, resolve_rat_settings,
                             resolve_sleep_channels)
        config = load_sorting_config(config_path, quiet=True)
        bad_hw, _ref, _eeg = resolve_rat_settings(source["name"], config)
        sleep_channels = resolve_sleep_channels(source["name"], config)
    except Exception as e:
        print(f"  [theta] the tracker config could not be read ({e}); no bad "
              f"channels are skipped and no pyr tetrode is known.")
        return [], {}
    return list(bad_hw), dict(sleep_channels or {})


def _pyr_column(source, sleep_channels, bad_hw):
    """LFP column of the rat's pyr tetrode (SLEEP_CHANNELS_<RAT>), or None.

    The same channel step 8 computes the theta/delta ratio on: both use
    sorting.sleep_role_column. None (with a note in the log) when the config
    names no pyr tetrode, the LFP has no channel map, or the tetrode has no
    usable channel in this LFP; the caller then picks a channel itself.
    """
    pyr_tetrode = sleep_channels.get("pyr")
    if pyr_tetrode is None:
        print("  [theta] no pyr tetrode in SLEEP_CHANNELS_<RAT>; using the "
              "channel with the highest theta/delta instead.")
        return None
    if not source["channel_map"]:
        print("  [theta] no channel map: the pyr tetrode cannot be found in "
              "the LFP; using the channel with the highest theta/delta instead.")
        return None

    # _rat_config already put src/sorter on sys.path (sleep_channels is only
    # non-empty when it read the config), so this import works.
    from sorting import sleep_role_column
    column = sleep_role_column(source["channel_map"], sleep_channels, "pyr", bad_hw)
    if column is None:
        print(f"  [theta] pyr tetrode NT{pyr_tetrode} has no usable channel in "
              f"this LFP (missing, or all bad); using the channel with the "
              f"highest theta/delta instead.")
    return column


def _bad_columns(source, bad_hw):
    """LFP columns of the rat's BAD_CHANNELS_<rat> (``bad_hw``, hardware ids
    from _rat_config), so a known-bad channel is never picked as the theta
    channel.

    Empty when the rat has no bad channels or the LFP has no channel map.
    """
    if not bad_hw:
        return set()
    if not source["channel_map"]:
        print("  [theta] no channel map: the bad channels cannot be found in "
              "the LFP, so none are skipped.")
        return set()

    # The config holds hardware channel ids, as the sorter uses them:
    # hardware id = (nTrode - 1) * 4 + (channel - 1).
    bad_hw = {int(c) for c in bad_hw}
    cols = set()
    for col, entry in enumerate(source["channel_map"]):
        try:
            hw = (int(entry["ntrode"]) - 1) * 4 + (int(entry["channel"]) - 1)
        except (TypeError, KeyError, ValueError):
            continue
        if hw in bad_hw:
            cols.add(col)
    if cols:
        labels = [_channel_label(source["channel_map"], c) for c in sorted(cols)]
        print(f"  [theta] skipping the rat's bad channels {labels}")
    return cols


def _channel_label(channel_map, col):
    """'nt05ch02' for an LFP column, or 'col17' when there is no channel map."""
    try:
        nt = int(channel_map[col]["ntrode"])
        ch = int(channel_map[col]["channel"])
        return f"nt{nt:02d}ch{ch:02d}"
    except (TypeError, KeyError, ValueError, IndexError):
        return f"col{col}"


def _find_npy_lfp(op_folder):
    """(lfp_dir, prefix) of the Trodes LFP export, or (None, None)."""
    lo = Path(op_folder) / "LFP_Output"
    if not lo.is_dir():
        return None, None
    # NB: the folder also holds ..._emg_from_lfp_timestamps.npy — a bare
    # *lfp_timestamps.npy glob would grab that and build a bogus prefix.
    hits = [p for p in lo.glob("*lfp_timestamps.npy")
            if not p.name.startswith("._") and "emg" not in p.name.lower()]
    pfxs = [p.name[:-len("lfp_timestamps.npy")] for p in hits]
    pfxs = [pf for pf in pfxs if (lo / f"{pf}lfp_data.npy").exists()]
    if not pfxs:
        return None, None
    return lo, pfxs[0]


def load_theta(op_folder, x, y, t, band=THETA_BAND, sel_block_s=300.0, config=None):
    """Session theta phase from the LFP, on the spike clock.

    The LFP is read from the session NWB (acquisition/lfp, written by step 8),
    or from the legacy LFP_Output/*.npy export when the NWB has none.

    Channel = the rat's pyr tetrode from SLEEP_CHANNELS_<RAT> in `config`
    (hm_tracker_paths.txt): its lowest channel that is not in BAD_CHANNELS, the
    same channel step 8 computes theta/delta on. Without one, the column with
    the highest theta/delta Welch-power ratio (snwb.theta_delta_ratio:
    6-12 / 1-4 Hz) inside a movement-rich `sel_block_s` block (found from the
    tracking), never a BAD_CHANNELS one. `band` is the phase
    band-pass only. The maze session's
    sample 0 is spike-time 0 (session_boundaries; maze-only exports start at 0).
    Reads ONE contiguous row-block for selection, then only the chosen column
    for the full trace. Returns dict or None:
      {t0, fs, lt (float32 times), uw (float32 unwrapped phase), label}"""
    # 1. Find the LFP: the NWB first, the old .npy export as a fallback.
    try:
        source = _lfp_source_from_nwb(op_folder)
    except Exception as e:
        print(f"  [theta] could not read the LFP from the NWB ({e}); "
              f"trying LFP_Output instead.")
        source = None
    if source is not None:
        why = _lfp_rate_problem(source, t)
        if why:
            print(f"  [theta] the NWB's LFP is not usable ({why}); trying LFP_Output instead.")
            source["close"]()
            source = None
    if source is None:
        source = _lfp_source_from_npy(op_folder)
        why = _lfp_rate_problem(source, t) if source is not None else None
        if why:
            n, expected = source["data"].shape[0], float(np.nanmax(t))
            fit = min(STANDARD_LFP_RATES, key=lambda r: abs(n / r - expected))
            if not source["boundaries"] and abs(n / fit - expected) < 0.2 * expected:
                print(f"  [theta] LFP_Output: {why}; using {fit:.0f} Hz, the standard rate "
                      f"whose duration fits.")
                source["fs"] = fit
            else:
                print(f"  [theta] LFP_Output is not usable either ({why}) — phase/precession "
                      f"pages will be skipped.")
                return None
    if source is None:
        print("  [theta] no LFP (no acquisition/lfp in the NWB and no LFP_Output) "
              "— phase/precession pages will be skipped.")
        return None

    # 2. Compute the theta phase. The file is closed again whatever happens;
    #    everything returned is an in-memory copy.
    try:
        bad_hw, sleep_channels = _rat_config(source, config)
        skip = _bad_columns(source, bad_hw)
        pyr_column = _pyr_column(source, sleep_channels, bad_hw)
        return _theta_from_lfp_source(source, x, y, t, band, sel_block_s, skip,
                                      column=pyr_column)
    except Exception as e:
        print(f"  [theta] failed to load LFP ({e}) — phase pages skipped.")
        return None
    finally:
        source["close"]()


def _theta_from_lfp_source(source, x, y, t, band, sel_block_s, skip=(), column=None):
    """The work behind load_theta, on an LFP source dict (see above).
    ``column``: the theta channel (the pyr tetrode's) when known; None picks
    the channel with the highest theta/delta in a movement-rich block.
    ``skip``: LFP columns that may not be picked that way (bad channels)."""
    ld = source["data"]
    fs = source["fs"]

    # spike-clock offset: the TASK session's start sample (0 for maze-only
    # exports). Task recordings are named maze/mazs/awake/hab (same set
    # scan_drive._classify_phase maps to 'task') — matching only "maze" would
    # silently leave off=0 on a pre+awake+post export and align every spike
    # to PRE-SLEEP LFP. If a multi-session export has no task boundary we
    # cannot align at all: fail loudly rather than compute phases on noise.
    off = 0.0
    sb = source["boundaries"]
    if sb:                             # None: maze-only export, off=0 is correct
        task_keys = ("maze", "mazs", "awake", "hab")
        task = [b for b in sb
                if any(k in str(b.get("name", "")).lower() for k in task_keys)]
        if task:
            off = float(task[0]["start"]) / fs
        elif len(sb) > 1:
            print(f"  [theta] no task-phase boundary among "
                  f"{[str(b.get('name', '')) for b in sb]} — cannot align the "
                  f"LFP to the spike clock; phase pages skipped.")
            return None

    # movement-rich selection block: highest mean speed over candidate starts
    sp = np.zeros_like(x)
    d = np.hypot(np.diff(x), np.diff(y)); dts = np.diff(t)
    sp[1:] = d / np.where(dts > 0, dts, np.inf)
    dur = len(ld) / fs
    cand = np.arange(0.0, max(dur - sel_block_s, 1.0), sel_block_s / 2)
    best_t0, best_v = 0.0, -1.0
    for c in cand:
        # c is an LFP-FILE time; file time F = spike/position time F - off
        # (the same mapping as lt below), so score the tracking at c - off.
        m = (t >= c - off) & (t <= c - off + sel_block_s)
        vv = sp[m]; vv = vv[np.isfinite(vv)]
        v = float(vv.mean()) if vv.size else 0.0
        if v > best_v:
            best_v, best_t0 = v, c
    s0 = int(best_t0 * fs); s1 = min(int((best_t0 + sel_block_s) * fs), len(ld))

    if column is not None:
        # The theta channel is given (the pyr tetrode's): read only its trace.
        # Its theta/delta in the movement block is computed for the log only,
        # so a pyr channel with weak theta shows up there.
        best = int(column)
        how = "pyr tetrode"
        sig = source["read_column"](best)
        f, P = welch(sig[s0:s1], fs=fs, nperseg=4096)
        best_ratio = snwb.theta_delta_ratio(f, P)
    else:
        # No pyr channel: pick the channel with the highest theta/delta ratio
        # in the movement block, then read the full trace of that one only.
        block = np.asarray(ld[s0:s1, :], dtype=np.float64)   # one contiguous read
        if len(skip) >= block.shape[1]:
            skip = ()                  # nothing would be left: allow every channel
        ratios = np.full(block.shape[1], -np.inf)   # -inf = never chosen
        for chn in range(block.shape[1]):
            if chn in skip:
                continue               # a bad channel is never the theta channel
            f, P = welch(block[:, chn], fs=fs, nperseg=4096)
            ratios[chn] = snwb.theta_delta_ratio(f, P)   # same bands as step 8
        best = int(np.argmax(ratios))
        best_ratio = float(ratios[best])
        how = "highest theta/delta"
        sig = source["read_column"](best)

    label = _channel_label(source["channel_map"], best)

    b, a = butter(3, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    thf = filtfilt(b, a, sig)
    uw = np.unwrap(np.angle(hilbert(thf)))
    lt = (np.arange(len(sig)) / fs - off)
    print(f"  [theta] channel {label} ({how}, theta/delta {best_ratio:.2f}) from "
          f"{source['origin']}, {fs:g} Hz, {len(sig) / fs:.0f}s, offset {off:g}s.")
    return {"fs": fs, "lt": lt.astype(np.float32), "uw": uw.astype(np.float32),
            "sig": sig.astype(np.float32),          # raw channel, for gamma bands
            "t0": float(lt[0]), "t1": float(lt[-1]), "label": label}


def spike_phases(theta, st):
    """(phase_rad 0..2pi, cycle_index) of each spike; spikes outside the LFP are
    dropped. Interpolates the UNWRAPPED phase (monotonic) then re-wraps."""
    st = np.asarray(st, dtype=float)
    st = st[(st >= theta["t0"]) & (st <= theta["t1"])]
    u = np.interp(st, theta["lt"], theta["uw"])
    return np.mod(u, 2 * np.pi), np.floor(u / (2 * np.pi)).astype(np.int64), st


# ------------------------------------------------------------
#                 circular statistics
# ------------------------------------------------------------
def rayleigh(ph):
    """(R, p, mu_deg): mean resultant length, Rayleigh p (Zar approximation),
    preferred phase in degrees 0..360."""
    n = len(ph)
    if n < 2:
        return np.nan, np.nan, np.nan
    C = np.exp(1j * ph).mean()
    R = float(np.abs(C)); mu = float(np.degrees(np.angle(C)) % 360.0)
    z = n * R * R
    p = np.exp(-z) * (1 + (2 * z - z * z) / (4 * n))
    return R, float(np.clip(p, 0, 1)), mu


def circ_lin_fit(lin, ph, max_slope=np.pi, n_grid=721):
    """Circular-linear regression phase ~ a*lin: the slope a (rad per unit of
    `lin`) maximising the resultant of (ph - a*lin), plus the circular-circular
    correlation rho (Jammalamadaka) between ph and the fitted a*lin, with its
    asymptotic p. Returns (slope_rad, rho, p)."""
    lin = np.asarray(lin, float); ph = np.asarray(ph, float)
    if len(ph) < 5 or np.ptp(lin) == 0:
        return np.nan, np.nan, np.nan
    slopes = np.linspace(-max_slope, max_slope, n_grid)
    # chunk the slope grid: the full (n_grid, n_spike) broadcast is GBs for a
    # 200k-spike interneuron; 32-slope blocks keep it ~100 MB.
    eph = np.exp(1j * ph)
    Rres = np.empty(n_grid)
    for i in range(0, n_grid, 32):
        blk = slopes[i:i + 32]
        Rres[i:i + 32] = np.abs((eph[None, :] * np.exp(-1j * blk[:, None] * lin[None, :])).mean(1))
    k = int(np.argmax(Rres))
    if k in (0, n_grid - 1):
        # with integer regressors R(a) is 2pi-periodic, so -pi and +pi are the SAME
        # model and tie exactly; argmax would always report -pi, deterministically
        # biasing the slope sign. Break the tie toward the side the peak leans to.
        k = 0 if Rres[1] >= Rres[n_grid - 2] else n_grid - 1
    a = float(slopes[k])
    x_ = np.mod(a * lin, 2 * np.pi)
    sx = np.sin(x_ - np.angle(np.exp(1j * x_).mean()))
    sy = np.sin(ph - np.angle(np.exp(1j * ph).mean()))
    den = np.sqrt((sx ** 2).sum() * (sy ** 2).sum())
    if den == 0:
        return a, np.nan, np.nan
    rho = float((sx * sy).sum() / den)
    n = len(ph)
    l20 = (sx ** 2).mean(); l02 = (sy ** 2).mean(); l22 = ((sx * sy) ** 2).mean()
    if l22 <= 0:
        return a, rho, np.nan
    z = rho * np.sqrt(n * l20 * l02 / l22)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return a, rho, float(p)


# ------------------------------------------------------------
#                 page renderers
# ------------------------------------------------------------
def _peth_rate(st, ev_times, rec=None):
    """(edges, coverage-corrected rate) of `st` around the events. `rec` = (t0, t1)
    of the spike record: bins outside it around an edge event carry no data, so
    each bin is normalised by the number of events actually COVERING it (else
    events near the record edges dilute the rate toward 0)."""
    edges = np.arange(-PETH_WIN, PETH_WIN + PETH_BIN, PETH_BIN)
    counts = np.zeros(len(edges) - 1)
    cover = np.zeros(len(edges) - 1)
    for e in ev_times:
        rel = st[(st >= e - PETH_WIN) & (st <= e + PETH_WIN)] - e
        counts += np.histogram(rel, bins=edges)[0]
        if rec is not None:
            cover += (edges[:-1] >= rec[0] - e) & (edges[1:] <= rec[1] - e)
        else:
            cover += 1
    rate = np.where(cover > 0, counts / (np.maximum(cover, 1) * PETH_BIN), np.nan)
    return edges, rate


def _peth_panel(axr, axh, st, ev_times, split, rec=None, ymax=None):
    """Raster (axr) + rate histogram (axh) of `st` around each event time. `ymax`
    puts every PETH panel of the unit on the SAME rate scale."""
    col = _SPLIT_COL[split]
    ne = len(ev_times)
    axr.set_title(f"{split}  (n={ne} events)", fontsize=9)
    if ne == 0:
        axr.text(0.5, 0.5, "no events", ha="center", va="center", transform=axr.transAxes)
        axr.set_xticks([]); axr.set_yticks([])
        axh.set_xticks([]); axh.set_yticks([])
        return
    # classic raster: true vertical tick segments, one row per event
    rels = [st[(st >= e - PETH_WIN) & (st <= e + PETH_WIN)] - e for e in ev_times]
    axr.eventplot(rels, lineoffsets=np.arange(ne), linelengths=0.85,
                  colors=[col], linewidths=0.7)
    edges, rate = _peth_rate(st, ev_times, rec)
    axr.set_xlim(-PETH_WIN, PETH_WIN); axr.set_ylim(-0.5, ne - 0.5)
    axr.axvline(0, color="crimson", lw=0.8)
    axr.set_ylabel("event #", fontsize=7); axr.tick_params(labelsize=6)
    axh.bar(edges[:-1], rate, width=PETH_BIN, align="edge", color=col, alpha=0.85)
    axh.axvline(0, color="crimson", lw=0.8)
    axh.set_xlim(-PETH_WIN, PETH_WIN)
    if ymax is not None and np.isfinite(ymax) and ymax > 0:
        axh.set_ylim(0, ymax)
    axh.set_xlabel("time from event (s)", fontsize=7)
    axh.set_ylabel("rate (Hz)", fontsize=7); axh.tick_params(labelsize=6)


def peth_pages(pdf, cid, st, events, rec=None):
    """One page per event type: 3 split columns of raster + PETH. Every PETH panel
    of the unit (all event types x all splits) shares ONE rate scale."""
    ymax = 0.0
    for by_split in events.values():
        for split in SPLITS:
            if len(by_split[split]):
                _e, r = _peth_rate(st, by_split[split], rec)
                if np.isfinite(r).any():
                    ymax = max(ymax, float(np.nanmax(r)))
    ymax = 1.05 * ymax if ymax > 0 else None
    for ev_name, by_split in events.items():
        fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
        for c, split in enumerate(SPLITS):
            _peth_panel(axes[0][c], axes[1][c], st, by_split[split], split,
                        rec=rec, ymax=ymax)
        fig.suptitle(f"Unit {cid} — spikes around {ev_name.upper()} "
                     f"(±{PETH_WIN:g}s, {PETH_BIN * 1e3:.0f}ms bins; shared rate scale)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)


def phase_page(pdf, cid, st, theta, wins):
    """Theta phase-coupling page: per split a double-plotted phase histogram +
    R / Rayleigh p / preferred phase. All three panels share ONE y scale."""
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 4.4))
    ph_all, _cyc, st_in = spike_phases(theta, st)
    edges = np.linspace(0, 360, 25)
    # pass 1: histograms for every split -> shared y limit
    hists = {}
    for split in SPLITS:
        ph = ph_all[_in_windows(st_in, wins[split])]
        if len(ph) >= 10:
            h = np.histogram(np.degrees(ph), bins=edges)[0].astype(float)
            hists[split] = (ph, h / h.sum())
    ymax = 1.1 * max((h.max() for _p, h in hists.values()), default=0.0)
    for ax, split in zip(axes, SPLITS):
        col = _SPLIT_COL[split]
        if split not in hists:
            n_ph = int(_in_windows(st_in, wins[split]).sum())
            ax.text(0.5, 0.5, f"{split}\n({n_ph} spikes)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ph, h = hists[split]
        R, p, mu = rayleigh(ph)
        ax.bar(np.r_[edges[:-1], edges[:-1] + 360], np.r_[h, h], width=15,
               align="edge", color=col, alpha=0.85)
        if ymax > 0:
            ax.set_ylim(0, ymax)
        ax.axvline(mu, color="crimson", lw=1); ax.axvline(mu + 360, color="crimson", lw=1)
        ax.set_xlim(0, 720)
        ax.set_xticks([0, 180, 360, 540, 720])
        ax.set_xlabel("theta phase (deg; 0=peak)", fontsize=8)
        ax.set_ylabel("fraction of spikes", fontsize=8)
        ax.set_title(f"{split}\nn={len(ph)}  R={R:.3f}  p={p:.2g}  pref={mu:.0f}°",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Unit {cid} — spike-theta phase coupling "
                 f"({THETA_BAND[0]:g}-{THETA_BAND[1]:g} Hz, ch {theta['label']})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)





# ------------------------------------------------------------
#          hexmaze linearization (graph distance to goal)
# ------------------------------------------------------------
def _graph_dist_to_goal(nodes, goal):
    """{node_id: shortest-path distance (m) to `goal` along the maze graph}, plus
    the segment list [(pa, pb, d_goal(a), d_goal(b), length)] for projection.
    Graph = NU._bridges segments (honeycomb edges + inter-island corridors)."""
    ids = list(nodes.keys())
    P = np.array([nodes[i] for i in ids], float)
    segs = NU._bridges(nodes)

    def _nid(pt):                       # endpoints come FROM P, so nearest = exact
        return int(np.argmin(np.hypot(P[:, 0] - pt[0], P[:, 1] - pt[1])))

    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import dijkstra
    n = len(ids)
    A = lil_matrix((n, n))
    epairs = []
    for p1, p2 in segs:
        i, j = _nid(p1), _nid(p2)
        w = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        A[i, j] = A[j, i] = w
        epairs.append((i, j, np.asarray(p1, float), np.asarray(p2, float), w))
    D = dijkstra(A.tocsr(), directed=False, indices=ids.index(goal))
    return ({ids[k]: float(D[k]) for k in range(n)},
            [(pa, pb, float(D[i]), float(D[j]), w) for (i, j, pa, pb, w) in epairs])


def linearize(x, y, t, nodes, goal, speed_thresh=0.05):
    """Project the tracking onto the maze graph: the 1-D coordinate d = graph
    distance to the goal (m) per FINITE sample, plus the classic direction/speed
    gate 'approach' (speed > speed_thresh AND d decreasing = running toward the
    goal). Every physical location has a FIXED d, so runs from different start
    nodes pool onto one axis. Returns {t, d, approach} or None."""
    if goal not in nodes:
        return None
    _, segs = _graph_dist_to_goal(nodes, goal)
    ok = np.isfinite(x) & np.isfinite(y)
    xf, yf, tf = x[ok], y[ok], t[ok]
    if xf.size < 10:
        return None
    best_perp = np.full(xf.shape, np.inf)
    d = np.full(xf.shape, np.inf)
    for (pa, pb, da, db, L) in segs:
        if not (np.isfinite(da) and np.isfinite(db)) or L <= 0:
            continue
        dxs, dys = pb[0] - pa[0], pb[1] - pa[1]
        u = np.clip(((xf - pa[0]) * dxs + (yf - pa[1]) * dys) / (L * L), 0.0, 1.0)
        perp = np.hypot(xf - (pa[0] + u * dxs), yf - (pa[1] + u * dys))
        lin = np.minimum(da + u * L, db + (1.0 - u) * L)
        better = perp < best_perp
        best_perp[better] = perp[better]
        d[better] = lin[better]
    sp = np.zeros_like(xf)
    dts = np.diff(tf)
    sp[1:] = np.hypot(np.diff(xf), np.diff(yf)) / np.where(dts > 0, dts, np.inf)
    dsm = gaussian_filter1d(np.where(np.isfinite(d), d, 0.0), 3)
    dd = np.zeros_like(d)
    dd[1:] = np.diff(dsm) / np.where(dts > 0, dts, np.inf)
    approach = (sp > speed_thresh) & (dd < 0) & np.isfinite(d)
    return {"t": tf, "d": d, "approach": approach, "x": xf, "y": yf}


# 1-D field conventions (classic precession protocol)
_LIN_BIN = 0.10          # m rate-map bin along the distance-to-goal axis
_FIELD_FRAC = 0.20       # field = contiguous region >= 20% of the unit's peak
_FIELD_MIN_PEAK = 1.0    # Hz in-field peak required
_FIELD_MIN_W = 0.30      # m minimum field width


def _lin_ratemap(lin, st, wins):
    """(centers, rate, (spike_d, spike_t)) on the distance-to-goal axis for one
    split: approach-gated occupancy and spikes only (speed + direction gate)."""
    m = lin["approach"] & _in_windows(lin["t"], wins)
    if m.sum() < 20:
        return None, None, None
    dt_pos = float(np.median(np.diff(lin["t"]))) or 1 / 30.
    dmax = float(np.max(lin["d"][m]))
    edges = np.arange(0.0, dmax + _LIN_BIN, _LIN_BIN)
    if len(edges) < 4:
        return None, None, None
    occ = np.histogram(lin["d"][m], bins=edges)[0].astype(float) * dt_pos
    st = np.asarray(st, float)
    st = st[(st >= lin["t"][0]) & (st <= lin["t"][-1])]
    si = np.searchsorted(lin["t"], st).clip(1, len(lin["t"]) - 1)
    near = np.where(np.abs(lin["t"][si] - st) < np.abs(lin["t"][si - 1] - st), si, si - 1)
    keep = m[near] & (np.abs(lin["t"][near] - st) <= 0.1)
    sd, stk = lin["d"][near[keep]], st[keep]
    spk = np.histogram(sd, bins=edges)[0].astype(float)
    occ_s = gaussian_filter1d(occ, 1.5)
    rate = np.where(occ_s > 1e-3, gaussian_filter1d(spk, 1.5) / occ_s, 0.0)
    rate[occ <= 0] = 0.0
    return edges[:-1] + _LIN_BIN / 2, rate, (sd, stk)


def _fields_1d(centers, rate):
    """[(d_lo, d_hi, peak_hz), ...]: contiguous regions >= _FIELD_FRAC * peak with
    an in-field peak >= _FIELD_MIN_PEAK and width >= _FIELD_MIN_W, strongest first."""
    r = np.nan_to_num(rate)
    if r.size == 0 or r.max() < _FIELD_MIN_PEAK:
        return []
    above = r >= _FIELD_FRAC * r.max()
    out, i = [], 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            if (j - i + 1) * _LIN_BIN >= _FIELD_MIN_W and r[i:j + 1].max() >= _FIELD_MIN_PEAK:
                out.append((float(centers[i] - _LIN_BIN / 2),
                            float(centers[j] + _LIN_BIN / 2), float(r[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    out.sort(key=lambda f: -f[2])
    return out


def _fields_2d(x, y, t, st, wins, speed_thresh=0.05, bin_m=0.05, sigma=2.0):
    """2-D place fields for one split: occupancy/spike maps accumulated over the
    split's windows (speed-gated, step-v conventions: 5 cm bins, sigma 2, fields
    >=50% peak via V.place_fields). Returns (field_masks, xedges, yedges)."""
    xmin, xmax, ymin, ymax = V.MAZE_EXTENT
    nx = max(5, int(round((xmax - xmin) / bin_m)))
    ny = max(5, int(round((ymax - ymin) / bin_m)))
    xe = np.linspace(xmin, xmax, nx + 1); ye = np.linspace(ymin, ymax, ny + 1)
    ok = np.isfinite(x) & np.isfinite(y) & _in_windows(t, wins)
    xs, ys, ts = x[ok], y[ok], t[ok]
    if xs.size < 50:
        return [], xe, ye
    sp = np.zeros_like(xs)
    dts = np.diff(ts)
    sp[1:] = np.hypot(np.diff(xs), np.diff(ys)) / np.where(dts > 0, dts, np.inf)
    mv = sp > speed_thresh
    dt_pos = float(np.median(dts[dts > 0])) if (dts > 0).any() else 1 / 30.
    occ = np.histogram2d(xs[mv], ys[mv], bins=[xe, ye])[0].T * dt_pos
    st = np.asarray(st, float)
    sm = _in_windows(st, wins)
    sxp = np.interp(st[sm], ts, xs, left=np.nan, right=np.nan)
    syp = np.interp(st[sm], ts, ys, left=np.nan, right=np.nan)
    svp = np.interp(st[sm], ts, sp, left=0.0, right=0.0)
    k = np.isfinite(sxp) & np.isfinite(syp) & (svp > speed_thresh)
    spk = np.histogram2d(sxp[k], syp[k], bins=[xe, ye])[0].T
    from scipy.ndimage import gaussian_filter
    visited = occ > 0
    occ_s = gaussian_filter(occ, sigma); spk_s = gaussian_filter(spk, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(occ_s > 0, spk_s / occ_s, 0.0)
    rate = np.ma.masked_where(~visited, rate)
    return V.place_fields(rate, field_frac=0.35), xe, ye


def _field_passes(lin, fmask, xe, ye):
    """Traversals of one 2-D field during APPROACH runs: contiguous in-field,
    approach-gated sample stretches with a net d-progress >= 0.10 m. Returns
    [(t_start, t_end, d_entry, d_exit), ...] (d decreasing: entry far, exit near)."""
    ix = np.clip(np.digitize(lin["x"], xe) - 1, 0, fmask.shape[1] - 1)
    iy = np.clip(np.digitize(lin["y"], ye) - 1, 0, fmask.shape[0] - 1)
    inf = fmask[iy, ix] & lin["approach"]
    passes = []
    i, n = 0, len(inf)
    while i < n:
        if inf[i]:
            j = i
            while j + 1 < n and (inf[j + 1] or (j + 2 < n and inf[j + 2]
                                 and lin["t"][j + 2] - lin["t"][j] <= 0.4)):
                j += 1
            d0, d1 = float(lin["d"][i]), float(lin["d"][j])
            if d0 - d1 >= 0.10 and lin["t"][j] > lin["t"][i]:
                passes.append((float(lin["t"][i]), float(lin["t"][j]), d0, d1))
            i = j + 1
        else:
            i += 1
    return passes


def precession_page(pdf, cid, st, theta, wins, lin, x, y, t):
    """CLASSIC phase precession: place fields are detected in 2-D (step-v
    conventions); every APPROACH pass through a field is normalised to
    position-in-field 0..1 (0 = entry, 1 = exit) via the linearised d coordinate,
    and ALL passes of ALL fields pool into ONE phase-vs-position plot per split
    with a single circular-linear fit. Negative slope = classic precession."""
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.0),
                             gridspec_kw={"height_ratios": [1, 1.8]})
    st = np.asarray(st, float)
    for c, split in enumerate(SPLITS):
        axm, axp = axes[0][c], axes[1][c]
        col = _SPLIT_COL[split]
        centers, rate1d, _spk = _lin_ratemap(lin, st, wins[split])
        fields, xe, ye = _fields_2d(x, y, t, st, wins[split])
        if centers is not None:
            axm.plot(centers, rate1d, color=col, lw=1.2)
            axm.invert_xaxis()
        axm.set_xlabel("distance to goal (m)", fontsize=7)
        axm.set_ylabel("rate (Hz)", fontsize=7)
        axm.tick_params(labelsize=6)
        axm.set_title(f"{split}: {len(fields)} 2-D field(s)", fontsize=9)
        axm.spines[["top", "right"]].set_visible(False)
        xs_all, ph_all, n_pass = [], [], 0
        for fmask in fields:
            for (t0p, t1p, d0, d1) in _field_passes(lin, fmask, xe, ye):
                sel = (st >= t0p) & (st <= t1p)
                if not sel.any():
                    continue
                d_sp = np.interp(st[sel], lin["t"], lin["d"])
                xs_all.append(np.clip((d0 - d_sp) / max(d0 - d1, 1e-9), 0, 1))
                u = np.interp(st[sel], theta["lt"], theta["uw"])
                ph_all.append(np.mod(u, 2 * np.pi))
                n_pass += 1
        if not xs_all or len(np.concatenate(ph_all)) < 15:
            n_sp = 0 if not ph_all else len(np.concatenate(ph_all))
            axp.text(0.5, 0.5, f"{split}\n{len(fields)} field(s), {n_pass} passes, "
                     f"{n_sp} spikes — too few", ha="center", va="center",
                     transform=axp.transAxes, fontsize=8)
            axp.set_xticks([]); axp.set_yticks([])
            continue
        xs = np.concatenate(xs_all); ph = np.concatenate(ph_all)
        a, rho, p = circ_lin_fit(xs, ph, max_slope=3 * np.pi, n_grid=1081)
        deg = np.degrees(ph)
        axp.scatter(np.r_[xs, xs], np.r_[deg, deg + 360], s=4, alpha=0.3,
                    color=col, linewidths=0, rasterized=True)
        if np.isfinite(a):
            mu0 = np.degrees(np.angle(np.exp(1j * (ph - a * xs)).mean())) % 360
            gx = np.linspace(0, 1, 20)
            axp.plot(gx, mu0 + np.degrees(a) * gx, color="crimson", lw=1.5)
        axp.set_xlim(0, 1); axp.set_ylim(0, 720)
        axp.set_yticks([0, 180, 360, 540, 720])
        axp.set_xlabel("normalised position in field (0=entry, 1=exit)", fontsize=7)
        axp.set_ylabel("theta phase (deg)", fontsize=7)
        axp.tick_params(labelsize=6)
        axp.set_title(f"{len(fields)} field(s), {n_pass} passes, n={len(ph)} spikes\n"
                      f"slope {np.degrees(a):+.0f}\u00b0/field  \u03c1={rho:.2f}  p={p:.2g}",
                      fontsize=8.5)
        axp.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Unit {cid} \u2014 classic phase precession (2-D fields \u226535% peak; "
                 f"approach passes through each field normalised 0\u21921 and pooled; "
                 f"1-D = graph distance to goal)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def _note_page(pdf, cid, msg):
    fig, ax = plt.subplots(figsize=(8.27, 3.5))
    ax.axis("off")
    ax.text(0.5, 0.5, f"Unit {cid}: {msg}", ha="center", va="center", fontsize=11)
    pdf.savefig(fig); plt.close(fig)


# ------------------------------------------------------------
#                 gamma: PAC, events, spike coupling
# ------------------------------------------------------------
GAMMA_BANDS = {"mid gamma": (50.0, 90.0), "high gamma": (90.0, 150.0)}
_BAND_COL = {"mid gamma": "#d95f02", "high gamma": "#7570b3"}
_PAC_NBINS = 18


def _count_events(env, m, lo, hi, min_dur, fs):
    """Number of gamma events inside mask `m`: contiguous stretches of env > lo
    (mask-gated, so a stretch never spans a window gap) lasting >= min_dur s whose
    peak exceeds hi."""
    a = (env > lo) & m
    if not a.any():
        return 0
    d = np.diff(a.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if a[0]:
        starts = np.r_[0, starts]
    if a[-1]:
        ends = np.r_[ends, len(a)]
    n = 0
    for s0, e0 in zip(starts, ends):
        if (e0 - s0) / fs >= min_dur and env[s0:e0].max() > hi:
            n += 1
    return n


def _gamma_prepare(theta, wins):
    """Session-level gamma quantities on the theta channel, computed ONCE:
      bands[name]["uw"]  unwrapped gamma phase (for per-unit spike coupling)
      pac[split][name]   (phase_bin_edges, normalised amp per theta-phase bin, Tort MI)
      ev[split][name]    (event_rate_hz, n_events, split_lfp_seconds)
    Gamma events: envelope > mean+2SD with peak > mean+3SD for >= 3 cycles of the
    band centre (thresholds from the WHOLE session, so splits are comparable)."""
    fs = theta["fs"]
    lt = theta["lt"]
    sig = np.asarray(theta["sig"], dtype=np.float64)
    nb = _PAC_NBINS
    edges = np.linspace(0, 2 * np.pi, nb + 1)
    th_bin = np.clip(np.digitize(np.mod(theta["uw"], 2 * np.pi), edges) - 1, 0, nb - 1)
    out = {"bands": {}, "pac": {s: {} for s in SPLITS}, "ev": {s: {} for s in SPLITS}}
    for name, (f1, f2) in GAMMA_BANDS.items():
        b, a = butter(3, [f1 / (fs / 2), f2 / (fs / 2)], btype="band")
        g = filtfilt(b, a, sig)
        an = hilbert(g)
        env = np.abs(an)
        out["bands"][name] = {"uw": np.unwrap(np.angle(an)).astype(np.float32)}
        mu, sd = float(env.mean()), float(env.std())
        lo, hi = mu + 2 * sd, mu + 3 * sd
        min_dur = 3.0 / ((f1 + f2) / 2.0)
        for split in SPLITS:
            m = _in_windows(lt, wins[split])
            if not m.any():
                out["pac"][split][name] = (edges, np.full(nb, 1.0 / nb), np.nan)
                out["ev"][split][name] = (np.nan, 0, 0.0)
                continue
            amp = np.bincount(th_bin[m], weights=env[m], minlength=nb)
            cnt = np.bincount(th_bin[m], minlength=nb).astype(float)
            mamp = amp / np.maximum(cnt, 1.0)
            tot = mamp.sum()
            p = mamp / tot if tot > 0 else np.full(nb, 1.0 / nb)
            H = -np.sum(p * np.log(p + 1e-12))
            mi = float((np.log(nb) - H) / np.log(nb))
            out["pac"][split][name] = (edges, p, mi)
            n_ev = _count_events(env, m, lo, hi, min_dur, fs)
            dur = float(m.sum()) / fs
            out["ev"][split][name] = (n_ev / max(dur, 1e-9), n_ev, dur)
    return out


def gamma_session_page(pdf, cid, gamma, theta):
    """SESSION-level gamma page (identical in every unit PDF, precomputed): per
    split, theta-phase vs normalised gamma amplitude (PAC, Tort MI) for mid and
    high gamma, and the gamma event rate."""
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 7.5),
                             gridspec_kw={"height_ratios": [1.6, 1]})
    pac_max = 0.0
    for split in SPLITS:
        for name in GAMMA_BANDS:
            pac_max = max(pac_max, float(np.max(gamma["pac"][split][name][1])))
    for c, split in enumerate(SPLITS):
        axp, axe = axes[0][c], axes[1][c]
        for name in GAMMA_BANDS:
            edges, p, mi = gamma["pac"][split][name]
            cen = np.degrees((edges[:-1] + edges[1:]) / 2)
            axp.step(np.r_[cen, cen + 360], np.r_[p, p], where="mid",
                     color=_BAND_COL[name], lw=1.4,
                     label=f"{name} MI={mi:.4f}" if np.isfinite(mi) else f"{name} (n/a)")
        axp.axhline(1.0 / _PAC_NBINS, color="0.6", lw=0.7, ls="--")
        axp.set_xlim(0, 720); axp.set_xticks([0, 180, 360, 540, 720])
        if pac_max > 0:
            axp.set_ylim(0, 1.15 * pac_max)
        axp.set_xlabel("theta phase (deg; 0=peak)", fontsize=7)
        axp.set_ylabel("normalised gamma amp", fontsize=7)
        axp.set_title(split, fontsize=9)
        axp.legend(fontsize=6.5, loc="upper right")
        axp.tick_params(labelsize=6)
        axp.spines[["top", "right"]].set_visible(False)
        names = list(GAMMA_BANDS)
        rates = [gamma["ev"][split][n][0] for n in names]
        nevs = [gamma["ev"][split][n][1] for n in names]
        bars = axe.bar(names, rates, color=[_BAND_COL[n] for n in names], alpha=0.85)
        for bar, nv in zip(bars, nevs):
            axe.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{nv}", ha="center", va="bottom", fontsize=7)
        axe.set_ylabel("gamma events / s", fontsize=7)
        axe.tick_params(labelsize=6.5)
        axe.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"SESSION theta-gamma coupling + gamma events (ch {theta['label']}; "
                 f"mid {GAMMA_BANDS['mid gamma'][0]:.0f}-{GAMMA_BANDS['mid gamma'][1]:.0f} Hz, "
                 f"high {GAMMA_BANDS['high gamma'][0]:.0f}-{GAMMA_BANDS['high gamma'][1]:.0f} Hz; "
                 f"events: env>2SD, peak>3SD, >=3 cycles)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def gamma_coupling_page(pdf, cid, st, theta, gamma, wins):
    """Per-unit spike-gamma phase coupling: rows = mid / high gamma, columns =
    splits; double-plotted phase histograms with R / Rayleigh p / preferred
    phase. All six panels share one y scale."""
    st = np.asarray(st, dtype=float)
    st2 = st[(st >= theta["t0"]) & (st <= theta["t1"])]
    edges = np.linspace(0, 360, 25)
    hists = {}
    for name in GAMMA_BANDS:
        u = np.interp(st2, theta["lt"], gamma["bands"][name]["uw"])
        ph_all = np.mod(u, 2 * np.pi)
        for split in SPLITS:
            ph = ph_all[_in_windows(st2, wins[split])]
            if len(ph) >= 10:
                h = np.histogram(np.degrees(ph), bins=edges)[0].astype(float)
                hists[(name, split)] = (ph, h / h.sum())
    ymax = 1.1 * max((h.max() for _p, h in hists.values()), default=0.0)
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 7.5))
    for r, name in enumerate(GAMMA_BANDS):
        for c, split in enumerate(SPLITS):
            ax = axes[r][c]
            if (name, split) not in hists:
                ax.text(0.5, 0.5, f"{name}\n{split}\n(too few spikes)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ph, h = hists[(name, split)]
            R, p, mu = rayleigh(ph)
            ax.bar(np.r_[edges[:-1], edges[:-1] + 360], np.r_[h, h], width=15,
                   align="edge", color=_BAND_COL[name], alpha=0.85)
            ax.axvline(mu, color="crimson", lw=1)
            ax.axvline(mu + 360, color="crimson", lw=1)
            if ymax > 0:
                ax.set_ylim(0, ymax)
            ax.set_xlim(0, 720); ax.set_xticks([0, 180, 360, 540, 720])
            ax.set_xlabel(f"{name} phase (deg; 0=peak)", fontsize=7)
            ax.set_ylabel("fraction of spikes", fontsize=7)
            ax.set_title(f"{split}\nn={len(ph)}  R={R:.3f}  p={p:.2g}  pref={mu:.0f}°",
                         fontsize=8.5)
            ax.tick_params(labelsize=6.5)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Unit {cid} — spike-gamma phase coupling (ch {theta['label']}; "
                 f"rows: mid 50-90 Hz / high 90-150 Hz)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig); plt.close(fig)


# ------------------------------------------------------------
#                 session-level orchestration
# ------------------------------------------------------------
def prepare(op_folder, x, y, t, trials, nodes, config=None):
    """Compute the session-wide inputs once: events, theta phase, split windows.
    Returns a bundle dict (theta may be None if the LFP export is absent)."""
    wins = split_windows(trials, t.min(), t.max())
    events = build_events(x, y, t, trials, nodes)
    n_ev = {k: {s: len(v) for s, v in d.items()} for k, d in events.items()}
    print(f"  [events] start {n_ev['start']}, goal {n_ev['goal arrival']}, "
          f"bridge {n_ev['bridge entry']}")
    theta = load_theta(op_folder, x, y, t, config=config)
    gamma = None
    if theta is not None:
        gamma = _gamma_prepare(theta, wins)
        mis = {s2: {n: round(gamma["pac"][s2][n][2], 4) for n in GAMMA_BANDS}
               for s2 in SPLITS}
        print(f"  [gamma] Tort MI {mis}")
    # spike-record extent for the PETH edge correction: the ephys (LFP) span when
    # known, else the tracking span (slightly generous — honest fallback).
    rec = (theta["t0"], theta["t1"]) if theta else (float(t.min()), float(t.max()))
    # linearised hexmaze (graph distance to the session's dominant goal node)
    gvals = [g for (_tt, g, _s, _a, _b) in trials if g is not None]
    goal = max(set(gvals), key=gvals.count) if gvals else None
    lin = linearize(x, y, t, nodes, goal) if goal is not None else None
    if lin is not None:
        print(f"  [linear] goal {goal}: d range 0..{np.max(lin['d'][np.isfinite(lin['d'])]):.1f} m, "
              f"{int(lin['approach'].sum())} approach samples.")
    else:
        print("  [linear] no goal node — classic precession page skipped.")
    return {"events": events, "theta": theta, "wins": wins, "rec": rec, "lin": lin,
            "gamma": gamma, "pos": (x, y, t)}


def unit_pages(pdf, cid, spike_times, bundle):
    """Append this unit's PETH / phase-coupling / precession pages to its PDF."""
    st = np.asarray(spike_times, dtype=float)
    peth_pages(pdf, cid, st, bundle["events"], rec=bundle.get("rec"))
    if bundle["theta"] is None:
        _note_page(pdf, cid, "no LFP export found — theta phase/precession skipped")
        return
    phase_page(pdf, cid, st, bundle["theta"], bundle["wins"])
    if bundle.get("gamma") is not None:
        gamma_coupling_page(pdf, cid, st, bundle["theta"], bundle["gamma"], bundle["wins"])
        gamma_session_page(pdf, cid, bundle["gamma"], bundle["theta"])
    if bundle.get("lin") is not None:
        precession_page(pdf, cid, st, bundle["theta"], bundle["wins"], bundle["lin"],
                        *bundle["pos"])
    else:
        _note_page(pdf, cid, "no goal node — classic precession skipped")
