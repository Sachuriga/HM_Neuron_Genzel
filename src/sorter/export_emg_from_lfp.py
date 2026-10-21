"""Step-8 companion — EMG-from-LFP (Buzsáki/Schomburg cross-channel correlation).

The method (see ``emg_from_lfp.py``) needs real 300–600 Hz content. With the LFP
now exported at 1500 Hz / ``-lfplowpass 700`` (Nyquist 750), that band survives in
the LFP itself — exactly how buzcode/anjal run it on 1250 Hz LFP. So the primary,
fast path reads the LFP export directly with numpy; no wideband/SpikeInterface.

PRIMARY — from the 1500 Hz LFP (session NWB, else ``<output>/LFP_Output/``):
    one channel per tetrode, leaving out the rat's BAD_CHANNELS / EEG_TETRODES
      -> band-pass 300–600 Hz -> cross-channel correlation EMG   (numpy/scipy)

FALLBACK — raw wideband via SpikeInterface (only when the LFP band is missing,
e.g. an old ``-lfplowpass 500`` export): read the ``.rec`` directly (or an old
``*.raw/*_group0.dat``), the same channel choice, band-pass 300–600, decimate,
correlate. (Runs in the SpikeInterface env, HM_neuron.)

Output (in ``<output>/LFP_Output/``):
    emg_from_lfp.npy            EMG at the LFP rate (upsampled), pipeline-aligned
    emg_from_lfp_5hz.npy        native EMG (default 5 Hz), smoothed
    emg_from_lfp_timestamps.npy time axis (s) for the 5 Hz signal
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se

try:
    from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
except ImportError:  # pragma: no cover - fallback when run as a module
    from .readTrodesExtractedDataFile3 import readTrodesExtractedDataFile

from emg_from_lfp import (emg_from_lfp, emg_correlation, _highfreq_passband,
                          _moving_average)
from session_prefix import session_prefix, find_output

RAW_FS = 30000.0        # raw acquisition rate (Hz)
GAIN = 0.195            # µV/bit (as in sorting.py); irrelevant to Pearson corr
CH_PER_TETRODE = 4
WORK_FS = 3000.0        # correlation rate; Nyquist 1500 keeps the 300–600 band.
                        # Chosen so 30 kHz -> WORK_FS is an integer decimation
                        # (factor 10), which lets us use the cheap spre.decimate.
EMG_FS = 5.0            # native EMG output rate (0.2 s bins), as in the reference
SMOOTH_S = 10.0         # EMG smoothing window (s)
N_PICK = 32             # channels fed to the correlation (spread across probe)


def find_raw_dats(input_folder):
    """Raw per-recording .dat files, chronological (Trodes name embeds datetime)."""
    base = Path(input_folder)
    return sorted(base.glob("**/*.raw/*_group0.dat"), key=lambda f: f.name)


def find_rec_files(input_folder):
    """Trodes .rec files, chronological. Read lazily via SpikeInterface, so no
    ``trodesexport -raw`` extraction is needed. (A ``<X>_merged.rec`` replaces
    its ``<X>.rec``.)"""
    from trodes_rec import find_rec_files as _find
    return _find(input_folder, recursive=True)


def spread_channels(n_channels, n_pick=N_PICK, skip=(), ch_per_tetrode=CH_PER_TETRODE):
    """One channel per tetrode, spread across the probe: the lowest-numbered
    channel of each tetrode that is not in ``skip`` (the rat's bad/EEG channels).

    Channel index = (tetrode)*4 + local. A tetrode whose lowest channel is in
    ``skip`` is represented by its next one; a tetrode with every channel in
    ``skip`` is left out. With more tetrodes than ``n_pick``, an even spread of
    ``n_pick`` of them is kept.
    """
    skip = set(int(c) for c in skip)
    picks = []
    for t in range(n_channels // ch_per_tetrode):
        good = [t * ch_per_tetrode + k for k in range(ch_per_tetrode)
                if t * ch_per_tetrode + k not in skip]
        if good:
            picks.append(good[0])
    if len(picks) > n_pick:
        keep = np.unique(np.linspace(0, len(picks) - 1, n_pick).round().astype(int))
        picks = [picks[i] for i in keep]
    return picks


# ──────────────────────────────────────────────────────────────────────────────
# PRIMARY PATH: EMG from the 1500 Hz LFP export (numpy only, no SpikeInterface)
# ──────────────────────────────────────────────────────────────────────────────

def _write_emg_to_nwb(lfp_dir, emg, timestamps, emg_fs):
    """Add the 5 Hz EMG-from-LFP to the session NWB (skipped when already there)."""
    nwb_path = _session_nwb(lfp_dir)
    if nwb_path is None:
        print("  ⚠  no session .nwb found — run the step-8 LFP export first; "
              "only the .npy EMG was written.")
        return None
    import sleep_nwb_writer as snwb
    from pynwb import NWBHDF5IO

    io = NWBHDF5IO(str(nwb_path), mode="r+")
    try:
        nwbfile = io.read()
        added = snwb.add_sleep_inputs(nwbfile, emg=emg, emg_timestamps=timestamps)
        if added:
            io.write(nwbfile)
            print(f"  ✓ {nwb_path.name}: acquisition/emg_from_lfp "
                  f"({emg.size} @ {emg_fs} Hz)")
        else:
            print(f"  = {nwb_path.name} already has acquisition/emg_from_lfp")
    finally:
        io.close()
    return nwb_path


def _session_nwb(lfp_dir):
    """The session NWB for an LFP_Output folder, or None."""
    # append, never insert(0): src/nwb has its own session_prefix.py that would
    # otherwise shadow this package's
    import sys
    nwb_dir = str(Path(__file__).resolve().parent.parent / "nwb")
    if nwb_dir not in sys.path:
        sys.path.append(nwb_dir)
    try:
        import sleep_nwb_writer as snwb
        return snwb.find_session_nwb(lfp_dir)
    except Exception as exc:
        print(f"   (could not look for a session NWB: {exc})")
        return None


def _open_lfp(lfp_dir):
    """Open the session LFP: returns ``(data, fs, close)``.

    ``data`` is ``(n_samples, n_channels)`` and sliced by column only, so the
    NWB path reads just the columns asked for straight out of HDF5 rather than
    loading the session. Prefers the NWB written by the LFP export and falls
    back to ``lfp_data.npy`` + ``lfp_timestamps.npy`` for older sessions.
    """
    nwb_path = _session_nwb(lfp_dir)
    if nwb_path is not None:
        from pynwb import NWBHDF5IO
        io = NWBHDF5IO(str(nwb_path), mode="r")
        try:
            series = io.read().get_acquisition("lfp")
        except Exception:
            io.close()
        else:
            fs = float(series.rate) if getattr(series, "rate", None) else None
            if fs is None and series.timestamps is not None:
                ts = np.asarray(series.timestamps[:1000]).ravel()
                dt = float(np.median(np.diff(ts))) if ts.size > 1 else 0.0
                fs = 1.0 / dt if dt > 0 else None
            if fs:
                print(f"   LFP from {nwb_path.name} ({series.data.shape} @ {fs:.0f} Hz)")
                return series.data, fs, io.close
            io.close()

    data_file = find_output(lfp_dir, "lfp_data.npy")       # prefixed or not
    if data_file is None:
        raise FileNotFoundError(
            "no LFP found: neither acquisition/lfp in the session .nwb nor "
            "lfp_data.npy in LFP_Output")
    ts_file = find_output(lfp_dir, "lfp_timestamps.npy")
    if ts_file is None:
        raise FileNotFoundError("no lfp_timestamps.npy in LFP_Output")
    ts = np.asarray(np.load(ts_file, mmap_mode="r")).ravel()
    span = float(ts[-1]) - float(ts[0]) if ts.size >= 2 else 0.0
    fs = (ts.size - 1) / span if span > 0 else None
    return np.load(data_file, mmap_mode="r"), fs, (lambda: None)


_INFO_CACHE = {}


def _session_info(lfp_dir):
    """``processing/sleep/session_info`` from the session NWB, else ``{}``.

    Holds the channel map, session boundaries and SNR scores that
    used to live in LFP_Output as small ``.npy``. Cached per folder.
    """
    key = str(lfp_dir)
    if key in _INFO_CACHE:
        return _INFO_CACHE[key]
    info = {}
    nwb_path = _session_nwb(lfp_dir)
    if nwb_path is not None:
        import sleep_nwb_writer as snwb
        inputs = None
        try:
            inputs = snwb.read_sleep_inputs(nwb_path)
            info = inputs.get("session_info") or {}
        except Exception as exc:
            print(f"   (could not read session_info from {nwb_path.name}: {exc})")
        finally:
            snwb.close_inputs(inputs)
    _INFO_CACHE[key] = info
    return info


def _hw_to_lfp_columns(lfp_dir, hw_channels):
    """Map hardware channel ids -> LFP column indices via the channel map.

    The channel map holds per-column {index, ntrode, channel, ...}; the hardware
    id of a column is (ntrode-1)*4 + (channel-1). Returns the columns present.
    Read from the session NWB, falling back to ``channel_map.npy``.
    """
    cmap = _session_info(lfp_dir).get("channel_map")
    if not cmap:
        cmap_file = find_output(lfp_dir, "channel_map.npy")   # prefixed or not
        if cmap_file is None:
            return None
        cmap = np.load(cmap_file, allow_pickle=True)
    hw_of_col = {}
    for entry in cmap:
        nt, ch = entry.get("ntrode"), entry.get("channel")
        if nt is not None and ch is not None:
            hw_of_col[(nt - 1) * CH_PER_TETRODE + (ch - 1)] = int(entry["index"])
    return [hw_of_col[c] for c in hw_channels if c in hw_of_col]


def _one_column_per_tetrode(lfp_dir, n_cols, skip=()):
    """One LFP column per nTrode: its lowest-numbered channel not in ``skip``.

    Channels of one tetrode sit within ~50 um of each other and carry nearly the
    same signal, so feeding all four to the cross-channel correlation would
    inflate it with near-duplicate pairs. ``skip`` holds LFP columns never to use
    (the rat's bad/EEG channels): a tetrode whose lowest channel is skipped is
    represented by its next one. An LFP that already holds one channel per
    nTrode (the exportLFP layout) comes back whole, minus ``skip``.
    """
    skip = set(int(c) for c in skip)
    cmap = _session_info(lfp_dir).get("channel_map")
    if not cmap:
        cmap_file = find_output(lfp_dir, "channel_map.npy")
        cmap = np.load(cmap_file, allow_pickle=True) if cmap_file else None
    if cmap is None or len(cmap) != n_cols:
        # no map: assume the 4-per-tetrode hardware order when it fits
        if n_cols >= 2 * CH_PER_TETRODE and n_cols % CH_PER_TETRODE == 0:
            return spread_channels(n_cols, n_pick=n_cols, skip=skip)
        return [c for c in range(n_cols) if c not in skip]
    first = {}
    for entry in cmap:
        nt, ch = entry.get("ntrode"), entry.get("channel")
        col = int(entry["index"])
        if nt is None or ch is None or col in skip:
            continue
        if nt not in first or ch < first[nt][0]:
            first[nt] = (ch, col)
    return (sorted(c for _ch, c in first.values())
            or [c for c in range(n_cols) if c not in skip])


READ_ROWS = 75_000      # rows per LFP read = one HDF5 chunk tall (50 s @ 1500 Hz)


def _read_columns(data, cols, rows=READ_ROWS):
    """``data[:, cols]`` as float64, read one block of rows at a time.

    Asking h5py for several columns at once (``data[:, cols]``) on the chunked,
    gzip-compressed NWB LFP is very slow: ~17x slower than this loop when
    measured, i.e. minutes for one session with nothing printed, so the run
    looks frozen. Here each block reads every column of a 50 s stretch (each
    chunk decompressed once) and numpy keeps the wanted columns.
    """
    n = data.shape[0]
    out = np.empty((n, len(cols)), dtype=np.float64)
    for s in tqdm(range(0, n, rows), desc="   Reading LFP", unit="block",
                  leave=False):
        out[s:s + rows] = np.asarray(data[s:s + rows])[:, cols]
    return out


SNR_FRAC = 0.3          # with --exclude_bad: drop LFP cols with SNR < SNR_FRAC * median
MIN_KEEP = 8            # fewer tetrodes than this left -> use every tetrode instead


def _low_snr_columns(lfp_dir, n_cols, snr_frac=SNR_FRAC):
    """LFP columns whose step-8 SNR score (``channel_snr_scores``) is below
    ``snr_frac * median``: probably noisy. Only used with ``--exclude_bad``."""
    scores = _session_info(lfp_dir).get("channel_snr_scores")
    snr_file = None if scores else find_output(lfp_dir, "channel_snr_scores.npy")
    if not scores and snr_file is None:
        return set()
    scores = np.asarray(scores if scores else np.load(snr_file)).ravel()
    finite = np.isfinite(scores)
    if scores.size != n_cols or not finite.any():
        return set()
    thr = snr_frac * float(np.median(scores[finite]))
    low = {int(c) for c in np.where(scores < thr)[0]}
    if low:
        print(f"   EMG: low-SNR columns (SNR < {thr:.0f}) left out: {sorted(low)}")
    return low


def emg_from_lfp_output(lfp_dir, emg_fs=EMG_FS, skip_hw=None, exclude_low_snr=False,
                        snr_frac=SNR_FRAC):
    """EMG-from-LFP computed directly from the 1500 Hz LFP export (numpy/scipy).

    The LFP already carries the 300–600 Hz band (exportLFP -lfplowpass 700), so no
    wideband/SpikeInterface read is needed — this is the fast primary path.

    Channels: one column per tetrode, spread across the probe (ideal for the
    cross-channel correlation; see :func:`_one_column_per_tetrode`). The
    hardware channels in ``skip_hw`` (the rat's bad and EEG channels) are never
    used, since a noisy channel dilutes every pairwise correlation it is part
    of. ``exclude_low_snr`` also leaves out low-SNR columns.
    """
    lfp_dir = Path(lfp_dir)
    data, fs, close = _open_lfp(lfp_dir)
    try:
        if fs is None:
            raise FileNotFoundError("could not determine the LFP sampling rate")
        if fs / 2.0 <= 600.0:
            raise ValueError(
                f"LFP fs={fs:.0f} Hz (Nyquist {fs/2:.0f}) is below the 300–600 Hz EMG "
                f"band. Re-export LFP at 1500 Hz / -lfplowpass 700.")

        # The columns never to use: the rat's bad/EEG channels (hardware ids,
        # found in the LFP through its channel map), plus low-SNR ones if asked.
        n_cols = data.shape[1]
        skip = set()
        if skip_hw:
            mapped = _hw_to_lfp_columns(lfp_dir, sorted(skip_hw))
            if mapped is None:
                print("   (no channel map: the bad/EEG channels cannot be found "
                      "in the LFP, so none are left out)")
            else:
                skip |= set(mapped)
        if exclude_low_snr:
            skip |= _low_snr_columns(lfp_dir, n_cols, snr_frac)

        cols = _one_column_per_tetrode(lfp_dir, n_cols, skip=skip)
        origin = f"{len(cols)} LFP channels (one per tetrode"
        origin += ", bad/EEG left out)" if skip else ")"
        if skip and len(cols) < MIN_KEEP:
            print(f"   ⚠ only {len(cols)} tetrodes left without the bad/EEG "
                  f"channels; using every tetrode instead")
            cols = _one_column_per_tetrode(lfp_dir, n_cols)
            origin = f"{len(cols)} LFP channels (one per tetrode)"

        cols = sorted(set(int(c) for c in cols))
        print(f"   Reading {len(cols)} LFP columns: {cols}")
        sel = _read_columns(data, cols)
        n_samples = data.shape[0]
    finally:
        close()

    print(f"   Band-pass + cross-channel correlation on {len(cols)} channels ...")
    res = emg_from_lfp(sel, fs, emg_fs=emg_fs, smooth_window_s=SMOOTH_S)
    res["duration_s"] = n_samples / fs
    res["picks"] = cols
    res["origin"] = origin
    res["fs"] = fs
    return res


# ──────────────────────────────────────────────────────────────────────────────
# FALLBACK PATH: EMG from the raw wideband (SpikeInterface), for LP-500 LFP
# ──────────────────────────────────────────────────────────────────────────────

def emg_from_recording(rec, work_fs=WORK_FS, emg_fs=EMG_FS, n_pick=N_PICK, skip=()):
    """Core: EMG-from-LFP for a SpikeInterface recording object (any source).

    Channels: up to ``n_pick``, one per tetrode spread across the probe, never
    one in ``skip`` (the rat's bad/EEG channels; see :func:`spread_channels`).
    Channel index == hardware id here (the raw export's column order).

    Band-passes to the EMG band, decimates/resamples to ``work_fs``, and
    correlates. Returns the ``emg_correlation`` dict plus ``duration_s``,
    ``picks`` and ``passband``.
    """
    n_channels = rec.get_num_channels()
    duration_s = rec.get_total_duration()
    src_fs = rec.get_sampling_frequency()

    ch_ids = rec.get_channel_ids()
    picks = spread_channels(n_channels, n_pick=n_pick, skip=skip)
    if len(picks) < 2:
        raise ValueError(f"Need >= 2 channels for the EMG; got {picks} "
                         f"(recording has {n_channels} channels).")
    rec_sel = rec.select_channels([ch_ids[i] for i in picks])

    lo, hi = _highfreq_passband(work_fs)                   # 300–600 Hz
    rec_bp = spre.bandpass_filter(rec_sel, freq_min=lo, freq_max=hi)

    # Reduce to work_fs. The band-pass already limits the signal to < hi Hz, so
    # when src_fs is an integer multiple of work_fs we can use the cheap
    # spre.decimate (a strided read, no resample margins -> far less network IO)
    # without aliasing (work_fs/2 stays above hi). Otherwise fall back to resample.
    factor = src_fs / work_fs
    if abs(round(factor) - factor) < 1e-9 and round(factor) >= 2:
        rec_ds = spre.decimate(rec_bp, int(round(factor)))
    else:
        rec_ds = spre.resample(rec_bp, int(round(work_fs)))

    # Materialise only the n_pick down-sampled channels (small), then correlate.
    # (Pearson correlation is scale-invariant, so raw/unscaled traces are fine.)
    traces = rec_ds.get_traces().astype(np.float64)
    res = emg_correlation(traces, rec_ds.get_sampling_frequency(),
                          emg_fs=emg_fs, smooth_window_s=SMOOTH_S)
    res["duration_s"] = duration_s
    res["picks"] = picks
    res["passband"] = (lo, hi)
    return res


def emg_for_recording(dat_file, work_fs=WORK_FS, emg_fs=EMG_FS, n_pick=N_PICK,
                      skip=()):
    """EMG-from-LFP for one extracted raw ``*.raw/*_group0.dat`` (30 kHz, 128 ch)."""
    raw = readTrodesExtractedDataFile(str(dat_file))
    voltage = raw["data"]["voltage"]                       # (n_samples, n_channels)
    rec = si.NumpyRecording(traces_list=[voltage], sampling_frequency=RAW_FS)
    rec.set_channel_gains(GAIN)
    rec.set_channel_offsets(0.0)
    return emg_from_recording(rec, work_fs=work_fs, emg_fs=emg_fs, n_pick=n_pick,
                              skip=skip)


def emg_for_rec_file(rec_file, work_fs=WORK_FS, emg_fs=EMG_FS, n_pick=N_PICK,
                     skip=()):
    """EMG-from-LFP for a Trodes ``.rec`` read lazily (no raw extraction needed).

    ``open_rec`` puts the channels in the raw export's order (neo alone mislabels
    them on our chanPerChip="0" headers), so bad/EEG channel ids mean the same
    here. Read without Trodes' rawRef, like step 8's LFP: a subtracted reference
    channel would put its own 300-600 Hz activity into every channel and so
    into the cross-channel correlation."""
    from trodes_rec import open_rec
    rec = open_rec(rec_file, apply_raw_ref=False)
    return emg_from_recording(rec, work_fs=work_fs, emg_fs=emg_fs, n_pick=n_pick,
                              skip=skip)


def _rat_skip_channels(stem, config, extra_eeg=None):
    """``{hardware channel: 'bad' | 'EEG'}`` to leave out of the EMG.

    From the rat's ``BAD_CHANNELS_<rat>`` and ``EEG_TETRODES_<rat>`` in the
    tracker config (matched on the recording ``stem``), plus ``extra_eeg``
    (``--eeg_channels``). Empty when the rat has no config entry.
    """
    bad, eeg = [], []
    if config:
        try:
            from sorting import resolve_rat_settings
            bad, _ref, eeg = resolve_rat_settings(stem, config)
        except Exception as exc:
            print(f"   (bad/EEG channels not read from the config: {exc})")
    skip = {int(c): "EEG" for c in list(eeg) + list(extra_eeg or [])}
    skip.update({int(c): "bad" for c in bad})
    if skip:
        print(f"   EMG leaves out bad channels "
              f"{sorted(c for c, why in skip.items() if why == 'bad')} and EEG "
              f"channels {sorted(c for c, why in skip.items() if why == 'EEG')}")
    return skip


def _session_stem(input_folder):
    """A recording stem for matching the rat config (from a .rec/.dat, else folder)."""
    recs = find_rec_files(input_folder) or find_raw_dats(input_folder)
    return recs[0].stem if recs else Path(input_folder).name


def _emg_from_raw(input_folder, work_fs, emg_fs, n_pick, skip_hw,
                  smoothed, timestamps, boundaries):
    """Fallback: EMG from raw wideband (.rec via SI, else an old extracted .dat)."""
    recs = find_rec_files(input_folder)
    sources = [(r, "rec") for r in recs] if recs else \
        [(d, "dat") for d in find_raw_dats(input_folder)]
    offset_s = 0.0
    for src, kind in sources:
        print(f"  • {src.name}")
        loader = emg_for_recording if kind == "dat" else emg_for_rec_file
        res = loader(src, work_fs=work_fs, emg_fs=emg_fs, n_pick=n_pick, skip=skip_hw)
        origin = f"{len(res['picks'])} ch spread (one per tetrode"
        origin += ", bad/EEG left out)" if skip_hw else ")"
        print(f"      {origin}, band {res['passband'][0]:.0f}-"
              f"{res['passband'][1]:.0f} Hz, {res['data'].size} EMG bins @ {emg_fs} Hz")
        smoothed.append(res["smoothed"])
        timestamps.append(res["timestamps"] + offset_s)
        boundaries.append({"name": src.stem, "start_s": offset_s,
                           "duration_s": res["duration_s"]})
        offset_s += res["duration_s"]


def run(input_folder, output_folder, work_fs=WORK_FS, emg_fs=EMG_FS, n_pick=N_PICK,
        config=None, eeg_channels=None, include_bad=False, exclude_low_snr=False,
        snr_frac=SNR_FRAC, keep_npy=False):
    # LFP_Output only exists for the .npy files; the NWB is the default home.
    output_dir = Path(output_folder) / "LFP_Output"
    if keep_npy:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"▶  EMG-from-LFP (Buzsáki cross-channel correlation)")
    print(f"   Input:  {input_folder}")
    print(f"   Output: {output_dir if keep_npy else 'the session NWB'}")

    stem = _session_stem(input_folder)
    # The rat's bad and EEG channels stay out of the correlation (unless
    # include_bad): a noisy channel dilutes every pair it is part of.
    skip_hw = {} if include_bad else _rat_skip_channels(stem, config, eeg_channels)

    smoothed, timestamps, boundaries = [], [], []
    # PRIMARY: straight from the 1500 Hz LFP export (fast, numpy only): one
    # channel per tetrode, never a bad/EEG one.
    try:
        res = emg_from_lfp_output(output_dir, emg_fs=emg_fs, skip_hw=skip_hw,
                                  exclude_low_snr=exclude_low_snr, snr_frac=snr_frac)
        print(f"   From 1500 Hz LFP export @ {res['fs']:.0f} Hz: {res['origin']}, "
              f"band {res['passband'][0]:.0f}-{res['passband'][1]:.0f} Hz, "
              f"{res['data'].size} EMG bins @ {emg_fs} Hz")
        smoothed = [res["smoothed"]]
        timestamps = [res["timestamps"]]
        boundaries = [{"name": "lfp", "start_s": 0.0, "duration_s": res["duration_s"]}]
    except (FileNotFoundError, ValueError) as exc:
        print(f"   LFP path unavailable ({exc})")
        print(f"   Falling back to the raw wideband via SpikeInterface.")
        _emg_from_raw(input_folder, work_fs, emg_fs, n_pick, skip_hw,
                      smoothed, timestamps, boundaries)
    if not smoothed:
        print("❌  No input found (LFP export, **/*.raw/*_group0.dat, or **/*.rec).")
        return

    emg_5hz = np.concatenate(smoothed).astype("float32")
    ts_5hz = np.concatenate(timestamps).astype("float64")
    # Min/max normalise across the whole (concatenated) session, as the reference does.
    rng = emg_5hz.max() - emg_5hz.min()
    emg_5hz_norm = ((emg_5hz - emg_5hz.min()) / (rng + 1e-12)).astype("float32")

    pfx = session_prefix(_session_stem(input_folder))    # rat_sessiondate_ prefix
    if keep_npy:
        np.save(output_dir / f"{pfx}emg_from_lfp_5hz.npy", emg_5hz_norm)
        np.save(output_dir / f"{pfx}emg_from_lfp_timestamps.npy", ts_5hz)
        print(f"  ✓ {pfx}emg_from_lfp_5hz.npy  ({emg_5hz_norm.size}) @ {emg_fs} Hz")

    # Into the session NWB, on its own 5 Hz timebase — this is what the sleep
    # scorer reads. The per-sample upsampled copy is derivable from it, so it is
    # no longer written for NWB-era sessions.
    nwb_path = _write_emg_to_nwb(output_dir, emg_5hz_norm, ts_5hz, emg_fs)

    # Legacy sessions (those still carrying lfp_timestamps.npy) also get the
    # upsampled emg_from_lfp.npy, so anything reading it keeps working.
    lfp_ts_file = find_output(output_dir, "lfp_timestamps.npy")
    if lfp_ts_file is not None:
        lfp_ts = np.load(lfp_ts_file).astype("float64")
        emg_up = np.interp(lfp_ts, ts_5hz, emg_5hz_norm).astype("float32")
        np.save(output_dir / f"{pfx}emg_from_lfp.npy", emg_up)
        print(f"  ✓ {pfx}emg_from_lfp.npy  ({emg_up.size}) upsampled to LFP time axis")

    if len(boundaries) > 1 and keep_npy:
        np.save(output_dir / f"{pfx}emg_from_lfp_boundaries.npy", boundaries)
        for b in boundaries:
            print(f"    {b['name']}: t0={b['start_s']:.1f}s  dur={b['duration_s']:.1f}s")

    print(f"{'=' * 60}")
    print(f"✅  EMG-from-LFP → {nwb_path.name if nwb_path is not None else output_dir}")


def _load_config(config_path):
    """Load the tracker config (hm_tracker_paths.txt) for EEG-channel resolution."""
    import os
    if not config_path:
        default = Path(os.path.expanduser("~")) / "Desktop" / "hm_tracker_paths.txt"
        config_path = str(default) if default.exists() else None
    if not config_path:
        return None
    try:
        from sorting import load_sorting_config
        return load_sorting_config(config_path, quiet=True)
    except Exception as exc:
        print(f"Warning: could not load config {config_path}: {exc}")
        return None


def _parse_eeg_arg(raw):
    """Parse --eeg_channels (space/comma list, plain ids or NT notation)."""
    if not raw:
        return None
    try:
        from sorting import _parse_channel_list
        return _parse_channel_list(raw)
    except Exception:
        return [int(t) for t in raw.replace(",", " ").split()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute EMG-from-LFP (Buzsáki cross-channel correlation) from "
                    "the session LFP (fallback: the raw .rec via SpikeInterface). "
                    "Uses one channel per tetrode and leaves out the rat's "
                    "BAD_CHANNELS and EEG_TETRODES channels from the config.")
    parser.add_argument("--input_folder", required=True,
                        help="Folder with raw export (**/*.raw/*_group0.dat) or **/*.rec.")
    parser.add_argument("--output_folder", required=True,
                        help="Destination (writes into <output>/LFP_Output/).")
    parser.add_argument("--work_fs", type=float, default=WORK_FS,
                        help=f"Correlation sample rate in Hz (default {WORK_FS:.0f}).")
    parser.add_argument("--n_pick", type=int, default=N_PICK,
                        help=f"Raw fallback only: at most this many channels, one "
                             f"per tetrode (default {N_PICK}).")
    parser.add_argument("--config", default=None,
                        help="hm_tracker_paths.txt with the rat's BAD_CHANNELS / "
                             "EEG_TETRODES (default ~/Desktop/hm_tracker_paths.txt).")
    parser.add_argument("--eeg_channels", default=None,
                        help="More channels to EXCLUDE from the EMG correlation, "
                             "e.g. 'NT5' or '16 17 18 19'. Adds to the config's "
                             "bad/EEG channels.")
    parser.add_argument("--include_bad", action="store_true",
                        help="Keep the rat's bad/EEG channels in the correlation "
                             "(they are left out by default).")
    parser.add_argument("--exclude_bad", action="store_true",
                        help="Also leave out low-SNR channels (SNR < snr_frac x "
                             "median). The config bad/EEG channels are left out "
                             "anyway.")
    parser.add_argument("--keep-npy", dest="keep_npy", action="store_true",
                        help="Also write the EMG .npy files into LFP_Output/. "
                             "They duplicate what the NWB holds; off by default.")
    parser.add_argument("--snr_frac", type=float, default=SNR_FRAC,
                        help=f"With --exclude_bad, drop channels with SNR < "
                             f"snr_frac*median (default {SNR_FRAC}).")
    args = parser.parse_args()
    run(args.input_folder, args.output_folder, work_fs=args.work_fs, n_pick=args.n_pick,
        config=_load_config(args.config), eeg_channels=_parse_eeg_arg(args.eeg_channels),
        include_bad=args.include_bad, exclude_low_snr=args.exclude_bad,
        snr_frac=args.snr_frac, keep_npy=args.keep_npy)
