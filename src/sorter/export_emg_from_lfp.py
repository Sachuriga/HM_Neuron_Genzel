"""Step-8 companion — EMG-from-LFP (Buzsáki/Schomburg cross-channel correlation).

The method (see ``emg_from_lfp.py``) needs real 300–600 Hz content. With the LFP
now exported at 1500 Hz / ``-lfplowpass 700`` (Nyquist 750), that band survives in
the LFP itself — exactly how buzcode/anjal run it on 1250 Hz LFP. So the primary,
fast path reads the LFP export directly with numpy; no wideband/SpikeInterface.

PRIMARY — from the 1500 Hz LFP export (``<output>/LFP_Output/``):
    load lfp_data.npy -> pick channels (EEG config, else all/one-per-tetrode)
      -> band-pass 300–600 Hz -> cross-channel correlation EMG   (numpy/scipy)

FALLBACK — raw wideband via SpikeInterface (only when the LFP band is missing,
e.g. an old ``-lfplowpass 500`` export): read ``*.raw/*_group0.dat`` or the
``.rec`` directly, select spread/EEG channels, band-pass 300–600, decimate,
correlate. (Runs in the SpikeInterface env, HM_neuron.)

Output (in ``<output>/LFP_Output/``):
    emg_from_lfp.npy            EMG at the LFP rate (upsampled), pipeline-aligned
    emg_from_lfp_5hz.npy        native EMG (default 5 Hz), smoothed
    emg_from_lfp_timestamps.npy time axis (s) for the 5 Hz signal
"""

import argparse
from pathlib import Path

import numpy as np

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
    ``trodesexport -raw`` extraction is needed."""
    base = Path(input_folder)
    return sorted(base.glob("**/*.rec"), key=lambda f: f.name)


def spread_channels(n_channels, n_pick=N_PICK, ch_per_tetrode=CH_PER_TETRODE):
    """Pick ``n_pick`` channels spread across the probe: one per tetrode first.

    Channel index = (tetrode)*4 + local, so ``range(0, n, 4)`` is local-0 of every
    tetrode — a clean spatial spread across the 8×4 tetrode grid. If more than the
    number of tetrodes are requested, the extra picks are the next local index.
    """
    n_tetrodes = n_channels // ch_per_tetrode
    picks = []
    local = 0
    while len(picks) < n_pick and local < ch_per_tetrode:
        picks += [t * ch_per_tetrode + local for t in range(n_tetrodes)]
        local += 1
    return sorted(picks[:n_pick])


# ──────────────────────────────────────────────────────────────────────────────
# PRIMARY PATH: EMG from the 1500 Hz LFP export (numpy only, no SpikeInterface)
# ──────────────────────────────────────────────────────────────────────────────

def _lfp_sampling_rate(lfp_dir):
    """LFP sampling rate (Hz) from lfp_timestamps.npy, else None."""
    ts_file = find_output(lfp_dir, "lfp_timestamps.npy")   # prefixed or not
    if ts_file is None:
        return None
    ts = np.asarray(np.load(ts_file, mmap_mode="r")).ravel()
    if ts.size < 2:
        return None
    span = float(ts[-1]) - float(ts[0])
    return (ts.size - 1) / span if span > 0 else None


def _hw_to_lfp_columns(lfp_dir, hw_channels):
    """Map hardware channel ids -> lfp_data.npy column indices via channel_map.npy.

    channel_map.npy holds per-column {index, ntrode, channel, ...}; the hardware
    id of a column is (ntrode-1)*4 + (channel-1). Returns the columns present.
    """
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


SNR_FRAC = 0.3          # exclude LFP cols with SNR < SNR_FRAC * median SNR
MIN_KEEP = 8            # never drop the EMG correlation below this many channels


def _bad_lfp_columns(lfp_dir, n_cols, snr_frac=SNR_FRAC, config=None, stem=None,
                     extra_exclude_hw=None, min_keep=MIN_KEEP):
    """LFP columns to drop from the EMG correlation.

    Union of (1) SNR outliers — ``channel_snr_scores.npy`` columns below
    ``snr_frac * median`` — and (2) the config ``BAD_CHANNELS_<rat>`` /
    ``EEG_TETRODES_<rat>`` (plus any ``extra_exclude_hw``) hardware ids mapped to
    LFP columns. Guards so at least ``min_keep`` channels survive (re-admitting the
    highest-SNR excluded columns first). Returns a set of column indices.
    """
    bad, reasons = set(), {}
    scores = None
    snr_file = find_output(lfp_dir, "channel_snr_scores.npy")   # prefixed or not
    if snr_file is not None:
        scores = np.asarray(np.load(snr_file)).ravel()
        finite = np.isfinite(scores)
        if scores.size == n_cols and finite.any():
            thr = snr_frac * float(np.median(scores[finite]))
            for c in np.where(scores < thr)[0]:
                bad.add(int(c)); reasons[int(c)] = f"SNR {scores[c]:.0f}<{thr:.0f}"

    hw = set()
    if config is not None and stem is not None:
        try:
            from sorting import resolve_rat_settings
            bad_hw, _ref, eeg_hw = resolve_rat_settings(stem, config)
            hw |= {int(c) for c in bad_hw} | {int(c) for c in eeg_hw}
        except Exception as exc:
            print(f"   (bad-channel config not resolved: {exc})")
    if extra_exclude_hw:
        hw |= {int(c) for c in extra_exclude_hw}
    for c in (_hw_to_lfp_columns(lfp_dir, sorted(hw)) or []):
        bad.add(int(c)); reasons.setdefault(int(c), "config bad/EEG")

    keep_floor = min(min_keep, n_cols)
    if n_cols - len(bad) < keep_floor:
        # re-admit the least-bad (highest-SNR) excluded columns until at the floor
        def _snr(c):
            return scores[c] if scores is not None and c < scores.size else 0.0
        for c in sorted(bad, key=_snr, reverse=True):
            if n_cols - len(bad) >= keep_floor:
                break
            bad.discard(c); reasons.pop(c, None)
    if bad:
        print("   EMG: excluding LFP cols " +
              ", ".join(f"{c}({reasons[c]})" for c in sorted(bad)))
    return bad


def emg_from_lfp_output(lfp_dir, emg_fs=EMG_FS, channels=None, exclude_bad=False,
                        snr_frac=SNR_FRAC, config=None, stem=None,
                        extra_exclude_hw=None):
    """EMG-from-LFP computed directly from the 1500 Hz LFP export (numpy/scipy).

    The LFP already carries the 300–600 Hz band (exportLFP -lfplowpass 700), so no
    wideband/SpikeInterface read is needed — this is the fast primary path.

    Channel selection:
      * ``channels`` (hardware ids) given and mappable -> use those LFP columns.
      * otherwise -> all LFP columns (the export is one-per-tetrode, i.e. already
        spread across the probe — ideal for the cross-channel correlation).
    With ``exclude_bad`` (default), SNR-outlier and config bad/EEG columns are
    then dropped (see :func:`_bad_lfp_columns`), since a noisy or reference
    channel dilutes every pairwise correlation.
    """
    lfp_dir = Path(lfp_dir)
    fs = _lfp_sampling_rate(lfp_dir)
    if fs is None:
        raise FileNotFoundError("no lfp_timestamps.npy in LFP_Output")
    if fs / 2.0 <= 600.0:
        raise ValueError(
            f"LFP fs={fs:.0f} Hz (Nyquist {fs/2:.0f}) is below the 300–600 Hz EMG "
            f"band. Re-export LFP at 1500 Hz / -lfplowpass 700.")

    data_file = find_output(lfp_dir, "lfp_data.npy")       # prefixed or not
    if data_file is None:
        raise FileNotFoundError("no lfp_data.npy in LFP_Output")
    data = np.load(data_file, mmap_mode="r")               # (n_samples, n_channels)

    cols, origin = None, None
    if channels:
        mapped = _hw_to_lfp_columns(lfp_dir, [int(c) for c in channels])
        if mapped and len(mapped) >= 2:
            cols, origin = sorted(set(mapped)), f"EEG channels -> LFP cols {sorted(set(mapped))}"
        else:
            print("   (requested EEG channels not all in the LFP export; "
                  "using all LFP channels instead)")
    if cols is None:
        cols = list(range(data.shape[1]))
        origin = f"all {data.shape[1]} LFP channels (one per tetrode)"

    if exclude_bad:
        bad = _bad_lfp_columns(lfp_dir, data.shape[1], snr_frac=snr_frac,
                               config=config, stem=stem,
                               extra_exclude_hw=extra_exclude_hw)
        kept = [c for c in cols if c not in bad]
        dropped = sorted(c for c in cols if c in bad)
        if dropped and len(kept) >= 2:
            cols, origin = kept, origin + f" minus {dropped} (bad/EEG)"

    sel = np.asarray(data[:, cols], dtype=np.float64)
    res = emg_from_lfp(sel, fs, emg_fs=emg_fs, smooth_window_s=SMOOTH_S)
    res["duration_s"] = data.shape[0] / fs
    res["picks"] = cols
    res["origin"] = origin
    res["fs"] = fs
    return res


# ──────────────────────────────────────────────────────────────────────────────
# FALLBACK PATH: EMG from the raw wideband (SpikeInterface), for LP-500 LFP
# ──────────────────────────────────────────────────────────────────────────────

def emg_from_recording(rec, work_fs=WORK_FS, emg_fs=EMG_FS, n_pick=N_PICK,
                       channels=None):
    """Core: EMG-from-LFP for a SpikeInterface recording object (any source).

    Channel selection:
      * ``channels`` given (e.g. the tracker's EEG channels) -> use ALL of them.
      * otherwise -> ``n_pick`` channels spread across the probe (one per tetrode).

    Band-passes to the EMG band, decimates/resamples to ``work_fs``, and
    correlates. Returns the ``emg_correlation`` dict plus ``duration_s``,
    ``picks`` and ``passband``.
    """
    n_channels = rec.get_num_channels()
    duration_s = rec.get_total_duration()
    src_fs = rec.get_sampling_frequency()

    ch_ids = rec.get_channel_ids()
    if channels:
        picks = sorted(int(c) for c in channels if 0 <= int(c) < n_channels)
        if len(picks) < 2:
            raise ValueError(f"Need >= 2 valid EEG channels; got {picks} "
                             f"(recording has {n_channels} channels).")
    else:
        picks = spread_channels(n_channels, n_pick=n_pick)
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
                      channels=None):
    """EMG-from-LFP for one extracted raw ``*.raw/*_group0.dat`` (30 kHz, 128 ch)."""
    raw = readTrodesExtractedDataFile(str(dat_file))
    voltage = raw["data"]["voltage"]                       # (n_samples, n_channels)
    rec = si.NumpyRecording(traces_list=[voltage], sampling_frequency=RAW_FS)
    rec.set_channel_gains(GAIN)
    rec.set_channel_offsets(0.0)
    return emg_from_recording(rec, work_fs=work_fs, emg_fs=emg_fs, n_pick=n_pick,
                              channels=channels)


def emg_for_rec_file(rec_file, work_fs=WORK_FS, emg_fs=EMG_FS, n_pick=N_PICK,
                     channels=None):
    """EMG-from-LFP for a Trodes ``.rec`` read lazily (no raw extraction needed)."""
    rec = se.read_spikegadgets(str(rec_file))
    return emg_from_recording(rec, work_fs=work_fs, emg_fs=emg_fs, n_pick=n_pick,
                              channels=channels)


def _resolve_eeg_channels(stem, config, eeg_override):
    """EEG channels to use for one recording. Priority: explicit override >
    tracker config (EEG_TETRODES_<rat>) > None (falls back to a probe spread)."""
    if eeg_override:
        return list(eeg_override)
    if config:
        try:
            from sorting import resolve_rat_settings
            _bad, _ref, eeg = resolve_rat_settings(stem, config)
            return eeg
        except Exception as exc:
            print(f"   (could not resolve EEG channels from config: {exc})")
    return None


def _session_stem(input_folder):
    """A recording stem for matching the rat config (from a .rec/.dat, else folder)."""
    recs = find_rec_files(input_folder) or find_raw_dats(input_folder)
    return recs[0].stem if recs else Path(input_folder).name


def _emg_from_raw(input_folder, work_fs, emg_fs, n_pick, chans,
                  smoothed, timestamps, boundaries):
    """Fallback: EMG from raw wideband (extracted .dat, else .rec via SI)."""
    dats = find_raw_dats(input_folder)
    sources = [(d, "dat") for d in dats] if dats else \
        [(r, "rec") for r in find_rec_files(input_folder)]
    offset_s = 0.0
    for src, kind in sources:
        print(f"  • {src.name}")
        loader = emg_for_recording if kind == "dat" else emg_for_rec_file
        res = loader(src, work_fs=work_fs, emg_fs=emg_fs, n_pick=n_pick, channels=chans)
        origin = f"EEG channels {res['picks']}" if chans else \
            f"{len(res['picks'])} ch spread (one per tetrode)"
        print(f"      {origin}, band {res['passband'][0]:.0f}-"
              f"{res['passband'][1]:.0f} Hz, {res['data'].size} EMG bins @ {emg_fs} Hz")
        smoothed.append(res["smoothed"])
        timestamps.append(res["timestamps"] + offset_s)
        boundaries.append({"name": src.stem, "start_s": offset_s,
                           "duration_s": res["duration_s"]})
        offset_s += res["duration_s"]


def run(input_folder, output_folder, work_fs=WORK_FS, emg_fs=EMG_FS, n_pick=N_PICK,
        config=None, eeg_channels=None, exclude_bad=False, snr_frac=SNR_FRAC):
    output_dir = Path(output_folder) / "LFP_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"▶  EMG-from-LFP (Buzsáki cross-channel correlation)")
    print(f"   Input:  {input_folder}")
    print(f"   Output: {output_dir}")

    stem = _session_stem(input_folder)
    chans = _resolve_eeg_channels(stem, config, eeg_channels)

    smoothed, timestamps, boundaries = [], [], []
    # PRIMARY: straight from the 1500 Hz LFP export (fast, numpy only). Use ALL
    # tetrodes minus bad/EEG ones — EEG tetrodes (``chans``) are *excluded* from
    # the correlation here, not used as its source.
    try:
        res = emg_from_lfp_output(output_dir, emg_fs=emg_fs, exclude_bad=exclude_bad,
                                  snr_frac=snr_frac, config=config, stem=stem,
                                  extra_exclude_hw=chans)
        print(f"   From 1500 Hz LFP export @ {res['fs']:.0f} Hz: {res['origin']}, "
              f"band {res['passband'][0]:.0f}-{res['passband'][1]:.0f} Hz, "
              f"{res['data'].size} EMG bins @ {emg_fs} Hz")
        smoothed = [res["smoothed"]]
        timestamps = [res["timestamps"]]
        boundaries = [{"name": "lfp", "start_s": 0.0, "duration_s": res["duration_s"]}]
    except (FileNotFoundError, ValueError) as exc:
        print(f"   LFP path unavailable ({exc})")
        print(f"   Falling back to the raw wideband via SpikeInterface.")
        _emg_from_raw(input_folder, work_fs, emg_fs, n_pick, chans,
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
    np.save(output_dir / f"{pfx}emg_from_lfp_5hz.npy", emg_5hz_norm)
    np.save(output_dir / f"{pfx}emg_from_lfp_timestamps.npy", ts_5hz)
    print(f"  ✓ {pfx}emg_from_lfp_5hz.npy  ({emg_5hz_norm.size}) @ {emg_fs} Hz")

    # Upsample to the LFP rate (1500 Hz) so it matches emg_rms.npy / lfp_data.npy
    # per-sample. Prefer the real LFP time axis when present.
    lfp_ts_file = find_output(output_dir, "lfp_timestamps.npy")
    if lfp_ts_file is not None:
        lfp_ts = np.load(lfp_ts_file).astype("float64")
        emg_up = np.interp(lfp_ts, ts_5hz, emg_5hz_norm).astype("float32")
        np.save(output_dir / f"{pfx}emg_from_lfp.npy", emg_up)
        print(f"  ✓ {pfx}emg_from_lfp.npy  ({emg_up.size}) upsampled to LFP time axis")
    else:
        print("  ⚠  lfp_timestamps.npy not found — saved only the 5 Hz EMG. "
              "Run step-8 LFP export first to also get the pipeline-rate emg_from_lfp.npy.")

    if len(boundaries) > 1:
        np.save(output_dir / f"{pfx}emg_from_lfp_boundaries.npy", boundaries)
        for b in boundaries:
            print(f"    {b['name']}: t0={b['start_s']:.1f}s  dur={b['duration_s']:.1f}s")

    print(f"{'=' * 60}")
    print(f"✅  EMG-from-LFP → {output_dir}")


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
        return load_sorting_config(config_path)
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
                    "the raw wideband recording (or .rec) via SpikeInterface. Uses "
                    "the tracker's EEG channels when configured, else 32 channels "
                    "spread across the probe (one per tetrode).")
    parser.add_argument("--input_folder", required=True,
                        help="Folder with raw export (**/*.raw/*_group0.dat) or **/*.rec.")
    parser.add_argument("--output_folder", required=True,
                        help="Destination (writes into <output>/LFP_Output/).")
    parser.add_argument("--work_fs", type=float, default=WORK_FS,
                        help=f"Correlation sample rate in Hz (default {WORK_FS:.0f}).")
    parser.add_argument("--n_pick", type=int, default=N_PICK,
                        help=f"Spread channels used when no EEG channels are given "
                             f"(default {N_PICK}).")
    parser.add_argument("--config", default=None,
                        help="hm_tracker_paths.txt for per-rat EEG channels "
                             "(default ~/Desktop/hm_tracker_paths.txt).")
    parser.add_argument("--eeg_channels", default=None,
                        help="Explicit EEG channels to EXCLUDE from the EMG "
                             "correlation, e.g. 'NT5' or '16 17 18 19'. Adds to the "
                             "config EEG/bad channels.")
    parser.add_argument("--exclude_bad", action="store_true",
                        help="Drop SNR-outlier + config bad/EEG channels from the EMG "
                             "correlation. Off by default: for a healthy probe the "
                             "cross-channel mean is insensitive to channel choice "
                             "(~identical correlation); useful only when several "
                             "channels are bad.")
    parser.add_argument("--snr_frac", type=float, default=SNR_FRAC,
                        help=f"With --exclude_bad, drop channels with SNR < "
                             f"snr_frac*median (default {SNR_FRAC}).")
    args = parser.parse_args()
    run(args.input_folder, args.output_folder, work_fs=args.work_fs, n_pick=args.n_pick,
        config=_load_config(args.config), eeg_channels=_parse_eeg_arg(args.eeg_channels),
        exclude_bad=args.exclude_bad, snr_frac=args.snr_frac)
