import gc
import re
import argparse
import numpy as np
from pathlib import Path
from scipy.signal import welch
from scipy.stats import zscore
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# TRODES OFFICIAL READER (from readTrodesExtractedDataFile3.py)
# ──────────────────────────────────────────────────────────────────────────────

def readTrodesExtractedDataFile(filename):
    with open(filename, 'rb') as f:
        if f.readline().decode('ascii').strip() != '<Start settings>':
            raise Exception("Settings format not supported")
        fields = True
        fieldsText = {}
        for line in f:
            if fields:
                line = line.decode('ascii').strip()
                if line != '<End settings>':
                    vals = line.split(': ')
                    fieldsText.update({vals[0].lower(): vals[1]})
                else:
                    fields = False
                    break
        dt = _parseFields(fieldsText['fields'])
        data = np.fromfile(f, dt)
        fieldsText.update({'data': data})
        return fieldsText


def _parseFields(fieldstr):
    sep = re.split(r'\s', re.sub(r"\>\<|\>|\<", ' ', fieldstr).strip())
    typearr = []
    for i in range(0, len(sep), 2):
        fieldname = sep[i]
        repeats = 1
        ftype = 'uint32'
        if '*' in sep[i + 1]:
            temptypes = re.split(r'\*', sep[i + 1])
            ftype = temptypes[temptypes[0].isdigit()]
            repeats = int(temptypes[temptypes[1].isdigit()])
        else:
            ftype = sep[i + 1]
        fieldtype = getattr(np, ftype)
        typearr.append((str(fieldname), fieldtype, repeats))
    return np.dtype(typearr)


# ──────────────────────────────────────────────────────────────────────────────
# DISCOVER & LOAD TRODES LFP .dat FILES
# ──────────────────────────────────────────────────────────────────────────────

def find_lfp_sessions(input_folder):
    """
    Find Trodes-exported LFP sessions under input_folder.

    Each .rec recording exports into its own '<recording>.LFP' folder:
        <recording>.LFP/<recording>.LFP_nt<N>ch<C>.dat
        <recording>.LFP/<recording>.timestamps.dat

    When a folder holds several recordings (separate sessions), each one
    becomes its own session here, returned in chronological order (the Trodes
    name embeds YYYYMMDD_HHMMSS, so sorting by name == sorting by time).

    Returns a list of dicts: {'name', 'lfp_files', 'ts_file'}.
    """
    base = Path(input_folder)

    sessions = []
    for d in sorted((d for d in base.glob("*.LFP") if d.is_dir()),
                    key=lambda d: d.name):
        lfp_files = sorted(
            (f for f in d.glob("*.dat")
             if 'timestamps' not in f.name.lower()),
            key=lambda f: parse_channel_info(f)
        )
        if not lfp_files:
            continue
        ts_list = (list(d.glob("*.timestamps.dat"))
                   or list(d.glob("*timestamps*.dat")))
        sessions.append({
            'name': d.stem,
            'lfp_files': lfp_files,
            'ts_file': ts_list[0] if ts_list else None,
        })

    # Fallback: flat *LFP*.dat files directly under input_folder (one session).
    if not sessions:
        flat = sorted(
            (f for f in base.glob("*LFP*.dat")
             if 'timestamps' not in f.name.lower()),
            key=lambda f: parse_channel_info(f)
        )
        if flat:
            ts_list = list(base.glob("*timestamps*.dat"))
            sessions.append({
                'name': base.name,
                'lfp_files': flat,
                'ts_file': ts_list[0] if ts_list else None,
            })

    return sessions


def load_and_concat_sessions(sessions):
    """
    Load every session and concatenate channels across sessions in order.

    Sessions are joined sample-after-sample per channel, matched by
    (ntrode, channel). Only channels common to all sessions are kept.

    Returns (channels, fs, boundaries) where:
      - channels: list of {'data', 'ntrode', 'channel', 'file'} (concatenated)
      - fs: sampling rate (Hz) from the first session header
      - boundaries: list of {'name', 'start', 'n'} marking each session's
        sample range within the concatenated arrays.
    """
    fs = None
    loaded = []
    for s in sessions:
        chans, _ts, this_fs = load_lfp_channels(s['lfp_files'], s['ts_file'])
        if fs is None:
            fs = this_fs
        cmap = {(c['ntrode'], c['channel']): c['data'] for c in chans}
        fmap = {(c['ntrode'], c['channel']): c['file'] for c in chans}
        loaded.append({
            'name': s['name'],
            'cmap': cmap,
            'fmap': fmap,
            'n': chans[0]['data'].shape[0] if chans else 0,
        })

    # Channels common to every session (sorted by ntrode, then channel).
    common = set(loaded[0]['cmap'])
    for sm in loaded[1:]:
        common &= set(sm['cmap'])
    if not common:
        raise RuntimeError("No channels common to all LFP sessions.")
    keys = sorted(common, key=lambda k: (k[0] is None, k[0] or 0, k[1] or 0))

    if len(loaded) > 1:
        names = ", ".join(f"{sm['name']} ({sm['n']} samples)" for sm in loaded)
        tqdm.write(f"  Concatenating {len(loaded)} session(s): {names}")

    channels = []
    for k in keys:
        data = np.concatenate([sm['cmap'][k] for sm in loaded])
        channels.append({
            'data': data,
            'ntrode': k[0],
            'channel': k[1],
            'file': loaded[0]['fmap'][k],
        })

    boundaries = []
    start = 0
    for sm in loaded:
        boundaries.append({'name': sm['name'], 'start': start, 'n': sm['n']})
        start += sm['n']

    return channels, fs, boundaries


def parse_channel_info(filename):
    """
    Extract ntrode and channel number from Trodes LFP filename.
    e.g. 'Recording.LFP_nt3ch1.dat' -> (3, 1)
    """
    stem = Path(filename).stem
    match = re.search(r'_nt(\d+)ch(\d+)', stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def load_lfp_channels(lfp_files, ts_file):
    """
    Read each Trodes LFP .dat using the official reader.
    Returns:
      - channels: list of dicts with 'data', 'ntrode', 'channel', 'file'
      - timestamps: 1-D array of raw timestamps (or None)
      - fs: sampling rate from file header
    """
    channels = []
    fs = None

    for f in tqdm(lfp_files, desc="  Reading LFP .dat files", unit='file'):
        result = readTrodesExtractedDataFile(str(f))
        data = result['data']

        # Get sampling rate from header
        if fs is None:
            fs = float(result.get('samplingrate',
                       result.get('clockrate', 1000)))

        # Extract voltage scaling
        voltage_scale = float(result.get('voltagescaling', 0.195))

        # Get the data field — first field in the structured array
        field_names = data.dtype.names
        data_field = field_names[0]
        raw = data[data_field].astype('float32').flatten()
        voltage = raw * voltage_scale

        nt, ch = parse_channel_info(f)
        channels.append({
            'data': voltage,
            'ntrode': nt,
            'channel': ch,
            'file': f,
        })

    # Load timestamps from separate file
    timestamps = None
    if ts_file is not None:
        tqdm.write(f"  Loading timestamps from {ts_file.name}")
        ts_result = readTrodesExtractedDataFile(str(ts_file))
        ts_data = ts_result['data']
        ts_field = ts_data.dtype.names[0]
        timestamps = ts_data[ts_field].flatten()

    return channels, timestamps, fs


# ──────────────────────────────────────────────────────────────────────────────
# CHANNEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def select_emg_channel(lfp_array, fs, segment_duration_s=60):
    """Channel with highest power in the EMG band (20–200 Hz)."""
    num_channels = lfp_array.shape[1]
    n = min(int(segment_duration_s * fs), lfp_array.shape[0])
    emg_power = np.zeros(num_channels)

    with tqdm(total=num_channels, unit='ch', desc="  Scoring EMG channels",
              leave=False) as pb:
        for ch in range(num_channels):
            seg = np.array(lfp_array[:n, ch], dtype='float64')
            f, psd = welch(seg, fs=fs, nperseg=int(2 * fs))
            emg_power[ch] = np.mean(psd[(f >= 20) & (f <= 200)])
            del seg
            pb.update(1)

    emg_ch = int(np.argmax(emg_power))
    tqdm.write(f"  ✓ EMG channel: {emg_ch}  "
               f"(power: {emg_power[emg_ch]:.4f})")
    return emg_ch


def select_cleanest_channels(lfp_array, fs, n_best=3, segment_duration_s=60):
    """Sleep-band SNR ranking."""
    n = min(int(segment_duration_s * fs), lfp_array.shape[0])
    n_ch = lfp_array.shape[1]

    scores = np.zeros(n_ch)
    with tqdm(total=n_ch, unit='ch', desc="  Scoring EEG channels",
              leave=False) as pb:
        for ch in range(n_ch):
            seg = np.array(lfp_array[:n, ch], dtype='float64')
            f, psd = welch(seg, fs=fs, nperseg=int(2 * fs))
            sleep_band = np.mean(psd[(f >= 0.5) & (f <= 30)])
            noise_band = np.mean(psd[f > 50])
            scores[ch] = sleep_band / (noise_band + 1e-12)
            pb.update(1)

    best_idx = np.argsort(scores)[::-1][:n_best]
    tqdm.write(f"  ✓ Cleanest {n_best} channels: {best_idx}  "
               f"(scores: {scores[best_idx].round(2)})")
    return best_idx, scores


# ──────────────────────────────────────────────────────────────────────────────
# AWAKENESS
# ──────────────────────────────────────────────────────────────────────────────

def compute_awakeness(lfp_array, emg_1d, fs, best_ch_idx, epoch_s=1):
    n_samples   = lfp_array.shape[0]
    n_per_epoch = int(fs * epoch_s)
    n_epochs    = n_samples // n_per_epoch
    eeg_ch      = int(best_ch_idx[0])

    emg_power   = np.zeros(n_epochs, dtype='float32')
    theta_delta = np.zeros(n_epochs, dtype='float32')

    with tqdm(total=n_epochs, unit='s', desc="  Awakeness epochs",
              leave=False) as pb:
        for i in range(n_epochs):
            s = i * n_per_epoch
            e = s + n_per_epoch
            emg_seg = np.array(emg_1d[s:e], dtype='float64')
            eeg_seg = np.array(lfp_array[s:e, eeg_ch], dtype='float64')

            emg_power[i] = np.sqrt(np.mean(emg_seg ** 2))

            f, psd = welch(eeg_seg, fs=fs,
                           nperseg=min(n_per_epoch, int(2 * fs)))
            delta          = np.mean(psd[(f >= 0.5) & (f <= 4)])
            theta          = np.mean(psd[(f >= 5)   & (f <= 9)])
            theta_delta[i] = theta / (delta + 1e-12)
            pb.update(1)

    emg_z = zscore(emg_power)
    td_z  = zscore(theta_delta)
    awakeness_epochs = (0.6 * emg_z + 0.4 * td_z).astype('float32')

    n_lfp_samples = lfp_array.shape[0]
    tqdm.write(f"  Upsampling awakeness from {n_epochs} epochs → "
               f"{n_lfp_samples} samples")
    epoch_times  = (np.arange(n_epochs) + 0.5) * n_per_epoch
    sample_times = np.arange(n_lfp_samples, dtype='float64')
    awakeness    = np.interp(sample_times, epoch_times,
                             awakeness_epochs).astype('float32')
    emg_power_up   = np.interp(sample_times, epoch_times,
                               emg_power).astype('float32')
    theta_delta_up = np.interp(sample_times, epoch_times,
                               theta_delta).astype('float32')

    return awakeness, emg_power_up, theta_delta_up


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

STEPS = [
    "Find & load LFP .dat files",
    "Convert to .npy (per channel + combined)",
    "Select EMG channel",
    "Select cleanest EEG channels",
    "Compute awakeness",
]


def run_pipeline(input_folder, output_folder):
    base_path  = Path(input_folder)
    output_dir = Path(output_folder) / "LFP_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = ("{l_bar}{bar}| {n_fmt}/{total_fmt} steps  "
           "[{elapsed}<{remaining}]")

    with tqdm(total=len(STEPS), unit='step', desc=f"[{STEPS[0]}]",
              bar_format=fmt) as sp:

        def advance(i):
            if i < len(STEPS):
                sp.set_description(f"[{STEPS[i]}]")
            sp.update(1)

        # ── 1. FIND & LOAD LFP .dat FILES ───────────────────────────────────
        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"▶  Input: {base_path}")
        tqdm.write("Step 1/5 — Finding Trodes-exported LFP .dat files")

        sessions = find_lfp_sessions(input_folder)
        if not sessions:
            tqdm.write("❌  No LFP .dat files found! Check your input folder.")
            tqdm.write("    Expected: <recording>.LFP/<recording>.LFP_nt*ch*.dat")
            return

        total_files = sum(len(s['lfp_files']) for s in sessions)
        tqdm.write(f"  Found {len(sessions)} session(s), "
                   f"{total_files} LFP channel file(s) total")
        for s in sessions:
            ts_name = s['ts_file'].name if s['ts_file'] else "(none)"
            tqdm.write(f"    • {s['name']}: {len(s['lfp_files'])} ch, ts={ts_name}")

        channels, fs, boundaries = load_and_concat_sessions(sessions)
        tqdm.write(f"  Sampling rate from header: {fs} Hz")
        advance(1)

        # ── 2. SAVE PER-CHANNEL .npy + COMBINED ARRAY ───────────────────────
        tqdm.write("Step 2/5 — Converting to .npy files")

        npy_dir = output_dir / "channels_npy"
        npy_dir.mkdir(exist_ok=True)

        n_samples = channels[0]['data'].shape[0]
        num_channels = len(channels)

        for i, ch_info in enumerate(channels):
            nt = ch_info['ntrode']
            ch = ch_info['channel']
            if nt is not None:
                fname = f"lfp_nt{nt:02d}_ch{ch:02d}.npy"
            else:
                fname = f"lfp_ch{i:03d}.npy"
            np.save(npy_dir / fname, ch_info['data'])

        tqdm.write(f"  ✓ Saved {num_channels} individual .npy files → {npy_dir}")

        # Build combined (n_samples, n_channels) array
        lfp_array = np.column_stack([ch['data'] for ch in channels])
        np.save(output_dir / "lfp_data.npy", lfp_array)

        # Timestamps: build a continuous, gapless time axis across the
        # concatenated sessions. Per-session Trodes timestamps reset at each
        # recording start, so a raw concatenation would be non-monotonic; we
        # instead derive seconds from the sample index at the export rate.
        ts_seconds = (np.arange(n_samples) / fs).astype('float64')
        np.save(output_dir / "lfp_timestamps.npy", ts_seconds)

        # Record where each session begins/ends within the concatenated data.
        np.save(output_dir / "session_boundaries.npy", boundaries)
        if len(boundaries) > 1:
            for b in boundaries:
                tqdm.write(f"    {b['name']}: samples {b['start']}.."
                           f"{b['start'] + b['n']}  (t0={b['start'] / fs:.2f}s)")

        # Save channel mapping
        ch_info_list = []
        for i, ch in enumerate(channels):
            ch_info_list.append({
                'index': i,
                'ntrode': ch['ntrode'],
                'channel': ch['channel'],
                'source_file': str(ch['file'].name),
            })
        np.save(output_dir / "channel_map.npy", ch_info_list)

        tqdm.write(f"  ✓ lfp_data.npy  ({n_samples}, {num_channels}) @ {fs} Hz")

        del channels
        gc.collect()
        advance(2)

        # ── 3. SELECT EMG CHANNEL ───────────────────────────────────────────
        tqdm.write("Step 3/5 — Selecting EMG channel")
        emg_ch = select_emg_channel(lfp_array, fs)
        emg_1d = lfp_array[:, emg_ch].copy()

        np.save(output_dir / "emg_data.npy", emg_1d[:, np.newaxis])
        np.save(output_dir / "emg_channel_index.npy", np.array([emg_ch]))
        tqdm.write(f"  ✓ emg_data.npy  ({emg_1d.shape[0]}, 1)")
        advance(3)

        # ── 4. SELECT CLEANEST EEG CHANNELS ─────────────────────────────────
        tqdm.write("Step 4/5 — Selecting cleanest EEG channels")
        best_ch_idx, scores = select_cleanest_channels(lfp_array, fs, n_best=3)
        np.save(output_dir / "cleanest_channel_indices.npy", best_ch_idx)
        np.save(output_dir / "channel_snr_scores.npy", scores)
        advance(4)

        # ── 5. COMPUTE AWAKENESS ────────────────────────────────────────────
        tqdm.write("Step 5/5 — Computing awakeness score")
        awakeness, emg_rms, theta_delta = compute_awakeness(
            lfp_array, emg_1d, fs, best_ch_idx,
        )
        np.save(output_dir / "awakeness.npy",        awakeness)
        np.save(output_dir / "emg_rms.npy",          emg_rms)
        np.save(output_dir / "theta_delta_ratio.npy", theta_delta)
        advance(5)

        del lfp_array, emg_1d
        gc.collect()

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"✅  All outputs → {output_dir}")
    tqdm.write(f"   channels_npy/                {num_channels} individual channel .npy files")
    tqdm.write(f"   lfp_data.npy                 ({n_samples}, {num_channels}) @ {fs} Hz")
    tqdm.write(f"   lfp_timestamps.npy           time axis (s)")
    tqdm.write(f"   session_boundaries.npy       per-session sample ranges")
    tqdm.write(f"   channel_map.npy              ntrode/channel mapping")
    tqdm.write(f"   cleanest_channel_indices.npy top-3 EEG ch: {best_ch_idx}")
    tqdm.write(f"   channel_snr_scores.npy       SNR scores (all channels)")
    tqdm.write(f"   emg_data.npy                 channel {emg_ch}")
    tqdm.write(f"   emg_channel_index.npy        EMG channel index")
    tqdm.write(f"   awakeness.npy                per-sample ({n_samples})")
    tqdm.write(f"   emg_rms.npy                  per-sample ({n_samples})")
    tqdm.write(f"   theta_delta_ratio.npy        per-sample ({n_samples})")
    tqdm.write(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Trodes-exported LFP .dat files to .npy and compute awakeness."
    )
    parser.add_argument('--input_folder',  required=True,
                        help="Folder containing Trodes LFP export (*.LFP/*.dat)")
    parser.add_argument('--output_folder', required=True,
                        help="Destination for all output files.")
    args = parser.parse_args()
    run_pipeline(args.input_folder, args.output_folder)