import os
import shutil
import traceback
from pathlib import Path
import numpy as np

# Built-in fallback used only when the config file has no entry for a rat.
# Keys are matched against the lowercased file stem prefix (e.g. "rat1").
DEFAULT_BAD_CHANNELS = {
    "rat1": [0, 1, 2, 3,
             4, 5, 6, 7,
             28, 29, 30, 31,
             36, 37, 38, 39,
             41, 42, 43,
             44, 45, 46, 47, 51,
             56, 59,
             68, 69, 70, 71,
             80, 92, 93, 94, 95,
             96, 97, 98, 99,
             100, 101, 102, 103,
             108, 109, 110, 111,
             124, 125, 126, 127],
}


import re

# Sorter selection. Stored in the config dict under a reserved key that can
# never collide with a rat token (rat keys are matched against file stems).
SORTER_CONFIG_KEY = "__sorter__"
DEFAULT_SORTER = "mountainsort5"
SUPPORTED_SORTERS = ("mountainsort5", "mountainsort4")


def resolve_sorter(config):
    """Return the configured sorter name, falling back to the default."""
    sorter = (config or {}).get(SORTER_CONFIG_KEY, DEFAULT_SORTER)
    if sorter not in SUPPORTED_SORTERS:
        print(f"Warning: unsupported sorter '{sorter}' in config; "
              f"using {DEFAULT_SORTER}. Supported: {', '.join(SUPPORTED_SORTERS)}.")
        sorter = DEFAULT_SORTER
    return sorter


# Global numeric sorting settings. Stored in the config dict under a reserved
# key. Maps the config-file key (upper case) -> (internal name, converter,
# default value).
SORTING_PARAMS_KEY = "__params__"
SORTING_PARAM_SPEC = {
    "FREQ_MIN": ("freq_min", float, 600.0),          # band-pass high-pass cutoff (Hz)
    "FREQ_MAX": ("freq_max", float, 8000.0),         # band-pass low-pass cutoff (Hz)
    "DETECT_THRESHOLD": ("detect_threshold", float, 5.0),  # spike detection threshold
    "DETECT_SIGN": ("detect_sign", int, 0),          # -1 neg, 0 both, 1 pos
}


def resolve_sorting_params(config):
    """Return the numeric sorting settings, filling in defaults for any
    keys not present in the config file."""
    stored = (config or {}).get(SORTING_PARAMS_KEY, {})
    return {name: stored.get(name, default)
            for (name, _conv, default) in SORTING_PARAM_SPEC.values()}


# 128 channels are wired as 32 tetrodes of 4: NT1..NT32, each with ch1..ch4.
# A token like "NT5ch3" maps to hardware channel (5-1)*4 + (3-1) = 18.
N_TETRODES = 32
CH_PER_TETRODE = 4
_NT_RE = re.compile(r"^nt(\d+)ch(\d+)$", re.IGNORECASE)


def nt_to_channel(tetrode, ch):
    """Map a 1-based NT tetrode / 1-based channel to a 0-based hardware id."""
    if not (1 <= tetrode <= N_TETRODES):
        raise ValueError(f"tetrode {tetrode} out of range 1..{N_TETRODES}")
    if not (1 <= ch <= CH_PER_TETRODE):
        raise ValueError(f"channel {ch} out of range 1..{CH_PER_TETRODE}")
    return (tetrode - 1) * CH_PER_TETRODE + (ch - 1)


def channel_to_nt(channel):
    """Inverse of nt_to_channel: 0-based hardware id -> (tetrode, ch), 1-based."""
    return channel // CH_PER_TETRODE + 1, channel % CH_PER_TETRODE + 1


def _parse_one_channel(token):
    """Parse a single token (plain int or NTxchY) into a 0-based channel id."""
    m = _NT_RE.match(token)
    if m:
        return nt_to_channel(int(m.group(1)), int(m.group(2)))
    return int(token)


def _parse_channel_list(raw_value):
    """
    Parse a space/comma separated list of channel ids into a list of ints.
    Each token may be a plain 0-based id (e.g. "18") or NT notation
    (e.g. "NT5ch3"). The two styles can be mixed.
    """
    if raw_value is None:
        return []
    tokens = raw_value.replace(",", " ").split()
    channels = []
    for tok in tokens:
        try:
            channels.append(_parse_one_channel(tok))
        except ValueError as e:
            print(f"Warning: ignoring invalid channel token '{tok}' in config ({e}).")
    return channels


# A whole-tetrode token: "NT5" or just "5" (1-based tetrode number).
_NT_TETRODE_RE = re.compile(r"^(?:nt)?(\d+)$", re.IGNORECASE)


def _parse_tetrode_token(token):
    """Parse a tetrode token ("NT5" or "5") into its four 0-based channel ids."""
    m = _NT_TETRODE_RE.match(token)
    if not m:
        raise ValueError(f"'{token}' is not a tetrode (expected NT<t> or <t>)")
    tetrode = int(m.group(1))
    if not (1 <= tetrode <= N_TETRODES):
        raise ValueError(f"tetrode {tetrode} out of range 1..{N_TETRODES}")
    base = (tetrode - 1) * CH_PER_TETRODE
    return [base + c for c in range(CH_PER_TETRODE)]


def _parse_tetrode_list(raw_value):
    """
    Parse a space/comma separated list of whole-tetrode tokens into the flat
    list of 0-based channel ids they cover. Each token is "NT<t>" or "<t>"
    (1-based tetrode number). Used for EEG tetrodes that are excluded from
    spike sorting entirely.
    """
    if raw_value is None:
        return []
    tokens = raw_value.replace(",", " ").split()
    channels = []
    for tok in tokens:
        try:
            channels.extend(_parse_tetrode_token(tok))
        except ValueError as e:
            print(f"Warning: ignoring invalid tetrode token '{tok}' in config ({e}).")
    return channels


# Sleep-scoring channels: one tetrode per role.
#   cortex -> NREM slow-wave/delta ; sr (stratum radiatum) -> REM theta ;
#   pyr (pyramidal) -> optional (ripples / EMG spread).
SLEEP_ROLES = ("cortex", "sr", "pyr")


def _parse_sleep_channels(raw_value):
    """Parse ``cortex:NT28 sr:NT10 pyr:NT5`` (or plain tetrode numbers) into
    ``{'cortex': 28, 'sr': 10, 'pyr': 5}`` — 1-based tetrode numbers per role."""
    out = {}
    if not raw_value:
        return out
    for tok in raw_value.replace(",", " ").split():
        if ":" not in tok:
            print(f"Warning: ignoring sleep-channel token '{tok}' (want role:tetrode).")
            continue
        role, ch = tok.split(":", 1)
        role = role.strip().lower()
        m = _NT_TETRODE_RE.match(ch.strip())
        if role not in SLEEP_ROLES:
            print(f"Warning: unknown sleep-channel role '{role}' (use {SLEEP_ROLES}).")
        elif not m or not (1 <= int(m.group(1)) <= N_TETRODES):
            print(f"Warning: invalid tetrode '{ch}' for sleep role '{role}'.")
        else:
            out[role] = int(m.group(1))
    return out


def resolve_sleep_channels(file_stem, config):
    """Return the ``{cortex, sr, pyr}`` sleep tetrodes for a recording (by rat), or {}."""
    stem = file_stem.lower()
    for rat in sorted(config, key=len, reverse=True):
        if isinstance(config[rat], dict) and stem.startswith(rat):
            return dict(config[rat].get("sleep_channels", {}))
    return {}


def load_sorting_config(config_path):
    """
    Read per-rat sorting settings from an hm_tracker_paths.txt style file.

    Recognised keys (RAT token is anything, e.g. RAT1, RAT2, MOUSEA):
        SORTER=mountainsort5             spike sorter for all rats
                                         (mountainsort5 or mountainsort4)
        FREQ_MIN=600                     band-pass high-pass cutoff (Hz)
        FREQ_MAX=8000                    band-pass low-pass cutoff (Hz)
        DETECT_THRESHOLD=5               spike detection threshold
        DETECT_SIGN=0                    -1 neg, 0 both, 1 pos
        BAD_CHANNELS_<RAT>=0 1 2 3 ...   channels to interpolate
        REF_CHANNEL_<RAT>=64             single (or space separated) reference
                                         channel(s) for common_reference.
                                         If omitted, global median is used.
        EEG_TETRODES_<RAT>=NT5 17 ...    whole tetrodes ("NT<t>" or "<t>") used
                                         for EEG; their channels are dropped
                                         from the recording before sorting.

    Returns a dict like:
        { "rat1": {"bad_channels": [...], "ref_channels": [...],
                   "eeg_channels": [...]}, ... }
    Rat tokens are lowercased so they can be matched against the file stem.
    """
    config = {}
    if not config_path:
        return config
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Sorting config not found at {config_path} — using built-in defaults.")
        return config

    def _entry(rat):
        return config.setdefault(
            rat, {"bad_channels": [], "ref_channels": [], "eeg_channels": [],
                  "sleep_channels": {}}
        )

    with open(config_path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "SORTER":
                config[SORTER_CONFIG_KEY] = value.lower()
            elif key in SORTING_PARAM_SPEC:
                name, conv, default = SORTING_PARAM_SPEC[key]
                if value == "":
                    continue  # empty -> keep default
                try:
                    config.setdefault(SORTING_PARAMS_KEY, {})[name] = conv(value)
                except ValueError:
                    print(f"Warning: invalid value '{value}' for {key}; "
                          f"using default {default}.")
            elif key.startswith("BAD_CHANNELS_"):
                rat = key[len("BAD_CHANNELS_"):].lower()
                _entry(rat)["bad_channels"] = _parse_channel_list(value)
            elif key.startswith("REF_CHANNEL_"):
                rat = key[len("REF_CHANNEL_"):].lower()
                _entry(rat)["ref_channels"] = _parse_channel_list(value)
            elif key.startswith("EEG_TETRODES_"):
                rat = key[len("EEG_TETRODES_"):].lower()
                _entry(rat)["eeg_channels"] = _parse_tetrode_list(value)
            elif key.startswith("SLEEP_CHANNELS_"):
                rat = key[len("SLEEP_CHANNELS_"):].lower()
                _entry(rat)["sleep_channels"] = _parse_sleep_channels(value)

    rats = [k for k in config if not k.startswith("__")]
    if rats:
        print(f"Loaded sorting config for rats: {', '.join(sorted(rats))}")
    if SORTER_CONFIG_KEY in config:
        print(f"Sorter selected in config: {config[SORTER_CONFIG_KEY]}")
    if SORTING_PARAMS_KEY in config:
        print(f"Sorting params from config: {config[SORTING_PARAMS_KEY]}")
    return config


def resolve_rat_settings(file_stem, config):
    """
    Match a recording's file stem against the configured rats and return its
    bad channels, reference channels and EEG channels (channels belonging to
    EEG tetrodes, excluded from sorting). Falls back to DEFAULT_BAD_CHANNELS.
    """
    stem = file_stem.lower()

    # Prefer the longest matching rat token so "rat10" beats "rat1".
    matched = None
    for rat in sorted(config, key=len, reverse=True):
        if stem.startswith(rat):
            matched = rat
            break

    if matched is not None:
        settings = config[matched]
        bad_channels = list(settings.get("bad_channels", []))
        ref_channels = list(settings.get("ref_channels", []))
        eeg_channels = list(settings.get("eeg_channels", []))
        eeg_note = ""
        if eeg_channels:
            eeg_note = f", {len(eeg_channels)} EEG channel(s) excluded"
        print(f"Using config for '{matched}': "
              f"{len(bad_channels)} bad channel(s), "
              f"ref={ref_channels if ref_channels else 'global median'}"
              f"{eeg_note}")
        return bad_channels, ref_channels, eeg_channels

    # No config entry: fall back to built-in defaults (bad channels only).
    for rat, channels in DEFAULT_BAD_CHANNELS.items():
        if stem.startswith(rat):
            print(f"No config entry for '{rat}'; using built-in default bad channels.")
            return list(channels), [], []

    return [], [], []

import probeinterface as pi
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import argparse

# Ensure the Trodes extractor is available
try:
    from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
except ImportError:
    print("Warning: 'readTrodesExtractedDataFile3.py' not found. Please ensure it is in the directory or PYTHONPATH.")


# Post-sorting analysis (analyzer, metrics, quality-check labels, Phy export)
# lives in sorter_common.py so it stays identical between the full pipeline and
# continue_sorting.py. QUALITY_CHECK_THRESHOLDS / label_units_quality_check are
# re-exported here for convenience/backwards-compatibility.
from sorter_common import (  # noqa: E402
    QUALITY_CHECK_THRESHOLDS,
    label_units_quality_check,
    analyze_and_export,
)


def process_single_file(file_path, output_parent, fs=30000.0, gain=0.195, offset=0.0, n_jobs=4,
                        bad_channel_ids=None, ref_channels=None, eeg_channel_ids=None,
                        sorter_name=DEFAULT_SORTER, sorting_params=None):
    """
    Runs the spike sorting pipeline on a single .dat file.
    Plotting is disabled, progress bars are enabled for all computations.
    All intermediate files are deleted at the end, leaving only phy_export.
    """
    file_path_obj = Path(file_path)
    file_stem = file_path_obj.stem

    # Resolve numeric sorting settings (band-pass cutoffs, detection params),
    # falling back to the built-in defaults for anything not supplied.
    if sorting_params is None:
        sorting_params = resolve_sorting_params(None)
    freq_min = sorting_params.get('freq_min', 600.0)
    freq_max = sorting_params.get('freq_max', 8000.0)
    detect_threshold = sorting_params.get('detect_threshold', 5.0)
    detect_sign = sorting_params.get('detect_sign', 0)

    # 1. SETUP PATHS
    output_parent_obj = Path(output_parent)
    output_dir = output_parent_obj / f"{file_stem}_{sorter_name}_sorting_output"

    # Wipe any leftover artifacts from a previous (possibly failed) run so
    # nothing collides with this attempt.
    if output_dir.exists():
        print(f"Cleaning previous output: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing {file_stem} ---")
    print(f"Output folder: {output_dir}")

    # 2. LOAD DATA
    print("Loading data...")
    raw = readTrodesExtractedDataFile(str(file_path_obj))
    full_traces_raw = raw['data']['voltage']

    # 3. CREATE SPIKEINTERFACE RECORDING
    print("Creating recording object...")
    rec = si.NumpyRecording(traces_list=[full_traces_raw], sampling_frequency=fs)
    rec.set_channel_gains(gain)
    rec.set_channel_offsets(offset)

    # 4. CUSTOM PROBE GEOMETRY (8x4 Tetrode Grid)
    print("Configuring probe geometry...")
    rows, cols = 8, 4
    inter_tetrode_spacing = 250.0 
    diamond_offsets = np.array([[0, 10], [10, 0], [0, -10], [-10, 0]])
    
    all_positions, all_device_indices, group_ids = [], [], []
    tetrode_idx = 0
    
    for r in range(rows):
        for c in range(cols):
            x_center = c * inter_tetrode_spacing
            y_center = r * inter_tetrode_spacing
            for local_idx in range(4):
                all_positions.append([x_center + diamond_offsets[local_idx, 0], y_center + diamond_offsets[local_idx, 1]])
                all_device_indices.append(tetrode_idx * 4 + local_idx)
                group_ids.append(tetrode_idx)
            tetrode_idx += 1

    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=all_positions, shapes='circle', shape_params={'radius': 5})
    probe.set_device_channel_indices(all_device_indices)

    rec = rec.set_probe(probe)
    rec.set_property("group", group_ids)
    print(f"Probe attached. Total Groups: {len(np.unique(group_ids))}")

    # 4.1. Drop EEG tetrode channels — they are not spike-sorted.
    eeg_channel_ids = list(eeg_channel_ids) if eeg_channel_ids else []
    if eeg_channel_ids:
        present = set(rec.get_channel_ids().tolist())
        to_remove = [c for c in eeg_channel_ids if c in present]
        if to_remove:
            n_tetrodes = len(set((c // CH_PER_TETRODE) for c in to_remove))
            print(f"Excluding {len(to_remove)} EEG channel(s) "
                  f"({n_tetrodes} tetrode(s)) from sorting: {sorted(to_remove)}")
            rec = rec.remove_channels(remove_channel_ids=to_remove)
            print(f"Remaining channels: {rec.get_num_channels()}, "
                  f"groups: {len(np.unique(rec.get_property('group')))}")
        # Don't interpolate/reference against channels we just removed.
        excluded = set(eeg_channel_ids)
        bad_channel_ids = [c for c in (bad_channel_ids or []) if c not in excluded]
        ref_channels = [c for c in (ref_channels or []) if c not in excluded]

    # 5. PREPROCESSING & SAVING
    print("Preprocessing and saving binary...")
    
    # 5.1. Bandpass filter
    print(f"Bandpass filter: {freq_min}-{freq_max} Hz")
    rec_filtered = spre.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max)
    
    # 5.2. Bad Channel Detection (per-rat list from hm_tracker_paths.txt)
    bad_channel_ids = list(bad_channel_ids) if bad_channel_ids else []
    ref_channels = list(ref_channels) if ref_channels else []

    # 5.3. Bad Channel Interpolation
    if bad_channel_ids:
        print(f"Interpolating {len(bad_channel_ids)} bad channels (50 µm radius)...")
        rec_interpolated = spre.interpolate_bad_channels(
            rec_filtered,
            bad_channel_ids=bad_channel_ids
        )
    else:
        print("No bad channels configured for this rat — skipping interpolation.")
        rec_interpolated = rec_filtered

    # 5.4. Common Reference
    #   - If ref channel(s) are configured for this rat, first reference
    #     against them ('single').
    #   - Then always apply a global median common average reference.
    if ref_channels:
        print(f"Referencing against channel(s) {ref_channels}, then global median.")
        rec_ref = spre.common_reference(
            rec_interpolated,
            reference='single',
            ref_channel_ids=ref_channels,
        )
    else:
        rec_ref = rec_interpolated
    rec_cmr = spre.common_reference(rec_ref, reference='global', operator='median')

    # 5.5. No manual whitening here — MountainSort does its own whitening
    #      (para['whiten']=True below), so whitening twice would be wrong.
    rec_preprocessed = rec_cmr

    processed_folder = output_dir / 'processed_binary'
    if processed_folder.exists():
        shutil.rmtree(processed_folder)

    rec_saved = rec_preprocessed.save(
        folder=processed_folder, 
        format='binary', 
        overwrite=True, 
        n_jobs=1,  # Keep this at 1 for Windows!
        chunk_duration="1s",
        progress_bar=True
    )

    # 6. SPIKE SORTING (MountainSort4 / MountainSort5, selected via config)
    win_temp = output_dir / "ms4_temp"
    win_temp.mkdir(exist_ok=True)
    os.environ['TEMPDIR'] = str(win_temp)

    para = si.get_default_sorter_params(sorter_name)
    # We've already band-pass filtered above, so let the sorter skip filtering
    # but still do its own whitening. Detect both spike polarities.
    para['filter'] = False
    para['whiten'] = True
    para['detect_sign'] = detect_sign
    para['detect_threshold'] = detect_threshold
    if sorter_name == 'mountainsort5':
        # scheme '2' is MS5's recommended multi-pass clustering scheme.
        para['scheme'] = '2'

    sorter_work_folder = output_dir / 'sorting_work_folder'
    if sorter_work_folder.exists():
        shutil.rmtree(sorter_work_folder)

    print(f"Starting sorting using {sorter_name}...")
    sorting = si.run_sorter_by_property(
        sorter_name=sorter_name,
        recording=rec_saved,
        grouping_property='group',
        folder=sorter_work_folder,
        verbose=True,
        engine="loop",
        engine_kwargs={},
        **para
    )

    final_output_folder = output_dir / 'final_sorting_result'
    if final_output_folder.exists():
        shutil.rmtree(final_output_folder)
    sorting.save(folder=final_output_folder)
    print(f"Sorting complete! Object saved to: {final_output_folder}")

    # 7-10. Analyzer, metrics, quality-check labels, Phy export, cleanup — shared
    #       with continue_sorting.py so both paths stay identical.
    analyze_and_export(
        sorting, rec_saved, output_dir,
        n_jobs=n_jobs, file_stem=file_stem, cleanup=True,
    )


def run_sorting_pipeline(base_data_folder, output_data_folder, n_jobs=4, config=None):
    """
    Scans the base_data_folder for .dat files inside .raw folders
    and processes each one, outputting strictly to output_data_folder.

    `config` is the dict returned by load_sorting_config(); per-recording bad
    channels and reference channels are resolved from it by file stem.
    """
    config = config or {}
    sorter_name = resolve_sorter(config)
    sorting_params = resolve_sorting_params(config)
    print(f"Using sorter: {sorter_name}")
    base_path = Path(base_data_folder)
    output_path = Path(output_data_folder)

    # Ensure the main output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    dat_files = list(base_path.glob("**/*.raw/*_group0.dat"))

    if not dat_files:
        print(f"No .dat files found inside .raw folders under '{base_data_folder}'.")
        return

    print(f"Found {len(dat_files)} recording(s) to process.")

    for i, dat_file in enumerate(dat_files, 1):
        print(f"\n{'='*60}")
        print(f"File {i}/{len(dat_files)}: {dat_file.name}")
        print(f"{'='*60}")

        bad_channel_ids, ref_channels, eeg_channel_ids = resolve_rat_settings(
            dat_file.stem, config)
        try:
            process_single_file(
                file_path=dat_file,
                output_parent=output_path,
                n_jobs=n_jobs,
                bad_channel_ids=bad_channel_ids,
                ref_channels=ref_channels,
                eeg_channel_ids=eeg_channel_ids,
                sorter_name=sorter_name,
                sorting_params=sorting_params,
            )
        except Exception as e:
            print(f"Error processing {dat_file.name}:\n{e}")
            traceback.print_exc()
            print("Skipping to next file...")


# =========================================================
# HOW TO RUN
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tracker Headless Mode")

    # Input and Output arguments
    parser.add_argument('--input_folder', required=True, help="Folder containing .raw folders")
    parser.add_argument('--output_folder', required=True, help="Folder to store all outputs and Phy exports")
    parser.add_argument('--config', default=None,
                        help="Path to hm_tracker_paths.txt for per-rat bad/ref channel settings. "
                             "Defaults to ~/Desktop/hm_tracker_paths.txt if present.")

    args = parser.parse_args()

    config_path = args.config
    if not config_path:
        default_config = Path(os.path.expanduser("~")) / "Desktop" / "hm_tracker_paths.txt"
        if default_config.exists():
            config_path = str(default_config)
    sorting_config = load_sorting_config(config_path)

    try:
        run_sorting_pipeline(args.input_folder, args.output_folder, n_jobs=4, config=sorting_config)
    except Exception as e:
        print(f"\n[FATAL] Pipeline crashed for folder '{args.input_folder}':\n{e}")
        traceback.print_exc()
        print("[FATAL] Exiting with code 0 so the batch runner continues to the next folder.")