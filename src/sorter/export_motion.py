import re
import argparse
import numpy as np
from pathlib import Path
from fractions import Fraction
from scipy.signal import resample_poly

# Target rate so motion lines up with the 1000 Hz LFP export.
TARGET_FS = 1000


# ──────────────────────────────────────────────────────────────────────────────
# TRODES OFFICIAL READER (from readTrodesExtractedDataFile3.py)
# ──────────────────────────────────────────────────────────────────────────────

def readTrodesExtractedDataFile(filename):
    with open(filename, 'rb') as f:
        if f.readline().decode('ascii').strip() != '<Start settings>':
            raise Exception("Settings format not supported")
        fieldsText = {}
        for line in f:
            line = line.decode('ascii').strip()
            if line != '<End settings>':
                vals = line.split(': ')
                fieldsText.update({vals[0].lower(): vals[1]})
            else:
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
        # Scalar fields use (name, type); (name, type, 1) triggers a numpy
        # FutureWarning and will later be reinterpreted as shape (1,).
        if repeats == 1:
            typearr.append((str(fieldname), fieldtype))
        else:
            typearr.append((str(fieldname), fieldtype, repeats))
    return np.dtype(typearr)


# ──────────────────────────────────────────────────────────────────────────────
# DISCOVER & LOAD ANALOG (IMU) .dat FILES
# ──────────────────────────────────────────────────────────────────────────────

# Headstage IMU axes we want, in order. motion.npy columns follow this order.
ACCEL_AXES = ('AccelX', 'AccelY', 'AccelZ')


def find_analog_roots(input_folder):
    """
    Return the per-session analog export folders in chronological order.

    Each .rec recording exports its analog channels into its own
    '<recording>.analog' folder. When a folder holds several recordings
    (separate sessions), each becomes its own root here. The Trodes name
    embeds YYYYMMDD_HHMMSS, so sorting by name == sorting by time.

    Returns (roots, names). Falls back to the input folder itself (flat
    layout) when no '*.analog' folders exist.
    """
    base = Path(input_folder)
    dirs = sorted((d for d in base.glob("*.analog") if d.is_dir()),
                  key=lambda d: d.name)
    if dirs:
        return [d for d in dirs], [d.stem for d in dirs]
    return [base], [base.name]


def find_axis_file(root, suffix):
    """Find the .dat in <root> whose name ends in <suffix>.dat (e.g. 'AccelX')."""
    matches = [f for f in Path(root).glob(f"*{suffix}.dat")
               if 'timestamps' not in f.name.lower()]
    return sorted(matches)[0] if matches else None


def load_channel(dat_file):
    """Read a single analog .dat and return its 1-D data plus header dict."""
    result = readTrodesExtractedDataFile(str(dat_file))
    data = result['data']
    # Analog channel files store the value in the last field (a leading 'time'
    # field is present in some export configurations).
    field_names = data.dtype.names
    data_field = field_names[-1]
    values = data[data_field].astype('float32').flatten()
    return values, result


def get_source_fs(header):
    """Native sample rate of an analog channel from its header."""
    return float(header.get('samplingrate', header.get('clockrate', 30000)))


def downsample(values, src_fs, target_fs):
    """Anti-aliased resample of a 1-D signal from src_fs to target_fs."""
    if abs(src_fs - target_fs) < 1e-6:
        return values.astype('float32')
    ratio = Fraction(target_fs).limit_denominator() / \
        Fraction(src_fs).limit_denominator()
    up, down = ratio.numerator, ratio.denominator
    return resample_poly(values, up, down).astype('float32')


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(input_folder, output_folder):
    base_path = Path(input_folder)
    # Save alongside the LFP outputs so motion and LFP share one folder.
    output_dir = Path(output_folder) / "LFP_Output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"▶  Motion (IMU) extraction")
    print(f"   Input:  {base_path}")
    print(f"   Output: {output_dir}")

    # Each .rec is a separate recording session with its own analog export
    # folder. Load + downsample every session, then concatenate them in
    # chronological order so motion.npy spans all sessions, matching the
    # concatenated LFP output.
    roots, names = find_analog_roots(input_folder)
    src_fs = None
    sessions_data = []   # [{'name', 'nd', 'axes': {axis: down_array}}]
    boundaries = []
    for root, name in zip(roots, names):
        axis_raw = {}
        for axis in ACCEL_AXES:
            dat_file = find_axis_file(root, axis)
            if dat_file is None:
                continue
            values, header = load_channel(dat_file)
            if src_fs is None:
                src_fs = get_source_fs(header)
            axis_raw[axis] = values
        if not axis_raw:
            if len(roots) > 1:
                print(f"  ⚠  {name}: no Accel .dat found — skipping session.")
            continue

        # Truncate ragged axes within this session, then downsample to 1000 Hz.
        n = min(v.shape[0] for v in axis_raw.values())
        if len({v.shape[0] for v in axis_raw.values()}) > 1:
            print(f"  ⚠  {name}: axis lengths differ; truncating to {n}.")
        down = {ax: downsample(v[:n], src_fs, TARGET_FS)
                for ax, v in axis_raw.items()}
        nd = min(len(v) for v in down.values())
        down = {ax: v[:nd] for ax, v in down.items()}
        for ax in ACCEL_AXES:
            if ax in down:
                print(f"  ✓ {name} {ax}  ({nd} samples @ {TARGET_FS} Hz)")
        sessions_data.append({'name': name, 'nd': nd, 'axes': down})

    if not sessions_data:
        print("❌  No Accel .dat files found. Did Step e / Step 1 export "
              "with -analogio?")
        print("    Expected: <recording>.analog/<recording>.analog_*AccelX.dat")
        return

    # Keep only axes present in every session so columns stay aligned.
    common_axes = set(sessions_data[0]['axes'])
    for sd in sessions_data[1:]:
        common_axes &= set(sd['axes'])
    found_axes = [ax for ax in ACCEL_AXES if ax in common_axes]
    if not found_axes:
        print("❌  No Accel axis is common to all sessions; cannot concatenate.")
        return

    if len(sessions_data) > 1:
        joined = ", ".join(f"{sd['name']} ({sd['nd']})" for sd in sessions_data)
        print(f"  Concatenating {len(sessions_data)} session(s): {joined}")

    cols = [np.concatenate([sd['axes'][ax] for sd in sessions_data])
            for ax in found_axes]
    motion = np.column_stack(cols).astype('float32')
    n_down = motion.shape[0]

    # Per-session sample ranges within the concatenated motion array.
    start = 0
    for sd in sessions_data:
        boundaries.append({'name': sd['name'], 'start': start, 'n': sd['nd']})
        start += sd['nd']

    out_file = output_dir / "motion.npy"
    np.save(out_file, motion)
    print(f"  ✓ motion.npy  {motion.shape}  columns={found_axes} @ {TARGET_FS} Hz")

    # Time axis at the downsampled rate, zero-referenced (seconds).
    ts_seconds = (np.arange(n_down) / TARGET_FS).astype('float64')
    np.save(output_dir / "motion_timestamps.npy", ts_seconds)
    print(f"  ✓ motion_timestamps.npy  ({n_down}) @ {TARGET_FS} Hz")

    np.save(output_dir / "motion_session_boundaries.npy", boundaries)
    if len(boundaries) > 1:
        for b in boundaries:
            print(f"    {b['name']}: samples {b['start']}.."
                  f"{b['start'] + b['n']}  (t0={b['start'] / TARGET_FS:.2f}s)")

    print(f"{'=' * 60}")
    print(f"✅  Motion data → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract headstage IMU Accel channels from Trodes "
                    "-analogio export into motion.npy (n_samples, 3)."
    )
    parser.add_argument('--input_folder', required=True,
                        help="Folder containing Trodes analogio export "
                             "(*.analog/*Accel*.dat)")
    parser.add_argument('--output_folder', required=True,
                        help="Destination for motion.npy.")
    args = parser.parse_args()
    run(args.input_folder, args.output_folder)
