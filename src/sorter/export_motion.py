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
        typearr.append((str(fieldname), fieldtype, repeats))
    return np.dtype(typearr)


# ──────────────────────────────────────────────────────────────────────────────
# DISCOVER & LOAD ANALOG (IMU) .dat FILES
# ──────────────────────────────────────────────────────────────────────────────

# Headstage IMU axes we want, in order. motion.npy columns follow this order.
ACCEL_AXES = ('AccelX', 'AccelY', 'AccelZ')


def find_analog_file(input_folder, suffix):
    """
    Find the analogio-exported .dat file whose name ends in <suffix>.dat
    (e.g. 'AccelX'). Searches the Trodes '*.analog' export folder, then the
    input folder itself as a fallback.
    """
    base = Path(input_folder)
    patterns = (f"*.analog/*{suffix}.dat", f"*{suffix}.dat")
    for pat in patterns:
        matches = [f for f in base.glob(pat)
                   if 'timestamps' not in f.name.lower()]
        if matches:
            return sorted(matches)[0]
    return None


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

    columns = []
    lengths = []
    found_axes = []
    src_fs = None
    for axis in ACCEL_AXES:
        dat_file = find_analog_file(input_folder, axis)
        if dat_file is None:
            print(f"  ⚠  {axis} .dat not found — skipping.")
            continue
        values, header = load_channel(dat_file)
        if src_fs is None:
            src_fs = get_source_fs(header)
        print(f"  ✓ {axis}: {dat_file.name}  ({values.shape[0]} samples)")
        columns.append(values)
        lengths.append(values.shape[0])
        found_axes.append(axis)

    if not columns:
        print("❌  No Accel .dat files found. Did Step e / Step 1 export "
              "with -analogio?")
        print("    Expected: <recording>.analog/<recording>.analog_*AccelX.dat")
        return

    # Guard against ragged channels (truncate to the shortest).
    n = min(lengths)
    if len(set(lengths)) > 1:
        print(f"  ⚠  Axis lengths differ {lengths}; truncating to {n}.")

    # Downsample each axis from its native rate to TARGET_FS (1000 Hz).
    print(f"  Downsampling {src_fs:g} Hz → {TARGET_FS} Hz")
    down_cols = [downsample(c[:n], src_fs, TARGET_FS) for c in columns]
    n_down = min(len(c) for c in down_cols)
    motion = np.column_stack([c[:n_down] for c in down_cols]).astype('float32')

    out_file = output_dir / "motion.npy"
    np.save(out_file, motion)
    print(f"  ✓ motion.npy  {motion.shape}  columns={found_axes} @ {TARGET_FS} Hz")

    # Time axis at the downsampled rate, zero-referenced (seconds).
    ts_seconds = (np.arange(n_down) / TARGET_FS).astype('float64')
    np.save(output_dir / "motion_timestamps.npy", ts_seconds)
    print(f"  ✓ motion_timestamps.npy  ({n_down}) @ {TARGET_FS} Hz")

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
