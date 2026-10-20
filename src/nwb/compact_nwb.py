"""
Give back the disk space that re-runs leave inside a session NWB.

WHY THIS EXISTS
---------------
Several steps replace part of the session NWB when they run again:
step u (the Units table, ~0.2 GB with its waveforms), step 4 (behaviour) and
step 8 (the LFP, several GB). They delete the old part and write a new one.
HDF5 files never shrink by themselves: the space of the deleted part stays in
the file, unused. After a few re-runs the NWB can be much bigger than its
content. This script writes a fresh, compact copy and swaps it in.

HOW IT WORKS
------------
1. Copy the NWB to <name>.compact.tmp.nwb, using
     - `h5repack` (the official HDF5 tool) when it is installed — it copies
       every object exactly, even ones pynwb does not know about;
     - otherwise pynwb's export, which re-writes everything pynwb can read.
2. Check that EVERY object of the original also exists in the copy.
3. Only then replace the original with the copy. If anything looks wrong, the
   copy is deleted and the original is left untouched.

USAGE
-----
    python src/nwb/compact_nwb.py <op folder or .nwb file>
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def human_size(n_bytes):
    """12345678 -> '11.8 MiB'."""
    size = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def all_object_paths(nwb_path):
    """Every group and dataset path inside an HDF5 file, as a set of strings."""
    import h5py

    paths = set()
    with h5py.File(str(nwb_path), "r") as f:
        # visit() calls the function once for every object below the root.
        f.visit(paths.add)
    return paths


def copy_with_h5repack(source, target):
    """Copy with the HDF5 command-line tool. Returns True on success."""
    h5repack = shutil.which("h5repack")
    if h5repack is None:
        return False
    print("  copying with h5repack ...")
    result = subprocess.run([h5repack, str(source), str(target)])
    return result.returncode == 0


def copy_with_pynwb_export(source, target):
    """Copy by reading the NWB with pynwb and exporting it to a new file."""
    from pynwb import NWBHDF5IO

    print("  copying with pynwb export (h5repack not found) ...")
    with NWBHDF5IO(str(source), mode="r", load_namespaces=True) as source_io:
        nwbfile = source_io.read()
        with NWBHDF5IO(str(target), mode="w") as target_io:
            target_io.export(src_io=source_io, nwbfile=nwbfile)
    return True


def compact(nwb_path):
    """Replace `nwb_path` by a compact copy. Returns the bytes freed (0 if none)."""
    nwb_path = Path(nwb_path)
    temp_copy = nwb_path.with_name(nwb_path.stem + ".compact.tmp.nwb")
    size_before = nwb_path.stat().st_size

    # The copy needs as much free space as the original at most.
    free_space = shutil.disk_usage(nwb_path.parent).free
    if free_space < size_before:
        print(f"  not enough free space for a temporary copy "
              f"({human_size(free_space)} free, {human_size(size_before)} needed) — skipped.")
        return 0

    try:
        if not copy_with_h5repack(nwb_path, temp_copy):
            copy_with_pynwb_export(nwb_path, temp_copy)

        # Safety check: nothing may go missing in the copy.
        missing = all_object_paths(nwb_path) - all_object_paths(temp_copy)
        if missing:
            print(f"  STOP: the copy lacks {len(missing)} object(s), e.g. "
                  f"{sorted(missing)[:3]} — the original is kept unchanged.")
            temp_copy.unlink(missing_ok=True)
            return 0

        size_after = temp_copy.stat().st_size
        if size_after >= size_before:
            print("  already compact — the original is kept.")
            temp_copy.unlink(missing_ok=True)
            return 0

        os.replace(temp_copy, nwb_path)      # atomic swap on the same drive
        freed = size_before - size_after
        print(f"  {human_size(size_before)} -> {human_size(size_after)} "
              f"(freed {human_size(freed)})")
        return freed
    except BaseException:
        # Whatever went wrong (including Ctrl-C), never leave a half copy behind.
        temp_copy.unlink(missing_ok=True)
        raise


def find_nwb_files(target):
    """The NWB file itself, or the session NWB(s) directly inside a folder."""
    target = Path(target)
    if target.is_file():
        return [target]
    return [p for p in sorted(target.glob("*.nwb"))
            if not p.name.endswith(".tmp.nwb") and not p.name.startswith("._")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rewrite session NWB files compactly, giving back the space "
                    "left by deleted (replaced) parts.")
    parser.add_argument("target", help="An op folder, or one .nwb file.")
    args = parser.parse_args()

    files = find_nwb_files(args.target)
    if not files:
        print(f"No .nwb file found at {args.target}.")
    total_freed = 0
    for nwb_file in files:
        print(f"{nwb_file.name}:")
        total_freed += compact(nwb_file)
    if len(files) > 1:
        print(f"Total freed: {human_size(total_freed)}")
