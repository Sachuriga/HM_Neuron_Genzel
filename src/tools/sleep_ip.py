"""Empty a sleep ip folder once everything made from its raw data is in the op folder.

A sleep ip folder (every recording in it is a "pre" or a "post") only holds a
COPY of the .rec, and it is as big as the recording. The later steps do not read
it any more once two things are in the op folder:

  - <op>/*_sorting_output/processed_binary/   written to the end by step 7
        (30 kHz, filtered) -> the sorter itself, Phy and step u read this
  - <op>/*.nwb with LFP and EMG               written by step 8
        -> sleep scoring reads this

So the ip folder is emptied as soon as both are there. That can be in the middle
of step 7: the sorting that follows reads processed_binary/, not the .rec, and a
sort that is stopped is continued from processed_binary/ on the next run.

Maze (task) folders are never touched: their videos and .rec are still needed
(re-track, step 2, ...).

Who calls this:
    sorting.py   right after a processed_binary/ is complete (empty_if_ready)
    runner.py    at the end of a run with step 7 or 8 (empty_sleep_ip_folders)
"""

import json
import re
import shutil
import time
from pathlib import Path

# The same phase rule as the sorter (src/sorter/sorting.py, _PHASE_KEYS). The
# order matters: "post" and the task words are tested before "pre".
_PHASE_KEYS = (("post", "post"), ("maze", "task"), ("mazs", "task"),
               ("awake", "task"), ("hab", "task"), ("pre", "pre"))
_REC_DATE_RE = re.compile(r"(\d{8})_\d{6}")
_RAT_NUMBER_RE = re.compile(r"[Rr]at\s*_?(\d+)")

# Written into the op folder when its ip folder was emptied. processed_binary_tool
# reads it: from then on processed_binary/ is the only 30 kHz copy on this disk,
# so --clean must not delete it. Keep the name the same in both files.
IP_CLEARED_NOTE = "RAW_IP_CLEARED.txt"

# The note step 7 writes into processed_binary/ after the last sample is on disk.
# Keep the name the same as sorting.PROCESSED_BINARY_NOTE.
PROCESSED_BINARY_NOTE = "write_complete.json"

# What step 8 must have written into the session NWB before the .rec may go.
# The motion (IMU) series is not asked for: a session without it is emptied too.
_NWB_SERIES_FROM_RAW = ("lfp", "emg_from_lfp")


def recording_phase(name):
    """'pre', 'post' or 'task' from a recording name; None when it says neither."""
    lower_name = name.lower()
    for key, phase in _PHASE_KEYS:
        if key in lower_name:
            return phase
    return None


# ------------------------------------------------------------
#                 THE CHECKS
# ------------------------------------------------------------
def processed_binary_is_complete(out_dir):
    """True when <out_dir>/processed_binary/ holds the whole recording.

    Two ways to know:
      - its note is there (step 7 writes it only after the last sample), and the
        file on disk still has the size the note says;
      - the sort finished (phy_export/params.py is written at the very end): a
        sort from before the note existed.
    """
    out_dir = Path(out_dir)
    binary_files = list((out_dir / "processed_binary").glob("*.raw"))
    bytes_on_disk = sum(f.stat().st_size for f in binary_files)
    if bytes_on_disk == 0:
        return False

    if (out_dir / "phy_export" / "params.py").is_file():
        return True

    note_file = out_dir / "processed_binary" / PROCESSED_BINARY_NOTE
    if not note_file.is_file():
        return False
    try:
        note = json.loads(note_file.read_text())
    except (OSError, ValueError):
        return False
    # A note from the first days has no num_bytes; the note itself then counts.
    expected_bytes = note.get("num_bytes")
    return expected_bytes is None or expected_bytes == bytes_on_disk


def _has_complete_binary(op, rec_file):
    """True when the op folder holds a complete processed_binary/ of this .rec."""
    # the sorter names the folder <recording>.raw_group0_<sorter>_sorting_output
    for out_dir in Path(op).glob(f"{rec_file.stem}.raw_group0_*_sorting_output"):
        if processed_binary_is_complete(out_dir):
            return True
    return False


def _nwb_problems(op):
    """Why the session NWB of `op` does not yet hold everything step 8 makes from
    the raw data. An empty list means it is all there."""
    try:
        import h5py
    except ImportError:
        return ["h5py is not installed, so the session NWB could not be checked"]
    nwb_files = [p for p in sorted(Path(op).glob("*.nwb"))
                 if not p.name.endswith(".tmp.nwb") and not p.name.startswith("._")]
    if not nwb_files:
        return ["no session NWB in the op folder (run step 8)"]
    try:
        with h5py.File(str(nwb_files[0]), "r") as f:
            missing = []
            for name in _NWB_SERIES_FROM_RAW:
                data_path = f"acquisition/{name}/data"
                if data_path not in f or f[data_path].shape[0] == 0:
                    missing.append(name)
    except OSError as e:
        return [f"the session NWB could not be read ({e})"]
    if missing:
        return [f"{nwb_files[0].name} has no {' / '.join(missing)} yet (run step 8)"]
    return []


def sleep_ip_problems(ip, op):
    """Decide whether a sleep ip folder may be emptied.

    Returns None when `ip` is not a sleep folder (no .rec, or a recording that is
    not pre/post) — such a folder is simply left alone. Otherwise returns the list
    of reasons to KEEP it; an empty list means it is safe to empty.
    """
    ip, op = Path(ip), Path(op)
    rec_files = sorted(ip.rglob("*.rec"))     # the sorter also looks in subfolders
    if not rec_files:
        return None
    phases = [recording_phase(rec.stem) for rec in rec_files]
    if any(phase not in ("pre", "post") for phase in phases):
        return None

    # Split parts of one phase (..._post + ..._post2, after a crash of the
    # acquisition PC) are sorted as ONE recording, into the output folder of the
    # first part. So the check is per (rat, date, phase) group, not per file.
    groups = {}
    for rec, phase in zip(rec_files, phases):
        rat = _RAT_NUMBER_RE.search(rec.stem)
        date = _REC_DATE_RE.search(rec.stem)
        key = (rat.group(1) if rat else None, date.group(1) if date else None, phase)
        groups.setdefault(key, []).append(rec)

    problems = []
    for parts in groups.values():
        if not any(_has_complete_binary(op, rec) for rec in parts):
            problems.append(f"{parts[0].name} has no complete processed_binary/ "
                            f"in the op folder yet (run step 7)")
    problems += _nwb_problems(op)
    return problems


# ------------------------------------------------------------
#                 THE DELETING
# ------------------------------------------------------------
def empty_sleep_ip_folder(ip, op):
    """Delete everything inside `ip` (the folder itself stays, so ipN still pairs
    with opN) and leave a note in `op`. Returns True when nothing is left."""
    ip, op = Path(ip), Path(op)
    rec_names = [rec.name for rec in sorted(ip.rglob("*.rec"))]
    for item in sorted(ip.iterdir()):
        try:
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
        except OSError as e:
            # e.g. a file still open in another program (Windows)
            print(f"    could not delete {item.name}: {e}")
    left = [item.name for item in ip.iterdir()]
    if left:
        print(f"    NOT empty yet, still there: {', '.join(left)}")
    if not any(ip.rglob("*.rec")):
        (op / IP_CLEARED_NOTE).write_text(
            f"The sleep ip folder of this session ({ip}) was emptied on "
            f"{time.strftime('%Y-%m-%d %H:%M')}, after processed_binary/ was complete "
            f"and the LFP was in the NWB.\n"
            f"Deleted recordings: {', '.join(rec_names)}\n\n"
            "On this disk the 30 kHz data now only exists as\n"
            "  <recording>_sorting_output/processed_binary/   (band-passed + re-referenced)\n"
            "so processed_binary_tool.py --clean does not delete it while this note is here.\n"
            "A sort that was not finished is continued from it by step 7.\n"
            "The unfiltered data (new LFP, other filter settings, a new sort from raw)\n"
            "needs the original .rec from the recording drive.\n")
    return not left


def empty_if_ready(ip, op):
    """Empty ONE ip folder when it is a sleep folder that passes the checks.

    Returns True when it was emptied. A folder that is kept gets one line per
    reason, so the log says what is still missing.
    """
    problems = sleep_ip_problems(ip, op)
    if problems is None:
        return False                     # a maze folder, or nothing to delete
    if problems:
        print(f"[SLEEP-IP] KEPT {ip}:")
        for reason in problems:
            print(f"    - {reason}")
        return False
    print(f"[SLEEP-IP] Emptying {ip} (its data is in {op})")
    emptied = empty_sleep_ip_folder(ip, op)
    if emptied:
        print(f"[SLEEP-IP] Emptied {ip}")
    return emptied


def empty_sleep_ip_folders(ops):
    """End of a run with step 7 or 8: empty every sleep ip folder that passes
    sleep_ip_problems(); say why for the ones that are kept."""
    print("\n" + "=" * 56)
    print("[MASTER] Sleep ip folders (pre/post): emptying the ones that have a complete "
          "processed_binary/ and their LFP in the NWB...")
    print("=" * 56)
    for ip, op in ops:
        empty_if_ready(ip, op)
