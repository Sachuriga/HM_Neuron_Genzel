"""
Free disk space by deleting processed_binary/ — and rebuild it when needed.

WHY THIS EXISTS
---------------
Every sorted recording keeps <op>/<recording>_<sorter>_sorting_output/processed_binary/:
the band-passed, re-referenced copy of the raw data that Phy's trace view
and step u read. It is as big as the raw data itself (~27 GiB per
recording hour), and it is only needed while you are still curating.

It can be deleted safely because it is NOT original data: it is rebuilt from
the .rec with exactly the code sorting.py used (sorting.build_preprocessed_recording),
which gives a byte-identical file as long as the per-rat channel settings and
FREQ_MIN / FREQ_MAX in hm_tracker_paths.txt have not changed since sorting.

WHAT IT DOES
------------
  --clean    For every sorting output whose units are ALREADY in the session NWB
             (step u done), delete processed_binary/. Before deleting, a small
             fingerprint of the file is saved, so --restore can check that the
             rebuilt file is identical. Dry run unless you add --yes.

  --restore  Rebuild processed_binary/ from the .rec files in --input_folder for
             every sorting output that is missing it, then compare it with the
             saved fingerprint.

USAGE
-----
    # 1. see what would be deleted and how much space it frees (nothing is deleted)
    python src/sorter/processed_binary_tool.py --clean --output_folder <op>

    # 2. really delete
    python src/sorter/processed_binary_tool.py --clean --output_folder <op> --yes

    # 3. later, to curate again / re-run step u: rebuild from the .rec
    python src/sorter/processed_binary_tool.py --restore --input_folder <ip> \
        --output_folder <op> --config ~/Desktop/hm_tracker_paths.txt

    # A WHOLE TREE: every <...>_sorting_output/processed_binary below a root, e.g.
    # the share (.../task/rat5_491390/20260623/<recording>_sorting_output/). The
    # same rules per session; first without --yes to see the list and the total.
    python src/sorter/processed_binary_tool.py --clean --root <root, e.g. the share's Rat_HM_Neuron>
    python src/sorter/processed_binary_tool.py --clean --root <root> --yes
    #   --force  also where the units are not in the session NWB yet
"""

import sys
import json
import shutil
import hashlib
import argparse
from pathlib import Path

# This file lives in src/sorter; the session-NWB helpers live in src/nwb.
SORTER_DIR = Path(__file__).resolve().parent
NWB_DIR = SORTER_DIR.parent / "nwb"
sys.path.insert(0, str(SORTER_DIR))
if str(NWB_DIR) not in sys.path:
    sys.path.append(str(NWB_DIR))    # append: src/nwb has its own session_prefix.py

FINGERPRINT_FILE = "processed_binary_fingerprint.json"
REMOVED_NOTE_FILE = "processed_binary_REMOVED.txt"


# ---------------------------------------------------------------------------
#   Small helpers
# ---------------------------------------------------------------------------
def human_size(n_bytes):
    """12345678 -> '11.8 MiB'."""
    size = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def folder_size(folder):
    """Total size in bytes of all files inside `folder`."""
    total = 0
    for item in Path(folder).rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def find_sorting_outputs(output_folder):
    """All <recording>_<sorter>_sorting_output folders directly in the op folder."""
    return sorted(Path(output_folder).glob("*_sorting_output"))


def read_phy_params(phy_folder):
    """Phy's params.py (plain `key = value` lines) as a dict."""
    namespace = {}
    exec((Path(phy_folder) / "params.py").read_text(), {}, namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__")}


def traces_file_name(phy_folder):
    """File name of the recording params.py points at, e.g. 'traces_cached_seg0.raw'."""
    dat_path = read_phy_params(phy_folder)["dat_path"]
    if isinstance(dat_path, (list, tuple)):
        dat_path = dat_path[0]
    # dat_path may be a Windows path written on the sorting PC.
    return str(dat_path).replace("\\", "/").split("/")[-1]


def file_fingerprint(path, block_bytes=16 * 2**20):
    """Size + SHA-256 of three 16 MiB blocks (start, middle, end) of a file.

    Hashing a 100+ GB file completely would take a long time; three blocks are
    enough to notice a rebuild that came out different (other bad channels,
    other filter settings, ...), because such a change touches every block.
    """
    path = Path(path)
    size = path.stat().st_size
    block_starts = [0, max(0, size // 2 - block_bytes // 2), max(0, size - block_bytes)]
    hashes = []
    with open(path, "rb") as f:
        for start in block_starts:
            f.seek(start)
            hashes.append(hashlib.sha256(f.read(block_bytes)).hexdigest())
    return {"size_bytes": size, "block_bytes": block_bytes, "block_sha256": hashes}


def sorting_groups_in_nwb(output_folder):
    """Names of the *_sorting_output folders whose units are in the session NWB.

    Step u writes the source folder of every unit into the Units column
    `sorting_group`. Returns an empty set when there is no NWB or no units.
    """
    import h5py
    import sleep_nwb_writer as snwb

    nwb_path = snwb.find_session_nwb(output_folder)
    if nwb_path is None:
        return set()
    with h5py.File(str(nwb_path), "r") as f:
        if "units/sorting_group" not in f:
            return set()
        values = f["units/sorting_group"][:]
    # h5py returns stored strings as bytes; turn them into normal str.
    names = set()
    for value in values:
        if isinstance(value, bytes):
            names.add(value.decode())
        else:
            names.add(str(value))
    return names


# ---------------------------------------------------------------------------
#   --clean
# ---------------------------------------------------------------------------
def clean(output_folder, really_delete=False, force=False):
    """Delete processed_binary/ of sorting outputs whose units are in the NWB.

    `force` also deletes it where the units are NOT in the session NWB yet (the
    NWB is then not opened at all). Returns the counts of this folder:
    {"bytes", "deleted", "kept", "failed"} — "bytes" is what was freed, or, in a
    dry run, what would be."""
    counts = {"bytes": 0, "deleted": 0, "kept": 0, "failed": 0}
    outputs = find_sorting_outputs(output_folder)
    if not outputs:
        print(f"No *_sorting_output folders in {output_folder}.")
        return counts

    in_nwb = set()
    if not force:
        try:
            in_nwb = sorting_groups_in_nwb(output_folder)
        except OSError as e:
            # an NWB that cannot be read (in use, damaged) says nothing: keep all
            print(f"  could not read the session NWB of {output_folder} ({e}).")
    total_freed = 0

    for out_dir in outputs:
        processed = out_dir / "processed_binary"
        phy = out_dir / "phy_export"
        print(f"\n{out_dir.name}")

        # Each check below protects data you might still need.
        if not processed.exists():
            print("  no processed_binary/ (already removed) — nothing to do.")
            continue
        if not (phy / "params.py").exists():
            print("  no phy_export/ — the sort is not finished; keeping processed_binary/.")
            counts["kept"] += 1
            continue
        if not force and out_dir.name not in in_nwb:
            print("  its units are not in the session NWB yet (run step u first) "
                  "— keeping processed_binary/ (--force deletes it anyway).")
            counts["kept"] += 1
            continue

        deleting = False
        try:
            traces_file = processed / traces_file_name(phy)
            if not traces_file.exists():
                print(f"  {traces_file.name} (named in params.py) is missing — keeping the folder.")
                counts["kept"] += 1
                continue

            size = folder_size(processed)
            if not really_delete:
                print(f"  would delete processed_binary/ ({human_size(size)}). "
                      f"Add --yes to delete.")
                total_freed += size
                counts["deleted"] += 1
                continue

            # Save the fingerprint BEFORE deleting, so --restore can verify the rebuild.
            print("  saving a fingerprint of the file (reads 48 MiB)...")
            fingerprint = file_fingerprint(traces_file)
            fingerprint["traces_file"] = traces_file.name
            (out_dir / FINGERPRINT_FILE).write_text(json.dumps(fingerprint, indent=2))

            deleting = True
            shutil.rmtree(processed)
        except Exception as e:
            # One folder that cannot be read or removed (a file open in Phy, the
            # share dropping out) must not stop the others. If the removal had
            # begun it may be half done: say so, and leave no note that claims a
            # clean removal.
            print(f"  ERROR: {type(e).__name__}: {e}")
            print(f"  processed_binary/ was NOT removed completely — check {processed}"
                  if deleting else "  processed_binary/ is left as it is.")
            counts["failed"] += 1
            continue
        (out_dir / REMOVED_NOTE_FILE).write_text(
            "processed_binary/ was deleted to save disk space. Phy's trace view and\n"
            "step u need it. Rebuild it from the .rec with:\n\n"
            "  python src/sorter/processed_binary_tool.py --restore "
            "--input_folder <ip folder> --output_folder <this op folder> --config <hm_tracker_paths.txt>\n")
        print(f"  deleted processed_binary/ — freed {human_size(size)}.")
        total_freed += size
        counts["deleted"] += 1

    verb = "freed" if really_delete else "would free"
    print(f"\nTotal {verb}: {human_size(total_freed)}")
    counts["bytes"] = total_freed
    return counts


# ---------------------------------------------------------------------------
#   --clean --root : every session below a root folder
# ---------------------------------------------------------------------------
# Folders that are never searched: they hold the files of a sort or of a pipeline
# step, not session folders. A sorting output alone is thousands of files, and
# listing them on the share is what would make the search slow.
_NOT_SEARCHED_NAMES = {"lfp_output", "visualization", "decoding", "videos", "umap",
                       "predictive_coding", "clock_qc", "clock_repair_backup",
                       "lfp_fix_backup", "channels_npy", "figures", "__pycache__"}
_NOT_SEARCHED_SUFFIXES = (".raw", ".dio", ".lfp", ".rec", ".analog", ".time",
                          ".spikes", ".mountainsort")


def find_session_folders_with_sorting(root, max_depth=8):
    """Every folder at or below `root` that directly holds a *_sorting_output
    folder (an op / session folder), sorted. A sorting output itself is never
    entered, so only the folders above it are listed."""
    import os

    found = []
    todo = [(Path(root), 0)]
    while todo:
        folder, depth = todo.pop()
        try:
            with os.scandir(folder) as entries:
                subdirs = [e.name for e in entries if e.is_dir(follow_symlinks=False)]
        except OSError as e:
            print(f"cannot list {folder}: {e}")
            continue
        holds_sorting = False
        for name in subdirs:
            low = name.lower()
            if low.endswith("_sorting_output"):
                holds_sorting = True
            elif (name.startswith((".", "~")) or "_sorting_output" in low   # .backup_<date> too
                    or low in _NOT_SEARCHED_NAMES or low.endswith(_NOT_SEARCHED_SUFFIXES)):
                continue
            elif depth < max_depth:
                todo.append((folder / name, depth + 1))
        if holds_sorting:
            found.append(folder)
    return sorted(found)


def clean_tree(root, really_delete=False, force=False, max_depth=8):
    """Delete processed_binary/ in every *_sorting_output folder below `root`, e.g.
    <root>/task/rat5_491390/20260623/<recording>_mountainsort4_sorting_output/.

    The same rules and the same fingerprint as clean(), one session folder after
    the other; a dry run unless `really_delete`. Returns the summed counts."""
    root = Path(root)
    if not root.is_dir():
        print(f"Root folder not found: {root}")
        return {"bytes": 0, "deleted": 0, "kept": 0, "failed": 0, "sessions": 0}
    print(f"Looking for *_sorting_output folders below {root} "
          f"(up to {max_depth} levels deep)...")
    sessions = find_session_folders_with_sorting(root, max_depth)
    print(f"{len(sessions)} session folder(s) with a sorting output.")

    total = {"bytes": 0, "deleted": 0, "kept": 0, "failed": 0, "sessions": len(sessions)}
    for i, session in enumerate(sessions, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(sessions)}] {session}")
        counts = clean(session, really_delete=really_delete, force=force)
        for key in ("bytes", "deleted", "kept", "failed"):
            total[key] += counts[key]

    print(f"\n{'=' * 60}")
    if really_delete:
        print(f"Deleted {total['deleted']} processed_binary folder(s) below {root}: "
              f"freed {human_size(total['bytes'])}.")
    else:
        print(f"DRY RUN — nothing was deleted. {total['deleted']} processed_binary folder(s) "
              f"below {root} would be deleted: {human_size(total['bytes'])}. Add --yes to delete.")
    if total["kept"]:
        print(f"{total['kept']} kept (the reason is given above"
              + ("" if force else "; --force also deletes those whose units are not in the NWB yet")
              + ").")
    if total["failed"]:
        print(f"{total['failed']} could NOT be removed completely — see the ERROR lines above.")
    return total


# ---------------------------------------------------------------------------
#   --restore
# ---------------------------------------------------------------------------
def find_recordings_of(out_dir, input_folder):
    """(recording files, sorter name) that produced this sorting output.

    The output folder is named after the FIRST part of a phase; a phase that
    was split over several .rec files was sorted as one, so all its parts are
    returned, in the same order sorting.py used.
    """
    import sorting

    rec_files = sorting.find_rec_files(input_folder, recursive=True)
    have_rec = {sorting._recording_name(f) for f in rec_files}
    old_dats = [f for f in Path(input_folder).glob("**/*.raw/*_group0.dat")
                if sorting._recording_name(f) not in have_rec]
    groups = sorting.group_split_recordings(rec_files + old_dats)

    for group in groups:
        for sorter_name in sorting.SUPPORTED_SORTERS:
            candidate = sorting.sorting_output_dir(group[0], out_dir.parent, sorter_name)
            if candidate.name == out_dir.name:
                return group, sorter_name
    return None, None


def restore(input_folder, output_folder, config_path=None):
    """Rebuild missing processed_binary/ folders from the .rec files."""
    import sorting

    config = sorting.load_sorting_config(config_path) if config_path else {}
    sorting_params = sorting.resolve_sorting_params(config)

    for out_dir in find_sorting_outputs(output_folder):
        processed = out_dir / "processed_binary"
        phy = out_dir / "phy_export"
        if processed.exists() or not (phy / "params.py").exists():
            continue                       # nothing missing, or not a finished sort

        print(f"\n{out_dir.name}: rebuilding processed_binary/ ...")
        recordings, sorter_name = find_recordings_of(out_dir, input_folder)
        if recordings is None:
            print(f"  could not find the .rec for this output in {input_folder} — skipped.")
            continue

        bad, ref, eeg = sorting.resolve_rat_settings(recordings[0].stem, config)
        rec = sorting.build_preprocessed_recording(
            recordings, bad_channel_ids=bad, ref_channels=ref, eeg_channel_ids=eeg,
            freq_min=sorting_params["freq_min"], freq_max=sorting_params["freq_max"])

        # Phy's spike files assume this exact channel count. A different count
        # means the EEG tetrodes in the config changed since sorting.
        expected_channels = int(read_phy_params(phy)["n_channels_dat"])
        if rec.get_num_channels() != expected_channels:
            print(f"  STOP: the rebuild has {rec.get_num_channels()} channels, Phy expects "
                  f"{expected_channels}. EEG_TETRODES for this rat changed since sorting; "
                  f"put the old setting back and try again.")
            continue

        rec.save(folder=processed, **sorting.PROCESSED_BINARY_SAVE_KWARGS)
        _check_against_fingerprint(out_dir, processed, phy)
        note = out_dir / REMOVED_NOTE_FILE
        if note.exists():
            note.unlink()


def _check_against_fingerprint(out_dir, processed, phy):
    """Compare the rebuilt file with the fingerprint --clean saved."""
    fingerprint_path = out_dir / FINGERPRINT_FILE
    traces_file = processed / traces_file_name(phy)
    if not traces_file.exists():
        print(f"  WARNING: rebuilt folder has no {traces_file.name}, which params.py expects.")
        return
    if not fingerprint_path.exists():
        print("  rebuilt (no fingerprint was saved, so it cannot be compared).")
        return

    saved = json.loads(fingerprint_path.read_text())
    rebuilt = file_fingerprint(traces_file, block_bytes=saved["block_bytes"])
    same = (rebuilt["size_bytes"] == saved["size_bytes"]
            and rebuilt["block_sha256"] == saved["block_sha256"])
    if same:
        print("  rebuilt — identical to the deleted original (fingerprint matches).")
    else:
        print("  WARNING: rebuilt, but it is NOT identical to the original. Spike times "
              "in Phy are still valid; waveforms may differ slightly. Did BAD_CHANNELS / "
              "REF_CHANNEL / FREQ_MIN / FREQ_MAX change since sorting?")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete processed_binary/ to free disk space, or rebuild it from the .rec.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--clean", action="store_true",
                        help="Delete processed_binary/ where the units are already in the NWB.")
    action.add_argument("--restore", action="store_true",
                        help="Rebuild missing processed_binary/ folders from the .rec files.")
    parser.add_argument("--output_folder", help="The op folder.")
    parser.add_argument("--root",
                        help="With --clean, instead of --output_folder: a folder above the "
                             "sessions, e.g. the share's Rat_HM_Neuron. Every "
                             "*_sorting_output/processed_binary below it is handled.")
    parser.add_argument("--depth", type=int, default=8,
                        help="With --root: how many folder levels below it are searched (default 8).")
    parser.add_argument("--force", action="store_true",
                        help="With --clean: also delete processed_binary/ where the units are "
                             "not in the session NWB yet (step u will then need --restore first).")
    parser.add_argument("--input_folder", help="The ip folder with the .rec files (--restore).")
    parser.add_argument("--config", default=None,
                        help="hm_tracker_paths.txt with the per-rat channel settings (--restore).")
    parser.add_argument("--yes", action="store_true",
                        help="With --clean: really delete (without it, only report).")
    args = parser.parse_args()

    if args.clean:
        if bool(args.root) == bool(args.output_folder):
            parser.error("--clean needs --output_folder (one op folder) or --root "
                         "(every session below a folder), not both.")
        if args.root:
            done = clean_tree(args.root, really_delete=args.yes, force=args.force,
                              max_depth=args.depth)
        else:
            done = clean(args.output_folder, really_delete=args.yes, force=args.force)
        sys.exit(1 if done["failed"] else 0)
    else:
        if args.root or not args.output_folder:
            parser.error("--restore works on one op folder: give --output_folder.")
        if not args.input_folder:
            parser.error("--restore needs --input_folder (the folder with the .rec files).")
        config_path = args.config
        default_config = Path.home() / "Desktop" / "hm_tracker_paths.txt"
        if config_path is None and default_config.exists():
            config_path = str(default_config)
        restore(args.input_folder, args.output_folder, config_path)
