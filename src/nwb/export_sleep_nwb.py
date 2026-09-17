"""STEP 8 — backfill the sleep-scoring inputs into the session NWB.

``export_lfp.py`` now writes the session NWB itself (and the EMG/motion exports
append to it), so for a freshly exported session this script finds everything
already in place and does nothing.

It stays in the pipeline for **legacy sessions** — those exported before the
NWB became the primary output, which only have the loose ``.npy`` files. For
those it creates ``<op>/<Rat>_<YYYYMMDD>.nwb`` (the same name ``create_nwb.py``
derives, so step w and step u later append to this very file) and fills it from
the ``.npy`` with everything ``HM_rat_sleep_score`` needs:

    acquisition/lfp             from  LFP_Output/*lfp_data.npy + *lfp_timestamps.npy
    acquisition/emg_from_lfp    from  LFP_Output/*emg_from_lfp_5hz.npy (+ its timestamps)
    acquisition/motion          from  LFP_Output/*motion_accel.npy (+ *motion_timestamps.npy)
    processing/sleep/sleep_channels   from  LFP_Output/*sleep_channels.npy

The ``.npy`` files stay where they are — this only adds a packaged copy, so
nothing downstream that still reads them breaks.

Re-running is safe: containers that already exist are left untouched, and an
NWB that step w/u already populated is appended to rather than replaced, so a
scoring stored in it is never lost.

Usage:
    python export_sleep_nwb.py --input_folder <ip> --output_folder <op>
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleep_nwb as snwb                                    # noqa: E402

LFP_ROWS_PER_CHUNK = 250_000        # ~1 s of writing per chunk at 1500 Hz x 64 ch


def find_output(folder, suffix):
    """The prefixed (or bare) ``<prefix><suffix>`` file in ``folder``, else None.

    Matches on the prefix boundary so ``lfp_timestamps.npy`` never resolves to
    ``..._emg_from_lfp_timestamps.npy``.
    """
    folder = Path(folder)
    exact = folder / suffix
    if exact.is_file():
        return exact
    for p in sorted(folder.glob(f"*{suffix}")):
        head = p.name[: -len(suffix)]
        if re.fullmatch(r"[A-Za-z]+\d+_\d{8}_\d{6}_", head):
            return p
    return None


def output_prefix(folder):
    """``'Rat6_20260629_143022_'`` from the LFP_Output folder's own files."""
    for suffix in ("lfp_data.npy", "lfp_timestamps.npy", "channel_map.npy"):
        p = find_output(folder, suffix)
        if p is not None:
            return p.name[: -len(suffix)]
    return ""


def session_start_time(prefix):
    """``Rat6_20260629_143022_`` -> an aware datetime for the NWB header."""
    m = re.search(r"(\d{8})_(\d{6})", prefix or "")
    if not m:
        return datetime.now(timezone.utc)
    stamp = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return stamp.replace(tzinfo=datetime.now().astimezone().tzinfo)


def _load(path, mmap=False):
    if path is None:
        return None
    try:
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=not mmap)
    except Exception as exc:
        print(f"  [warn] could not read {Path(path).name}: {exc}")
        return None


def _lfp_chunks(arr, buffer_gb=0.25):
    """Stream a big ``(n_samples, n_channels)`` memmap into HDF5 block by block,
    so a multi-GB LFP never has to sit in RAM at once."""
    from hdmf.data_utils import GenericDataChunkIterator

    class _ArrayChunks(GenericDataChunkIterator):
        def _get_data(self, selection):
            return np.asarray(arr[selection], dtype=np.float32)

        def _get_maxshape(self):
            return tuple(int(n) for n in arr.shape)

        def _get_dtype(self):
            return np.dtype("float32")

    return _ArrayChunks(buffer_gb=buffer_gb)


def collect_inputs(lfp_dir):
    """Gather the sleep-scoring arrays from an ``LFP_Output`` folder."""
    lfp_file = find_output(lfp_dir, "lfp_data.npy")
    lfp = _load(lfp_file, mmap=True)
    ts = _load(find_output(lfp_dir, "lfp_timestamps.npy"), mmap=True)

    emg = _load(find_output(lfp_dir, "emg_from_lfp_5hz.npy"))
    emg_ts = _load(find_output(lfp_dir, "emg_from_lfp_timestamps.npy"))
    if emg is None:                       # fall back to the per-sample version
        emg = _load(find_output(lfp_dir, "emg_from_lfp.npy"), mmap=True)
        emg_ts = ts

    motion = _load(find_output(lfp_dir, "motion_accel.npy"), mmap=True)
    motion_ts = _load(find_output(lfp_dir, "motion_timestamps.npy"), mmap=True)

    channels = _load(find_output(lfp_dir, "sleep_channels.npy"))
    if channels is not None:
        try:
            channels = channels.item()
        except Exception:
            channels = None

    return {"lfp": lfp, "lfp_timestamps": ts, "emg": emg, "emg_timestamps": emg_ts,
            "motion": motion, "motion_timestamps": motion_ts,
            "sleep_channels": channels}


def run(output_folder, rat_nr=None):
    from pynwb import NWBHDF5IO, NWBFile

    op = Path(output_folder)
    lfp_dir = op / "LFP_Output"
    if not lfp_dir.is_dir():
        print(f"[sleep-nwb] No LFP_Output in {op} — run the LFP export first. Skipping.")
        return 0

    # The LFP export writes the NWB itself now; this script is the backfill for
    # legacy sessions that only ever produced .npy. Nothing to do when the
    # session NWB already carries the LFP.
    existing = snwb.find_session_nwb(op)
    if existing is not None:
        inputs = None
        try:
            inputs = snwb.read_sleep_inputs(existing)
            if inputs.get("lfp") is not None:
                have = [n for n in ("emg", "motion") if inputs.get(n) is not None]
                print(f"[sleep-nwb] {existing.name} already holds the LFP "
                      f"({'+'.join(['lfp'] + have)}) — nothing to backfill.")
                return 0
        except Exception as exc:
            print(f"[sleep-nwb] Could not inspect {existing.name}: {exc}")
        finally:
            snwb.close_inputs(inputs)

    prefix = output_prefix(lfp_dir)
    if not prefix:
        print(f"[sleep-nwb] No prefixed LFP files in {lfp_dir}; skipping.")
        return 0

    data = collect_inputs(lfp_dir)
    if data["lfp"] is None:
        print(f"[sleep-nwb] No lfp_data.npy in {lfp_dir} and no LFP in the NWB; "
              f"skipping.")
        return 0

    nwb_path = snwb.find_session_nwb(op) or (op / snwb.session_nwb_name(prefix, op))
    print(f"[sleep-nwb] Session: {prefix.rstrip('_')}")
    print(f"[sleep-nwb] Target:  {nwb_path}")

    # Reconcile LFP length against its timebase HERE: once wrapped in a chunk
    # iterator the array is opaque and can no longer be trimmed downstream.
    lfp = data.pop("lfp")
    ts = data["lfp_timestamps"]
    if ts is not None:
        ts = np.asarray(ts, dtype=float).ravel()
        if ts.shape[0] != lfp.shape[0]:
            n = min(ts.shape[0], lfp.shape[0])
            print(f"  [warn] lfp ({lfp.shape[0]}) vs timestamps ({ts.shape[0]}): "
                  f"truncating both to {n}")
            lfp, ts = lfp[:n], ts[:n]
        data["lfp_timestamps"] = ts
    big = lfp.shape[0] > LFP_ROWS_PER_CHUNK
    data["lfp"] = _lfp_chunks(lfp) if big else np.asarray(lfp, dtype=np.float32)

    if Path(nwb_path).is_file():
        # append into the existing session file — never rewrite it, so anything
        # step w/u already added (and any scoring) survives
        io = NWBHDF5IO(str(nwb_path), mode="r+")
        try:
            nwbfile = io.read()
            added = snwb.add_sleep_inputs(nwbfile, **data)
            if added:
                io.write(nwbfile)
        finally:
            io.close()
    else:
        nwbfile = NWBFile(
            session_description="Sleep-scoring inputs (LFP / EMG / motion).",
            identifier=str(uuid4()),
            session_start_time=session_start_time(prefix),
            session_id=prefix.rstrip("_"),
            lab="Genzel Lab", institution="Radboud University",
        )
        added = snwb.add_sleep_inputs(nwbfile, **data)
        tmp = nwb_path.with_suffix(".tmp.nwb")
        try:
            with NWBHDF5IO(str(tmp), "w") as io:
                io.write(nwbfile)
            tmp.replace(nwb_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    print(f"[sleep-nwb] Added: {', '.join(added) if added else '(nothing new)'}")
    for name in ("lfp", "emg_from_lfp", "motion", "sleep_channels"):
        mark = "+" if name in added else "="
        print(f"   {mark} {name}")
    print(f"[sleep-nwb] Done -> {nwb_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Package the sleep-scoring inputs (LFP/EMG/motion) into the "
                    "session NWB so HM_rat_sleep_score can read one file.")
    ap.add_argument("--input_folder", required=False, help="Unused; kept for step-8 symmetry.")
    ap.add_argument("--output_folder", required=True,
                    help="Session op folder (the one holding LFP_Output/).")
    args = ap.parse_args()
    try:
        return run(args.output_folder)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[sleep-nwb] FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
