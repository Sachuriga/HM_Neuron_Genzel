"""
Rewrite the LFP inside a session NWB from the old LFP_Output export.

WHY THIS EXISTS
---------------
Sessions exported before step 8 wrote the NWB itself got their LFP into the NWB
through create_nwb.py, from LFP_Output/*lfp_data.npy. Two things went wrong there:
  - create_nwb paired the LFP with the 5 Hz EMG timestamps
    (*_emg_from_lfp_timestamps.npy, which its file match picked up) and cut it to
    their length: the NWB holds a ~20 000-sample stub stamped at 5 Hz;
  - some exports stamped a 1 kHz LFP at the 30 kHz acquisition rate (Trodes put the
    raw rate in the LFP header), so the trace "lasts" 1/30 of the session — in the
    NWB and in LFP_Output/lfp_timestamps.npy alike.
The export's samples are right in both cases; the NWB copy and the stamps are not.

WHAT IT DOES (one op folder)
----------------------------
1. Works on a LOCAL copy of the NWB and the export: the share is slow, and an
   interrupted write must never leave a broken file there.
2. Finds the export's true rate: the stamped rate when the export lasts as long as
   the spikes, else the standard rate that does (or that the LFP's high-frequency
   power follows the population spike rate at, for an export that runs on after
   the spikes). The rate is always cross-checked that way when there are spikes.
3. Drops the old acquisition/lfp and writes the export in step 8's layout
   (sleep_nwb_writer.add_sleep_inputs: float32 on a rate, 50 s x 4-channel gzip
   chunks, processing/sleep/session_info with the channel map). An export at
   another rate than LFP_OUTPUT_RATE (1500 Hz) is resampled to it (polyphase, exact
   ratio; session_info records it).
4. Verifies the copy: every other object unchanged, the LFP equal to what was meant
   to be written, the theta reader accepts it.
5. Only then replaces the NWB: the original goes to <op>/lfp_fix_backup/, with a
   record in lfp_fix_backup/lfp_fix.json. A mis-stamped LFP_Output lfp_timestamps.npy
   is rewritten too (original in LFP_Output/lfp_fix_backup/).
A session whose NWB already holds the rewritten LFP, or step 8's own LFP, is left
alone, and so is a session without LFP_Output — except that a mis-stamped
lfp_timestamps.npy left by an interrupted earlier run is still restamped.

Freed space: dropping a full-length old LFP leaves its space inside the HDF5 file.
It is given back with h5repack when that is installed (conda: hdf5 package), else
the file keeps it until `python src/nwb/compact_nwb.py <op>`.

USAGE
-----
    python src/nwb/rewrite_nwb_lfp.py --output_folder <op> [--rate 1500] [--dry-run]
Exit code 0: rewritten, or nothing to do. 1: stopped — the op folder is untouched,
unless the log says "STOPPED after replacing the NWB" (then run it again to finish).
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sleep_nwb_writer as snwb                                    # noqa: E402

STANDARD_LFP_RATES = (1000.0, 1250.0, 1500.0, 2000.0, 2500.0)
BACKUP_DIR = "lfp_fix_backup"
REF_EXPORTLFP = "trodes_exportlfp"        # = lfp_from_raw.REF_EXPORTLFP: Trodes' LFP reference
# objects this tool replaces; everything else must come out bit-identical (EMG,
# motion, sleep_channels and any scorings in processing/sleep included)
REPLACED = ("/acquisition/lfp", "/processing/sleep/session_info", "/processing/sleep/layout",
            "/processing/sleep/theta_delta_ratio", "/specifications")
TIME_AXIS = ("seconds from the first sample of the first concatenated session; "
             "spike times share this origin")


def say(msg):
    print(f"[lfp-fix {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------- #
#  Files
# ---------------------------------------------------------------------------- #
def find_output(folder, suffix):
    """``<prefix><suffix>`` (or bare ``<suffix>``) in an LFP_Output folder, else None.
    Only a session prefix may precede the suffix, so lfp_timestamps.npy never
    resolves to ..._emg_from_lfp_timestamps.npy, nor session_boundaries.npy to
    ..._motion_session_boundaries.npy (the same rule as export_sleep_nwb)."""
    folder = Path(folder)
    exact = folder / suffix
    if exact.is_file():
        return exact
    for p in sorted(folder.glob(f"*{suffix}")):
        if re.fullmatch(r"[A-Za-z]+\d+_\d{8}_\d{6}_", p.name[:-len(suffix)]):
            return p
    return None


def copy_with_sha256(src, dst, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            b = fi.read(bufsize)
            if not b:
                break
            h.update(b)
            fo.write(b)
        fo.flush()
        os.fsync(fo.fileno())
    return h.hexdigest()


def sha256(path, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def drop_old_lfp(nwb_path):
    """Delete the LFP and what describes or derives from it per sample (session_info,
    theta_delta_ratio). Unlike snwb.drop_lfp_outputs, the EMG stays: it has its own
    clock and nothing here writes it again."""
    names = [f"acquisition/{snwb.LFP_NAME}"] + [
        f"processing/{snwb.SLEEP_MODULE}/{n}" for n in (snwb.SESSION_INFO_NAME,) + snwb.DERIVED_NAMES]
    with h5py.File(str(nwb_path), "r+") as f:
        removed = [p for p in names if p in f]
        for p in removed:
            del f[p]
    return removed


def find_h5repack():
    """h5repack on PATH or in this Python environment (a conda env ships it with
    the hdf5 package: bin/ on Linux/macOS, Library/bin/ on Windows), else None."""
    for cand in (shutil.which("h5repack"),
                 Path(sys.prefix) / "bin" / "h5repack",
                 Path(sys.prefix) / "Library" / "bin" / "h5repack.exe"):
        if cand and Path(cand).is_file():
            return str(cand)
    return None


def unique_path(path):
    """`path`, or `path` with a time stamp before the suffix when it exists."""
    path = Path(path)
    if not path.exists():
        return path
    return path.with_name(f"{path.name}.{time.strftime('%Y%m%d_%H%M%S')}")


def sidecars(*dirs):
    """macOS AppleDouble files ("._*") in these folders. A Mac writes one next to
    each file it creates on an SMB share; they are not data."""
    return {str(p) for d in dirs if Path(d).is_dir() for p in Path(d).glob("._*")}


class NwbReplacedError(RuntimeError):
    """Stopped AFTER the NWB was replaced: the op folder is NOT untouched."""


def append_record(backup, rec):
    """Add rec to <op>/lfp_fix_backup/lfp_fix.json (written whole, then renamed)."""
    log = backup / "lfp_fix.json"
    history = json.loads(log.read_text()) if log.exists() else []
    history.append(rec)
    tmp_log = log.with_name(log.name + ".tmp")
    tmp_log.write_text(json.dumps(history, indent=1, default=str))
    os.replace(tmp_log, log)


def stamps_rate(ts):
    """The rate a timestamps array claims, in Hz."""
    # k/fs stamps give fs back only to ~1e-9 (1499.999999998): take the rate itself
    return round(float(1.0 / np.median(np.diff(ts[:200000]))), 3)


def restamp_export(op, lfp_dir, tsf, fs):
    """Rewrite LFP_Output's lfp_timestamps.npy at the export's true rate fs when it
    is stamped at the acquisition rate (the Trodes header bug); the original goes to
    LFP_Output/lfp_fix_backup/. Returns the record for lfp_fix.json, or None when the
    stamps need no fix."""
    # into memory, never memory-mapped: Windows cannot replace a mapped file
    ts = np.load(tsf)
    if len(ts) < 2:
        return None
    stamped = stamps_rate(ts)
    if stamped < 20000 or abs(stamped - fs) < 1e-6:
        return None
    (lfp_dir / BACKUP_DIR).mkdir(exist_ok=True)
    kept_ts = lfp_dir / BACKUP_DIR / tsf.name
    if not kept_ts.exists():
        shutil.copy2(tsf, kept_ts)
    if sha256(kept_ts) != sha256(tsf):
        raise RuntimeError(f"{kept_ts} is not a copy of {tsf.name}; left as is")
    new_ts = np.arange(len(ts), dtype=np.float64) / fs
    tmp_ts = lfp_dir / (tsf.name + ".lfpfix.tmp")
    try:
        with open(tmp_ts, "wb") as fo:
            np.save(fo, new_ts)
        os.replace(tmp_ts, tsf)
    finally:
        # a failed replace must not leave the half-done copy on the share
        if tmp_ts.exists():
            tmp_ts.unlink()
    if not np.array_equal(np.load(tsf), new_ts):
        raise RuntimeError(f"{tsf} did not read back")
    return {"file": str(tsf.relative_to(op)), "was_hz": stamped, "now_hz": fs,
            "backup": str(kept_ts.relative_to(op))}


# ---------------------------------------------------------------------------- #
#  Fingerprints: proof that nothing outside the LFP changed
# ---------------------------------------------------------------------------- #
def _value(f, v):
    if isinstance(v, h5py.Reference):
        return ("ref", f[v].name if v else None)
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return ("arr", [_value(f, x) for x in v.ravel().tolist()])
        return ("arr", str(v.dtype), v.shape,
                hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest())
    if isinstance(v, bytes):
        return ("bytes", v.decode("utf-8", "replace"))
    if isinstance(v, np.generic):
        return ("np", str(v.dtype), v.item())
    return ("py", repr(v))


def _has_refs(dtype):
    if dtype.names:
        return any(_has_refs(dtype[n]) for n in dtype.names)
    return bool(h5py.check_dtype(ref=dtype)) or dtype == object


def _dataset_digest(f, ds):
    if ds.shape is None:
        return "null"
    if _has_refs(ds.dtype):
        # references are compared by what they point to (h5repack moves objects)
        v = np.atleast_1d(ds[()])
        if ds.dtype.names:
            rows = [tuple(_value(f, r[n]) for n in ds.dtype.names) for r in v]
            return hashlib.sha1(repr(rows).encode()).hexdigest()
        return hashlib.sha1(repr([_value(f, x) for x in v.ravel().tolist()]).encode()).hexdigest()
    h = hashlib.sha1()
    if ds.ndim and ds.size:
        row_bytes = ds.dtype.itemsize * int(np.prod(ds.shape[1:], dtype=np.int64) or 1)
        step = max(1, (64 << 20) // max(1, row_bytes))
        for i in range(0, ds.shape[0], step):
            h.update(np.ascontiguousarray(ds[i:i + step]).tobytes())
    else:
        h.update(np.ascontiguousarray(ds[()]).tobytes())
    return h.hexdigest()


def fingerprint(path):
    """{object path: digest of its type, data and attributes} for everything
    outside REPLACED, links included."""
    out = {}
    with h5py.File(path, "r") as f:
        def attrs(o):
            return [(k, _value(f, o.attrs[k])) for k in sorted(o.attrs.keys())]

        def walk(group, prefix):
            for k in group:
                p = f"{prefix}/{k}"
                if any(p == r or p.startswith(r + "/") for r in REPLACED):
                    continue
                link = group.get(k, getlink=True)
                if isinstance(link, h5py.SoftLink):
                    out[p] = ("softlink", link.path)
                elif isinstance(link, h5py.ExternalLink):
                    out[p] = ("extlink", link.filename, link.path)
                else:
                    o = group[k]
                    if isinstance(o, h5py.Dataset):
                        out[p] = ("dataset", str(o.dtype), o.shape, _dataset_digest(f, o), attrs(o))
                    else:
                        out[p] = ("group", attrs(o))
                        walk(o, p)
        out["/"] = ("group", attrs(f))
        walk(f, "")
    return out


# ---------------------------------------------------------------------------- #
#  The export's true rate
# ---------------------------------------------------------------------------- #
def hf_power(lfp, block=100):
    """log high-frequency power (squared first difference, median over channels)
    per `block` samples."""
    n = lfp.shape[0] // block * block
    out = np.empty(n // block)
    step = block * 5000                  # ~128 MB of a 32-channel LFP at a time
    for i in range(0, n, step):
        x = np.asarray(lfp[i:min(i + step, n)], np.float64)
        d = np.diff(x, axis=0, prepend=x[:1])
        out[i // block:(i + len(x)) // block] = np.median(
            (d ** 2).reshape(-1, block, x.shape[1]).mean(1), axis=1)
    return np.log(out + 1e-12), block


def spike_covariation(power, block, spikes, rates):
    """Correlation of the LFP's high-frequency power with the population spike rate
    (1-s bins), with the LFP placed in time at each candidate rate. Both rise and
    fall with the animal's state, so they only line up at the true rate."""
    edges = np.arange(0.0, float(np.max(spikes)) + 1.0, 1.0)
    rate = np.histogram(spikes, edges)[0].astype(float)
    out = {}
    for r in rates:
        t = (np.arange(len(power)) * block + block / 2) / r
        keep = t < edges[-1]
        if keep.sum() < 1000:
            out[r] = float("nan")
            continue
        idx = np.digitize(t[keep], edges) - 1
        s = np.bincount(idx, weights=power[keep], minlength=len(rate))[:len(rate)]
        c = np.bincount(idx, minlength=len(rate))[:len(rate)]
        ok = c > 0
        out[r] = float(np.corrcoef(s[ok] / c[ok], rate[ok])[0, 1])
    return out


def export_rate(n, stamped, spikes_end, tracking_end, cov=None, given=None):
    """(rate, why) of an n-sample export, or (None, why)."""
    if given:
        return float(given), f"--export_rate {given:g} given"
    if spikes_end:
        # LFP and spikes come from the same recording: they end together
        tol = max(2.0, 0.002 * spikes_end)
        if stamped and abs(n / stamped - spikes_end) <= tol:
            return stamped, f"stamped {stamped:g} Hz; the export ends with the spikes"
        fits = [r for r in STANDARD_LFP_RATES if abs(n / r - spikes_end) <= tol]
        if len(fits) == 1:
            return fits[0], (f"stamped {stamped or 0:g} Hz would last {n / stamped if stamped else 0:.0f} s; "
                             f"at {fits[0]:g} Hz the export ends with the spikes "
                             f"({n / fits[0]:.1f} s vs {spikes_end:.1f} s)")
        if cov:
            # an export that runs on after the spikes: the rate the LFP follows them at
            ranked = sorted((v, r) for r, v in cov.items() if r in STANDARD_LFP_RATES and np.isfinite(v))
            if len(ranked) >= 2 and ranked[-1][0] > 0.15 and ranked[-1][0] - ranked[-2][0] > 0.05 \
                    and n / ranked[-1][1] > spikes_end:
                r = ranked[-1][1]
                return r, (f"the LFP follows the spikes at {r:g} Hz (r={ranked[-1][0]:.2f}, next "
                           f"{ranked[-2][0]:.2f}); the export runs {n / r:.0f} s, the spikes end "
                           f"at {spikes_end:.0f} s")
        return None, f"no rate makes the export ({n} samples) end with the spikes ({spikes_end:.1f} s)"
    # no units in the NWB: the tracking is the only clock to compare with (ephys and
    # camera are started by hand, so 0.5-2x is still a plausible standard-rate fit)
    if tracking_end:
        if stamped and any(abs(stamped - r) <= 0.01 * r for r in STANDARD_LFP_RATES) \
                and 0.5 <= n / stamped / tracking_end <= 2.0:
            return stamped, f"stamped {stamped:g} Hz (no spikes; checked against the tracking only)"
        fit = min(STANDARD_LFP_RATES, key=lambda r: abs(n / r - tracking_end))
        if abs(n / fit - tracking_end) < 0.2 * tracking_end:
            return fit, f"{fit:g} Hz fits the tracking ({n / fit:.0f} s vs {tracking_end:.0f} s; no spikes)"
    return None, "no spikes and no tracking to check the rate against"


# ---------------------------------------------------------------------------- #
#  The NWB as it is
# ---------------------------------------------------------------------------- #
def stored_lfp(nwb_path):
    """What the NWB holds now: {rate, timestamps_hz, shape, info} or None."""
    with h5py.File(str(nwb_path), "r") as f:
        g = f.get(f"acquisition/{snwb.LFP_NAME}")
        if g is None:
            return None
        out = {"shape": tuple(g["data"].shape), "rate": None, "timestamps_hz": None, "info": None}
        if "starting_time" in g:
            out["rate"] = float(g["starting_time"].attrs["rate"])
        elif "timestamps" in g and g["timestamps"].shape[0] > 1:
            out["timestamps_hz"] = float(1.0 / np.median(np.diff(g["timestamps"][:5000])))
        p = f"processing/{snwb.SLEEP_MODULE}/{snwb.SESSION_INFO_NAME}"
        if p in f:
            try:
                out["info"] = json.loads(f[p].attrs.get("description", "") or "{}")
            except ValueError:
                out["info"] = None
    return out


def leave_alone(stored, target, force=False):
    """Why this NWB's LFP must be kept as it is, or None."""
    if force or stored is None or not stored["info"]:
        return None
    info = stored["info"]
    if "rewritten" in info:
        if stored["rate"] and abs(stored["rate"] - target) < 1e-6 \
                and len(info.get("channel_map") or []) == stored["shape"][1]:
            return f"already rewritten ({stored['shape'][0]} samples at {target:g} Hz)"
        return None
    if info.get("lfp_reference") is not None:
        return "step 8 wrote this LFP itself (session_info.lfp_reference is set)"
    return None


# ---------------------------------------------------------------------------- #
#  One session
# ---------------------------------------------------------------------------- #
def rewrite(op, target_rate=1500.0, work=None, dry_run=False, given_rate=None, force=False,
            keep_work=False):
    """Rewrite op's NWB LFP. Returns the record (status 'replaced', 'restamped',
    'rehearsed' or 'skipped'); raises when it stops (the op folder is then untouched,
    except after NwbReplacedError: the NWB is replaced, the export's stamps are not).
    A dry run keeps its rewritten local copy only with keep_work."""
    t0 = time.time()
    op = Path(op).resolve()
    nwb = snwb.find_session_nwb(op)
    lfp_dir = op / "LFP_Output"
    rec = {"op": str(op), "started": dt.datetime.now().isoformat(timespec="seconds"),
           "target_rate_hz": target_rate}
    if nwb is None:
        rec.update(status="skipped", why="no session NWB")
        return rec
    rec["nwb"] = nwb.name
    files = {k: find_output(lfp_dir, f"{k}.npy") if lfp_dir.is_dir() else None
             for k in ("lfp_data", "lfp_timestamps", "channel_map", "session_boundaries",
                       "channel_snr_scores", "cleanest_channel_indices")}
    if files["lfp_data"] is None:
        rec.update(status="skipped", why="no LFP_Output/lfp_data.npy (the LFP comes from step 8)")
        return rec
    stored = stored_lfp(nwb)
    keep = leave_alone(stored, target_rate, force)
    if keep:
        rec.update(status="skipped", why=keep)
        # an earlier run may have replaced the NWB but stopped before restamping the
        # export's lfp_timestamps.npy (on Windows the stamps used to stay memory-mapped,
        # and a mapped file cannot be replaced): finish that part now
        export_rate_hz = ((stored["info"] or {}).get("rewritten") or {}).get("export_rate_hz")
        if export_rate_hz and files["lfp_timestamps"] and not dry_run:
            done = restamp_export(op, lfp_dir, files["lfp_timestamps"], float(export_rate_hz))
            if done:
                rec.update(status="restamped", export_timestamps=done,
                           minutes=round((time.time() - t0) / 60, 1))
                append_record(op / BACKUP_DIR, rec)
                say(f"{nwb.name}: NWB already rewritten; {files['lfp_timestamps'].name} restamped "
                    f"{done['was_hz']:g} -> {done['now_hz']:g} Hz")
        return rec
    export = np.load(files["lfp_data"], mmap_mode="r")
    if stored and stored["shape"][1] > export.shape[1] and not force:
        rec.update(status="skipped", why=f"the NWB's LFP has more channels ({stored['shape'][1]}) "
                                         f"than the export ({export.shape[1]})")
        return rec
    rec["source"] = {k: (str(v.relative_to(op)) if v else None) for k, v in files.items()}
    rec["old_lfp"] = {"shape": list(stored["shape"]) if stored else None,
                      "rate_hz": stored and (stored["rate"] or stored["timestamps_hz"])}

    work = Path(work or Path(tempfile.gettempdir()) / "hm_lfp_fix") / f"{nwb.stem}_{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        rec = _rewrite_in(work, op, nwb, lfp_dir, files, stored, rec, target_rate,
                          dry_run, given_rate, t0)
        if rec["status"] == "rehearsed" and not keep_work:
            rec["work"] = None                       # removed below
        return rec
    finally:
        if not (dry_run and keep_work):
            shutil.rmtree(work, ignore_errors=True)


def _rewrite_in(work, op, nwb, lfp_dir, files, stored, rec, target_rate, dry_run, given_rate, t0):
    # 1. local copies
    st = nwb.stat()
    rec["orig"] = {"size": st.st_size, "mtime": st.st_mtime}
    local = work / nwb.name
    rec["orig"]["sha256"] = copy_with_sha256(nwb, local)
    local_export = work / "lfp_data.npy"
    shutil.copyfile(files["lfp_data"], local_export)
    lfp = np.load(local_export, mmap_mode="r")
    n, n_ch = lfp.shape
    # read the stamps into memory, NOT memory-mapped: step 6 may replace this file,
    # and Windows refuses to replace a file that is still mapped ("[WinError 5]
    # Access is denied"); for float64 stamps asarray() would not even make a copy
    ts = (np.asarray(np.load(files["lfp_timestamps"]), float)
          if files["lfp_timestamps"] else None)
    say(f"{nwb.name}: copied ({st.st_size / 1e9:.2f} GB NWB, export {n} x {n_ch})")

    # 2. the clocks to check the rate against
    before = fingerprint(local)
    with h5py.File(local, "r") as f:
        spikes = np.asarray(f["units/spike_times"][:], float) if "units/spike_times" in f else None
        tracking_end = None
        pos = f.get("processing/Behavior/Position")
        if pos is not None and len(pos):
            first = pos[sorted(pos.keys())[0]]
            if "timestamps" in first:
                tracking_end = float(np.nanmax(first["timestamps"][:]))
    spikes = spikes if spikes is not None and spikes.size > 1000 else None
    spikes_end = float(spikes.max()) if spikes is not None else None

    # 3. the export's true rate
    stamped, uniform = None, None
    if ts is not None and len(ts) > 1:
        stamped = stamps_rate(ts)
        # evenly spaced, and (stamps in seconds) sample 0 at time 0 like the spikes
        uniform = bool(np.max(np.abs(np.diff(ts) - 1.0 / stamped)) < 1e-3 / stamped
                       and (stamped < 100 or abs(ts[0]) < 0.5 / stamped))
    cov = None
    if spikes is not None:
        power, block = hf_power(lfp)
        rates = set(STANDARD_LFP_RATES) | {r for r in (stamped, given_rate) if r and 100 <= r < 20000}
        cov = spike_covariation(power, block, spikes, sorted(rates))
    fs, why = export_rate(n, stamped, spikes_end, tracking_end, cov, given_rate)
    rec["rate"] = {"stamped_hz": stamped, "stamps_uniform_from_0": uniform, "export_hz": fs, "why": why,
                   "spikes_end_s": spikes_end, "tracking_end_s": tracking_end,
                   "spike_covariation": {f"{k:g}": round(v, 4) for k, v in (cov or {}).items()}}
    say(f"{nwb.name}: {why}" + (f"; LFP vs spikes r = " + ", ".join(
        f"{k:g} Hz {v:.2f}" for k, v in cov.items()) if cov else ""))
    if fs is None:
        raise RuntimeError(why)
    if cov and not given_rate:
        # a weak correlation proves nothing; another rate that clearly beats it does
        better = {r: v for r, v in cov.items()
                  if np.isfinite(v) and v > 0.15 and v > cov.get(fs, -1.0) + 0.05}
        if better:
            raise RuntimeError(f"the LFP follows the spikes better at another rate than {fs:g} Hz: {cov}")
    if uniform is False:
        raise RuntimeError("the export's timestamps are not evenly spaced from 0 — a rate cannot hold them")

    # what goes into the NWB: the export, resampled to the target rate if need be
    data, out_rate = lfp, float(target_rate)
    if abs(out_rate - fs) > 1e-6:
        from scipy.signal import resample_poly
        ratio = Fraction(int(round(out_rate)), int(round(fs)))
        n_out = -(-n * ratio.numerator // ratio.denominator)
        data = np.lib.format.open_memmap(work / "lfp_resampled.npy", mode="w+",
                                         dtype=np.float32, shape=(n_out, n_ch))
        for c in range(n_ch):
            y = resample_poly(np.asarray(lfp[:, c], np.float64), ratio.numerator, ratio.denominator)
            data[:, c] = y[:n_out].astype(np.float32)
        data.flush()
        # at the instants both clocks share, the resampled trace must follow the export
        k = min(n // ratio.denominator, n_out // ratio.numerator, 300000)
        a = np.asarray(lfp[:k * ratio.denominator:ratio.denominator], np.float64)
        b = np.asarray(data[:k * ratio.numerator:ratio.numerator], np.float64)
        r = float(np.median([np.corrcoef(a[:, c], b[:, c])[0, 1] for c in range(n_ch)]))
        rec["resampled"] = {"from_hz": fs, "to_hz": out_rate, "up": ratio.numerator,
                            "down": ratio.denominator, "n_samples": n_out,
                            "method": "scipy.signal.resample_poly (Kaiser-window FIR), per channel",
                            "median_corr_at_shared_instants": round(r, 6)}
        say(f"{nwb.name}: resampled {fs:g} -> {out_rate:g} Hz ({n} -> {n_out} samples, "
            f"r = {r:.5f} at the shared instants)")
        if r < 0.99:
            raise RuntimeError(f"the resampled trace does not follow the export (r = {r:.4f})")

    # 4. drop the old LFP, give its space back, write the new one
    rec["dropped"] = drop_old_lfp(local)
    old_bytes = int(np.prod(stored["shape"])) * 4 if stored else 0
    if old_bytes > 20e6:
        h5repack = find_h5repack()
        if h5repack:
            packed = work / (local.stem + ".packed.nwb")
            subprocess.run([h5repack, str(local), str(packed)], check=True)
            os.replace(packed, local)
            rec["compacted"] = True
        else:
            rec["compacted"] = False
            say(f"{nwb.name}: no h5repack here — the old LFP's ~{old_bytes / 1e9:.1f} GB stay inside "
                f"the file until `python src/nwb/compact_nwb.py {op}`")
    cmap = [dict(x) for x in np.load(files["channel_map"], allow_pickle=True)] \
        if files["channel_map"] else []
    if cmap and len(cmap) != n_ch:
        raise RuntimeError(f"the channel map has {len(cmap)} entries for {n_ch} columns")
    meta = {"lfp_rate_hz": out_rate, "n_samples": int(data.shape[0]), "n_channels": int(n_ch),
            "time_axis": TIME_AXIS, "channel_map": cmap, "lfp_reference": REF_EXPORTLFP}
    if files["cleanest_channel_indices"]:
        meta["cleanest_channel_indices"] = np.load(files["cleanest_channel_indices"]).tolist()
    if files["channel_snr_scores"]:
        meta["channel_snr_scores"] = np.load(files["channel_snr_scores"]).tolist()
    if files["session_boundaries"]:
        scale = out_rate / fs
        meta["session_boundaries"] = [
            {"name": b["name"], "start_sample": int(round(int(b["start"]) * scale)),
             "n_samples": int(round(int(b["n"]) * scale)),
             "start_time": int(b["start"]) / fs, "stop_time": (int(b["start"]) + int(b["n"])) / fs}
            for b in np.load(files["session_boundaries"], allow_pickle=True)]
    old = rec["old_lfp"]
    meta["rewritten"] = {
        "date": dt.date.today().isoformat(), "tool": "src/nwb/rewrite_nwb_lfp.py",
        "from": f"LFP_Output/{files['lfp_data'].name} (Trodes exportLFP, one channel per nTrode)",
        "export_rate_hz": fs, "why": why,
        "replaced": (f"{old['shape'][0]}-sample acquisition/lfp at {old['rate_hz'] or 0:g} Hz"
                     if old["shape"] else "no LFP"),
        **({"resampled": rec["resampled"]} if "resampled" in rec else {})}
    with h5py.File(local, "r") as f:
        version = (f.attrs.get("nwb_version"), f.attrs.get_id("nwb_version").dtype) \
            if "nwb_version" in f.attrs else None
    from pynwb import NWBHDF5IO
    io = NWBHDF5IO(str(local), mode="r+")
    try:
        nwbfile = io.read()
        rec["added"] = snwb.add_sleep_inputs(nwbfile, lfp=data, lfp_rate=out_rate, metadata=meta)
        io.write(nwbfile)
    finally:
        io.close()
    # pynwb stamps its own schema version on a file it appends to; the file keeps its
    # own (the containers written here, TimeSeries, are the same in both versions)
    with h5py.File(local, "r+") as f:
        if version is not None and f.attrs.get("nwb_version") != version[0]:
            rec["nwb_version"] = {"pynwb_stamped": str(f.attrs.get("nwb_version")),
                                  "kept": str(version[0])}
            del f.attrs["nwb_version"]
            f.attrs.create("nwb_version", version[0], dtype=version[1])

    # 5. verify the copy
    after = fingerprint(local)
    changed = [k for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)
               and not (k == f"/processing/{snwb.SLEEP_MODULE}" and k not in before)]  # new module
    if changed:
        raise RuntimeError(f"objects outside the LFP changed: {changed[:8]}")
    with h5py.File(local, "r") as f:
        g = f[f"acquisition/{snwb.LFP_NAME}"]
        ds = g["data"]
        if ds.shape != tuple(data.shape) or ds.dtype != np.float32 or "timestamps" in g \
                or float(g["starting_time"][()]) != 0.0 \
                or abs(float(g["starting_time"].attrs["rate"]) - out_rate) > 1e-9:
            raise RuntimeError("the written LFP is not laid out as intended")
        step = (ds.chunks or (65536,))[0] * 8
        for i in range(0, ds.shape[0], step):
            if not np.array_equal(ds[i:i + step], np.asarray(data[i:i + step], np.float32)):
                raise RuntimeError(f"the written LFP differs from the source at rows {i}..{i + step}")
        rec["new_lfp"] = {"shape": list(ds.shape), "rate_hz": out_rate, "chunks": list(ds.chunks or []),
                          "compression": ds.compression}
    rec["theta_reader"] = _theta_reader_check(work, tracking_end, out_rate, len(cmap))
    say(f"{nwb.name}: verified ({len(before)} other objects unchanged, LFP equal to its source, "
        f"theta reader: {rec['theta_reader']})")
    if dry_run:
        rec.update(status="rehearsed", work=str(work))
        return rec

    # 6. replace the NWB (and fix the export's stamps), keeping the originals
    cur = nwb.stat()
    if (cur.st_size, cur.st_mtime) != (rec["orig"]["size"], rec["orig"]["mtime"]):
        raise RuntimeError("the NWB changed while this ran — nothing replaced")
    backup = op / BACKUP_DIR
    known = sidecars(op, backup, lfp_dir, lfp_dir / BACKUP_DIR)
    tmp = op / (nwb.name + ".lfpfix.tmp")
    new_sha = copy_with_sha256(local, tmp)
    if sha256(tmp) != new_sha:
        tmp.unlink()
        raise RuntimeError("the copy on the share does not read back identically — nothing replaced")
    backup.mkdir(exist_ok=True)
    kept = unique_path(backup / (nwb.name + ".prelfpfix"))
    os.replace(nwb, kept)
    os.replace(tmp, nwb)
    rec["new"] = {"size": nwb.stat().st_size, "sha256": new_sha}
    rec["backup"] = str(kept.relative_to(op))
    rec["status"] = "replaced"
    say(f"{nwb.name}: replaced (original -> {rec['backup']})")

    # from here on the NWB IS replaced: whatever happens to the stamps, record it
    try:
        if files["lfp_timestamps"]:
            done = restamp_export(op, lfp_dir, files["lfp_timestamps"], fs)
            if done:
                rec["export_timestamps"] = done
                say(f"{nwb.name}: {files['lfp_timestamps'].name} restamped "
                    f"{done['was_hz']:g} -> {done['now_hz']:g} Hz")
    except Exception as e:
        rec["export_timestamps"] = {"error": f"{type(e).__name__}: {e}"}
        raise NwbReplacedError(
            f"the NWB was replaced (original in {rec['backup']}), but "
            f"LFP_Output/{files['lfp_timestamps'].name} was not restamped: "
            f"{type(e).__name__}: {e} — run step l again to finish it") from e
    finally:
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        append_record(backup, rec)
        for p in sorted(sidecars(op, backup, lfp_dir, lfp_dir / BACKUP_DIR) - known):
            os.remove(p)
    return rec


def _theta_reader_check(folder, tracking_end, rate, n_ch):
    """Would theta_events (visualize_nwb's theta pages) use this LFP?"""
    try:
        import theta_events as te
    except Exception as e:                  # its plotting imports are not needed here
        return f"not checked ({type(e).__name__})"
    src = te._lfp_source_from_nwb(folder)
    try:
        t = np.array([0.0, tracking_end]) if tracking_end else None
        problem = te._lfp_rate_problem(src, t)
        cmap = len(src["channel_map"] or [])
        if problem or abs(src["fs"] - rate) > 1e-6 or cmap != n_ch:
            raise RuntimeError(f"theta_events would not use it: {problem or ''} "
                               f"fs={src['fs']:g} channel map {cmap}/{n_ch}")
        return f"ok ({src['fs']:g} Hz, channel map {cmap})"
    finally:
        src["close"]()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output_folder", required=True, help="Session op folder (holds the NWB and LFP_Output/).")
    ap.add_argument("--config", default=None, help="Unused; accepted like the other per-op steps.")
    ap.add_argument("--rate", type=float, default=float(os.environ.get("LFP_OUTPUT_RATE", "") or 1500),
                    help="Rate of the LFP in the NWB (default LFP_OUTPUT_RATE, else 1500 Hz).")
    ap.add_argument("--export_rate", type=float, default=None,
                    help="The export's true rate, when it cannot be worked out.")
    ap.add_argument("--work", default=None, help="Local scratch folder (default: the system temp).")
    # the GUI's HM_LFP_FIX_DRY_RUN wins over the config file's LFP_FIX_DRY_RUN
    dry_env = next((os.environ[k].strip() for k in ("HM_LFP_FIX_DRY_RUN", "LFP_FIX_DRY_RUN")
                    if os.environ.get(k, "").strip()), "")
    ap.add_argument("--dry-run", action="store_true",
                    default=dry_env.lower() in ("1", "true", "yes", "on"),
                    help="Do and verify everything on a local copy; change nothing in the op folder "
                         "(also LFP_FIX_DRY_RUN=1 in hm_tracker_paths.txt, e.g. for a first pass of step l).")
    ap.add_argument("--keep-work", action="store_true",
                    help="With --dry-run: keep the rewritten local copy to inspect.")
    ap.add_argument("--force", action="store_true",
                    help="Rewrite even an LFP this tool or step 8 already wrote.")
    a = ap.parse_args()
    try:
        rec = rewrite(a.output_folder, a.rate, a.work, a.dry_run, a.export_rate, a.force, a.keep_work)
    except NwbReplacedError as e:
        import traceback
        traceback.print_exc()
        say(f"STOPPED after replacing the NWB: {e}")
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        say(f"STOPPED: {type(e).__name__}: {e} — {a.output_folder} is untouched.")
        return 1
    if rec["status"] == "skipped":
        say(f"{rec.get('nwb', a.output_folder)}: nothing to do — {rec['why']}")
    elif rec["status"] == "rehearsed":
        say(f"{rec['nwb']}: dry run OK — would be rewritten; nothing in the op folder changed"
            + (f" (rewritten copy: {rec['work']})" if rec.get("work") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
