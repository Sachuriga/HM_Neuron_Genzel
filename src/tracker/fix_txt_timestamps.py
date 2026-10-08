"""
Runs at the end of runner step 4 (former step [f]): repair tracker .txt files
whose timestamps are unix time.

An earlier tracker version renamed only stitched_framewise_ts.csv (the unix
'Corrected Time Stamp' clock) at the end of a run and then preferred that file
on re-runs — so a re-tracked session's .txt summary carried unix timestamps
(~1.7e9 s) instead of session seconds. The two sync CSVs are frame-by-frame
aligned, so unix -> seconds is an exact per-frame mapping: build it from
*framewise_ts.csv and *framewise_seconds.csv and rewrite the .txt in place
(the original is kept once as <name>.txt.unixbak).

What gets converted (only numeric values > 1e6, so the step is idempotent and
a healthy seconds-clock .txt is left byte-identical):
  - 'Trial End (Sync Seconds): <t>'
  - the '(t_start, t_end)' pair of every node-transition row
Durations / lengths / velocities are untouched (the clock change is an offset,
differences are identical on both clocks). 'N/A' and non-numeric tokens are
left alone.

The sync step writes seconds = unix - constant for the same frames, so a valid
CSV pair has one fixed offset; a pair from two different sync runs (one file
renamed by an older tracker, the other re-synced) does not, and is refused
rather than used to rewrite the .txt with wrong times.

Usage:
    python fix_txt_timestamps.py --output_folder <op> [--dry_run]
"""
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UNIX_MIN = 1e6          # anything above this is not a session-seconds value
OFFSET_TOL = 1e-3       # s; spread of (unix - seconds) allowed for one sync run

_TXT_RE = re.compile(r"^\d{8}_Rat\d+\.txt$")
_TRIAL_END_RE = re.compile(r"(Trial End \(Sync Seconds\): )(\S+)")
_TRANSITION_RE = re.compile(r"(\('\d+',\s*'\d+'\)\s+\()([^,]+)(,\s*)([^)]+)(\))")


def _csvs(op, pat):
    """Files in op matching pat, skipping macOS AppleDouble sidecars."""
    return [p for p in sorted(Path(op).glob(pat)) if not p.name.startswith("._")]


def _prefix(p, suffix):
    return p.name[:-len(suffix)]


def _pair_mapping(ts_p, sec_p):
    """(unix, seconds, offset_spread) for one CSV pair joined by frame index."""
    ts = pd.read_csv(ts_p, index_col=0)
    sec = pd.read_csv(sec_p, index_col=0)
    tcol = next((c for c in ts.columns if "stamp" in c.lower()), ts.columns[0])
    scol = next((c for c in sec.columns if "second" in c.lower()), sec.columns[0])
    j = ts[[tcol]].join(sec[[scol]], how="inner").dropna()
    if len(j) < 2:
        return None
    u = j[tcol].to_numpy(float)
    s = j[scol].to_numpy(float)
    off = u - s
    order = np.argsort(u)                       # np.interp needs ascending x
    return u[order], s[order], float(off.max() - off.min())


def _load_mapping(op):
    """(unix_array, seconds_array) sorted by unix, from the frame-aligned sync
    CSVs (*framewise_ts.csv <-> *framewise_seconds.csv). Pairs with the same
    prefix (one sync run) are tried first; the first pair whose unix-seconds
    offset is constant is used. None if no consistent pair exists."""
    ts_all = _csvs(op, "*framewise_ts.csv")
    sec_all = _csvs(op, "*framewise_seconds.csv")
    pairs = sorted(((t, s) for t in ts_all for s in sec_all),
                   key=lambda ts: _prefix(ts[0], "framewise_ts.csv")
                   != _prefix(ts[1], "framewise_seconds.csv"))
    for ts_p, sec_p in pairs:
        m = _pair_mapping(ts_p, sec_p)
        if m is None:
            continue
        u, s, spread = m
        if spread > OFFSET_TOL:
            print(f"  {ts_p.name} <-> {sec_p.name}: unix-seconds offset varies by "
                  f"{spread:.3f} s — not from the same sync run, not used.")
            continue
        print(f"  mapping: {ts_p.name} <-> {sec_p.name} ({len(u)} frames)")
        return u, s
    return None


def _fix_text(text, unix, secs):
    """(new_text, n_converted, n_out_of_range). Only numeric tokens > UNIX_MIN
    inside the two timestamp positions are converted."""
    stats = {"n": 0, "oob": 0}

    def conv(tok):
        t = tok.strip()
        try:
            v = float(t)
        except ValueError:
            return tok
        if v <= UNIX_MIN:
            return tok
        if v < unix[0] - 1.0 or v > unix[-1] + 1.0:
            stats["oob"] += 1
        stats["n"] += 1
        return repr(float(np.interp(v, unix, secs)))

    def trial_end(m):
        return m.group(1) + conv(m.group(2))

    def transition(m):
        return (m.group(1) + conv(m.group(2)) + m.group(3)
                + conv(m.group(4)) + m.group(5))

    text = _TRIAL_END_RE.sub(trial_end, text)
    text = _TRANSITION_RE.sub(transition, text)
    return text, stats["n"], stats["oob"]


def run(output_folder, dry_run=False):
    op = Path(output_folder)
    txts = [p for p in sorted(op.glob("*_Rat*.txt"))
            if _TXT_RE.match(p.name) and not p.name.startswith("._")]
    if not txts:
        print(f"No tracker .txt (<date>_Rat<N>.txt) in {op} — nothing to do.")
        return 0
    mapping = None
    fixed = 0
    for p in txts:
        text = p.read_text(errors="replace")
        # quick probe: any unix-sized value in a timestamp position?
        probe, n_probe, _ = _fix_text(text, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        if n_probe == 0:
            print(f"  {p.name}: already on the seconds clock — untouched.")
            continue
        if mapping is None:
            mapping = _load_mapping(op)
            if mapping is None:
                print(f"  {p.name}: has unix timestamps but no consistent framewise "
                      f"ts+seconds CSV pair in {op} — cannot convert, skipped.")
                continue
        new_text, n, oob = _fix_text(text, *mapping)
        if oob:
            print(f"  WARNING {p.name}: {oob}/{n} values outside the sync CSV's "
                  f"time range were clamped to its edges.")
        if dry_run:
            print(f"  {p.name}: would convert {n} unix timestamps (dry run).")
            continue
        bak = p.with_suffix(p.suffix + ".unixbak")
        if not bak.exists():                    # first backup = the true original
            bak.write_text(text)
        p.write_text(new_text)
        print(f"  {p.name}: converted {n} unix timestamps -> seconds "
              f"(original kept as {bak.name}).")
        fixed += 1
    if fixed:
        print(f"Fixed {fixed} file(s). Step 4 writes the NWB right after this, so "
              f"its Trials_Data gets the corrected values.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert unix-time timestamps in tracker .txt files back to "
                    "session seconds via the frame-aligned sync CSVs.")
    ap.add_argument("--output_folder", required=True, help="op/session folder.")
    ap.add_argument("--config", default=None, help="Accepted for runner consistency (unused).")
    ap.add_argument("--dry_run", action="store_true", help="report only, write nothing.")
    a = ap.parse_args()
    raise SystemExit(run(a.output_folder, dry_run=a.dry_run))
