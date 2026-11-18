"""Remove a <session>_Coordinates_Full_with_frames.csv left over from an earlier
tracker run.

Step d (tracking_eyes.py) writes that file as a copy of the tracker's
<session>_Coordinates_Full.csv plus region_id / extracted_frame_idx (and later the
DeepLabCut keypoints). Rerunning the tracker replaces Coordinates_Full.csv but not
the copy, and every later step reads the copy first — so a session re-tracked into
26 trials was still plotted with the 17 trials of the run before."""
from pathlib import Path

import numpy as np
import pandas as pd

# What the tracker decides. 'Timestamp' is left out: the clock repair rewrites it
# in place, in both files, without the tracking having changed.
_TRACKING_COLS = ("Frame_Index", "Trial_Num", "Rat_X", "Rat_Y")
_SUFFIX = "_with_frames.csv"


def same_tracking(full_csv, with_frames_csv) -> bool:
    """True when both files hold the same tracking (frames, trials, rat position)."""
    a, b = (pd.read_csv(p, usecols=lambda c: c in _TRACKING_COLS)
            for p in (full_csv, with_frames_csv))
    if len(a) != len(b) or set(a.columns) != set(b.columns):
        return False
    return all(
        np.allclose(pd.to_numeric(a[c], errors="coerce"), pd.to_numeric(b[c], errors="coerce"),
                    rtol=0, atol=1e-6, equal_nan=True)
        for c in a.columns)


def drop_stale_with_frames(folder) -> list:
    """Delete every *_Coordinates_Full_with_frames.csv in `folder` whose tracking no
    longer matches the *_Coordinates_Full.csv beside it. Returns the stale paths —
    including one that could not be deleted (open in another program), so that the
    caller can still leave it unread."""
    stale = []
    folder = Path(folder)
    if not folder.is_dir():
        return stale
    for full in sorted(folder.glob("*Coordinates_Full.csv")):
        if full.name.startswith("._"):          # macOS sidecar, not a CSV
            continue
        copy = full.with_name(full.stem + _SUFFIX)
        if not copy.exists():
            continue
        try:
            if same_tracking(full, copy):
                continue
        except Exception as e:
            print(f"[WARNING] Could not compare {copy.name} with {full.name} ({e}); left as it is.")
            continue
        stale.append(copy)
        try:
            copy.unlink()
            print(f"[INFO] Deleted {copy.name}: it came from an earlier tracker run and no longer "
                  f"matches {full.name}. Rerun step d to rebuild it.")
        except OSError as e:
            print(f"[WARNING] {copy.name} is from an earlier tracker run but could not be "
                  f"deleted ({e}). Delete it by hand, then rerun step d.")
    return stale
