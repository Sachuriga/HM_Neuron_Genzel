"""clock_qc --auto on a synthetic op folder: a 0.75 % stretched video clock is detected,
repaired in place across every video-clock file, backed up, and left alone on re-run."""
import re

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import clock_qc

FS, DUR, STRETCH, UNIX0 = 250.0, 1800.0, 0.0075, 1_782_000_000.0


def make_op(tmp_path):
    rng = np.random.default_rng(3)
    t = np.arange(0, DUR, 1 / FS)                                   # true = ephys clock
    bouts = np.cumsum(rng.random(len(t)) < 0.0008) % 2
    loc = gaussian_filter1d(bouts * (1 + rng.random(len(t))), FS * 0.4)
    lead = int(round(clock_qc.PHYS_LAG_S * FS))                     # theta leads speed
    amp = 1 + 3 * np.r_[loc[lead:], np.repeat(loc[-1], lead)]
    theta = amp * np.sin(2 * np.pi * 8 * t) + 0.3 * rng.standard_normal(len(t))
    lfp = np.c_[0.3 * rng.standard_normal(len(t)), theta].astype(np.float32)
    (tmp_path / "LFP_Output").mkdir()
    np.save(tmp_path / "LFP_Output" / "Rat1_20260101_lfp_data.npy", lfp)
    np.save(tmp_path / "LFP_Output" / "Rat1_20260101_lfp_timestamps.npy", t)

    tf = np.arange(0, DUR, 1 / 30)                                  # true frame times
    s_anchor_true = 1350 / 30
    s = tf + STRETCH * (tf - s_anchor_true)                         # stretched video clock
    x = np.cumsum(np.interp(tf, t, loc)) * 5
    crlf = dict(index=False, lineterminator="\r\n")                 # as the tracker PC writes
    pd.DataFrame({"Frame Number": np.arange(len(s)), "Seconds From Creation": s}).to_csv(
        tmp_path / "stitched_framewise_seconds.csv", **crlf)
    pd.DataFrame({"Frame Number": np.arange(len(s)), "Corrected Time Stamp": s + UNIX0}).to_csv(
        tmp_path / "20260101_Rat1_framewise_ts.csv", **crlf)
    trial = np.where(np.arange(len(s)) % 7 == 0, np.nan, 1.0)        # empty cells stay empty
    pd.DataFrame({"Frame_Index": np.arange(len(s)), "Timestamp": s, "Trial_Num": trial,
                  "Rat_X": x, "Rat_Y": 0.0, "Researcher_X": 166.0}).to_csv(
        tmp_path / "20260101_Rat1_Coordinates_Full_with_frames.csv", **crlf)
    (tmp_path / "log_20260101_Rat1.log").write_bytes(
        ("INFO : Recording Trial 1\r\n" + "".join(
            f"INFO : 00:16:40.{k:03d} {float(s[30000 + k] + UNIX0)!r} : The rat position is: (1, 2) @ 112\r\n"
            for k in range(3))).encode())
    a, b = float(s[30000]), float(s[30600])                         # frames 1000 s, 1020 s
    (tmp_path / "20260101_Rat1.txt").write_text(
        "Rat number: 1 , Date: 20260101 \nSummary Trial 1\n"
        f"Trial End (Sync Seconds): {b!r}\n"
        "Start-Next Nodes // Sync Time (s) // Diff (s) // Length (m) // Velocity (m/s)\n"
        f"('112', '111') ({a!r}, {b!r}) {b - a:.3f} 0.3 {0.3 / (b - a):.3f}\n"
        f"('111', '110') ({a + UNIX0!r}, {b + UNIX0!r}) {b - a:.3f} 0.3 {0.3 / (b - a):.3f}\n")
    return tf, s


def test_auto_repairs_every_video_clock_file(tmp_path):
    tf, s_old = make_op(tmp_path)
    assert clock_qc.run(tmp_path, auto=True) == clock_qc.EXIT_REPAIRED
    backup = tmp_path / clock_qc.BACKUP_DIR
    for name in ("stitched_framewise_seconds.csv", "20260101_Rat1_framewise_ts.csv",
                 "20260101_Rat1_Coordinates_Full_with_frames.csv", "20260101_Rat1.txt",
                 "log_20260101_Rat1.log", "clock_repair.json"):
        assert (backup / name).exists(), name

    s_new = pd.read_csv(tmp_path / "stitched_framewise_seconds.csv").iloc[:, 1].to_numpy()
    assert np.max(np.abs(s_new - tf)) < 0.3                         # stretch gone
    ts_new = pd.read_csv(tmp_path / "20260101_Rat1_framewise_ts.csv").iloc[:, 1].to_numpy()
    assert np.allclose(ts_new - s_new, UNIX0, atol=1e-4)            # pair stays consistent
    coords = tmp_path / "20260101_Rat1_Coordinates_Full_with_frames.csv"
    c = pd.read_csv(coords)
    assert np.allclose(c.Timestamp, s_new, atol=1e-6)
    # every other cell is byte-identical, and the CRLF line ending is kept
    new_rows = coords.read_bytes().split(b"\r\n")
    old_rows = (backup / coords.name).read_bytes().split(b"\r\n")
    assert len(new_rows) == len(old_rows)
    for a, b in zip(new_rows[:2000], old_rows[:2000]):
        a, b = a.split(b","), b.split(b",")
        assert a[:1] + a[2:] == b[:1] + b[2:]
    log = (tmp_path / "log_20260101_Rat1.log").read_bytes()
    assert log.count(b"\r\n") == 4
    utc = [float(v) for v in re.findall(rb"\.\d{3} (\d+\.\d+) :", log)]
    assert np.allclose(utc, s_new[30000:30003] + UNIX0, atol=1e-3)

    txt = (tmp_path / "20260101_Rat1.txt").read_text()
    nums = [float(v) for v in re.findall(r"\(([\d.]+), ([\d.]+)\)", txt)[0]]
    assert abs(nums[0] - s_new[30000]) < 1e-6 and abs(nums[1] - s_new[30600]) < 1e-6
    unix = [float(v) for v in re.findall(r"\(([\d.]+), ([\d.]+)\)", txt)[1]]
    assert abs(unix[0] - (s_new[30000] + UNIX0)) < 1e-3
    diff = float(txt.splitlines()[4].split()[-3])
    assert abs(diff - (s_new[30600] - s_new[30000])) < 1e-3           # duration recomputed

    # second run: passes, changes nothing
    before = (tmp_path / "stitched_framewise_seconds.csv").read_text()
    assert clock_qc.run(tmp_path, auto=True) == clock_qc.EXIT_OK
    assert (tmp_path / "stitched_framewise_seconds.csv").read_text() == before
    # the backup still holds the true original
    s_bak = pd.read_csv(backup / "stitched_framewise_seconds.csv").iloc[:, 1].to_numpy()
    assert np.allclose(s_bak, s_old)


def test_healthy_session_is_not_touched(tmp_path):
    global STRETCH
    old, STRETCH = STRETCH, 0.0
    try:
        make_op(tmp_path)
    finally:
        STRETCH = old
    assert clock_qc.run(tmp_path, auto=True) == clock_qc.EXIT_OK
    assert not (tmp_path / clock_qc.BACKUP_DIR).exists()
