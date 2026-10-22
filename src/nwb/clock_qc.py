"""Ephys-vs-video clock check, and repair, for one op folder.

The video clock (framewise seconds from step 2) and the ephys clock (spike times and
LFP as sample/fs) are tied together only through the LED/DIO sync. This checks the
tie with the data itself: running speed from the video and a locomotion signal
recorded on the ephys clock — theta amplitude from the LFP export, and the population
spike rate — follow each other at a small, constant lag. Their cross-correlation in
2-min windows gives

    D(t) = video label - ephys label of the same moment   (s)

Theta amplitude leads speed by ~0.15 s (median over 39 well-synced HexMaze sessions),
so a healthy session shows D ~ +0.15 s everywhere. A drift (D growing over the
session) or a jump means the two clocks disagree: Rat5_20260630's step-2 fit was
0.75 % fast and D grew by ~27 s per hour.

The clock line D(t) = c + d*t is the consensus of the windows (the line most windows
agree with to 0.3 s), so spurious peaks and a poor theta channel do not move it. A
signal is used only when its windows agree along its whole recording; two used
signals must agree with each other, and the agreeing windows must cover the session.

    PASS     clocks agree
    WARN     a drift beyond the limit that is not significant, many windows disagree, or
             theta and spike rate (one clock) give different drifts — including a signal
             too patchy to be used whose own line contradicts the verdict
    FAIL     the clocks disagree (drift > 0.3 s/h and > 3 SE, or a lag > 0.5 s off)
    UNKNOWN  the data cannot tell (too few or patchy windows, signals disagree)

Repair, s -> s' (the fit is repeated on the corrected clock until it holds):

    s' = s - d*(s - s_anchor)        slope only, anchored at the frame where step 2
                                     anchored its fit
    s' = s - (c + d*s - phys_lag)    also the constant part

    python src/nwb/clock_qc.py <op>            check; report in <op>/clock_qc/
    python src/nwb/clock_qc.py <op> --auto     on FAIL, re-time the op folder's video-clock
                                               files in place if the repaired clock passes
                                               (exit 10); else change nothing (exit 3)
    python src/nwb/clock_qc.py <op> --repair   repaired framewise copies only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert, welch

DT = 0.05                    # analysis bin (s)
PHYS_LAG_S = 0.15            # theta amplitude leads running speed (median, well-synced sessions)
VIDEO_FPS = 30.0
SYNC_START_SEC = 45.0        # step 2 anchors its fit at the first frame after this
CONSENSUS_TOL = 0.3          # s: a window this close to the clock line agrees with it
MAX_DRIFT_S_PER_H = 0.3      # healthy sessions scatter +-0.11 s/h (SE ~0.07); every session
                             # beyond 0.3 s/h had an odd step-2 frame rate predicting its drift
MIN_AGREE = 8                # agreeing windows behind a verdict (16 min of 2-min windows)
MIN_AGREE_SIGNAL = 6         # ... behind using one signal
N_SEGMENTS = 4               # the line must be confirmed in 3 of 4 quarters of the session
MAX_AUTO_DRIFT_S_PER_H = 60  # beyond this (1.7 %) the video is suspect, not its clock rate


# ----------------------------------------------------------------------------- inputs
def _one(pattern_list, folder, exclude=("_repaired", ".prerepair")):
    for pat in pattern_list:
        hits = [p for p in sorted(Path(folder).glob(pat)) if not any(e in p.name for e in exclude)
                and not p.name.startswith("._")]
        if hits:
            return hits[0]
    return None


def find_inputs(op):
    op = Path(op)
    lfp = _one(["LFP_Output/*lfp_data.npy", "*lfp_data.npy"], op,
               exclude=("_repaired", ".prerepair", "emg", "motion"))
    # the LFP's OWN timestamps (a bare *lfp_timestamps.npy glob also matches the 5 Hz
    # *_emg_from_lfp_timestamps.npy, which sorts first)
    lfp_ts = lfp.with_name(lfp.name.replace("lfp_data", "lfp_timestamps")) if lfp else None
    return dict(
        coords=_one(["*Coordinates_Full_with_frames.csv"], op),
        seconds=_one(["*_framewise_seconds.csv", "stitched_framewise_seconds.csv"], op),
        ts=_one(["*_framewise_ts.csv", "stitched_framewise_ts.csv"], op),
        lfp=lfp,
        lfp_ts=lfp_ts if lfp_ts is not None and lfp_ts.exists() else None,
        phy=_one(["*_sorting_output/phy_export", "phy_export"], op),
    )


def video_speed(coords_csv, seconds_csv, dt=DT):
    """(bin times, speed px/s) on the video clock, frames matched by Frame_Index."""
    c = pd.read_csv(coords_csv, usecols=lambda k: k in ("Frame_Index", "Rat_X", "Rat_Y"))
    s = pd.read_csv(seconds_csv)
    sec = pd.Series(s.iloc[:, 1].to_numpy(float), index=s.iloc[:, 0].to_numpy(int))
    fr = c["Frame_Index"].to_numpy(int) if "Frame_Index" in c else np.arange(len(c))
    t = sec.reindex(fr).to_numpy()
    x, y = c["Rat_X"].to_numpy(float), c["Rat_Y"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    tb = np.arange(max(t[0], 0.0), t[-1], dt) + dt / 2
    xb, yb = np.interp(tb, t, x), np.interp(tb, t, y)
    sp = np.r_[0.0, np.hypot(np.diff(xb), np.diff(yb))] / dt
    return tb, sp


STANDARD_LFP_RATES = (1000.0, 1250.0, 1500.0, 2000.0, 2500.0)


def trusted_lfp_rate(fs, n, expected):
    """Can `fs` be the rate of this n-sample LFP of a session lasting `expected` s?
    Yes when the duration matches, or when fs is a standard LFP rate and the recording
    is merely longer or shorter than the video (ephys and camera are started and
    stopped by hand: 0.6-1.2x seen). A 1 kHz LFP stamped at 30 kHz lasts 1/30."""
    if not expected:
        return True
    ratio = n / fs / expected
    standard = any(abs(fs - r) <= 0.01 * r for r in STANDARD_LFP_RATES)
    return 0.8 <= ratio <= 1.25 or (standard and 0.5 <= ratio <= 2.0)


def theta_amplitude(lfp_npy, lfp_ts_npy=None, channel=None, dt=DT, band=(6.0, 12.0),
                    expected_duration=None):
    """(bin times on the ephys clock, theta amplitude) from the LFP export.

    Some exports stamp a 1 kHz LFP with 30 kHz timestamps. When the stamped rate cannot
    be right (trusted_lfp_rate), the standard rate whose duration fits the session
    (`expected_duration`) is used instead, with a warning.
    """
    data = np.load(lfp_npy, mmap_mode="r")
    n = data.shape[0]
    if lfp_ts_npy is not None and Path(lfp_ts_npy).exists():
        ts = np.load(lfp_ts_npy, mmap_mode="r")
        if len(ts) != n:
            # a few samples' difference is an export quirk; more is a different recording
            if abs(len(ts) - n) > max(10, 1e-4 * n):
                raise ValueError(f"{Path(lfp_ts_npy).name} has {len(ts)} samples, the LFP {n}")
            n = min(n, len(ts))
        fs = 1.0 / float(np.median(np.diff(np.asarray(ts[:5000], float))))
        t0 = float(ts[0])
    else:
        raise FileNotFoundError("LFP timestamps missing; cannot tell the LFP sampling rate")
    if not trusted_lfp_rate(fs, n, expected_duration):
        fit_fs = min(STANDARD_LFP_RATES, key=lambda r: abs(n / r - expected_duration))
        if abs(n / fit_fs - expected_duration) < 0.2 * expected_duration:
            print(f"[clock-qc] WARNING: the LFP timestamps say {fs:.0f} Hz ({n / fs:.0f} s) but "
                  f"the session lasts ~{expected_duration:.0f} s; using {fit_fs:.0f} Hz")
            fs, t0 = fit_fs, 0.0
    if fs < 100:
        raise ValueError(f"LFP sampling rate {fs:.1f} Hz is implausible")
    if channel is None:
        probe = np.asarray(data[: int(min(400 * fs, n))], np.float32)
        f, p = welch(probe, fs=fs, nperseg=int(8 * fs), axis=0)
        ratio = p[(f >= 6) & (f <= 10)].mean(0) / p[(f >= 2) & (f <= 4)].mean(0)
        channel = int(np.argmax(ratio))
    sig = np.empty(n, np.float64)
    step = int(fs * 600)
    for a in range(0, n, step):
        sig[a:min(a + step, n)] = data[a:min(a + step, n), channel]
    b, a_ = butter(2, [band[0] / (fs / 2), band[1] / (fs / 2)], "band")
    amp = np.abs(hilbert(filtfilt(b, a_, sig), N=next_fast_len(n))[:n])
    k = int(round(dt * fs))
    m = n // k
    tb = t0 + (np.arange(m) + 0.5) * k / fs
    return tb, amp[: m * k].reshape(m, k).mean(1), channel, fs


def spike_rate(phy_dir, dt=DT):
    """(bin times, population rate) from a phy export (sample/fs clock)."""
    phy = Path(phy_dir)
    fs = 30000.0
    params = phy / "params.py"
    if params.exists():
        for line in params.read_text().splitlines():
            if line.strip().startswith("sample_rate"):
                fs = float(line.split("=")[1].strip())
    st = np.load(phy / "spike_times.npy").astype(np.int64).ravel() / fs
    edges = np.arange(0.0, st.max() + dt, dt)
    return edges[:-1] + dt / 2, np.histogram(st, edges)[0].astype(float)


# ----------------------------------------------------------------------------- lags
def _z(v):
    v = np.nan_to_num(np.asarray(v, float))
    return (v - v.mean()) / (v.std() + 1e-12)


def window_lags(t_ephys, ephys, t_video, speed, window=120.0, max_lag=25.0, smooth=0.5, dt=DT):
    """D (video label - ephys label of the same moment) per window, with quality.

    Both signals are put on one grid of `dt` bins, smoothed, z-scored per window, and
    cross-correlated over +-max_lag. Returns a DataFrame: t (window centre, s),
    D (s), r_peak, r_zero, edge (peak at the end of the search range: the true lag
    may lie beyond it).
    """
    t0, t1 = max(t_ephys[0], t_video[0]), min(t_ephys[-1], t_video[-1])
    grid = np.arange(t0, t1, dt)
    e = gaussian_filter1d(np.interp(grid, t_ephys, ephys), smooth / dt)
    v = gaussian_filter1d(np.interp(grid, t_video, speed), smooth / dt)
    w, L = int(window / dt), int(max_lag / dt)
    lags = np.arange(-L, L + 1)
    rows = []
    for i in range(0, len(grid) - w + 1, w):
        a, b = _z(e[i:i + w]), _z(v[i:i + w])
        n = len(a)
        xc = np.array([np.mean(a[max(0, l):n + min(0, l)] * b[max(0, -l):n - max(0, l)])
                       for l in lags])
        k = int(np.argmax(xc))
        rows.append(dict(t=float(grid[i] + window / 2), D=float(-lags[k] * dt),
                         r_peak=float(xc[k]), r_zero=float(xc[L]),
                         edge=bool(k == 0 or k == len(lags) - 1)))
    return pd.DataFrame(rows)


def consensus_line(t, D, tol=CONSENSUS_TOL):
    """(slope, intercept, agree mask): the line the most windows agree with to `tol`.

    Every line through two windows, and the flat line through each, is a candidate;
    the winner is refitted by least squares on the windows that agree with it. Single
    windows lock onto spurious peaks tens of seconds away (rest, handling) — however
    many there are, they do not move this line, where they pull a regression.
    """
    t, D = np.asarray(t, float), np.asarray(D, float)
    n = len(t)
    if n == 0:
        return np.nan, np.nan, np.zeros(0, bool)
    i, j = np.triu_indices(n, 1)
    ok = t[j] != t[i]
    i, j = i[ok], j[ok]
    slope = np.r_[(D[j] - D[i]) / (t[j] - t[i]), np.zeros(n)]
    icpt = np.r_[D[i] - slope[:len(i)] * t[i], D]
    best = (-1, np.inf, 0)
    for a in range(0, len(slope), 4096):                  # bounded memory
        res = np.abs(D[None, :] - icpt[a:a + 4096, None] - slope[a:a + 4096, None] * t[None, :])
        agree = res <= tol
        count = agree.sum(1)
        cost = np.where(agree, res, 0.0).sum(1)
        k = int(np.lexsort((cost, -count))[0])
        if (count[k], -cost[k]) > (best[0], -best[1]):
            best = (int(count[k]), float(cost[k]), a + k)
    d, c = float(slope[best[2]]), float(icpt[best[2]])
    keep = np.abs(D - (c + d * t)) <= tol
    for _ in range(5):
        if keep.sum() >= 3 and np.ptp(t[keep]) > 0:
            d, c = (float(v) for v in np.polyfit(t[keep], D[keep], 1))
        new = np.abs(D - (c + d * t)) <= tol
        if (new == keep).all() or new.sum() < keep.sum():
            break
        keep = new
    return d, c, keep


def _segments(t, keep, lo, hi, n=N_SEGMENTS):
    """(confirmed, contradicted) quarters of [lo, hi]: confirmed when >= 2 windows and
    40 % of the reliable ones agree with the line; contradicted when >= 3 reliable
    windows and < 25 % agree (a jump, or the clocks parting late in the session).
    One window agreeing by chance confirms nothing."""
    seg = np.clip(np.searchsorted(np.linspace(lo, hi, n + 1), t, side="right") - 1, 0, n - 1)
    confirmed, contradicted = [], []
    for k in range(n):
        m = seg == k
        a = int(keep[m].sum())
        if a >= 2 and a >= 0.4 * m.sum():
            confirmed.append(k)
        elif m.sum() >= 3 and a < 0.25 * m.sum():
            contradicted.append(k)
    return confirmed, contradicted


def _clock_line(t, D, lo, hi):
    d, c, keep = consensus_line(t, D)
    n = int(keep.sum())
    se = np.inf
    if n >= 3:
        r = D[keep] - (c + d * t[keep])
        sxx = float(np.sum((t[keep] - t[keep].mean()) ** 2))
        if sxx > 0:
            se = float(np.sqrt(np.sum(r ** 2) / max(n - 2, 1) / sxx))
    confirmed, contradicted = _segments(t, keep, lo, hi)
    return dict(slope=d, offset_s=c, keep=keep, n_agree=n, se=se,
                confirmed=confirmed, contradicted=contradicted)


def assess(lags, phys_lag=PHYS_LAG_S, min_r=0.3, max_drift_s_per_h=MAX_DRIFT_S_PER_H,
           max_dev_s=0.5):
    """Fit the clock line D(t) = c + d*t and judge the session (statuses: module doc).

    A window is reliable when the coupling is clear (r_peak >= min_r) and its peak lies
    inside the search range. Each signal's clock line is the consensus of its reliable
    windows; a signal is used when at least MIN_AGREE_SIGNAL windows, and half of its
    reliable ones, agree and the line is confirmed in 3 of 4 quarters of its recording.
    The verdict is the consensus of the used signals' windows, which must be confirmed
    in 3 of 4 quarters of the session.
    """
    if len(lags) == 0:
        return dict(n_windows=0, n_reliable=0, status="UNKNOWN",
                    reason="the ephys and video signals do not overlap by one window")
    lags = lags.copy()
    if "signal" not in lags:
        lags["signal"] = "signal"
    if "edge" not in lags:
        lags["edge"] = False
    rel = lags[(lags.r_peak >= min_r) & ~lags.edge.astype(bool)]
    out = dict(n_windows=len(lags), n_reliable=len(rel), status="UNKNOWN", per_signal={})
    used = []
    for name, g in rel.groupby("signal"):
        span = lags[lags.signal == name].t
        f = _clock_line(g.t.to_numpy(float), g.D.to_numpy(float), float(span.min()), float(span.max()))
        ok = (f["n_agree"] >= MIN_AGREE_SIGNAL and f["n_agree"] >= 0.5 * len(g)
              and len(f["confirmed"]) >= N_SEGMENTS - 1)
        out["per_signal"][name] = dict(drift_s_per_h=f["slope"] * 3600, se_s_per_h=f["se"] * 3600,
                                       n_agree=f["n_agree"], n_reliable=len(g),
                                       quarters_confirmed=len(f["confirmed"]), used=bool(ok))
        if ok:
            used.append((name, f))
    out["signals_used"] = [n for n, _ in used]
    if not used:
        out["reason"] = "no signal with windows that agree along the recording"
        return out
    # LFP and spikes share the sample/fs clock, so two used signals that differ mean one
    # is misleading: such a session is never repaired automatically.
    disagree = None
    if len(used) >= 2:
        (n1, f1), (n2, f2) = used[:2]
        tol = max(max_drift_s_per_h / 3600, 3 * np.hypot(f1["se"], f2["se"]),
                  0.1 * max(abs(f1["slope"]), abs(f2["slope"])))
        if abs(f1["slope"] - f2["slope"]) > tol:
            disagree = (f"{n1} and {n2} disagree ({f1['slope'] * 3600:+.2f} vs "
                        f"{f2['slope'] * 3600:+.2f} s/h)")
    g = rel[rel.signal.isin(out["signals_used"])]
    t, D = g.t.to_numpy(float), g.D.to_numpy(float)
    lo, hi = float(lags.t.min()), float(lags.t.max())
    f = _clock_line(t, D, lo, hi)
    d, c, keep = f["slope"], f["offset_s"], f["keep"]
    dev = np.abs(D[keep] - phys_lag)
    out.update(drift_s_per_h=d * 3600, drift_se_s_per_h=f["se"] * 3600, offset_s=c, slope=d,
               n_agree=f["n_agree"], n_outliers=int(len(g) - f["n_agree"]),
               quarters_confirmed=len(f["confirmed"]),
               median_D=float(np.median(D[keep])) if keep.any() else np.nan,
               max_dev_from_phys_s=float(dev.max()) if keep.any() else np.nan)
    if f["n_agree"] < MIN_AGREE:
        out["reason"] = f"only {f['n_agree']} windows agree"
        return out
    q = (hi - lo) / N_SEGMENTS
    quarter = lambda k: f"{(lo + k * q) / 60:.0f}-{(lo + (k + 1) * q) / 60:.0f} min"
    if len(f["confirmed"]) < N_SEGMENTS - 1:
        out["reason"] = ("the clock line is confirmed only in "
                         + (", ".join(quarter(k) for k in f["confirmed"]) or "no part")
                         + f" of {lo / 60:.0f}-{hi / 60:.0f} min")
        return out
    drifts = abs(d * 3600) > max_drift_s_per_h
    significant = abs(d) > 3 * f["se"]
    off = np.median(dev) > max_dev_s
    if (drifts and significant) or off:
        out["status"] = "FAIL"
        out["reason"] = (f"drift {d * 3600:+.2f} +- {f['se'] * 3600:.2f} s/h" if drifts and significant
                         else f"lag {np.median(D[keep]):+.2f} s, not ~{phys_lag:+.2f} s")
    elif drifts or f["contradicted"] or f["n_agree"] < 0.5 * len(g) or dev.max() > 2 * max_dev_s:
        out["status"] = "WARN"
        out["reason"] = ("drift beyond the limit but not significant" if drifts else
                         "the windows in " + ", ".join(quarter(k) for k in f["contradicted"])
                         + " disagree with the clock line" if f["contradicted"] else
                         "fewer than half the reliable windows agree" if f["n_agree"] < 0.5 * len(g)
                         else "some agreeing windows are > 1 s from the physiological lag")
    else:
        out["status"] = "PASS"
    # a signal too patchy to be used, but with a line of its own, must not contradict
    # the verdict either (Rat6_20260708: spikes -0.65, theta -1.56 s/h; place coding
    # agreed with neither, so no repair could be trusted)
    for name, p in out["per_signal"].items():
        if disagree or p["used"] or p["n_agree"] < MIN_AGREE_SIGNAL:
            continue
        tol = max(max_drift_s_per_h, 3 * np.hypot(p["se_s_per_h"], out["drift_se_s_per_h"]),
                  0.1 * max(abs(p["drift_s_per_h"]), abs(out["drift_s_per_h"])))
        if abs(p["drift_s_per_h"] - out["drift_s_per_h"]) > tol:
            disagree = (f"{name} (not used) gives {p['drift_s_per_h']:+.2f} s/h against "
                        f"{out['drift_s_per_h']:+.2f}")
    if disagree:
        out["status"] = "WARN"
        out["reason"] = disagree + (f"; pooled: {out['reason']}" if out.get("reason") else "")
    return out


def all_lags(signals, tv, sp, window):
    """Window lags of every ephys signal against video speed, pooled (column 'signal')."""
    parts = []
    for name, (te, sig) in signals.items():
        L = window_lags(te, sig, tv, sp, window=window)
        L.insert(0, "signal", name)
        parts.append(L)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ----------------------------------------------------------------------------- repair
EXIT_OK, EXIT_FAIL, EXIT_REVIEW, EXIT_REPAIRED = 0, 1, 3, 10
BACKUP_DIR = "clock_repair_backup"    # a subfolder: find_paths only reads top-level files
REPORT_DIR = "clock_qc"
UNIX_MIN = 1e6                        # larger values are unix times, smaller session seconds
TXT_NAME_RE = re.compile(r"^\d{8}_Rat\d+\.txt$")
TRIAL_END_RE = re.compile(r"(Trial End \(Sync Seconds\): )(\S+)")
# "('112', '111') (t_start, t_end) diff length velocity" (fix_txt_timestamps.py's
# pattern, extended by the three numbers that follow)
TRANSITION_RE = re.compile(r"(\('\d+',\s*'\d+'\)\s+\()([^,]+)(,\s*)([^)]+)(\))"
                           r"(?:([ \t]+)(\S+)([ \t]+)(\S+)([ \t]+)(\S+))?")
# "INFO : 00:01:50.733 1784020013.6818247 : The rat position is: ..."
LOG_RE = re.compile(r"^(INFO : \d+:\d\d:\d\d\.\d+ )(\d+\.\d+)( : )", re.M)


def correction_fn(fit, s_anchor, with_offset=False, phys_lag=PHYS_LAG_S):
    """corr(s): seconds to subtract from a video-clock time s (session seconds)."""
    d, c = fit["slope"], fit["offset_s"]
    if with_offset:
        return lambda s: c + d * np.asarray(s, float) - phys_lag
    return lambda s: d * (np.asarray(s, float) - s_anchor)


def repair_seconds(s, fit, s_anchor, with_offset=False, phys_lag=PHYS_LAG_S):
    s = np.asarray(s, float)
    return s - correction_fn(fit, s_anchor, with_offset, phys_lag)(s)


def converge_repair(signals, tv, sp, window, fit, s_anchor, with_offset=False, max_iter=3):
    """(corr, assessment of the corrected clock, fits used).

    The fit is repeated on the corrected clock: windows whose lag lay beyond the search
    range (a fast drift late in a session) are only seen once the clock is close. A
    further step is kept only if it reduces the drift.
    """
    steps, fits = [], []

    def corr(s):
        s = np.asarray(s, float)
        out = np.zeros_like(s)
        for f in steps:
            out = out + f(s - out)
        return out

    after, cur = None, fit
    for _ in range(max_iter):
        steps.append(correction_fn(cur, s_anchor, with_offset))
        nxt = assess(all_lags(signals, tv - corr(tv), sp, window))
        if after is not None and not abs(nxt.get("slope", np.inf)) < abs(after["slope"]):
            steps.pop()
            break
        fits.append(dict(slope=cur["slope"], offset_s=cur["offset_s"],
                         drift_s_per_h=cur["drift_s_per_h"]))
        after = nxt
        if after["status"] != "FAIL" or "slope" not in after:
            break
        cur = after
    return corr, after, fits


def video_clock_files(op):
    """Every top-level file of an op folder that carries video-clock times."""
    op = Path(op)

    def top(pattern):
        return [p for p in sorted(op.glob(pattern)) if not p.name.startswith("._")]

    return dict(seconds=top("*framewise_seconds.csv"), ts=top("*framewise_ts.csv"),
                coords=top("*Coordinates_Full*.csv"),
                txt=[p for p in top("*.txt") if TXT_NAME_RE.match(p.name)],
                log=top("*.log"))


def unix_offsets(files):
    """({ts file: offset}, common offset or None): `unix - seconds` of the sync run,
    from ts/seconds files of equal length whose difference is constant."""
    secs = [pd.read_csv(s).iloc[:, 1].to_numpy(float) for s in files["seconds"]]
    per = {}
    for p in files["ts"]:
        tv = pd.read_csv(p).iloc[:, 1].to_numpy(float)
        per[p] = next((float(np.median(tv - sv)) for sv in secs
                       if len(sv) == len(tv) and np.ptp(tv - sv) < 1e-3), None)
    vals = [v for v in per.values() if v is not None]
    return per, (vals[0] if vals and np.ptp(vals) < 1e-3 else None)


def _shift(v, corr, unix_off):
    """New value of one video-clock time, unix or session seconds (None: cannot map)."""
    if v > UNIX_MIN:
        return None if unix_off is None else v - float(corr(v - unix_off))
    return v - float(corr(v))


def _retime_values(v, corr, unix_off):
    v = np.asarray(v, float)
    nv = v.copy()
    sec = v <= UNIX_MIN
    nv[sec] = v[sec] - corr(v[sec])
    if unix_off is not None:
        nv[~sec] = v[~sec] - corr(v[~sec] - unix_off)
    return nv


def _write_atomic(path, data: bytes):
    tmp = path.with_name(f".{path.name}.clockqc.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _rewrite_csv_column(path, column, fn):
    """Apply fn to the finite values of one column. Every other cell is kept as the
    text it was, and the file keeps its line ending. Returns the number changed."""
    nl = "\r\n" if b"\r\n" in path.read_bytes()[:65536] else "\n"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    col = df.columns[column] if isinstance(column, int) else column
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    ok = np.isfinite(v)
    cells = df[col].to_numpy(object)
    cells[ok] = [repr(float(x)) for x in fn(v[ok])]
    df[col] = cells
    _write_atomic(path, df.to_csv(index=False, lineterminator=nl).encode())
    return int(ok.sum())


def retime_txt(text, corr, unix_off):
    """Tracker .txt with every Sync Seconds time moved to the repaired clock; transition
    durations and speeds are recomputed as the tracker writes them (round(x, 3))."""
    stats = dict(n=0, skipped=0)

    def conv(tok):
        try:
            v = float(tok.strip())
        except ValueError:
            stats["skipped"] += 1
            return tok, None
        nv = _shift(v, corr, unix_off)
        if nv is None:
            stats["skipped"] += 1
            return tok, None
        stats["n"] += 1
        return repr(float(nv)), nv

    def trial_end(m):
        return m.group(1) + conv(m.group(2))[0]

    def transition(m):
        (t0s, t0), (t1s, t1) = conv(m.group(2)), conv(m.group(4))
        out = m.group(1) + t0s + m.group(3) + t1s + m.group(5)
        if m.group(7) is not None:
            diff_s, length_s, vel_s = m.group(7), m.group(9), m.group(11)
            if t0 is not None and t1 is not None and t1 > t0:
                diff_s = str(round(t1 - t0, 3))
                try:
                    vel_s = str(round(float(length_s) / (t1 - t0), 3))
                except ValueError:
                    pass
            out += m.group(6) + diff_s + m.group(8) + length_s + m.group(10) + vel_s
        return out

    text = TRIAL_END_RE.sub(trial_end, text)
    text = TRANSITION_RE.sub(transition, text)
    return text, stats


def retime_log(text, corr, unix_off):
    """Tracker .log with its UTC synchronised timestamps moved to the repaired clock."""
    stats = dict(n=0, skipped=0)

    def sub(m):
        nv = _shift(float(m.group(2)), corr, unix_off)
        if nv is None:
            stats["skipped"] += 1
            return m.group(0)
        stats["n"] += 1
        return m.group(1) + repr(float(nv)) + m.group(3)

    return LOG_RE.sub(sub, text), stats


def apply_repair_in_place(op, corr):
    """Re-time every video-clock file of the op folder. Each original is copied once into
    <op>/clock_repair_backup/ before it is changed (never overwritten, so a second repair
    keeps the true originals). Returns a list of what changed."""
    op = Path(op)
    files = video_clock_files(op)
    per_ts, unix_off = unix_offsets(files)     # before anything changes
    backup = op / BACKUP_DIR
    backup.mkdir(exist_ok=True)
    changed = []

    def keep_original(p):
        dst = backup / p.name
        if not dst.exists():
            shutil.copy2(p, dst)

    for p in files["seconds"]:
        keep_original(p)
        n = _rewrite_csv_column(p, 1, lambda v: v - corr(v))
        changed.append(f"{p.name} ({n})")
    for p in files["ts"]:
        off = per_ts.get(p) if per_ts.get(p) is not None else unix_off
        if off is None:
            print(f"[clock-qc] {p.name}: no seconds file pairs with it; left unchanged")
            continue
        keep_original(p)
        n = _rewrite_csv_column(p, 1, lambda v, off=off: v - corr(v - off))
        changed.append(f"{p.name} ({n})")
    for p in files["coords"]:
        if "Timestamp" not in pd.read_csv(p, nrows=0).columns:
            continue
        keep_original(p)
        n = _rewrite_csv_column(p, "Timestamp", lambda v: _retime_values(v, corr, unix_off))
        changed.append(f"{p.name} ({n})")
    for kind, files_, fn in (("txt", files["txt"], retime_txt), ("log", files["log"], retime_log)):
        for p in files_:
            raw = p.read_bytes().decode("utf-8", errors="surrogateescape")
            new, st = fn(raw, corr, unix_off)
            if st["n"]:
                keep_original(p)
                _write_atomic(p, new.encode("utf-8", errors="surrogateescape"))
                changed.append(f"{p.name} ({st['n']}" + (f", {st['skipped']} left" if st["skipped"]
                                                        else "") + ")")
            elif st["skipped"]:
                print(f"[clock-qc] {p.name}: {st['skipped']} unix times left (no unix offset)")
    return changed


# ----------------------------------------------------------------------------- run
def _fmt(res):
    if "drift_s_per_h" not in res:
        return f"{res['status']} ({res.get('reason', '')})"
    return (f"{res['status']}: drift {res['drift_s_per_h']:+.2f} +- "
            f"{res.get('drift_se_s_per_h', np.nan):.2f} s/h, median D {res['median_D']:+.2f} s "
            f"(healthy ~{PHYS_LAG_S:+.2f}), {res.get('n_agree', 0)}/{res['n_reliable']} windows "
            f"agree" + (f" — {res['reason']}" if res.get("reason") else ""))


def run(op, out_dir=None, repair=False, with_offset=False, in_place=False, auto=False,
        channel=None, window=120.0, signal="auto", anchor_sec=SYNC_START_SEC, strict=False,
        dry_run=False):
    """Check one op folder; optionally repair. Returns an exit code:
    0 ok (or nothing to do), 1 failed (--strict), 3 needs manual review (--auto could
    not verify a repair), 10 repaired in place (--auto / --in-place)."""
    op = Path(op)
    out_dir = Path(out_dir) if out_dir else op / REPORT_DIR
    inp = find_inputs(op)
    if inp["coords"] is None or inp["seconds"] is None:
        print(f"[clock-qc] {op.name}: no coordinates/framewise seconds; skipped")
        return EXIT_OK
    tv, sp = video_speed(inp["coords"], inp["seconds"])
    # Two independent ephys-clock signals; a spurious window in one does not move the fit.
    signals, sources = {}, []
    if signal in ("auto", "theta") and inp["lfp"] is not None:
        te, sig, ch, fs = theta_amplitude(inp["lfp"], inp["lfp_ts"], channel=channel,
                                          expected_duration=float(tv[-1] - max(tv[0], 0.0)))
        signals["theta"] = (te, sig)
        sources.append(f"theta amplitude (LFP ch {ch}, {fs:.0f} Hz)")
    if signal in ("auto", "spikes") and inp["phy"] is not None:
        signals["spikes"] = spike_rate(inp["phy"])
        sources.append("population spike rate (phy)")
    if not signals:
        print(f"[clock-qc] {op.name}: no LFP export or phy folder; skipped")
        return EXIT_OK
    out_dir.mkdir(parents=True, exist_ok=True)
    source = " + ".join(sources)

    lags = all_lags(signals, tv, sp, window)
    res = assess(lags)
    print(f"[clock-qc] {op.name}: {source}")
    for name, p in res.get("per_signal", {}).items():
        print(f"[clock-qc]   {name:6s}: drift {p['drift_s_per_h']:+.2f} +- {p['se_s_per_h']:.2f} s/h, "
              f"{p['n_agree']}/{p['n_reliable']} windows agree"
              + ("" if p["used"] else "  (not used: windows do not agree along the recording)"))
    print(f"[clock-qc] {_fmt(res)}")
    report = dict(op=str(op), source=source, assessment=res, windows=lags.to_dict("records"))
    code = EXIT_OK

    eligible = res["status"] == "FAIL" or (repair and not auto and res["status"] == "WARN")
    if (repair or auto) and eligible and "slope" in res:
        s = pd.read_csv(inp["seconds"]).iloc[:, 1].to_numpy(float)
        s_anchor = float(s[min(int(round(anchor_sec * VIDEO_FPS)), len(s) - 1)])
        chosen = None
        if auto and abs(res["drift_s_per_h"]) > MAX_AUTO_DRIFT_S_PER_H:
            print(f"[clock-qc] a drift of {res['drift_s_per_h']:+.1f} s/h is beyond what a clock "
                  f"rate error explains; not repairing automatically")
        else:
            # --auto: slope only first (step 2 anchors its fit there), then slope + offset;
            # applied only if the repaired clock passes the same check.
            for off in ([False, True] if auto else [with_offset]):
                corr, after, fits = converge_repair(signals, tv, sp, window, res, s_anchor, off)
                entry = dict(with_offset=off, s_anchor=s_anchor, steps=fits,
                             implied_lag_at_anchor_s=res["offset_s"] + res["slope"] * s_anchor,
                             assessment_after=after)
                report.setdefault("repair_trials", []).append(entry)
                print(f"[clock-qc] repair ({'slope + offset' if off else f'slope only, anchored at {s_anchor:.1f} s'}"
                      f", {len(fits)} step{'s' if len(fits) > 1 else ''}) -> {_fmt(after)}")
                if after["status"] == "PASS" or not auto:
                    chosen = (corr, entry)
                    break
        if chosen is None:
            print("*" * 72)
            print(f"[clock-qc] {op.name}: clocks disagree and no repair verified — NOT changed. "
                  f"Re-run step 2 for this session (raw .meta + DIO needed); details in "
                  f"{out_dir / 'clock_qc.json'}.")
            print("*" * 72)
            code = EXIT_REVIEW
        else:
            corr, entry = chosen
            report["repair"] = entry
            if (in_place or auto) and dry_run:
                files = video_clock_files(op)
                entry["would_change"] = [p.name for k in files for p in files[k]]
                print(f"[clock-qc] DRY RUN — would re-time in place: {', '.join(entry['would_change'])}")
                code = EXIT_REPAIRED
            elif in_place or auto:
                entry["changed"] = apply_repair_in_place(op, corr)
                log_path = op / BACKUP_DIR / "clock_repair.json"
                history = json.loads(log_path.read_text()) if log_path.exists() else []
                history.append(dict(source=source, assessment_before=res, repair=entry))
                log_path.write_text(json.dumps(history, indent=1, default=float))
                print(f"[clock-qc] repaired IN PLACE: {', '.join(entry['changed'])}; originals in "
                      f"{op / BACKUP_DIR}. Refresh the NWB (step 4 end) to propagate.")
                code = EXIT_REPAIRED
            else:
                written = []
                for src in (inp["seconds"], inp["ts"]):
                    if src is None:
                        continue
                    df = pd.read_csv(src)
                    col = df.iloc[:, 1].to_numpy(float)
                    base = col if src == inp["seconds"] else col - (unix_offsets(
                        dict(seconds=[inp["seconds"]], ts=[src]))[1] or 0.0)
                    df.iloc[:, 1] = col - corr(base)
                    dst = out_dir / (src.stem + "_repaired.csv")
                    df.to_csv(dst, index=False)
                    written.append(str(dst))
                entry["written"] = written
                print(f"[clock-qc] repaired copies: {', '.join(written)} (originals untouched; "
                      f"use --auto to apply)")
    with open(out_dir / "clock_qc.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    if res["status"] == "FAIL" and code == EXIT_OK:
        print("*" * 72)
        print(f"[clock-qc] {op.name}: behaviour and ephys clocks DISAGREE "
              f"({res.get('drift_s_per_h', np.nan):+.1f} s/h). Re-run step 2, or "
              f"`python src/nwb/clock_qc.py {op} --auto`.")
        print("*" * 72)
        code = EXIT_FAIL if strict else EXIT_OK
    elif res["status"] == "UNKNOWN":
        print(f"[clock-qc] {op.name}: clock not verified ({res.get('reason', '')})")
    return code


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("op", help="op (session output) folder")
    ap.add_argument("--auto", action="store_true",
                    help="if the check FAILS, re-time the op folder's video-clock files in place "
                         "(framewise CSVs, coordinates Timestamp, tracker .txt/.log; originals "
                         "kept in <op>/clock_repair_backup/) when the repaired clock passes the "
                         "same check (exit 10); otherwise change nothing (exit 3)")
    ap.add_argument("--repair", action="store_true",
                    help="write repaired copies of the framewise CSVs into <op>/clock_qc/")
    ap.add_argument("--repair-offset", action="store_true",
                    help="--repair: also remove the constant offset (minus the physiological lag)")
    ap.add_argument("--in-place", action="store_true",
                    help="--repair: re-time in place, as --auto does")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --auto/--in-place: decide and report, change nothing")
    ap.add_argument("--out-dir", default=None,
                    help="where reports/copies go (default: <op>/clock_qc/)")
    ap.add_argument("--channel", type=int, default=None, help="LFP channel (default: best theta)")
    ap.add_argument("--signal", choices=["auto", "theta", "spikes"], default="auto")
    ap.add_argument("--window", type=float, default=120.0, help="window length (s)")
    ap.add_argument("--anchor-sec", type=float, default=SYNC_START_SEC,
                    help="video second at which step 2 anchored its fit (SYNC_START_SEC)")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the check fails")
    a = ap.parse_args()
    sys.exit(run(a.op, out_dir=a.out_dir, repair=a.repair, with_offset=a.repair_offset,
                 in_place=a.in_place, auto=a.auto, channel=a.channel, window=a.window,
                 signal=a.signal, anchor_sec=a.anchor_sec, strict=a.strict, dry_run=a.dry_run))
