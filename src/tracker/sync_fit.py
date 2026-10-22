"""Robust LED/DIO clock fit for step 2 (``Video_LED_Sync_using_ICA.py``).

The original fit paired the k-th LED blink seen in the video with the k-th DIO edge
and regressed one on the other. A single missed or extra blink (LED occluded, a
dropped frame, a spurious ICA edge) shifts every later pair by a whole LED period,
and the global regression absorbs the accumulated shift into its slope:
Rat5_20260630 came out with its video clock 0.75 % fast, i.e. behaviour drifting
~27 s per hour against the spikes, and nothing flagged it.

Here blinks are paired with DIO edges by TIME:

1. a whole-hour difference between the two clocks (the camera and Trodes writers
   use different time-zone conventions) is removed first;
2. in short sliding windows, each blink's offset to its nearest DIO edge is taken
   modulo the LED period and averaged on the circle. Missed or extra blinks only
   drop out of (or barely dilute) the average; they cannot shift it by a period;
3. the window offsets are unwrapped and fitted with a line (slope = clock-rate
   ratio), with outlier windows removed;
4. every blink is paired with the DIO edge that line predicts, pairs further than
   ``tol`` periods away are rejected, and the line is refitted on the pairs.

Both LED trains are periodic, so the absolute offset between the clocks is only
defined up to a whole number of LED periods. ``frame_mapping`` therefore keeps the
camera-clock anchor of the original pipeline at ``x0`` and moves it only by the
LED-measured sub-period offset, when that is unambiguous (``anchor="auto"``).

numpy only, so it can be tested without OpenCV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

FRAME_S = 1 / 30.0
LED_PERIOD_S = {"blue": 0.4, "red": 2.0}      # 2.5 Hz and 0.5 Hz


class SyncFitError(RuntimeError):
    """The LED trains cannot be aligned at all (too few events, no overlap)."""


@dataclass
class SyncFit:
    """dio ~= intercept + slope * video, both in seconds."""

    slope: float
    intercept: float
    hour_shift: float            # whole-hour clock difference removed before pairing
    period: float
    n_video: int
    n_dio: int
    n_pairs: int
    match_frac_video: float      # paired blinks / blinks inside the DIO span
    match_frac_dio: float        # paired DIO edges / DIO edges inside the video span
    resid_med_abs: float         # s
    resid_p95_abs: float         # s
    segment_resid: list = field(default_factory=list)    # [(video t start, median resid)]
    window_offsets: list = field(default_factory=list)   # [(video t, offset, resultant)]
    notes: list = field(default_factory=list)

    def predict(self, video_t):
        """Video-clock times -> DIO-clock times."""
        return self.intercept + self.slope * np.asarray(video_t, float)

    def summary(self):
        d = asdict(self)
        d["slope_ppm"] = (self.slope - 1.0) * 1e6
        d["drift_s_per_h"] = (self.slope - 1.0) * 3600.0
        return d


def _nearest(sorted_ref, x):
    """Index of the element of ``sorted_ref`` nearest to each ``x``."""
    i = np.clip(np.searchsorted(sorted_ref, x), 1, len(sorted_ref) - 1)
    left, right = sorted_ref[i - 1], sorted_ref[i]
    return np.where(np.abs(x - left) <= np.abs(right - x), i - 1, i)


def _clean(t):
    t = np.asarray(t, float)
    return np.sort(t[np.isfinite(t)])


def fit_led_clock(video_t, dio_t, period, window=30.0, step=10.0, tol=0.25,
                  min_events=10, min_resultant=0.5, segment=300.0):
    """Fit the camera-clock -> DIO-clock line from two LED edge trains.

    Args:
        video_t: blink times seen in the video (camera clock, s).
        dio_t: the same LED's DIO edge times (Trodes-derived wall clock, s).
        period: LED period in s (0.4 blue, 2.0 red).
        window, step: sliding-window length and step for the phase tracking (s).
            The offset may change by less than ``period / 2`` per step, i.e. the
            clocks may differ in rate by up to ``period / (2 * step)`` (2 % here).
        tol: maximum |residual| of an accepted pair, in periods.
    """
    v, d = _clean(video_t), _clean(dio_t)
    if len(v) < min_events or len(d) < min_events:
        raise SyncFitError(f"too few LED events (video {len(v)}, DIO {len(d)})")

    notes = []
    hour_shift = 3600.0 * np.round((np.median(d) - np.median(v)) / 3600.0)
    if hour_shift:
        notes.append(f"removed a whole-hour clock difference of {hour_shift:+.0f} s")
    vs = v + hour_shift

    t_lo, t_hi = max(vs[0], d[0]), min(vs[-1], d[-1])
    if t_hi - t_lo < 2 * window:
        raise SyncFitError(f"video and DIO trains overlap for only {max(t_hi - t_lo, 0):.1f} s")

    # --- 2. local offset modulo the period, averaged on the circle -------------
    centres = np.arange(t_lo + window / 2, t_hi - window / 2 + 1e-9, step)
    offs = np.full(len(centres), np.nan)
    res = np.zeros(len(centres))
    for k, c in enumerate(centres):
        sel = vs[(vs >= c - window / 2) & (vs < c + window / 2)]
        if len(sel) < min_events:
            continue
        z = np.exp(2j * np.pi * (d[_nearest(d, sel)] - sel) / period).mean()
        offs[k] = period * np.angle(z) / (2 * np.pi)
        res[k] = abs(z)
    good = np.isfinite(offs) & (res >= min_resultant)
    if good.sum() < 3:
        raise SyncFitError("the LED phase could not be tracked (blink trains do not "
                           "line up in any window)")

    # --- 3. unwrap and fit a line, dropping outlier windows --------------------
    tw = centres[good]
    ow = np.unwrap(2 * np.pi * offs[good] / period) * period / (2 * np.pi)
    tref = float(np.median(tw))
    keep = np.ones(len(tw), bool)
    for _ in range(3):
        b1, a1 = np.polyfit(tw[keep] - tref, ow[keep], 1)
        r = ow - (a1 + b1 * (tw - tref))
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-12
        new_keep = np.abs(r) <= max(5 * 1.4826 * mad, FRAME_S / 2)
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if (~keep).any():
        notes.append(f"{(~keep).sum()} of {len(keep)} phase windows rejected as outliers")

    # --- 4. pair every blink with its predicted DIO edge, refit ----------------
    slope_off, icpt_off = b1, a1                      # offset(t) = icpt + slope*(t - tref)
    for _ in range(3):
        target = vs + icpt_off + slope_off * (vs - tref)
        j = _nearest(d, target)
        r = d[j] - target
        ok = np.abs(r) < tol * period
        # a DIO edge claimed by two blinks keeps the closer one
        order = np.argsort(np.abs(r))
        seen = np.zeros(len(d), bool)
        uniq = np.zeros(len(vs), bool)
        for i in order:
            if ok[i] and not seen[j[i]]:
                seen[j[i]] = True
                uniq[i] = True
        if uniq.sum() < min_events:
            raise SyncFitError("too few blink/DIO pairs after time-based matching")
        b, a = np.polyfit(vs[uniq] - tref, d[j[uniq]] - vs[uniq], 1)
        if abs(b - slope_off) < 1e-9 and abs(a - icpt_off) < 1e-6:
            break
        slope_off, icpt_off = b, a

    pv, pd_ = v[uniq], d[j[uniq]]
    slope = 1.0 + slope_off
    intercept = hour_shift + icpt_off - slope_off * (tref - hour_shift)
    # i.e. d = intercept + slope * v  (v on the camera clock)
    resid = pd_ - (intercept + slope * pv)

    span_v = (vs >= d[0]) & (vs <= d[-1])
    span_d = (d >= vs[0]) & (d <= vs[-1])
    seg = []
    for s0 in np.arange(pv[0], pv[-1], segment):
        m = (pv >= s0) & (pv < s0 + segment)
        if m.sum() >= min_events:
            seg.append((float(s0), float(np.median(resid[m]))))
    return SyncFit(
        slope=float(slope), intercept=float(intercept), hour_shift=float(hour_shift),
        period=float(period), n_video=len(v), n_dio=len(d), n_pairs=int(uniq.sum()),
        match_frac_video=float(uniq.sum() / max(span_v.sum(), 1)),
        match_frac_dio=float(uniq.sum() / max(span_d.sum(), 1)),
        resid_med_abs=float(np.median(np.abs(resid))),
        resid_p95_abs=float(np.percentile(np.abs(resid), 95)),
        segment_resid=seg,
        window_offsets=[(float(t - hour_shift), float(o), float(q))
                        for t, o, q in zip(tw, ow, res[good])],
        notes=notes)


def qc_failures(fit, max_slope_dev=2e-4, max_resid_med=FRAME_S, max_resid_p95=2 * FRAME_S,
                max_segment_resid=FRAME_S, min_match=0.8):
    """Reasons this fit must not be used (empty list = fit is fine).

    Two crystal clocks differ by tens of ppm, so ``|slope - 1|`` beyond 2e-4
    (0.7 s per hour) means mispaired edges, not a real rate difference.
    """
    out = []
    dev = fit.slope - 1.0
    if abs(dev) > max_slope_dev:
        out.append(f"clock-rate ratio off by {dev * 1e6:+.0f} ppm ({dev * 3600:+.2f} s/h); "
                   f"limit {max_slope_dev * 1e6:.0f} ppm")
    if fit.resid_med_abs > max_resid_med:
        out.append(f"median |residual| {fit.resid_med_abs * 1000:.1f} ms > {max_resid_med * 1000:.1f} ms")
    if fit.resid_p95_abs > max_resid_p95:
        out.append(f"95th pct |residual| {fit.resid_p95_abs * 1000:.1f} ms > {max_resid_p95 * 1000:.1f} ms")
    bad_seg = [(t, r) for t, r in fit.segment_resid if abs(r) > max_segment_resid]
    if bad_seg:
        out.append(f"{len(bad_seg)} segment(s) offset by more than one frame — the camera "
                   f"clock jumps there (first at video t={bad_seg[0][0]:.0f} s, "
                   f"{bad_seg[0][1] * 1000:+.0f} ms)")
    if min(fit.match_frac_video, fit.match_frac_dio) < min_match:
        out.append(f"only {fit.match_frac_video:.0%} of blinks / {fit.match_frac_dio:.0%} of "
                   f"DIO edges paired (need {min_match:.0%})")
    return out


def frame_mapping(fit, x0, anchor="auto", confirm=None, max_confirm_diff=FRAME_S):
    """Map camera-clock frame times onto the session clock convention of step 2.

    The original pipeline anchored the corrected clock on the camera clock at ``x0``
    (``x0 + slope * (t - x0)``) because the LED trains cannot fix whole periods. The
    LED does measure the sub-period offset at ``x0``, ``delta``; it is applied when
    unambiguous:

        anchor="pc"    keep the camera-clock anchor (original behaviour, new slope);
        anchor="led"   always apply delta;
        anchor="auto"  apply delta if |delta| < period/4, or if a second LED's fit
                       (``confirm``) measures the same delta within one frame.

    Returns (mapping function, decision dict).
    """
    delta = float(fit.predict(x0) - fit.hour_shift - x0)
    decision = dict(anchor=anchor, delta_s=delta, applied=False, reason="")
    if anchor == "led":
        decision.update(applied=True, reason="forced")
    elif anchor == "auto":
        if abs(delta) < fit.period / 4:
            decision.update(applied=True, reason=f"|delta| < period/4 ({fit.period / 4:.2f} s)")
        elif confirm is not None:
            d2 = float(confirm.predict(x0) - confirm.hour_shift - x0)
            decision["confirm_delta_s"] = d2
            if abs(d2 - delta) <= max_confirm_diff:
                decision.update(applied=True, reason="confirmed by the other LED")
            else:
                decision["reason"] = "ambiguous and not confirmed by the other LED"
        else:
            decision["reason"] = "ambiguous (|delta| >= period/4) and no second LED"
    shift = delta if decision["applied"] else 0.0
    slope = fit.slope

    def mapping(t):
        t = np.asarray(t, float)
        return x0 + shift + slope * (t - x0)

    return mapping, decision


def legacy_index_pair_fit(video_t, dio_t):
    """The pre-fix fit, kept for comparison: trim, pair edge k with blink k, regress.

    Mirrors ``trim_ts_before_first_overlap`` + ``pred_dio_ts_from_ica_ts_and_verify``
    (slope and intercept of dio ~ video on index-paired, trimmed trains).
    """
    v, d = _clean(video_t), _clean(dio_t)
    d_tr = d[1:-2]
    v_tr = v[v > d[0]][:len(d_tr)]
    n = min(len(d_tr), len(v_tr))
    b, a = np.polyfit(v_tr[:n], d_tr[:n], 1)
    return float(b), float(a)
