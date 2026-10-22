"""Step-2 clock fit: time-paired fit vs the old index-paired regression.

Simulates one session: the DIO edges of the blue LED on the Trodes-derived clock, and
the blinks the ICA finds in the video on the camera clock (offset by a whole time
zone + 70 ms, 30 ppm fast, frame-quantised, detection starting 45 s in), with missed
and spurious blinks — the conditions under which Rat5_20260630's old fit came out
0.75 % fast.
"""
import numpy as np
import pytest

import sync_fit

T0 = 1_782_000_000.0          # DIO wall clock at session start (unix s)
HOURS = 7200.0                # camera clock written in local time
OFFSET = 0.070                # camera clock behind the LED by 70 ms
EPS = 30e-6                   # camera clock 30 ppm fast
PERIOD = sync_fit.LED_PERIOD_S["blue"]


def simulate(duration=3800.0, p_miss=0.02, p_extra=0.01, miss_burst=(1000.0, 1008.0),
             jump=None, seed=0):
    rng = np.random.default_rng(seed)
    t_true = np.arange(5.0, duration, PERIOD)                  # LED on-centre, true time
    dio = T0 + t_true
    cam = T0 + HOURS + OFFSET + (1 + EPS) * t_true
    if jump is not None:                                       # camera clock jumps
        cam = cam + np.where(t_true > jump[0], jump[1], 0.0)
    cam = np.round(cam * 30) / 30 + rng.uniform(-0.5, 0.5, len(cam)) / 30
    keep = (t_true > 45.0) & (rng.random(len(cam)) > p_miss)
    keep &= ~((t_true > miss_burst[0]) & (t_true < miss_burst[1]))
    video = cam[keep]
    n_extra = int(p_extra * len(video))
    video = np.sort(np.r_[video, rng.uniform(video[0], video[-1], n_extra)])
    return video, dio, t_true, cam


def test_old_index_pairing_drifts_new_fit_does_not():
    video, dio, *_ = simulate()
    old_slope, _ = sync_fit.legacy_index_pair_fit(video, dio)
    fit = sync_fit.fit_led_clock(video, dio, PERIOD)
    true_slope = 1 / (1 + EPS)
    # the old fit is off by far more than a crystal (the bug)
    assert abs(old_slope - true_slope) > 1e-3
    # the new one recovers the clock rate to within 2 ppm
    assert abs(fit.slope - true_slope) < 2e-6
    assert fit.hour_shift == -HOURS
    assert fit.resid_med_abs < 1 / 30
    assert sync_fit.qc_failures(fit) == []


def test_mapping_puts_frames_on_the_led_clock():
    video, dio, t_true, cam = simulate()
    fit = sync_fit.fit_led_clock(video, dio, PERIOD)
    x0 = video[0]
    mapping, decision = sync_fit.frame_mapping(fit, x0, anchor="auto")
    assert decision["applied"] and abs(decision["delta_s"] + OFFSET) < 1 / 30
    # every true LED event, seen on the camera clock, lands on its DIO time
    # (expressed in the camera's whole-hour convention) within a frame
    err = mapping(cam) - (dio + HOURS)
    assert np.max(np.abs(err)) < 1.5 / 30
    # 'pc' keeps the camera anchor: off by the 70 ms the LED measures, no drift
    mapping_pc, _ = sync_fit.frame_mapping(fit, x0, anchor="pc")
    err_pc = mapping_pc(cam) - (dio + HOURS)
    assert abs(np.median(err_pc) - OFFSET) < 1 / 30
    n = len(err_pc) // 10                       # no drift: same error at start and end
    assert abs(np.median(err_pc[:n]) - np.median(err_pc[-n:])) < 0.2 / 30


def test_clock_jump_fails_qc():
    video, dio, *_ = simulate(jump=(2000.0, 0.25))
    fit = sync_fit.fit_led_clock(video, dio, PERIOD)
    reasons = " ".join(sync_fit.qc_failures(fit))
    assert "segment" in reasons or "residual" in reasons


def test_large_rate_error_is_tracked_and_refused():
    rng = np.random.default_rng(1)
    t = np.arange(5.0, 3600.0, PERIOD)
    dio = T0 + t
    video = T0 + (1 + 1e-3) * t + rng.uniform(-0.5, 0.5, len(t)) / 30
    fit = sync_fit.fit_led_clock(video, dio, PERIOD)
    assert abs(fit.slope - 1 / (1 + 1e-3)) < 5e-6
    assert any("ppm" in r for r in sync_fit.qc_failures(fit))


def test_too_few_events():
    with pytest.raises(sync_fit.SyncFitError):
        sync_fit.fit_led_clock([1.0, 2.0], [1.0, 2.0], PERIOD)
