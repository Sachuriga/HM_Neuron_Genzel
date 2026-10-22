"""clock_qc: a stretched video clock is detected and repaired from the data alone."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

import clock_qc

DT = clock_qc.DT


def locomotion(duration, seed=0):
    """A bursty running-speed-like signal on a true-time grid."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, DT)
    state = np.cumsum(rng.random(len(t)) < 0.004) % 2           # run / rest bouts
    return t, gaussian_filter1d(state * (1 + rng.random(len(t))), 8) + 0.05 * rng.random(len(t))


def session(stretch, offset=0.0, phys=clock_qc.PHYS_LAG_S, duration=3600.0):
    """Ephys signal on the ephys clock; video speed labelled on a stretched clock."""
    t, x = locomotion(duration)
    ephys_t, ephys = t, x                                         # theta ~ speed, leads by `phys`
    video_t = (t + phys) * (1 + stretch) + offset                 # same moment, later video label
    return ephys_t, ephys, video_t, x


def test_healthy_session_passes():
    et, e, vt, v = session(stretch=0.0)
    res = clock_qc.assess(clock_qc.window_lags(et, e, vt, v))
    assert res["status"] == "PASS"
    assert abs(res["median_D"] - clock_qc.PHYS_LAG_S) < 0.1


def test_stretched_clock_fails_and_repairs():
    et, e, vt, v = session(stretch=0.0075)                        # Rat5_20260630
    res = clock_qc.assess(clock_qc.window_lags(et, e, vt, v))
    assert res["status"] == "FAIL"
    assert abs(res["drift_s_per_h"] - 27.0) < 1.5
    vt_new = clock_qc.repair_seconds(vt, res, s_anchor=vt[0])
    res2 = clock_qc.assess(clock_qc.window_lags(et, e, vt_new, v))
    assert res2["status"] == "PASS"
    assert abs(res2["drift_s_per_h"]) < 0.3


def _windows(signal, t, D, r=0.6):
    import pandas as pd
    return pd.DataFrame(dict(signal=signal, t=t, D=D, r_peak=r, r_zero=0.1, edge=False))


def test_spurious_windows_and_a_useless_signal_do_not_fail_a_healthy_session():
    """Rat6_20260701: a third of the spike windows on spurious peaks, theta all noise."""
    import pandas as pd
    rng = np.random.default_rng(1)
    t = np.arange(60.0, 3600.0, 120.0)
    D = clock_qc.PHYS_LAG_S + 0.05 * rng.standard_normal(len(t))
    bad = rng.choice(len(t), size=len(t) // 3, replace=False)
    D[bad] = rng.uniform(-25, -2, len(bad))                       # all pulling one way
    lags = pd.concat([_windows("spikes", t, D),
                      _windows("theta", t, rng.uniform(-25, 25, len(t)), r=0.35)])
    res = clock_qc.assess(lags)
    assert res["status"] == "PASS", res
    assert res["signals_used"] == ["spikes"]


def test_signals_that_disagree_are_never_failed():
    import pandas as pd
    t = np.arange(60.0, 3600.0, 120.0)
    lags = pd.concat([_windows("spikes", t, 0.15 + 0 * t),
                      _windows("theta", t, 0.15 + 3.0 / 3600 * t)])
    res = clock_qc.assess(lags)
    assert res["status"] == "WARN" and "disagree" in res["reason"]


def test_patchy_coverage_gives_no_verdict():
    """Agreeing windows only in the first half: the second half is unverified."""
    import pandas as pd
    rng = np.random.default_rng(2)
    t = np.arange(60.0, 3600.0, 120.0)
    D = np.where(t < 1800, 0.15, rng.uniform(-25, 25, len(t)))
    res = clock_qc.assess(_windows("spikes", t, D))
    assert res["status"] == "UNKNOWN"


def test_window_beyond_the_search_range_is_not_used():
    et, e, vt, v = session(stretch=0.0075)                        # D reaches ~27 s > 25 s
    lags = clock_qc.window_lags(et, e, vt, v)
    assert lags.edge.any()
    res = clock_qc.assess(lags)
    assert abs(res["drift_s_per_h"] - 27.0) < 0.5                 # edge windows not pulling it


def test_lfp_rate_trust():
    """30 kHz stamps on a 1 kHz LFP are caught; an ephys recording merely longer or
    shorter than the video (Rat6_20260708: 1.22x, Rat6_20260701: 0.58x) is not."""
    assert not clock_qc.trusted_lfp_rate(30000.0, 3_800_000, 3800.0)
    assert clock_qc.trusted_lfp_rate(1500.0, int(3908 * 1500), 3215.0)
    assert clock_qc.trusted_lfp_rate(1500.0, int(2214 * 1500), 3807.0)
    assert clock_qc.trusted_lfp_rate(1000.0, 3_800_000, 3800.0)


def test_an_unused_signal_that_contradicts_blocks_the_verdict():
    """Rat6_20260708: spikes say -0.65 s/h; theta, too patchy to be used, says -1.56."""
    import pandas as pd
    t = np.arange(60.0, 3600.0, 120.0)
    rng = np.random.default_rng(4)
    theta = 0.15 - 1.56 / 3600 * t
    theta[rng.choice(len(t), size=len(t) // 2, replace=False)] = rng.uniform(-25, 25, len(t) // 2)
    theta[(t > 900) & (t < 2700)] = rng.uniform(-25, 25, ((t > 900) & (t < 2700)).sum())
    lags = pd.concat([_windows("spikes", t, 0.15 - 0.65 / 3600 * t),
                      _windows("theta", t, theta)])
    res = clock_qc.assess(lags)
    assert not res["per_signal"]["theta"]["used"]
    assert res["status"] == "WARN" and "not used" in res["reason"], res
