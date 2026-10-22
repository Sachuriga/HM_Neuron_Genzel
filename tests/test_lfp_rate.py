"""theta_events: an LFP whose rate or length cannot be the session's is not used."""
import numpy as np
import pytest

pytest.importorskip("pynwb")             # theta_events reads NWBs
import theta_events as TE                # noqa: E402


def source(n, fs, boundaries=None):
    return dict(fs=fs, data=np.zeros((n, 1), np.float32), boundaries=boundaries)


def test_good_lfp_is_accepted():
    assert TE._lfp_rate_problem(source(3_600_000, 1000.0), np.array([0.0, 3600.0])) is None


def test_stub_and_misstamped_lfp_are_rejected():
    t = np.array([-80.0, 3600.0])
    assert "not an LFP rate" in TE._lfp_rate_problem(source(18_000, 5.0), t)      # EMG-stamped stub
    assert "last" in TE._lfp_rate_problem(source(3_600_000, 30000.0), t)        # 30 kHz stamps


def test_multi_session_export_is_only_rate_checked():
    t = np.array([0.0, 3600.0])
    assert TE._lfp_rate_problem(source(10_800_000, 1000.0, boundaries=[{}]), t) is None


def test_ephys_longer_or_shorter_than_the_video_is_accepted():
    assert TE._lfp_rate_problem(source(int(3908 * 1500), 1500.0), np.array([0.0, 3215.0])) is None
    assert TE._lfp_rate_problem(source(int(2214 * 1500), 1500.0), np.array([0.0, 3807.0])) is None
