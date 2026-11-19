"""shared_noise.py: removing interference that reaches the tetrodes with different
gains (which a common reference cannot remove), without touching the spikes."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("spikeinterface")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sorter"))
import shared_noise as sn                                   # noqa: E402

FS = 30000
N_TETRODES = 16
GROUPS = np.repeat(np.arange(N_TETRODES), 4)                # tetrode of each of the 64 channels
NOISE_UV = 10.0
SPIKE = np.array([-1.0, -0.6, 0.2, 0.35, 0.2, 0.1])         # trough first, short rebound
PROFILE = np.array([1.0, 0.7, 0.5, 0.3])                    # the 4 wires of the tetrode


def channel_noise(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, NOISE_UV, (n_samples, GROUPS.size)).astype(np.float32)


def interference(n_samples, seed=1):
    """A pulse train (one 300 uV pulse every 8 ms) with its own gain on every tetrode,
    as left behind by a gain-1 reference: gain - 1 = -0.6 .. +1.4."""
    rng = np.random.default_rng(seed)
    pulses = np.zeros(n_samples, dtype=np.float32)
    pulses[::240] = 300.0
    pulses = np.convolve(pulses, [0.3, 1.0, -0.5, -0.2], mode="same")
    gain = np.repeat(rng.uniform(-0.6, 1.4, N_TETRODES), 4).astype(np.float32)
    return pulses[:, None] * gain[None, :]


def add_spikes(traces, tetrode, times, amplitude_uv):
    channels = np.flatnonzero(GROUPS == tetrode)
    for t in times:
        traces[t:t + SPIKE.size, channels] += amplitude_uv * SPIKE[:, None] * PROFILE[None, :]
    return traces


def test_interference_with_a_gain_per_tetrode_is_removed():
    noise = channel_noise(FS)
    dirty = noise + interference(FS)
    cleaned, strengths = sn.remove_shared_noise(dirty, GROUPS, max_components=2)

    assert strengths and strengths[0] > sn.TRIGGER_STRENGTH
    left_before = np.sqrt(((dirty - noise) ** 2).mean())
    left_after = np.sqrt(((cleaned - noise) ** 2).mean())
    assert left_after < 0.15 * left_before                   # the interference is gone ...
    assert left_after < 0.3 * NOISE_UV                       # ... down to a fraction of the noise


def test_a_spike_survives_the_removal():
    spike_times = np.arange(1000, FS - 1000, 1500)
    without = channel_noise(FS) + interference(FS)
    with_spikes = add_spikes(without.copy(), tetrode=5, times=spike_times, amplitude_uv=150.0)

    patterns, _ = sn.shared_patterns(with_spikes, GROUPS, max_components=2)
    assert patterns

    def clean(y):
        for w in patterns:
            y = y - sn.weighted_median_score(y, w)[:, None] * w[None, :]
        return y

    spike_after = clean(with_spikes) - clean(without)        # what is left of the spikes alone
    own = np.flatnonzero(GROUPS == 5)
    kept = spike_after[spike_times][:, own[0]] / -150.0      # the trough, on the biggest wire
    assert np.median(kept) > 0.97
    assert kept.min() > 0.85
    other = np.abs(spike_after[spike_times][:, GROUPS != 5])
    assert np.median(other.max(axis=1)) < 0.5 * NOISE_UV     # and it does not appear elsewhere


def test_a_block_without_interference_comes_back_unchanged():
    quiet = channel_noise(FS)
    for tetrode in range(N_TETRODES):                        # every tetrode has its own units
        add_spikes(quiet, tetrode, np.arange(500 + 37 * tetrode, FS - 500, 3000), 200.0)
    cleaned, strengths = sn.remove_shared_noise(quiet, GROUPS, max_components=2)
    assert strengths == []
    assert np.array_equal(cleaned, quiet)


def test_strong_units_on_a_few_tetrodes_are_not_a_pattern():
    """Two tetrodes that record the SAME big unit share a signal, but only they do:
    that is not interference, and must not be subtracted."""
    traces = channel_noise(FS)
    times = np.arange(300, FS - 300, 150)                    # 200 Hz, 400 uV: a huge variance
    add_spikes(traces, 2, times, 400.0)
    add_spikes(traces, 3, times, 400.0)
    patterns, _ = sn.shared_patterns(traces, GROUPS, max_components=2)
    assert patterns == []


def make_recording(traces):
    import spikeinterface.full as si
    rec = si.NumpyRecording(traces_list=[traces], sampling_frequency=FS)
    rec.set_property("group", GROUPS)
    return rec


def test_recording_gives_the_same_samples_however_it_is_read():
    """Blocks are fixed on the sample axis, so chunked reading (rec.save, Phy, the
    rebuild tool) cannot change the result. 2.4 blocks: the last one takes the rest."""
    n = int(2.4 * FS)
    traces = np.rint(channel_noise(n) + interference(n)).astype(np.int16)
    rec = sn.remove_shared_noise_recording(make_recording(traces), max_components=2)

    whole = rec.get_traces()
    assert whole.dtype == np.int16 and whole.shape == traces.shape
    assert not np.array_equal(whole, traces)                 # it did something
    for start, end in ((0, 1), (0, FS), (FS - 7, FS + 7), (12345, 54321), (2 * FS, n), (n - 3, n)):
        assert np.array_equal(rec.get_traces(start_frame=start, end_frame=end), whole[start:end])
    some = [3, 0, 17]
    assert np.array_equal(rec.get_traces(start_frame=100, end_frame=40000, channel_ids=some),
                          whole[100:40000][:, some])

    # last block = samples FS .. n (1.4 blocks), cleaned as ONE block
    expected, _ = sn.remove_shared_noise(traces[FS:], GROUPS, max_components=2)
    assert np.array_equal(whole[FS:], np.rint(expected).astype(np.int16))
    assert "interference found and removed in 2 of 2 blocks" in rec.removed_summary()


def test_recording_is_off_with_zero_components_or_too_few_tetrodes():
    traces = np.rint(channel_noise(FS) + interference(FS)).astype(np.int16)
    off = sn.remove_shared_noise_recording(make_recording(traces), max_components=0)
    assert np.array_equal(off.get_traces(), traces)

    import spikeinterface.full as si
    few = si.NumpyRecording(traces_list=[traces[:, :16]], sampling_frequency=FS)
    few.set_property("group", GROUPS[:16])                   # 4 tetrodes < MIN_TETRODES
    assert np.array_equal(sn.remove_shared_noise_recording(few).get_traces(), traces[:, :16])


def test_config_key_is_read_and_off_by_default(tmp_path):
    sorting = pytest.importorskip("sorting")
    assert sorting.resolve_sorting_params({})["shared_noise_components"] == 0

    config_file = tmp_path / "hm_tracker_paths.txt"
    config_file.write_text("SORTER = mountainsort4\nSHARED_NOISE_COMPONENTS = 2\n")
    config = sorting.load_sorting_config(config_file, quiet=True)
    assert sorting.resolve_sorting_params(config)["shared_noise_components"] == 2
