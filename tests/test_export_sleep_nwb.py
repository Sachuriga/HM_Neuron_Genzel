"""export_sleep_nwb (step 8 backfill): an old export's LFP stamps are checked
against the session's spikes before they are packaged into the NWB."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pynwb")
from pynwb import NWBFile, NWBHDF5IO                       # noqa: E402

import export_sleep_nwb as E                               # noqa: E402

PFX = "Rat1_20260101_120000_"
FS, SECONDS = 1000, 200


def make_op(tmp_path, stamped, with_nwb=True):
    op = tmp_path / "20260101"
    lo = op / "LFP_Output"
    lo.mkdir(parents=True)
    n = FS * SECONDS
    np.save(lo / f"{PFX}lfp_data.npy", np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32))
    np.save(lo / f"{PFX}lfp_timestamps.npy", np.arange(n) / stamped)
    np.save(lo / f"{PFX}channel_map.npy",
            np.array([{"index": i, "ntrode": i + 1, "channel": 1} for i in range(4)], dtype=object))
    np.save(lo / f"{PFX}session_boundaries.npy",
            np.array([{"name": "Rat1_maze_merged", "start": 0, "n": n}], dtype=object))
    if with_nwb:                       # step 4/u ran first: units, but no LFP yet
        nwb = NWBFile(session_description="t", identifier="t",
                      session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
        spikes = np.sort(np.random.default_rng(1).uniform(0, SECONDS - 0.01, 3000))
        nwb.add_unit(spike_times=spikes)
        with NWBHDF5IO(str(op / "Rat1_20260101.nwb"), "w") as io:
            io.write(nwb)
    return op


def lfp_in(op):
    with h5py.File(op / "Rat1_20260101.nwb", "r") as f:
        info = json.loads(f["processing/sleep/session_info"].attrs["description"])
        return np.asarray(f["acquisition/lfp/timestamps"][:]), info


def test_30khz_stamps_are_packaged_at_the_rate_the_spikes_imply(tmp_path):
    op = make_op(tmp_path, stamped=30000)
    assert E.run(op) == 0
    ts, info = lfp_in(op)
    assert np.allclose(ts, np.arange(FS * SECONDS) / FS)
    assert [c["ntrode"] for c in info["channel_map"]] == [1, 2, 3, 4]
    assert "lfp_reference" not in info                    # step l may still replace it


def test_right_stamps_are_packaged_as_they_are(tmp_path):
    op = make_op(tmp_path, stamped=FS)
    assert E.run(op) == 0
    ts, _ = lfp_in(op)
    assert np.array_equal(ts, np.load(op / "LFP_Output" / f"{PFX}lfp_timestamps.npy"))


def test_acquisition_rate_stamps_with_nothing_to_check_are_not_packaged(tmp_path):
    op = make_op(tmp_path, stamped=30000, with_nwb=False)
    assert E.run(op) == 0
    assert not list(op.glob("*.nwb"))
