"""rewrite_nwb_lfp: the NWB's LFP is rebuilt from the LFP_Output export on its true
rate, everything else in the file stays bit-identical, and the original is kept."""
import json
from datetime import datetime, timezone

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pynwb")
pytest.importorskip("scipy")
from pynwb import NWBFile, NWBHDF5IO, TimeSeries          # noqa: E402
from pynwb.behavior import Position, SpatialSeries        # noqa: E402

import rewrite_nwb_lfp as R                               # noqa: E402

PFX = "Rat1_20260101_120000_"
SECONDS = 240


def make_session(op, fs, stamped, old_lfp, n_ch=4):
    """A session whose LFP power and spiking follow the same slow state changes,
    with the export stamped at `stamped` Hz and the NWB holding `old_lfp`
    ('stub': 5 Hz-stamped head of the export; 'stamped': all of it at `stamped`)."""
    rng = np.random.default_rng(1)
    n = SECONDS * fs
    state = np.exp(rng.normal(0, 0.8, SECONDS + 1))
    amp = np.interp(np.arange(n) / fs, np.arange(SECONDS + 1), state)
    lfp = (rng.normal(0, 1, (n, n_ch)) * amp[:, None] * 50).astype(np.float32)
    spikes = np.sort(np.concatenate([s + rng.random(rng.poisson(40 * state[s])) for s in range(SECONDS)]))

    lo = op / "LFP_Output"
    lo.mkdir(parents=True)
    np.save(lo / f"{PFX}lfp_data.npy", lfp)
    np.save(lo / f"{PFX}lfp_timestamps.npy", np.arange(n) / stamped)
    np.save(lo / f"{PFX}channel_map.npy",
            np.array([{"index": i, "ntrode": i + 1, "channel": 1} for i in range(n_ch)], dtype=object))
    np.save(lo / f"{PFX}session_boundaries.npy",
            np.array([{"name": "Rat1_maze_merged", "start": 0, "n": n}], dtype=object))
    # look-alikes that must not be taken for the LFP's own files
    np.save(lo / f"{PFX}emg_from_lfp_timestamps.npy", np.arange(1, SECONDS * 5 + 1) / 5.0)
    np.save(lo / f"{PFX}motion_session_boundaries.npy",
            np.array([{"name": "motion", "start": 7, "n": 3}], dtype=object))

    nwb = NWBFile(session_description="test", identifier="rewrite-test",
                  session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
    for u in range(3):
        nwb.add_unit(spike_times=spikes[u::3])
    behaviour = nwb.create_processing_module("Behavior", "behaviour")
    pos = Position(name="Position")
    pt = np.arange(0, SECONDS, 1 / 30)
    pos.add_spatial_series(SpatialSeries(name="Rat", data=np.c_[np.sin(pt), np.cos(pt)],
                                         timestamps=pt, reference_frame="maze"))
    behaviour.add(pos)
    if old_lfp == "stub":
        k = SECONDS * 5
        nwb.add_acquisition(TimeSeries(name="lfp", description="lfp voltage", data=lfp[:k],
                                       unit="uV", timestamps=np.arange(1, k + 1) / 5.0))
    else:
        nwb.add_acquisition(TimeSeries(name="lfp", description="lfp voltage", data=lfp,
                                       unit="uV", timestamps=np.arange(n) / stamped))
    with NWBHDF5IO(str(op / "Rat1_20260101.nwb"), "w") as io:
        io.write(nwb)
    return lfp


def stored(op):
    with h5py.File(op / "Rat1_20260101.nwb", "r") as f:
        g = f["acquisition/lfp"]
        info = json.loads(f["processing/sleep/session_info"].attrs["description"])
        return np.asarray(g["data"][:]), float(g["starting_time"].attrs["rate"]), "timestamps" in g, info


def test_stub_is_replaced_by_the_whole_export(tmp_path):
    op = tmp_path / "20260101"
    lfp = make_session(op, fs=1500, stamped=1500, old_lfp="stub")
    rec = R.rewrite(op, 1500.0, work=tmp_path / "work")
    assert rec["status"] == "replaced" and rec["rate"]["export_hz"] == 1500
    data, rate, has_ts, info = stored(op)
    assert rate == 1500 and not has_ts and np.array_equal(data, lfp)
    assert [c["ntrode"] for c in info["channel_map"]] == [1, 2, 3, 4]
    assert info["session_boundaries"][0]["start_sample"] == 0          # not the motion look-alike
    assert info["session_boundaries"][0]["n_samples"] == len(lfp)
    backup = op / "lfp_fix_backup" / "Rat1_20260101.nwb.prelfpfix"
    new = R.fingerprint(op / "Rat1_20260101.nwb")
    assert new.pop("/processing/sleep")[0] == "group"     # new: holds session_info
    assert R.fingerprint(backup) == new                   # everything else bit-identical
    assert len(json.loads((op / "lfp_fix_backup" / "lfp_fix.json").read_text())) == 1
    # stamps were right: the export is left as it is
    assert np.array_equal(np.load(op / "LFP_Output" / f"{PFX}lfp_timestamps.npy"), np.arange(len(lfp)) / 1500)
    # and a second run has nothing to do
    assert R.rewrite(op, 1500.0, work=tmp_path / "work")["status"] == "skipped"


def test_30khz_stamped_1khz_export_is_restamped_and_resampled(tmp_path):
    op = tmp_path / "20260101"
    lfp = make_session(op, fs=1000, stamped=30000, old_lfp="stamped")
    original_ts = np.load(op / "LFP_Output" / f"{PFX}lfp_timestamps.npy")
    rec = R.rewrite(op, 1500.0, work=tmp_path / "work")
    assert rec["rate"]["export_hz"] == 1000
    assert (rec["resampled"]["up"], rec["resampled"]["down"]) == (3, 2)
    assert rec["theta_reader"].startswith("ok")
    data, rate, _, info = stored(op)
    assert rate == 1500 and data.shape == (len(lfp) * 3 // 2, lfp.shape[1])
    assert info["rewritten"]["export_rate_hz"] == 1000
    # every second output sample sits on an export sample: same signal there
    k = len(lfp) // 2
    assert np.corrcoef(data[:3 * k:3, 0], lfp[:2 * k:2, 0])[0, 1] > 0.99
    ts = np.load(op / "LFP_Output" / f"{PFX}lfp_timestamps.npy")
    assert np.array_equal(ts, np.arange(len(lfp)) / 1000.0)
    assert np.array_equal(np.load(op / "LFP_Output" / "lfp_fix_backup" / f"{PFX}lfp_timestamps.npy"),
                          original_ts)


def test_dry_run_and_step8_lfp_leave_the_folder_alone(tmp_path):
    op = tmp_path / "20260101"
    make_session(op, fs=1500, stamped=1500, old_lfp="stub")
    nwb = op / "Rat1_20260101.nwb"
    before = R.sha256(nwb)
    assert R.rewrite(op, 1500.0, work=tmp_path / "work", dry_run=True)["status"] == "rehearsed"
    assert R.sha256(nwb) == before and not (op / "lfp_fix_backup").exists()

    R.rewrite(op, 1500.0, work=tmp_path / "work2")
    with h5py.File(nwb, "r+") as f:                       # make it look like step 8's own LFP
        a = f["processing/sleep/session_info"].attrs
        info = json.loads(a["description"])
        info.pop("rewritten")
        info["lfp_reference"] = "none"
        a["description"] = json.dumps(info)
    rec = R.rewrite(op, 1500.0, work=tmp_path / "work3")
    assert rec["status"] == "skipped" and "step 8" in rec["why"]


def test_the_files_own_nwb_version_is_kept(tmp_path):
    # a file last written by a newer pynwb than the one appending now
    op = tmp_path / "20260101"
    make_session(op, fs=1500, stamped=1500, old_lfp="stub")
    with h5py.File(op / "Rat1_20260101.nwb", "r+") as f:
        f.attrs["nwb_version"] = "9.9.9"
    rec = R.rewrite(op, 1500.0, work=tmp_path / "work")
    assert rec["status"] == "replaced" and rec["nwb_version"]["kept"] == "9.9.9"
    with h5py.File(op / "Rat1_20260101.nwb", "r") as f:
        assert f.attrs["nwb_version"] == "9.9.9"
