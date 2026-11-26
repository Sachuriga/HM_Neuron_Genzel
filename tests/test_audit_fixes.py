"""Regression tests for the bugs fixed in the project-wide review: one test per
fix, each named after what used to go wrong."""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ------------------------------------------------------------------ create_nwb
@pytest.fixture(scope="module")
def create_nwb():
    pytest.importorskip("pynwb")
    # colorama.init() wraps sys.stdout on Windows, which pytest is capturing: make
    # create_nwb take its no-colour fallback instead
    saved = sys.modules.get("colorama", "absent")
    sys.modules["colorama"] = None
    try:
        return importlib.import_module("create_nwb")
    finally:
        if saved == "absent":
            sys.modules.pop("colorama", None)
        else:
            sys.modules["colorama"] = saved


def test_time_zero_row_when_the_video_starts_after_time_zero(create_nwb):
    # every time is >= 0: the old lookup of "the row before the first non-negative
    # one" raised KeyError(-1)
    assert create_nwb.find_index_time_zero(pd.DataFrame({"Time (seconds)": [0.4, 0.43, 0.47]})) == 0
    # times across zero: the row nearest to it, as before
    assert create_nwb.find_index_time_zero(pd.DataFrame({"Time (seconds)": [-0.05, -0.02, 0.01]})) == 2
    # every time negative (old: IndexError), rows without a time, no time at all
    assert create_nwb.find_index_time_zero(pd.DataFrame({"Time (seconds)": [-3.0, -2.0]})) == 1
    assert create_nwb.find_index_time_zero(pd.DataFrame({"Time (seconds)": [np.nan, 0.2, 0.1]})) == 2
    assert create_nwb.find_index_time_zero(pd.DataFrame({"Time (seconds)": [np.nan, np.nan]})) is None


def test_session_seconds_are_not_taken_for_a_unix_time(create_nwb):
    # the coordinate file's Timestamp holds session seconds when the tracker read
    # the seconds file; as a unix time that was a session start in 1970
    assert create_nwb.unix_time_or_none(0.033) is None
    assert create_nwb.unix_time_or_none(float("nan")) is None
    assert create_nwb.unix_time_or_none("N/A") is None
    assert create_nwb.unix_time_or_none(1784020013.68) == 1784020013.68


# --------------------------------------------------------------- spike_metrics
def test_acg_of_a_fast_cell_is_the_same_on_every_run():
    import spike_metrics
    spikes = np.sort(np.random.default_rng(3).uniform(0, 600, 60000))   # over the 50000 cap
    first, _ = spike_metrics.autocorrelogram(spikes, 0.05, 0.0005)
    again, _ = spike_metrics.autocorrelogram(spikes, 0.05, 0.0005)
    assert np.array_equal(first, again)


# -------------------------------------------------------------- session_prefix
def test_file_prefix_lists_the_folder_once(tmp_path, monkeypatch):
    import session_prefix
    (tmp_path / "Rat6_HM_20260707_091045_task_ms5_sorting_output").mkdir()
    (tmp_path / "Rat6_20260707.nwb").write_text("")
    session_prefix._file_prefix.cache_clear()
    assert session_prefix.file_prefix(tmp_path) == "Rat6_20260707_091045_"

    def no_more_listing(*a, **k):
        raise AssertionError("the folder was listed again")
    monkeypatch.setattr(Path, "glob", no_more_listing)
    assert session_prefix.file_prefix(tmp_path) == "Rat6_20260707_091045_"
    assert session_prefix.prefixed(tmp_path, "summary.pdf") == "Rat6_20260707_091045_summary.pdf"


# ------------------------------------------------------------------- add_units
def test_step_u_does_not_take_a_macos_sidecar_for_the_nwb(tmp_path):
    pytest.importorskip("pynwb")
    import add_units
    (tmp_path / "._Rat5_20260630.nwb").write_text("not hdf5")        # sorts first
    (tmp_path / "Rat5_20260630.tmp.nwb").write_text("")
    (tmp_path / "Rat5_20260630.nwb").write_text("")
    assert add_units.find_nwb_file(tmp_path).name == "Rat5_20260630.nwb"


# ---------------------------------------------------------------------- runner
def test_worker_gate_uses_the_limits_of_the_config_file(monkeypatch, capsys):
    import runner
    monkeypatch.delenv("DISABLE_RESOURCE_CHECK", raising=False)
    load = {"cpu": 50}                            # busy now, idle after one wait
    monkeypatch.setattr(runner, "get_cpu_load", lambda: load["cpu"])
    monkeypatch.setattr(runner, "get_gpu_load", lambda: 0)
    monkeypatch.setattr(runner, "get_mem_usage", lambda: 10)
    waits = []

    def waited(seconds):
        waits.append(seconds)
        load["cpu"] = 5
    monkeypatch.setattr(runner.time, "sleep", waited)
    monkeypatch.setenv("MAX_CPU", "40")           # as load_config() exports it, after import
    monkeypatch.setenv("WAIT_SECONDS", "7")
    runner.wait_for_resources()
    # with the import-time limit (90 %) a 50 % load never waited at all
    assert waits == [7]
    assert "High CPU: 50%" in capsys.readouterr().out


# ----------------------------------------------------------------- plot_trials
def test_local_speed_is_what_the_per_sample_loop_gave():
    import matplotlib
    matplotlib.use("Agg")
    import plot_trials

    def loop(v, half=15):
        return np.array([np.median(v[max(0, i - half):min(v.size, i + half + 1)])
                         for i in range(v.size)])
    rng = np.random.default_rng(1)
    for n in (0, 1, 5, 30, 31, 32, 400):
        v = rng.random(n)
        assert np.array_equal(plot_trials._local_speed(v), loop(v)), n


# ---------------------------------------------------------------- tracking_eyes
def test_step_d_skips_sidecars_and_backup_copies(tmp_path):
    pytest.importorskip("cv2")
    sys.path.insert(0, str(REPO / "src" / "dlc"))
    import tracking_eyes
    (tmp_path / "clock_repair_backup").mkdir()
    for name in ("._20260630_Rat5_Coordinates_Full.csv",
                 "clock_repair_backup/20260630_Rat5_Coordinates_Full.csv",
                 "20260630_Rat5_Coordinates_Full.csv"):
        (tmp_path / name).write_text("Frame_Index\n")
    found = tracking_eyes._find_files(tmp_path, "*_Full.csv")
    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["20260630_Rat5_Coordinates_Full.csv"]


def test_step_d_maps_the_rat_into_the_eye_frame():
    pytest.importorskip("cv2")
    sys.path.insert(0, str(REPO / "src" / "dlc"))
    import tracking_eyes as T
    # Top row, camera 3 (region 2): canvas = display x 2, tile starts at 2 * 392,
    # and the stitch cut 104 px off the left and nothing off the top.
    assert T.rat_position_in_eye_frame(542.0, 300.0, 2) == (2 * 542 - 784 + 104, 600)
    # Bottom row, camera 8 (region 7): the tile starts at canvas y = 709 and the
    # stitch cut 91 px off the top of the (rotated) frame.
    assert T.rat_position_in_eye_frame(226.0, 554.5, 7) == (2 * 226 - 392 + 104, 1109 - 709 + 91)


def test_step_d_crop_box_stays_inside_the_frame():
    pytest.importorskip("cv2")
    sys.path.insert(0, str(REPO / "src" / "dlc"))
    import tracking_eyes as T
    size = T.crop_size_in_pixels(40)
    assert size == 106 and T.crop_size_in_pixels(0) == 0
    assert T.crop_box_origin(300, 400, size, 600, 800) == (247, 347)      # centred on the rat
    assert T.crop_box_origin(5, 795, size, 600, 800) == (0, 800 - size)   # pushed back inside


def test_step_d_cropped_video_is_enlarged_and_maps_back(tmp_path):
    """The saved crop is crop_scale times larger, has one frame per written frame,
    and a point in it maps back to the eye frame through crop_x0 / crop_scale."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    sys.path.insert(0, str(REPO / "src" / "dlc"))
    import tracking_eyes as T

    small = np.zeros((106, 106, 3), np.uint8)
    assert T.enlarge_crop(small, 4).shape == (424, 424, 3)
    assert T.enlarge_crop(small, 1) is small                    # 1 = left as it is
    # crop pixel (212, 100) of a 4x crop whose box starts at (247, 347)
    assert T.crop_pixel_to_eye_pixel(212, 100, 247, 347, 4) == (247 + 53, 347 + 25)

    video = tmp_path / "crop.mp4"
    writer = T.HighQualityVideoWriter(video, 30.0, (424, 424))
    for shade in range(5):
        writer.write(np.full((424, 424, 3), shade * 50, np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(video))
    frames = 0
    while cap.read()[0]:
        frames += 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    assert (frames, width) == (5, 424)


# ------------------------------------------------------------------ tracker_gui
def test_gui_keeps_a_character_split_between_two_reads_whole():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    import codecs
    text = "[TREE] FAILED Rat5_20260630 (3.1 min) — see log\n".encode("utf-8")
    cut = text.index("—".encode("utf-8")) + 1           # in the middle of the dash
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    assert decoder.decode(text[:cut]) + decoder.decode(text[cut:]) == text.decode("utf-8")


def test_step_d_writes_only_the_cropped_video_and_keeps_frames_in_step(tmp_path):
    """Step d part 1 saves the box around the rat (not the whole eye frame), one
    frame per row it could read, and skips the super-resolved video on request."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO / "src" / "dlc"))
    import tracking_eyes as T

    ip, op = tmp_path / "ip", tmp_path / "op"
    ip.mkdir(); op.mkdir()
    writer = cv2.VideoWriter(str(ip / "eye03_a.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (600, 800))
    for shade in range(6):
        writer.write(np.full((800, 600, 3), shade * 40, np.uint8))
    writer.release()
    rows = pd.DataFrame({"Frame_Index": range(6), "Rat_X": 542.0, "Rat_Y": 300.0})
    rows.loc[2, "Rat_X"] = np.nan                       # the rat was not tracked here
    rows.to_csv(op / "s_Coordinates_Full.csv", index=False)
    (op / T.SR_VIDEO_NAME).write_bytes(b"left over from an earlier run")

    T.process_tracking_data(str(ip), str(op), show_stream=False, super_resolve=False)

    assert not (op / "collected_frames.mp4").exists()   # whole frames are not saved
    assert not (op / T.SR_VIDEO_NAME).exists()          # stale video removed
    out = pd.read_csv(op / "s_Coordinates_Full_with_frames.csv")
    assert out["extracted_frame_idx"].notna().sum() == 5
    assert out.loc[0, ["crop_x0", "crop_y0", "crop_scale"]].tolist() == [351, 547, 4]
    cap = cv2.VideoCapture(str(op / T.CROPPED_VIDEO_NAME))
    frames = 0
    while cap.read()[0]:
        frames += 1
    assert (frames, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) == (5, 424)
    cap.release()
