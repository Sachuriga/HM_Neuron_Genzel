"""runner.py: a sleep ip folder (pre/post) is emptied only when its recordings are
sorted (processed_binary kept) and the session NWB holds LFP + EMG + motion.
processed_binary_tool --clean then keeps processed_binary/ (the last 30 kHz copy)."""
import sys
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "sorter"))
import runner  # noqa: E402  (chdirs to the repo root, as when it runs)
import processed_binary_tool as pbt  # noqa: E402

PRE = "Rat6_HM_Neurons_20260707_091045_pre"
POST = "Rat6_HM_Neurons_20260707_131500_post"
POST2 = "Rat6_HM_Neurons_20260707_140000_post2"
MAZE = "Rat6_HM_Neurons_20260707_101500_maze"


def touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def add_recording(ip, name):
    touch(ip / f"{name}.rec")
    touch(ip / f"{name}.trodesconf")
    touch(ip / f"{name}.DIO" / "dio.dat")


def add_sort(op, name, binary=True):
    out = op / f"{name}.raw_group0_mountainsort4_sorting_output"
    touch(out / "phy_export" / "params.py", "dat_path = 'traces_cached_seg0.raw'\n")
    if binary:
        touch(out / "processed_binary" / "traces_cached_seg0.raw")
    return out


def add_nwb(op, series=("lfp", "emg_from_lfp", "motion")):
    op.mkdir(parents=True, exist_ok=True)
    with h5py.File(op / "Rat6_20260707_pre.nwb", "w") as f:
        for name in series:
            f.create_dataset(f"acquisition/{name}/data", data=[1.0, 2.0])


@pytest.fixture
def pair(tmp_path):
    return tmp_path / "ip1", tmp_path / "op1"


def test_sorted_sleep_folder_with_lfp_is_emptied(pair):
    ip, op = pair
    add_recording(ip, PRE)
    add_sort(op, PRE)
    add_nwb(op)
    assert runner.sleep_ip_problems(ip, op) == []
    runner.empty_sleep_ip_folders([(ip, op)])
    assert ip.is_dir() and list(ip.iterdir()) == []       # the folder itself stays
    assert (op / runner.IP_CLEARED_NOTE).is_file()
    assert (op / f"{PRE}.raw_group0_mountainsort4_sorting_output" / "processed_binary").is_dir()


def test_maze_folder_is_never_a_sleep_folder(pair):
    ip, op = pair
    add_recording(ip, MAZE)
    add_recording(ip, PRE)                                # mixed: still not a sleep folder
    add_sort(op, MAZE)
    add_sort(op, PRE)
    add_nwb(op)
    assert runner.sleep_ip_problems(ip, op) is None
    runner.empty_sleep_ip_folders([(ip, op)])
    assert (ip / f"{MAZE}.rec").exists() and (ip / f"{PRE}.rec").exists()


def test_kept_without_a_finished_sort(pair):
    ip, op = pair
    add_recording(ip, PRE)
    add_nwb(op)                                           # preset 18: LFP only
    problems = runner.sleep_ip_problems(ip, op)
    assert len(problems) == 1 and "step 7" in problems[0]
    runner.empty_sleep_ip_folders([(ip, op)])
    assert (ip / f"{PRE}.rec").exists()


def test_kept_when_processed_binary_is_gone(pair):
    ip, op = pair
    add_recording(ip, PRE)
    add_sort(op, PRE, binary=False)
    add_nwb(op)
    assert runner.sleep_ip_problems(ip, op)


def test_kept_without_lfp_emg_or_motion_in_the_nwb(pair):
    ip, op = pair
    add_recording(ip, POST)
    add_sort(op, POST)
    assert "no session NWB" in runner.sleep_ip_problems(ip, op)[0]
    add_nwb(op, series=("lfp", "emg_from_lfp"))
    assert "motion" in runner.sleep_ip_problems(ip, op)[0]


def test_split_parts_of_one_phase_need_one_sort(pair):
    ip, op = pair
    add_recording(ip, POST)
    add_recording(ip, POST2)                              # sorted together with POST
    add_sort(op, POST)
    add_nwb(op)
    assert runner.sleep_ip_problems(ip, op) == []


def test_pre_and_post_in_one_folder_both_need_a_sort(pair):
    ip, op = pair
    add_recording(ip, PRE)
    add_recording(ip, POST)
    add_sort(op, PRE)
    add_nwb(op)
    problems = runner.sleep_ip_problems(ip, op)
    assert len(problems) == 1 and POST in problems[0]


def test_clean_keeps_processed_binary_after_the_ip_was_emptied(pair):
    ip, op = pair
    add_recording(ip, PRE)
    out = add_sort(op, PRE)
    add_nwb(op)
    runner.empty_sleep_ip_folders([(ip, op)])
    counts = pbt.clean(op, really_delete=True, force=True)
    assert counts["deleted"] == 0 and counts["kept"] == 1
    assert (out / "processed_binary" / "traces_cached_seg0.raw").exists()
