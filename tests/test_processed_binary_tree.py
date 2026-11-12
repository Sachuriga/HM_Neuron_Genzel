"""processed_binary_tool --clean --root: deleting processed_binary/ in every
sorting output below a root folder (the share's layout), with clean()'s rules."""
import sys
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sorter"))
import processed_binary_tool as pbt                       # noqa: E402

SORT = "Rat5_HM_Neurons_{date}_105525_maze_merged.raw_group0_mountainsort4_sorting_output"


def make_session(folder, date, units_in_nwb=True, finished=True):
    """A session folder as step 7 (+ step u) leaves it; returns its processed_binary."""
    out = folder / SORT.format(date=date)
    processed = out / "processed_binary"
    processed.mkdir(parents=True)
    (processed / "traces_cached_seg0.raw").write_bytes(b"\x01\x02" * 5000)
    (processed / "binary.json").write_text("{}")
    if finished:
        (out / "phy_export").mkdir()
        (out / "phy_export" / "params.py").write_text(
            "dat_path = r'C:\\sorting pc\\processed_binary\\traces_cached_seg0.raw'\n"
            "n_channels_dat = 124\n")
    with h5py.File(folder / f"Rat5_{date}.nwb", "w") as f:
        if units_in_nwb:
            f.create_dataset("units/sorting_group", data=np.array([out.name.encode()], dtype="S"))
    return processed


@pytest.fixture
def share(tmp_path):
    """<root>/task/rat5_491390/<date>/, as on the share."""
    rat = tmp_path / "Rat_HM_Neuron" / "task" / "rat5_491390"
    return tmp_path / "Rat_HM_Neuron", {
        "done": make_session(rat / "20260623", "20260623"),
        "not_in_nwb": make_session(rat / "20260624", "20260624", units_in_nwb=False),
        "unfinished": make_session(rat / "20260625", "20260625", finished=False),
    }


def test_sorting_outputs_are_found_without_looking_inside_them(share):
    root, folders = share
    # a sorting output kept inside another one, and inside a step's own folder:
    # neither is a session's
    nested = folders["done"].parent / "phy_export" / "old" / SORT.format(date="19990101")
    (nested / "processed_binary").mkdir(parents=True)
    (root / "task" / "rat5_491390" / "20260623" / "clock_repair_backup"
     / SORT.format(date="19990102")).mkdir(parents=True)
    found = pbt.find_session_folders_with_sorting(root)
    assert [p.name for p in found] == ["20260623", "20260624", "20260625"]
    assert pbt.find_session_folders_with_sorting(root, max_depth=2) == []      # too shallow


def test_a_dry_run_deletes_nothing(share, capsys):
    root, folders = share
    total = pbt.clean_tree(root)
    assert all(p.is_dir() for p in folders.values())
    assert (total["sessions"], total["deleted"], total["kept"], total["failed"]) == (3, 1, 2, 0)
    assert total["bytes"] == 10002
    assert "DRY RUN" in capsys.readouterr().out


def test_only_sorts_whose_units_are_in_the_nwb_are_deleted(share):
    root, folders = share
    total = pbt.clean_tree(root, really_delete=True)
    assert not folders["done"].exists()
    assert folders["not_in_nwb"].is_dir() and folders["unfinished"].is_dir()
    assert (total["deleted"], total["kept"], total["bytes"]) == (1, 2, 10002)
    out = folders["done"].parent
    assert (out / pbt.FINGERPRINT_FILE).exists() and (out / pbt.REMOVED_NOTE_FILE).exists()
    assert (out / "phy_export" / "params.py").exists()            # nothing else is touched
    # a second run finds nothing left to do there
    assert pbt.clean_tree(root, really_delete=True)["deleted"] == 0


def test_force_also_deletes_where_step_u_has_not_run(share):
    root, folders = share
    total = pbt.clean_tree(root, really_delete=True, force=True)
    assert not folders["done"].exists() and not folders["not_in_nwb"].exists()
    assert folders["unfinished"].is_dir()                 # no phy_export: never deleted
    assert (total["deleted"], total["kept"]) == (2, 1)
